"""pyALDIC (``al-dic``) engine wrapper + result adapter.

Bridges the optional third-party 2D AL-DIC solver to the app's DVC data model:

1. **Input adapter** — normalize app frames (``(H, W)`` numpy) to the float64
   ``[0, 1]`` grayscale + boolean-mask list ``al_dic.core.pipeline.run_aldic``
   expects, and map the node's params to a ``DICPara``.
2. **Solve** — call ``run_aldic`` once over the whole ordered image series
   (accumulative vs incremental handled natively by ``reference_mode``), with a
   progress + cancel callback wired through.
3. **Output adapter** — resample each per-frame result (which lives on pyALDIC's
   *adaptive quadtree FE mesh*, node coords ``[x, y]``, displacement interleaved
   ``[u, v, ...]``) onto a **regular grid** so it fits :class:`DVCResult`
   (``(Gy, Gx, 2)``, axis order ``[y, x]``, displacement ``[dy, dx]``), which the
   existing :class:`~nd2studios.widgets.dvc_panel.DVCPanel` renders unchanged.

Pure backend module — no PySide6. ``al_dic`` is imported lazily (see
:func:`_require_al_dic`) so importing this module never requires the extra.
"""
from __future__ import annotations

import importlib
import importlib.util
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nd2studios.core.dvc_registry import DVCResult

ProgressCb = Optional[Callable[[int], None]]
CancelledCb = Optional[Callable[[], bool]]

_INSTALL_HINT = (
    "The 2D DIC node needs the optional 'al-dic' package (pyALDIC).\n"
    "Install it with:  pip install al-dic\n"
    "(it pulls numba + PySide6>=6.6; it is intentionally not a core dependency)."
)


def al_dic_available() -> bool:
    """True if the optional ``al_dic`` package is importable."""
    return importlib.util.find_spec("al_dic") is not None


def _require_al_dic():
    """Import and return the ``al_dic`` entry points, or raise a friendly error."""
    if not al_dic_available():
        raise ImportError(_INSTALL_HINT)
    config = importlib.import_module("al_dic.core.config")
    pipeline = importlib.import_module("al_dic.core.pipeline")
    return config, pipeline


def _refinement_policy(refinement: Optional[Dict[str, Any]], half_win: int):
    """Build an ``al_dic`` ``RefinementPolicy`` from the refine node's spec, or None."""
    if not refinement:
        return None
    try:
        refine = importlib.import_module("al_dic.mesh.refinement")
    except Exception:  # noqa: BLE001 — refinement is best-effort
        return None
    crit = refinement.get("criteria", {}) if isinstance(refinement, dict) else {}
    brush = refinement.get("brush")
    mask = None
    if brush is not None:
        mask = np.asarray(brush).astype(np.float64)
    try:
        return refine.build_refinement_policy(
            refine_inner_boundary=bool(crit.get("mask_boundary", False)),
            refine_outer_boundary=bool(crit.get("roi_edge", False)),
            refinement_mask=(mask if bool(crit.get("brush", mask is not None)) else None),
            min_element_size=int(refinement.get("min_element_size", 8) or 8),
            half_win=int(half_win),
        )
    except Exception:  # noqa: BLE001 — never fail the solve on a bad policy
        return None


# ── input normalization ──────────────────────────────────────────────────────
def _to_float01(img: np.ndarray) -> np.ndarray:
    """Grayscale ``(H, W)`` float64 in [0, 1] (dtype-aware scaling)."""
    a = np.asarray(img)
    if a.ndim == 3:                       # collapse an accidental channel axis
        a = a.max(axis=0) if a.shape[0] <= 4 else a[a.shape[0] // 2]
    a = a.astype(np.float64)
    lo = float(a.min())
    hi = float(a.max())
    if hi <= lo:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _snap_pow2(v: int) -> int:
    """Nearest power of two >= 2 (``winstepsize`` / ``winsize_min`` must be pow2)."""
    v = max(2, int(v))
    return int(2 ** round(np.log2(v)))


def _build_dicpara(config, params: Dict[str, Any], img_size: Tuple[int, int],
                   reference_mode: str, use_masks: bool):
    """Map node params → an ``al_dic`` ``DICPara`` (validated by ``dicpara_default``)."""
    winsize = max(2, int(params.get("winsize", 40) or 40))
    if winsize % 2:                       # winsize must be even
        winsize += 1
    winstep = _snap_pow2(int(params.get("winstepsize", 16) or 16))
    winmin = min(_snap_pow2(int(params.get("winsize_min", 8) or 8)), winstep)
    overrides: Dict[str, Any] = {
        "winsize": winsize,
        "winstepsize": winstep,
        "winsize_min": winmin,
        "init_guess_mode": str(params.get("init_guess_mode", "auto") or "auto"),
        "mu": float(params.get("mu", 1e-3) or 1e-3),
        "tol": float(params.get("tol", 1e-2) or 1e-2),
        "admm_max_iter": max(1, int(params.get("admm_max_iter", 3) or 3)),
        "icgn_max_iter": max(1, int(params.get("icgn_max_iter", 100) or 100)),
        "disp_smoothness": max(0.0, float(params.get("disp_smoothness", 5e-4) or 0.0)),
        "strain_smoothness": max(0.0, float(params.get("strain_smoothness", 1e-5) or 0.0)),
        "reference_mode": reference_mode,
        "img_size": (int(img_size[0]), int(img_size[1])),
        "use_masks": bool(use_masks),
        # Keep displacement in *pixels* — DVCResult stores voxels and converts to
        # µm itself via ``voxel_size_um`` (matching the DVC engine).
        "um2px": 1.0,
        "show_plots": False,
    }
    return config.dicpara_default(**overrides)


# ── output adapter (adaptive FE mesh → regular grid) ─────────────────────────
def _mesh_coords(mesh) -> np.ndarray:
    """``(N, 2)`` node coordinates ``[x, y]`` from an ``al_dic`` ``DICMesh``."""
    return np.asarray(getattr(mesh, "coordinates_fem"), dtype=float).reshape(-1, 2)


def _regular_grid(coords_xy: np.ndarray, step: int
                  ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Regular grid (pitch = ``step``) spanning the node bbox.

    Returns ``(Yq, Xq, (Gy, Gx))`` where ``Yq``/``Xq`` are ``(Gy, Gx)`` meshes in
    image coordinates (row = y, col = x)."""
    x_min, y_min = coords_xy[:, 0].min(), coords_xy[:, 1].min()
    x_max, y_max = coords_xy[:, 0].max(), coords_xy[:, 1].max()
    step = max(1, int(step))
    xs = np.arange(x_min, x_max + 1e-6, step)
    ys = np.arange(y_min, y_max + 1e-6, step)
    if xs.size < 1:
        xs = np.asarray([x_min])
    if ys.size < 1:
        ys = np.asarray([y_min])
    Xq, Yq = np.meshgrid(xs, ys)          # (Gy, Gx)
    return Yq, Xq, (int(ys.size), int(xs.size))


def _interp_to_grid(coords_xy: np.ndarray, values: np.ndarray,
                    Yq: np.ndarray, Xq: np.ndarray) -> np.ndarray:
    """Interpolate node ``values`` onto the query grid (linear + nearest fill)."""
    from scipy.interpolate import griddata
    pts = coords_xy                       # (N, 2) as (x, y)
    query = np.column_stack([Xq.ravel(), Yq.ravel()])
    out = np.full(query.shape[0], np.nan, dtype=np.float64)
    if pts.shape[0] >= 4:
        out = griddata(pts, values, query, method="linear")
    holes = ~np.isfinite(out)
    if holes.any() and pts.shape[0] >= 1:
        fill = griddata(pts, values, query, method="nearest")
        out[holes] = fill[holes]
    return out.reshape(Yq.shape)


def _frame_to_dvcresult(mesh, U: np.ndarray, step: int,
                        voxel_size_um: Tuple[float, ...], method: str,
                        params: Dict[str, Any], reference_mode: str,
                        notes: str = "") -> DVCResult:
    """Convert one pyALDIC frame result (mesh + interleaved U) to a 2D DVCResult.

    pyALDIC: coords ``[x, y]``, ``U = [u0, v0, u1, v1, ...]`` (u = x-disp, v = y-disp).
    DVCResult: ``grid_coords[..., 0] = y, [..., 1] = x``; displacement
    ``[..., 0] = dy, [..., 1] = dx`` — hence the swap below."""
    coords = _mesh_coords(mesh)           # (N, 2) [x, y]
    u = np.asarray(U, dtype=float).reshape(-1)
    n = coords.shape[0]
    if u.size >= 2 * n and n > 0:
        u_x = u[0:2 * n:2]                # x-displacement per node
        v_y = u[1:2 * n:2]               # y-displacement per node
    else:                                 # defensive: degenerate result
        u_x = np.zeros(n)
        v_y = np.zeros(n)
    Yq, Xq, grid_shape = _regular_grid(coords, step)
    dy = _interp_to_grid(coords, v_y, Yq, Xq)     # (Gy, Gx) y-disp
    dx = _interp_to_grid(coords, u_x, Yq, Xq)     # (Gy, Gx) x-disp
    grid_coords = np.stack([Yq, Xq], axis=-1)     # (Gy, Gx, 2) [y, x]
    disp = np.stack([dy, dx], axis=-1)            # (Gy, Gx, 2) [dy, dx]
    return DVCResult(
        dim=2,
        grid_coords=grid_coords.astype(np.float64),
        displacement_field=disp.astype(np.float64),
        voxel_size_um=tuple(float(v) for v in voxel_size_um),
        strain_field=None,                 # DVCPanel derives strain from displacement
        strain_type="infinitesimal",
        converged=True,
        iterations=int(params.get("admm_max_iter", 3) or 3),
        mu=float(params.get("mu", 1e-3) or 1e-3),
        method=method,
        notes=notes,
        diagnostics={
            "engine": "pyALDIC (al-dic)",
            "n_nodes": int(n),
            "grid_shape": tuple(int(v) for v in grid_shape),
            "winsize": int(params.get("winsize", 40) or 40),
            "winstepsize": int(step),
            "reference_mode": reference_mode,
        },
    )


# ── public entry points ──────────────────────────────────────────────────────
def run_pyaldic_series(
    images: Sequence[np.ndarray],
    masks: Optional[Sequence[np.ndarray]],
    params: Dict[str, Any],
    voxel_size_um: Tuple[float, ...],
    *,
    reference_mode: str = "accumulative",
    refinement: Optional[Dict[str, Any]] = None,
    progress_cb: ProgressCb = None,
    cancelled_cb: CancelledCb = None,
) -> List[Dict[str, DVCResult]]:
    """Run AL-DIC over an ordered image series and adapt the results.

    ``images[0]`` is the reference. Returns a list of length ``len(images) - 1``
    (one entry per deformed frame, in order) — each a dict
    ``{"primary": DVCResult, "increment": DVCResult}`` where *primary* is the
    cumulative field (from pyALDIC's ``U_accum`` when present) and *increment* is
    the raw per-step field (``U``). Cumulative vs incremental behaviour is chosen
    natively by ``reference_mode`` ("accumulative" | "incremental").

    Raises ``ImportError`` (friendly) when ``al-dic`` is not installed and
    ``RuntimeError`` when the user cancels.
    """
    config, pipeline = _require_al_dic()
    imgs = [_to_float01(im) for im in images]
    if len(imgs) < 2:
        raise ValueError("AL-DIC needs at least 2 images (reference + deformed).")
    H, W = imgs[0].shape[:2]
    if masks is None:
        mask_list = [np.ones((H, W), dtype=np.float64) for _ in imgs]
        use_masks = False
    else:
        mask_list = [np.asarray(m).astype(np.float64) for m in masks]
        use_masks = any(float(m.min()) < 1.0 for m in mask_list)
    para = _build_dicpara(config, params, (H, W), reference_mode, use_masks)
    winsize = int(getattr(para, "winsize", params.get("winsize", 40)))
    policy = _refinement_policy(refinement, half_win=max(1, winsize // 2))

    def _progress(frac: float, _msg: str = "") -> None:
        if progress_cb is not None:
            progress_cb(int(max(0.0, min(1.0, float(frac))) * 100))

    def _stop() -> bool:
        return bool(cancelled_cb()) if cancelled_cb is not None else False

    result = pipeline.run_aldic(
        para, imgs, mask_list,
        progress_fn=_progress, stop_fn=_stop,
        compute_strain=bool(params.get("compute_strain", True)),
        refinement_policy=policy,
    )

    disp = list(getattr(result, "result_disp", []) or [])
    meshes = list(getattr(result, "result_fe_mesh_each_frame", []) or [])
    canonical = getattr(result, "dic_mesh", None)
    step = int(getattr(para, "winstepsize", params.get("winstepsize", 16)))
    method = "pyALDIC"
    out: List[Dict[str, DVCResult]] = []
    for i, fr in enumerate(disp):
        mesh = meshes[i] if i < len(meshes) and meshes[i] is not None else canonical
        if mesh is None:
            continue
        u_incr = getattr(fr, "U", None)
        u_accum = getattr(fr, "U_accum", None)
        if u_accum is None:
            u_accum = u_incr
        primary = _frame_to_dvcresult(
            mesh, u_accum, step, voxel_size_um, method, params, reference_mode,
            notes="cumulative")
        increment = _frame_to_dvcresult(
            mesh, u_incr, step, voxel_size_um, method, params, reference_mode,
            notes="increment")
        out.append({"primary": primary, "increment": increment})
    return out


def run_pyaldic_pair(
    ref_img: np.ndarray,
    def_img: np.ndarray,
    voxel_size_um: Tuple[float, ...],
    params: Dict[str, Any],
    progress_cb: ProgressCb = None,
    cancelled_cb: CancelledCb = None,
    *,
    roi_mask: Optional[np.ndarray] = None,
    refinement: Optional[Dict[str, Any]] = None,
) -> DVCResult:
    """Single reference/deformed pair convenience (headless + the DVCMethod ABC).

    Returns the cumulative :class:`DVCResult` for ``def_img`` vs ``ref_img``."""
    masks = None
    if roi_mask is not None:
        m = np.asarray(roi_mask).astype(np.float64)
        masks = [m, m]
    series = run_pyaldic_series(
        [ref_img, def_img], masks, params, voxel_size_um,
        reference_mode="accumulative", refinement=refinement,
        progress_cb=progress_cb, cancelled_cb=cancelled_cb)
    if not series:
        raise RuntimeError("AL-DIC produced no result for the image pair.")
    return series[0]["primary"]
