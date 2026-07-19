"""Portable DVC-result bundle I/O (V1.76) — Qt-free.

A DVC node produces, for every multipoint and frame, a dense
:class:`~nd2studios.core.dvc_registry.DVCResult` (grid coordinates, displacement +
strain fields, q-factor, diagnostics), plus per-frame image backdrops and — for a
Granule Volume Mask → DVC edge on "Objects" scope — a full per-granule bundle
(field series + cropped mask + crop origin). The DVC viewer (``DVCPanel``) plays
through all of that; nothing about it is derivable from the raw ND2 once the run is
gone.

This module serializes that whole payload to a **single portable file** (default
extension ``.nd2dvc``) so it can be reloaded into a *new* ND2Studios run and
re-rendered by the same viewer with no recomputation — the export/reload asked for
in V1.76. The file is an NPZ (zip) written by :func:`numpy.savez_compressed`:

* one ``__manifest__`` entry — a UTF-8 JSON byte array (**no pickle**) describing the
  structure, the metadata the viewer needs (frame counts, pixel / voxel sizes,
  channel names, timestamps) and a read-only **provenance** record (how the field was
  produced); and
* the numpy arrays, keyed ``a0, a1, …``, **deduplicated by object identity** — an
  object mask broadcast across every frame (``{t: mask for t in series}``) is written
  once, not once per frame.

Because it is a zip, :func:`numpy.load` auto-detects it by content, so the custom
``.nd2dvc`` extension loads fine. This module deliberately imports only the standard
library + numpy + the pure :class:`DVCResult` dataclass (mirrors
``pipeline_graph/checkpoint_io.py`` and ``pipeline_graph/io.py``): no PySide6, so it
runs headless and off a worker thread.

Public API::

    save_dvc_bundle(path, *, series_by_m, incr_by_m, bg_by_mt, whole_masks_by_m,
                    obj_full_by_m, display_mask_by_m, meta, provenance)
    bundle = load_dvc_bundle(path)   # dict of the same stores, reconstructed

The ``bundle`` dict returned by :func:`load_dvc_bundle` mirrors the page's DVC stores
(``_dvc_series_by_m`` etc.) so ``PipelinesPage`` can assign it directly.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nd2studios.core.dvc_registry import DVCResult

DVC_BUNDLE_KIND = "nd2studios.dvc-bundle"
DVC_BUNDLE_VERSION = 1
DVC_BUNDLE_EXTENSION = ".nd2dvc"
_MANIFEST_KEY = "__manifest__"

# DVCResult array fields serialized via the identity-deduped array registry
# (order-independent — each is looked up by name on load).
_RESULT_ARRAY_FIELDS = ("grid_coords", "displacement_field", "strain_field",
                        "qfactor")


# ── JSON coercion (numpy scalars/arrays/tuples/sets → plain JSON) ────────────
def _json_safe(obj: Any) -> Any:
    """Deep-coerce ``obj`` so it is JSON-serializable.

    Diagnostics dicts and provenance/meta blocks can carry numpy scalars, tuples,
    small arrays and sets; this normalizes them (arrays → lists) so
    :func:`json.dumps` never raises. Large numpy arrays should go through the
    array registry, not here.
    """
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return [_json_safe(v) for v in sorted(obj)]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def json_safe(obj: Any) -> Any:
    """Public entry to :func:`_json_safe` — coerce ``obj`` (a provenance / meta
    block) to plain JSON so callers can store it on a pipeline node and round-trip
    it through ``save_pipeline`` without a numpy-type serialization error."""
    return _json_safe(obj)


# ── identity-deduped array registry ──────────────────────────────────────────
class _ArrayRegistry:
    """Collects arrays for the NPZ, deduplicating by ``id()``.

    The DVC object stores broadcast one mask array across all frames and repeat the
    largest object's mask as the display mask, so the *same* ndarray appears many
    times. Keying on identity writes each distinct array once and hands back a stable
    ``a<n>`` key; :meth:`stash` returns ``None`` for a ``None`` input (an absent
    ``strain_field`` / ``qfactor``).
    """

    def __init__(self) -> None:
        self._by_id: Dict[int, str] = {}
        self.arrays: Dict[str, np.ndarray] = {}
        self._n = 0

    def stash(self, arr: Any) -> Optional[str]:
        if arr is None:
            return None
        oid = id(arr)
        key = self._by_id.get(oid)
        if key is not None:
            return key
        key = f"a{self._n}"
        self._n += 1
        self._by_id[oid] = key
        self.arrays[key] = np.asarray(arr)
        return key


# ── DVCResult <-> JSON ────────────────────────────────────────────────────────
def _result_to_json(r: DVCResult, reg: _ArrayRegistry) -> Dict[str, Any]:
    """Serialize one :class:`DVCResult` — arrays via ``reg``, scalars inline."""
    out: Dict[str, Any] = {
        "dim": int(r.dim),
        "voxel_size_um": [float(v) for v in (r.voxel_size_um or ())],
        "strain_type": str(r.strain_type or ""),
        "converged": bool(r.converged),
        "iterations": int(r.iterations),
        "mu": float(r.mu),
        "beta": float(r.beta),
        "method": str(r.method or ""),
        "notes": str(r.notes or ""),
        "diagnostics": _json_safe(r.diagnostics or {}),
    }
    for name in _RESULT_ARRAY_FIELDS:
        out[name] = reg.stash(getattr(r, name, None))
    return out


def _result_from_json(d: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> DVCResult:
    """Rebuild a :class:`DVCResult` from its JSON dict + the loaded arrays."""
    def _arr(name: str) -> Optional[np.ndarray]:
        key = d.get(name)
        return arrays.get(key) if key is not None else None

    gc = _arr("grid_coords")
    disp = _arr("displacement_field")
    return DVCResult(
        dim=int(d.get("dim", 3)),
        grid_coords=gc if gc is not None else np.zeros((0,), dtype=float),
        displacement_field=disp if disp is not None else np.zeros((0,), dtype=float),
        voxel_size_um=tuple(float(v) for v in d.get("voxel_size_um", ()) or ()),
        strain_field=_arr("strain_field"),
        strain_type=str(d.get("strain_type", "")),
        qfactor=_arr("qfactor"),
        converged=bool(d.get("converged", False)),
        iterations=int(d.get("iterations", 0)),
        mu=float(d.get("mu", 0.0)),
        beta=float(d.get("beta", 0.0)),
        method=str(d.get("method", "")),
        notes=str(d.get("notes", "")),
        diagnostics=dict(d.get("diagnostics", {}) or {}),
    )


# ── {t: DVCResult} series <-> JSON ────────────────────────────────────────────
def _series_to_json(series: Optional[Dict[int, DVCResult]],
                    reg: _ArrayRegistry) -> Dict[str, Any]:
    return {str(int(t)): _result_to_json(r, reg)
            for t, r in (series or {}).items() if r is not None}


def _series_from_json(d: Optional[Dict[str, Any]],
                      arrays: Dict[str, np.ndarray]) -> Dict[int, DVCResult]:
    return {int(t): _result_from_json(rj, arrays) for t, rj in (d or {}).items()}


def _frames_arr_to_json(frames: Optional[Dict[int, Any]],
                        reg: _ArrayRegistry) -> Dict[str, Any]:
    """``{t: ndarray}`` (backdrops / masks) → ``{str(t): array_key}`` (deduped)."""
    out: Dict[str, Any] = {}
    for t, arr in (frames or {}).items():
        key = reg.stash(arr)
        if key is not None:
            out[str(int(t))] = key
    return out


def _frames_arr_from_json(d: Optional[Dict[str, Any]],
                          arrays: Dict[str, np.ndarray]) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for t, key in (d or {}).items():
        arr = arrays.get(key)
        if arr is not None:
            out[int(t)] = arr
    return out


# ── per-granule object bundle <-> JSON ────────────────────────────────────────
def _object_to_json(bundle: Dict[str, Any], reg: _ArrayRegistry) -> Dict[str, Any]:
    """Serialize one ``_dvc_obj_full_by_m`` granule bundle."""
    origin = bundle.get("origin", (0, 0, 0)) or (0, 0, 0)
    return {
        "series": _series_to_json(bundle.get("series"), reg),
        "increment": _series_to_json(bundle.get("increment"), reg),
        "bg": _frames_arr_to_json(bundle.get("bg"), reg),
        "mask": _frames_arr_to_json(bundle.get("mask"), reg),
        "n_voxels": int(bundle.get("n_voxels", 0)),
        "origin": [int(v) for v in origin],
    }


def _object_from_json(d: Dict[str, Any],
                      arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    return {
        "series": _series_from_json(d.get("series"), arrays),
        "increment": _series_from_json(d.get("increment"), arrays),
        "bg": _frames_arr_from_json(d.get("bg"), arrays),
        "mask": _frames_arr_from_json(d.get("mask"), arrays),
        "n_voxels": int(d.get("n_voxels", 0)),
        "origin": tuple(int(v) for v in d.get("origin", (0, 0, 0)) or (0, 0, 0)),
    }


# ── save / load ───────────────────────────────────────────────────────────────
def save_dvc_bundle(
    path: str,
    *,
    series_by_m: Dict[int, Dict[int, DVCResult]],
    incr_by_m: Optional[Dict[int, Dict[int, DVCResult]]] = None,
    bg_by_mt: Optional[Dict[int, Dict[int, np.ndarray]]] = None,
    whole_masks_by_m: Optional[Dict[int, Dict[int, np.ndarray]]] = None,
    obj_full_by_m: Optional[Dict[int, Dict[int, Dict[str, Any]]]] = None,
    display_mask_by_m: Optional[Dict[int, Dict[int, np.ndarray]]] = None,
    meta: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> None:
    """Write every DVC output (all multipoints) to a portable ``.nd2dvc`` bundle.

    All ``*_by_m`` maps are keyed by multipoint index. ``series_by_m`` is the
    cumulative field series; ``incr_by_m`` the raw per-step increments; ``bg_by_mt``
    the per-frame backdrops; ``whole_masks_by_m`` the whole-frame drawn masks (3D Mask
    Drawing path); ``obj_full_by_m`` the per-granule bundles (Objects-scope path);
    ``display_mask_by_m`` the largest object's mask for the single-surface path.
    ``meta`` and ``provenance`` are opaque JSON blocks (coerced defensively). Written
    through an open file handle so the ``.nd2dvc`` extension is preserved (``savez``
    would otherwise append ``.npz``).
    """
    reg = _ArrayRegistry()
    incr_by_m = incr_by_m or {}
    bg_by_mt = bg_by_mt or {}
    whole_masks_by_m = whole_masks_by_m or {}
    obj_full_by_m = obj_full_by_m or {}
    display_mask_by_m = display_mask_by_m or {}

    all_ms = sorted({int(m) for m in (
        set(series_by_m) | set(incr_by_m) | set(bg_by_mt) | set(whole_masks_by_m)
        | set(obj_full_by_m) | set(display_mask_by_m))})

    multipoints: Dict[str, Any] = {}
    for m in all_ms:
        objects = {
            str(int(oid)): _object_to_json(b, reg)
            for oid, b in (obj_full_by_m.get(m, {}) or {}).items()
        }
        multipoints[str(m)] = {
            "series": _series_to_json(series_by_m.get(m), reg),
            "increments": _series_to_json(incr_by_m.get(m), reg),
            "backgrounds": _frames_arr_to_json(bg_by_mt.get(m), reg),
            "whole_masks": _frames_arr_to_json(whole_masks_by_m.get(m), reg),
            "display_mask": _frames_arr_to_json(display_mask_by_m.get(m), reg),
            "objects": objects,
        }

    manifest = {
        "kind": DVC_BUNDLE_KIND,
        "version": DVC_BUNDLE_VERSION,
        "provenance": _json_safe(provenance or {}),
        "meta": _json_safe(meta or {}),
        "multipoints": multipoints,
    }
    payload = dict(reg.arrays)
    payload[_MANIFEST_KEY] = np.frombuffer(
        json.dumps(manifest).encode("utf-8"), dtype=np.uint8)
    with open(path, "wb") as fh:
        np.savez_compressed(fh, **payload)


def load_dvc_bundle(path: str) -> Dict[str, Any]:
    """Read a ``.nd2dvc`` bundle written by :func:`save_dvc_bundle`.

    Returns a dict mirroring the page's DVC stores so it can be assigned directly::

        {"kind", "version", "provenance", "meta",
         "series_by_m": {m: {t: DVCResult}},
         "incr_by_m":   {m: {t: DVCResult}},
         "bg_by_mt":    {m: {t: ndarray}},
         "whole_masks_by_m": {m: {t: ndarray}},
         "display_mask_by_m": {m: {t: ndarray}},
         "obj_full_by_m": {m: {oid: {"series","increment","bg","mask",
                                     "n_voxels","origin"}}}}

    Raises ``ValueError`` if the file is not an ND2Studios DVC bundle or is newer than
    this build understands.
    """
    with np.load(path, allow_pickle=False) as z:
        if _MANIFEST_KEY not in z.files:
            raise ValueError("Not an ND2Studios DVC bundle (no manifest).")
        manifest = json.loads(bytes(z[_MANIFEST_KEY].tobytes()).decode("utf-8"))
        arrays = {k: z[k] for k in z.files if k != _MANIFEST_KEY}

    if not isinstance(manifest, dict) or manifest.get("kind") != DVC_BUNDLE_KIND:
        raise ValueError("Not an ND2Studios DVC bundle (wrong kind).")
    if int(manifest.get("version", 0)) > DVC_BUNDLE_VERSION:
        raise ValueError(
            f"DVC bundle version {manifest.get('version')} is newer than this "
            f"build supports ({DVC_BUNDLE_VERSION}).")

    series_by_m: Dict[int, Dict[int, DVCResult]] = {}
    incr_by_m: Dict[int, Dict[int, DVCResult]] = {}
    bg_by_mt: Dict[int, Dict[int, np.ndarray]] = {}
    whole_masks_by_m: Dict[int, Dict[int, np.ndarray]] = {}
    display_mask_by_m: Dict[int, Dict[int, np.ndarray]] = {}
    obj_full_by_m: Dict[int, Dict[int, Dict[str, Any]]] = {}

    for m_str, mp in (manifest.get("multipoints") or {}).items():
        m = int(m_str)
        series_by_m[m] = _series_from_json(mp.get("series"), arrays)
        incr = _series_from_json(mp.get("increments"), arrays)
        if incr:
            incr_by_m[m] = incr
        bg = _frames_arr_from_json(mp.get("backgrounds"), arrays)
        if bg:
            bg_by_mt[m] = bg
        wm = _frames_arr_from_json(mp.get("whole_masks"), arrays)
        if wm:
            whole_masks_by_m[m] = wm
        dm = _frames_arr_from_json(mp.get("display_mask"), arrays)
        if dm:
            display_mask_by_m[m] = dm
        objects = {int(oid): _object_from_json(od, arrays)
                   for oid, od in (mp.get("objects") or {}).items()}
        if objects:
            obj_full_by_m[m] = objects

    return {
        "kind": manifest.get("kind"),
        "version": int(manifest.get("version", 0)),
        "provenance": manifest.get("provenance") or {},
        "meta": manifest.get("meta") or {},
        "series_by_m": series_by_m,
        "incr_by_m": incr_by_m,
        "bg_by_mt": bg_by_mt,
        "whole_masks_by_m": whole_masks_by_m,
        "display_mask_by_m": display_mask_by_m,
        "obj_full_by_m": obj_full_by_m,
    }
