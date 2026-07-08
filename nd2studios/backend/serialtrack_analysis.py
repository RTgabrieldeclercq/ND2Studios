"""SerialTrack PTV analysis data layer (numpy/scipy, Qt-free).

Turns ND2Studios tracked measurement rows into Particle-Tracking-Velocimetry
(PTV) / digital-volume-correlation analysis products for the SerialTrack viewer
tab. This is the data layer described in ``CodeLog/ClaudesPlan/V1.47_serialtrack_ptv_plots.md``.

Why this module exists
----------------------
In ND2Studios, SerialTrack is run only to chain ``track_id`` onto measurement
rows (``backend/object_tracker._link_group_serialtrack`` sets
``strain_n_neighbors=0`` and discards the ``TrackingSession``). The rich
displacement / strain / deformation-gradient fields SerialTrack can compute are
never persisted. This module **rebuilds** those fields from the tracked rows
using SerialTrack's own field maths (:mod:`nd2studios.backend.serialtrack.fields`),
so the PTV plots work on any tracked experiment with no engine change.

Coordinate & data conventions
------------------------------
* Positions are stored ``(y, x)`` for 2D or ``(y, x, z)`` for 3D — matching the
  ``centroid_y_px`` / ``centroid_x_px`` / ``centroid_z_px`` row keys and image
  ``(row, col)`` indexing. Component index 0 is the *y* component, 1 is *x*,
  2 is *z*. All derived quantities (divergence, curl, Jacobian) are computed
  generically from the deformation-gradient tensor, so the convention is
  internally consistent regardless of axis order.
* Dimensionality ``D`` is inferred per dataset: 3 when any row carries a
  non-null ``centroid_z_px``, else 2. (Today's ND2Studios pipeline collapses Z
  at load, so ``D == 2`` in practice; 3D lights up automatically for imported /
  volumetric 3D centroids.)
* All backend functions are pure — no PySide6 imports — per project convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .serialtrack.fields import (
    DisplacementField, StrainField, compute_gridded_strain,
)

Row = Dict[str, Any]
ProgressCB = Callable[[float, str], None]


# ═══════════════════════════════════════════════════════════════
#  Row extraction helpers
# ═══════════════════════════════════════════════════════════════

def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def has_z(rows: List[Row]) -> bool:
    """True if any row carries a non-null ``centroid_z_px`` (⇒ 3D dataset)."""
    return any(r.get("centroid_z_px") is not None for r in rows)


def _coord(row: Row, ndim: int) -> List[float]:
    if ndim == 3:
        return [_f(row.get("centroid_y_px")), _f(row.get("centroid_x_px")),
                _f(row.get("centroid_z_px"))]
    return [_f(row.get("centroid_y_px")), _f(row.get("centroid_x_px"))]


# ═══════════════════════════════════════════════════════════════
#  Track data container
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrackData:
    """Per-frame particle positions + chained trajectories for one group.

    Attributes
    ----------
    frames : list[int]
        Sorted unique frame indices that carry at least one detection.
    coords : dict[int, (N, D)]
        Particle positions at each frame (``(y, x[, z])``).
    track_ids : dict[int, (N,)]
        Track id per particle at each frame (``-1`` = untracked).
    trajectories : (N_tracks, n_frames, D)
        NaN-gapped position matrix, one row per distinct track id.
    traj_ids : (N_tracks,)
        The track id for each trajectory row.
    ndim : int
        Spatial dimensionality (2 or 3).
    pixel_size_um, z_step_um : float | None
        Physical scaling for lateral / axial axes.
    time_step : float
        Time between frames (physical units); velocity = displacement / time_step.
    """
    frames: List[int]
    coords: Dict[int, np.ndarray]
    track_ids: Dict[int, np.ndarray]
    trajectories: np.ndarray
    traj_ids: np.ndarray
    ndim: int
    pixel_size_um: Optional[float] = None
    z_step_um: Optional[float] = None
    time_step: float = 1.0

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def n_tracks(self) -> int:
        return int(self.trajectories.shape[0]) if self.trajectories.size else 0

    def frame_index(self, t: int) -> int:
        """Position of frame ``t`` within :attr:`frames` (``-1`` if absent)."""
        try:
            return self.frames.index(int(t))
        except ValueError:
            return -1

    def pixel_steps(self) -> np.ndarray:
        """``(D,)`` physical size per pixel for scaling to physical units."""
        px = self.pixel_size_um or 1.0
        if self.ndim == 3:
            return np.array([px, px, self.z_step_um or px], dtype=float)
        return np.array([px, px], dtype=float)


def build_track_data(
    rows: List[Row],
    m: Optional[int] = None,
    *,
    pixel_size_um: Optional[float] = None,
    z_step_um: Optional[float] = None,
    time_step: float = 1.0,
) -> TrackData:
    """Build :class:`TrackData` from ND2Studios measurement rows.

    Parameters
    ----------
    rows : list of measurement-row dicts
        Must carry ``frame``, ``centroid_y_px``, ``centroid_x_px`` and (for
        tracked rows) ``track_id``; ``centroid_z_px`` enables 3D.
    m : int, optional
        Restrict to one multipoint (``m_position``). ``None`` keeps all.
    pixel_size_um, z_step_um, time_step : physical scaling passed through.
    """
    if m is not None:
        rows = [r for r in rows if int(r.get("m_position", 0)) == int(m)]
    ndim = 3 if has_z(rows) else 2

    by_frame: Dict[int, List[Row]] = defaultdict(list)
    for r in rows:
        by_frame[int(r.get("frame", 0))].append(r)
    frames = sorted(f for f, rs in by_frame.items() if rs)

    coords: Dict[int, np.ndarray] = {}
    track_ids: Dict[int, np.ndarray] = {}
    for fr in frames:
        rs = by_frame[fr]
        coords[fr] = np.array([_coord(r, ndim) for r in rs], dtype=float)
        tids = []
        for r in rs:
            tid = r.get("track_id")
            tids.append(int(tid) if tid is not None else -1)
        track_ids[fr] = np.asarray(tids, dtype=np.int64)

    # Chain trajectories: distinct positive track ids → NaN-gapped matrix.
    all_ids = sorted({int(t) for fr in frames for t in track_ids[fr] if t >= 0})
    n_frames = len(frames)
    traj = np.full((len(all_ids), n_frames, ndim), np.nan, dtype=float)
    id_to_row = {tid: i for i, tid in enumerate(all_ids)}
    for fi, fr in enumerate(frames):
        c = coords[fr]
        for pi, tid in enumerate(track_ids[fr]):
            if tid >= 0:
                traj[id_to_row[int(tid)], fi] = c[pi]

    return TrackData(
        frames=frames, coords=coords, track_ids=track_ids,
        trajectories=traj, traj_ids=np.asarray(all_ids, dtype=np.int64),
        ndim=ndim, pixel_size_um=pixel_size_um, z_step_um=z_step_um,
        time_step=time_step,
    )


# ═══════════════════════════════════════════════════════════════
#  Per-particle displacement
# ═══════════════════════════════════════════════════════════════

def particle_displacement(
    td: TrackData, t: int, mode: str = "cumulative", ref_frame: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-particle positions and displacement vectors at frame ``t``.

    * ``mode="cumulative"`` — displacement from ``ref_frame`` (default: first
      frame) to ``t``, for tracks present in both. Positions returned are the
      *deformed* (frame-``t``) positions.
    * ``mode="incremental"`` — displacement from the previous present frame to
      ``t``.

    Returns ``(coords (M, D), disp (M, D))`` for the ``M`` tracked particles that
    have a valid displacement at ``t``. May be empty.
    """
    fi = td.frame_index(t)
    if fi < 0 or td.n_tracks == 0:
        return np.empty((0, td.ndim)), np.empty((0, td.ndim))

    if mode == "incremental":
        if fi == 0:
            return np.empty((0, td.ndim)), np.empty((0, td.ndim))
        ref_fi = fi - 1
    else:
        ref_fi = 0 if ref_frame is None else td.frame_index(ref_frame)
        if ref_fi < 0:
            ref_fi = 0

    pos_t = td.trajectories[:, fi, :]
    pos_r = td.trajectories[:, ref_fi, :]
    valid = np.isfinite(pos_t).all(axis=1) & np.isfinite(pos_r).all(axis=1)
    return pos_t[valid], (pos_t[valid] - pos_r[valid])


# ═══════════════════════════════════════════════════════════════
#  Gridded fields
# ═══════════════════════════════════════════════════════════════

@dataclass
class FieldBundle:
    """Gridded displacement + strain fields for one frame, plus the scattered
    per-particle vectors they were interpolated from."""
    t: int
    coords: np.ndarray            # (M, D) particle positions
    disp: np.ndarray              # (M, D) per-particle displacement
    disp_field: DisplacementField
    strain_field: StrainField
    ndim: int
    time_step: float

    @property
    def grids(self) -> Tuple[np.ndarray, ...]:
        return self.disp_field.grids

    @property
    def grid_shape(self) -> Tuple[int, ...]:
        return self.disp_field.shape


def compute_field_bundle(
    td: TrackData, t: int, *,
    mode: str = "cumulative",
    ref_frame: Optional[int] = None,
    grid_step: float = 24.0,
    smoothness: float = 0.0,
    physical: bool = False,
) -> Optional[FieldBundle]:
    """Scatter per-particle displacement to a grid and compute strain at ``t``.

    Displacement is gridded with the bounded Cell-Tracker interpolation (linear
    inside the tracked-particle convex hull, zero outside — never extrapolated),
    so the field cannot exceed the range of the per-particle vectors. Here
    ``smoothness`` is a **Gaussian smoothing sigma in grid cells** (0 = none),
    not an RBF weight.

    Returns ``None`` when there are too few tracked particles (< D+1) to form a
    field. ``physical=True`` scales displacements to physical units via the
    dataset pixel steps before gridding.
    """
    coords, disp = particle_displacement(td, t, mode=mode, ref_frame=ref_frame)
    if len(coords) < td.ndim + 1:
        return None
    ps = td.pixel_steps()
    gstep = np.full(td.ndim, float(grid_step), dtype=float)
    dfield, sfield = compute_gridded_strain(
        coords, disp, gstep, smoothness=smoothness,
        pixel_steps=ps if physical else None,
    )
    dfield.time_step = td.time_step
    return FieldBundle(
        t=int(t), coords=coords, disp=disp,
        disp_field=dfield, strain_field=sfield,
        ndim=td.ndim, time_step=td.time_step,
    )


# ═══════════════════════════════════════════════════════════════
#  Derived scalar / vector fields
# ═══════════════════════════════════════════════════════════════

def displacement_magnitude(fb: FieldBundle) -> np.ndarray:
    """``|u|`` on the grid — ``(*grid_shape,)``."""
    return np.sqrt(np.sum(fb.disp_field.components ** 2, axis=0))


def velocity_components(fb: FieldBundle) -> np.ndarray:
    """Velocity field ``u / time_step`` — ``(D, *grid_shape)``."""
    return fb.disp_field.velocity


def divergence(fb: FieldBundle) -> np.ndarray:
    """Divergence (dilatation / volumetric strain rate) = tr(∂u/∂x)."""
    F = fb.strain_field.F_tensor
    return np.trace(F, axis1=0, axis2=1)


def curl(fb: FieldBundle) -> np.ndarray:
    """Curl / vorticity of the displacement field.

    2D → scalar out-of-plane component ``∂u_x/∂y − ∂u_y/∂x`` (``(*grid,)``).
    3D → vector ``(3, *grid)`` = (ω_x, ω_y, ω_z) in (y, x, z) index space.

    Component index 0 = y, 1 = x, 2 = z; ``F[i, j] = ∂u_i/∂x_j`` where axis j is
    the same (y, x, z) order.
    """
    F = fb.strain_field.F_tensor
    if fb.ndim == 2:
        # ∂u_x/∂y − ∂u_y/∂x = F[1,0] − F[0,1]
        return F[1, 0] - F[0, 1]
    # 3D vorticity ω = ∇ × u, with axes (y=0, x=1, z=2):
    #   ω_y = ∂u_z/∂x − ∂u_x/∂z = F[2,1] − F[1,2]
    #   ω_x = ∂u_y/∂z − ∂u_z/∂y = F[0,2] − F[2,0]
    #   ω_z = ∂u_x/∂y − ∂u_y/∂x = F[1,0] − F[0,1]
    return np.stack([F[0, 2] - F[2, 0], F[2, 1] - F[1, 2], F[1, 0] - F[0, 1]])


def jacobian(fb: FieldBundle) -> np.ndarray:
    """Jacobian ``J = det(I + ∂u/∂x)`` — local volume change (1.0 = none)."""
    F = fb.strain_field.F_tensor
    ndim = fb.ndim
    eye = np.eye(ndim).reshape(ndim, ndim, *([1] * (F.ndim - 2)))
    defgrad = F + eye
    # Move the two tensor axes last so np.linalg.det operates per grid point.
    moved = np.moveaxis(defgrad, (0, 1), (-2, -1))
    return np.linalg.det(moved)


def effective_strain(fb: FieldBundle) -> np.ndarray:
    """Von-Mises-equivalent (effective) strain from the strain tensor.

    ``ε_eff = sqrt( (2/3) · e_ij e_ij )`` with ``e`` the deviatoric strain.
    """
    eps = fb.strain_field.eps_tensor
    ndim = fb.ndim
    tr = np.trace(eps, axis1=0, axis2=1) / ndim
    dev = eps.copy()
    for i in range(ndim):
        dev[i, i] = dev[i, i] - tr
    return np.sqrt((2.0 / 3.0) * np.sum(dev ** 2, axis=(0, 1)))


# ═══════════════════════════════════════════════════════════════
#  Stress (constitutive models) + Von Mises
# ═══════════════════════════════════════════════════════════════

def compute_stress(
    fb: FieldBundle, *,
    youngs_modulus: float = 1.0,
    poisson_ratio: float = 0.3,
    model: str = "linear_isotropic",
) -> Tuple[np.ndarray, np.ndarray]:
    """Cauchy stress tensor and Von Mises stress from the strain field.

    Parameters
    ----------
    youngs_modulus : E (stress units of choice; scales the result linearly).
    poisson_ratio : ν in ``[0, 0.5)``.
    model : currently ``"linear_isotropic"`` (Hooke's law). Placeholder for
        future Neo-Hookean / Mooney-Rivlin ports from SerialTrack_Python.

    Returns ``(sigma (D, D, *grid), von_mises (*grid))``.
    """
    eps = fb.strain_field.eps_tensor
    ndim = fb.ndim
    E, nu = float(youngs_modulus), float(poisson_ratio)
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    tr = np.trace(eps, axis1=0, axis2=1)
    sigma = 2.0 * mu * eps.copy()
    for i in range(ndim):
        sigma[i, i] = sigma[i, i] + lam * tr

    vm = von_mises_from_sigma(sigma, ndim, poisson_ratio=nu)
    return sigma, vm


def von_mises_from_sigma(sigma: np.ndarray, ndim: int,
                         poisson_ratio: float = 0.3) -> np.ndarray:
    """Von Mises stress from a Cauchy stress tensor ``(D, D, *grid)``.

    For 2D we assume plane strain: ``σ_zz = ν (σ_xx + σ_yy)``.
    """
    if ndim == 3:
        sxx, syy, szz = sigma[0, 0], sigma[1, 1], sigma[2, 2]
        sxy, syz, szx = sigma[0, 1], sigma[1, 2], sigma[2, 0]
    else:
        sxx, syy = sigma[0, 0], sigma[1, 1]
        sxy = sigma[0, 1]
        szz = poisson_ratio * (sxx + syy)
        syz = szx = np.zeros_like(sxx)
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2)
    )


# ═══════════════════════════════════════════════════════════════
#  Named-scalar dispatcher (for the panel's "Field" combo)
# ═══════════════════════════════════════════════════════════════

#: Component index by axis letter (matches the (y, x, z) storage order).
_AXIS = {"y": 0, "x": 1, "z": 2}


def scalar_field(fb: FieldBundle, key: str, **stress_kw) -> np.ndarray:
    """Return a named 2D/3D scalar field for plotting.

    Supported keys: ``disp_mag``, ``u_y``/``u_x``/``u_z``, ``v_y``/``v_x``/``v_z``,
    ``e_yy``/``e_xx``/``e_zz``/``e_xy``/``e_yz``/``e_xz``, ``div``, ``curl``
    (2D scalar / 3D ω_z), ``detF``, ``eff_strain``, ``von_mises``,
    ``s_yy``/``s_xx``/…​ (stress components).
    """
    if key == "disp_mag":
        return displacement_magnitude(fb)
    if key.startswith("u_"):
        return fb.disp_field.components[_AXIS[key[2]]]
    if key.startswith("v_"):
        return velocity_components(fb)[_AXIS[key[2]]]
    if key.startswith("e_"):
        i, j = _AXIS[key[2]], _AXIS[key[3]]
        return fb.strain_field.eps_tensor[i, j]
    if key == "div":
        return divergence(fb)
    if key == "curl":
        c = curl(fb)
        return c if fb.ndim == 2 else c[2]  # ω_z for 3D
    if key == "detF":
        return jacobian(fb)
    if key == "eff_strain":
        return effective_strain(fb)
    if key == "von_mises":
        _, vm = compute_stress(fb, **stress_kw)
        return vm
    if key.startswith("s_"):
        sigma, _ = compute_stress(fb, **stress_kw)
        i, j = _AXIS[key[2]], _AXIS[key[3]]
        return sigma[i, j]
    raise KeyError(f"Unknown scalar field key: {key!r}")
