"""viz3d.overlays — turn measurement results into 3-D geometry primitives.

Pure numpy: no Qt, no PyVista/VTK. Results are consumed **duck-typed** so this
module never imports the (possibly Qt-tainted) modules that define them:

- DVC: a ``DVCResult``-like object with ``grid_coords`` ``(*grid, d)``,
  ``displacement_field`` ``(*grid, d)``, ``voxel_size_um`` ``(z, y, x)`` /
  ``(y, x)``, optional ``strain_field`` ``(*grid, d, d)`` and ``qfactor``
  ``(*grid,)``. Native axis order is slowest-first: ``(z, y, x)`` (3-D) or
  ``(y, x)`` (2-D). Coordinates/displacements are in voxels; we scale by
  ``voxel_size_um`` to micrometers and reorder to world ``(x, y, z)``.
- PTV/SerialTrack: a ``TrackData``-like object with ``trajectories``
  ``(N_tracks, n_frames, D)`` NaN-gapped, ``traj_ids`` ``(N_tracks,)``, ``ndim``,
  ``pixel_size_um``, ``z_step_um`` and ``time_step``. Coordinate order is
  ``(y, x[, z])`` in pixels. When ``D == 2`` (all current ND2Studios data), the
  third render axis is time (space-time tubes); ``D == 3`` renders true depth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nd2studios.backend.viz3d.mdm import MDMResult


# ── DVC ──────────────────────────────────────────────────────────────────────
@dataclass
class DVCField:
    """A DVC displacement/strain field as world-space glyph data.

    ``points_um`` / ``vectors_um`` are ``(N, 3)`` in world ``(x, y, z)`` µm
    (z padded to 0 for 2-D DIC). ``scalars`` maps a field name to a ``(N,)``
    array for coloring. ``grid_shape`` is the native ``(Gz, Gy, Gx)`` /
    ``(Gy, Gx)`` so the widget can build a ``StructuredGrid`` for ``warp_by_vector``.
    """

    points_um: np.ndarray
    vectors_um: np.ndarray
    scalars: Dict[str, np.ndarray]
    grid_shape: Tuple[int, ...]
    default_scalar: str = "disp_mag"

    @property
    def n_points(self) -> int:
        return int(self.points_um.shape[0])


@dataclass
class PtvTracks:
    """Particle trajectories as world-space polyline segments.

    Each entry of ``segments`` is a ``(P_i, 3)`` array of world ``(x, y, z_or_t)``
    points for one contiguous (NaN-free) run of a track; ``seg_track_ids`` and
    ``seg_scalars`` (per-vertex color values) are parallel lists.
    """

    segments: List[np.ndarray] = field(default_factory=list)
    seg_track_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), int))
    seg_scalars: List[np.ndarray] = field(default_factory=list)
    third_axis: str = "time"      # "time" | "z"
    color_by: str = "time"        # "time" | "velocity"

    @property
    def n_segments(self) -> int:
        return len(self.segments)


def _reorder_to_xyz(vec_nd: np.ndarray, d: int) -> np.ndarray:
    """Map native ``(z, y, x)`` (3-D) / ``(y, x)`` (2-D) rows to world ``(x, y, z)``."""
    n = vec_nd.shape[0]
    out = np.zeros((n, 3), dtype=np.float64)
    if d >= 3:
        out[:, 0] = vec_nd[:, 2]   # x
        out[:, 1] = vec_nd[:, 1]   # y
        out[:, 2] = vec_nd[:, 0]   # z
    else:
        out[:, 0] = vec_nd[:, 1]   # x
        out[:, 1] = vec_nd[:, 0]   # y
        out[:, 2] = 0.0
    return out


def dvc_field(result: Any,
              scalar_keys: Optional[List[str]] = None) -> DVCField:
    """Adapt a ``DVCResult``-like object into a :class:`DVCField`.

    Scalars produced (subset available depends on what the result carries):
    ``disp_mag`` (µm), ``u_x``/``u_y``/``u_z`` (µm), strain components
    (``e_xx``, ``e_yy``, ``e_zz``, ``e_xy``, ``e_xz``, ``e_yz`` as available),
    ``eff_strain`` (von-Mises-style deviatoric magnitude) and ``qfactor``.
    Pass ``scalar_keys`` to restrict the returned set.
    """
    gc = np.asarray(result.grid_coords, dtype=np.float64)
    df = np.asarray(result.displacement_field, dtype=np.float64)
    d = int(gc.shape[-1])
    grid_shape = tuple(int(s) for s in gc.shape[:-1])
    n = int(np.prod(grid_shape)) if grid_shape else 0

    coords = gc.reshape(n, d)
    disp = df.reshape(n, d)

    vsz = tuple(float(v) for v in (getattr(result, "voxel_size_um", ()) or ()))
    if len(vsz) != d:
        vsz = (1.0,) * d
    scale = np.asarray(vsz, dtype=np.float64)     # native (z,y,x) / (y,x) order

    points = _reorder_to_xyz(coords * scale, d)
    vectors = _reorder_to_xyz(disp * scale, d)

    scalars: Dict[str, np.ndarray] = {}
    scalars["disp_mag"] = np.linalg.norm(vectors, axis=1)
    scalars["u_x"] = vectors[:, 0]
    scalars["u_y"] = vectors[:, 1]
    if d >= 3:
        scalars["u_z"] = vectors[:, 2]

    sf = getattr(result, "strain_field", None)
    if sf is not None:
        strain = np.asarray(sf, dtype=np.float64).reshape(n, d, d)
        sym = 0.5 * (strain + np.transpose(strain, (0, 2, 1)))
        # Index→axis: native order is (z,y,x) in 3-D → 0=z,1=y,2=x; (y,x) in 2-D.
        if d >= 3:
            comp = {"e_zz": (0, 0), "e_yy": (1, 1), "e_xx": (2, 2),
                    "e_yz": (0, 1), "e_xz": (0, 2), "e_xy": (1, 2)}
        else:
            comp = {"e_yy": (0, 0), "e_xx": (1, 1), "e_xy": (0, 1)}
        for name, (i, j) in comp.items():
            scalars[name] = sym[:, i, j]
        trace = np.trace(sym, axis1=1, axis2=2)
        dev = sym - (trace / d)[:, None, None] * np.eye(d)[None, :, :]
        scalars["eff_strain"] = np.sqrt((2.0 / 3.0) * np.sum(dev * dev, axis=(1, 2)))

    qf = getattr(result, "qfactor", None)
    if qf is not None:
        q = np.asarray(qf, dtype=np.float64).reshape(-1)
        if q.shape[0] == n:
            scalars["qfactor"] = q

    if scalar_keys is not None:
        scalars = {k: v for k, v in scalars.items() if k in scalar_keys}

    return DVCField(points_um=points, vectors_um=vectors, scalars=scalars,
                    grid_shape=grid_shape, default_scalar="disp_mag")


# ── PTV / SerialTrack ────────────────────────────────────────────────────────
def _contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return ``[start, end)`` index pairs for each run of True in ``mask``."""
    runs: List[Tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _track_world(seg_px: np.ndarray, frames: np.ndarray, d: int,
                 px: float, z_step: float, time_step: float,
                 third_axis: str) -> np.ndarray:
    """Map one track segment's native ``(y, x[, z])`` px rows to world ``(x, y, z_or_t)``."""
    p = seg_px.shape[0]
    out = np.zeros((p, 3), dtype=np.float64)
    out[:, 0] = seg_px[:, 1] * px      # x
    out[:, 1] = seg_px[:, 0] * px      # y
    if d >= 3:
        out[:, 2] = seg_px[:, 2] * z_step
    elif third_axis == "time":
        out[:, 2] = frames.astype(np.float64) * time_step
    else:
        out[:, 2] = 0.0
    return out


def _segment_velocity(world: np.ndarray, frames: np.ndarray,
                      d: int, time_step: float) -> np.ndarray:
    """Per-vertex speed (world units / time) from spatial displacement."""
    p = world.shape[0]
    v = np.zeros(p, dtype=np.float64)
    if p < 2:
        return v
    spatial = world[:, :3] if d >= 3 else world[:, :2]
    dist = np.linalg.norm(np.diff(spatial, axis=0), axis=1)
    dt = np.diff(frames).astype(np.float64) * (time_step if time_step > 0 else 1.0)
    dt[dt == 0] = time_step if time_step > 0 else 1.0
    seg_v = dist / dt
    v[:-1] = seg_v
    v[-1] = seg_v[-1]
    return v


def ptv_polylines(track_data: Any,
                  up_to_frame: Optional[int] = None,
                  third_axis: str = "time",
                  max_tracks: int = 800,
                  color_by: str = "time") -> PtvTracks:
    """Adapt a ``TrackData``-like object into :class:`PtvTracks`.

    NaN-gapped trajectories are split into contiguous finite runs (each ≥ 2
    points becomes a polyline). Coloring is per-vertex: ``"time"`` → fraction
    along the full trajectory; ``"velocity"`` → spatial speed. ``up_to_frame``
    truncates to a playback position; ``max_tracks`` caps track count.
    """
    traj = np.asarray(getattr(track_data, "trajectories", np.zeros((0, 0, 2))))
    if traj.ndim != 3 or traj.shape[0] == 0 or traj.shape[1] == 0:
        return PtvTracks(third_axis=third_axis, color_by=color_by)

    n_tracks, n_frames, d = traj.shape
    ids = np.asarray(getattr(track_data, "traj_ids", np.arange(n_tracks)))
    px = float(getattr(track_data, "pixel_size_um", 1.0) or 1.0)
    z_step = float(getattr(track_data, "z_step_um", 1.0) or 1.0)
    time_step = float(getattr(track_data, "time_step", 1.0) or 1.0)

    fi_max = n_frames if up_to_frame is None else min(n_frames, int(up_to_frame) + 1)
    if fi_max <= 0:
        return PtvTracks(third_axis=third_axis, color_by=color_by)

    denom = max(1, n_frames - 1)
    n_use = min(n_tracks, int(max_tracks))

    segments: List[np.ndarray] = []
    seg_ids: List[int] = []
    seg_scalars: List[np.ndarray] = []

    frame_axis = np.arange(fi_max)
    for k in range(n_use):
        path = traj[k, :fi_max, :]                    # (fi_max, d)
        finite = np.isfinite(path).all(axis=1)
        for a, b in _contiguous_runs(finite):
            if b - a < 2:
                continue
            seg_px = path[a:b]
            frames = frame_axis[a:b]
            world = _track_world(seg_px, frames, d, px, z_step, time_step, third_axis)
            if color_by == "velocity":
                sc = _segment_velocity(world, frames, d, time_step)
            else:
                sc = frames.astype(np.float64) / denom
            segments.append(world)
            seg_ids.append(int(ids[k]))
            seg_scalars.append(sc)

    return PtvTracks(segments=segments,
                     seg_track_ids=np.asarray(seg_ids, dtype=int),
                     seg_scalars=seg_scalars,
                     third_axis=third_axis, color_by=color_by)


def label_volume(masks: np.ndarray) -> np.ndarray:
    """Coerce a label mask to a ``(Z, H, W)`` int volume (2-D → single plane)."""
    a = np.asarray(masks)
    if a.ndim == 2:
        a = a[None, ...]
    return a.astype(np.int32, copy=False)


# ── DVC-on-object (mask) ─────────────────────────────────────────────────────
@dataclass
class MaskedField:
    """A DVC scalar field interpolated onto a hand-drawn 3-D object mask.

    The sparse DVC subset grid is up-sampled (linear) onto the mask's voxel grid,
    so the object's **surface boundary** and **interior** can be colored by the
    field at much finer resolution than the DVC sampling. Blocks are cropped to the
    object's bounding box (and strided to a voxel budget) to stay renderable.

    - ``mask`` ``(Z, H, W)`` bool — the object within its bounding box.
    - ``scalars`` maps a field name to a ``(Z, H, W)`` float volume, ``NaN``
      outside the object.
    - ``spacing`` ``(dz, dy, dx)`` µm (stride-adjusted), VTK ``ImageData`` axis
      order; ``origin_um`` is the block origin in world ``(x, y, z)`` µm.
    """

    mask: np.ndarray
    scalars: Dict[str, np.ndarray]
    spacing: Tuple[float, float, float]
    origin_um: Tuple[float, float, float]
    default_scalar: str = "disp_mag"
    n_object_voxels: int = 0

    @property
    def scalar_names(self) -> List[str]:
        return list(self.scalars.keys())

    @property
    def is_empty(self) -> bool:
        return self.mask.size == 0 or self.n_object_voxels == 0


def _bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int, int, int]]:
    """Tight ``(z0, z1, y0, y1, x0, x1)`` bounds of a boolean volume, or None."""
    if not mask.any():
        return None
    zz, yy, xx = np.where(mask)
    return (int(zz.min()), int(zz.max()) + 1, int(yy.min()), int(yy.max()) + 1,
            int(xx.min()), int(xx.max()) + 1)


def _grid_axes_vox(grid_coords: np.ndarray, d: int) -> List[np.ndarray]:
    """Per-axis 1-D coordinate arrays (voxels) from a regular ``grid_coords``.

    ``grid_coords`` is ``(*grid, d)`` with the last axis holding native
    ``(z, y, x)`` (3-D) / ``(y, x)`` (2-D) coordinates; the grid is regular so each
    axis's coordinates are constant along the other axes.
    """
    if d >= 3:
        return [np.asarray(grid_coords[:, 0, 0, 0], dtype=np.float64),
                np.asarray(grid_coords[0, :, 0, 1], dtype=np.float64),
                np.asarray(grid_coords[0, 0, :, 2], dtype=np.float64)]
    return [np.asarray(grid_coords[:, 0, 0], dtype=np.float64),
            np.asarray(grid_coords[0, :, 1], dtype=np.float64)]


def dvc_object_field(result: Any,
                     mask: np.ndarray,
                     mask_voxel_size_um: Optional[Tuple[float, float, float]] = None,
                     *,
                     z_offset_um: float = 0.0,
                     scalar_keys: Optional[List[str]] = None,
                     max_box_voxels: int = 4_000_000) -> MaskedField:
    """Interpolate a DVC field onto a 3-D object ``mask`` → :class:`MaskedField`.

    The DVC field lives on a sparse regular grid of subset centers (in voxels);
    the ``mask`` is a full-resolution ``(Z, H, W)`` boolean object drawn on the raw
    volume. Both are placed in physical µm — DVC nodes at
    ``grid_coords × voxel_size_um`` (which already folds in any XY downsample),
    mask voxels at ``index × mask_voxel_size_um`` — and each requested scalar is
    linearly interpolated (``scipy.interpolate.RegularGridInterpolator``,
    extrapolating past the grid inset) onto the object's voxels. The result is
    cropped to the object's bounding box and strided so it never exceeds
    ``max_box_voxels``.

    ``z_offset_um`` accounts for a DVC ``z_start`` crop (DVC ``z=0`` maps to raw
    ``z = z_start``); leave 0 for a full-Z DVC run (the common case). A 2-D DIC
    field (or a single-Z-plane grid) is interpolated in-plane and broadcast over Z.
    """
    from scipy.interpolate import RegularGridInterpolator

    mask = np.asarray(mask)
    if mask.ndim == 2:
        mask = mask[None, ...]
    mask = mask.astype(bool)
    if mask_voxel_size_um and len(mask_voxel_size_um) == 3:
        dz, dy, dx = (float(mask_voxel_size_um[0]), float(mask_voxel_size_um[1]),
                      float(mask_voxel_size_um[2]))
    else:
        dz = dy = dx = 1.0
    dz = dz if dz > 0 else 1.0
    dy = dy if dy > 0 else 1.0
    dx = dx if dx > 0 else 1.0

    field = dvc_field(result, scalar_keys=scalar_keys)
    grid_shape = field.grid_shape
    gc = np.asarray(result.grid_coords, dtype=np.float64)
    d = int(gc.shape[-1])
    vsz = tuple(float(v) for v in (getattr(result, "voxel_size_um", ()) or ()))
    if len(vsz) != d:
        vsz = (1.0,) * d

    axes_um = [np.asarray(a) * vsz[i] for i, a in enumerate(_grid_axes_vox(gc, d))]
    if d >= 3:
        axes_um[0] = axes_um[0] + float(z_offset_um)

    bounds = _bbox(mask)
    empty = MaskedField(mask=np.zeros((0, 0, 0), bool), scalars={},
                        spacing=(dz, dy, dx), origin_um=(0.0, 0.0, 0.0),
                        default_scalar=field.default_scalar)
    if bounds is None or field.n_points == 0:
        return empty
    z0, z1, y0, y1, x0, x1 = bounds

    stride = 1
    def _n(v: int, s: int) -> int:
        return (v + s - 1) // s
    while (_n(z1 - z0, stride) * _n(y1 - y0, stride) * _n(x1 - x0, stride)
           > int(max_box_voxels)) and stride < 64:
        stride += 1

    zi = np.arange(z0, z1, stride)
    yi = np.arange(y0, y1, stride)
    xi = np.arange(x0, x1, stride)
    sub_mask = mask[z0:z1:stride, y0:y1:stride, x0:x1:stride]
    if sub_mask.size == 0:
        return empty

    # Interpolate in 3-D when the grid has depth; otherwise interpolate in-plane
    # (2-D DIC or a single-Z-plane grid) and broadcast the result across Z.
    use_3d = d >= 3 and all(len(a) >= 2 for a in axes_um)
    zq, yq, xq = zi * dz, yi * dy, xi * dx

    def _interp_scalar(flat: np.ndarray) -> np.ndarray:
        grid = np.asarray(flat, dtype=np.float64).reshape(grid_shape)
        if use_3d:
            rgi = RegularGridInterpolator(tuple(axes_um), grid,
                                          bounds_error=False, fill_value=None)
            ZZ, YY, XX = np.meshgrid(zq, yq, xq, indexing="ij")
            pts = np.stack([ZZ.ravel(), YY.ravel(), XX.ravel()], axis=1)
            return rgi(pts).reshape(sub_mask.shape)
        # 2-D / shallow: collapse any leading size-1 grid axis to a (Gy,Gx) plane.
        grid2 = grid.reshape(grid_shape[-2:]) if d >= 3 else grid
        ay, ax = (axes_um[-2], axes_um[-1])
        rgi = RegularGridInterpolator((ay, ax), grid2,
                                      bounds_error=False, fill_value=None)
        YY, XX = np.meshgrid(yq, xq, indexing="ij")
        plane = rgi(np.stack([YY.ravel(), XX.ravel()], axis=1)).reshape(
            sub_mask.shape[1], sub_mask.shape[2])
        return np.repeat(plane[None, ...], sub_mask.shape[0], axis=0)

    scalars_out: Dict[str, np.ndarray] = {}
    for name, flat in field.scalars.items():
        try:
            vals = _interp_scalar(flat)
        except Exception:  # noqa: BLE001 — degenerate grid; skip this scalar
            continue
        scalars_out[name] = np.where(sub_mask, vals, np.nan).astype(np.float32)

    return MaskedField(
        mask=sub_mask,
        scalars=scalars_out,
        spacing=(dz * stride, dy * stride, dx * stride),
        origin_um=(float(x0) * dx, float(y0) * dy, float(z0) * dz),
        default_scalar=(field.default_scalar if field.default_scalar in scalars_out
                        else (next(iter(scalars_out), "disp_mag"))),
        n_object_voxels=int(sub_mask.sum()),
    )


# ── DVC-on-object surface (V1.68 — Stout et al. 2016 MDM / UV-wrap) ────────────
@dataclass
class SurfaceField:
    """A DVC displacement field carried on a **smoothed closed surface mesh**.

    Supersedes :class:`MaskedField` as the primary object render (V1.68): instead
    of coloring an interpolated voxel block, the object's boundary ``∂V`` is a
    marching-cubes + Taubin mesh and the measured displacement is sampled onto its
    vertices and decomposed into normal / tangential parts (Stout et al. 2016).

    - ``vertices_um`` ``(Nv, 3)`` world ``(x, y, z)`` µm; ``faces`` ``(Nf, 3)`` int.
    - ``vertex_normals`` ``(Nv, 3)`` unit outward normals.
    - ``scalars`` per-vertex ``(Nv,)`` arrays: ``disp_mag``, ``u_x``/``u_y``/``u_z``,
      ``u_perp`` (signed, +outward), ``u_par`` (tangential magnitude), plus any DVC
      strain / ``qfactor`` scalars sampled on the surface.
    - ``u_par_vec`` ``(Nv, 3)`` tangential displacement vectors (for streamlines).
    - ``centroid_um`` ``(3,)`` mesh centroid (the UV-unwrap parameterization pole).
    - ``mdm`` the :class:`~nd2studios.backend.viz3d.mdm.MDMResult` for this frame
      (``None`` if metrics were skipped); ``interior`` an optional voxel
      :class:`MaskedField` for the inside-the-object render.
    """

    vertices_um: np.ndarray
    faces: np.ndarray
    vertex_normals: np.ndarray
    scalars: Dict[str, np.ndarray]
    u_par_vec: np.ndarray
    centroid_um: np.ndarray
    default_scalar: str = "disp_mag"
    mdm: Optional[MDMResult] = None
    interior: Optional["MaskedField"] = None
    dim: int = 3
    n_object_voxels: int = 0

    @property
    def n_vertices(self) -> int:
        return int(self.vertices_um.shape[0])

    @property
    def scalar_names(self) -> List[str]:
        return list(self.scalars.keys())

    @property
    def is_empty(self) -> bool:
        return self.vertices_um.size == 0 or self.faces.size == 0


@dataclass
class UnwrapMap:
    """A 2-D cartographic unwrap of a surface scalar (Stout et al. Fig 4E–G).

    ``values`` ``(H, W)`` is the chosen scalar rasterized into the projection
    (``NaN`` off the map); ``coverage`` marks valid pixels. ``vec_u`` / ``vec_v``
    are the tangential displacement's east / north components for ``u∥``
    streamlines. ``projection`` is ``"mollweide"`` (equal-area) or
    ``"equirectangular"``.
    """

    values: np.ndarray
    coverage: np.ndarray
    vec_u: np.ndarray
    vec_v: np.ndarray
    projection: str = "mollweide"
    scalar: str = "u_perp"

    @property
    def is_empty(self) -> bool:
        return self.values.size == 0 or not bool(np.any(self.coverage))


def _empty_surface_field() -> SurfaceField:
    z3 = np.zeros((0, 3), dtype=np.float64)
    return SurfaceField(vertices_um=z3, faces=np.zeros((0, 3), dtype=np.int64),
                        vertex_normals=z3.copy(), scalars={}, u_par_vec=z3.copy(),
                        centroid_um=np.zeros(3, dtype=np.float64))


def dvc_object_surface(result: Any,
                       mask: np.ndarray,
                       mask_voxel_size_um: Optional[Tuple[float, float, float]] = None,
                       *,
                       z_offset_um: float = 0.0,
                       scalar_keys: Optional[List[str]] = None,
                       smooth_iterations: int = 10,
                       with_interior: bool = False,
                       with_metrics: bool = True) -> SurfaceField:
    """Build the DVC-on-object **surface** render (V1.68 primary object adapter).

    Orchestrates :mod:`~nd2studios.backend.viz3d.surface` +
    :mod:`~nd2studios.backend.viz3d.mdm`: marching-cubes + Taubin the ``mask`` into
    a closed surface (physical µm), sample the DVC displacement onto its vertices,
    decompose into ``u⊥`` / ``u∥``, compute the Mean Deformation Metrics, and
    (optionally) attach a voxel :class:`MaskedField` for the interior. Consumes
    ``result`` (a ``DVCResult``) **read-only** — nothing in ``backend/dvc/`` is
    imported or modified. ``z_offset_um`` mirrors :func:`dvc_object_field` (a DVC
    ``z_start`` crop; 0 for the common full-Z run).
    """
    from nd2studios.backend.viz3d.surface import (
        build_object_surface, decompose_surface_displacement,
        sample_displacement_on_surface, sample_scalars_on_surface,
    )
    from nd2studios.backend.viz3d.mdm import (
        deformation_metrics, mean_displacement_gradient,
    )

    surf = build_object_surface(mask, mask_voxel_size_um,
                                smooth_iterations=int(smooth_iterations))
    if surf.is_empty or np.asarray(result.grid_coords).size == 0:
        return _empty_surface_field()

    u_vert = sample_displacement_on_surface(result, surf, z_offset_um=z_offset_um)
    u_perp, u_par_vec, u_par_mag = decompose_surface_displacement(
        u_vert, surf.vertex_normals)

    # Start from the DVC strain / qfactor scalars sampled on the surface, then let
    # the displacement-derived scalars (consistent with u_perp/u_par) win.
    scalars = sample_scalars_on_surface(result, surf, scalar_keys=scalar_keys,
                                        z_offset_um=z_offset_um)
    scalars["disp_mag"] = np.linalg.norm(u_vert, axis=1)
    scalars["u_x"] = u_vert[:, 0]
    scalars["u_y"] = u_vert[:, 1]
    scalars["u_z"] = u_vert[:, 2]
    scalars["u_perp"] = u_perp
    scalars["u_par"] = u_par_mag

    mdm = None
    if with_metrics:
        grad_u = mean_displacement_gradient(surf, u_vert)
        mdm = deformation_metrics(grad_u, dim=int(getattr(result, "dim", 3)))

    interior = None
    if with_interior:
        interior = dvc_object_field(result, mask, mask_voxel_size_um,
                                    z_offset_um=z_offset_um, scalar_keys=scalar_keys)

    return SurfaceField(
        vertices_um=surf.vertices_um, faces=surf.faces,
        vertex_normals=surf.vertex_normals, scalars=scalars,
        u_par_vec=u_par_vec, centroid_um=surf.centroid_um,
        default_scalar="disp_mag", mdm=mdm, interior=interior,
        dim=int(getattr(result, "dim", 3)),
        n_object_voxels=int(np.asarray(mask).astype(bool).sum()),
    )


def merge_surface_fields(fields: List[SurfaceField],
                         offsets_um: Optional[List[Tuple[float, float, float]]] = None
                         ) -> SurfaceField:
    """Concatenate several per-object :class:`SurfaceField`\\s into one composite
    (the DVC "All granules" render, V1.74).

    Each granule's surface is built independently in its **own crop-local** frame
    (vertices near the origin), so ``offsets_um`` (one world ``(x, y, z)`` µm
    translation per field, index-aligned with ``fields`` **before** empties are
    dropped) places each component back at its true relative position; omit it to
    merge in-place (surfaces already share a frame). The composite is a plain
    concatenation with a running vertex-index offset applied to each field's
    ``faces``, so rendering one merged mesh reuses the existing single-surface path
    (``PyVista3DViewer._add_surface_field_overlay``) with **no viewer changes** —
    the disjoint components draw as separate blobs under one shared colour scale
    (good for cross-granule comparison).

    Scalars are the **union** of every field's keys; a field lacking a key
    contributes ``NaN`` for its vertices (ignored by the viewer's percentile clim
    and ``nan_to_num``), so a scalar present on only some granules still colours
    those it exists on. ``mdm`` is ``None`` (a composite Mean-Deformation-Metric is
    ill-defined — pick a single granule for MDM) and ``interior`` is dropped.
    Empty / ``None`` fields are skipped; an all-empty input yields the empty
    surface; a single non-empty input is returned (translated by its offset).
    """
    offs = list(offsets_um) if offsets_um is not None else None
    pairs = [(f, (offs[i] if offs is not None and i < len(offs) else None))
             for i, f in enumerate(fields)
             if f is not None and not getattr(f, "is_empty", True)]
    if not pairs:
        return _empty_surface_field()

    def _shift(v: np.ndarray, off) -> np.ndarray:
        if off is None:
            return v
        return v + np.asarray(off, dtype=np.float64).reshape(1, 3)

    if len(pairs) == 1:
        f, off = pairs[0]
        if off is None:
            return f
        v = _shift(np.asarray(f.vertices_um, dtype=np.float64), off)
        return SurfaceField(
            vertices_um=v, faces=np.asarray(f.faces, dtype=np.int64),
            vertex_normals=np.asarray(f.vertex_normals, dtype=np.float64),
            scalars={k: np.asarray(a, dtype=np.float64).reshape(-1)
                     for k, a in f.scalars.items()},
            u_par_vec=np.asarray(f.u_par_vec, dtype=np.float64),
            centroid_um=v.mean(axis=0), default_scalar=f.default_scalar,
            mdm=None, interior=None, dim=int(getattr(f, "dim", 3)),
            n_object_voxels=int(getattr(f, "n_object_voxels", 0)))

    live = [f for f, _off in pairs]
    verts: List[np.ndarray] = []
    normals: List[np.ndarray] = []
    upar: List[np.ndarray] = []
    faces: List[np.ndarray] = []
    offset = 0
    per_field_nv: List[int] = []
    all_keys: List[str] = []
    for f, off in pairs:
        v = _shift(np.asarray(f.vertices_um, dtype=np.float64), off)
        nv = int(v.shape[0])
        verts.append(v)
        normals.append(np.asarray(f.vertex_normals, dtype=np.float64))
        upar.append(np.asarray(f.u_par_vec, dtype=np.float64))
        faces.append(np.asarray(f.faces, dtype=np.int64) + offset)
        offset += nv
        per_field_nv.append(nv)
        for k in f.scalars.keys():
            if k not in all_keys:
                all_keys.append(k)

    scalars: Dict[str, np.ndarray] = {}
    for k in all_keys:
        parts = []
        for f, nv in zip(live, per_field_nv):
            arr = f.scalars.get(k)
            if arr is None:
                parts.append(np.full((nv,), np.nan, dtype=np.float64))
            else:
                parts.append(np.asarray(arr, dtype=np.float64).reshape(-1))
        scalars[k] = np.concatenate(parts)

    merged_verts = np.concatenate(verts, axis=0)
    default = ("disp_mag" if "disp_mag" in scalars
               else (all_keys[0] if all_keys else "disp_mag"))
    return SurfaceField(
        vertices_um=merged_verts,
        faces=np.concatenate(faces, axis=0),
        vertex_normals=np.concatenate(normals, axis=0),
        scalars=scalars,
        u_par_vec=np.concatenate(upar, axis=0),
        centroid_um=merged_verts.mean(axis=0),
        default_scalar=default,
        mdm=None,
        interior=None,
        dim=int(getattr(live[0], "dim", 3)),
        n_object_voxels=int(sum(int(getattr(f, "n_object_voxels", 0))
                               for f in live)),
    )


def _sphere_lonlat(vertices_um: np.ndarray, centroid_um: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Genus-0 spherical parameterization: vertex direction → (lon, lat) + basis.

    Returns ``(lon, lat, dirs)`` where ``lon ∈ (−π, π]``, ``lat ∈ [−π/2, π/2]`` and
    ``dirs`` is the unit direction from the centroid (used to build the local
    east/north tangent frame for the ``u∥`` field).
    """
    d = np.asarray(vertices_um, dtype=np.float64) - np.asarray(centroid_um, float)
    r = np.linalg.norm(d, axis=1)
    r_safe = np.where(r > 1e-9, r, 1.0)
    dirs = d / r_safe[:, None]
    lon = np.arctan2(dirs[:, 1], dirs[:, 0])
    lat = np.arcsin(np.clip(dirs[:, 2], -1.0, 1.0))
    return lon, lat, dirs


def _tangent_east_north(dirs: np.ndarray, u_par_vec: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Project the tangential displacement onto local east/north unit vectors."""
    z_axis = np.array([0.0, 0.0, 1.0])
    east = np.cross(z_axis[None, :], dirs)
    en = np.linalg.norm(east, axis=1)
    degenerate = en < 1e-9
    east = np.where(degenerate[:, None], np.array([1.0, 0.0, 0.0])[None, :], east)
    en = np.where(degenerate, 1.0, en)
    east = east / en[:, None]
    north = np.cross(dirs, east)
    u = np.einsum("ij,ij->i", u_par_vec, east)
    v = np.einsum("ij,ij->i", u_par_vec, north)
    return u, v


def unwrap_surface(surface: Any, scalar: str = "u_perp", *,
                   projection: str = "mollweide",
                   width: int = 512, height: int = 256) -> UnwrapMap:
    """Rasterize a :class:`SurfaceField` scalar into a 2-D cartographic map.

    Each vertex is parameterized to the sphere (direction from the centroid) and
    the scalar (+ tangential ``u∥`` east/north field for streamlines) is
    interpolated onto a regular grid via ``scipy.interpolate.griddata``, with the
    ``±π`` seam handled by triplicating the sample points in longitude.
    ``"equirectangular"`` returns the ``(lon, lat)`` grid directly;
    ``"mollweide"`` (default, equal-area, the paper's choice) resamples that grid
    through the inverse Mollweide, masking pixels outside the ellipse.
    """
    from scipy.interpolate import griddata

    W, H = int(width), int(height)
    empty = UnwrapMap(values=np.full((H, W), np.nan), coverage=np.zeros((H, W), bool),
                      vec_u=np.zeros((H, W)), vec_v=np.zeros((H, W)),
                      projection=projection, scalar=scalar)
    verts = np.asarray(getattr(surface, "vertices_um", np.zeros((0, 3))), float)
    scal_map = getattr(surface, "scalars", {}) or {}
    if verts.shape[0] < 3:
        return empty                          # too few points for a triangulation
    if scalar not in scal_map:
        # Fall back to the surface's default scalar if the requested one is absent.
        scalar = getattr(surface, "default_scalar", "disp_mag")
        if scalar not in scal_map:
            return empty
    vals = np.asarray(scal_map[scalar], dtype=np.float64)
    centroid = np.asarray(getattr(surface, "centroid_um", verts.mean(axis=0)), float)
    lon, lat, dirs = _sphere_lonlat(verts, centroid)
    u_par_vec = np.asarray(getattr(surface, "u_par_vec", np.zeros_like(verts)), float)
    u_east, u_north = _tangent_east_north(dirs, u_par_vec)

    # Equirectangular target grid in (lon, lat); triplicate points across the seam.
    lon3 = np.concatenate([lon - 2 * np.pi, lon, lon + 2 * np.pi])
    lat3 = np.tile(lat, 3)
    pts = np.stack([lon3, lat3], axis=1)
    lon_ax = np.linspace(-np.pi, np.pi, W)
    lat_ax = np.linspace(np.pi / 2, -np.pi / 2, H)        # row 0 = north pole
    LON, LAT = np.meshgrid(lon_ax, lat_ax)

    def _grid(v: np.ndarray) -> np.ndarray:
        g = griddata(pts, np.tile(v, 3), (LON, LAT), method="linear")
        return g

    eq_vals = _grid(vals)
    eq_u = np.nan_to_num(_grid(u_east))
    eq_v = np.nan_to_num(_grid(u_north))
    eq_cov = np.isfinite(eq_vals)

    if projection == "equirectangular":
        return UnwrapMap(values=eq_vals, coverage=eq_cov, vec_u=eq_u, vec_v=eq_v,
                         projection="equirectangular", scalar=scalar)

    # Mollweide: for each output pixel, inverse-project to (lon, lat), then sample
    # the equirectangular grid (nearest-cell) — pixels outside the ellipse → NaN.
    xs = np.linspace(-1.0, 1.0, W)
    ys = np.linspace(1.0, -1.0, H)
    MX, MY = np.meshgrid(xs, ys)
    with np.errstate(invalid="ignore"):
        theta = np.arcsin(np.clip(MY, -1.0, 1.0))
        lat_i = np.arcsin(np.clip((2.0 * theta + np.sin(2.0 * theta)) / np.pi, -1, 1))
        cos_t = np.cos(theta)
        lon_i = np.where(np.abs(cos_t) > 1e-6,
                         np.pi * MX / np.maximum(cos_t, 1e-6), np.nan)
    inside = (MX ** 2 + MY ** 2) <= 1.0
    valid = inside & np.isfinite(lon_i) & (np.abs(lon_i) <= np.pi)

    col = np.clip(((lon_i + np.pi) / (2 * np.pi) * (W - 1)), 0, W - 1)
    row = np.clip(((np.pi / 2 - lat_i) / np.pi * (H - 1)), 0, H - 1)
    col = np.where(valid, col, 0).astype(np.int64)
    row = np.where(valid, row, 0).astype(np.int64)

    out_vals = np.full((H, W), np.nan)
    out_u = np.zeros((H, W))
    out_v = np.zeros((H, W))
    sampled = eq_vals[row, col]
    out_vals[valid] = sampled[valid]
    out_u[valid] = eq_u[row, col][valid]
    out_v[valid] = eq_v[row, col][valid]
    coverage = valid & np.isfinite(out_vals)
    return UnwrapMap(values=out_vals, coverage=coverage, vec_u=out_u, vec_v=out_v,
                     projection="mollweide", scalar=scalar)


# ── Granule node viewers (V1.73) ─────────────────────────────────────────────
# One composite overlay the PyVista viewer renders for any of the five granule
# nodes (points as crosshairs/colored dots, inter-centroid edges, an assembled
# label volume, and/or a new-vs-previous boundary pair). Pure numpy: all
# ``(z,y,x) voxel → (x,y,z) µm`` reorders and edge derivation live here; the widget
# only issues ``pv.*`` calls and applies ``granule_color`` per label id.

@dataclass
class GranuleScene:
    """3-D scene for a granule node. All coords are world ``(x, y, z)`` µm.

    - ``points_um`` ``(N,3)`` + optional ``point_labels`` ``(N,)`` int (per-granule
      color) and ``edges`` ``(E,2)`` int (indices into ``points_um``).
    - ``draw_points_as`` — ``"crosshair"`` (Bead Detect), ``"dot"`` (cluster /
      tessellation), or ``"none"``.
    - ``label_volume`` ``(Z,H,W)`` int — assembled mask (Node 4), rendered as one
      colored isosurface per label.
    - ``boundary_label_volume`` / ``prev_label_volume`` ``(Z,H,W)`` int — Node 5's
      new (solid) and previous (dotted) boundaries.
    - ``spacing`` ``(dz,dy,dx)`` µm and ``origin_um`` ``(x,y,z)`` for the label
      volumes' ``ImageData``.
    """

    points_um: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), float))
    point_labels: Optional[np.ndarray] = None
    draw_points_as: str = "dot"                 # "crosshair" | "dot" | "none"
    edges: Optional[np.ndarray] = None          # (E,2) indices into points_um
    label_volume: Optional[np.ndarray] = None
    boundary_label_volume: Optional[np.ndarray] = None
    prev_label_volume: Optional[np.ndarray] = None
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)   # (dz, dy, dx) µm
    origin_um: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    point_color: Tuple[int, int, int] = (255, 255, 0)       # unlabeled crosshair

    @property
    def n_points(self) -> int:
        return int(self.points_um.shape[0])


def _points_zyx_to_xyz_um(points_zyx: np.ndarray,
                          voxel_size_um: Tuple[float, float, float]) -> np.ndarray:
    """``(N,3)`` ``(z,y,x)`` voxel indices → world ``(x,y,z)`` µm (viz3d convention)."""
    pts = np.asarray(points_zyx, dtype=float).reshape(-1, 3)
    dz, dy, dx = (float(voxel_size_um[0]), float(voxel_size_um[1]),
                  float(voxel_size_um[2]))
    out = np.zeros_like(pts)
    out[:, 0] = pts[:, 2] * dx      # x
    out[:, 1] = pts[:, 1] * dy      # y
    out[:, 2] = pts[:, 0] * dz      # z
    return out


def granule_centroid_edges(points_zyx: np.ndarray, mode: str) -> np.ndarray:
    """Undirected edge index pairs ``(E,2)`` among the input points (degenerate-safe).

    ``mode="voronoi"`` → Voronoi ridge pairs; anything else → Delaunay simplex
    edges. Operates on the raw point cloud ("lines between all centroids"). Collapses
    a planar/near-planar cloud to a 2-D triangulation so QHull won't fail; returns
    ``(0,2)`` for < 2 points or on any QHull error.
    """
    from itertools import combinations
    pts = np.asarray(points_zyx, dtype=float).reshape(-1, 3)
    n = pts.shape[0]
    if n < 2:
        return np.zeros((0, 2), dtype=np.int64)
    use3d = n >= 4 and float(np.ptp(pts[:, 0])) > 1e-6
    P = pts if use3d else pts[:, 1:3]           # (y, x) when planar
    try:
        from scipy.spatial import Delaunay, Voronoi, QhullError
    except Exception:                           # pragma: no cover
        return np.zeros((0, 2), dtype=np.int64)
    try:
        if str(mode) == "voronoi":
            vor = Voronoi(P)
            edges = np.asarray(vor.ridge_points, dtype=np.int64)
        else:
            tri = Delaunay(P)
            s = tri.simplices
            pairs = np.vstack([s[:, list(c)]
                               for c in combinations(range(s.shape[1]), 2)])
            edges = np.unique(np.sort(pairs, axis=1), axis=0)
    except (QhullError, ValueError, IndexError):
        return np.zeros((0, 2), dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.int64)
    return np.ascontiguousarray(edges, dtype=np.int64)


def granule_points_scene(points_zyx: np.ndarray,
                         voxel_size_um: Tuple[float, float, float], *,
                         labels: Optional[np.ndarray] = None,
                         edges: Optional[np.ndarray] = None,
                         draw_as: str = "dot",
                         point_color: Tuple[int, int, int] = (255, 255, 0)
                         ) -> GranuleScene:
    """Scene for Bead Detect (crosshairs), Cluster (colored dots), or Tessellation
    (dots + ``edges``). ``labels`` colors dots per granule; ``edges`` ``(E,2)`` are
    indices into the point array."""
    return GranuleScene(
        points_um=_points_zyx_to_xyz_um(points_zyx, voxel_size_um),
        point_labels=(np.asarray(labels).astype(np.int64).ravel()
                      if labels is not None else None),
        draw_points_as=str(draw_as),
        edges=(np.asarray(edges).astype(np.int64).reshape(-1, 2)
               if edges is not None and np.asarray(edges).size else None),
        spacing=(float(voxel_size_um[0]), float(voxel_size_um[1]),
                 float(voxel_size_um[2])),
        point_color=tuple(int(c) for c in point_color),
    )


def granule_mask_scene(label_volume_zhw: np.ndarray,
                       voxel_size_um: Tuple[float, float, float]) -> GranuleScene:
    """Scene for the Volume Mask node — assembled per-label isosurfaces."""
    lv = np.asarray(label_volume_zhw)
    if lv.ndim == 2:
        lv = lv[None, ...]
    return GranuleScene(
        label_volume=lv.astype(np.int32, copy=False),
        spacing=(float(voxel_size_um[0]), float(voxel_size_um[1]),
                 float(voxel_size_um[2])),
    )


def granule_boundary_scene(new_label_volume_zhw: np.ndarray,
                           prev_label_volume_zhw: Optional[np.ndarray],
                           voxel_size_um: Tuple[float, float, float]) -> GranuleScene:
    """Scene for the Boundary node — new boundary (solid) + previous (dotted)."""
    new = np.asarray(new_label_volume_zhw)
    if new.ndim == 2:
        new = new[None, ...]
    prev = None
    if prev_label_volume_zhw is not None:
        prev = np.asarray(prev_label_volume_zhw)
        if prev.ndim == 2:
            prev = prev[None, ...]
        prev = prev.astype(np.int32, copy=False)
    return GranuleScene(
        boundary_label_volume=new.astype(np.int32, copy=False),
        prev_label_volume=prev,
        spacing=(float(voxel_size_um[0]), float(voxel_size_um[1]),
                 float(voxel_size_um[2])),
    )
