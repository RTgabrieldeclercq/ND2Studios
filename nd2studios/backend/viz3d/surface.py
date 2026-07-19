"""viz3d.surface — build a smoothed closed object surface from a 3-D mask and
sample a DVC displacement field onto it.

Pure numpy / scipy / scikit-image — **Qt-free and PyVista/VTK-free**, like
:mod:`nd2studios.backend.viz3d.overlays`. The mesh is produced here so the
widget layer only wraps ``vertices``/``faces`` in ``pyvista.PolyData``; keeping
VTK out keeps every routine headless-testable.

Follows Stout et al. 2016 (PNAS, *Materials and Methods → "Calculating
Discretized Cell Surfaces"*): a binary object mask →
**marching cubes** (Lorensen & Cline 1987) → **Taubin λ|μ smoothing**
(Taubin 1995) → a closed surface ``∂V`` in physical µm. The measured DVC
displacement is then interpolated onto the surface vertices and split into a
**normal** component ``u⊥ = u·n`` and a **tangential** component
``u∥ = u − u⊥ n`` (paper Fig 4C/D).

Pipeline::

    mask (Z,H,W) bool ──marching_cubes──▶ triangle mesh (µm)
                      ──_taubin_smooth──▶ ∂V (non-shrinking)
    DVCResult         ──µm-aligned RGI──▶ u at vertices (world x,y,z µm)
                      ──decompose──────▶ u⊥, u∥

The DVC field is consumed **read-only** (only ``DVCResult`` attributes are
read); nothing in ``backend/dvc/`` is imported or modified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from nd2studios.backend.viz3d.overlays import _grid_axes_vox, dvc_field


@dataclass
class ObjectSurface:
    """A smoothed closed surface mesh in world µm, ready for coloring + MDM.

    - ``vertices_um`` ``(Nv, 3)`` world ``(x, y, z)`` µm; ``faces`` ``(Nf, 3)``
      int triangle indices, **outward-oriented** (face normals point out of the
      object, so the divergence-theorem enclosed volume is positive).
    - ``vertex_normals`` / ``face_normals`` unit; ``face_areas`` ``(Nf,)`` µm²;
      ``face_centroids_um`` ``(Nf, 3)``.
    - ``enclosed_volume_um3`` via the divergence theorem (same triangulation the
      MDM surface integral uses, so ``V`` and ``∮`` are consistent).
    - ``scalars`` maps a name to a per-vertex ``(Nv,)`` array for coloring.
    """

    vertices_um: np.ndarray
    faces: np.ndarray
    vertex_normals: np.ndarray
    face_normals: np.ndarray
    face_areas: np.ndarray
    face_centroids_um: np.ndarray
    enclosed_volume_um3: float
    centroid_um: np.ndarray
    scalars: Dict[str, np.ndarray] = field(default_factory=dict)
    default_scalar: str = "disp_mag"

    @property
    def n_vertices(self) -> int:
        return int(self.vertices_um.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0]) if self.faces.ndim == 2 else 0

    @property
    def is_empty(self) -> bool:
        return self.vertices_um.size == 0 or self.faces.size == 0

    @property
    def scalar_names(self) -> List[str]:
        return list(self.scalars.keys())


def _empty_surface() -> ObjectSurface:
    z3 = np.zeros((0, 3), dtype=np.float64)
    return ObjectSurface(
        vertices_um=z3, faces=np.zeros((0, 3), dtype=np.int64),
        vertex_normals=z3.copy(), face_normals=z3.copy(),
        face_areas=np.zeros((0,), dtype=np.float64),
        face_centroids_um=z3.copy(), enclosed_volume_um3=0.0,
        centroid_um=np.zeros(3, dtype=np.float64))


# ── mesh geometry ─────────────────────────────────────────────────────────────
def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Signed enclosed volume ``V = (1/6) Σ_faces v0·(v1×v2)`` (µm³).

    Positive when faces are wound so their normals point outward.
    """
    if faces.size == 0:
        return 0.0
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0)


def _face_geometry(vertices: np.ndarray,
                   faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unit face normals, face areas (µm²), and face centroids (µm)."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    twice_area = np.linalg.norm(cross, axis=1)
    areas = 0.5 * twice_area
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = cross / twice_area[:, None]
    normals = np.nan_to_num(normals)
    centroids = (v0 + v1 + v2) / 3.0
    return normals, areas, centroids


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray,
                    face_normals: np.ndarray, face_areas: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals."""
    vn = np.zeros_like(vertices)
    weighted = face_normals * face_areas[:, None]
    for k in range(3):
        np.add.at(vn, faces[:, k], weighted)
    norms = np.linalg.norm(vn, axis=1)
    norms[norms == 0] = 1.0
    return vn / norms[:, None]


def _taubin_smooth(vertices: np.ndarray, faces: np.ndarray, iterations: int,
                   lam: float, mu: float) -> np.ndarray:
    """Taubin λ|μ smoothing over the mesh's edge adjacency (non-shrinking).

    Each iteration is a positive-λ Laplacian pass followed by a negative-μ pass,
    using the uniform (umbrella) Laplacian ``L(v) = mean(neighbours) − v``. Pure
    numpy + a scipy sparse adjacency; keeps this module VTK-free.
    """
    if iterations <= 0 or faces.size == 0 or vertices.shape[0] == 0:
        return vertices.astype(np.float64, copy=True)
    from scipy.sparse import coo_matrix

    n = vertices.shape[0]
    a = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    b = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    rows = np.concatenate([a, b])
    cols = np.concatenate([b, a])
    adj = coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()
    adj.data[:] = 1.0                      # dedup summed edges → binary adjacency
    deg = np.asarray(adj.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    inv_deg = (1.0 / deg)[:, None]

    v = vertices.astype(np.float64, copy=True)
    for _ in range(int(iterations)):
        v = v + lam * (adj.dot(v) * inv_deg - v)
        v = v + mu * (adj.dot(v) * inv_deg - v)
    return v


def build_object_surface(mask: np.ndarray,
                         voxel_size_um: Optional[Tuple[float, float, float]] = None,
                         *,
                         smooth_iterations: int = 10,
                         taubin_lambda: float = 0.5,
                         taubin_mu: float = -0.53,
                         step_size: int = 1) -> ObjectSurface:
    """Marching cubes + Taubin → a closed :class:`ObjectSurface` in world µm.

    ``mask`` is a ``(Z, H, W)`` boolean object (2-D promoted to a single plane).
    ``voxel_size_um`` is ``(dz, dy, dx)`` µm (defaults to isotropic 1). The mask
    is padded by one background voxel so the surface is **closed** even when the
    object touches the array border. Faces are re-wound to guarantee outward
    normals (positive enclosed volume) — required by the MDM surface integral.
    """
    from skimage.measure import marching_cubes

    mask = np.asarray(mask)
    if mask.ndim == 2:
        mask = mask[None, ...]
    mask = mask.astype(bool)
    if voxel_size_um and len(voxel_size_um) == 3:
        dz, dy, dx = (float(voxel_size_um[0]), float(voxel_size_um[1]),
                      float(voxel_size_um[2]))
    else:
        dz = dy = dx = 1.0
    dz = dz if dz > 0 else 1.0
    dy = dy if dy > 0 else 1.0
    dx = dx if dx > 0 else 1.0

    if not mask.any():
        return _empty_surface()

    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    try:
        verts, faces, _normals, _values = marching_cubes(
            padded.astype(np.float32), level=0.5,
            spacing=(dz, dy, dx), step_size=int(max(1, step_size)))
    except (ValueError, RuntimeError):
        return _empty_surface()
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        return _empty_surface()

    # marching_cubes vertices are (z, y, x) µm (input-axis order, scaled by
    # spacing). Undo the one-voxel pad, then reorder to world (x, y, z).
    verts = verts - np.array([dz, dy, dx], dtype=np.float64)[None, :]
    verts_xyz = verts[:, ::-1].astype(np.float64)          # (z,y,x) → (x,y,z)
    faces = np.asarray(faces, dtype=np.int64)

    verts_xyz = _taubin_smooth(verts_xyz, faces, smooth_iterations,
                               taubin_lambda, taubin_mu)

    # Guarantee outward orientation (positive enclosed volume) so face normals
    # point out of the object; the axis reversal above flips handedness, so the
    # winding fix is data-driven rather than assumed.
    if _signed_volume(verts_xyz, faces) < 0:
        faces = faces[:, ::-1].copy()

    fn, fa, fc = _face_geometry(verts_xyz, faces)
    vn = _vertex_normals(verts_xyz, faces, fn, fa)
    vol = abs(_signed_volume(verts_xyz, faces))
    centroid = verts_xyz.mean(axis=0)
    return ObjectSurface(
        vertices_um=verts_xyz, faces=faces, vertex_normals=vn,
        face_normals=fn, face_areas=fa, face_centroids_um=fc,
        enclosed_volume_um3=float(vol), centroid_um=centroid)


# ── on-surface displacement sampling (µm-aligned, from overlays' scheme) ───────
def _make_surface_interpolator(result: Any, verts: np.ndarray,
                               z_offset_um: float = 0.0):
    """Build the µm-aligned per-vertex interpolator shared by the samplers.

    Mirrors :func:`nd2studios.backend.viz3d.overlays.dvc_object_field`: the DVC
    grid nodes sit at ``grid_coords × voxel_size_um`` (folding in any XY
    downsample) and the surface vertices are already in world µm. Returns
    ``(interp, d, use_3d, vsz)`` where ``interp(grid_native)`` linearly samples a
    native-shaped grid at every vertex (extrapolating past the grid inset).
    """
    from scipy.interpolate import RegularGridInterpolator

    gc = np.asarray(result.grid_coords, dtype=np.float64)
    d = int(gc.shape[-1])
    vsz = tuple(float(v) for v in (getattr(result, "voxel_size_um", ()) or ()))
    if len(vsz) != d:
        vsz = (1.0,) * d
    axes_um = [np.asarray(a) * vsz[i] for i, a in enumerate(_grid_axes_vox(gc, d))]
    if d >= 3:
        axes_um[0] = axes_um[0] + float(z_offset_um)
    grid_shape = tuple(int(s) for s in gc.shape[:-1])
    verts = np.asarray(verts, dtype=np.float64)

    use_3d = d >= 3 and all(len(a) >= 2 for a in axes_um)
    if use_3d:
        query = np.stack([verts[:, 2], verts[:, 1], verts[:, 0]], axis=1)  # (z,y,x)
        interp_axes: Tuple[np.ndarray, ...] = tuple(axes_um)
    else:
        query = np.stack([verts[:, 1], verts[:, 0]], axis=1)               # (y,x)
        interp_axes = (axes_um[-2], axes_um[-1])

    def interp(grid_native: np.ndarray) -> np.ndarray:
        g = np.asarray(grid_native, dtype=np.float64).reshape(grid_shape)
        if not use_3d and d >= 3:
            g = g.reshape(grid_shape[-2:])
        rgi = RegularGridInterpolator(interp_axes, g, bounds_error=False,
                                      fill_value=None)
        return rgi(query)

    return interp, d, use_3d, vsz


def sample_displacement_on_surface(result: Any, surface: ObjectSurface, *,
                                   z_offset_um: float = 0.0) -> np.ndarray:
    """Interpolate the DVC displacement at the surface vertices → ``(Nv, 3)`` µm.

    Returns per-vertex ``u`` in world ``(x, y, z)`` µm. A 2-D DIC (or single-Z)
    field is interpolated in-plane and broadcast over Z (``u_z`` then 0).
    """
    verts = np.asarray(surface.vertices_um, dtype=np.float64)
    u_world = np.zeros((verts.shape[0], 3), dtype=np.float64)
    if verts.shape[0] == 0 or np.asarray(result.grid_coords).size == 0:
        return u_world
    try:
        interp, d, _use_3d, vsz = _make_surface_interpolator(
            result, verts, z_offset_um)
        disp_um = (np.asarray(result.displacement_field, dtype=np.float64)
                   * np.asarray(vsz))
        # native component index → world axis: 3-D (z→2, y→1, x→0); 2-D (y→1, x→0).
        native_to_world = ({0: 2, 1: 1, 2: 0} if d >= 3 else {0: 1, 1: 0})
        for ci in range(d):
            u_world[:, native_to_world[ci]] = np.nan_to_num(interp(disp_um[..., ci]))
    except Exception:  # noqa: BLE001 — degenerate grid; return zero displacement
        return np.zeros((verts.shape[0], 3), dtype=np.float64)
    return u_world


def sample_scalars_on_surface(result: Any, surface: ObjectSurface, *,
                              scalar_keys: Optional[List[str]] = None,
                              z_offset_um: float = 0.0) -> Dict[str, np.ndarray]:
    """Sample the DVC-derived scalar fields (``dvc_field``) at surface vertices."""
    verts = np.asarray(surface.vertices_um, dtype=np.float64)
    out: Dict[str, np.ndarray] = {}
    if verts.shape[0] == 0 or np.asarray(result.grid_coords).size == 0:
        return out
    fld = dvc_field(result, scalar_keys=scalar_keys)
    interp, _d, _use_3d, _vsz = _make_surface_interpolator(result, verts, z_offset_um)
    for name, flat in fld.scalars.items():
        try:
            out[name] = np.nan_to_num(interp(flat)).astype(np.float64)
        except Exception:  # noqa: BLE001 — degenerate grid; skip this scalar
            continue
    return out


def decompose_surface_displacement(u_vert: np.ndarray, vertex_normals: np.ndarray
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split per-vertex ``u`` into normal / tangential parts (paper Fig 4C/D).

    Returns ``(u_perp, u_par_vec, u_par_mag)`` — the signed normal component
    ``u⊥ = u·n`` (positive = outward), the tangential vector ``u∥ = u − u⊥ n``,
    and its magnitude.
    """
    u = np.asarray(u_vert, dtype=np.float64)
    n = np.asarray(vertex_normals, dtype=np.float64)
    if u.shape[0] == 0:
        return (np.zeros((0,)), np.zeros((0, 3)), np.zeros((0,)))
    u_perp = np.einsum("ij,ij->i", u, n)
    u_par_vec = u - u_perp[:, None] * n
    u_par_mag = np.linalg.norm(u_par_vec, axis=1)
    return u_perp, u_par_vec, u_par_mag
