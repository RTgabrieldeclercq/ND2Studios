"""Shared contracts for the V1.70 Granule Separation node chain (pure, Qt-free).

This is the P0 layer every granule sub-plan (P1–P7) builds against — the single
source of truth for the dataclasses, the point-cloud DATA-row schema, the coordinate
convention, and the ``record`` storage-attribute names. Keeping it central means the
five independently-authored backend modules stay consistent.

Coordinate convention (obeyed everywhere downstream — see also the P0 sub-plan doc):

* **Point-cloud arrays are stored ``(z, y, x)``** to match the app-wide measurement /
  track convention (``serialtrack_analysis`` reads ``centroid_z/y/x``). Note that
  ``backend/serialtrack/detection.ParticleDetector.detect`` returns ``(x, y, z)`` —
  the P1 bead-detect node is the ONLY place that flip happens.
* **Mesh vertices are world ``(x, y, z)`` µm** to match ``backend/viz3d`` (so P4/Node-4
  feed ``viz3d.surface.build_object_surface`` unchanged).
* **Voxel size is ``(dz, dy, dx)`` µm** everywhere (``viz3d.prep.Spacing``).
* **Mask/label bbox is ``(z0, z1, y0, y1, x0, x1)`` half-open**
  (``object_scope.ObjectRegion``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── record storage attribute names (set on both the page and the record) ──────
# Each is a nested dict ``{m: {t: <value>}}`` keyed per multipoint / timepoint,
# mirroring ``record._mask3d_by_m``. Kept as named constants so producers and
# consumers never disagree on the attribute string.
GRANULE_POINTS_ATTR = "_granule_points_by_m"    # {m:{t: (N,3) float (z,y,x) voxels}}
GRANULE_LABELS_ATTR = "_granule_labels_by_m"     # {m:{t: (N,) int initial GMM labels}}
GRANULE_TESS_ATTR = "_granule_tess_by_m"         # {m:{t: GranuleTessellation}}
GRANULE_MASKS_ATTR = "_granule_masks_by_m"       # {m:{t: {gid: (Z,H,W) bool, "_labels": (Z,H,W) int32}}}
GRANULE_BANDS_ATTR = "_granule_bands_by_m"       # {m:{t: {gid: (Z,H,W) bool, "_labels": (Z,H,W) int32}}}

# Reserved dict key inside the masks/bands per-(m,t) dict for the combined
# integer-label volume ("all volumes"), sitting beside the per-granule bool masks.
COMBINED_LABELS_KEY = "_labels"

# Sentinel granule id for an unclustered / noise point.
NOISE_LABEL = -1


@dataclass
class GranuleBoundary:
    """One granule's boundary + the density the merge compares.

    ``vertices_um``/``faces`` are a triangle mesh in world ``(x, y, z)`` µm (the
    ``viz3d`` convention). ``delaunay`` is retained (when built by the alpha-shape
    path) so P4's voxel inside-test is a cheap ``find_simplex`` with no
    re-triangulation; it is ``None`` for Voronoi-mode boundaries (P4 rebuilds one).
    """

    granule_id: int
    vertices_um: np.ndarray                      # (Nv, 3) world (x, y, z) µm
    faces: np.ndarray                            # (Nf, 3) int triangle indices
    enclosed_volume_um3: float
    n_points: int
    density: float                               # n_points / enclosed_volume_um3 (or 1/cell_vol)
    delaunay: Optional[Any] = None               # scipy.spatial.Delaunay | None


@dataclass
class GranuleTessellation:
    """P3 output → P4 input. Carries the FINAL (post-merge) granule assignment.

    ``point_labels`` is the final granule id per INPUT point (index-aligned with the
    ``(N,3)`` cloud in ``record._granule_points_by_m[m][t]``); ``merged_from`` maps
    each final id to the list of original GMM ids folded into it.
    """

    mode: str                                    # "alpha_shape" | "voronoi"
    boundaries: Dict[int, GranuleBoundary]       # final granule_id -> boundary
    point_labels: np.ndarray                     # (N,) int final granule id per point
    voxel_size_um: Tuple[float, float, float]    # (dz, dy, dx) the boundaries are in
    merged_from: Dict[int, List[int]] = field(default_factory=dict)


# ── point-cloud DATA-row schema helpers ──────────────────────────────────────
# The point cloud rides the existing ``PortType.DATA`` = ``List[Dict]`` row
# convention. P1 writes the first ``centroid_z_px`` in the app; the read path
# already exists (``serialtrack_analysis``).

POINT_ROW_KEYS = (
    "bead_id", "m_position", "frame",
    "centroid_z_px", "centroid_y_px", "centroid_x_px",
    "centroid_z_um", "centroid_y_um", "centroid_x_um",
    "granule_id",
)


def make_point_rows(points_zyx: np.ndarray,
                    m: int, t: int,
                    voxel_size_um: Tuple[float, float, float],
                    labels: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """Build DATA rows (P0 §3 schema) from an ``(N, 3)`` ``(z, y, x)`` voxel cloud.

    ``voxel_size_um`` is ``(dz, dy, dx)``. ``labels`` (optional) fills ``granule_id``;
    ``None`` leaves it ``None`` (pre-clustering).
    """
    pts = np.asarray(points_zyx, dtype=float).reshape(-1, 3)
    dz, dy, dx = (float(voxel_size_um[0]), float(voxel_size_um[1]),
                  float(voxel_size_um[2]))
    rows: List[Dict[str, Any]] = []
    for i in range(pts.shape[0]):
        z, y, x = float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])
        gid = None
        if labels is not None:
            gid = int(labels[i])
        rows.append({
            "bead_id": int(i),
            "m_position": int(m),
            "frame": int(t),
            "centroid_z_px": z, "centroid_y_px": y, "centroid_x_px": x,
            "centroid_z_um": z * dz, "centroid_y_um": y * dy, "centroid_x_um": x * dx,
            "granule_id": gid,
        })
    return rows


def points_from_rows(rows: List[Dict[str, Any]]) -> np.ndarray:
    """Inverse of :func:`make_point_rows`: ``List[Dict]`` → ``(N, 3)`` ``(z, y, x)``."""
    if not rows:
        return np.zeros((0, 3), dtype=float)
    return np.array(
        [[float(r.get("centroid_z_px", 0.0)),
          float(r.get("centroid_y_px", 0.0)),
          float(r.get("centroid_x_px", 0.0))] for r in rows],
        dtype=float,
    )


# ── categorical granule coloring (V1.73 viewers) ─────────────────────────────
# A fixed, set-INDEPENDENT palette so the granule *viewers* (2-D cluster/mask/
# boundary overlays and their 3-D counterparts) all agree on one color per granule
# id. This is deliberately NOT ``backend/track_overlays.generate_track_colormap``,
# which permutes colors as a function of the *set* of ids present (so the same id
# changes color when the id set changes) — see V1.71 plan §2. Kept here (backend,
# Qt-free) and copied verbatim from ``pages/analysis_page._LABEL_PALETTE`` so the
# label-overlay painter and these helpers render identically (a test asserts the two
# tuples stay equal — if you edit one, edit both).
GRANULE_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100),
    (255, 100, 255), (100, 255, 255), (255, 165, 0),   (180, 255, 130),
    (130, 180, 255), (255, 130, 180), (200, 200, 100), (100, 200, 200),
    (200, 100, 200), (150, 255, 200), (255, 200, 150), (200, 150, 255),
    (80, 180, 80),   (180, 80, 80),   (80, 80, 180),   (180, 180, 80),
)
NOISE_COLOR: Tuple[int, int, int] = (128, 128, 128)


def granule_color(gid: int) -> Tuple[int, int, int]:
    """Deterministic, set-independent ``(R, G, B)`` (0–255) for a granule id.

    The same id always maps to the same color regardless of what else is on
    screen, which is what lets the mask node match the cluster node: both call
    ``granule_color(<the same id>)``. ``gid <= 0`` (background / :data:`NOISE_LABEL`)
    → gray. Cycles the palette for ids beyond its length.
    """
    g = int(gid)
    if g <= 0:
        return NOISE_COLOR
    return GRANULE_PALETTE[(g - 1) % len(GRANULE_PALETTE)]
