"""Headless tests for ``backend.analysis.bead_detect`` (V1.70 · P1).

Covers what the P1 spec pins down:

1. K planted Gaussian blobs at known ``(z, y, x)`` are detected — exact count and
   sub-voxel positions within tolerance — in both ``log`` and ``components`` modes.
2. **Axis order is ``(z, y, x)``** (regression against the detector's ``(x,y,z)``
   flip): a volume with distinct ``Z < H < W`` extents and one blob whose x-coord is
   larger than the entire Z extent must come back with that value in column 2, not 0.
3. Anisotropic voxel size → per-axis ``_um`` scaling is correct (``_px * (dz|dy|dx)``)
   and cannot be produced by an axis swap (``dz != dy``).
4. ``Z == 1`` 2-D fallback → every ``centroid_z_px == 0`` and the cloud keeps its
   ``(N, 3)`` shape.
5. Row schema matches P0 §3 exactly; ``m_position`` / ``frame`` propagate from params.

``pytest.importorskip`` skips the whole module cleanly when the SerialTrack
detector's optional deps (numba / scipy) are absent.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")
pytest.importorskip("scipy")

from nd2studios.backend.analysis.bead_detect import detect_beads
from nd2studios.backend.analysis.granule_types import POINT_ROW_KEYS

# Strongly anisotropic voxel size (thick Z) to exercise the µm-scaling path.
VOXEL_SIZE_UM = (2.0, 0.5, 0.5)  # (dz, dy, dx) — dz != dy catches axis swaps


def _plant(shape: tuple, centers, sigma: float = 1.6, amp: float = 1.0) -> np.ndarray:
    """Sum of isotropic Gaussian blobs (in *voxel* space) on a zero background."""
    vol = np.zeros(shape, dtype=np.float32)
    if len(shape) == 2:
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        for cy, cx in centers:
            vol += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2)))
        return vol
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    for cz, cy, cx in centers:
        vol += amp * np.exp(
            -(((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))
        )
    return vol


def _nearest(points: np.ndarray, gt: tuple) -> float:
    """Distance from ``gt`` to the nearest detected point."""
    return float(np.linalg.norm(points - np.asarray(gt, dtype=float), axis=1).min())


# ── centres well separated (> min_distance) in a Z < H < W volume ─────────────
_SHAPE = (10, 40, 90)
_CENTERS = [(3.2, 20.3, 60.1), (6.0, 10.0, 15.0), (4.0, 30.0, 80.0)]


@pytest.mark.parametrize("mode,tol", [("log", 0.5), ("components", 0.7)])
def test_detects_all_blobs_with_subvoxel_positions(mode: str, tol: float) -> None:
    vol = _plant(_SHAPE, _CENTERS)
    pts, rows = detect_beads(
        vol, VOXEL_SIZE_UM,
        {"detect_mode": mode, "threshold": 0.2, "min_distance_px": 4, "subpixel": True},
    )
    assert pts.shape == (len(_CENTERS), 3)
    assert len(rows) == len(_CENTERS)
    for gt in _CENTERS:
        assert _nearest(pts, gt) <= tol


def test_output_axis_order_is_zyx() -> None:
    """A blob at x=60 (> the Z extent of 10) must land in column 2, not column 0."""
    vol = _plant(_SHAPE, _CENTERS)
    pts, _ = detect_beads(
        vol, VOXEL_SIZE_UM,
        {"detect_mode": "log", "threshold": 0.2, "min_distance_px": 4},
    )
    z, h, w = _SHAPE
    # Every z-coord is a valid plane index; every x-coord uses the full W extent.
    assert np.all((pts[:, 0] >= 0.0) & (pts[:, 0] < z))
    assert pts[:, 2].max() > z  # only possible if column 2 really is x

    # The (3.2, 20.3, 60.1) blob must match column-for-column.
    row = min(pts, key=lambda p: np.linalg.norm(p - np.array([3.2, 20.3, 60.1])))
    assert row[0] == pytest.approx(3.2, abs=0.5)   # z
    assert row[1] == pytest.approx(20.3, abs=0.5)  # y
    assert row[2] == pytest.approx(60.1, abs=0.5)  # x


def test_um_scaling_is_anisotropic_and_per_axis() -> None:
    vol = _plant(_SHAPE, _CENTERS)
    dz, dy, dx = VOXEL_SIZE_UM
    _, rows = detect_beads(
        vol, VOXEL_SIZE_UM,
        {"detect_mode": "log", "threshold": 0.2, "min_distance_px": 4},
    )
    assert rows
    for r in rows:
        assert r["centroid_z_um"] == pytest.approx(r["centroid_z_px"] * dz)
        assert r["centroid_y_um"] == pytest.approx(r["centroid_y_px"] * dy)
        assert r["centroid_x_um"] == pytest.approx(r["centroid_x_px"] * dx)
    # Tie scaling to a known planted centre (z=3.2 → ~6.4 µm at dz=2.0).
    zmatch = min(rows, key=lambda r: abs(r["centroid_z_px"] - 3.2))
    assert zmatch["centroid_z_um"] == pytest.approx(3.2 * dz, abs=1.0)


def test_z1_2d_fallback() -> None:
    shape = (1, 40, 90)
    centers2d = [(15.2, 40.3), (25.0, 70.0)]
    vol = _plant((shape[1], shape[2]), centers2d)[None, ...]
    pts, rows = detect_beads(
        vol, VOXEL_SIZE_UM,
        {"detect_mode": "components", "threshold": 0.2, "min_distance_px": 4},
    )
    assert pts.shape == (len(centers2d), 3)
    assert np.all(pts[:, 0] == 0.0)              # centroid_z_px == 0
    assert all(r["centroid_z_px"] == 0.0 for r in rows)
    assert all(r["centroid_z_um"] == 0.0 for r in rows)
    for cy, cx in centers2d:
        assert _nearest(pts, (0.0, cy, cx)) <= 0.7


def test_bare_2d_array_also_falls_back() -> None:
    """A plain (H, W) array is accepted and treated as a single Z plane."""
    vol2d = _plant((40, 90), [(15.0, 40.0), (25.0, 70.0)])
    pts, rows = detect_beads(
        vol2d, VOXEL_SIZE_UM,
        {"detect_mode": "components", "threshold": 0.2},
    )
    assert pts.shape[1] == 3
    assert np.all(pts[:, 0] == 0.0)


def test_row_schema_and_indices() -> None:
    vol = _plant(_SHAPE, _CENTERS)
    _, rows = detect_beads(
        vol, VOXEL_SIZE_UM,
        {"detect_mode": "log", "threshold": 0.2, "m_position": 7, "frame": 3},
    )
    assert rows
    for i, r in enumerate(rows):
        assert tuple(r.keys()) == POINT_ROW_KEYS
        assert r["bead_id"] == i
        assert r["m_position"] == 7
        assert r["frame"] == 3
        assert r["granule_id"] is None


def test_empty_volume_returns_empty_cloud() -> None:
    pts, rows = detect_beads(np.zeros((5, 20, 20), np.float32), (1.0, 1.0, 1.0), {})
    assert pts.shape == (0, 3)
    assert rows == []


def test_min_intensity_floor_drops_dim_blobs() -> None:
    """A bright and a dim blob; a min_intensity between them keeps only the bright."""
    vol = _plant(_SHAPE, [(5.0, 20.0, 30.0)], amp=1.0)
    vol += _plant(_SHAPE, [(5.0, 20.0, 60.0)], amp=0.3)
    pts, _ = detect_beads(
        vol, VOXEL_SIZE_UM,
        {"detect_mode": "components", "threshold": 0.1, "min_intensity": 0.5},
    )
    assert pts.shape[0] == 1
    assert _nearest(pts, (5.0, 20.0, 30.0)) <= 0.7


def test_subpixel_false_rounds_to_integers() -> None:
    vol = _plant(_SHAPE, _CENTERS)
    pts, _ = detect_beads(
        vol, VOXEL_SIZE_UM,
        {"detect_mode": "components", "threshold": 0.2, "subpixel": False},
    )
    assert pts.shape[0] == len(_CENTERS)
    assert np.array_equal(pts, np.rint(pts))
