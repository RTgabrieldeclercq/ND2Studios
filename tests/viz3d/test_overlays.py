"""Headless tests for ``nd2studios.backend.viz3d.overlays`` (pure numpy)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pytest

from nd2studios.backend.viz3d import overlays


# ── DVC fakes ────────────────────────────────────────────────────────────────
@dataclass
class FakeDVCResult:
    grid_coords: np.ndarray
    displacement_field: np.ndarray
    voxel_size_um: Tuple[float, ...]
    strain_field: Optional[np.ndarray] = None
    qfactor: Optional[np.ndarray] = None


def _make_3d_result(Gz=2, Gy=3, Gx=4):
    grid = (Gz, Gy, Gx)
    # coords native (z,y,x); make them distinct integers so reorder is checkable.
    zz, yy, xx = np.meshgrid(np.arange(Gz), np.arange(Gy), np.arange(Gx),
                             indexing="ij")
    coords = np.stack([zz, yy, xx], axis=-1).astype(float)   # (Gz,Gy,Gx,3)
    disp = np.ones(grid + (3,), dtype=float)                 # unit disp all axes
    strain = np.zeros(grid + (3, 3), dtype=float)
    strain[..., 0, 0] = 0.1   # e_zz
    strain[..., 1, 1] = 0.2   # e_yy
    strain[..., 2, 2] = 0.3   # e_xx
    qf = np.full(grid, 0.9)
    return FakeDVCResult(coords, disp, (2.0, 0.5, 0.25), strain, qf)


def test_dvc_field_3d_shapes_and_reorder():
    res = _make_3d_result()
    fld = overlays.dvc_field(res)
    assert fld.grid_shape == (2, 3, 4)
    assert fld.n_points == 24
    assert fld.points_um.shape == (24, 3)
    assert fld.vectors_um.shape == (24, 3)
    # Native disp (1,1,1) in (z,y,x) × voxel (2,.5,.25) → world (x,y,z)=(.25,.5,2)
    assert np.allclose(fld.vectors_um[0], [0.25, 0.5, 2.0])
    # disp_mag = norm of that world vector
    assert fld.scalars["disp_mag"][0] == pytest.approx(np.linalg.norm([0.25, 0.5, 2.0]))


def test_dvc_field_3d_point_scaling():
    res = _make_3d_result()
    fld = overlays.dvc_field(res)
    # Last grid point native (z,y,x)=(1,2,3) × (2,.5,.25) → um (2,1,.75)
    # reordered world (x,y,z) = (.75, 1.0, 2.0)
    assert np.allclose(fld.points_um[-1], [0.75, 1.0, 2.0])


def test_dvc_field_strain_components_and_effstrain():
    fld = overlays.dvc_field(_make_3d_result())
    assert fld.scalars["e_zz"][0] == pytest.approx(0.1)
    assert fld.scalars["e_yy"][0] == pytest.approx(0.2)
    assert fld.scalars["e_xx"][0] == pytest.approx(0.3)
    assert "eff_strain" in fld.scalars
    assert np.all(fld.scalars["eff_strain"] >= 0)
    assert "qfactor" in fld.scalars and fld.scalars["qfactor"][0] == pytest.approx(0.9)


def test_dvc_field_2d_pads_z_zero():
    Gy, Gx = 3, 4
    yy, xx = np.meshgrid(np.arange(Gy), np.arange(Gx), indexing="ij")
    coords = np.stack([yy, xx], axis=-1).astype(float)     # (Gy,Gx,2) native (y,x)
    disp = np.ones((Gy, Gx, 2))
    strain = np.zeros((Gy, Gx, 2, 2)); strain[..., 0, 1] = 0.05
    res = FakeDVCResult(coords, disp, (0.5, 0.5), strain)
    fld = overlays.dvc_field(res)
    assert fld.grid_shape == (3, 4)
    assert np.all(fld.points_um[:, 2] == 0.0)   # z padded
    assert np.all(fld.vectors_um[:, 2] == 0.0)
    assert "e_xy" in fld.scalars and "u_z" not in fld.scalars


def test_dvc_field_scalar_key_filter():
    fld = overlays.dvc_field(_make_3d_result(), scalar_keys=["disp_mag", "qfactor"])
    assert set(fld.scalars) == {"disp_mag", "qfactor"}


def test_dvc_field_missing_optional_fields():
    Gy, Gx = 2, 2
    coords = np.zeros((Gy, Gx, 2)); disp = np.zeros((Gy, Gx, 2))
    res = FakeDVCResult(coords, disp, (1.0, 1.0))    # no strain, no qfactor
    fld = overlays.dvc_field(res)
    assert "eff_strain" not in fld.scalars and "qfactor" not in fld.scalars
    assert "disp_mag" in fld.scalars


# ── PTV fakes ────────────────────────────────────────────────────────────────
@dataclass
class FakeTrackData:
    trajectories: np.ndarray
    traj_ids: np.ndarray
    ndim: int = 2
    pixel_size_um: float = 0.5
    z_step_um: float = 2.0
    time_step: float = 1.0


def test_ptv_polylines_2d_time_axis_and_scaling():
    # Two tracks, 4 frames, 2-D (y,x). Track 0 moves; track 1 has a NaN gap.
    traj = np.array([
        [[0, 0], [0, 2], [0, 4], [0, 6]],          # track 0, all finite
        [[1, 1], [np.nan, np.nan], [1, 3], [1, 5]],  # track 1, gap at frame 1
    ], dtype=float)
    td = FakeTrackData(traj, np.array([10, 20]))
    tracks = overlays.ptv_polylines(td, third_axis="time", color_by="time")
    # Track 0 -> one segment (4 pts); track 1 -> one run of >=2 pts (frames 2,3)
    assert tracks.n_segments == 2
    assert list(tracks.seg_track_ids) == [10, 20]
    seg0 = tracks.segments[0]
    # world x = x_px*px, y = y_px*px, z = frame*time_step
    assert np.allclose(seg0[:, 0], [0, 1, 2, 3])   # x = (0,2,4,6)*0.5
    assert np.allclose(seg0[:, 1], 0.0)            # y = 0
    assert np.allclose(seg0[:, 2], [0, 1, 2, 3])   # time axis = frame*1.0
    # color-by-time is fraction along full trajectory (denom = n_frames-1 = 3)
    assert np.allclose(tracks.seg_scalars[0], [0, 1 / 3, 2 / 3, 1.0])


def test_ptv_polylines_drops_single_point_runs():
    # Track with only isolated finite frames (no run >= 2) yields no segment.
    traj = np.array([[[0, 0], [np.nan, np.nan], [0, 4], [np.nan, np.nan]]], float)
    td = FakeTrackData(traj, np.array([1]))
    tracks = overlays.ptv_polylines(td)
    assert tracks.n_segments == 0


def test_ptv_polylines_up_to_frame_truncates():
    traj = np.array([[[0, 0], [0, 1], [0, 2], [0, 3]]], float)
    td = FakeTrackData(traj, np.array([0]))
    tracks = overlays.ptv_polylines(td, up_to_frame=1)
    assert tracks.segments[0].shape[0] == 2      # only frames 0..1


def test_ptv_polylines_velocity_coloring():
    traj = np.array([[[0, 0], [0, 2], [0, 6]]], float)  # x px 0,2,6
    td = FakeTrackData(traj, np.array([0]), pixel_size_um=1.0, time_step=1.0)
    tracks = overlays.ptv_polylines(td, color_by="velocity")
    v = tracks.seg_scalars[0]
    # spatial speeds: |Δx| between (0,2)=2, (2,6)=4; last repeats
    assert np.allclose(v, [2.0, 4.0, 4.0])


def test_ptv_polylines_3d_uses_z_step():
    # 3-D track (y,x,z); third axis is real z scaled by z_step_um.
    traj = np.array([[[0, 0, 0], [0, 0, 1], [0, 0, 2]]], float)
    td = FakeTrackData(traj, np.array([7]), ndim=3, pixel_size_um=1.0, z_step_um=5.0)
    tracks = overlays.ptv_polylines(td, third_axis="z")
    assert np.allclose(tracks.segments[0][:, 2], [0, 5, 10])


def test_ptv_polylines_max_tracks_cap():
    traj = np.tile(np.array([[[0, 0], [0, 1]]], float), (10, 1, 1))
    td = FakeTrackData(traj, np.arange(10))
    tracks = overlays.ptv_polylines(td, max_tracks=3)
    assert tracks.n_segments == 3


def test_ptv_polylines_empty():
    td = FakeTrackData(np.zeros((0, 0, 2)), np.zeros((0,), int))
    tracks = overlays.ptv_polylines(td)
    assert tracks.n_segments == 0


# ── label_volume ─────────────────────────────────────────────────────────────
def test_label_volume_promotes_2d():
    out = overlays.label_volume(np.ones((4, 4), np.uint16))
    assert out.shape == (1, 4, 4) and out.dtype == np.int32


def test_label_volume_keeps_3d():
    out = overlays.label_volume(np.ones((3, 4, 4), np.int64))
    assert out.shape == (3, 4, 4)
