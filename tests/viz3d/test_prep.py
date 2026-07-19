"""Headless tests for ``nd2studios.backend.viz3d.prep`` (pure numpy, no display)."""
from __future__ import annotations

import numpy as np
import pytest

from nd2studios.backend.viz3d import prep


class FakeVolume:
    """Minimal MaterializedDataset-like stand-in for tests."""

    def __init__(self, channels, pixel_size_um=0.5, z_step_um=2.0):
        self._channels = channels                      # name -> (M,T,Z,H,W)
        self.channel_names = list(channels.keys())
        first = next(iter(channels.values()))
        self.n_multipoints, self.n_timepoints, self.n_zslices = first.shape[:3]
        self.height, self.width = first.shape[3:]
        self.pixel_size_um = pixel_size_um
        self.z_step_um = z_step_um

    def get_volume(self, c, m=0, t=0, z_start=None, z_end=None):
        arr = self._channels[self.channel_names[int(c)]][int(m), int(t)]
        zs = 0 if z_start is None else int(z_start)
        ze = arr.shape[0] if z_end is None else int(z_end)
        return arr[zs:ze]


def _ref_apply_lut(frame, lo, hi, gamma):
    """Independent re-implementation of widgets/lut_histogram.apply_lut."""
    f = frame.astype(np.float32)
    if hi <= lo:
        hi = lo + 1.0
    f = (f - lo) / (hi - lo)
    f = np.clip(f, 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-3:
        f = f ** (1.0 / max(gamma, 0.05))
    return (f * 255).astype(np.uint8)


def test_to_uint8_matches_apply_lut_per_plane():
    rng = np.random.default_rng(0)
    vol = (rng.random((5, 8, 8)) * 4000).astype(np.uint16)
    out = prep.to_uint8(vol, 100.0, 3000.0, 1.0)
    assert out.dtype == np.uint8 and out.shape == vol.shape
    for z in range(vol.shape[0]):
        assert np.array_equal(out[z], _ref_apply_lut(vol[z], 100.0, 3000.0, 1.0))


def test_to_uint8_gamma_and_degenerate_bounds():
    vol = np.linspace(0, 1000, 64, dtype=np.float32).reshape(1, 8, 8)
    assert np.array_equal(prep.to_uint8(vol, 0.0, 1000.0, 2.2)[0],
                          _ref_apply_lut(vol[0], 0.0, 1000.0, 2.2))
    flat = prep.to_uint8(np.zeros((1, 4, 4), np.float32), 5.0, 5.0, 1.0)
    assert flat.min() == 0 and flat.max() == 0


def test_spacing_normalized_preserves_ratio():
    sp = prep.Spacing(dz=2.0, dy=0.5, dx=0.5)
    assert sp.as_tuple == (2.0, 0.5, 0.5)
    nz, ny, nx = sp.normalized()
    assert (nx, ny, nz) == pytest.approx((1.0, 1.0, 4.0))


def test_spacing_from_volume_defaults_and_reads():
    vol = FakeVolume({"c0": np.zeros((1, 1, 3, 4, 4), np.uint16)},
                     pixel_size_um=0.325, z_step_um=1.0)
    sp = prep.spacing_from_volume(vol)
    assert (sp.dx, sp.dy, sp.dz) == pytest.approx((0.325, 0.325, 1.0))
    assert prep.spacing_from_volume(object()).as_tuple == (1.0, 1.0, 1.0)


@pytest.mark.parametrize("n_z,zs,ze,expect", [
    (10, None, None, (0, 10)),
    (10, 3, 7, (3, 7)),
    (10, -5, 999, (0, 10)),
    (10, 8, 8, (8, 9)),
    (1, None, None, (0, 1)),
    (0, 0, 5, (0, 0)),
])
def test_clamp_z_range(n_z, zs, ze, expect):
    assert prep.clamp_z_range(n_z, zs, ze) == expect


def test_channel_volume_shape_and_zrange():
    data = np.arange(6 * 4 * 4, dtype=np.uint16).reshape(1, 1, 6, 4, 4)
    vol = FakeVolume({"dapi": data})
    assert prep.channel_volume(vol, 0).shape == (6, 4, 4)
    sub = prep.channel_volume(vol, 0, z_start=1, z_end=3)
    assert sub.shape == (2, 4, 4) and np.array_equal(sub, data[0, 0, 1:3])


def test_channel_volume_single_z_promotes_to_3d():
    class SingleZ(FakeVolume):
        def get_volume(self, c, m=0, t=0, z_start=None, z_end=None):
            return np.ones((4, 4), np.uint16)
    vol = SingleZ({"c": np.ones((1, 1, 1, 4, 4), np.uint16)})
    assert prep.channel_volume(vol, 0).shape == (1, 4, 4)


def test_build_channel_volumes_respects_enabled_and_lut():
    rng = np.random.default_rng(2)
    chans = {"green": (rng.random((1, 1, 3, 4, 4)) * 1000).astype(np.uint16),
             "red": (rng.random((1, 1, 3, 4, 4)) * 1000).astype(np.uint16)}
    vol = FakeVolume(chans)
    cd = {"green": {"enabled": True, "color": "green",
                    "lut_lo": 0.0, "lut_hi": 500.0, "lut_gamma": 1.0},
          "red": {"enabled": False, "color": "red"}}
    built = prep.build_channel_volumes(vol, channel_display=cd)
    assert [b.name for b in built] == ["green"]
    b = built[0]
    assert b.data.dtype == np.uint8 and b.data.shape == (3, 4, 4)
    assert b.color == pytest.approx((0.0, 1.0, 0.0)) and b.lut == (0.0, 500.0, 1.0)
    assert np.array_equal(b.data, prep.to_uint8(chans["green"][0, 0], 0.0, 500.0, 1.0))


def test_build_channel_volumes_auto_and_progress():
    chans = {"a": np.zeros((1, 1, 2, 3, 3), np.uint16),
             "b": np.zeros((1, 1, 2, 3, 3), np.uint16)}
    seen = []
    built = prep.build_channel_volumes(FakeVolume(chans), channel_display={},
                                       progress_cb=seen.append)
    assert len(built) == 2 and seen and seen[-1] == 100


def test_color_rgb_float_variants():
    assert prep.color_rgb_float("red") == pytest.approx((1.0, 0.0, 0.0))
    assert prep.color_rgb_float((0, 255, 0)) == pytest.approx((0.0, 1.0, 0.0))
    assert prep.color_rgb_float(None) == pytest.approx((1.0, 1.0, 1.0))
