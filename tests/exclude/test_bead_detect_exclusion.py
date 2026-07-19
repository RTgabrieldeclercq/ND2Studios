"""V1.75 — an Exclude node must also blank the region for BEAD DETECTION.

Bead detection (`special:bead_detect`) reads its volume through the granule
worker's own path (`_GranuleJob._read_registered_cropped`), not the analysis /
DVC seams, so exclusion has to be applied there too. Regression for: "Exclude
node did not exclude areas drawn by the mask — bead detection detected beads in
those areas." Headless (constructs the Qt-free read helper only).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pp = pytest.importorskip("nd2studios.pages.pipelines_page")


class _Vol:
    """Minimal volume stub with the `get_volume(c, m=, t=)` read protocol."""

    def __init__(self, base):
        self._base = base
        self.channel_names = ["c0"]

    def get_volume(self, c, m, t, **kw):
        return self._base.copy()


def _job():
    return pp._GranuleJob("k", "detect", {})


def test_bead_read_zeros_exclusion_no_crop():
    Z, H, W = 3, 20, 20
    base = np.full((Z, H, W), 100.0, np.float32)
    exv = np.zeros((Z, H, W), bool)
    exv[:, 4:10, 5:12] = True
    v, oy, ox = _job()._read_registered_cropped(
        _Vol(base), c_idx=0, m=0, t=0, transforms=None, interp_order=1,
        rect=None, exclude={0: exv})
    assert (oy, ox) == (0, 0)
    assert np.all(v[exv] == 0)            # excluded voxels blanked
    assert np.all(v[~exv] == 100.0)       # the rest is the real volume


def test_bead_read_exclusion_crop_aligned():
    Z, H, W = 2, 20, 20
    base = np.full((Z, H, W), 50.0, np.float32)
    exv = np.zeros((Z, H, W), bool)
    exv[:, 6:12, 7:14] = True
    rect = (5, 4, 10, 10)                 # x, y, w, h → crop [4:14, 5:15]
    v, oy, ox = _job()._read_registered_cropped(
        _Vol(base), 0, 0, 0, None, 1, rect, exclude={0: exv})
    assert (oy, ox) == (4, 5)
    assert v.shape == (Z, 10, 10)
    exv_crop = exv[:, 4:14, 5:15]
    assert np.all(v[exv_crop] == 0)
    assert np.all(v[~exv_crop] == 50.0)


def test_bead_read_no_exclusion_is_untouched():
    Z, H, W = 2, 10, 10
    base = np.full((Z, H, W), 7.0, np.float32)
    v, _oy, _ox = _job()._read_registered_cropped(
        _Vol(base), 0, 0, 0, None, 1, None, exclude={})
    assert np.all(v == 7.0)


def test_bead_read_exclusion_z_mismatch_projects_footprint():
    # A (1,H,W) exclusion (or any Z != volume Z) is projected over Z and applied.
    Z, H, W = 4, 12, 12
    base = np.full((Z, H, W), 9.0, np.float32)
    exv = np.zeros((1, H, W), bool)
    exv[:, 2:5, 3:6] = True
    v, _oy, _ox = _job()._read_registered_cropped(
        _Vol(base), 0, 0, 0, None, 1, None, exclude={0: exv})
    fp = exv.any(axis=0)
    assert np.all(v[:, fp] == 0)          # footprint blanked on every Z
    assert np.all(v[:, ~fp] == 9.0)
