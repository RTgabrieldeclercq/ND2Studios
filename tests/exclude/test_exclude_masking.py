"""V1.75 Exclude node — the masking math (headless, offscreen Qt).

The exclusion helpers live on the Qt-importing Pipelines page, but they run no Qt
themselves, so we borrow the self-contained ones onto a light stub (mirroring
``tests/scope/test_dvc_object_job.py``) and exercise them directly:

- ``_union_frames_bool`` / ``_accumulate_mask`` / ``_merge_exclusion`` — build the
  ``record._exclude_by_m`` volume (union over T + multiple sources; optional dilate).
- ``_apply_exclusion_channels`` — zeroes the excluded footprint in the analysis
  channels across all T, crop-aligned.
- ``_DVCJob._read_volume`` — zeroes the excluded voxels in the DVC read.
"""
from __future__ import annotations

import os
import types

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pp = pytest.importorskip("nd2studios.pages.pipelines_page")
PipelinesPage = pp.PipelinesPage


class _Stub:
    """A minimal object that borrows the real exclusion methods (no widget)."""

    _accumulate_mask = staticmethod(PipelinesPage._accumulate_mask)
    _union_frames_bool = staticmethod(PipelinesPage._union_frames_bool)
    _apply_exclusion_channels = PipelinesPage._apply_exclusion_channels
    _merge_exclusion = PipelinesPage._merge_exclusion

    def __init__(self, rect=None):
        self._rect = rect

    def _crop_rect(self):
        return self._rect


def _record(**kw):
    return types.SimpleNamespace(**kw)


# ── union / accumulate ──────────────────────────────────────────────────────

def test_union_frames_bool_over_t():
    m0 = np.zeros((2, 4, 4), bool)
    m0[:, 0, 0] = True
    m1 = np.zeros((2, 4, 4), bool)
    m1[:, 3, 3] = True
    u = PipelinesPage._union_frames_bool({0: m0, 5: m1})
    assert u.shape == (2, 4, 4)
    assert u[0, 0, 0] and u[0, 3, 3]          # union of both timepoints
    assert int(u.sum()) == 4                  # 2 pixels × 2 Z


def test_union_frames_bool_promotes_2d():
    u = PipelinesPage._union_frames_bool(np.ones((4, 4), bool))
    assert u.shape == (1, 4, 4)


def test_accumulate_mask_ors_into_dict():
    out = {}
    a = np.zeros((1, 4, 4), bool)
    a[0, 1, 1] = True
    PipelinesPage._accumulate_mask(out, 0, a)
    b = np.zeros((1, 4, 4), bool)
    b[0, 2, 2] = True
    PipelinesPage._accumulate_mask(out, 0, b)
    assert int(out[0].sum()) == 2


def test_merge_exclusion_or_and_dilate():
    stub = _Stub()
    rec = _record()
    a = np.zeros((1, 6, 6), bool)
    a[0, 2:4, 2:4] = True                     # 4 px
    stub._merge_exclusion(rec, 0, a, dilate=0)
    assert rec._exclude_by_m[0].shape == (1, 6, 6)
    assert int(rec._exclude_by_m[0].sum()) == 4
    b = np.zeros((1, 6, 6), bool)
    b[0, 0, 0] = True
    stub._merge_exclusion(rec, 0, b, dilate=0)  # OR into the same M
    assert int(rec._exclude_by_m[0].sum()) == 5
    rec2 = _record()
    stub._merge_exclusion(rec2, 0, a, dilate=1)  # grow the footprint
    assert int(rec2._exclude_by_m[0].sum()) > 4


# ── apply to analysis channels ──────────────────────────────────────────────

def test_apply_exclusion_channels_zeros_footprint_all_t():
    Z, H, W, T = 3, 8, 10, 4
    vol = np.zeros((Z, H, W), bool)
    vol[:, 2:5, 3:6] = True
    rec = _record(_exclude_by_m={0: vol})
    out = {"ch": np.ones((T, H, W), np.float32)}
    res = _Stub(rect=None)._apply_exclusion_channels(rec, out, 0)
    fp = vol.any(axis=0)
    a = res["ch"]
    assert np.all(a[:, fp] == 0)              # ignored across every timepoint
    assert np.all(a[:, ~fp] == 1)             # everything else untouched


def test_apply_exclusion_channels_crop_aligned():
    Z, H, W, T = 2, 10, 10, 2
    vol = np.zeros((Z, H, W), bool)
    vol[:, 4:7, 5:8] = True
    rec = _record(_exclude_by_m={0: vol})
    x, y, w, h = 3, 2, 6, 6                    # crop rect; channel already cropped
    out = {"ch": np.ones((T, h, w), np.float32)}
    res = _Stub(rect=(x, y, w, h))._apply_exclusion_channels(rec, out, 0)
    fp_crop = vol.any(axis=0)[y:y + h, x:x + w]
    a = res["ch"]
    assert a.shape == (T, h, w)
    assert np.all(a[:, fp_crop] == 0)
    assert np.all(a[:, ~fp_crop] == 1)


def test_apply_exclusion_channels_noop_without_mask():
    out = {"ch": np.ones((2, 5, 5), np.float32)}
    res = _Stub()._apply_exclusion_channels(_record(_exclude_by_m={}), out, 0)
    assert np.all(res["ch"] == 1)


# ── DVC read path ───────────────────────────────────────────────────────────

def test_dvc_read_volume_zeros_exclusion():
    Z, H, W = 3, 6, 6
    base = np.arange(Z * H * W, dtype=np.float32).reshape(Z, H, W) + 1.0  # all > 0

    class _Vol:
        def get_volume(self, c, m, t, z_start=0, z_end=None):
            ze = Z if z_end is None else z_end
            return base[(z_start or 0):ze].copy()

    exv = np.zeros((Z, H, W), bool)
    exv[:, 1:3, 2:4] = True
    job = pp._DVCJob(
        "k", vol=_Vol(), c_idx=0, m_list=[0], frames=[1], ref_frame=0,
        mode="cumulative", z_start=0, z_end=None, downsample=1,
        voxel_size_um=(1.0, 1.0, 1.0), params={}, rect=None,
        exclude_by_m={0: exv})
    v = np.asarray(job._read_volume(0, 1))
    assert v.shape == (Z, H, W)
    assert np.all(v[exv] == 0)                # excluded voxels zeroed
    assert np.all(v[~exv] == base[~exv])      # the rest is the raw volume
