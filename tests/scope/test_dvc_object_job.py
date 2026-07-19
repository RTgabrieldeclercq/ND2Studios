"""Headless test for the V1.68 per-object `_DVCJob` crop-mask alignment logic.

`_DVCJob` lives in the (Qt-importing) pipelines page, but its constructor and the
`_object_crop_mask` helper run no Qt / no DVC engine, so they can be exercised
headlessly. This pins the object-crop mask to the (rect ∩ bbox) footprint that the
per-object DVC field is computed on, so the object's field and its mask stay
aligned for the 3-D object surface.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pp = pytest.importorskip("nd2studios.pages.pipelines_page")
from nd2studios.backend.analysis.object_scope import ObjectRegion  # noqa: E402


def _job(object_regions=None):
    # Constructor stores state only — no volume read, no engine.
    return pp._DVCJob(
        "k", vol=object(), c_idx=0, m_list=[0], frames=[1], ref_frame=0,
        mode="cumulative", z_start=0, z_end=None, downsample=1,
        voxel_size_um=(1.0, 1.0, 1.0), params={}, rect=None,
        object_regions=object_regions)


def test_whole_frame_job_has_no_object_regions():
    assert _job()._object_regions is None


def test_object_regions_stored():
    region = ObjectRegion(1, (0, 4, 2, 8, 3, 9), np.ones((4, 6, 6), bool), "mask3d")
    job = _job([region])
    assert job._object_regions is not None
    assert len(job._object_regions) == 1


def test_object_crop_mask_3d_full_bbox():
    # No base-rect intersection: crop footprint == full bbox, mask unchanged.
    mask = np.zeros((4, 6, 6), bool)
    mask[:, 1:4, 2:5] = True
    region = ObjectRegion(1, (0, 4, 2, 8, 3, 9), mask, "mask3d")
    job = _job([region])
    out = job._object_crop_mask(region, ax0=3, ay0=2, ax1=9, ay1=8)
    assert out.shape == (4, 6, 6)
    assert np.array_equal(out, mask)


def test_object_crop_mask_3d_intersected():
    # A base rect shrinks the crop to the bbox interior → mask cropped to match.
    mask = np.zeros((4, 6, 6), bool)
    mask[:, :, :] = True
    region = ObjectRegion(1, (0, 4, 2, 8, 3, 9), mask, "mask3d")   # bbox y[2:8] x[3:9]
    # crop footprint y[4:7] x[5:8] (in full-frame coords) → local mask [2:5, 2:5]
    out = _job([region])._object_crop_mask(region, ax0=5, ay0=4, ax1=8, ay1=7)
    assert out.shape == (4, 3, 3)
    assert np.array_equal(out, mask[:, 2:5, 2:5])


def test_object_crop_mask_2d_label_region():
    mask2d = np.zeros((6, 6), bool)
    mask2d[1:4, 1:4] = True
    region = ObjectRegion(2, (0, 0, 10, 16, 20, 26), mask2d, "label")  # y[10:16] x[20:26]
    job = _job([region])
    out = job._object_crop_mask(region, ax0=20, ay0=10, ax1=26, ay1=16)
    assert out.ndim == 2 and out.shape == (6, 6)
    assert np.array_equal(out, mask2d)
