"""Headless tests for ``nd2studios.backend.analysis.object_scope`` (V1.68
Frame/Object scope toggle) — object enumeration from mask / label volumes.
"""
from __future__ import annotations

import numpy as np
import pytest

from nd2studios.backend.analysis.object_scope import ObjectRegion, iter_objects


def test_two_blobs_3d_bool_mask():
    vol = np.zeros((20, 40, 40), bool)
    vol[2:6, 5:10, 5:10] = True         # blob A (Z-scoped)
    vol[10:14, 25:32, 25:35] = True     # blob B
    regs = iter_objects(vol)
    assert len(regs) == 2
    for r in regs:
        assert r.source == "mask3d"
        assert r.z_scoped                     # a 3-D object scopes Z
    # Boxes match the drawn blobs (order by object_id / discovery).
    boxes = sorted(r.bbox for r in regs)
    assert boxes[0] == (2, 6, 5, 10, 5, 10)
    assert boxes[1] == (10, 14, 25, 32, 25, 35)
    # Each region's mask matches its box footprint.
    for r in regs:
        z0, z1, y0, y1, x0, x1 = r.bbox
        assert r.mask.shape == (z1 - z0, y1 - y0, x1 - x0)
        assert r.n_voxels == int(vol[z0:z1, y0:y1, x0:x1].sum())


def test_label_mask_unions_over_time():
    # (T,H,W) int labels: object 1 drifts in XY across T; its box is the union.
    labels = np.zeros((3, 30, 30), np.int32)
    labels[0, 2:6, 2:6] = 1
    labels[1, 4:8, 4:8] = 1
    labels[2, 6:10, 6:10] = 1
    labels[0, 20:24, 20:24] = 2
    regs = iter_objects(labels)
    assert len(regs) == 2
    by_id = {r.object_id: r for r in regs}
    r1 = by_id[1]
    assert r1.source == "label"
    assert not r1.z_scoped                    # 2-D label source keeps full Z
    # XY box is the union over all three frames: y,x in [2,10).
    _z0, _z1, y0, y1, x0, x1 = r1.bbox
    assert (y0, y1, x0, x1) == (2, 10, 2, 10)
    assert r1.mask.ndim == 2 and r1.mask.shape == (8, 8)


def test_pad_dilates_and_clamps():
    vol = np.zeros((10, 20, 20), bool)
    vol[4:6, 8:12, 8:12] = True
    regs = iter_objects(vol, pad=3)
    assert len(regs) == 1
    z0, z1, y0, y1, x0, x1 = regs[0].bbox
    assert (y0, y1, x0, x1) == (5, 15, 5, 15)     # 8-3 .. 12+3
    assert (z0, z1) == (1, 9)                       # 4-3 .. 6+3

    # Clamp at the array border.
    edge = np.zeros((10, 20, 20), bool)
    edge[0:2, 0:3, 0:3] = True
    r = iter_objects(edge, pad=5)[0]
    z0, z1, y0, y1, x0, x1 = r.bbox
    assert z0 == 0 and y0 == 0 and x0 == 0        # not negative


def test_min_voxels_drops_specks():
    vol = np.zeros((10, 20, 20), bool)
    vol[5, 5, 5] = True                            # 1-voxel speck
    vol[2:6, 10:15, 10:15] = True                  # real blob
    assert len(iter_objects(vol, min_voxels=1)) == 2
    regs = iter_objects(vol, min_voxels=10)
    assert len(regs) == 1
    assert regs[0].n_voxels >= 10


def test_empty_and_all_background():
    assert iter_objects(np.zeros((5, 5, 5), bool)) == []
    assert iter_objects(np.zeros((3, 8, 8), np.int32)) == []
    assert iter_objects(np.zeros((0, 0, 0), bool)) == []


def test_rect_xywh_matches_bbox():
    vol = np.zeros((8, 20, 20), bool)
    vol[1:4, 6:11, 7:13] = True
    r = iter_objects(vol)[0]
    assert r.rect_xywh == (7, 6, 6, 5)             # (x0, y0, w, h)


def test_float_mask_treated_as_boolean():
    vol = np.zeros((6, 12, 12), np.float32)
    vol[1:3, 3:6, 3:6] = 0.9
    regs = iter_objects(vol)
    assert len(regs) == 1
    assert regs[0].source == "mask3d"
