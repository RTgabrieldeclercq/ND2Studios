"""Headless test for ``PipelinesPage._store_dvc_objects`` (V1.74): it must keep
EVERY granule's field / backdrop / mask / crop-origin (previously only the largest
survived) so the DVC panel's per-granule selector + "All granules" composite have
data to show.

The method touches only ``self._dvc_obj_by_m`` / ``_dvc_obj_full_by_m`` /
``_dvc_display_mask_by_m`` (all freshly assigned), so it is exercised on a
lightweight stand-in ``self`` — no Qt widget tree, no DVC engine.
"""
from __future__ import annotations

import os
import types

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pp = pytest.importorskip("nd2studios.pages.pipelines_page")


def _res(tag):
    return types.SimpleNamespace(tag=tag)   # DVCResult stand-in (stored, not read)


def _value():
    """Two granules, one M (0), two frames (1, 2); granule 2 is larger."""
    mask1 = np.zeros((3, 4, 4), bool); mask1[:, 1:3, 1:3] = True     # 12 vox
    mask2 = np.zeros((3, 6, 6), bool); mask2[:, 1:5, 1:5] = True     # 48 vox
    objects = {
        1: {0: {"primary": {1: (_res("g1t1"), np.zeros((4, 4))),
                            2: (_res("g1t2"), np.ones((4, 4)))},
               "increment": {2: _res("g1i2")}}},
        2: {0: {"primary": {1: (_res("g2t1"), np.zeros((6, 6))),
                            2: (_res("g2t2"), np.ones((6, 6)))},
               "increment": {2: _res("g2i2")}}},
    }
    return {
        "objects": objects,
        "object_masks": {1: mask1, 2: mask2},
        "object_origins": {1: (0, 10, 20), 2: (1, 30, 40)},
    }


def test_store_keeps_all_objects_full_bundles():
    ns = types.SimpleNamespace()
    disp = pp.PipelinesPage._store_dvc_objects(ns, _value())

    # Full store: both granules present for M0.
    full = ns._dvc_obj_full_by_m[0]
    assert set(full) == {1, 2}

    g1, g2 = full[1], full[2]
    # Per-object series / backgrounds / increments preserved (were discarded before).
    assert set(g1["series"]) == {1, 2}
    assert g1["series"][1].tag == "g1t1"
    assert set(g1["bg"]) == {1, 2}
    assert g1["increment"][2].tag == "g1i2"
    # Voxel counts + crop origins carried through for sorting / composite offsets.
    assert g1["n_voxels"] == 12
    assert g2["n_voxels"] == 48
    assert g1["origin"] == (0, 10, 20)
    assert g2["origin"] == (1, 30, 40)
    # Mask broadcast across the object's computed frames.
    assert set(g1["mask"]) == {1, 2}
    assert g1["mask"][1].shape == (3, 4, 4)


def test_store_returns_largest_object_bundle_for_default_display():
    ns = types.SimpleNamespace()
    disp = pp.PipelinesPage._store_dvc_objects(ns, _value())
    # Largest = granule 2; the returned per-m bundle is granule 2's (default view),
    # and the display mask is granule 2's, matching pre-V1.74 behavior.
    assert disp[0]["primary"][1][0].tag == "g2t1"
    dm = ns._dvc_display_mask_by_m[0]
    assert set(dm) == {1, 2}
    assert int(dm[1].sum()) == 48


def test_store_empty_objects_returns_empty():
    ns = types.SimpleNamespace()
    assert pp.PipelinesPage._store_dvc_objects(ns, {"objects": {}}) == {}
