"""Headless test for the V1.74 per-granule selector in ``widgets.dvc_panel.DVCPanel``.

Drives ``set_data(objects=...)`` + ``_on_object_changed`` on an offscreen panel and
asserts the combo is populated (largest granule default, "All granules" entry) and
that selecting a granule / "All" swaps the active field series + masks. No DVC
engine, no 3-D viewer (surface builds are off-thread and not awaited here).
"""
from __future__ import annotations

import os
import types

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtWidgets import QApplication  # noqa: E402

dvc_panel = pytest.importorskip("nd2studios.widgets.dvc_panel")

_app = QApplication.instance() or QApplication([])


def _result():
    """A minimal 3-D DVCResult-like usable by field_bundle_from_result."""
    g = np.arange(2, dtype=float)
    GZ, GY, GX = np.meshgrid(g, g, g, indexing="ij")
    coords = np.stack([GZ, GY, GX], axis=-1)                 # (2,2,2,3)
    return types.SimpleNamespace(
        dim=3, grid_coords=coords,
        displacement_field=np.zeros_like(coords),
        voxel_size_um=(1.0, 1.0, 1.0), strain_field=None, qfactor=None,
        method="test", iterations=1, converged=True, beta=1.0,
        diagnostics={"grid_shape": (2, 2, 2), "median_zncc": 0.0})


def _bundle(n_voxels, origin, mzhw):
    mask = np.ones(mzhw, bool)
    return {
        "series": {1: _result(), 2: _result()},
        "bg": {1: np.zeros((4, 4)), 2: np.ones((4, 4))},
        "increment": {2: _result()},
        "mask": {1: mask, 2: mask},
        "n_voxels": n_voxels,
        "origin": origin,
    }


def _feed(panel, objects, largest):
    lb = objects[largest]
    panel.set_data(lb["series"], lb["bg"], [1, 2], m=0, n_frames_total=3,
                   masks=lb["mask"], mask_voxel_size=(1.0, 1.0, 1.0),
                   objects=objects)


def test_selector_populated_defaults_to_largest():
    panel = dvc_panel.DVCPanel()
    objects = {1: _bundle(12, (0, 10, 20), (2, 2, 2)),
               2: _bundle(48, (1, 30, 40), (2, 4, 4))}
    _feed(panel, objects, largest=2)

    cb = panel.cmb_object
    assert cb.count() == 3
    assert cb.itemData(0) == "__all__"            # composite entry first
    assert cb.itemData(1) == 2 and cb.itemData(2) == 1   # largest-first order
    assert cb.currentIndex() == 1                 # default = largest granule
    assert panel._active_oid == 2
    assert panel._show_all_objects is False


def test_selecting_a_granule_swaps_series_and_masks():
    panel = dvc_panel.DVCPanel()
    objects = {1: _bundle(12, (0, 10, 20), (2, 2, 2)),
               2: _bundle(48, (1, 30, 40), (2, 4, 4))}
    _feed(panel, objects, largest=2)

    panel.cmb_object.setCurrentIndex(panel.cmb_object.findData(1))
    assert panel._active_oid == 1
    assert panel._show_all_objects is False
    # Masks now come from granule 1's bundle (2×2×2), not granule 2's (2×4×4).
    assert panel._masks[1].shape == (2, 2, 2)


def test_selecting_all_sets_composite_flag_and_keeps_largest_series():
    panel = dvc_panel.DVCPanel()
    objects = {1: _bundle(12, (0, 10, 20), (2, 2, 2)),
               2: _bundle(48, (1, 30, 40), (2, 4, 4))}
    _feed(panel, objects, largest=2)

    panel.cmb_object.setCurrentIndex(0)           # "All granules"
    assert panel._show_all_objects is True
    # 2-D views still reference a single granule (the largest) while composited.
    assert panel._active_oid == 2
    assert panel._masks[1].shape == (2, 4, 4)


def test_multi_surface_worker_builds_and_merges_offset_composite():
    """The off-thread composite builder (run synchronously here) turns per-granule
    (result, mask, offset) triples into ONE merged, spatially-separated SurfaceField."""
    mask = np.ones((2, 4, 4), bool)
    items = [(_result(), mask, (0.0, 0.0, 0.0)),
             (_result(), mask, (100.0, 0.0, 0.0))]   # 2nd granule far in +x
    key = (("__all__", False, 1, "disp_mag", 0))
    worker = dvc_panel._MultiSurfaceFieldWorker(
        items, (1.0, 1.0, 1.0), "disp_mag", key, gen=7, smooth_iterations=0)
    captured = []
    worker.done.connect(lambda k, sf, g: captured.append((k, sf, g)))
    worker.run()                                     # synchronous (not .start())

    assert len(captured) == 1
    k, sf, g = captured[0]
    assert k == key and g == 7
    assert sf is not None and not sf.is_empty
    assert sf.mdm is None                            # composite MDM intentionally None
    assert float(np.ptp(sf.vertices_um[:, 0])) > 50.0   # two components separated in x


def test_no_objects_hides_selector():
    panel = dvc_panel.DVCPanel()
    panel.set_data({1: _result()}, {1: np.zeros((4, 4))}, [1], m=0,
                   n_frames_total=2, mask_voxel_size=(1.0, 1.0, 1.0))
    assert panel._objects == {}
    assert panel.cmb_object.count() == 0
    assert panel._active_oid is None
    assert panel._show_all_objects is False
