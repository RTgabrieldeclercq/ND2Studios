"""Headless tests for the V1.68 edge-scope model helpers + ObjectCropVolume +
node_produces_objects (pure numpy / dataclasses, no Qt, no display).
"""
from __future__ import annotations

import numpy as np
import pytest

from nd2studios.pipeline_graph.model import (
    SCOPE_OBJECTS, SCOPE_WHOLE, Edge, edge_scope, set_edge_scope,
)
from nd2studios.pipeline_graph.executor import ObjectCropVolume
from nd2studios.pipeline_graph.registry_adapter import (
    ANALYSIS_PREFIX, SPECIAL_MASK3D_OP_KEY, SPECIAL_TRACK_OP_KEY,
    op_produces_objects,
)
from nd2studios.backend.analysis.object_scope import ObjectRegion


def _edge(**params):
    return Edge(id="e1", src_node="a", src_port="a.out", dst_node="b",
                dst_port="b.in", params=dict(params))


def test_edge_scope_default_is_whole_frame():
    assert edge_scope(_edge()) == SCOPE_WHOLE               # legacy graphs unchanged
    assert edge_scope(_edge(scope="garbage")) == SCOPE_WHOLE  # unknown → whole


def test_set_edge_scope_roundtrips_and_stays_tidy():
    e = _edge()
    set_edge_scope(e, SCOPE_OBJECTS)
    assert edge_scope(e) == SCOPE_OBJECTS
    assert e.params["scope"] == SCOPE_OBJECTS
    # Setting back to whole_frame clears the key (legacy-identical serialization).
    set_edge_scope(e, SCOPE_WHOLE)
    assert edge_scope(e) == SCOPE_WHOLE
    assert "scope" not in e.params


def test_edge_scope_survives_to_dict_from_dict():
    e = _edge()
    set_edge_scope(e, SCOPE_OBJECTS)
    d = e.to_dict()
    assert d["params"]["scope"] == SCOPE_OBJECTS
    e2 = Edge.from_dict(d)
    assert edge_scope(e2) == SCOPE_OBJECTS
    # A legacy edge dict with no scope key loads as whole_frame.
    legacy = {"id": "e", "src_node": "a", "src_port": "p", "dst_node": "b",
              "dst_port": "q"}
    assert edge_scope(Edge.from_dict(legacy)) == SCOPE_WHOLE


def test_op_produces_objects():
    assert op_produces_objects(SPECIAL_MASK3D_OP_KEY)
    assert op_produces_objects(SPECIAL_TRACK_OP_KEY)
    assert op_produces_objects(ANALYSIS_PREFIX + "nuclei")
    assert not op_produces_objects("enhancement:blur")
    assert not op_produces_objects("special:dvc")
    assert not op_produces_objects("special:register")


class FakeVolume:
    """(M,T,Z,H,W)-per-channel MaterializedDataset-like stand-in."""

    def __init__(self, arr):                        # arr: (M,T,Z,H,W)
        self._a = np.asarray(arr)
        self.n_multipoints, self.n_timepoints, self.n_zslices = self._a.shape[:3]
        self.height, self.width = self._a.shape[3:]
        self.channel_names = ["ch0"]
        self.pixel_size_um = 0.5
        self.z_step_um = 2.0
        self.dtype = self._a.dtype

    def get_volume(self, c=0, m=0, t=0, z_start=None, z_end=None):
        v = self._a[int(m), int(t)]
        zs = 0 if z_start is None else int(z_start)
        ze = v.shape[0] if z_end is None else int(z_end)
        return v[zs:ze]

    def get_frame(self, c=0, m=0, t=0, z=0, z_mode="none", z_start=None,
                  z_end=None, **kw):
        v = self._a[int(m), int(t)]
        if z_mode == "none":
            return v[int(z)]
        zs = 0 if z_start is None else int(z_start)
        ze = v.shape[0] if z_end is None else int(z_end)
        sub = v[zs:ze]
        return {"max": sub.max, "mean": sub.mean, "min": sub.min}[z_mode](axis=0)


def _fake_volume(M=2, T=3, Z=8, H=20, W=24):
    # Encode coords into pixel value so crops can be verified exactly.
    a = np.zeros((M, T, Z, H, W), np.uint16)
    zz, yy, xx = np.mgrid[0:Z, 0:H, 0:W]
    for m in range(M):
        for t in range(T):
            a[m, t] = (m * 100000 + t * 10000 + zz * 1000 + yy * 30 + xx) % 65535
    return a


def test_object_crop_volume_conserves_tmzc_and_crops_3d():
    vol = FakeVolume(_fake_volume())
    region = ObjectRegion(object_id=1, bbox=(2, 6, 5, 12, 7, 15),
                          mask=np.ones((4, 7, 8), bool), source="mask3d")
    ocv = ObjectCropVolume(vol, region)
    # T / M / channels conserved; Z + XY cropped to the object box.
    assert ocv.n_multipoints == vol.n_multipoints
    assert ocv.n_timepoints == vol.n_timepoints
    assert ocv.channel_names == vol.channel_names
    assert ocv.n_zslices == 4          # z1 - z0
    assert (ocv.height, ocv.width) == (7, 8)
    # get_volume returns the object sub-box, Z-window mapped to absolute Z.
    sub = ocv.get_volume(0, m=1, t=2)
    assert sub.shape == (4, 7, 8)
    expected = vol.get_volume(0, m=1, t=2)[2:6, 5:12, 7:15]
    assert np.array_equal(sub, expected)
    # get_frame local z=0 maps to absolute z=2, XY-cropped.
    plane = ocv.get_frame(0, m=1, t=2, z=0, z_mode="none")
    assert plane.shape == (7, 8)
    assert np.array_equal(plane, vol.get_volume(0, m=1, t=2)[2, 5:12, 7:15])


def test_object_crop_volume_label_source_keeps_full_z():
    vol = FakeVolume(_fake_volume())
    # z1 <= z0 → not z-scoped (a 2-D label source), full Z preserved.
    region = ObjectRegion(object_id=3, bbox=(0, 0, 4, 10, 6, 14),
                          mask=np.ones((6, 8), bool), source="label")
    ocv = ObjectCropVolume(vol, region)
    assert ocv.n_zslices == vol.n_zslices        # full Z conserved
    assert (ocv.height, ocv.width) == (6, 8)
    sub = ocv.get_volume(0, m=0, t=1)
    assert sub.shape == (vol.n_zslices, 6, 8)
    assert np.array_equal(sub, vol.get_volume(0, m=0, t=1)[:, 4:10, 6:14])


def test_object_crop_volume_mask_out_projection_uses_z_union():
    # A Z-projected get_frame with mask_out must keep voxels the object occupies in
    # ANY plane (union), not just plane 0 (regression for the single-plane bug).
    vol = FakeVolume(_fake_volume(Z=4, H=12, W=12))
    mask = np.zeros((2, 5, 5), bool)
    mask[0, 1, 1] = True          # object present only in plane 0 here…
    mask[1, 3, 3] = True          # …and only in plane 1 here
    region = ObjectRegion(object_id=1, bbox=(1, 3, 2, 7, 3, 8), mask=mask,
                          source="mask3d")
    ocv = ObjectCropVolume(vol, region, mask_out=True)
    plane = ocv.get_frame(0, m=0, t=0, z=0, z_mode="max")   # projection over Z
    # Both object voxels survive the union mask (neither is zeroed).
    assert plane[1, 1] != 0
    assert plane[3, 3] != 0
    # A voxel the object never occupies is zeroed.
    assert plane[0, 0] == 0


def test_object_crop_volume_mask_out_zeroes_exterior():
    vol = FakeVolume(_fake_volume(Z=4, H=12, W=12))
    mask = np.zeros((2, 5, 5), bool)
    mask[:, 1:4, 1:4] = True                      # interior 3×3 of the 5×5 box
    region = ObjectRegion(object_id=1, bbox=(1, 3, 2, 7, 3, 8), mask=mask,
                          source="mask3d")
    ocv = ObjectCropVolume(vol, region, mask_out=True)
    sub = ocv.get_volume(0, m=0, t=0)
    assert sub.shape == (2, 5, 5)
    # Exterior voxels zeroed, interior preserved.
    assert np.all(sub[:, 0, :] == 0) and np.all(sub[:, :, 0] == 0)
    assert np.all(sub[mask] == vol.get_volume(0)[1:3, 2:7, 3:8][mask])
