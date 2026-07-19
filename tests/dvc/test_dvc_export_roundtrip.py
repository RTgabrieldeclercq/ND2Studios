"""Headless round-trip tests for the V1.76 portable DVC bundle (``dvc_export``).

Builds a small synthetic all-multipoints DVC payload (cumulative + increment
series, per-frame backdrops, whole-frame masks, and a per-granule object bundle
with a mask broadcast across frames), saves it to a ``.nd2dvc`` file, loads it back,
and asserts the reconstruction is faithful — including that identity-shared arrays
(the broadcast object mask) are deduplicated to a single stored array.
"""
from __future__ import annotations

import numpy as np

from nd2studios.backend.dvc_export import (
    DVC_BUNDLE_EXTENSION, load_dvc_bundle, save_dvc_bundle,
)
from nd2studios.core.dvc_registry import DVCResult


def _make_result(dim: int = 3, seed: int = 0) -> DVCResult:
    rng = np.random.default_rng(seed)
    if dim == 3:
        grid = (2, 3, 4)
    else:
        grid = (3, 4)
    coords = rng.random(grid + (dim,)).astype(np.float64)
    disp = rng.standard_normal(grid + (dim,)).astype(np.float64)
    strain = rng.standard_normal(grid + (dim * dim,)).astype(np.float32)
    qf = rng.random(grid).astype(np.float32)
    return DVCResult(
        dim=dim, grid_coords=coords, displacement_field=disp,
        voxel_size_um=tuple(0.1 * (i + 1) for i in range(dim)),
        strain_field=strain, strain_type="infinitesimal", qfactor=qf,
        converged=True, iterations=4, mu=1e-3, beta=0.5,
        method="ALDVC", notes="synthetic",
        diagnostics={"grid_shape": list(grid), "median_zncc": 0.987},
    )


def _assert_result_equal(a: DVCResult, b: DVCResult) -> None:
    assert a.dim == b.dim
    assert tuple(a.voxel_size_um) == tuple(b.voxel_size_um)
    assert a.method == b.method and a.strain_type == b.strain_type
    assert a.converged == b.converged and a.iterations == b.iterations
    assert a.mu == b.mu and a.beta == b.beta and a.notes == b.notes
    np.testing.assert_array_equal(a.grid_coords, b.grid_coords)
    np.testing.assert_array_equal(a.displacement_field, b.displacement_field)
    np.testing.assert_array_equal(a.strain_field, b.strain_field)
    np.testing.assert_array_equal(a.qfactor, b.qfactor)
    assert list(a.diagnostics["grid_shape"]) == list(b.diagnostics["grid_shape"])
    assert float(a.diagnostics["median_zncc"]) == float(b.diagnostics["median_zncc"])


def test_roundtrip_all_stores(tmp_path):
    # Two multipoints; frame 0 is the reference (no field), frames 1-2 have fields.
    series_by_m = {
        0: {1: _make_result(3, 1), 2: _make_result(3, 2)},
        1: {1: _make_result(3, 3)},
    }
    incr_by_m = {0: {2: _make_result(3, 20)}}
    bg_by_mt = {
        0: {0: np.arange(12, dtype=np.uint16).reshape(3, 4),
            1: np.arange(12, 24, dtype=np.uint16).reshape(3, 4)},
        1: {1: np.ones((3, 4), dtype=np.uint16)},
    }
    whole_masks_by_m = {0: {1: (np.arange(24).reshape(2, 3, 4) % 2).astype(bool)}}

    # One granule object on M0: its mask is the SAME array across both frames
    # (broadcast in _store_dvc_objects) — must dedupe to one stored array.
    shared_mask = (np.arange(24).reshape(2, 3, 4) > 10)
    obj_full_by_m = {
        0: {
            7: {
                "series": {1: _make_result(3, 71), 2: _make_result(3, 72)},
                "increment": {2: _make_result(3, 73)},
                "bg": {1: np.full((3, 4), 5, np.uint16),
                       2: np.full((3, 4), 6, np.uint16)},
                "mask": {1: shared_mask, 2: shared_mask},
                "n_voxels": int(shared_mask.sum()),
                "origin": (1, 2, 3),
            }
        }
    }
    display_mask_by_m = {0: {1: shared_mask, 2: shared_mask}}

    meta = {
        "n_multipoints": 2,
        "context_channels": ["GFP", "RFP"],
        "by_m": {
            "0": {"n_frames_total": 3, "pixel_size": 0.325,
                  "field_shape": [3, 4], "mask_voxel_size": [1.0, 0.325, 0.325],
                  "frame_times_s": [0.0, 10.0, 20.0],
                  "surface_smooth_iterations": 10},
            "1": {"n_frames_total": 3, "pixel_size": 0.325,
                  "field_shape": [3, 4], "mask_voxel_size": [1.0, 0.325, 0.325],
                  "frame_times_s": None, "surface_smooth_iterations": 10},
        },
    }
    provenance = {
        "app_version": "1.76", "created": "2026-07-14T00:00:00",
        "source_file": "WellB1.nd2",
        "dvc": {"method": "ALDVC", "subset_size": 16, "tracking_mode": "cumulative"},
        "upstream": [{"title": "3D Mask Drawing", "op_key": "special:mask3d",
                      "params": {"surface_smooth_iterations": 10}}],
        "summary": {"n_multipoints": 2, "n_frames": 3, "n_objects": 1, "dim": 3},
    }

    path = str(tmp_path / ("bundle" + DVC_BUNDLE_EXTENSION))
    save_dvc_bundle(
        path, series_by_m=series_by_m, incr_by_m=incr_by_m, bg_by_mt=bg_by_mt,
        whole_masks_by_m=whole_masks_by_m, obj_full_by_m=obj_full_by_m,
        display_mask_by_m=display_mask_by_m, meta=meta, provenance=provenance)

    b = load_dvc_bundle(path)

    # Series + increments.
    assert set(b["series_by_m"]) == {0, 1}
    assert set(b["series_by_m"][0]) == {1, 2}
    _assert_result_equal(series_by_m[0][1], b["series_by_m"][0][1])
    _assert_result_equal(series_by_m[0][2], b["series_by_m"][0][2])
    _assert_result_equal(series_by_m[1][1], b["series_by_m"][1][1])
    _assert_result_equal(incr_by_m[0][2], b["incr_by_m"][0][2])

    # Backdrops + whole masks.
    np.testing.assert_array_equal(bg_by_mt[0][1], b["bg_by_mt"][0][1])
    np.testing.assert_array_equal(whole_masks_by_m[0][1], b["whole_masks_by_m"][0][1])
    assert b["whole_masks_by_m"][0][1].dtype == bool

    # Object bundle.
    obj = b["obj_full_by_m"][0][7]
    _assert_result_equal(obj_full_by_m[0][7]["series"][1], obj["series"][1])
    _assert_result_equal(obj_full_by_m[0][7]["increment"][2], obj["increment"][2])
    np.testing.assert_array_equal(shared_mask, obj["mask"][1])
    assert obj["n_voxels"] == int(shared_mask.sum())
    assert obj["origin"] == (1, 2, 3)

    # Meta + provenance survive.
    assert b["meta"]["n_multipoints"] == 2
    assert b["meta"]["context_channels"] == ["GFP", "RFP"]
    assert b["meta"]["by_m"]["0"]["pixel_size"] == 0.325
    assert b["provenance"]["source_file"] == "WellB1.nd2"
    assert b["provenance"]["upstream"][0]["op_key"] == "special:mask3d"


def test_broadcast_mask_deduplicated(tmp_path):
    """The object mask shared across frames must be stored exactly once."""
    import zipfile

    shared_mask = (np.arange(60).reshape(3, 4, 5) > 25)
    obj_full_by_m = {
        0: {
            1: {
                "series": {t: _make_result(3, 100 + t) for t in (1, 2, 3, 4, 5)},
                "increment": {},
                "bg": {},
                "mask": {t: shared_mask for t in (1, 2, 3, 4, 5)},
                "n_voxels": int(shared_mask.sum()),
                "origin": (0, 0, 0),
            }
        }
    }
    display_mask_by_m = {0: {t: shared_mask for t in (1, 2, 3, 4, 5)}}
    path = str(tmp_path / ("dedup" + DVC_BUNDLE_EXTENSION))
    save_dvc_bundle(
        path, series_by_m={0: obj_full_by_m[0][1]["series"]},
        obj_full_by_m=obj_full_by_m, display_mask_by_m=display_mask_by_m,
        meta={}, provenance={})

    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n != "__manifest__.npy"]
    # 5 distinct grid_coords + 5 displacement + 5 strain + 5 qfactor = 20 result
    # arrays; the mask (shared across 5 frames AND the display map) is exactly 1.
    n_mask_candidates = 1
    n_result_arrays = 5 * 4
    assert len(names) == n_result_arrays + n_mask_candidates

    b = load_dvc_bundle(path)
    for t in (1, 2, 3, 4, 5):
        np.testing.assert_array_equal(shared_mask, b["obj_full_by_m"][0][1]["mask"][t])
        np.testing.assert_array_equal(shared_mask, b["display_mask_by_m"][0][t])


def test_dvc_checkpoint_node_builds_and_roundtrips(tmp_path):
    """The portable DVC Checkpoint node builds port-less + white and survives
    save_pipeline / load_pipeline with its bundle_path + provenance params."""
    from nd2studios.pipeline_graph import (
        NodeCategory, PipelineDoc, SPECIAL_DVC_CHECKPOINT_OP_KEY, Stage,
        build_node, dvc_checkpoint_spec, load_pipeline, save_pipeline,
    )

    node = build_node(dvc_checkpoint_spec(), pos=(12.0, 34.0))
    assert node.op_key == SPECIAL_DVC_CHECKPOINT_OP_KEY
    assert node.category is NodeCategory.CHECKPOINT
    assert node.inputs == [] and node.outputs == []      # terminal source, no ports
    node.params["bundle_path"] = str(tmp_path / "exp.nd2dvc")
    node.params["provenance"] = {"source_file": "WellB1.nd2",
                                 "summary": {"n_frames": 3}}

    doc = PipelineDoc.empty()
    doc.slice_for(Stage.ANALYSIS).add_node(node)
    path = str(tmp_path / ("pipe" + ".nd2s_pipeline.json"))
    save_pipeline(path, doc)

    loaded = load_pipeline(path)
    got = [n for n in loaded.analysis.nodes.values()
           if n.op_key == SPECIAL_DVC_CHECKPOINT_OP_KEY]
    assert len(got) == 1
    n2 = got[0]
    assert n2.category is NodeCategory.CHECKPOINT
    assert n2.params.get("bundle_path", "").endswith(".nd2dvc")
    assert n2.params["provenance"]["source_file"] == "WellB1.nd2"


def test_dvc_node_has_output_port_and_connects_to_output_node():
    """V1.79: the DVC node now emits an ANY output port that connects (via the
    ANY-wildcard rule) to an Output node so its data can be auto-saved on Run."""
    from nd2studios.pipeline_graph import (
        PortType, SPECIAL_DVC_OP_KEY, analysis_output_spec, build_node,
        can_connect, special_specs,
    )
    dvc_spec = next(s for s in special_specs() if s.op_key == SPECIAL_DVC_OP_KEY)
    assert dvc_spec.output_types == [PortType.ANY]

    dvc = build_node(dvc_spec, pos=(0.0, 0.0))
    out = build_node(analysis_output_spec(), pos=(200.0, 0.0))
    dvc_out = [p for p in dvc.outputs if p.type is not PortType.CHANNEL]
    out_in = [p for p in out.inputs if p.type is not PortType.CHANNEL]
    assert len(dvc_out) == 1 and dvc_out[0].type is PortType.ANY
    assert len(out_in) == 1
    assert can_connect(dvc_out[0], out_in[0])   # ANY → BINARY (analysis output)


def test_output_folder_save_load(tmp_path):
    """Mimic the Output-node auto-save target: results/<node title>/<title>.nd2dvc
    (a title with a space and '#' must survive as a folder + file base)."""
    series = {0: {1: _make_result(3, 1), 2: _make_result(3, 2)}}
    out_dir = tmp_path / "results" / "Analysis #1"
    out_dir.mkdir(parents=True)
    path = str(out_dir / ("Analysis #1" + DVC_BUNDLE_EXTENSION))
    save_dvc_bundle(path, series_by_m=series, meta={"n_multipoints": 1},
                    provenance={"source_file": "x.nd2"})
    b = load_dvc_bundle(path)
    _assert_result_equal(series[0][1], b["series_by_m"][0][1])
    _assert_result_equal(series[0][2], b["series_by_m"][0][2])
    assert b["provenance"]["source_file"] == "x.nd2"


def test_not_a_bundle_raises(tmp_path):
    import pytest
    bad = tmp_path / "bad.npz"
    np.savez(str(bad), foo=np.zeros(3))          # a valid NPZ, but no manifest
    with pytest.raises(ValueError):
        load_dvc_bundle(str(bad))
