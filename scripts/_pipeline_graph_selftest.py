"""Headless self-test for the Qt-free pipeline_graph core (V1.45).

Run: ``python scripts/_pipeline_graph_selftest.py`` from the repo root.
Asserts the graph model, type-checked connect, cycle rejection, fan-out,
recipe linearization, recipe execution, and JSON round-trip — none of which
should require PySide6.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

# Ensure repo root on path when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Populate the enhancement registry.
import nd2studios.plugins.enhancement  # noqa: F401,E402

from nd2studios.pipeline_graph import (  # noqa: E402
    Edge, GraphRunner, Node, NodeCategory, NodeRole, PipelineDoc, Port,
    PortType, ShapeKind, Stage, apply_recipe, build_node, can_connect,
    enhancement_specs, evaluate_simple_condition, if_else_spec, load_pipeline,
    merged_action_specs, new_id, recipe_for_node, save_pipeline, special_specs,
    topological_order, would_create_cycle,
)
from nd2studios.pipeline_graph.conditions import (  # noqa: E402
    Condition, ConditionBlock, describe_condition, evaluate_condition,
    make_block,
)
from nd2studios.backend.object_tracker import link_objects  # noqa: E402
from nd2studios.pipeline_graph.io import (  # noqa: E402
    _migrate_v1_to_v2, _migrate_v2_to_v3,
)
from nd2studios.pipeline_graph.registry_adapter import (  # noqa: E402
    INPUT_OP_KEY, OUTPUT_OP_KEY, SPECIAL_DISMISS_OP_KEY, SPECIAL_REVIEW_OP_KEY,
    SPECIAL_TRACK_OP_KEY, SPECIAL_VALIDATE_OP_KEY, analysis_input_spec,
    default_params_for, param_specs_for, results_input_spec, results_specs,
)


def _image_port(is_input: bool, name: str = "img") -> Port:
    return Port(id=new_id("p"), name=name, type=PortType.IMAGE, is_input=is_input)


def _make_input(sl) -> Node:
    n = Node(id=new_id("in"), stage=Stage.PROCESSING, role=NodeRole.INPUT,
             op_key=INPUT_OP_KEY, title="Input",
             outputs=[_image_port(False, "channels")])
    return sl.add_node(n)


def _make_action(sl, plugin_name: str, params: dict) -> Node:
    n = Node(id=new_id("act"), stage=Stage.PROCESSING, role=NodeRole.ACTION,
             op_key=f"enhancement:{plugin_name}", title=plugin_name, params=params,
             inputs=[_image_port(True)], outputs=[_image_port(False)])
    return sl.add_node(n)


def _make_output(sl, title: str) -> Node:
    n = Node(id=new_id("out"), stage=Stage.PROCESSING, role=NodeRole.OUTPUT,
             op_key=OUTPUT_OP_KEY, title=title, inputs=[_image_port(True)])
    return sl.add_node(n)


def _wire(sl, src: Node, dst: Node) -> Edge:
    e = Edge(id=new_id("e"), src_node=src.id, src_port=src.output_port().id,
             dst_node=dst.id, dst_port=dst.input_port().id)
    return sl.add_edge(e)


def main() -> int:
    # 1. Registry-driven palette.
    specs = enhancement_specs()
    names = {s.title for s in specs}
    assert len(specs) >= 15, f"expected many enhancement specs, got {len(specs)}"
    assert "Gaussian Blur" in names, names
    print(f"[ok] enhancement_specs(): {len(specs)} ops")

    # 2. Type-checked connect.
    out_p = _image_port(False)
    in_p = _image_port(True)
    bin_in = Port(id=new_id("p"), name="mask", type=PortType.BINARY, is_input=True)
    assert can_connect(out_p, in_p) is True
    assert can_connect(out_p, bin_in) is False, "type mismatch must reject"
    assert can_connect(in_p, out_p) is False, "input->output must reject"
    print("[ok] can_connect type-checking")

    # 3. Build a branching DAG: input -> blur -> out1 ; input -> median -> out2.
    doc = PipelineDoc.empty()
    sl = doc.processing
    inp = _make_input(sl)
    blur = _make_action(sl, "Gaussian Blur", {"sigma": 2.0})
    med = _make_action(sl, "Median Filter", {})
    out1 = _make_output(sl, "Processing #1")
    out2 = _make_output(sl, "Processing #2")
    _wire(sl, inp, blur)
    _wire(sl, blur, out1)
    _wire(sl, inp, med)          # fan-out from the input's single output port
    _wire(sl, med, out2)
    assert len(sl.edges) == 4
    print("[ok] branching DAG with fan-out built")

    # 4. Cycle rejection.
    assert would_create_cycle(sl, blur.id, inp.id) is True, "back-edge is a cycle"
    assert would_create_cycle(sl, inp.id, out1.id) is False
    print("[ok] would_create_cycle")

    # 5. Recipe linearization per output (each branch is its own recipe).
    r1 = recipe_for_node(sl, out1.id)
    r2 = recipe_for_node(sl, out2.id)
    assert r1 == [("Gaussian Blur", {"sigma": 2.0})], r1
    assert r2 == [("Median Filter", {})], r2
    print(f"[ok] recipe_for_node: out1={r1} out2={r2}")

    # 6. Execute a recipe on a synthetic (T, H, W) stack.
    channels = {"DAPI": (np.random.rand(3, 32, 32) * 255).astype(np.uint16)}
    res = apply_recipe(channels, r1, normalized=False)
    assert "DAPI" in res and res["DAPI"].shape[0] == 3, res["DAPI"].shape
    # Single-frame preview path.
    res1 = apply_recipe(channels, r1, normalized=False, frame=1)
    assert res1["DAPI"].shape[0] == 1, res1["DAPI"].shape
    print(f"[ok] apply_recipe full={res['DAPI'].shape} frame={res1['DAPI'].shape}")

    # 7. JSON round-trip.
    path = os.path.join(tempfile.gettempdir(), "selftest.nd2s_pipeline.json")
    save_pipeline(path, doc, name="selftest")
    doc2 = load_pipeline(path)
    assert doc2.to_dict() == doc.to_dict(), "round-trip mismatch"
    os.remove(path)
    print("[ok] save_pipeline / load_pipeline round-trip")

    # 8. remove_node clears its edges.
    sl.remove_node(med.id)
    assert med.id not in sl.nodes
    assert all(med.id not in (e.src_node, e.dst_node) for e in sl.edges.values())
    print("[ok] remove_node prunes edges")

    # 9. Merge (V1.45): new node specs carry category + shape.
    ie = build_node(if_else_spec())
    assert ie.category is NodeCategory.LOGIC and ie.shape_kind is ShapeKind.TRIANGLE
    assert [p.name for p in ie.outputs] == ["true", "false"], ie.outputs
    sp = {s.op_key: s for s in special_specs()}
    dismiss = build_node(sp[SPECIAL_DISMISS_OP_KEY])
    assert dismiss.shape_kind is ShapeKind.RECT and dismiss.category is NodeCategory.SPECIAL
    assert len(merged_action_specs()) >= 9
    # Track Objects (was "Validate Tracked Objects") is in the palette with its
    # tracking params: shared (method / distance / length) + centroid-only
    # (size / gap) + SerialTrack-only (mode / neighbors / global+local solver /
    # smoothness / outlier + ghost-cull / ADMM iteration budget / warm start) +
    # Cell-Tracker topology (ct_n_neighbors / ct_topo_weight) and fingerprint
    # (ct_area_weight / ct_max_gap). The param popup shows the method-relevant
    # subset via each spec's visible_when.
    assert SPECIAL_TRACK_OP_KEY in sp and sp[SPECIAL_TRACK_OP_KEY].title == "Track Objects"
    assert not any(s.title == "Validate Tracked Objects" for s in special_specs())
    track = build_node(sp[SPECIAL_TRACK_OP_KEY])
    assert set(track.params) == {
        "method", "max_distance", "distance_unit", "max_size_diff",
        "min_track_length", "max_frame_gap",
        "st_mode", "st_n_neighbors", "st_n_neighbors_min", "st_solver",
        "st_loc_solver", "st_smoothness", "st_outlier_threshold", "st_max_iter",
        "st_iter_stop_threshold", "st_dist_missing", "st_use_prev_results",
        "ct_n_neighbors", "ct_topo_weight", "ct_area_weight", "ct_max_gap"}, track.params
    st_specs = {s.name: s for s in param_specs_for(SPECIAL_TRACK_OP_KEY)}
    method_choices = st_specs["method"].choices
    assert "SerialTrack (topology PTV)" in method_choices
    assert "Cell-Tracker: Topology (Hungarian)" in method_choices
    assert "Cell-Tracker: Spatial Fingerprint" in method_choices
    # The ADMM global solver is routed in as a SerialTrack option.
    assert set(st_specs["st_solver"].choices) == {"MLS", "Regularization", "ADMM"}
    # CellTracker supportive special nodes: Metrics augments rows; Field Maps
    # renders Eulerian heatmaps. Both carry flat params (no custom editor).
    from nd2studios.pipeline_graph.registry_adapter import (
        SPECIAL_CT_METRICS_OP_KEY, SPECIAL_CT_FIELDS_OP_KEY,
    )
    assert SPECIAL_CT_METRICS_OP_KEY in sp and SPECIAL_CT_FIELDS_OP_KEY in sp
    ctm = build_node(sp[SPECIAL_CT_METRICS_OP_KEY])
    assert set(ctm.params) == {"n_neighbors", "intensity_channel"}, ctm.params
    ctf = build_node(sp[SPECIAL_CT_FIELDS_OP_KEY])
    assert "field" in ctf.params and "grid_step" in ctf.params, ctf.params
    # Review Objects (was "Review Object") carries a 'mode' choice.
    assert sp[SPECIAL_REVIEW_OP_KEY].title == "Review Objects"
    review = build_node(sp[SPECIAL_REVIEW_OP_KEY])
    assert review.params.get("mode") == "Single objects", review.params
    print(f"[ok] merged specs: if-else triangle/logic, {len(special_specs())} specials")

    # 10. GraphRunner branch pruning + topo order on a merged slice.
    mdoc = PipelineDoc.empty()
    msl = mdoc.merged
    minp = build_node(analysis_input_spec()); minp.id = "minp"; msl.add_node(minp)
    mif = build_node(if_else_spec()); mif.id = "mif"; msl.add_node(mif)
    vtrue = build_node(sp[SPECIAL_TRACK_OP_KEY])
    vtrue.id = "vtrue"; msl.add_node(vtrue)
    vfalse = build_node(sp[SPECIAL_DISMISS_OP_KEY]); vfalse.id = "vfalse"
    msl.add_node(vfalse)
    msl.add_edge(Edge(new_id("e"), "minp", minp.outputs[0].id, "mif", mif.inputs[0].id))
    tport = next(p for p in mif.outputs if p.name == "true")
    fport = next(p for p in mif.outputs if p.name == "false")
    msl.add_edge(Edge(new_id("e"), "mif", tport.id, "vtrue", vtrue.inputs[0].id))
    msl.add_edge(Edge(new_id("e"), "mif", fport.id, "vfalse", vfalse.inputs[0].id))
    assert topological_order(msl)[:2] == ["minp", "mif"], topological_order(msl)
    runner = GraphRunner(msl)
    assert runner.reachable_nodes() == {"minp", "mif", "vtrue", "vfalse"}
    seq = []
    while True:
        nid = runner.next_ready()
        if nid is None:
            break
        seq.append(nid)
        runner.complete(nid, prune_ports={fport.id} if nid == "mif" else None)
    assert seq == ["minp", "mif", "vtrue"], seq  # false branch pruned
    assert runner.is_finished()
    print(f"[ok] GraphRunner true-branch walk: {seq}")

    # 11. Condition evaluator over measurement rows.
    rows = [{"area_um2": 5.0}, {"area_um2": 40.0}, {"area_um2": 12.0}]
    assert evaluate_simple_condition(
        {"metric": "object_count", "comparator": ">", "value": 2}, rows) is True
    assert evaluate_simple_condition(
        {"metric": "area_um2", "aggregate": "any", "comparator": ">", "value": 30}, rows) is True
    assert evaluate_simple_condition(
        {"metric": "area_um2", "aggregate": "all", "comparator": ">", "value": 30}, rows) is False
    print("[ok] evaluate_simple_condition")

    # 12. io v2 round-trip + v1→v2 migration folds results into the merged slice.
    path2 = os.path.join(tempfile.gettempdir(), "selftest_merged.nd2s_pipeline.json")
    save_pipeline(path2, mdoc, name="merged")
    mdoc2 = load_pipeline(path2)
    assert mdoc2.to_dict() == mdoc.to_dict(), "merged round-trip mismatch"
    os.remove(path2)
    old = PipelineDoc.empty()
    rin = build_node(results_input_spec()); rin.id = "rin"; old.results.add_node(rin)
    ract = build_node(results_specs()[0]); ract.id = "ract"; old.results.add_node(ract)
    old.results.add_edge(Edge(new_id("e"), "rin", rin.outputs[0].id, "ract", ract.inputs[0].id))
    mig = _migrate_v1_to_v2(old)
    assert "ract" in mig.analysis.nodes and "rin" not in mig.analysis.nodes
    assert len(mig.results.nodes) == 0
    print("[ok] io v2 round-trip + v1->v2 results-merge migration")

    # 12b. io v2->v3: a legacy "Validate Tracked Objects" node becomes "Track
    # Objects" with the new tracking params.
    legacy = PipelineDoc.empty()
    vn = build_node(sp[SPECIAL_TRACK_OP_KEY]); vn.id = "vn"
    vn.op_key = SPECIAL_VALIDATE_OP_KEY        # pretend it's an old saved node
    vn.title = "Validate Tracked Objects"; vn.params = {}
    legacy.analysis.add_node(vn)
    rv = build_node(sp[SPECIAL_REVIEW_OP_KEY]); rv.id = "rv"
    rv.title = "Review Object"; rv.params = {}     # pretend it's an old saved node
    legacy.analysis.add_node(rv)
    mig3 = _migrate_v2_to_v3(legacy)
    assert mig3.analysis.nodes["vn"].op_key == SPECIAL_TRACK_OP_KEY
    assert mig3.analysis.nodes["vn"].title == "Track Objects"
    assert mig3.analysis.nodes["vn"].params == default_params_for(SPECIAL_TRACK_OP_KEY)
    assert mig3.analysis.nodes["rv"].title == "Review Objects"
    assert mig3.analysis.nodes["rv"].params.get("mode") == "Single objects"
    print("[ok] io v2->v3 migration: Validate->Track, Review Object->Review Objects")

    # 13. Condition model (Phase 2): build, round-trip, evaluate, describe.
    cond = Condition("ALL", [
        make_block("object_count"),
        make_block("metric"),
        ConditionBlock("empty_field", {}, negate=True),
    ])
    cond.blocks[0].params.update({"aggregate": "total", "comparator": ">", "value": 2})
    cond.blocks[1].params.update({"metric": "area_um2", "aggregate": "any",
                                  "comparator": ">", "value": 30})
    assert Condition.from_dict(cond.to_dict()).to_dict() == cond.to_dict()
    crows = [{"frame": 0, "segmentation_channel": "c", "label_id": 1, "area_um2": 10.0},
             {"frame": 0, "segmentation_channel": "c", "label_id": 2, "area_um2": 40.0},
             {"frame": 0, "segmentation_channel": "c", "label_id": 3, "area_um2": 5.0}]
    assert evaluate_condition(cond, crows) is True       # 3>2, one area>30, not empty
    assert evaluate_condition(cond, []) is False         # count fails + empty negated
    # timelapse over tracked rows
    trows = [{"frame": 0, "track_id": 1, "track_length": 3},
             {"frame": 1, "track_id": 1, "track_length": 3},
             {"frame": 0, "track_id": 2, "track_length": 2}]
    tc = Condition("ALL", [ConditionBlock("track_count", {"comparator": ">=", "value": 2})])
    assert evaluate_condition(tc, trows) is True
    assert isinstance(describe_condition(cond), str) and describe_condition(cond)
    print(f"[ok] conditions: AND/OR/NOT + families; '{describe_condition(cond)}'")

    # 13b. partition_rows — object-lens split (per-track vs per-row).
    from nd2studios.pipeline_graph.conditions import (
        partition_rows, GROUP_TRACK, GROUP_ROW)
    prows = [  # track 1 has one fast frame (speed 50); track 2 is slow throughout
        {"m_position": 0, "track_id": 1, "frame": 0, "speed": 0.0},
        {"m_position": 0, "track_id": 1, "frame": 1, "speed": 50.0},
        {"m_position": 0, "track_id": 2, "frame": 0, "speed": 0.0},
        {"m_position": 0, "track_id": 2, "frame": 1, "speed": 3.0},
    ]
    keepc = Condition("ALL", [ConditionBlock(
        "metric", {"metric": "speed", "aggregate": "all", "comparator": "<=", "value": 20.0})])
    keep, drop = partition_rows(keepc, prows, group_by=GROUP_TRACK)
    assert {r["track_id"] for r in keep} == {2} and {r["track_id"] for r in drop} == {1}
    assert len(drop) == 2  # the whole fast track routes together
    keep_r, drop_r = partition_rows(keepc, prows, group_by=GROUP_ROW)
    assert len(drop_r) == 1 and drop_r[0]["speed"] == 50.0  # only the fast frame
    assert partition_rows(Condition("ALL", []), prows) == (list(prows), [])  # no blocks → all pass
    print("[ok] partition_rows: per-track routes whole cell, per-row routes each frame")

    # 14. object_tracker: frame-to-frame centroid linking + thresholds.
    def _det(frame, cy, cx, area, ch="c", m=0):
        return {"segmentation_channel": ch, "m_position": m, "frame": frame,
                "centroid_y_px": cy, "centroid_x_px": cx, "area_px": area}

    # One object drifting 10 px/frame links into a single 3-frame track.
    moving = [_det(0, 10, 10, 100), _det(1, 10, 20, 100), _det(2, 10, 30, 100)]
    link_objects(moving, max_displacement_px=15.0, min_track_length=2)
    assert len({r["track_id"] for r in moving}) == 1, moving
    assert moving[0]["track_length"] == 3

    # The reference centroid must *update* frame-to-frame: a 25 px total drift
    # over 2 steps links only because each step (12.5 px) is under threshold.
    drift = [_det(0, 0, 0, 100), _det(1, 0, 12, 100), _det(2, 0, 25, 100)]
    link_objects(drift, max_displacement_px=15.0, min_track_length=2)
    assert len({r["track_id"] for r in drift}) == 1, drift

    # Distance gate: a 40 px jump exceeds the 15 px threshold -> two tracks.
    jump = [_det(0, 0, 0, 100), _det(1, 0, 40, 100)]
    link_objects(jump, max_displacement_px=15.0, min_track_length=1)
    assert jump[0]["track_id"] != jump[1]["track_id"], jump

    # Size gate: same place, area doubles (size_diff 0.5) -> blocked at 0.3.
    grow = [_det(0, 0, 0, 100), _det(1, 0, 1, 200)]
    link_objects(grow, max_displacement_px=15.0, min_track_length=1,
                 max_size_diff_frac=0.3)
    assert grow[0]["track_id"] != grow[1]["track_id"], grow
    # ...but links when the size gate is relaxed.
    grow2 = [_det(0, 0, 0, 100), _det(1, 0, 1, 200)]
    link_objects(grow2, max_displacement_px=15.0, min_track_length=1,
                 max_size_diff_frac=0.9)
    assert grow2[0]["track_id"] == grow2[1]["track_id"], grow2

    # Frame gap: a detection missing on frame 1 re-links on frame 2 only when a
    # gap is allowed.
    gap = [_det(0, 0, 0, 100), _det(2, 0, 5, 100)]
    link_objects(gap, max_displacement_px=15.0, min_track_length=1, max_frame_gap=0)
    assert gap[0]["track_id"] != gap[1]["track_id"], gap
    gap2 = [_det(0, 0, 0, 100), _det(2, 0, 5, 100)]
    link_objects(gap2, max_displacement_px=15.0, min_track_length=2, max_frame_gap=1)
    assert gap2[0]["track_id"] == gap2[1]["track_id"], gap2
    print("[ok] object_tracker: centroid linking + distance/size/gap thresholds")

    print("\nALL PIPELINE_GRAPH SELF-TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
