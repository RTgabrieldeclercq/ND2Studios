"""V1.75 Exclude node — registry / node-construction / serialization (Qt-free).

Exercises the pure ``nd2studios.pipeline_graph`` layer: the ``special:exclude``
spec is registered, exposes exactly the ``dilate_px`` param, builds a node with
``ANY`` in/out and (crucially) **no** rainbow CHANNEL ports, and round-trips
through ``Node.to_dict``/``from_dict``.
"""
from __future__ import annotations

from nd2studios.pipeline_graph import (
    Node,
    NodeCategory,
    PortType,
    ShapeKind,
    SPECIAL_EXCLUDE_OP_KEY,
    build_node,
    default_params_for,
    param_specs_for,
    special_specs,
)


def _exclude_spec():
    specs = [s for s in special_specs() if s.op_key == SPECIAL_EXCLUDE_OP_KEY]
    assert len(specs) == 1, "Exclude must appear exactly once in special_specs()"
    return specs[0]


def test_exclude_spec_registered():
    spec = _exclude_spec()
    assert spec.title == "Exclude"
    assert spec.input_types == [PortType.ANY]
    assert spec.output_types == [PortType.ANY]           # pass-through
    assert spec.effective_category() == NodeCategory.SPECIAL
    assert spec.shape_kind == ShapeKind.HEXAGON


def test_exclude_params():
    names = [s.name for s in param_specs_for(SPECIAL_EXCLUDE_OP_KEY)]
    assert names == ["dilate_px"]
    assert default_params_for(SPECIAL_EXCLUDE_OP_KEY) == {"dilate_px": 0}


def test_exclude_build_node_has_no_channel_ports():
    node = build_node(_exclude_spec())
    assert node.op_key == SPECIAL_EXCLUDE_OP_KEY
    assert [p.type for p in node.inputs] == [PortType.ANY]
    assert [p.type for p in node.outputs] == [PortType.ANY]
    # ANY input (not IMAGE) → it consumes a region, not a channel → no rainbow ports.
    assert all(p.type is not PortType.CHANNEL
               for p in list(node.inputs) + list(node.outputs))


def test_exclude_node_round_trip():
    node = build_node(_exclude_spec())
    node.params["dilate_px"] = 5
    restored = Node.from_dict(node.to_dict())
    assert restored.op_key == SPECIAL_EXCLUDE_OP_KEY
    assert restored.params.get("dilate_px") == 5
    assert [p.type for p in restored.inputs] == [PortType.ANY]
    assert [p.type for p in restored.outputs] == [PortType.ANY]
