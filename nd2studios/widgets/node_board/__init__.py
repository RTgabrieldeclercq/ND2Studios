"""Custom ``QGraphicsScene`` node editor for the Pipelines tab (V1.45).

Built on Qt Graphics View per the project's dependency-conservatism rule (no
NodeGraphQt). The Qt-free graph model lives in ``nd2studios/pipeline_graph/``;
this package only renders + edits it.
"""
from __future__ import annotations

from nd2studios.widgets.node_board.add_node_dialog import AddNodeDialog
from nd2studios.widgets.node_board.condition_builder_dialog import (
    ConditionBuilderDialog,
)
from nd2studios.widgets.node_board.edge_item import EdgeItem
from nd2studios.widgets.node_board.measurement_select_dialog import (
    MeasurementSelectDialog,
)
from nd2studios.widgets.node_board.node_item import NodeItem
from nd2studios.widgets.node_board.node_scene import NodeScene
from nd2studios.widgets.node_board.param_popup import ParamPopup
from nd2studios.widgets.node_board.port_item import PortItem, port_color

__all__ = [
    "AddNodeDialog", "ConditionBuilderDialog", "EdgeItem",
    "MeasurementSelectDialog", "NodeItem", "NodeScene", "ParamPopup",
    "PortItem", "port_color",
]
