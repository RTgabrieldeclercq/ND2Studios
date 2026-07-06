"""Qt-free node-graph core for the Pipelines tab (V1.45).

Exposes the data model, the registry-driven node catalog, the recipe executor,
and JSON serialization. No PySide6 imports live in this package — the board UI
in ``widgets/node_board/`` and ``pages/pipelines_page.py`` own all Qt code.
"""
from __future__ import annotations

from nd2studios.pipeline_graph.conditions import (
    BLOCK_KINDS,
    Condition,
    ConditionBlock,
    GROUP_ROW,
    GROUP_TRACK,
    LENS_FRAME,
    LENS_OBJECT,
    block_param_schema,
    default_block_params,
    default_condition,
    describe_condition,
    evaluate_condition,
    families,
    make_block,
    partition_rows,
)
from nd2studios.pipeline_graph.executor import (
    CroppedVolume,
    GraphRunner,
    PinnedProcessedVolume,
    ProcessedFrameVolume,
    apply_recipe,
    evaluate_simple_condition,
    input_node,
    output_nodes,
    predecessor,
    recipe_for_node,
    recipe_hash,
    topological_order,
)
from nd2studios.pipeline_graph.io import (
    PIPELINE_EXTENSION,
    PIPELINE_KIND,
    PIPELINE_VERSION,
    load_pipeline,
    save_pipeline,
)
from nd2studios.pipeline_graph.model import (
    Bridge,
    Edge,
    GraphSlice,
    Node,
    NodeCategory,
    NodeRole,
    PipelineDoc,
    Port,
    PortType,
    ShapeKind,
    Stage,
    can_connect,
    category_for_stage,
    clone_node,
    new_id,
    would_create_cycle,
)
from nd2studios.pipeline_graph.registry_adapter import (
    IF_ELSE_OP_KEY,
    SPECIAL_CT_FIELDS_OP_KEY,
    SPECIAL_CT_METRICS_OP_KEY,
    SPECIAL_DISMISS_OP_KEY,
    SPECIAL_EXPORT_OP_KEY,
    SPECIAL_INTERP_MAP_OP_KEY,
    SPECIAL_PAUSE_OP_KEY,
    SPECIAL_REVIEW_OP_KEY,
    SPECIAL_SEND_RESULTS_OP_KEY,
    SPECIAL_TRACK_OP_KEY,
    SPECIAL_VALIDATE_OP_KEY,
    NodeSpec,
    analysis_input_spec,
    analysis_output_spec,
    analysis_pipeline_name_for_op_key,
    analysis_specs,
    build_node,
    default_params_for,
    enhancement_specs,
    if_else_spec,
    merged_action_specs,
    param_specs_for,
    plugin_name_for_op_key,
    processing_input_spec,
    processing_output_spec,
    results_input_spec,
    results_op_name_for_op_key,
    results_output_spec,
    results_specs,
    special_op_name_for_op_key,
    special_specs,
)

__all__ = [
    # model
    "Bridge", "Edge", "GraphSlice", "Node", "NodeCategory", "NodeRole",
    "PipelineDoc", "Port", "PortType", "ShapeKind", "Stage", "can_connect",
    "category_for_stage", "clone_node", "new_id", "would_create_cycle",
    # registry
    "NodeSpec", "build_node", "default_params_for", "enhancement_specs",
    "param_specs_for", "plugin_name_for_op_key", "processing_input_spec",
    "processing_output_spec", "analysis_specs", "analysis_input_spec",
    "analysis_output_spec", "analysis_pipeline_name_for_op_key",
    "results_specs", "results_input_spec", "results_output_spec",
    "results_op_name_for_op_key", "if_else_spec", "special_specs",
    "merged_action_specs", "special_op_name_for_op_key",
    "IF_ELSE_OP_KEY", "SPECIAL_DISMISS_OP_KEY", "SPECIAL_EXPORT_OP_KEY",
    "SPECIAL_PAUSE_OP_KEY", "SPECIAL_REVIEW_OP_KEY",
    "SPECIAL_SEND_RESULTS_OP_KEY", "SPECIAL_TRACK_OP_KEY",
    "SPECIAL_VALIDATE_OP_KEY", "SPECIAL_CT_METRICS_OP_KEY",
    "SPECIAL_CT_FIELDS_OP_KEY", "SPECIAL_INTERP_MAP_OP_KEY",
    # conditions (if-else DSL)
    "BLOCK_KINDS", "Condition", "ConditionBlock", "block_param_schema",
    "default_block_params", "default_condition", "describe_condition",
    "evaluate_condition", "families", "make_block", "partition_rows",
    "LENS_FRAME", "LENS_OBJECT", "GROUP_TRACK", "GROUP_ROW",
    # executor
    "CroppedVolume", "GraphRunner", "PinnedProcessedVolume", "ProcessedFrameVolume",
    "apply_recipe", "evaluate_simple_condition", "input_node", "output_nodes",
    "predecessor", "recipe_for_node", "recipe_hash", "topological_order",
    # io
    "PIPELINE_EXTENSION", "PIPELINE_KIND", "PIPELINE_VERSION",
    "load_pipeline", "save_pipeline",
]
