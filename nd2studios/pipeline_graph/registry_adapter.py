"""Turn the live plugin / pipeline registries into node specs (V1.45).

The Pipelines palette is **registry-driven**: it enumerates the same
registries the Recipe / Analysis / Results pages use, so it stays in sync
automatically when plugins are added. We never hardcode operation names —
they are read at call time from :class:`PluginBase` (and, in later phases,
:class:`AnalysisPipeline` and ``results_engine``).

This module is Qt-free; it only touches the pure registry in
``core/plugin_registry.py``. A :class:`NodeSpec` is the recipe the board UI
follows to build a node: which ports it has, its title, and its default
params.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from nd2studios.core.analysis_registry import AnalysisPipeline
from nd2studios.core.plugin_registry import ParamSpec, PluginBase
from nd2studios.pipeline_graph.model import (
    Node, NodeCategory, NodeRole, Port, PortType, ShapeKind, Stage,
    category_for_stage, new_id,
)


# op_key prefixes — the synthetic source/sink nodes and the operation ops.
INPUT_OP_KEY = "input:image"
OUTPUT_OP_KEY = "output:image"
ENHANCEMENT_PREFIX = "enhancement:"
# Analysis stage (V1.45.5): processed-image source, pipeline actions, result sink.
ANALYSIS_INPUT_OP_KEY = "input:processed"
ANALYSIS_OUTPUT_OP_KEY = "output:analysis"
ANALYSIS_PREFIX = "analysis:"
# Logic + Special nodes (V1.45 merge) — branch control flow + interactive actions.
LOGIC_PREFIX = "logic:"
SPECIAL_PREFIX = "special:"
IF_ELSE_OP_KEY = "logic:if_else"
SPECIAL_TRACK_OP_KEY = "special:track_objects"
# Retired in favour of SPECIAL_TRACK_OP_KEY (V1.45). Kept defined so old saved
# graphs migrate (io._migrate_v2_to_v3) and any stray node still dispatches.
SPECIAL_VALIDATE_OP_KEY = "special:validate_tracks"
SPECIAL_REVIEW_OP_KEY = "special:review_object"
SPECIAL_DISMISS_OP_KEY = "special:dismiss"
SPECIAL_EXPORT_OP_KEY = "special:export"
SPECIAL_SEND_RESULTS_OP_KEY = "special:send_to_results"
SPECIAL_PAUSE_OP_KEY = "special:pause"
# CellTracker-derived analyses (V1.45) — supportive nodes for cell-tracking
# pipelines. Metrics augments tracked rows with per-cell columns; Field Maps
# renders Eulerian gridded heatmaps to image files.
SPECIAL_CT_METRICS_OP_KEY = "special:ct_metrics"
SPECIAL_CT_FIELDS_OP_KEY = "special:ct_fields"
# Interpolated Spatial Maps (V1.45) — generic: linearly interpolate any measurement
# column at object centroids (with an optional per-track temporal fill) onto a grid.
# Categorised under Results alongside the migrated CellTracker Spatial Field Maps.
SPECIAL_INTERP_MAP_OP_KEY = "special:interp_map"
# Digital Volume Correlation (V1.51) — ALDVC displacement/strain field between a
# reference and a deformed timepoint of the wired channel. A Special node (like
# Track Objects) with an IMAGE input so it gets a rainbow channel port; it runs
# the DVC engine off-thread and opens the DVC viewer tab. Reads full (Z,H,W)
# volumes for true 3D DVC (2D DIC when there is no Z).
SPECIAL_DVC_OP_KEY = "special:dvc"
# Image registration (V1.56) — align a (T,H,W) series onto a reference frame
# (temporal drift correction, rigid/affine). Like DVC, a Special node with an
# IMAGE input (rainbow channel wiring): the transform is estimated on the wired
# reference channel and applied to all channels (register once, apply to all),
# then results (before/after + drift plot) open in the Registration viewer tab.
SPECIAL_REGISTER_OP_KEY = "special:register"
# Checkpoint (V1.53) — a white pass-through node that freezes everything computed
# upstream (analysis label masks, measurement rows, tracks) on a Run. A later Run
# whose upstream graph is unchanged resumes *from* the checkpoint with the frozen
# data restored, so the expensive upstream work (segmentation / tracking) never
# re-runs — the user iterates on the downstream pipeline instantly. Ports are
# wildcard (ANY) so it accepts any upstream and passes it through.
SPECIAL_CHECKPOINT_OP_KEY = "special:checkpoint"
# Channel source nodes (V1.48) — one per loaded channel + an "All" node, rendered
# under the Input node. Each emits a CHANNEL payload wired (rainbow port) into a
# process to say "run on this channel". ``channel:__all__`` emits every channel.
CHANNEL_PREFIX = "channel:"
CHANNEL_ALL_OP_KEY = "channel:__all__"
# Port names for the rainbow channel-flow ports appended to process nodes.
RAINBOW_IN_NAME = "ch_in"
RAINBOW_OUT_NAME = "ch_out"


@dataclass
class NodeSpec:
    """A blueprint for one node type in the palette."""

    op_key: str
    title: str
    description: str
    stage: Stage
    role: NodeRole
    input_types: List[PortType] = field(default_factory=list)
    output_types: List[PortType] = field(default_factory=list)
    # V1.45 merge: category drives node color + Add-dialog grouping; shape_kind
    # the silhouette; port names label the if-else's true/false outputs.
    category: NodeCategory = None  # None → derived from stage at build time
    shape_kind: ShapeKind = ShapeKind.RECT
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)

    def effective_category(self) -> NodeCategory:
        return self.category or category_for_stage(self.stage)


# ── Processing (enhancement plugins) ────────────────────────────────────────

def processing_input_spec() -> NodeSpec:
    """The source node for a Processing slice — the loaded channels."""
    return NodeSpec(
        op_key=INPUT_OP_KEY,
        title="Input",
        description="Loaded image channels from Import.",
        stage=Stage.PROCESSING,
        role=NodeRole.INPUT,
        input_types=[],
        output_types=[PortType.IMAGE],
    )


def processing_output_spec() -> NodeSpec:
    """The sink node for a Processing slice — a processed-image bridge."""
    return NodeSpec(
        op_key=OUTPUT_OP_KEY,
        title="Output",
        description="Processed channels — bridges to Analysis / Export.",
        stage=Stage.PROCESSING,
        role=NodeRole.OUTPUT,
        input_types=[PortType.IMAGE],
        output_types=[],
    )


# ── Channel source nodes (V1.48) ─────────────────────────────────────────────

def channel_source_spec(channel_name: str) -> NodeSpec:
    """A single-channel source node (rendered under the Input node).

    Emits a ``CHANNEL`` payload; wiring it into a process's rainbow port means
    that process (and everything downstream) runs on this channel.
    """
    return NodeSpec(
        op_key=f"{CHANNEL_PREFIX}{channel_name}",
        title=channel_name,
        description=f"Channel '{channel_name}' — wire into a process's rainbow "
                    "port to run that process on this channel.",
        stage=Stage.PROCESSING,          # stage is cosmetic for channel nodes
        role=NodeRole.ACTION,            # a source (no structural input)
        input_types=[],
        output_types=[PortType.CHANNEL],
        category=NodeCategory.CHANNEL,
        shape_kind=ShapeKind.PILL,
        output_names=["channel"],
    )


def channel_all_spec() -> NodeSpec:
    """The "All" channel source node — emits every loaded channel at once."""
    return NodeSpec(
        op_key=CHANNEL_ALL_OP_KEY,
        title="All",
        description="All channels — wire into a process's rainbow port to run "
                    "that process on every channel.",
        stage=Stage.PROCESSING,
        role=NodeRole.ACTION,
        input_types=[],
        output_types=[PortType.CHANNEL],
        category=NodeCategory.CHANNEL,
        shape_kind=ShapeKind.PILL,
        output_names=["channel"],
    )


def channel_name_for_op_key(op_key: str) -> str:
    """``"channel:GFP"`` -> ``"GFP"``; ``"channel:__all__"`` -> ``""`` (all)."""
    if op_key == CHANNEL_ALL_OP_KEY:
        return ""
    if op_key.startswith(CHANNEL_PREFIX):
        return op_key[len(CHANNEL_PREFIX):]
    return op_key


def is_channel_source_op(op_key: str) -> bool:
    return op_key.startswith(CHANNEL_PREFIX)


def spec_takes_channels(op_key: str, role: NodeRole, input_types: List[PortType]) -> bool:
    """True for a process node that operates on image channels (enhancement or
    analysis) — the nodes that get rainbow channel-flow ports (V1.48)."""
    return (role is NodeRole.ACTION
            and not is_channel_source_op(op_key)
            and PortType.IMAGE in list(input_types or []))


def enhancement_specs() -> List[NodeSpec]:
    """One :class:`NodeSpec` per registered enhancement plugin.

    Read live from ``PluginBase.get_plugins("enhancement")`` so the palette
    reflects whatever plugins are installed (19 built-ins as of V1.44, but the
    count is never hardcoded). Each is an ``Image -> Image`` action.
    """
    specs: List[NodeSpec] = []
    for cls in PluginBase.get_plugins("enhancement"):
        specs.append(
            NodeSpec(
                op_key=f"{ENHANCEMENT_PREFIX}{cls.name}",
                title=cls.name,
                description=getattr(cls, "description", ""),
                stage=Stage.PROCESSING,
                role=NodeRole.ACTION,
                input_types=[PortType.IMAGE],
                output_types=[PortType.IMAGE],
            )
        )
    specs.sort(key=lambda s: s.title.lower())
    return specs


def plugin_name_for_op_key(op_key: str) -> str:
    """``"enhancement:Gaussian Blur"`` -> ``"Gaussian Blur"``."""
    if op_key.startswith(ENHANCEMENT_PREFIX):
        return op_key[len(ENHANCEMENT_PREFIX):]
    return op_key


# ── Analysis (AnalysisPipeline registry) ────────────────────────────────────

def analysis_input_spec() -> NodeSpec:
    """The source node for an Analysis slice — the processed channels.

    Carries the processed-image bridge from Processing (``record.processed_view``).
    """
    return NodeSpec(
        op_key=ANALYSIS_INPUT_OP_KEY,
        title="Processed",
        description="Processed image channels bridged from Processing.",
        stage=Stage.ANALYSIS,
        role=NodeRole.INPUT,
        input_types=[],
        output_types=[PortType.IMAGE],
    )


def analysis_output_spec() -> NodeSpec:
    """The sink node for an Analysis slice — a result (label masks) bridge."""
    return NodeSpec(
        op_key=ANALYSIS_OUTPUT_OP_KEY,
        title="Output",
        description="Analysis result (label masks + measurements) — bridges to "
                    "Results / Export.",
        stage=Stage.ANALYSIS,
        role=NodeRole.OUTPUT,
        input_types=[PortType.BINARY],
        output_types=[],
    )


def analysis_specs() -> List[NodeSpec]:
    """One :class:`NodeSpec` per registered :class:`AnalysisPipeline`.

    Read live from ``AnalysisPipeline.get_pipelines()`` so the palette tracks
    whatever pipelines are installed (names never hardcoded). Each consumes an
    ``Image`` and produces a ``Binary`` (label masks); measurements ride along
    on the result the output bridge carries.
    """
    specs: List[NodeSpec] = []
    for cls in AnalysisPipeline.get_pipelines():
        specs.append(
            NodeSpec(
                op_key=f"{ANALYSIS_PREFIX}{cls.name}",
                title=cls.name,
                description=getattr(cls, "description", ""),
                stage=Stage.ANALYSIS,
                role=NodeRole.ACTION,
                input_types=[PortType.IMAGE],
                output_types=[PortType.BINARY],
            )
        )
    specs.sort(key=lambda s: s.title.lower())
    return specs


def analysis_pipeline_name_for_op_key(op_key: str) -> str:
    """``"analysis:Bright / Dark Spots"`` -> ``"Bright / Dark Spots"``."""
    if op_key.startswith(ANALYSIS_PREFIX):
        return op_key[len(ANALYSIS_PREFIX):]
    return op_key


# ── Results (results_engine ops) ────────────────────────────────────────────
# Results actions wrap measurement/summary computation over a committed
# AnalysisResult chosen by the ``analysis_result`` param. (Overlay-frame and
# label-mask TIFF *export* nodes are deferred to the Export integration.)
RESULTS_INPUT_OP_KEY = "input:results"
RESULTS_OUTPUT_OP_KEY = "output:results"
RESULTS_PREFIX = "results:"

# (name, description, input type, output type)
_RESULTS_OPS = [
    ("Compute Measurements",
     "Per-object measurements (area in px and µm², centroids, perimeter, "
     "eccentricity, solidity, bbox and per-channel intensity) for the chosen "
     "analysis result — the rows shown in the results table.",
     PortType.BINARY, PortType.DATA),
    ("Summary",
     "Aggregate statistics for the chosen analysis result: object count, "
     "frames with objects, and mean / std area.",
     PortType.BINARY, PortType.DATA),
]


def results_input_spec() -> NodeSpec:
    """The source node for a Results slice — the analysis results."""
    return NodeSpec(
        op_key=RESULTS_INPUT_OP_KEY,
        title="Analysis results",
        description="Label masks + measurements bridged from Analysis.",
        stage=Stage.RESULTS,
        role=NodeRole.INPUT,
        input_types=[],
        output_types=[PortType.BINARY],
    )


def results_output_spec() -> NodeSpec:
    """The sink node for a Results slice — exportable measurements."""
    return NodeSpec(
        op_key=RESULTS_OUTPUT_OP_KEY,
        title="Output",
        description="Exportable results (measurements table) — surfaced in the "
                    "Export source selector.",
        stage=Stage.RESULTS,
        role=NodeRole.OUTPUT,
        input_types=[PortType.DATA],
        output_types=[],
    )


def results_specs() -> List[NodeSpec]:
    """Result-stage action nodes wrapping ``backend/results_engine``."""
    return [
        NodeSpec(
            op_key=f"{RESULTS_PREFIX}{name}",
            title=name,
            description=desc,
            stage=Stage.RESULTS,
            role=NodeRole.ACTION,
            input_types=[in_t],
            output_types=[out_t],
        )
        for (name, desc, in_t, out_t) in _RESULTS_OPS
    ]


def results_op_name_for_op_key(op_key: str) -> str:
    """``"results:Compute Measurements"`` -> ``"Compute Measurements"``."""
    if op_key.startswith(RESULTS_PREFIX):
        return op_key[len(RESULTS_PREFIX):]
    return op_key


# ── Logic (if-else branch) + Special (interactive action) nodes ──────────────
# These live in the merged Analysis scene alongside analysis + results nodes.
# An if-else routes the whole downstream pipeline down its true or false branch
# (aggregate gate); special nodes drive interactive steps during a Run.

def if_else_spec() -> NodeSpec:
    """The purple branch node: one input (top), true + false outputs (bottom)."""
    return NodeSpec(
        op_key=IF_ELSE_OP_KEY,
        title="If / Else",
        description="Branch the pipeline: evaluate a condition over the upstream "
                    "result and route downstream to the True or False output. "
                    "Accepts any upstream node (image, masks or measurements).",
        stage=Stage.RESULTS,
        role=NodeRole.ACTION,
        # Wildcard ports so any upstream (analysis masks, measurements, …) can
        # feed the branch and either output can drive any downstream node.
        input_types=[PortType.ANY],
        output_types=[PortType.ANY, PortType.ANY],
        category=NodeCategory.LOGIC,
        shape_kind=ShapeKind.TRIANGLE,
        input_names=["in"],
        output_names=["true", "false"],
    )


# (op_key, title, description, shape, output_types, category) for the special
# actions. Ports are wildcard (PortType.ANY) so a special node accepts any upstream
# and passes it through; a sink ([]) has no output. ``category`` is optional and
# defaults to SPECIAL (orange); the two spatial-map sinks override it to RESULTS
# (green) so they group under the Results tab while keeping their special-op
# dispatch + custom Run/preview handlers.
_SPECIAL_OPS = [
    (SPECIAL_TRACK_OP_KEY, "Track Objects",
     "Link objects across frames into tracks (centroid tracking). Configure the "
     "distance / size-change / track-length / frame-gap thresholds that decide "
     "when an object is no longer the same track.",
     ShapeKind.HEXAGON, [PortType.ANY]),
    (SPECIAL_REVIEW_OP_KEY, "Review Objects",
     "Inspect identified objects and accept / reject them — either object by "
     "object (cropped panels) or by stepping through whole frames with label "
     "ids drawn on each object.",
     ShapeKind.HEXAGON, [PortType.ANY]),
    (SPECIAL_DISMISS_OP_KEY, "Dismiss",
     "Drop the selected object data / results from the stream.",
     ShapeKind.RECT, [PortType.ANY]),
    (SPECIAL_EXPORT_OP_KEY, "Export Objects / Frames",
     "Export objects, or the frames containing them, in a chosen image format.",
     ShapeKind.HEXAGON, []),
    (SPECIAL_SEND_RESULTS_OP_KEY, "Send to Results",
     "Hand the measurements to the Results page for plotting and further "
     "data processing.",
     ShapeKind.HEXAGON, []),
    (SPECIAL_PAUSE_OP_KEY, "Pause",
     "Halt the run and drop the whole page back to editor mode so you can "
     "modify the downstream pipeline before continuing.",
     ShapeKind.HEXAGON, [PortType.ANY]),
    (SPECIAL_CHECKPOINT_OP_KEY, "Checkpoint",
     "Freeze everything computed upstream (segmentation / analysis masks, "
     "measurements, tracks) when the run reaches this node. A later Run whose "
     "upstream graph is unchanged resumes from here with the frozen data "
     "restored — the expensive upstream work never re-runs, so you can iterate "
     "on the downstream pipeline instantly. Edit anything upstream and the "
     "checkpoint re-freezes automatically on the next full Run.",
     ShapeKind.HEXAGON, [PortType.ANY], NodeCategory.CHECKPOINT),
    (SPECIAL_CT_METRICS_OP_KEY, "Cell-Tracker Metrics",
     "Augment tracked objects with per-cell spatial metrics (neighbor distance, "
     "cell density, local divergence / curl), motion (speed + velocity_x/y) and "
     "self-fold-change (each cell's intensity vs. its own time-average). A "
     "downstream If / Else can branch on any of these. Place after a Track "
     "Objects node — the velocity / fold metrics need track ids.",
     ShapeKind.HEXAGON, [PortType.ANY]),
    (SPECIAL_CT_FIELDS_OP_KEY, "Cell-Tracker Spatial Maps",
     "Open the interactive Spatial Maps tab in the image viewer (Eulerian "
     "gridded heatmaps: density, velocity, divergence, curl, intensity, "
     "fold-change). Carries saved spatial-map template(s) that auto-load when "
     "the node runs. Velocity-derived fields need track ids from an upstream "
     "Track Objects node.",
     ShapeKind.HEXAGON, [PortType.ANY], NodeCategory.RESULTS),
    (SPECIAL_INTERP_MAP_OP_KEY, "Interpolated Spatial Maps",
     "Open the interactive Spatial Maps tab in the image viewer to interpolate "
     "any measurement column (intensity, area, speed, a custom metric …) at "
     "object centroids onto a grid. Carries saved spatial-map template(s) that "
     "auto-load when the node runs.",
     ShapeKind.HEXAGON, [PortType.ANY], NodeCategory.RESULTS),
    # 7th element overrides the input port type: DVC takes an IMAGE (rainbow
    # channel wiring), unlike the other specials which pass ANY through.
    (SPECIAL_DVC_OP_KEY, "DVC (ALDVC)",
     "Augmented-Lagrangian Digital Volume Correlation: measure the dense "
     "displacement + strain field between a reference and a deformed timepoint "
     "of the wired channel. Reads full Z-volumes for true 3D DVC (2D DIC when "
     "there is no Z). Set the reference / deformed frames + subset size in the "
     "node settings; results open in the DVC viewer tab.",
     ShapeKind.HEXAGON, [], NodeCategory.SPECIAL, [PortType.IMAGE]),
    # Registration (V1.56) — like DVC, takes an IMAGE (rainbow channel wiring):
    # estimate the drift/rigid transform on the wired reference channel, apply to
    # all channels. Unlike DVC (a terminal viewer), registration is a *transform*
    # in the pipeline: it emits an ANY output so it wires UPSTREAM of analysis /
    # tracking, and its correction is applied to every channel that downstream
    # nodes read (so analyses run on drift-free images).
    (SPECIAL_REGISTER_OP_KEY, "Registration",
     "Align a (T,H,W) series onto a reference frame (temporal drift correction; "
     "translation / rigid / affine). The transform is estimated on the wired "
     "reference channel and applied to every channel (register once, apply to all) "
     "so colocalization is preserved. Wire it upstream of analysis / tracking nodes "
     "— they then run on the drift-corrected image. Results (before/after + a "
     "drift-vs-time plot) open in the Registration viewer tab.",
     ShapeKind.HEXAGON, [PortType.ANY], NodeCategory.SPECIAL, [PortType.IMAGE]),
]


def special_specs() -> List[NodeSpec]:
    """The orange action nodes (Dismiss is a red rectangle; the spatial-map sinks
    are green Results nodes — same special-op dispatch, different category)."""
    specs: List[NodeSpec] = []
    for entry in _SPECIAL_OPS:
        op_key, title, desc, shape, out_types = entry[:5]
        category = entry[5] if len(entry) > 5 else NodeCategory.SPECIAL
        in_types = entry[6] if len(entry) > 6 else [PortType.ANY]
        specs.append(
            NodeSpec(
                op_key=op_key,
                title=title,
                description=desc,
                stage=Stage.RESULTS,
                role=NodeRole.ACTION,
                input_types=in_types,
                output_types=out_types,
                category=category,
                shape_kind=shape,
            )
        )
    return specs


def merged_action_specs() -> List[NodeSpec]:
    """All action specs for the merged Analysis sub-tab: analysis pipelines,
    results ops, the if-else logic node, and the special action nodes."""
    return analysis_specs() + results_specs() + [if_else_spec()] + special_specs()


def special_op_name_for_op_key(op_key: str) -> str:
    """``"special:export"`` -> ``"export"`` (for routing in the Run executor)."""
    if op_key.startswith(SPECIAL_PREFIX):
        return op_key[len(SPECIAL_PREFIX):]
    return op_key


# ── param specs / defaults ──────────────────────────────────────────────────

def param_specs_for(op_key: str) -> List[ParamSpec]:
    """Fresh :class:`ParamSpec` list for an action node's op.

    Returns an empty list for source/sink nodes and unknown ops. A new plugin
    instance is built each call so the page can mutate choice options (e.g.
    inject channel names) without leaking state across nodes — mirrors how
    ``analysis_page`` calls ``instance.get_params()`` on activation.
    """
    if op_key.startswith(ENHANCEMENT_PREFIX):
        cls = PluginBase.get_plugin("enhancement", plugin_name_for_op_key(op_key))
        if cls is not None:
            return cls().get_params()
    if op_key.startswith(ANALYSIS_PREFIX):
        pcls = AnalysisPipeline.get_pipeline(analysis_pipeline_name_for_op_key(op_key))
        if pcls is not None:
            # V1.48: the segmentation / counterstain channel is chosen by wiring a
            # channel node into the process's rainbow port, not by a param — so
            # hide those selectors from the node popup. The pipeline's run() still
            # reads params["channel_name"]; the page injects it per wired channel.
            return [s for s in pcls().get_params()
                    if s.name not in ("channel_name", "counterstain_channel")]
    if op_key.startswith(RESULTS_PREFIX):
        # Results ops aren't plugin-backed; their only knob is which committed
        # analysis result to operate on. Choices are injected at pop-up time.
        return [
            ParamSpec(
                name="analysis_result",
                label="Analysis result",
                param_type="choice",
                default="",
                choices=[],
                tooltip="Which committed analysis result to measure / overlay.",
            )
        ]
    if op_key == IF_ELSE_OP_KEY:
        # The condition tree itself is edited by the ConditionBuilderDialog (via
        # the popup's Edit button) and lives in node.params["condition"]. These
        # two flat params choose the *lens*: route the whole result (frame) or
        # split objects into the true/false branches (object lens).
        from nd2studios.pipeline_graph.conditions import (
            LENS_FRAME, LENS_OBJECT, GROUP_TRACK, GROUP_ROW,
        )
        return [
            ParamSpec(
                name="lens", label="Apply to", param_type="choice",
                default=LENS_FRAME, choices=[LENS_FRAME, LENS_OBJECT],
                tooltip="'Whole frame' routes the entire result down one branch "
                        "(the condition is an aggregate gate). 'Each object' "
                        "evaluates the condition per object and sends passing "
                        "objects to the TRUE branch, failing ones to FALSE — so a "
                        "downstream Dismiss drops only the failing objects and the "
                        "rest of the pipeline sees only the kept ones.",
            ),
            ParamSpec(
                name="group_by", label="Group objects by", param_type="choice",
                default=GROUP_TRACK, choices=[GROUP_TRACK, GROUP_ROW],
                visible_when={"lens": LENS_OBJECT},
                tooltip="'Per track' routes a whole tracked cell as a unit (e.g. "
                        "'any frame-frame vector > 20 µm' excludes the entire "
                        "cell); untracked objects fall back to per-row. 'Per row' "
                        "routes each object-frame independently.",
            ),
        ]
    if op_key == SPECIAL_CHECKPOINT_OP_KEY:
        # Checkpoint (V1.53): the only knob is whether the frozen data is kept in
        # session RAM (default) or also persisted to a companion cache on pipeline
        # save so it survives restarts. A per-node choice: disk persistence
        # materializes every per-M label mask, which can be large, so it's opt-in.
        return [
            ParamSpec(
                name="persist_to_disk", label="Persist frozen data to disk",
                param_type="bool", default=False,
                tooltip="Off: the frozen data lives only in this session's memory "
                        "(fastest, no disk writes). On: saving the pipeline also "
                        "writes this checkpoint's masks / rows / tracks to a "
                        "companion '<pipeline>.checkpoints/' cache, so a later "
                        "session can resume from it without re-running the upstream "
                        "work. Persisting materializes every per-multipoint label "
                        "mask, which can be large.",
            ),
        ]
    if op_key == SPECIAL_DVC_OP_KEY:
        # DVC node params: how to walk the timelapse (tracking mode + reference),
        # multipoint scope, Z-range / XY-downsample for huge stacks, + the ALDVC
        # engine knobs from ALDVCMethod.get_params() (the single source of truth).
        # The channel is chosen by wiring a channel pill into the rainbow port
        # (like analysis nodes), not by a param. DVC computes a field for **every**
        # frame (a playable series in the DVC tab), so there is no single
        # "deformed frame" — the deformed frames are all frames.
        from nd2studios.backend.dvc.method import ALDVCMethod
        specs = [
            ParamSpec(
                name="tracking_mode", label="Tracking mode", param_type="choice",
                default="cumulative", choices=["cumulative", "incremental"],
                tooltip="How ALDVC walks the timelapse. 'Cumulative' correlates "
                        "the fixed reference frame against every frame (total "
                        "deformation from reference). 'Incremental' correlates each "
                        "frame against the previous one (frame-to-frame change) — "
                        "more robust for large accumulating motion.",
            ),
            ParamSpec(
                name="ref_frame", label="Reference frame (T)", param_type="int",
                default=0, min_val=0, max_val=100000, step=1,
                visible_when={"tracking_mode": "cumulative"},
                tooltip="The fixed undeformed reference timepoint (cumulative mode).",
            ),
            ParamSpec(
                name="all_multipoints", label="All multipoints", param_type="bool",
                default=False,
                tooltip="Run the DVC series for every multipoint (switch between "
                        "them in the DVC tab). Off = only the currently viewed M.",
            ),
            ParamSpec(
                name="z_start", label="Z start", param_type="int",
                default=0, min_val=0, max_val=100000, step=1,
                tooltip="First Z-slice (0-based) of the sub-volume to correlate "
                        "(3D only). Limit Z to keep a deep stack tractable.",
            ),
            ParamSpec(
                name="z_end", label="Z end (0 = all)", param_type="int",
                default=0, min_val=0, max_val=100000, step=1,
                tooltip="Last Z-slice, exclusive (3D only). 0 = to the end.",
            ),
            ParamSpec(
                name="downsample", label="XY downsample", param_type="int",
                default=1, min_val=1, max_val=16, step=1,
                tooltip="Block-average the volume by this factor in X and Y before "
                        "correlating — essential to make large (e.g. 4k²) stacks "
                        "tractable. Displacements are reported in physical units "
                        "(the voxel size is scaled accordingly).",
            ),
        ]
        specs.extend(ALDVCMethod().get_params())
        return specs
    if op_key == SPECIAL_REGISTER_OP_KEY:
        # Registration node params: multipoint scope + apply-to-all toggle, then
        # the RigidRegistration engine knobs (model / reference / upsample / …),
        # which are the single source of truth. The reference channel is chosen by
        # wiring a channel pill into the rainbow port (like analysis/DVC nodes).
        from nd2studios.backend.registration.method import RigidRegistration
        specs = [
            ParamSpec(
                name="apply_to_all_channels", label="Apply to all channels",
                param_type="bool", default=True,
                tooltip="Estimate the transform on the wired reference channel and "
                        "apply the SAME transform to every channel (preserves "
                        "colocalization). Off = only the wired channel(s).",
            ),
            ParamSpec(
                name="all_multipoints", label="All multipoints", param_type="bool",
                default=False,
                tooltip="Register every multipoint (switch between them in the "
                        "Registration tab). Off = only the currently viewed M.",
            ),
            ParamSpec(
                name="crop_to_common", label="Crop to common region",
                param_type="bool", default=False,
                visible_when={"model": "translation"},
                tooltip="After registration, crop every frame (all multipoints) to "
                        "the largest rectangle that is real (non-padded) data in ALL "
                        "registered frames — so frames are equal-size, recentred, and "
                        "free of the black drift borders. Applies everywhere "
                        "downstream (analysis / viewer / export). Translation only.",
            ),
        ]
        specs.extend(RigidRegistration().get_params())
        # ROI (V1.60): estimate the transform on a chosen sub-region (a static
        # landmark), apply it full-frame — locks onto a stable structure when
        # moving cells / artifacts would corrupt whole-frame correlation. Captured
        # via the popup's "Pick ROI…" button (draws on the viewer); stored as a
        # serializable spec. Hidden so it survives the popup's full-params rewrite.
        specs.append(
            ParamSpec(name="roi", label="ROI", param_type="hidden", default=None,
                      tooltip="Region the transform is estimated on (whole frame if "
                              "unset). Set it with 'Pick ROI…'."))
        return specs
    if op_key == SPECIAL_REVIEW_OP_KEY:
        return [
            ParamSpec(
                name="mode", label="Review by", param_type="choice",
                default="Single objects",
                choices=["Single objects", "Whole frame"],
                tooltip="'Single objects' reviews one object (or one track, when "
                        "a Track Objects node is wired in) at a time in cropped "
                        "panels. 'Whole frame' steps through whole frames with "
                        "each object's label id drawn in white; click an object "
                        "to reject it, or accept / reject the whole frame.",
            ),
        ]
    if op_key == SPECIAL_TRACK_OP_KEY:
        # Tracking thresholds. ``method`` is a choice that selects the linker; the
        # rows below it use ``visible_when`` so only the knobs that apply to the
        # chosen method are shown. Names come from object_tracker so the node and
        # backend stay in sync.
        from nd2studios.backend.object_tracker import (
            METHOD_CENTROID, METHOD_SERIALTRACK,
            METHOD_CT_TOPOLOGY, METHOD_CT_FINGERPRINT, METHOD_CT_OVERLAP,
            TRACKING_METHODS,
        )
        return [
            ParamSpec(
                name="method", label="Tracking method", param_type="choice",
                default=METHOD_CENTROID, choices=list(TRACKING_METHODS),
                tooltip="How objects are linked across frames. 'Centroid' matches "
                        "each object to the nearest object in the next frame "
                        "(global assignment) within the limits below. "
                        "'SerialTrack' uses scale/rotation-invariant topology "
                        "matching — more robust when objects move or rotate a lot. "
                        "'Cell-Tracker: Topology' blends a rotation-invariant "
                        "neighbor descriptor with distance; 'Cell-Tracker: Spatial "
                        "Fingerprint' combines position + area and fills short "
                        "detection gaps. 'Cell-Tracker: Mask Overlap (IoU)' links "
                        "by how much each object's segmentation mask overlaps the "
                        "next frame — the most robust choice for dense, "
                        "slowly-moving nuclei (recommended for StarDist masks).",
            ),
            ParamSpec(
                name="max_distance", label="Max distance", param_type="float",
                default=100.0, min_val=0.0, max_val=100000.0, step=1.0,
                tooltip="Objects whose centroids move farther than this between "
                        "frames are no longer linked (a new track starts). For "
                        "SerialTrack this is the field-of-search radius.",
            ),
            ParamSpec(
                name="distance_unit", label="Distance unit", param_type="choice",
                default="pixels", choices=["pixels", "µm"],
                tooltip="Unit for the max distance. 'µm' uses the ND2 pixel size.",
            ),
            ParamSpec(
                name="max_size_diff", label="Max size change", param_type="float",
                default=1.0, min_val=0.0, max_val=1.0, step=0.05,
                visible_when={"method": METHOD_CENTROID},
                tooltip="Max fractional change in object area between frames "
                        "(|Δarea| / larger area). 1.0 disables the size gate; "
                        "0.3 links only when area changes by ≤30%.",
            ),
            ParamSpec(
                name="min_track_length", label="Min track length",
                param_type="int", default=2, min_val=1, max_val=100000, step=1,
                tooltip="Tracks spanning fewer than this many frames are "
                        "discarded (not assigned a track id).",
            ),
            ParamSpec(
                name="max_frame_gap", label="Max frame gap", param_type="int",
                default=0, min_val=0, max_val=1000, step=1,
                visible_when={"method": METHOD_CENTROID},
                tooltip="Allow a track to survive this many missed frames "
                        "(occlusion / missed detection) and re-link afterwards. "
                        "0 = must be re-detected in the next frame.",
            ),
            ParamSpec(
                name="st_mode", label="SerialTrack mode", param_type="choice",
                default="Incremental", choices=["Incremental", "Cumulative"],
                visible_when={"method": METHOD_SERIALTRACK},
                tooltip="'Incremental' links each frame to the previous one "
                        "(objects may appear / disappear). 'Cumulative' links "
                        "every frame back to the first — best for large total "
                        "motion when all objects are present from the start.",
            ),
            ParamSpec(
                name="st_n_neighbors", label="Max neighbors", param_type="int",
                default=25, min_val=2, max_val=100, step=1,
                visible_when={"method": METHOD_SERIALTRACK},
                tooltip="SerialTrack topology-descriptor size (number of nearest "
                        "neighbors, n_neighbors_max). Higher for dense seeding, "
                        "lower for sparse.",
            ),
            ParamSpec(
                name="st_n_neighbors_min", label="Min neighbors",
                param_type="int", default=1, min_val=1, max_val=100, step=1,
                visible_when={"method": METHOD_SERIALTRACK},
                tooltip="Floor for the exponential neighbor-count decay across "
                        "iterations (n_neighbors_min). At ≤2 the matcher becomes a "
                        "nearest-neighbor search. Keep at 1 unless you know better.",
            ),
            ParamSpec(
                name="st_solver", label="Global solver", param_type="choice",
                default="Regularization",
                choices=["MLS", "Regularization", "ADMM"],
                visible_when={"method": METHOD_SERIALTRACK},
                tooltip="Global displacement solver. 'MLS' is mesh-free and fast; "
                        "'Regularization' scatters to a grid and smooths (robust "
                        "default); 'ADMM' is the augmented-Lagrangian solver with "
                        "automatic L-curve α — most faithful to the paper and best "
                        "for noisy / large-deformation data, but slower.",
            ),
            ParamSpec(
                name="st_loc_solver", label="Local matcher", param_type="choice",
                default="Topology",
                choices=["Topology", "Histogram then Topology"],
                visible_when={"method": METHOD_SERIALTRACK},
                tooltip="Local matching strategy. 'Topology' uses the "
                        "scale/rotation-invariant descriptor directly; 'Histogram "
                        "then Topology' pre-matches by a displacement histogram "
                        "first.",
            ),
            ParamSpec(
                name="st_smoothness", label="Smoothness", param_type="float",
                default=0.1, min_val=0.0, max_val=100.0, step=0.05,
                visible_when={"method": METHOD_SERIALTRACK},
                tooltip="Global smoothing strength (the α/µ knob; used by the "
                        "Regularization and ADMM solvers). Higher = smoother, more "
                        "noise rejection, less local detail. Paper range 1e-3…1e-1.",
            ),
            ParamSpec(
                name="st_outlier_threshold", label="Outlier threshold",
                param_type="float", default=5.0, min_val=0.0, max_val=100.0,
                step=0.5,
                visible_when={"method": METHOD_SERIALTRACK},
                tooltip="Westerweel normalized-median-residual cutoff for rejecting "
                        "spurious displacement vectors (typ. 2–5). 0 disables it.",
            ),
            ParamSpec(
                name="st_max_iter", label="Max iterations", param_type="int",
                default=20, min_val=1, max_val=1000, step=1,
                visible_when={"method": METHOD_SERIALTRACK},
                tooltip="Maximum ADMM iterations per frame pair. 20 is usually "
                        "plenty; raise for very large deformations.",
            ),
            ParamSpec(
                name="st_iter_stop_threshold", label="Convergence tol.",
                param_type="float", default=0.01, min_val=0.0, max_val=10.0,
                step=0.001,
                visible_when={"method": METHOD_SERIALTRACK},
                tooltip="ADMM convergence threshold on the displacement-update "
                        "norm. Smaller = tighter convergence, more iterations.",
            ),
            ParamSpec(
                name="st_dist_missing", label="Ghost-cull distance",
                param_type="float", default=5.0, min_val=0.0, max_val=100000.0,
                step=0.5,
                visible_when={"method": METHOD_SERIALTRACK},
                tooltip="Critical distance ε_d (px) for culling ghost particles "
                        "(detected in only one frame), active in late iterations.",
            ),
            ParamSpec(
                name="st_use_prev_results", label="Use previous results",
                param_type="bool", default=False,
                visible_when={"method": METHOD_SERIALTRACK},
                tooltip="Warm-start each frame's solve with a data-driven initial "
                        "guess (extrapolation, then POD-GPR from frame 7). Helps "
                        "very large deformation; the POD-GPR stage needs "
                        "scikit-learn (pip install scikit-learn).",
            ),
            ParamSpec(
                name="ct_n_neighbors", label="Topology neighbors",
                param_type="int", default=5, min_val=1, max_val=100, step=1,
                visible_when={"method": METHOD_CT_TOPOLOGY},
                tooltip="Number of nearest neighbors in Cell-Tracker's "
                        "rotation-invariant topology descriptor.",
            ),
            ParamSpec(
                name="ct_topo_weight", label="Topology weight", param_type="float",
                default=0.3, min_val=0.0, max_val=1.0, step=0.05,
                visible_when={"method": METHOD_CT_TOPOLOGY},
                tooltip="Blend of topology cost vs. raw distance (0 = pure "
                        "distance, 1 = pure topology). Raise it when motion is "
                        "coherent and neighborhoods stay stable.",
            ),
            ParamSpec(
                name="ct_area_weight", label="Area weight", param_type="float",
                default=0.3, min_val=0.0, max_val=1.0, step=0.05,
                visible_when={"method": METHOD_CT_FINGERPRINT},
                tooltip="Weight of area similarity vs. distance in the fingerprint "
                        "cost (0 = pure distance). Raise when size is a stable cue.",
            ),
            ParamSpec(
                name="ct_max_gap", label="Max frame gap", param_type="int",
                default=3, min_val=0, max_val=1000, step=1,
                visible_when={"method": [METHOD_CT_FINGERPRINT, METHOD_CT_OVERLAP]},
                tooltip="Frames a track may vanish (missed detection) and still "
                        "re-link afterwards (fingerprint and mask-overlap linkers). "
                        "For overlap this keeps the last mask as the target while "
                        "a detection is briefly missing.",
            ),
            ParamSpec(
                name="ct_min_iou", label="Min overlap (IoU)", param_type="float",
                default=0.1, min_val=0.0, max_val=1.0, step=0.05,
                visible_when={"method": METHOD_CT_OVERLAP},
                tooltip="Minimum mask intersection-over-union (0–1) to link an "
                        "object to the next frame. Objects overlapping less than "
                        "this start a new track instead of being force-linked. "
                        "Lower it (e.g. 0.05) if masks jitter or the field drifts; "
                        "run drift correction first for large frame-to-frame shifts.",
            ),
        ]
    if op_key == SPECIAL_CT_METRICS_OP_KEY:
        # Per-cell spatial metrics + self-fold-change. ``intensity_channel``
        # choices are injected at pop-up time (live channel names); leaving it
        # empty computes neighbor / divergence / curl but skips self-fold-change.
        return [
            ParamSpec(
                name="n_neighbors", label="Neighbors", param_type="int",
                default=6, min_val=1, max_val=100, step=1,
                tooltip="Number of nearest neighbors used for neighbor distance "
                        "and local divergence / curl. Larger = smoother, more "
                        "global.",
            ),
            ParamSpec(
                name="intensity_channel", label="Intensity channel",
                param_type="choice", default="", choices=[],
                tooltip="Channel whose mean intensity drives self-fold-change "
                        "(each cell vs. its own time-average). Leave empty to skip "
                        "self-fold-change.",
            ),
        ]
    if op_key in (SPECIAL_CT_FIELDS_OP_KEY, SPECIAL_INTERP_MAP_OP_KEY):
        # Both spatial-map nodes are now interactive: they open the Spatial Maps
        # tab in the viewer. Their only stored state is a list of saved template
        # names (edited via the popup's "Spatial map templates…" button →
        # SpatialTemplatePicker); the sidebar controls live in the tab itself.
        return [
            ParamSpec(
                name="templates", label="Templates", param_type="hidden",
                default=[],
                tooltip="Saved Spatial Maps configuration(s) auto-loaded into the "
                        "tab when this node runs.",
            ),
        ]
    if op_key == SPECIAL_EXPORT_OP_KEY:
        # Frame-organization choices come from results_engine (lazy import so the
        # Qt-free graph layer stays light at module load).
        try:
            from nd2studios.backend.results_engine import (
                EXPORT_ORG_DEFAULT, EXPORT_ORG_MAP,
            )
            org_choices = list(EXPORT_ORG_MAP.keys())
            org_default = EXPORT_ORG_DEFAULT
        except Exception:  # noqa: BLE001
            org_choices = ["T folds into M (one file per M)"]
            org_default = org_choices[0]
        return [
            ParamSpec(
                name="content", label="Export", param_type="choice",
                default="objects",
                choices=["objects", "frames_with_objects"],
                tooltip="Export label masks (objects), or whole frames that "
                        "contain objects (image + overlay).",
            ),
            ParamSpec(
                name="image_format", label="Image format", param_type="choice",
                default="TIFF", choices=["TIFF", "PNG", "JPEG"],
                tooltip="Output image format. TIFF stores multi-page stacks; "
                        "PNG/JPG write a folder of frames when an axis folds.",
            ),
            ParamSpec(
                name="frame_organization", label="Frame organization",
                param_type="choice", default=org_default, choices=org_choices,
                tooltip="Which axes (M / T / Z) stack inside a file vs split into "
                        "separate files. e.g. 'T folds into M' = one file per "
                        "multipoint containing all its timepoints.",
            ),
            ParamSpec(
                name="include_overlay", label="Burn in overlay",
                param_type="bool", default=False,
                tooltip="When exporting objects, also burn the label-mask overlay "
                        "onto the image instead of writing plain label masks.",
            ),
        ]
    return []


def default_params_for(op_key: str) -> Dict[str, Any]:
    """Default param dict for an op, derived from its :class:`ParamSpec`s.

    The if-else node carries a nested ``condition`` block tree (not flat params),
    seeded with a default condition.
    """
    if op_key == IF_ELSE_OP_KEY:
        from nd2studios.pipeline_graph.conditions import default_condition
        out = {spec.name: spec.default for spec in param_specs_for(op_key)}
        out["condition"] = default_condition().to_dict()
        return out
    out: Dict[str, Any] = {}
    for spec in param_specs_for(op_key):
        out[spec.name] = spec.default
    return out


def build_node(
    spec: NodeSpec,
    pos: tuple = (0.0, 0.0),
    bridge_id: str | None = None,
) -> Node:
    """Instantiate a model :class:`Node` from a :class:`NodeSpec`.

    Ports are created from the spec's type signature; params are seeded with
    registry defaults. Kept here (not in the board UI) so node construction
    stays in the Qt-free layer.
    """
    def _name(names: List[str], i: int, prefix: str) -> str:
        return names[i] if i < len(names) else f"{prefix}{i}"

    inputs = [
        Port(id=new_id("p"), name=_name(spec.input_names, i, "in"),
             type=t, is_input=True)
        for i, t in enumerate(spec.input_types)
    ]
    outputs = [
        Port(id=new_id("p"), name=_name(spec.output_names, i, "out"),
             type=t, is_input=False)
        for i, t in enumerate(spec.output_types)
    ]
    # V1.48: process nodes (enhancement / analysis) get rainbow channel-flow
    # ports — one free CHANNEL input (more spawn as channels are wired) on the
    # left edge, and a CHANNEL output on the right that re-emits the node's
    # channel set so channels can be threaded on downstream ("both sides").
    if spec_takes_channels(spec.op_key, spec.role, spec.input_types):
        inputs.append(Port(id=new_id("p"), name=RAINBOW_IN_NAME,
                           type=PortType.CHANNEL, is_input=True))
        outputs.append(Port(id=new_id("p"), name=RAINBOW_OUT_NAME,
                            type=PortType.CHANNEL, is_input=False))
    return Node(
        id=new_id("node"),
        stage=spec.stage,
        role=spec.role,
        op_key=spec.op_key,
        title=spec.title,
        params=default_params_for(spec.op_key),
        inputs=inputs,
        outputs=outputs,
        pos=(float(pos[0]), float(pos[1])),
        bridge_id=bridge_id,
        category=spec.effective_category(),
        shape_kind=spec.shape_kind,
    )
