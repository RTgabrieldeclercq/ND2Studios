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
from nd2studios.pipeline_graph.granule_ops import (
    SPECIAL_BEAD_DETECT_OP_KEY,
    SPECIAL_GRANULE_BOUNDARY_OP_KEY,
    SPECIAL_GRANULE_CLUSTER_OP_KEY,
    SPECIAL_GRANULE_MASK_OP_KEY,
    SPECIAL_GRANULE_TESSELLATE_OP_KEY,
)
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
# Save Data (V1.71) — a pass-through node that writes the dataset *as it is at
# this point in the pipeline* (recipe + registration + any upstream crop applied)
# to disk (TIFF hyperstack / NPZ) when a Run reaches it, then passes the stream
# through unchanged so it can sit mid-pipeline.
SPECIAL_SAVE_DATA_OP_KEY = "special:save_data"
# Crop (V1.71, manual mode only) — a pass-through node that applies a user-picked
# rectangular crop to every downstream node. Like the Registration "crop to
# common region", it publishes the rect (record._pipeline_crop) where the page's
# _crop_rect() composition reads it, so the whole downstream pipeline sees the
# cropped image with no per-node changes.
SPECIAL_CROP_OP_KEY = "special:crop"
# Exclude (V1.75) — a pass-through node that marks a wired region (a 3D Mask
# Drawing / Granule Volume Mask object) as "ignore": every downstream analysis
# node skips the voxels inside it, across all Z / M / T. The inverse of the V1.68
# Frame/Object scope lever (which crops analysis TO an object). Like the Crop
# node it publishes a record side-artifact (record._exclude_by_m, a per-M (Z,H,W)
# boolean volume) that the page's shared image-read chokepoints zero out, so the
# whole downstream pipeline honours it with no per-node changes.
SPECIAL_EXCLUDE_OP_KEY = "special:exclude"
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
# 3D Mask Drawing (V1.65) — like DVC/Registration, takes an IMAGE (rainbow channel
# wiring): the user draws/edits a 3D object mask over the wired channel's raw
# (Z,H,W) volume (per-Z shapes, propagate-across-Z, or threshold-seed + edit). It
# emits a BINARY output so the drawn mask can feed analysis / DVC-render nodes. The
# drawn shapes live in node.params['mask_shapes'] (hidden); Run rasterizes them into
# a (Z,H,W) boolean volume per (m,t). Phase 1 of the DVC-on-object feature.
SPECIAL_MASK3D_OP_KEY = "special:mask3d"
# 2D Digital Image Correlation (V1.77) — pyALDIC (the optional 'al-dic' package):
# the 2D sibling of the DVC node. Like DVC, a Special node with an IMAGE input
# (rainbow channel wiring) that reads the wired channel's frames, correlates them
# with the al-dic AL-DIC solver off-thread and opens its own "DIC" viewer tab
# (a 2D DVCResult series rendered by the reused DVCPanel). Terminal (no output).
SPECIAL_DIC_OP_KEY = "special:dic"
# DIC Mesh Region (V1.77) — reproduces pyALDIC's ROI toolbar: draw the mesh domain
# (rectangle / polygon / circle, Add/Cut, Refine Brush, Invert/Clear). A pass-through
# (IMAGE in → ANY out) wired UPSTREAM of the DIC node; on Run it rasterizes the drawn
# shapes into a boolean ROI mask published to record._dic_roi_by_m, which the DIC job
# uses as the AL-DIC mesh domain. Shapes live in node.params['roi_shapes'] (hidden).
SPECIAL_DIC_ROI_OP_KEY = "special:dic_roi"
# DIC Mesh Refinement (V1.77) — pyALDIC's adaptive-quadtree workflow: paint a brush
# region + pick refinement criteria (mask boundary / ROI edge / brush). A pass-through
# wired UPSTREAM of the DIC node; on Run it publishes record._dic_refine_by_m, which
# the DIC job turns into an al-dic RefinementPolicy. Brush lives in
# node.params['brush_shapes'] (hidden).
SPECIAL_DIC_REFINE_OP_KEY = "special:dic_refine"
# Prism (V1.77) — a channel splicer / overlay converger, rendered as a 2.5D faceted
# gem (``ShapeKind.GEM``). Like DVC/Registration/Mask3D it takes an IMAGE (so it grows
# rainbow channel ports) and emits ANY (so it wires UPSTREAM of DVC / analysis, e.g.
# input → … → Prism → DVC). Fed a channel via its rainbow port, it can **add / remove /
# replace** the analysis channels flowing downstream (``channel_op`` param), and — on a
# **view-only** (dotted) outgoing edge — **converge that channel into another node as a
# viewer-only overlay** (never into analysis). Its flagship use converges a granule
# boundary channel into DVC as a volumetric overlay clipped to the shell between the
# volume-mask boundary and the boundary-extraction boundary.
SPECIAL_PRISM_OP_KEY = "special:prism"
# Checkpoint (V1.53) — a white pass-through node that freezes everything computed
# upstream (analysis label masks, measurement rows, tracks) on a Run. A later Run
# whose upstream graph is unchanged resumes *from* the checkpoint with the frozen
# data restored, so the expensive upstream work (segmentation / tracking) never
# re-runs — the user iterates on the downstream pipeline instantly. Ports are
# wildcard (ANY) so it accepts any upstream and passes it through.
SPECIAL_CHECKPOINT_OP_KEY = "special:checkpoint"
# DVC Checkpoint (V1.76) — a portable, self-contained *input* node created only by the
# DVC viewer's Import flow (NOT offered in the Add dialog — an empty one is
# meaningless). It carries a reloaded ``.nd2dvc`` bundle (every multipoint's DVCResult
# series, backdrops, masks and per-granule bundles) via ``params["bundle_path"]`` plus
# a read-only ``params["provenance"]`` record (how the field was produced). On import /
# pipeline-load it re-publishes the saved field to the DVC viewer with NO Run. Modelled
# on the white Checkpoint node (``NodeCategory.CHECKPOINT``); DVC is terminal, so it has
# no input and no output ports (a pure source that drives the viewer).
SPECIAL_DVC_CHECKPOINT_OP_KEY = "special:dvc_checkpoint"
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


def op_produces_objects(op_key: str) -> bool:
    """True if a node with this ``op_key`` yields **objects** (masks / labels).

    'Objects' = something :func:`nd2studios.backend.analysis.object_scope.iter_objects`
    can enumerate for per-object scoping (V1.68): the 3D Mask Drawing node
    (``special:mask3d`` — a drawn ``(Z,H,W)`` mask, output ``PortType.ANY``), the
    Track Objects node (``special:track_objects``), or any analysis node (output
    ``PortType.BINARY`` = ``AnalysisResult.label_masks``). Gating on the op_key
    (not the port type) is deliberate: mask3d emits ``ANY`` so it would pair with a
    DVC ``IMAGE`` input, and a pure BINARY-port check would miss it.
    """
    return (op_key == SPECIAL_MASK3D_OP_KEY
            or op_key == SPECIAL_TRACK_OP_KEY
            or op_key == SPECIAL_GRANULE_MASK_OP_KEY  # V1.70 per-granule (Z,H,W) bool masks
            or op_key.startswith(ANALYSIS_PREFIX))


def node_produces_objects(node: Any) -> bool:
    """:func:`op_produces_objects` for a model ``Node`` (drives the scope lever)."""
    return op_produces_objects(getattr(node, "op_key", ""))


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
    # Save Data (V1.71) — a pass-through (ANY→ANY) so it can sit mid-pipeline: it
    # writes the dataset as it is at this point (recipe + registration + upstream
    # crop applied) to disk and passes the stream through unchanged.
    (SPECIAL_SAVE_DATA_OP_KEY, "Save Data",
     "Write the dataset exactly as it is at this point in the pipeline "
     "(enhancement recipe + registration + any upstream crop applied) to disk — "
     "a multi-channel TIFF hyperstack or an NPZ, one file per multipoint. A "
     "pass-through: place it anywhere and the run continues unchanged.",
     ShapeKind.HEXAGON, [PortType.ANY]),
    # Crop (V1.71, manual) — a pass-through (ANY→ANY) so it wires UPSTREAM of
    # analysis / tracking / export; its rect is published where _crop_rect() picks
    # it up, cropping every downstream node.
    (SPECIAL_CROP_OP_KEY, "Crop",
     "Crop the image to a rectangle you pick, and apply that crop to the rest of "
     "the pipeline — every downstream node (analysis, tracking, export, Save "
     "Data) and the viewer read the cropped image. Manual mode only for now: set "
     "the region with 'Pick crop region…' in the node settings. Pixel size is "
     "preserved; only width/height shrink.",
     ShapeKind.HEXAGON, [PortType.ANY]),
    # Exclude (V1.75) — a pass-through (ANY→ANY): wire a region-producing node
    # (3D Mask Drawing / Granule Volume Mask) INTO it and every downstream analysis
    # ignores the voxels inside that region. Its exclusion volume is published where
    # the page's analysis / DVC image reads zero it out — the inverse of the crop.
    (SPECIAL_EXCLUDE_OP_KEY, "Exclude",
     "Ignore a region during analysis. Wire an object / mask / region node (e.g. "
     "3D Mask Drawing or Granule Volume Mask) into this node and every downstream "
     "analysis node (segmentation, spots, DVC, measurements, spatial maps) skips "
     "the voxels inside that region — across all Z, all multipoints and all "
     "timepoints. Use it to blank out artefacts, dead cells or air bubbles so they "
     "never enter the analysis. The image itself is untouched (Save Data / Export "
     "still write the full frame); only what analysis 'sees' is masked.",
     ShapeKind.HEXAGON, [PortType.ANY]),
    (SPECIAL_CHECKPOINT_OP_KEY, "Checkpoint",
     "Freeze everything computed upstream (segmentation / analysis masks, "
     "measurements, tracks) when the run reaches this node. A later Run whose "
     "upstream graph is unchanged resumes from here with the frozen data "
     "restored — the expensive upstream work never re-runs, so you can iterate "
     "on the downstream pipeline instantly. Edit anything upstream and the "
     "checkpoint re-freezes automatically on the next full Run.",
     ShapeKind.HEXAGON, [PortType.ANY], NodeCategory.CHECKPOINT),
    (SPECIAL_DVC_CHECKPOINT_OP_KEY, "DVC Checkpoint",
     "A reloaded DVC result bundle (.nd2dvc file). Created by the DVC viewer's "
     "Export → Import flow (not from the Add menu): it re-publishes every saved "
     "multipoint / frame / object displacement + strain field to the DVC viewer with "
     "no Run, and records how the field was produced (provenance). A portable "
     "checkpoint you can open in a fresh session with no ND2 file loaded. Terminal "
     "(no ports) — it drives the DVC viewer, like the DVC node.",
     ShapeKind.HEXAGON, [], NodeCategory.CHECKPOINT, []),
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
     "node settings; results open in the DVC viewer tab. Wire its output into an "
     "Output node to auto-save every field (the .nd2dvc bundle) to a folder named "
     "after that Output node on Run.",
     ShapeKind.HEXAGON, [PortType.ANY], NodeCategory.SPECIAL, [PortType.IMAGE]),
    # 2D DIC (V1.77) — pyALDIC. Like DVC, takes an IMAGE (rainbow channel wiring)
    # and is terminal (opens its own DIC viewer tab). Reads 2D frames of the wired
    # channel and correlates them with the optional 'al-dic' AL-DIC solver.
    (SPECIAL_DIC_OP_KEY, "DIC (pyALDIC)",
     "2D Augmented-Lagrangian Digital Image Correlation (pyALDIC): measure the "
     "dense in-plane displacement + strain field across the timelapse of the wired "
     "channel. The 2D sibling of the DVC node — hybrid local IC-GN + global ADMM "
     "over an adaptive mesh. Wire a DIC Mesh Region node in front to restrict the "
     "mesh to a drawn ROI, and a DIC Mesh Refinement node to refine it. Needs the "
     "optional 'al-dic' package (pip install al-dic); results open in the DIC "
     "viewer tab.",
     ShapeKind.HEXAGON, [], NodeCategory.SPECIAL, [PortType.IMAGE]),
    # DIC Mesh Region (V1.77) — pyALDIC's ROI toolbar as a node: draw the mesh
    # domain (rectangle / polygon / circle, Add/Cut, Refine Brush, Invert/Clear).
    # A pass-through (IMAGE in → ANY out) so it wires input → DIC Mesh Region → DIC;
    # on Run its shapes rasterize to record._dic_roi_by_m (the AL-DIC mesh domain).
    (SPECIAL_DIC_ROI_OP_KEY, "DIC Mesh Region",
     "Draw the region of interest / mesh domain for a downstream DIC node, exactly "
     "like pyALDIC's ROI tools: add or cut rectangles, polygons and circles, paint "
     "a freehand brush, and invert / clear. Wire it in front of a DIC node "
     "(input → DIC Mesh Region → DIC) — on Run the drawn shapes are rasterized to a "
     "boolean ROI mask that becomes the AL-DIC finite-element mesh domain (the "
     "correlation runs only inside it). Edit it with 'Draw mesh region…'.",
     ShapeKind.HEXAGON, [PortType.ANY], NodeCategory.SPECIAL, [PortType.IMAGE]),
    # DIC Mesh Refinement (V1.77) — pyALDIC's adaptive-quadtree workflow as a node:
    # paint a brush region + pick refinement criteria. A pass-through wired UPSTREAM
    # of the DIC node; on Run it publishes record._dic_refine_by_m (an al-dic
    # RefinementPolicy spec) so the DIC mesh refines where it matters.
    (SPECIAL_DIC_REFINE_OP_KEY, "DIC Mesh Refinement",
     "Drive pyALDIC's adaptive quadtree mesh refinement for a downstream DIC node. "
     "Paint a 'brush' region to resolve finer displacement where you need it, and "
     "toggle refinement criteria (mask boundary, ROI edge, brush region). Wire it "
     "in front of a DIC node (… → DIC Mesh Refinement → DIC) — on Run it builds an "
     "AL-DIC refinement policy so the mesh subdivides in the chosen regions. Edit "
     "it with 'Draw refinement brush…'.",
     ShapeKind.HEXAGON, [PortType.ANY], NodeCategory.SPECIAL, [PortType.IMAGE]),
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
    # 3D Mask Drawing (V1.65) — like Registration, a pass-through in the image
    # stream: takes an IMAGE (rainbow channel wiring) so the user draws over the
    # wired channel's raw (Z,H,W) volume, and emits ANY so it wires UPSTREAM of DVC /
    # analysis (e.g. input → 3D Mask Drawing → DVC), which both fixes connectivity
    # (an ANY output pairs with DVC's IMAGE input, unlike a BINARY one) and orders
    # the run so the mask is published to record._mask3d_by_m *before* the DVC node's
    # 3-D Object view reads it. The drawn mask is a side artifact on the record, not
    # a payload threaded through the wire. Edited via the popup's "Draw 3D mask…"
    # button; rasterized to a (Z,H,W) volume on Run.
    (SPECIAL_MASK3D_OP_KEY, "3D Mask Drawing",
     "Draw and edit a 3D object mask across the Z-stack of the wired channel, then "
     "wire it in front of a DVC node (input → 3D Mask Drawing → DVC) so the DVC "
     "viewer's '3D Object' tab can render the field on the object. Draw a rectangle "
     "/ ellipse / polygon on each Z plane (Manual), draw a few planes and fill the "
     "rest by copying or smoothly interpolating between them (Propagate across Z), "
     "or seed each plane's outline from an intensity threshold and hand-correct it "
     "(Threshold seed + edit). On Run the shapes are rasterized into a (Z,H,W) mask "
     "volume the DVC object render consumes.",
     ShapeKind.HEXAGON, [PortType.ANY], NodeCategory.SPECIAL, [PortType.IMAGE]),
    # Prism (V1.77) — a 2.5D faceted gem that splices into any analysis chain. It
    # takes an IMAGE (rainbow channel wiring) and emits ANY (so it wires upstream of
    # DVC / analysis). Fed a channel via its rainbow port, it ADDs / REMOVEs / REPLACEs
    # the analysis channels flowing downstream, and on a VIEW-ONLY (dotted) outgoing
    # edge converges that channel into a downstream node as a viewer-only overlay.
    (SPECIAL_PRISM_OP_KEY, "Prism",
     "Load in, remove, or converge a channel at this point in the pipeline. Wire a "
     "channel (a rainbow channel-source pill) into the Prism, then wire the Prism into "
     "any analysis node. On a normal (solid) edge it ADDs / REMOVEs / REPLACEs the "
     "channels that node analyses (choose with 'Channel operation'). On a VIEW-ONLY "
     "edge — click the wire to make it dotted — the channel is fed to the viewers only, "
     "never to analysis, so it can't confuse the analysis pipeline. Its flagship use: "
     "wire Granule Boundary Extraction → Prism (green) → DVC on a dotted edge so DVC "
     "keeps correlating its own channel while the green channel is drawn in DVC's 3-D "
     "object overlay, clipped to the shell between the volume-mask boundary and the "
     "boundary-extraction boundary.",
     ShapeKind.GEM, [PortType.ANY], NodeCategory.SPECIAL, [PortType.IMAGE]),
    # ── Granule Separation (V1.70) ──────────────────────────────────────────
    # A five-node chain that separates a 3-D point cloud of bead centroids into the
    # hydrogel granules they belong to. Bead Detection takes an IMAGE (rainbow
    # channel wiring, like DVC/Mask3D) and emits a DATA point cloud; the middle
    # nodes flow DATA; the Volume Mask emits ANY (like Mask3D) so it wires into DVC
    # / a Boundary node. Artifacts live on record._granule_*_by_m, not on the wire.
    (SPECIAL_BEAD_DETECT_OP_KEY, "Bead Detection",
     "Detect bead / particle centroids in the wired channel's raw (Z,H,W) volume "
     "and emit them as a 3-D point cloud (measurement rows carrying centroid_z/y/x). "
     "First stage of granule separation — feed it into a Granule Clustering node.",
     ShapeKind.HEXAGON, [PortType.DATA], NodeCategory.SPECIAL, [PortType.IMAGE]),
    (SPECIAL_GRANULE_CLUSTER_OP_KEY, "Granule Clustering",
     "Assign each bead in the point cloud to a hydrogel granule with a Gaussian "
     "mixture model, choosing the granule count by a BIC sweep over your seeded "
     "estimate relaxed by a percentage. Wire a Bead Detection node in; feed the "
     "labelled points into a Granule Tessellation node.",
     ShapeKind.HEXAGON, [PortType.DATA], NodeCategory.SPECIAL, [PortType.DATA]),
    (SPECIAL_GRANULE_TESSELLATE_OP_KEY, "Granule Tessellation",
     "Turn each granule's beads into a boundary (per-granule alpha-shape or global "
     "Voronoi) and merge neighbouring regions of similar point density into one "
     "granule. Produces the final granule boundaries for the Granule Volume Mask.",
     ShapeKind.HEXAGON, [PortType.ANY], NodeCategory.SPECIAL, [PortType.DATA]),
    (SPECIAL_GRANULE_MASK_OP_KEY, "Granule Volume Mask",
     "Voxelize the tessellated granule boundaries onto the confocal grid (with a "
     "smoothing parameter) into per-granule (Z,H,W) masks plus a combined label "
     "volume. Object-producing: wire it into DVC and flip the edge to Objects to "
     "run one DVC field per granule, or into a Granule Boundary node.",
     ShapeKind.HEXAGON, [PortType.ANY], NodeCategory.SPECIAL, [PortType.ANY]),
    (SPECIAL_GRANULE_BOUNDARY_OP_KEY, "Granule Boundary Extraction",
     "Extract an outward boundary band (N voxels, by dilation or Euclidean "
     "distance transform) around each granule mask — reaching into background and "
     "neighbouring granules — to recover the data surrounding each granule.",
     ShapeKind.HEXAGON, [PortType.ANY], NodeCategory.SPECIAL, [PortType.ANY]),
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


def dvc_checkpoint_spec() -> NodeSpec:
    """The portable DVC Checkpoint input node spec (V1.76).

    Created only by the DVC viewer's Import flow — the page passes this to
    ``NodeScene.add_node_from_spec`` directly. It is kept **out** of the Add dialog
    (``PipelinesPage`` filters it from ``scene.action_specs``) but stays in
    :func:`special_specs` so its white ``CHECKPOINT`` category round-trips through
    ``save_pipeline`` / ``load_pipeline`` (``io.py`` re-derives category from
    ``special_specs`` on load)."""
    for s in special_specs():
        if s.op_key == SPECIAL_DVC_CHECKPOINT_OP_KEY:
            return s
    raise KeyError(SPECIAL_DVC_CHECKPOINT_OP_KEY)


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
    if op_key == SPECIAL_DIC_OP_KEY:
        # 2D DIC node params (V1.78): how to walk the timelapse + XY downsample,
        # then the pyALDIC engine knobs from PyALDICMethod.get_params() (the single
        # source of truth). No Z knobs — DIC is 2D. The channel comes from a wired
        # channel pill (rainbow port), the ROI/refinement from upstream DIC Mesh
        # nodes; both are not params here.
        from nd2studios.backend.dic.method import PyALDICMethod
        specs = [
            ParamSpec(
                name="tracking_mode", label="Tracking mode", param_type="choice",
                default="cumulative", choices=["cumulative", "incremental"],
                tooltip="How AL-DIC walks the timelapse. 'Cumulative' correlates "
                        "the fixed reference frame against every frame (total "
                        "deformation from reference → accumulative mode). "
                        "'Incremental' correlates each frame against the previous "
                        "one — more robust for large accumulating motion.",
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
                tooltip="Run the DIC series for every multipoint (switch between "
                        "them in the DIC tab). Off = only the currently viewed M.",
            ),
            ParamSpec(
                name="downsample", label="XY downsample", param_type="int",
                default=1, min_val=1, max_val=16, step=1,
                tooltip="Block-average the frame by this factor in X and Y before "
                        "correlating — for very large frames. Displacements are "
                        "reported in physical units (the pixel size is scaled).",
            ),
        ]
        specs.extend(PyALDICMethod().get_params())
        return specs
    if op_key == SPECIAL_DIC_ROI_OP_KEY:
        # DIC Mesh Region node (V1.78): a live mesh-grid preview pitch for the editor
        # + the hidden drawn ROI shapes (keyed {str(m): [shape,...]}). The shapes are
        # edited via the popup's "Draw mesh region…" button and rasterized on Run.
        return [
            ParamSpec(
                name="mesh_preview_step", label="Mesh preview pitch (px)",
                param_type="int", default=16, min_val=2, max_val=128, step=2,
                tooltip="Grid spacing used only for the live mesh-dot preview in the "
                        "drawing editor. The actual mesh pitch is the downstream DIC "
                        "node's 'Subset spacing / step'.",
            ),
            ParamSpec(name="roi_shapes", label="ROI shapes",
                      param_type="hidden", default={}),
        ]
    if op_key == SPECIAL_DIC_REFINE_OP_KEY:
        # DIC Mesh Refinement node (V1.78): the adaptive-quadtree criteria toggles +
        # the hidden brush shapes (keyed {str(m): [shape,...]}), edited via the
        # popup's "Draw refinement brush…" button and turned into an al-dic
        # RefinementPolicy on Run.
        return [
            ParamSpec(
                name="refine_mask_boundary", label="Refine at mask/inner boundary",
                param_type="bool", default=False,
                tooltip="Subdivide mesh elements straddling holes / inner ROI "
                        "boundaries (pyALDIC MaskBoundaryCriterion).",
            ),
            ParamSpec(
                name="refine_roi_edge", label="Refine at ROI edge",
                param_type="bool", default=True,
                tooltip="Subdivide elements along the outer ROI boundary so the "
                        "field resolves the domain edge (pyALDIC ROIEdgeCriterion).",
            ),
            ParamSpec(
                name="refine_brush", label="Refine in brush region",
                param_type="bool", default=True,
                tooltip="Subdivide elements inside the painted brush region "
                        "(pyALDIC BrushRegionCriterion).",
            ),
            ParamSpec(
                name="min_element_size", label="Min element size (px)",
                param_type="int", default=8, min_val=2, max_val=128, step=2,
                tooltip="Smallest element the quadtree may refine to (power of two).",
            ),
            ParamSpec(name="brush_shapes", label="Brush shapes",
                      param_type="hidden", default={}),
        ]
    if op_key == SPECIAL_PRISM_OP_KEY:
        # Prism (V1.77). The channel(s) are chosen by wiring a channel-source pill into
        # the rainbow port (like analysis/DVC nodes) — no channel dropdown here.
        return [
            ParamSpec(
                name="channel_op", label="Channel operation", param_type="choice",
                default="add", choices=["add", "remove", "replace"],
                tooltip="How the channel(s) fed into this Prism combine with the "
                        "channels already flowing here, for ANALYSIS (solid) edges:\n"
                        "  add — also analyse the Prism's channel(s) downstream\n"
                        "  remove — stop analysing the Prism's channel(s) downstream\n"
                        "  replace — analyse ONLY the Prism's channel(s) downstream.\n"
                        "Ignored on a view-only (dotted) edge — that always feeds the "
                        "viewers only, never analysis.",
            ),
            ParamSpec(
                name="overlay_render", label="Overlay render", param_type="choice",
                default="Shell (iso)",
                choices=["Shell (iso)", "Cloud (volume)", "Cloud (MIP)"],
                tooltip="How a VIEW-ONLY (dotted-edge) channel is drawn in the DVC "
                        "viewer's 3-D object overlay: an iso-surface shell, a "
                        "translucent volume cloud, or a maximum-intensity projection. "
                        "The overlay is auto-clipped to the shell between the two "
                        "granule boundaries when a Granule Boundary node is upstream.",
            ),
        ]
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
    if op_key == SPECIAL_EXCLUDE_OP_KEY:
        # Exclude node (V1.75). No region is drawn here — the region comes from the
        # wired upstream node (3D Mask Drawing / Granule Volume Mask); this only
        # tunes how it is applied. `dilate_px` grows the ignored footprint by a
        # safety margin (e.g. to also skip the halo around a bright artefact).
        return [
            ParamSpec(
                name="dilate_px", label="Grow margin (px)", param_type="int",
                default=0, min_val=0, max_val=200, step=1,
                tooltip="Expand the excluded region outward by this many pixels "
                        "before analysis ignores it — a safety margin around the "
                        "wired mask (e.g. to also skip the halo of a bright "
                        "artefact). 0 = ignore exactly the region as drawn.",
            ),
        ]
    if op_key == SPECIAL_MASK3D_OP_KEY:
        # 3D Mask Drawing node params. The mask itself is drawn interactively via
        # the popup's "Draw 3D mask…" button and stored as vector shapes in the
        # hidden ``mask_shapes`` param; these flat params only choose how the drawn
        # planes are turned into a (Z,H,W) volume on Run.
        return [
            ParamSpec(
                name="mode", label="Creation mode", param_type="choice",
                default="Propagate across Z",
                choices=["Manual (per-plane)", "Propagate across Z",
                         "Threshold seed + edit"],
                tooltip="How the 3D mask is built. 'Manual' fills only the Z planes "
                        "you draw on. 'Propagate across Z' fills the planes between "
                        "the ones you draw (see Propagate). 'Threshold seed + edit' "
                        "seeds each plane's outline from an intensity threshold that "
                        "you then correct by drawing.",
            ),
            ParamSpec(
                name="propagate", label="Propagate", param_type="choice",
                default="Interpolate between planes",
                choices=["Copy to all Z", "Interpolate between planes"],
                visible_when={"mode": "Propagate across Z"},
                tooltip="'Copy to all Z' extrudes the union of the drawn planes "
                        "through the whole stack (a prism). 'Interpolate between "
                        "planes' smoothly morphs the outline between consecutive "
                        "drawn planes (signed-distance blend) so the object surface "
                        "is higher-resolution than the planes you drew.",
            ),
            ParamSpec(
                name="apply_all_frames", label="Apply to all frames (T)",
                param_type="bool", default=True,
                tooltip="Reuse the drawn mask on every timepoint (the object is the "
                        "same across the timelapse). Off = only the frame(s) you "
                        "drew on carry a mask.",
            ),
            # V1.68 — the only *surface* knob on the node (all other DVC-on-object
            # display choices — colour scalar, context channel, 2D projection —
            # live in the DVC panel and need no re-Run). Feeds the Taubin smoothing
            # pass in backend.viz3d.surface.build_object_surface when the drawn
            # mask is rendered as a deformation surface (Stout et al. 2016).
            ParamSpec(
                name="surface_smooth_iterations", label="Surface smoothing (iter)",
                param_type="int", default=10, min_val=0, max_val=100, step=1,
                tooltip="Taubin smoothing iterations for the marching-cubes surface "
                        "built from this mask in the DVC '3D Object' view. 0 = raw "
                        "voxel surface (blocky); higher = smoother object boundary. "
                        "Does not change the drawn mask or any measurement.",
            ),
            # The drawn shapes: {str(m): {str(t): {z_key: [shape, …]}}} with z_key an
            # int Z index or 'all'. Hidden so the popup's full-params rewrite never
            # clobbers it; captured via "Draw 3D mask…".
            ParamSpec(
                name="mask_shapes", label="Mask shapes", param_type="hidden",
                default={},
                tooltip="The drawn per-Z shapes. Edit with 'Draw 3D mask…'."),
        ]
    if op_key == SPECIAL_BEAD_DETECT_OP_KEY:
        # Bead detection knobs (backend/analysis/bead_detect.detect_beads). The
        # channel is chosen by wiring a channel pill into the rainbow port.
        return [
            ParamSpec(
                name="detect_mode", label="Detection mode", param_type="choice",
                default="log", choices=["log", "components"],
                tooltip="'log' = Laplacian-of-Gaussian blob maxima (bright compact "
                        "beads). 'components' = connected-component centroids "
                        "(after thresholding). Both are sub-voxel refined."),
            ParamSpec(
                name="min_distance_px", label="Min bead distance (px)",
                param_type="int", default=3, min_val=1, max_val=1000, step=1,
                tooltip="Minimum separation between detected beads (suppresses "
                        "duplicate maxima on one bead)."),
            ParamSpec(
                name="threshold", label="Intensity threshold (0 = auto)",
                param_type="float", default=0.0, min_val=0.0, max_val=1e9, step=1.0,
                tooltip="Absolute intensity threshold for detection. 0 = auto "
                        "(Otsu / percentile)."),
            ParamSpec(
                name="min_intensity", label="Min peak intensity",
                param_type="float", default=0.0, min_val=0.0, max_val=1e9, step=1.0,
                tooltip="Drop detected peaks dimmer than this."),
            ParamSpec(
                name="subpixel", label="Sub-voxel refinement", param_type="bool",
                default=True,
                tooltip="Refine each centroid to sub-voxel accuracy (parabola / "
                        "radial-symmetry fit)."),
            ParamSpec(
                name="all_multipoints", label="All multipoints", param_type="bool",
                default=False,
                tooltip="Detect in every multipoint. Off = only the current M."),
        ]
    if op_key == SPECIAL_GRANULE_CLUSTER_OP_KEY:
        # GMM (or KMeans) granule assignment with a BIC sweep over the relaxed
        # count (backend/analysis/granule_cluster.cluster_granules). Needs
        # scikit-learn (optional/lazy, find_spec-gated like cellpose/stardist).
        return [
            ParamSpec(
                name="n_granules", label="Granule count (seed)", param_type="int",
                default=10, min_val=1, max_val=100000, step=1,
                tooltip="Your initial estimate of the number of granules in the "
                        "volume. The actual count is chosen within ± the relaxation "
                        "below by lowest BIC."),
            ParamSpec(
                name="relax_pct", label="Relax count by (%)", param_type="float",
                default=25.0, min_val=0.0, max_val=100.0, step=5.0,
                tooltip="Sweep the granule count over [n·(1−p), n·(1+p)] and pick "
                        "the model with the lowest BIC. 0 = fixed at the seed."),
            ParamSpec(
                name="method", label="Method", param_type="choice",
                default="gmm", choices=["gmm", "kmeans"],
                tooltip="'gmm' = full-covariance Gaussian mixture (handles "
                        "elongated granules). 'kmeans' = spherical, faster, weaker "
                        "for elongated shapes."),
            ParamSpec(
                name="n_init", label="Fit restarts", param_type="int",
                default=3, min_val=1, max_val=50, step=1,
                tooltip="Number of random restarts per candidate count (best kept)."),
            ParamSpec(
                name="all_multipoints", label="All multipoints", param_type="bool",
                default=False,
                tooltip="Cluster every multipoint. Off = only the current M."),
        ]
    if op_key == SPECIAL_GRANULE_TESSELLATE_OP_KEY:
        # Boundary construction + density-merge
        # (backend/analysis/granule_tessellate.tessellate_granules).
        return [
            ParamSpec(
                name="tess_mode", label="Tessellation", param_type="choice",
                default="alpha_shape", choices=["alpha_shape", "voronoi"],
                tooltip="'alpha_shape' = per-granule concave hull (captures "
                        "roughness / elongation). 'voronoi' = global Voronoi cells; "
                        "a granule is the union of its beads' cells."),
            ParamSpec(
                name="alpha", label="Alpha (concavity)", param_type="float",
                default=0.0, min_val=0.0, max_val=1e6, step=1.0,
                visible_when={"tess_mode": "alpha_shape"},
                tooltip="Alpha-shape radius (µm). Smaller = tighter, more concave "
                        "boundary; 0 = auto; very large ⇒ the convex hull."),
            ParamSpec(
                name="merge_tol", label="Density-merge tolerance",
                param_type="float", default=0.2, min_val=0.0, max_val=1.0, step=0.05,
                tooltip="Merge adjacent granules whose point densities differ by no "
                        "more than this fraction (|ρi−ρj|/max ≤ tol), iteratively. "
                        "0 = never merge."),
            ParamSpec(
                name="adj_dist_um", label="Adjacency distance (µm)",
                param_type="float", default=0.0, min_val=0.0, max_val=1e6, step=1.0,
                visible_when={"tess_mode": "alpha_shape"},
                tooltip="Two alpha-shape granules are neighbours if their hulls "
                        "touch or their centroids are within this distance. "
                        "0 = auto (from the mean nearest-neighbour spacing)."),
            ParamSpec(
                name="min_granule_points", label="Min beads per granule",
                param_type="int", default=4, min_val=1, max_val=100000, step=1,
                tooltip="Granules with fewer beads than this are dropped or merged "
                        "into a neighbour (a hull needs ≥4 points)."),
        ]
    if op_key == SPECIAL_GRANULE_MASK_OP_KEY:
        # Voxelize + SDF-Gaussian smoothing
        # (backend/analysis/granule_mask.build_granule_masks).
        return [
            ParamSpec(
                name="smooth_sigma", label="Surface smoothing (µm)",
                param_type="float", default=0.0, min_val=0.0, max_val=1e4, step=0.5,
                tooltip="Signed-distance Gaussian smoothing of each granule volume "
                        "(µm). 0 = raw voxelized boundary; higher = smoother, "
                        "non-shrinking surface."),
            ParamSpec(
                name="fill_holes", label="Fill interior holes", param_type="bool",
                default=True,
                tooltip="Fill any interior voids so each granule is solid."),
            ParamSpec(
                name="min_object_voxels", label="Min granule voxels",
                param_type="int", default=8, min_val=1, max_val=10**9, step=1,
                tooltip="Drop voxelized granules smaller than this (specks)."),
            ParamSpec(
                name="all_multipoints", label="All multipoints", param_type="bool",
                default=False,
                tooltip="Mask every multipoint. Off = only the current M."),
        ]
    if op_key == SPECIAL_GRANULE_BOUNDARY_OP_KEY:
        # Outward boundary band (backend/analysis/granule_boundary.extract_boundary_bands).
        return [
            ParamSpec(
                name="band_voxels", label="Band thickness (voxels)",
                param_type="int", default=3, min_val=1, max_val=1000, step=1,
                tooltip="How many voxels outward from each granule surface to "
                        "include in the boundary band."),
            ParamSpec(
                name="band_method", label="Band method", param_type="choice",
                default="dilation", choices=["dilation", "edt"],
                tooltip="'dilation' = N binary dilations (voxel band). 'edt' = "
                        "Euclidean distance transform ≤ a physical distance "
                        "(honours anisotropic voxel size)."),
            ParamSpec(
                name="band_um", label="Band distance (µm, EDT)",
                param_type="float", default=0.0, min_val=0.0, max_val=1e4, step=0.5,
                visible_when={"band_method": "edt"},
                tooltip="Metric band thickness for the EDT method (µm). "
                        "0 = use band thickness × smallest voxel dimension."),
            ParamSpec(
                name="include_neighbors", label="Reach into neighbour granules",
                param_type="bool", default=True,
                tooltip="Include voxels belonging to adjacent granules in the band "
                        "(the shared matrix between packed granules). Off = "
                        "background only."),
            ParamSpec(
                name="all_multipoints", label="All multipoints", param_type="bool",
                default=False,
                tooltip="Band every multipoint. Off = only the current M."),
        ]
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
    if op_key == SPECIAL_SAVE_DATA_OP_KEY:
        # Save the current pipeline image data to disk. TIFF hyperstack reuses
        # backend.exporters.tiff_exporter.export_tiff_hyperstack (which accepts
        # both (T,H,W) and (T,Z,H,W)); NPZ writes the raw channel arrays. The
        # enhancement recipe is a 2-D operation on the Z-projected view, so it is
        # only available for the projection modes — the full Z-stack saves raw
        # voxels (registration + crop applied).
        return [
            ParamSpec(
                name="data", label="Data", param_type="choice",
                default="Full Z-stack (raw voxels)",
                choices=["Full Z-stack (raw voxels)",
                         "Z-projection (recipe-processed)",
                         "Z-projection (raw)"],
                tooltip="What to save. 'Full Z-stack (raw voxels)' writes every Z "
                        "plane as a (T,Z,H,W) hyperstack (registration + any "
                        "upstream crop applied; no enhancement recipe — that is a "
                        "2-D projection operation). 'Z-projection (recipe-"
                        "processed)' writes the Z-collapsed image the pipeline "
                        "analyzes (recipe + registration + crop). 'Z-projection "
                        "(raw)' writes the Z-collapsed raw image (registration + "
                        "crop, no recipe).",
            ),
            ParamSpec(
                name="image_format", label="Format", param_type="choice",
                default="TIFF hyperstack", choices=["TIFF hyperstack", "NPZ"],
                tooltip="'TIFF hyperstack' writes one multi-channel ImageJ TZCYX "
                        ".tif per multipoint (opens in Fiji / napari with the "
                        "correct pixel size). 'NPZ' writes a compressed .npz of "
                        "the channel arrays keyed by channel name.",
            ),
            ParamSpec(
                name="bit_depth", label="Bit depth", param_type="choice",
                default="passthrough", choices=["passthrough", "uint16", "uint8"],
                visible_when={"image_format": "TIFF hyperstack"},
                tooltip="TIFF output bit depth. 'passthrough' keeps the source "
                        "dtype; uint16 / uint8 apply a per-channel percentile "
                        "stretch so each channel keeps its own contrast.",
            ),
            ParamSpec(
                name="all_multipoints", label="All multipoints", param_type="bool",
                default=True,
                tooltip="Save every multipoint — one file per M "
                        "(pipeline_data_M01.tif, …) — so the full M axis is "
                        "preserved. Off = only the currently viewed multipoint.",
            ),
        ]
    if op_key == SPECIAL_CROP_OP_KEY:
        # Manual crop node. The rectangle is picked interactively via the popup's
        # "Pick crop region…" button (reusing the page's crop dialog) and stored
        # in the hidden ``rect`` param as (x, y, w, h) raw-image pixels; ``mode``
        # is a single-choice placeholder documenting that only manual is wired.
        return [
            ParamSpec(
                name="mode", label="Crop mode", param_type="choice",
                default="Manual (draw / enter rectangle)",
                choices=["Manual (draw / enter rectangle)"],
                tooltip="How the crop rectangle is chosen. Manual: pick it "
                        "yourself with 'Pick crop region…'. (Automatic "
                        "content-based cropping may be added later.)",
            ),
            ParamSpec(
                name="rect", label="Crop rectangle", param_type="hidden",
                default=None,
                tooltip="(x, y, w, h) in raw-image pixels. Set it with "
                        "'Pick crop region…'.",
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
