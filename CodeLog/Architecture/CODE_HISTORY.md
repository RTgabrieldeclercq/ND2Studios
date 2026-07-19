# ND2Studios — Consolidated Engineering Ledger (CODE_HISTORY.md)

## 0. What this is

This document compacts ~114 per-version implementation plans from `CodeLog/ClaudesPlan/`
(V1.0 through V1.70) into one reference optimized for building **future** software on
ND2Studios. It captures durable decisions, data contracts/APIs, reusable primitives, and
known gotchas/supersessions — it is not a narrative changelog (see
`CodeLog/Updates/CHANGELOG.md` for that) and not an architecture diagram (see
`CodeLog/Architecture/ARCHITECTURE.md` for that).

**The original plan files in `CodeLog/ClaudesPlan/` remain the authoritative, detailed
record** for any given version — this document is a lossy but densely cross-referenced
index into them. Where a later version reversed or replaced an earlier decision, that is
called out explicitly ("superseded by...").

Structure:
1. Thematic sections (primary value) — grouped by subsystem.
2. Chronology table — one row per version/feature, in shipping order.
3. Key reusable primitives index — flat lookup by name.

---

## 1. Pipeline-graph core (`nd2studios/pipeline_graph/`)

The node-graph editor (Pipelines tab) is the newest and most actively evolving subsystem.
It began as a Processing-only MVP (V1.45) and grew into a unified, checkpointable,
loop-capable, channel-wire-aware DAG executor by V1.62/V1.68.

### 1.1 Model & schema evolution

| Schema version | Introduced | What changed |
|---|---|---|
| v1 | V1.45 Pipelines tab | Separate Processing/Analysis/Results `GraphSlice`s + `Bridge` objects linking them by name only (no real edge). |
| v2 | V1.45 Merged Analysis+Results | Folds Results into Analysis (`PipelineDoc.merged` aliases `analysis`); adds `NodeCategory`/`ShapeKind` enums, if-else triangle node, SPECIAL hexagon action nodes. `_migrate_v1_to_v2` in `io.py`. |
| v3 | V1.45 Track Objects node | `special:validate_tracks` retargeted to `special:track_objects`; `_migrate_v2_to_v3`. |
| v4 | V1.49 Loop connector | `Edge.kind: "structural"\|"loop"` + `Edge.params` (loop config) added; no-op migration (`Edge.from_dict` already defaulted these). |
| v5 → v6 | V1.61 Pipelines overhaul | v5 interim; v6 (`_migrate_to_connected`) **removes Bridge nodes entirely** — Processing tail is reconnected directly to the Analysis head via `_processing_tail_node`; all three stages now render as ONE scene/one `GraphSlice`. `node.stage` is retained only for execution-dispatch bookkeeping. |

Core dataclasses (`pipeline_graph/model.py`): `PortType`, `Stage`, `NodeRole` enums;
`Port`, `Node` (`category`/`shape_kind`/`pos`/`bridge_id`, defaulted in `__post_init__`),
`Edge` (`kind='structural'|'loop'`, `params: Dict`), `GraphSlice`
(`add_node`/`remove_node`/`add_edge`/`remove_edge`, `would_create_cycle`,
`structural_edges()`/`structural_incoming()`/`structural_outgoing()`/`loop_edges()` —
**the required filtered accessors for any traversal that must ignore loop back-edges**),
`Bridge`, `PipelineDoc` (3 slices + bridges pre-V1.61; one slice post-V1.61).

**Gotcha (recurring):** any new code walking `GraphSlice` edges for structural purposes
(recipe linearization, channel propagation, topological order, `GraphRunner` indegree)
**must** use the `structural_*` filtered accessors, or loop edges silently corrupt the
traversal (V1.49).

### 1.2 Registry (`pipeline_graph/registry_adapter.py`)

Golden rule stated explicitly in the V1.45 plans: **read plugin/pipeline names at
runtime from the existing registries, never hardcode them.**

- `enhancement_specs()` — enumerates `PluginBase.get_plugins('enhancement')`, op_key
  `enhancement:<name>`.
- `analysis_specs()` — enumerates `AnalysisPipeline.get_pipelines()`, op_key
  `analysis:<name>`, Image→Binary.
- `results_specs()` — Compute Measurements / Summary, Binary→Data.
- `if_else_spec()` — purple triangle, aggregate gate (ALL/ANY condition), one input,
  true/false outputs.
- `special_specs()` / `merged_action_specs()` — orange hexagon SPECIAL action nodes,
  catalogued in `_SPECIAL_OPS`: category, `ShapeKind`, `input_types`/`output_names`.
- `op_produces_objects(op_key)` — `True` for `SPECIAL_MASK3D_OP_KEY`,
  `SPECIAL_TRACK_OP_KEY`, `SPECIAL_GRANULE_MASK_OP_KEY`, or any analysis node whose
  pipeline yields `label_masks` — drives the V1.68 Frame/Object scope-lever gating.
- `param_specs_for(op_key)` — per-op `ParamSpec` list; `visible_when` gates
  method-specific params (now also accepts a **list** of values, added V1.58).
- `build_node(...)` carries `category`/`shape_kind` + port names.

**Special op-key catalog** (all `ShapeKind.HEXAGON`, `NodeCategory.SPECIAL` unless noted):
`special:validate_tracks` (superseded), `special:track_objects` (V1.45, refined V1.58),
`special:review_object`/"Review Objects" (V1.45), `special:checkpoint` (V1.53,
`NodeCategory.CHECKPOINT`, white), `special:ct_fields`/"Cell-Tracker Spatial Maps"
(re-homed to `NodeCategory.RESULTS` in the interp-maps plan), `special:interp_map`
("Interpolated Spatial Maps", `NodeCategory.RESULTS`), `special:mask3d`/"3D Mask
Drawing" (V1.65, output `PortType.ANY` — corrected from an originally-specced `BINARY`
because `BINARY→IMAGE` is rejected by `can_connect`), DVC's `SPECIAL_DVC_OP_KEY`
("DVC (ALDVC)", V1.51), Registration's `SPECIAL_REGISTER_OP_KEY` (V1.56), Deconvolution's
`SPECIAL_DECONV_OP_KEY` (proposed, V1.63), and the 5 granule ops (V1.70,
`special:bead_detect`/`special:granule_cluster`/`special:granule_tessellate`/
`special:granule_mask`/`special:granule_boundary`, all imported from the new
`pipeline_graph/granule_ops.py`, not re-hardcoded).

`_normalize_special_categories` (`io.py`) re-derives a loaded node's `NodeCategory` from
its *current* registry spec at load time — the mechanism that let `ct_fields`/`interp_map`
be recategorized from SPECIAL→RESULTS with **no schema bump**. Reuse this whenever a
node's category changes in a later version.

### 1.3 Executor & GraphRunner (`pipeline_graph/executor.py`)

- `recipe_for_node(slice, node_id)`, `apply_recipe(channels, recipe, normalized, frame,
  progress_cb, cancelled_cb)` — reimplements the recipe-apply loop itself (does **not**
  import `EnhancedDataset`) so `pipeline_graph/` stays fully Qt-free/decoupled.
- `topological_order`, `GraphRunner` — state machine: `reachable_nodes`/`next_ready`/
  `complete(prune_ports=set())`/`is_finished`. `prune_ports` is the general mechanism for
  routing different row subsets down different branch outputs (if-else branching,
  V1.45; later reused conceptually for object-lens If/Else).
- `GraphRunner(..., frozen: Optional[set])` (V1.53) — each frozen node is marked
  reached+done+dispatched so a topological walk treats frozen checkpoint ancestors as
  complete without re-adding them to `next_ready`. `frozen=None`/empty = unmodified run.
- Lazy-volume wrappers, all mirroring the same `get_frame`/shape/`channel_names`
  protocol so viewer code needs zero changes to consume them:
  - `ProcessedFrameVolume` / `PinnedProcessedVolume(set_planes({(m,t):{ch:frame}}))` —
    per-displayed-frame or per-pinned-plane recipe application (V1.45).
  - `CroppedVolume` (V1.46, Preview Crop) — wraps raw/processed volume, reports
    cropped `height`/`width`, `get_frame` slices `[y:y+h, x:x+w]`.
  - `RegisteredFrameVolume` (V1.59) — lazy per-(m,t) registration-transform wrapper,
    `bypass_pyramid=True`, mirrors `channel_names`/`n_*`/`height`/`width`/`dtype`/
    `pixel_size_um`/`z_step_um`.
- **Register-then-crop is the established ordering rule** wherever registration and
  cropping interact (V1.59): register the full frame first, then crop last — a rectangle
  drawn in registered-display coordinates then maps directly onto the sliced region.
  `_processed_channels_for_m` and `_DVCJob._read_volume` both follow this order.

### 1.4 Conditions DSL (`pipeline_graph/conditions.py`)

Qt-free. `Condition` (ALL/ANY) of negatable `ConditionBlock`s; `BLOCK_KINDS` catalog
(results-number / object-population / timelapse families); `evaluate_condition` (whole-
frame gate, robust to missing data); `describe_condition`.

- **Lens** (V1.45 follow-up): `partition_rows(cond, rows, group_by)` with
  `LENS_FRAME`/`LENS_OBJECT` + `GROUP_*` constants — the general mechanism for any node
  needing per-object vs per-frame branching, not just If/Else. If-else node params gained
  `lens` + `group_by` (additive, `.get`-read, no schema bump).
- `TRACKING_METRICS` + `metric_choices(extra)` (V1.45 follow-up) — `celltracker_bridge`
  adds `speed_um`/`velocity_*_um`/`cell_density` columns; `ConditionBuilderDialog` and
  `pipelines_page._available_metric_columns`/`_has_upstream_celltracker` discover columns
  live by walking incoming edges.
- **Track persistence rewrite** (V1.46): redefined from a whole-frame population
  threshold to a genuine per-track min/max frame-count filter — `mode` (choice "At least"
  /"At most") + `frames` (int), restricted to the object lens only
  (`BLOCK_KINDS[...]["object_lens_only"]=True`, `is_object_lens_only(kind)`;
  `condition_builder_dialog._build_add_menu` accepts a `lens` kwarg and skips
  object-lens-only blocks outside `LENS_OBJECT`).
- **Track displacement/motility extension** (V1.50): new `basis` param (Net
  first→last | Cumulative path | Per-frame step, default Net) and `outlier` param
  (Reject object | Reject frame only, default Reject object, meaningful only for
  Per-frame basis). Per-track reduction for Per-frame basis = **max** consecutive step
  distance (feeds the existing aggregate/comparator UI unchanged).
  `scrub_outlier_frames(rows)` — new pre-partition row-scrubbing hook, called once at the
  top of `partition_rows`; removes only the "to"-frame row of an over-threshold step
  (frame-only rejection). **Gotcha:** frame-only rejection degrades to reject-object
  semantics inside whole-frame `evaluate_condition` (which returns one bool per frame and
  cannot drop individual rows) — only the object-lens `partition_rows` path can do true
  per-frame scrubbing.
- New block kinds for loop "until" stop conditions (V1.49): `nondup_object_count`,
  `tracking_coverage` (backed by `tracking_ratio(rows, n_frames)` from
  `pipeline_graph/loop.py`).

### 1.5 Terminal Dismiss (V1.46)

Redefined `Dismiss` from a dead no-op (checked `track_validation == "rejected"`, a value
Review never sets) into a genuine terminal discard:
`_discard_objects(drop_rows)` (`pages/pipelines_page.py`) — removes rows by `id()` from
`_run_context["rows"]`/`_results_rows`/`_track_overlay_rows`, and **zeros each dropped
object's `label_id` out of `label_masks[seg_channel][frame]`** in `_run_results_by_m[m]`
(mutates masks in place — destructive/session-scoped, not reversible without re-running).
Reusable pattern for any future terminal-discard node.

### 1.6 Checkpoint node (V1.53, hardened V1.59)

- White hexagon pass-through node. On Run, freezes everything computed upstream
  (masks, rows, tracks) keyed by an upstream-subgraph hash; if a later Run's current
  hash matches, resumes instead of recomputing from Input.
- `_checkpoint_upstream_hash`: sha1 of ancestor nodes' `(op_key, sorted params)`, edges
  feeding them, any loop-edge config touching them, plus the Processing recipe/
  per-channel recipes/normalized flag. **Crop was originally in this hash and was
  REMOVED in V1.59** — the preview crop is a downstream scoping concern and the
  registration crop is redundant (fully determined by already-hashed params), and
  crop is cleared at the START of every Run before hash comparison anyway, guaranteeing
  a spurious mismatch.
- `persist_to_disk` bool param (default off) → compressed NPZ + `index.json` manifest
  in `<pipeline>.checkpoints/`; **NOT extended to registration state** — a disk-reloaded
  checkpoint after an app restart will silently no-op registration restore.
- `_run_checkpoint` snapshots `_run_results_by_m`, `_run_all_rows`, `_run_context`,
  `_results_rows`, `_run_results_crop`, track overlay state — masks held **by reference**
  (session RAM), not copied.
- V1.59 also extended freeze/restore to registration state
  (`record._registration_by_m`/`_registration_interp_order`/`_registration_crop`,
  `_reg_by_m` viewer bundle) via `_restore_registration_state`, and fixed DVC's
  `_DVCJob` to honor registration + crop (previously read raw volumes ignoring both).
- Frozen masks are **re-scoped to the crop on restore**, not assumed full-frame:
  `_effective_run_crop()`, `_crop_contains(outer, inner)`, `_crop_analysis_result(res,
  rect, origin)` (origin-aware slice — `x0 = x - origin_x` since masks may be frozen in
  registration-crop space), `_recrop_restored_checkpoint(target, stored)` — **re-derives**
  measurement rows from sliced masks via `_ensure_run_rows` rather than offsetting frozen
  rows (correctness over cheap-offset, because the row schema has many spatial columns).
  Tracks re-derived via `_set_track_overlay_state` using the **default** per-M linker,
  NOT whatever method an upstream Track Objects node was configured with.
- Checkpoint-selection containment guard is **mask-gated**: only applies when the
  checkpoint actually froze masks — a registration-only checkpoint (no masks) is always
  resumable.
- Philosophy (stated twice, V1.53 and V1.59): **correctness over cache aggressiveness** —
  a param that doesn't actually affect output still triggers an unnecessary re-run when
  changed; a checkpoint frozen at a crop only serves the same-or-contained crop.

### 1.7 Loop / iteration connector (V1.49)

A visually distinct amber back-edge (`Edge.kind="loop"`) drawn left of the standard
bridge, from a node's bottom back to its (or an upstream node's) top — defines a loop
region with an iteration plan, invisible to every existing DAG codepath.

- `pipeline_graph/loop.py` (Qt-free): `loop_region(sl, loop_edge) -> LoopRegion`
  (entry/exit/body via forward∩backward reachability); `iteration_plan(config,
  body_nodes)` — Cartesian-product grid across swept `ParamSpec` axes (or `zip` for
  paired sweeps), `mode="count"`→N empty assignments, `mode="until"`→capped generator;
  `deduplicate_objects(rows, masks, *, metric, threshold)` — mask-IoU primary,
  centroid-distance fallback (reuses the greedy-matching idea from
  `object_tracker.link_objects`); `combine_iterations(results, rule, ...)` implementing
  `union_dedup`/`best`/`last`/`keep_all`; `tracking_ratio(rows, n_frames) -> (ratio,
  max_continuous_run)` — reusable tracking-quality metric (used by both the
  `tracking_coverage` condition block and the "best"-rule ranking).
- `can_connect_loop(src, dst)` requires src=output-on-bottom, dst=input-on-top, both
  structural ports (not CHANNEL); `would_create_cycle` is **skipped** for loop-edge
  validation.
- Loop body sub-run reuses existing node dispatch via a **scoped `GraphRunner` over a
  temporary `GraphSlice`** containing only body nodes + structural edges — outer run
  scratch (`_runner_obj`, `_run_context`, `_run_port_rows`) is saved/restored around each
  sub-run. Reusable template for running any subgraph in isolation while keeping outer
  Run state intact.
- `LoopSettingsDialog` (`widgets/node_board/loop_dialog.py`): Iteration / Stop / Combine
  results / Multipoint scope tabs.

### 1.8 Channel-wire model (V1.48)

Turns "which channel a process runs on" from a per-node parameter into graph wiring.

- `PortType.CHANNEL` (rainbow-colored, only CHANNEL↔CHANNEL, fan-in allowed unlike
  normal single-wire inputs); channel source node `op_key = "channel:<name>"` /
  `"channel:__all__"`, role `NodeRole.ACTION`, no structural input, `ShapeKind.PILL`.
- `channel_source_names(node)`, `has_channel_wiring(sl) -> bool`,
  `channel_sets(sl, all_names) -> Dict[node_id, Set[str]]` — topological propagation:
  `set(node) = union(channel-edge sources) ∪ union(structural-pred sets)`, falling back
  to **all names per node** when `has_channel_wiring` is False (backward-compat
  default for every pre-V1.48 saved graph). **Reusable pattern** for propagating any
  future per-node attribute set along the graph.
- `channel_recipes(sl, output_id, all_names) -> Dict[str, Recipe]` — per-channel recipe:
  a channel only gets enhancement steps at/after the node it was first wired into
  (monotonic suffix); unwired channels stay raw.
- `EnhancedDataset`/`ProcessedFrameVolume` gained `recipe_by_channel: Dict[str, Recipe]`
  override (raw for channels with empty/absent recipe); **not serialized in V1.48** —
  recomputed from the graph each time.
- **Critical, unrelated bugfix bundled in the same plan:** `record._raw_channels`/
  `record.processed_view()` hold exactly **one M** (whichever was last loaded/navigated)
  — any Run-like feature iterating M must re-materialize per-M channels via the new
  `_processed_channels_for_m(record, m)` (built the same way as the existing all-frames
  `ProcessedFrameVolume` base-image pattern), mirrored later by DVC/registration/granule
  work. Do **not** read `_raw_channels`/`processed_view()` directly when per-M
  correctness matters.

### 1.9 Frame/Object scope toggle (V1.68)

Generalizes the pre-existing single-crop Run seam (`_crop_rect()`/
`_effective_run_crop()`/`_maybe_crop_volume()`, `_DVCJob(rect=...)`) from one rectangle
to N per-object rectangles.

- `Edge.params['scope']` ∈ `{SCOPE_WHOLE='whole_frame', SCOPE_OBJECTS='objects'}`,
  default `whole_frame` (no schema bump — absent key = old behavior).
  `edge_scope(edge)`, `set_edge_scope(edge, scope)`.
- `backend/analysis/object_scope.py::ObjectRegion(object_id, bbox=(z0,z1,y0,y1,x0,x1),
  mask, source)`; `iter_objects(obj, voxel_size_um, *, min_voxels=1, pad=0)` dispatches on
  dtype: `(Z,H,W)` bool → 3D connected components (`scipy.ndimage.label`); `(T,H,W)`
  int32 label masks → per-label bbox unioned over T; tracked objects → per-track bbox
  union over frames.
- `executor.ObjectCropVolume(raw_volume, region, *, mask_out=False)` — same
  `get_frame`/`get_volume` protocol as `CroppedVolume`; `mask_out=False` default keeps
  surrounding matrix context (needed by DVC); T/M/C axes always conserved.
- Scope lever UI: `widgets/node_board/edge_item.py` pill ('▭ Frame'/'◈ Objects', orange
  tint), `node_scene.py::edge_scope_changed`/`toggle_edge_scope`.
- **First cut wired only DVC end-to-end**; generic per-object execution for
  analysis/results/export nodes explicitly deferred. `_apply_edge_scope_lever` initially
  gated the lever on `dst.op_key==SPECIAL_DVC_OP_KEY` specifically — **widened in the
  V1.70 granule integration (P6)** to gate on `edge.source node_produces_objects` instead
  (any object-producing source, not just DVC destinations).
- V1.70's granule `special:granule_mask` node reuses this exact mechanism for free —
  because its output data shape matches `record._mask3d_by_m` exactly (see §7).

### 1.10 The V1.61 Pipelines-tab overhaul (master plan, phased V1.61–V1.64)

Ground-up plan (R1–R14) merging Processing/Analysis into one auto-arranging graph with
a per-node POV multi-viewer. **Key grounding finding:** pre-V1.61, Processing and
Analysis were joined only by a Bridge **name label**, not a real graph edge — two
different engines (recipe linearization vs `GraphRunner`).

- R1 (V1.61): merge to one scene/slice, schema v4→v5→v6, bridges removed
  (see §1.1).
- R2 (V1.61): per-category add-menu (grouped by `spec.effective_category()`).
- R4 (V1.61): double-click rename for input/output nodes (rename moved to
  click-on-name; double-click now promotes "previewed" node instead — a UX change
  worth remembering since earlier plans described double-click-to-rename).
- R5 (V1.61): pan + marquee gesture mode machine (`_BoardView._Mode` enum:
  IDLE=pan/SCISSORS/LOOP/LABEL).
- R3/R8/R9 (V1.62, Phase 2): **multi-input files** — every loaded file in the Import
  tab gets its own input node bound to its `ND2StudiosRecord` by `exp_id`
  (`node.params['source_exp_id']`); `_active_record()` resolves through the **focused**
  input node (the one feeding the previewed/running chain) — chosen over rewriting the
  ~52 existing `_active_record()` call sites individually (a reusable "resolve through
  one indirection point" refactor pattern). `_sync_input_nodes()` replaces the old
  single-universal-input `_ensure_input_node`; closed files disable (grey, keep) their
  node. R8 (channels on input-node body) and R9 (channel-bridge spacing) were explicitly
  **deferred** within V1.62, not fully built.
- R6 (auto-arrange DAG layout) requires a `Research/dag_layout.md` literature review
  before code, per CLAUDE.md — check whether this landed before extending layout logic.
- R13 (multi-input rendering: one tab per feeding input, not composited) and R7 (label
  groups, persisted in `.nd2s_pipeline.json`, schema bump) are part of the master plan
  but their delivery status should be verified against later changelogs before assuming
  complete.

**Gotcha:** merging into one runnable scene requires `GraphRunner` to **pass through**
enhancement/PROCESSING-stage nodes (no run handler pre-merge); ~15 execution-critical
call sites (`_graph_recipe`, `_graph_channel_recipes`, `_apply_processing`,
`_do_processing_preview`, bridge counters) must filter to PROCESSING-stage outputs —
getting this subtly wrong produces wrong scientific results, not a crash.

---

## 2. Import, tile-layout & multipoint stitching

### 2.1 Tile-layout algorithm history (important supersession chain)

The stage-XY→pixel layout algorithm was rewritten **seven times** before landing on a
from-scratch regime-aware rebuild in V1.54. When touching stitching/tile-layout code,
treat **V1.54's `backend/stitch/` package as the only current implementation** —
everything before it is superseded history, kept here for context on *why* certain
edge cases are handled the way they are:

| Version | Algorithm | Fate |
|---|---|---|
| V1.1 | Basic stage-XY stitch dialog, `compute_tile_layout` | Baseline |
| V1.4 | Fixed a bug where `frame_metadata(i)` was misread as indexing M directly (it indexes the flattened M×T×Z×C sequence) — new `_flat_index(coords, dim_order, sizes)` helper, one metadata fetch per M | Bugfix, `_flat_index` durable |
| V1.5 | Flush-grid: `_cluster_axis`/`_assign_index` sweep-gap clustering, 30%-of-tile-size tolerance | Superseded by V1.6 (fixed tolerance too coarse) |
| V1.6 | Adaptive clustering tolerance derived from the data's own sorted-gap structure (unimodal vs bimodal via gap-ratio > 3.0); scrollable `TileLayoutWidget` (drops fixed 240px height) | Tolerance formula durable; layout algorithm superseded by V1.7/V1.8 |
| V1.7 | Duplicate-position horizontal expansion (`max_dup` sub-columns per physical cell) | Superseded by V1.8 for the revisit case (looked like real structure on degenerate axes) |
| V1.8 | Serpentine/boustrophedon layout when `max_dup > 1` (row-scan alternating direction, M=0 top-left) | Superseded by V1.12 (dropped for pure physical) |
| V1.10 | Extended serpentine trigger to also fire on `fill_ratio < 0.9` (sparse/irregular ROI scans), new `StitchLayout.serpentine_reason` field | Trigger condition concept carried forward |
| V1.11 | Partial-metadata tolerance: truncate `m_indices` to `len(stage_xy_um)` when ND2 reports more positions (P) than valid frame_metadata | **Preserved through every subsequent rewrite** including V1.54 |
| V1.12 | Pure physical placement (no clustering, no serpentine) — literal stage-XY pixel offsets | Superseded next version |
| V1.13 | Row-flush, center-aligned (compress tile-sized gaps) | Reverted the very next version — "loses true spatial relationships" |
| V1.14 | Physical rebuild (reverts V1.13); switched primary stage-XY source to `f.experiment` (XYPosLoop) over per-frame `frame_metadata()` readback | Was "authoritative" until V1.54 |
| **V1.54** | **Regime-aware rebuild**: `backend/stitch/` package, auto-detects overlap-vs-zero-overlap per dataset, routes to registration (m2stitch/ashlar/phase-correlation) or coordinate-only placement | **Current implementation** |

### 2.2 V1.54 — the current stitching pipeline (`nd2studios/backend/stitch/`)

Core insight: overlapping tiles can be registered from image content; zero-overlap tiles
share no pixels and **must** be placed from stage coordinates only (registering them
would lock onto spurious noise). Regime auto-detected from overlap fraction; user can
override in the dialog.

- `config.py::StitchConfig`, `dataset.py::Tile`/`Dataset` + `build_dataset(volume,
  stage_xy_um, meta, config)` (kept the V1.14 orientation math: negate stage XY,
  `x=(sx-min)/px`, `y=(max-sy)/px`), `regime.py::decide_regime(dataset, config) ->
  'overlap'|'zero_overlap'`, `positions.py::coordinate_positions(dataset)` (also
  exports `StitchLayout` + `compute_tile_layout` compatibility shim), `register.py`
  (built-in phase-correlation: `skimage.registration.phase_cross_correlation` + Hann
  window + band-pass pre-filter + NCC confidence + spanning-tree global optimization),
  `engines.py::compute_positions()` for `auto|coordinate|phase_correlation|m2stitch|
  ashlar`, `compositor.py::composite(...)` (feather/average/max/none blending, float32
  accumulation cast back to source dtype), `illumination.py` (BaSiC gated, built-in
  flatfield fallback), `writer.py::write_ome_tiff()` (pyramidal/tiled/BigTIFF/TZCYX),
  `qc.py::write_qc()`, `pipeline.py::run_stitch(...) -> StitchResult`.
- `backend/exporters/stitch_exporter.py` is now a **thin compatibility shim**
  re-exporting `StitchLayout`, `compute_tile_layout`, `_cluster_axis` — every prior
  import site keeps working; `export_stitched_tiff` is deprecated and delegates to
  `run_stitch`.
- Optional gated engines (never in `requirements.txt`): `m2stitch` 0.7.2 (primary
  overlap engine, regular grids), `ashlar` 1.20.0 (wired via `jdk4py` — a packaged,
  pip-installable OpenJDK that auto-sets `JAVA_HOME` so `import jnius` works without a
  system JDK; reusable pattern for any future JVM-dependent library), `basicpy`
  (BaSiC illumination — not currently importable due to API drift; built-in smoothed
  flatfield is the default).
- Output format: pyramidal, tiled, BigTIFF OME-TIFF, always (dtype/channels/pixel size
  preserved). `backend/tiff_loader.py` gained `read_ome_tiff_metadata` so the "reload
  stitched TIFF" path understands it.
- 7 real bugs found and fixed by adversarial review post-implementation (worth knowing
  if debugging stitching edge cases): single-row/strip fracturing into bogus rows from
  sub-tile jitter; register.py's hard per-tile clip pinning far tiles incorrectly;
  compositor negative-offset placement; BigTIFF/pyramid-size threshold miscalculation;
  per-timepoint registration needing a shared reference origin; Windows memmap handle
  not released before temp-file removal; illumination silently skipping on shape
  mismatch instead of erroring.
- Zero-overlap gaps rendering as fill (0) is **correct behavior**, not a bug — never let
  zero-overlap data reach a registrar.

### 2.3 Related fixes/features

- **V1.0 "Fix Black Stitch Output"**: removed a bare `except: pass` around per-tile
  frame reads (silently produced all-black "success" output); fixed `resolutionunit=
  "MICROMETER"` (not a valid tifffile value — silently dropped, not an error) to
  `None` across `stitch_exporter.py`, `tiff_exporter.py`, `composite_exporter.py`.
  **Anti-pattern to grep for anywhere:** swallowing per-frame read exceptions inside a
  `BaseWorker` task silently turns total failure into a "finished" success signal.
- **V1.30 "Preserve Z-stacks through stitched exporter"**: `z_mode='none'` on a
  multi-Z ND2 now correctly preserves ALL Z planes as `(T,Z,C,H,W)` instead of
  collapsing to Z=1; fixed `LazyTIFFChannel` Z-stride bug (`base_page + z*n_c`, was
  stride-1, wrong for multi-channel TZCYX).
- **V1.15**: stitch multi-channel export switched from forced 8-bit RGB downcast to
  per-channel `minisblack` pages at native dtype (ImageJ-hyperstack compatible).
- **V1.16**: fixed re-import of stitched multi-channel TIFFs — loader now reads
  ImageJ hyperstack metadata (`read_imagej_tiff_metadata`, `load_imagej_tiff_channels`
  in `backend/tiff_loader.py`) instead of dumping all channels onto the T slider.
- **V1.27/V1.28 — multi-file reconstruction**: V1.27 added Z-only multi-file import
  (`LazyMultiFileND2Volume`/`MultiFileLazyZChannel`); **fully superseded by V1.28's
  generalized `chain_axis` (T/M/Z/C)** — `MultiFileLazyZChannel` renamed
  `MultiFileLazyChannel`, V1.27's multi-select Browse UX removed in favor of a
  `ReconstructDialog`. `chain_offsets` cumulative-offset-table routing is the reusable
  core abstraction. V1.31 added optional `chain_mapping: list[(file_idx, local_idx)]`
  for drag-reorderable interleave/reverse/repeat via `AxisBlocksWidget`.
- **V1.29**: unified Z-Projection TIFF export and Stitched TIFF export onto one
  `export_tiff_hyperstack(channels, enabled, filepath, bit_depth, pixel_size_um,
  progress_cb)` writer — single `(T,Z,C,H,W)` ImageJ TZCYX hyperstack for both paths.
- **V1.57 — Export from Import tab**: adds export directly on `FilePanel` (previously
  only reachable via Import→Confirm→Export), honoring the T/M/Z tile-strip crop
  (already baked into `record._raw_volume`) plus a new spatial XY ROI rubber-band crop
  (`self._xy_crop`, stored not mutated, realized only at export time). Reuses
  `ExportWorker`/exporters wholesale — no new export backend.

### 2.4 Multi-file import & reconstruction contracts (current, post-V1.31)

- `LazyMultiFileND2Volume(filepaths, chain_axis, chain_mapping=None)` /
  `LazyMultiFileTIFFVolume(...)` — mirror the full `LazyND2Volume` surface (`shape`,
  `n_multipoints`, `get_frame`, `to_lazy_channel`, `all_channels_as_lazy`, `reopen()`).
  `reopen()` on both single- and multi-file volumes is the interface method every
  threaded reader (prefetch, IOWorker, DVC) depends on.
- `ReconstructDialog` (`pages/reconstruct_dialog.py`) — QTableWidget of file shapes,
  chain-axis combo (auto-suggested from filename patterns, always overridable),
  file-type lock (ND2 or TIFF, never mixed).
- **V1.9 defensive fix**: `MultiAxisViewer._normalize_to_2d(frame)` — coerces
  arbitrary loader output (RGB/RGBA TIFFs, hyperstacks, 1D) to displayable 2D grayscale
  via Rec.601 luminance (`0.299R+0.587G+0.114B`, alpha dropped) rather than crashing on
  `ValueError: too many values to unpack`. Applied at every frame-read call site
  (`_compose_current_frame`, `_populate_lut_samples_from_volume`). **Reusable general
  primitive** for any viewer consuming heterogeneous image sources; house convention for
  RGB→grayscale collapse anywhere in the codebase.

---

## 3. Viewers — 2D (`MultiAxisViewer`/pyqtgraph) and PyVista 3D

### 3.1 2D viewer evolution

- **V1.1**: introduced `MultiAxisViewer` + `LazyND2Volume` for M/Z scrolling, per-channel
  `LutHistogramWidget` (draggable contrast), stage-XY stitch dialog.
- **V1.2**: split into image canvas + `ChannelChip` strip + collapsible `LutSidebar`
  (260px, animates min/maxWidth not `setVisible` — avoids QHBoxLayout flicker; house
  convention for any collapsible panel), plus a `TilePreviewWidget` corner inset
  (superseded by V1.3).
- **V1.3**: `TileLayoutWidget` (navigate/select dual mode, `set_mode()`) replaces the
  fixed corner preview; `compute_tile_layout()` remains the single geometry source of
  truth for navigate/select/stitch previews alike.
- **V1.0 perf work** (Smooth Playback + Overlay Performance Cache): `PreRenderWorker`
  (background `QThread`) + three-tier display fast-path — QPixmap cache → numpy
  `_render_cache` → live compose. Overlay-perf added a second tier (`_pp_cache`,
  `_pp_version`) layered on top specifically for polygon/analysis overlays so T-slider
  scrubbing stays instant with overlays active. Cache-invalidate-and-restart-worker is
  the standard response to any LUT/channel-color change.
- **V1.17**: `FrameCache` (thread-safe LRU, `backend/frame_cache.py`) +
  `PrefetchManager(QThread)` — prefetches ±5 neighbor T/Z frames on its own reopened
  volume handle (nd2 file handles are **not thread-safe** — every new worker must hold
  its own via `volume.reopen()`, a rule repeated in every later threading phase).
- **V1.34 (Phase 2)**: `utils/threading.py::IOWorker(QThread)` — moves cache-miss reads
  off the GUI thread; `utils/resources.py::recommended_cache_budget_bytes()` /
  `recommended_worker_count()` — the shared RAM/core-count source of truth reused by
  every later phase.
- **V1.35 (Phase 3)**: velocity-aware prefetch (forward/backward-biased window based on
  scrub direction), key pinning (`FrameCache.pin/unpin`, per-channel-plane granularity,
  never per-composite), `CacheStats` telemetry, immutable cached arrays
  (`arr.flags.writeable = False`).
- **V1.36 (Phase 4)**: `GpuImageCanvas` (`widgets/gpu_image_canvas.py`) — pyqtgraph-
  backed, one `pg.ImageItem` per channel + GPU-side LUT/levels +
  `CompositionMode_Plus` blending, feature-flagged via `Settings.USE_GPU_DISPLAY`
  (default True) / `ND2_DISABLE_GPU_DISPLAY` env override, per-instance defensive
  fallback to legacy `ImageCanvas` on any construction failure. **Gamma is ignored on
  the GPU path** (pyqtgraph `ImageItem` only supports linear levels) — a known
  "gamma does nothing" symptom. Legacy `ImageCanvas` (`widgets/image_viewer.py`) is
  NOT removed — still used by export preview/recipe/export/stitch dialogs/lut_sidebar.
- **V1.42 (industry-comparison optimizations)**: `_invalidate_render_cache(*,
  lut_only=False)` — LUT/contrast drags on GPU canvas become a no-op instead of full
  cache invalidation (pyqtgraph reapplies levels via `setLevels()`); `.nd2idx.json`
  sidecar metadata cache (`read_or_cache_nd2_metadata`, mtime-validity only); a
  cancel-first single-slot "mailbox of size 1" request queue
  (`utils/request_queue.py`); LRU on `LazyND2Channel._read_frame` sized via
  `recommended_cache_budget_bytes()`. `PyramidReader.pick_level_for_viewport` (from
  V1.39) is consumed here as the stable viewport→level API.
- **V1.43**: `FrameStrip` (`widgets/frame_strip.py`) — single-paint-surface tile-based
  navigator replacing the plain QSlider per axis (M/T/Z), click/shift/ctrl-click/
  keyboard selection, right-click crop-to-selection; the hidden QSlider stays the model
  of record for backward compat. `MaterializedDataset.subset(*, m,t,z)` — non-mutating
  advanced-indexing crop copy; `CropWorker(BaseWorker)` runs it off-thread.
  `_frame_meta_text(axis, idx)` (per-frame metadata string) reused directly by V1.44's
  pinned metadata panel.
- **V1.44 (GUI Overhaul)**: top tab bar replaces the left collapsible sidebar (the
  60↔240px animated sidebar described in CLAUDE.md is **now historical, not current
  architecture**); `widgets/icon_button.py` — `ui_scale()`/`scaled(px)` (DPI factor),
  `make_icon`/`icon_button`/`tool_button`/`bind_toggle_icon` (qtawesome-based, guarded
  import — missing install degrades to text buttons); `LutSidebar.set_metadata_text`
  pins per-file metadata permanently at the sidebar bottom. **House convention going
  forward for any new button/icon** — see the `nd2studios-widget-layout` skill.
- **V1.64**: `screen_scale()` — screen-height-derived scale factor (1.0 at/below 1080p,
  slope 0.5 up to a 1.5 cap) folded into `ui_scale()`; `core/theme.py::build_stylesheet()`
  scales every QSS pt-font and px-dimension by `screen_scale()` only (not the DPI
  factor — QSS px was never DPI-multiplied). Canvas/QPainter-drawn text (node board,
  filmstrip) explicitly out of scope.

### 3.2 Pipelines-page viewer additions (V1.45–V1.46)

- Node-graph editor got its own reused `MultiAxisViewer` pane; overlay tab bar
  (Image/Segmentation/Tracks/Vectors) added in the V1.46 viewer overhaul —
  `_overlay_result_for(m,t)` prefers previewed planes, falls back to the full per-M Run
  result; `_composite_pipeline_overlay` dispatches per active tab. Bottom `_data_tabs`
  QTabWidget hosts Measurements + lazily-built `MplCanvas` plot tabs.
  `backend/track_overlays.py` (Qt-free) — colormap/overlay/vector-drawing primitives
  (`generate_track_colormap`, `make_colored_overlay`, `draw_cell_vectors`/
  `draw_grid_velocity`, cv2 arrowheads).
- **V1.46.3**: `widgets/popout_window.py::PopOutWindow(QDialog)` — reparents a **live**
  widget (not a copy) into a maximized standalone window and docks it back on close via
  a guarded `on_restore` callback; generic, page-agnostic, reuse for any future
  "maximize this panel" affordance. `_set_panel_visible(kind, visible)` indirection
  layer is required — direct `setVisible` calls fight a currently-popped-out panel.
- **Recurring V1.46 bugfix theme**: preview and Run are two separate code paths that
  both build overlay/track/plot state; several bugs were "X works on Run but not
  Preview" because the async preview-completion path never called
  `_update_overlay_tabs_available()` or built track-overlay state. Fixed via a shared
  `_set_track_overlay_state(rows)` helper. **When adding a third execution path (e.g.
  batch), call the shared helper too rather than reimplementing.**
- **Spatial Maps tab** (V1.46, final form): `special:ct_fields` and `special:interp_map`
  both open a SHARED interactive "Spatial Maps" tab (`widgets/spatial_maps_panel.py`)
  instead of exporting to disk — see §8.
- **SerialTrack tab** (V1.47): `widgets/serialtrack_panel.py` — mirrors
  `SpatialMapsPanel`'s compact/full sidebar + FrameStrip + template + cache pattern.
  See §5.
- **DVC panel** (V1.51+): `widgets/dvc_panel.py` — adapts `serialtrack_panel.py`; gained
  a "3D Object" subtab in V1.67/V1.68. See §6.

### 3.3 PyVista 3D viewer (V1.65)

Design doc status: Proposed → later confirmed built (referenced by V1.67/V1.68 as
`widgets/viewer3d/pyvista_viewer.py`, off-screen VTK → QLabel).

- `PYVISTA_AVAILABLE = find_spec('pyvista') and find_spec('pyvistaqt')` checked before
  any import; unavailable → `Missing3DDeps` placeholder, never crashes; never pinned in
  `requirements.txt` (3rd instance of this optional-heavy-dependency pattern after
  cellpose/stardist).
- Layering extends backend-purity to be VTK-free too: `backend/viz3d/{prep,overlays}.py`
  pure numpy; `widgets/viewer3d/*` is the only Qt+pyvista mixing point;
  `workers/volume3d_worker.py` (`BaseWorker`) reads `(Z,H,W)`+LUT off-thread, returns
  **only numpy** — VTK actors/mappers are created/mutated **exclusively on the GUI
  thread** (non-negotiable threading contract).
- Renders `record._raw_volume` (raw, un-collapsed Z) — never processed/Z-collapsed
  channels.
- `PyVista3DViewer` API deliberately mirrors `MultiAxisViewer`'s surface (`set_volume`,
  `set_channels`, `coords()`, `channel_state()`/`apply_channel_state`,
  `set_frame_timestamps`, `refresh()`) so a 2D↔3D toggle can hand LUT/color state back
  and forth with zero call-site changes.
- Anisotropic voxel spacing always applied: `ImageData.spacing=(dz,dy,dx)`; axis order
  (Z,Y,X) matches the DVC/PTV `(z,y,x)` convention (avoids transpose bugs).
- 4 render modes: volume (additive default, composite alternative), MIP, slices,
  isosurface. Time playback via an LRU cache keyed `(name,m,t,lut-sig,z-range)`; when
  builds can't keep up, **drops frames** (skip to latest requested t) rather than
  queuing.
- `backend/viz3d/prep.py::Spacing` dataclass + `to_uint8` — the Z-generalization of the
  2D `apply_lut`; must match numerically.
- `backend/viz3d/overlays.py::DVCField`/`PtvTracks` dataclasses — pure adapters over
  existing `DVCResult`/`TrackData`, explicitly **not re-deriving** strain math (reuses
  `field_bundle_from_result` + `serialtrack_analysis.scalar_field`).
- `MaterializedDataset.get_volume(c,m,t,z_start,z_end) -> (Z,H,W)` is "the true 3D read
  path" (preserves Z, unlike `get_frame` which projects) — already used by DVC, reused
  here; duck-typed against any volume exposing `get_volume` + extents.
- Extended in V1.67/V1.68 with an "overlay-only render path" (no backing raw volume
  needed) — see §6.

---

## 4. Enhancement / Analysis / Segmentation

### 4.1 Analysis framework (V1.19)

`core/analysis_registry.py::AnalysisPipeline` ABC + registry (separate from
`PluginBase._registry`), `AnalysisResult` dataclass, `workers/analysis_worker.py`.
**The extension point for all future analysis pipelines** — implement `get_params()` +
`run(cancelled_cb, ...)`, force-import once in `__main__.py` for the registration side
effect (recurring gotcha: forgetting the import means the pipeline silently doesn't
appear). `analysis_results` is intentionally **not serialized** in `.nd2s` sessions
(masks can be hundreds of MB) except where later explicitly extended (workspace
staging, V1.38).

Pipelines registered over time, each an independent module force-imported in
`__main__.py`:
- **Nuclei Segmentation** (Cellpose 3, V1.19) — optional lazy dep via
  `importlib.util.find_spec`.
- **Tear/dark-region detection** (V1.19, classical Phase-1 baseline) — rank-normalized
  intensity/variance/entropy score map, robust to intensity drift across
  batches/microscopes; `analysis_page.py`'s CSV export switched from a hardcoded
  7-field list to dynamic `list(measurements[0].keys())` — **durable contract for every
  pipeline since**.
- **Histogram Threshold Segmenter** (V1.20) — `backend/analysis/histothresh/` 7-module
  Qt-free subpackage (validation/histogram/thresholds/morphology/config/identifier);
  `compute_histogram(image, bit_depth).percentile(p)` reused verbatim by V1.21's spots
  intensity gate.
- **Bright & Dark Spots** (V1.21, GA3-style) — `backend/analysis/spots/` 6-module
  subpackage mirroring histothresh's template; scale-space LoG/DoG blob detector +
  circularity symmetry gate + histothresh intensity gate; `secondary_label_masks`/
  `secondary_overlay_color`/`secondary_overlay_alpha` added to `AnalysisResult` (V1.24)
  as a **generic extension point** any pipeline can populate with zero
  `analysis_page.py` changes.
- **Manual Mask** (V1.30, renamed **Mask Analysis** in V1.40) —
  `backend/analysis/manual_mask.py`; drawing tools on `ImageCanvas`
  (`set_draw_mode('rect'|'ellipse'|'polygon')`, `shape_drawn` signal); a `"hidden"`
  `ParamSpec` type (added to `ParamEditor`) smuggles `frame_shapes` through the params
  dict without UI rendering — reusable for any future non-UI pipeline state.
  V1.40 added `generate_tiled_label_boxes`/`generate_tiled_mask` (evenly-spaced
  rectangle-strip auto-objects with stable IDs across time, `label_offset` kwarg keeps
  drawn-shape IDs disjoint from tiled IDs — general layering technique for auto-shapes
  under user shapes). **Gotcha:** any hardcoded `"Manual Mask"` string instead of the
  `MANUAL_MASK_PIPELINE_NAME` constant breaks after the rename.
- **StarDist Segmentation** (V1.45) — `backend/celltracker/segmentation.py` (vendored
  from CellTracker), `backend/analysis/stardist_segmentation.py`; optional lazy dep
  exactly like Cellpose; `_ensure_tf_ready` (renamed from `_ensure_tf_threading` in
  V1.46, function **kept** — pins TF to single inter/intra-op thread, **do not remove**,
  see §4.3) + `_patch_windows_symlink`. `AnalysisResult.overlay_outline: bool` field —
  outline rendering hint honored by both the interactive overlay composer and export
  overlays; primary mask outline uses `find_boundaries(mode='outer')` single-colour
  (matches CellTracker exactly); export/results_engine path uses `mode='inner')` +
  per-label palette (boundary pixels stay object-side/attributable) — **do not conflate
  the two modes**.
- **3D Mask Drawing** (V1.65) and the **Granule** node chain (V1.70) are analysis-
  adjacent SPECIAL nodes — see §7 and §11.

### 4.2 `filter_and_relabel` — the vectorized area-filter+relabel fix (V1.46)

Root cause of StarDist/Cellpose being ~2x slower than sibling CellTracker: an
O(n_objects × pixels) per-frame loop (`mask[mask==region.label]=0` per out-of-range
object, then per-object relabel). Fixed with a single O(pixels) pass:
`backend/analysis/source_utils.py::filter_and_relabel(mask, min_area, max_area)` —
`np.bincount` for per-label areas, LUT mapping surviving labels to contiguous ids,
`lut[mask]` applies both filter and relabel at once. Measured: 2048² frame, ~6k labels:
12.81s → 0.030s (~430×). **Reusable for any labeled-mask post-processing needing area
filter + contiguous relabel together.**

### 4.3 TensorFlow threading — CRITICAL gotcha, corrected twice

CellTracker (and by extension ND2Studios' StarDist port) pins TF to
**single inter/intra-op thread** before any TF op — required because StarDist tiles
large frames into many small `predict_instances` passes, and multi-threaded intra-op
oversubscribes cores and thrashes. An interim V1.46 revision REMOVED this cap on the
theory that CellTracker used TF's multi-threaded default; this was **wrong** and caused
a **~10x slowdown**. The cap was restored (function renamed `_ensure_tf_ready`) and is
now the confirmed-correct final state. **Do not repeat this mistake** — any future
"restore CellTracker's TF threading" work must keep the single-thread pin.

### 4.4 Multiprocessing for StarDist (V1.46)

`backend/analysis/mp_stardist.py::run_stardist_multiprocess(...)` — process-parallel
path (ProcessPoolExecutor, spawn context) because the StarDist model is a module-level
singleton not thread-safe across threads; each worker process loads its own model+TF
(still 1 op-thread each). GPU path stays sequential (avoids VRAM contention).
`utils/resources.py::recommended_process_count()` — RAM/core-gated (~1.5-2GB/worker,
cap 8), mirrors `recommended_worker_count()`. **Not extended to Cellpose** (has its own
batching/GPU path).

### 4.5 Streaming analysis (V1.46) — see also §9

`backend/analysis/plane_runner.py::run_planes_to_labels(...)` — routes each frame to
either an in-RAM array or a `LabelStackWriter` (disk-spilling) depending on
`should_stream_analysis()`. `AnalysisPipeline.needs_full_stack: bool = False` — opt back
into in-RAM per pipeline (reserved for future temporal pipelines). Spots and Nuclei
segmentation stay `n_workers=1` (Cellpose/segmenter not thread-safe); histogram/tear
detection stay parallel.

---

## 5. Tracking / SerialTrack / PTV

### 5.1 Object linking — algorithm evolution

- **V1.0**: `backend/object_tracker.py::link_objects` — first cross-frame linker,
  `scipy.optimize.linear_sum_assignment` (full Hungarian). `TrackValidationDialog`
  interactive accept/reject (tracking recomputed on every "Compute Measurements" click,
  never persisted).
- **V1.45 "Track Objects node"**: renamed from "Validate Tracked Objects"; per-track
  moving reference centroid, gap-tolerant re-linking (`max_frame_gap`), size-difference
  gate; `link_objects_with_params(rows, params, pixel_size_um)` — the canonical
  Qt-free param-translation function any future node/UI should call rather than
  `link_objects` directly. Track vs Review split: linking and accept/reject are now
  two separate single-purpose nodes.
- **V1.45 CellTracker integration**: vendored `backend/celltracker/tracking.py`
  (Qt-free) — topology (`compute_topology_features`, `link_frames`) and fingerprint
  (`fingerprint_cost_matrix`, `track_fingerprint`) trackers, bridged via
  `_link_group_celltracker` which remaps CellTracker's per-call 1-based `track_id`
  onto a shared global counter.
- **V1.45 SerialTrack**: vendored headless subset `backend/serialtrack/` (9 modules,
  excludes io/results/run which pull h5py/tifffile/PySide6) as `METHOD_SERIALTRACK`;
  uses `track_coordinates` (not `detection` — centroids are pre-segmented);
  Incremental (default, handles appear/disappear) vs Cumulative mode. `numba` is the
  one new hard dependency. Later extended (2026-07-05) with the **full solver surface**
  including ADMM: `st_solver` (MLS/Regularization/ADMM), `st_loc_solver`
  (Topology/Histogram-then-Topology), `st_smoothness`, `st_outlier_threshold`,
  `st_max_iter`, etc.
- **V1.58 — LAP birth/death + mask-overlap linker** (major rewrite): root cause of two
  failure modes (spurious long links at high `max_distance`; chain-linking "currents"
  at low `max_distance`) was that plain `linear_sum_assignment` returns a **complete**
  matching with no way to leave a detection unmatched. Fixed by porting the
  **Jaqaman et al. (2008) LAP** (u-track/TrackMate) augmented cost matrix with dummy
  birth/death nodes: `backend/celltracker/tracking.py::solve_lap(cost, no_match_cost)`
  — now **the shared assignment primitive for every linker** (centroid, topology,
  fingerprint, overlap); splits the gated cost matrix into connected components
  (exact, since a node's only non-linking option is its own dummy) to stay
  near-O(sum of small blocks) instead of O((N+M)³). New `METHOD_CT_OVERLAP = 'Cell-
  Tracker: Mask Overlap (IoU)'` — gates directly on IoU (`min_iou`), far more
  discriminative than centroid-only linking for slow dense nuclei; `track_overlap(df,
  masks, min_iou, max_gap, no_match_cost)`. Measured on 3 real monolayer TIFFs:
  old topology gave 861/489/2554 "implausible one-frame steps"; new topology-LAP gave
  711/443/1297; new overlap-IoU gave 7/3/31 (near-elimination). **Anti-pattern to
  recognize:** bare `linear_sum_assignment` for any many-to-many frame-linking problem
  silently forces complete matchings — always route through `solve_lap`.
  `widgets/common.py` `visible_when` extended to accept a **list** of values (row shows
  if current choice is any value in the list).
- **V1.52 — SerialTrack field bounded fix**: `scatter_to_grid` with any
  `smoothness > 0` routed through unbounded `RBFInterpolator(thin_plate_spline)`,
  which extrapolates without limit outside the convex hull — vectors up to ~134px
  appeared when real displacements were <21px. Fixed with a **bounded** variant
  matching CellTracker's approach: `scatter_to_grid_bounded`/
  `scatter_to_grid_multi_bounded` (`backend/serialtrack/regularization.py`) — griddata
  linear inside the convex hull + `nan_to_num` + Gaussian blur (sigma redefined in grid
  cells, not RBF weight). **Only the field-*output*/display path changed** — tracking's
  internal global-step solvers still use the original RBF path (smoothness there is a
  genuine iterative regularizer, not a display concern). **General lesson: prefer
  griddata(linear)+nan_to_num+blur for display fields; reserve RBF/thin-plate-spline
  for solver-internal regularization only** — this exact lesson recurs in the V1.46
  spatial-maps saga (§8).

### 5.2 Performance/UX fixes around tracking (all V1.46)

- **Tracking off the GUI thread**: both synchronous linker call sites moved to a
  `_TrackJob(AnalysisJob)` worker; in-place row mutation across the hand-off accepted
  as safe (only additive keys, individually atomic under the GIL).
- **Progress plumbing + vectorized speedups**: root cause of "no progress until it
  jumps to done" was `progress_cb` never being threaded through
  `link_objects_with_params → link_objects → _link_group_celltracker → track_timeseries`.
  Fixed + vectorized 3 of 4 hot spots (final track_id assignment via `df.apply`→list
  comprehension; O(T²) boolean masks → one `groupby('frame')`;
  `compute_topology_features`'s per-cell loop vectorized). **The Hungarian
  `linear_sum_assignment` step itself was explicitly left untouched** — it's inherent
  O(n³) cost; changing it changes results. `(fraction, message)` progress_cb convention
  is now the standard shape for any backend-pure long-running callable.
- **Dense-TIFF "tracks not computed" bug**: root cause was `compute_measurements`
  taking ~52s/frame (fed tracking, made it *look* hung) — a full-frame boolean scan per
  object for per-channel intensity. Fixed to a single labelled `scipy.ndimage.mean`/
  `standard_deviation` pass per channel, then index by label — ~52s→~6s/frame (~8x).
  **Reusable fix pattern**: avoid `frame_data[mask_frame==prop.label]` inside a
  per-object loop anywhere in the codebase; do one labelled pass instead. Secondary bug
  fixed same plan: `read_imagej_tiff_metadata` mishandled a bare-string ImageJ
  `'Labels'` value for single-channel files (`"H2B-iRFP670"[:1]` → `"H"`).
  **SerialTrack itself was NOT optimized** — inherently slow on thousands of dense
  objects; documented as a known limitation (prefer Centroid/Cell-Tracker linkers for
  dense fields).

### 5.3 SerialTrack PTV Analysis tab (V1.47)

`backend/serialtrack_analysis.py` (Qt-free) rebuilds displacement/strain/stress fields
**from persisted tracked measurement rows** — the existing tracker only uses SerialTrack
to chain `track_id` and **discards the `TrackingSession`** (phase P5 "session
persistence" not yet done as of this plan). D-agnostic (2D/3D, inferred from
`centroid_z_px` presence). Reuses `backend/serialtrack/fields.py`
(`DisplacementField`, `StrainField`) and `regularization.py::scatter_to_grid_multi` —
do not reimplement gridding/strain math. Stress + Von Mises ported from
`SerialTrack_Python/pages/stress_page.py`. `widgets/serialtrack_panel.py` mirrors
`SpatialMapsPanel`'s UI conventions. **`widgets/results/*_view.py` is confirmed dead
code with broken imports** — do not build on it.

---

## 6. DVC & ALDVC & MDM/surface (object-level 3D deformation)

### 6.1 DVC/ALDVC core (V1.51, fidelity hardened V1.55)

Clean-room port of FranckLab's ALDVC (Yang/Hazlett/Landauer/Franck 2020). Integrated as
a **Special hexagon node + companion viewer panel** (not `AnalysisPipeline` — dense
vector/tensor fields fit neither the mask/rows nor the `(T,H,W)→(T,H,W)` plugin
contract) — this is the template later reused for Registration.

- `core/dvc_registry.py::DVCResult`/`DVCParams`/`DVCMethod` (registry, Qt-free, mirrors
  the later SerialTrack/Registration registries). `DVCResult` fields: `grid_coords`
  `(*grid,d)` voxels, `displacement_field` `(*grid,d)` voxels (component 0 = z in 3D),
  `strain_field` `(*grid,d,d)` full rank-2 tensor, `qfactor` (ZNCC 0-1),
  `voxel_size_um` `(z,y,x)`. Axis convention: **slowest-first `(z,y,x)`**,
  `F[i,j]=∂u_i/∂x_j`.
- `backend/dvc/` package: `mesh.py` (subset grid + DOF pack/unpack, 0-based
  `order='F'`), `integer_search.py` (FFT seed), `icgn.py` (12-DOF local IC-GN, cached
  ref Hessian, ZNCC ddof=1), `outliers.py`, `global_step.py` (F-coupled augmented-
  Lagrangian FD solve), `admm.py` (outer loop ≤4 + dual updates), `strain.py`, `
  engine.py::run_aldvc(ref, def, voxel, params, progress_cb, cancelled_cb) -> DVCResult`.
- Reads the un-collapsed volume directly via `record._raw_volume.get_frame(c, m, t,
  z_mode='none')` — bypasses the Z-collapse recipe path entirely (degrades to 2D DIC
  with a status hint if the record is Z-collapsed). This accessor pattern (used ~14×) is
  the standard way to read raw Z volumes anywhere in the codebase.
- **V1.55 fidelity additions** (needed for large-deformation/time-series robustness on
  real 121GB confocal data): (A) `backend/dvc/tracking.py::accumulate_incremental` —
  Lagrangian point-tracking composes per-step increments into cumulative displacement
  (reusable "compose per-step field into cumulative field" pattern); (B) cross-frame
  warm-start — FFT-seed only frame 2, then every later frame warm-starts from the
  previous frame's solution (`u0_seed` param on `run_aldvc`); (C) coarse-to-fine
  multigrid FFT integer search (`integer_search_multigrid`) to bracket large
  displacement a single-scale search would miss entirely. Verified: multigrid recovers
  an 18-voxel shift a single-scale radius-6 search fails on (0.015 vs 15.5 voxel
  error).
- Third registry instance confirmed: registry → method subclass → Special hexagon node
  → off-thread `AnalysisJob` → viewer panel with overlay tab — **this is now THE
  template for adding any dense-field analysis as a node** (SerialTrack, DVC, later
  Registration).
- Deferred as of V1.55: FE global step, GPU IC-GN, `.nd2s` field persistence,
  process-pool fan-out (some later delivered — V1.51's own phase-6 addendum added
  multicore `n_workers` + gated GPU FFT seed).

### 6.2 DVC-on-object rendering (Phases 1–3, V1.65/V1.67/V1.68)

- **Phase 1 (V1.65, 3D Mask Drawing node)**: `special:mask3d` — draws/edits a 3D
  boolean mask over a wired channel's raw `(Z,H,W)` volume via 3 modes (Manual,
  Propagate across Z, Threshold seed+edit). Output port `PortType.ANY` (not BINARY —
  corrected for `can_connect` compatibility). `backend/analysis/mask3d.py`:
  `rasterize_plane`, `build_mask_volume(shapes_by_z, Z, H, W, propagate)`,
  `threshold_seed_shapes`. **Storage contract**: `record._mask3d_by_m = {m:{t:(Z,H,W)
  bool}}` — the durable store Phase 2 consumes, and later matched exactly by V1.70's
  granule masks so the same downstream machinery (scope lever, `iter_objects`,
  viz3d surface) works for free.
- **Phase 2 (V1.67, DVC-on-object voxel render)**: interpolates the sparse DVC grid
  onto the Phase-1 mask, renders surface + interior colored by a DVC scalar, in a new
  "3D Object" DVCPanel subtab, reusing `PyVista3DViewer`. Alignment done entirely in
  physical µm (grid_coords × voxel_size_um vs mask index × mask_voxel_size_um) —
  handles anisotropy/downsampling automatically. `backend/viz3d/overlays.py::
  MaskedField` + `dvc_object_field(result, mask, mask_voxel_size_um, ...)` via
  `scipy.interpolate.RegularGridInterpolator`, cropped to bbox, capped at
  `max_box_voxels=4e6`. New "overlay-only render path" added to `PyVista3DViewer`
  (`_render_overlay_only`) because it previously required a backing raw volume.
- **Phase 3 (V1.68, surface deformation metrics / MDM)** — **supersedes Phase 2's
  voxel-first render**: replaced with a smoothed closed-surface mesh (marching cubes +
  Taubin smoothing) carrying DVC displacement sampled ON the boundary (not interior),
  plus the **Stout et al. 2016 PNAS Mean Deformation Metrics (MDM)** suite and a 2D
  Mollweide/equirectangular UV-unwrap. `MaskedField`/`dvc_object_field` demoted to an
  optional `with_interior` helper only.
  - `backend/viz3d/surface.py::ObjectSurface` (vertices_um, faces, normals, face_areas,
    `enclosed_volume_um3` via divergence theorem, `centroid_um`, `scalars`);
    `build_object_surface(mask, voxel_size_um, *, smooth_iterations=10,
    taubin_lambda=0.5, taubin_mu=-0.53, step_size=1)`; `sample_displacement_on_surface`;
    `decompose_surface_displacement` (perpendicular/parallel components).
  - `backend/viz3d/mdm.py::MDMResult` (grad_u, F, J, R, U, stretches, principal_dirs,
    theta_deg); `mean_displacement_gradient` (Eq. 6: `(1/V)Σ_faces(ū_f⊗n_f)A_f`);
    `deformation_metrics` (polar decomposition F=WΣVᵀ⇒R=WVᵀ,U=VΣVᵀ,
    `cosθ=(tr R−1)/2`); `cumulative_rotation(thetas_deg, times_s)` — stateful,
    accumulated per-frame on the panel using `record._frame_timestamps`.
  - `dvc_object_surface(...)` is the new primary object adapter; `unwrap_surface(...,
    projection='mollweide'|'equirectangular')` is display-only (does not affect the 3D
    surface or MDM scalars).
  - Validated: 4 canonical cases from the paper (stretch, 45° rotation, simple shear,
    Eshelby inclusion) recover `⟨F⟩` to ~1e-12–1e-15.
  - **`backend/dvc/` is never imported/modified by any of this** — only `DVCResult` is
    consumed read-only; regression-checked via empty `git diff` under `backend/dvc/`.

---

## 7. Masks / objects / per-object scope

Cross-references §1.9 (Frame/Object scope toggle), §6.2 (mask3d + surface render), and
§11 (granule separation, which builds directly on this section's data shapes).

- `backend/analysis/mask3d.py` + `record._mask3d_by_m = {m:{t:(Z,H,W) bool}}` (V1.65)
  is the canonical single-object-mask storage shape.
- `backend/analysis/object_scope.py::ObjectRegion`/`iter_objects` (V1.68) is the
  generic per-object crop-region extractor, dispatching on dtype/source
  (`(Z,H,W)` bool → 3D connected components; `(T,H,W)` int32 → per-label bbox union
  over T; tracked objects → per-track bbox union over frames). Extended in V1.70 (P6)
  with `iter_objects_3d_labels` for a **combined** `(Z,H,W)` int32 label volume
  (treats it as one 3D region per label id via `scipy.ndimage.find_objects` — otherwise
  it would be misinterpreted as `(T,H,W)`).
- `executor.ObjectCropVolume(raw_volume, region, *, mask_out=False)` — the object-scope
  analogue of `CroppedVolume`; `mask_out=False` default preserves surrounding matrix
  context (needed by correlation-style nodes like DVC).
- **Review UI generalization** (V1.45): "Review Objects" node with two modes —
  `TrackValidationDialog` generalized via a `_ReviewUnit(uid, rows, track_id=None)`
  abstraction (one review unit per track, or per untracked object with
  `include_untracked=True`); `WholeFrameReviewDialog` (new) shows label ids across
  T/M with click-to-toggle-reject + whole-frame Accept/Reject, driven by a lazy
  per-M channel provider callable to bound memory. Both dialogs expose
  `rejected_rows`/`accepted_rows` (row-reference decisions — the established pattern
  across all review dialogs; safe to extend without breaking Run/preview apply paths).
  `WholeFrameReviewDialog` later (same version) gained a click-to-inspect corner inset
  (zoomed, always-centered crop that follows T/plays across a track's frames) reusing
  `analysis_page._overlay_labels(outline=True)`.

---

## 8. Results & spatial maps

- **V1.0 Results page**: grouped column-selector sidebar (`_COLUMN_GROUPS`),
  `_ResultsConfigDialog` two-step wizard, `results_config` persistence.
  `widgets/config_wizard.py::ConfigWizard`/`SavePreviewDialog` — reusable guided
  pick-then-diff-preview dialog pattern.
- **V1.22 Results tab**: `backend/results_engine.py::compute_measurements` (regionprops
  + per-channel intensity + absolute stage-µm centroids),
  `export_overlay_frames`/`export_label_masks_tiff`.
- **V1.26**: All-M support propagated through Results/Batch — `_channels_for_m(exp, m)`
  helper (volume-aware with flat-channel fallback) is the reusable idiom for any future
  per-M downstream consumer. **Gotcha carried forward**: `exp.analysis_results[name]`
  must be read defensively as either a bare `AnalysisResult` or `Dict[int,
  AnalysisResult]` (format changed in V1.25).
- **Interpolated Spatial Maps node** (V1.45): generic node interpolating any measurement
  column onto a grid per (M, frame) via `griddata`; `temporal_fill` (per-track
  `np.interp`, no extrapolation). Re-homed (with CellTracker's Spatial Field Maps) from
  `NodeCategory.SPECIAL` → `NodeCategory.RESULTS` since both are file-writing image
  sinks, not measurement ops — **the template for "image-producing sink = SPECIAL op +
  category override to RESULTS."**
- **Spatial Maps construction saga (V1.46, 3 sequential revisions — read carefully
  before touching `compute_spatial_fields`)**:
  1. First rewrite: binned Gaussian-weighted local mean (matching how cell density is
     computed) + a `cell_footprint` mask to stop values bleeding into empty space —
     **fully reverted the very next plan.**
  2. Reverted to the **original Cell-Tracker construction**: `scipy.interpolate.griddata`
     linear interpolation + `nan_to_num(fill=frame-mean)` + Gaussian blur — deliberately
     matches the sibling repo exactly, including the property that griddata fills the
     WHOLE convex hull with the frame mean outside the cell footprint (an accepted
     trade-off, not a bug). Needs ≥4 cells for scalar fields, >3 tracked cells for
     velocity. Reference files: `Cell-Tracker/CellTracker/backend/fields.py`,
     `pages/spatial_page.py`.
  3. **Converted to an interactive in-viewer tab** (both `special:ct_fields` and
     `special:interp_map` now open the SAME "Spatial Maps" tab instead of exporting to
     disk): `widgets/spatial_maps_panel.py::SpatialMapsPanel`, `set_compact(bool)`
     dual-sidebar reparenting (compact top-bar vs full vertical sidebar, same widgets),
     a global JSON template library `backend/spatial_templates.py`
     (`~/.nd2studios/spatial_map_templates`), `backend/celltracker_bridge.
     build_tracked_df(rows, m)` (rows → CellTracker DataFrame shape). Node params
     collapsed to a single hidden `templates: list` param — old saved graphs with
     legacy per-field params still load fine (extra keys harmless).
  - **Durable lesson from this saga** (also true of the SerialTrack V1.52 fix):
    a normalized local mean (sum/count) plateaus a cell's value flat across the whole
    smoothing kernel, bridging gaps between cells — check whether numerator and
    denominator share the same kernel before trusting a "smoothed ratio fades with
    distance" assumption.

---

## 9. Performance & resource-aware streaming

A 7-phase initiative (V1.33–V1.41) plus later hardening (V1.46, V1.66).

| Phase | Version | Delivered |
|---|---|---|
| 1 | V1.33 | Qt-free profiling harness `nd2studios/utils/profiling.py` + `profiling/harness/run_all <label>` → `profiling/baselines/<label>.json`. Real-code mapping table for future phases. |
| 2 | V1.34 | Gap audit found most lazy-loading infra already existed (V1.0/V1.17/V1.28). Added `utils/resources.py` (`SystemResources.detect()`, `recommended_cache_budget_bytes(reserve_fraction=0.6)`, `recommended_worker_count()`) and `utils/threading.py::IOWorker(QThread)` (own `volume.reopen()` handle; request-id staleness discard, not coord-equality). `FrameCache.max_bytes` now RAM-adaptive (was flat 300MB). |
| 3 | V1.35 | Pinning, velocity-biased prefetch, `CacheStats` — see §3.1. |
| 4 | V1.36 | pyqtgraph GPU display — see §3.1. |
| 5 | V1.37 | `nd2studios/compute/` job-runner layer: `CancellationToken` (threading.Event-backed, `is_cancelled()`/`check()`), `JobRunner` (QThreadPool sized by `recommended_worker_count()`, key-based job coalescing — submitting under an in-flight key cancels the prior job), `ResultStore`, `pipeline_jobs.py`. **Additive only** — `BaseWorker`/`AnalysisWorker` kept, only docstring-deprecated. `JobRunner`'s QThreadPool is the **only** place in the app using QThreadPool; everything else stays on QThread/BaseWorker. |
| 6 | V1.38 | Auto-managed on-disk workspace (`~/.nd2studios/workspace/sessions/<source_hash>/`), orthogonal to `.nd2s`. `PipelineStage` ABC (`commit`/`is_committed`/`release_in_memory`/`rehydrate`); `RecipeStage`/`AnalysisStage`; `storage.py::write_label_stack`/`read_label_stack` (Zarr preferred, NPZ fallback). Source hash = `sha256(first MiB + last MiB + size + mtime)[:16]` — reused verbatim by V1.39's pyramid stage. |
| 7 | V1.39 | Optional GPU acceleration (`compute/gpu/` — `cucim`/CuPy with CPU fallback, default-off, skip dispatch below 256×256) + multi-resolution Zarr pyramid (`PyramidStage`, levels 1–4 only, level 0 stays the live reader). `PyramidReader.pick_level_for_viewport` — reused directly by V1.42. |
| Addendum | V1.40 | Multi-core parallelism via `ThreadPoolExecutor` (GIL-releasing scipy/skimage hot loops) — NOT Dask/Numba/multiprocessing for compute. `_AtomicCounter` (10-line helper) for cross-thread progress counting — `done += 1` is **not** atomic once the GIL is released inside a C-extension call. `utils/storage.py::is_fast_storage`/`recommended_io_thread_count` (Linux-sysfs, non-Linux conservative fallback). |
| 8 | V1.41 | `utils/resource_strategy.py::LoadStrategy` (EAGER_FULL/EAGER_REDUCED/LAZY_CACHED) + `choose_strategy(meta, path)` decision matrix; `core/memory_monitor.py` singleton (warning 80%/critical 90%/emergency 95%, hysteresis bands); `utils/progress.py::FrameProgress` (accurate T×M×Z×C progress, ≤101 emissions via 1%-threshold gating — the general progress contract going forward). Streaming exporters (per-frame TiffWriter/imageio writes, RAM peak drops to single-frame). **Explicit rejection**: mid-session eager→lazy demotion (reference-tearing risk with in-flight workers) — modal-block is the only sanctioned mechanism. |

### Later hardening

- **V1.46 "Restore CellTracker segmentation speed"** — see §4.2/§4.3.
- **V1.46 "Streaming, resource-aware analysis"**: `LabelStackWriter`
  (`pipeline/storage.py`) — Zarr `(1,H,W)`-chunked or memmap `.npy` random-access
  streaming label sink; `should_stream_analysis(volume, *, decision, monitor, force)`
  adaptive gate; `backend/analysis/plane_runner.py::run_planes_to_labels`. **Never call
  `write_label_stack` with `np.ascontiguousarray` on a value that might already be a
  memmap/lazy reader** — silently re-materializes the whole stack, defeating streaming
  (the exact bug this plan fixed in `analysis_stage.py`'s `commit_m`).
- **V1.66 "Streaming / memory-mapped loader" (status: Implemented core)** — the
  culminating fix. Root cause: since V1.41 the loader eagerly built the **entire**
  `(M,T,Z,H,W)` array per channel regardless of the LAZY_CACHED strategy (only honored
  for single-file ND2; TIFF/multi-file always materialized). Measured on a real
  56.25GB BigTIFF: eager = ~5min + needs 56GB RAM; streaming = ~20ms open, 0.17s cold
  frame then cached instant, peak RSS <1GB.
  - `backend/streaming_dataset.py::StreamingDataset` — implements the **exact**
    `MaterializedDataset` public surface (`get_frame`, `get_volume`, `to_lazy_channel`,
    `all_channels_as_lazy`, `channel_array`, `subset`, `reopen`, `close`, shape/n_*/
    dtype/pixel_size_um/z_step_um/z_mode) — a pure drop-in requiring **zero** downstream
    changes. `TiffMemmapSource` (true OS-paged mmap via `tifffile.memmap(mode='r')` for
    uncompressed/contiguous TIFF, falls back to per-page read for compressed/tiled).
    `LazyVolumeSource` wraps any lazy volume exposing `get_volume` (falls back to
    stacking per-plane `get_frame` reads for composites lacking `get_volume`).
  - Bounded byte-capped LRU frame cache + single background prefetch thread with its
    own `reopen()`'d source.
  - Blocked Z-projection (max/mean/min over ~256MB Z-blocks) — verified byte-identical
    to full-materialization.
  - `Settings.STREAM_ALWAYS = True` (default) — **every** file now gets `LAZY_CACHED`.
    Eager load is opt-in only via `FORCED_LOAD_STRATEGY='eager_full'`.
    `EAGER_MAX_BYTES_DEFAULT=2GB` absolute cap replaces the old 50%-of-RAM fraction.
  - **View streams; compute materializes when it fits, streams when it doesn't**:
    `fits_resident(volume)` (RAM-relative check, ignores the small absolute view-open
    cap) drives `should_stream_analysis`/`materialize_channels_if_fits`. MemoryMonitor
    CRITICAL+ pressure forces streaming regardless of size.
  - `Settings.DISPLAY_PREVIEW_Z_PLANES` (default 0 = exact) — optional bounded
    Z-subsampled preview for fast display (4.2× faster) while `to_lazy_channel`/
    `get_volume` (recipe/export/DVC) stay exact.
  - **Reusable pattern**: "drop-in replacement implementing the exact same public
    surface as the thing it replaces" (mirrors `RegisteredFrameVolume` mirroring
    `ProcessedFrameVolume`) — zero downstream call-site changes needed.

---

## 10. Registration & deconvolution

### 10.1 Image Registration (V1.56, hardened V1.59/V1.60)

Two integration paths sharing one pure engine (`backend/registration/estimate.py`):
(1) a Processing-stage `EnhancementPlugin` (per-channel temporal drift stabilization,
drops into the recipe/Pipelines palette), (2) a cross-channel
`RegistrationMethod`/Special hexagon node ("register once on a reference channel, apply
to all channels") — the **3rd instance** of the registry→method→special-node→job→panel
template (after SerialTrack, DVC).

- `estimate_translation(reference, moving, upsample, highpass_sigma) -> (shift, ncc)`
  via `skimage.registration.phase_cross_correlation`; `apply_shift`; `ecc_align`
  (`cv2.findTransformECC`, translation/euclidean/affine/homography, seeded from
  phase-correlation); `apply_warp`; `stabilize(volume, model, reference, ...)` —
  reference modes `'first'`/`'previous'`/`'mean'`.
- `core/registration_registry.py::RegistrationMethod`/`RegistrationResult` (mirrors
  `core/dvc_registry.py` exactly).
- `estimate_series`/`apply_series` — per-frame **absolute effective transforms**
  (translation shifts or composed ECC affine warps); the "apply to all channels" half.
- `record._registration_by_m` (per-M transforms), `record._registration_crop` — storage
  keys later read/frozen by the Checkpoint node (§1.6) and preview/base-image code.
- **V1.60 late-frame robustness** (root causes: no confidence gating, 'previous'-mode
  drift accumulation, photobleaching, phase-correlation wraparound):
  `disambiguate=True` on `phase_cross_correlation`; hold-last-good gating below
  `min_confidence` (reuse previous good transform, or skip increment for 'previous'
  mode, instead of snapping to identity); new `reference='template'` two-pass anchor
  (cheap 'previous'-pass rough-stabilize → average → register every frame to that
  averaged template) — new default, with `min_confidence=0.2` and a `normalize`
  param (e.g. `'zscore'` for photobleaching). Reproduction: late-frame error 285px →
  10px. Also added ROI-based (rectangle bbox crop or freeform masked phase
  correlation) and ORB+RANSAC feature-based registration models, plus
  `crop_to_common` (`estimate.common_translation_crop` — rectangle intersection of
  per-frame valid footprints across all frames of one M, published as
  `record._registration_crop`), and a memory-budgeted LRU cache (512MB) on
  `RegisteredFrameVolume` fixing slow registered playback.
- **V1.59 "Pause-node crop preview on the registered image"**: fixed 3 gaps — preview
  readers never applied registration at all; arming Preview Crop reverted to the
  un-registered base image (destroying the registration writeback); crop-then-register
  ordering was backwards. Fix established **register-then-crop** as the standing
  convention (see §1.3). `estimate.apply_frame` (single-frame analogue of
  `apply_series`) + `executor.RegisteredFrameVolume` (lazy per-(m,t) wrapper,
  `bypass_pyramid=True`) are the reusable single-frame/lazy-volume primitives.
  **Scope explicitly preview-only** — a resumed Run still ignores any preview crop, to
  avoid mixing full-frame-frozen-state with cropped downstream size.

### 10.2 Deconvolution (V1.63 — status: Proposed, verify current implementation state)

Fourth registry (`core/deconvolution_registry.py::DeconvolutionMethod`/
`DeconvolutionResult`, mirrors `DVCMethod` byte-for-byte). Dimension-agnostic
Richardson-Lucy engine over `d∈{2,3}` reading raw un-collapsed `(Z,H,W)` volumes,
streamed per-`(t,c,m)` via `get_volume(c,m,t)` (never materializes the whole
`(T,Z,H,W)` stack). PSF via Gibson-Lanni (depth-dependent) or Born-Wolf fallback, or
an imported measured PSF. FFT-based convolution (`scipy.fft`), optional CuPy GPU path
gated by `find_spec`. Output modes: `'project'` (default, Z-project back into the
normal `(T,H,W)` stream — the key bridge letting a 3D-first feature coexist with the
Z-collapsing pipeline) vs `'volume'` (keep full `(T,Z,H,W)`). Nikon Table A
auto-iteration caps reproduced as defaults. Only Richardson-Lucy is deeply built;
Landweber/Wiener/Blind/MAP are registered stubs. **Status check required** — this plan
is marked Proposed, not confirmed shipped.

---

## 11. V1.70 — Granule Separation (P0–P7)

A 5-node chain (bead detect → cluster → tessellate → mask → boundary) separating a 3D
bead point cloud into hydrogel granules, deliberately designed to **reuse the entire
V1.65–V1.68 mask/object/viewer/DVC stack for free** by matching data shapes exactly.

| Sub-plan | Node | Delivers |
|---|---|---|
| P0 | (contracts only) | Shared data contracts every other sub-plan implements against. Artifacts flow via `record._granule_*_by_m` (like `_mask3d_by_m`), not edge payloads — ports exist only for `can_connect` wiring validation. **Coordinate order pinned to `(z,y,x)`** (matches app-wide measurement/track convention); mesh vertices pinned to world `(x,y,z)` µm (matches viz3d convention); voxel size pinned `(dz,dy,dx)` µm everywhere. `backend/analysis/granule_types.py`: `GranuleBoundary`, `GranuleTessellation`. `pipeline_graph/granule_ops.py`: the 5 op-key constants (imported, never re-hardcoded, by `registry_adapter.py`/`pipelines_page.py`). |
| P1 | `special:bead_detect` | Reuses `backend/serialtrack/detection.py::ParticleDetector.detect` (LoG/CC + radial-symmetry sub-voxel refinement); **transposes detector-native `(x,y,z)` → `(z,y,x)` — the ONLY place in the app that performs this transpose.** Publishes `(N,3)` voxel coords + DATA rows carrying the app's **first 3D-centroid field**, `centroid_z_px`. |
| P2 | `special:granule_cluster` | Full-covariance GaussianMixture (scikit-learn, optional/lazy via `find_spec`, never in `requirements.txt`), user-seeded `n_granules` relaxed ±p%, actual k chosen by BIC sweep; KMeans fallback for sparse/unstable clouds. Points scaled to isotropic µm before fitting so anisotropic Z doesn't bias the fit. |
| P3 | `special:granule_tessellate` | Per-granule alpha-shape (Delaunay + circumradius filter, concave hull) OR global Voronoi; **owns the density-merge** (region-adjacency-graph, union-find, iterative merge of adjacent granules within `merge_tol` density ratio) — its `point_labels` become the FINAL granule assignment (not P2's initial GMM labels). |
| P4 | `special:granule_mask` | Voxelizes each granule's boundary with SDF-Gaussian smoothing (`distance_transform_edt` then threshold — non-shrinking, smooth-surface, reuses the `mask3d.py` idiom). **Publishes `record._granule_masks_by_m[m][t]` with the IDENTICAL value type `{granule_id:(Z,H,W) bool}` as a single `_mask3d_by_m` object** — the deliberate reuse trick: `iter_objects`, the V1.68 scope lever → per-object DVC, and `viz3d.surface.build_object_surface` all consume it with **zero changes** to those modules. Also publishes a combined `(Z,H,W)` int32 label volume. |
| P5 | `special:granule_boundary` | Extracts an outward boundary band (dilation or EDT-metric method, honors anisotropic voxel size) around each granule for correlating surrounding-matrix data. |
| P6 | (graph integration, shared files, done once/last) | The **only** sub-plan touching `registry_adapter.py`/`pipelines_page.py` (deliberately serialized last to avoid merge conflicts). Widens `node_scene._apply_edge_scope_lever` to gate on **source** `node_produces_objects` generally (was DVC-destination-only). Adds `iter_objects_3d_labels` to `object_scope.py` for the combined int32 volume. |
| P7 | (viewer path + docs) | "Node 4 / View granule in 3-D" is **not a new pipeline node** — a reused view path on the existing PyVista stack (same "View in 3-D" button pattern as DVC/PTV). Originates this very CODE_HISTORY.md compaction task. |

**Payoff of the P0/P4 data-shape-matching decision**: when a `granule_mask` feeds DVC
with `scope=objects`, `_dvc_scoped_object_regions` already calls `iter_objects` on the
per-granule bool masks, yielding one DVC field + surface + MDM **per granule** with
zero new DVC code.

---

## 12. Chronology (V1.0 → V1.70)

| Version | Feature(s) | Significance |
|---|---|---|
| V1.0 | Initial scaffold | Layered core/widgets/backend/plugins/workers/pages architecture; ND2 import, trial/accept/reject recipe, TIFF/composite/movie export. Sibling-not-fork of CellTracker. |
| V1.0 | Macro Recorder | Action-level macro record/replay across Recipe/Analysis/Export; `.nd2s_macro.json`. |
| V1.0 | Multi-file Recipe page + Play All + symbol fixes | Fixed invisible Unicode glyphs on Windows (use ASCII); multi-file before/after columns. |
| V1.0 | Multi-file side-by-side Import view | `FilePanel` widget; multiple ND2/TIFF files viewable simultaneously. |
| V1.0 | Overlay performance cache | `_pp_cache` tier on top of the render cache; instant T-scrub with overlays. |
| V1.0 | Results page column selector + config dialog | Grouped sidebar, `ConfigWizard`/`SavePreviewDialog`. |
| V1.0 | Smooth playback: pre-render cache | `PreRenderWorker` + 3-tier display fast-path; up to 60fps. |
| V1.0 | Fix black stitch output | Removed silent except-pass; fixed invalid `resolutionunit`. |
| V1.0 | Tracked object validation | Hungarian cross-frame linker + `TrackValidationDialog`. |
| V1.1 | M/Z scrolling, LUT histogram, M stitching | `LazyND2Volume`, `MultiAxisViewer`, first stitch dialog. |
| V1.2 | Collapsible LUT sidebar + tile preview | Corner `TilePreviewWidget` (superseded V1.3). |
| V1.3 | Interactive tile layout | `TileLayoutWidget` navigate/select modes replace corner preview. |
| V1.4 | Per-M stage XY bugfix | `_flat_index` — fixed misread of `frame_metadata` indexing. |
| V1.5 | Flush-grid tile layout | `_cluster_axis` sweep-gap clustering (superseded V1.6). |
| V1.6 | Adaptive clustering tolerance + scrollable tile widget | Data-derived tolerance; fixed 240px height removed. |
| V1.7 | Duplicate-position expansion | Horizontal sub-columns for revisits (superseded V1.8). |
| V1.8 | Serpentine layout | Boustrophedon scan-order for revisits (superseded V1.12). |
| V1.9 | Frame-shape normalization | `_normalize_to_2d` — tolerate RGB/hyperstack/non-2D frames. |
| V1.10 | Sparse-grid detection | `fill_ratio` trigger extends serpentine to irregular ROI scans. |
| V1.11 | Partial-metadata tolerance | Truncate `m_indices` to `stage_xy_um` length — preserved through all later rewrites. |
| V1.12 | Pure physical tile layout | No clustering/serpentine (superseded V1.13). |
| V1.13 | Row-flush layout | Compresses gaps (reverted next version). |
| V1.14 | Physical rebuild + `f.experiment` stage source | "Authoritative" until V1.54. |
| V1.15 | TIFF bit-depth fix | LUT auto-percentile guard; stitch multi-channel no longer forced 8-bit RGB. |
| V1.16 | Stitched TIFF multi-channel reload | ImageJ hyperstack metadata now read on import. |
| V1.17 | Frame prefetch cache | `FrameCache` + `PrefetchManager(QThread)`, ±5 neighbor prefetch. |
| V1.18 | Non-destructive crop | `_original_raw_channels` moved to `ND2StudiosRecord`. |
| V1.19 | Analysis tab + Cellpose | `AnalysisPipeline` registry — the extension point for all pipelines. |
| V1.19 | Tear/dark-region detection | Classical rank-normalized baseline; dynamic CSV fieldnames contract. |
| V1.20 | Histogram Threshold Segmenter | `backend/analysis/histothresh/` subpackage template. |
| V1.21 | Bright & Dark Spots (GA3-style) | `backend/analysis/spots/` subpackage, reuses histothresh histogram. |
| V1.22 | Results tab | `results_engine.compute_measurements`, overlay/mask export. |
| V1.23 | Batch tab | `.nd2st.json` template, sequential per-file `BatchWorker`. |
| V1.24 | Background mask overlay for Spots | `secondary_label_masks` generic `AnalysisResult` extension. |
| V1.25 | Analysis: all-M run, per-M overlay, hide toggle | `Dict[int, AnalysisResult]` per-M storage format. |
| V1.26 | Results & Batch: all-M support | `_channels_for_m` helper; defensive per-M unwrap everywhere. |
| V1.27 | Multi-file Z-stack import | `LazyMultiFileND2Volume` (Z-only, superseded V1.28). |
| V1.28 | Universal multi-file reconstruction | `chain_axis` (T/M/Z/C) generalizes V1.27; `ReconstructDialog`. |
| V1.29 | Unify Z-projection + stitched TIFF export | `export_tiff_hyperstack` single writer for both paths. |
| V1.30 | Manual mask drawing pipeline | `ManualMaskPipeline`; `"hidden"` ParamSpec type; ΔArea columns. |
| V1.30 | Preserve Z-stacks through stitched exporter | `z_mode='none'` now keeps all Z planes; TIFF Z-stride fix. |
| V1.31 | Reconstruct block UI + chain mapping | Drag-reorderable `AxisBlocksWidget`, `chain_mapping`. |
| V1.32 | Export preview dialog + image-sequence exporter | `ImageAdjustments` pipeline; `ExportPreviewDialog`. |
| V1.33 | Phase 1: profiling harness | `utils/profiling.py`; `profiling/harness/run_all`. |
| V1.34 | Phase 2: lazy loading & threading | `utils/resources.py`, `IOWorker`; RAM-adaptive cache budget. |
| V1.35 | Phase 3: velocity-aware prefetch, pinning | `CacheStats`, pin/unpin, direction-biased prefetch window. |
| V1.36 | Phase 4: pyqtgraph GPU display | `GpuImageCanvas`, feature-flagged, per-instance fallback. |
| V1.37 | Phase 5: background analysis with cancellation | `nd2studios/compute/` job-runner layer, `JobRunner`. |
| V1.38 | Phase 6: pipeline workspace | On-disk `~/.nd2studios/workspace/`, `PipelineStage` ABC. |
| V1.39 | Phase 7: GPU accel & pyramids | `compute/gpu/`, `PyramidStage` (levels 1-4). |
| V1.40 | Mask Analysis rename + custom mask creator | Manual Mask → Mask Analysis; tiled label boxes. |
| V1.40 | Phase addendum: multi-core parallelism | `ThreadPoolExecutor` hot loops; `_AtomicCounter`. |
| V1.41 | Pipeline smoothness + resource-aware loading | `resource_strategy.choose_strategy`, `memory_monitor.py`, `FrameProgress`. |
| V1.42 | Viewer optimizations (industry comparison) | LUT-only cache bypass, `.nd2idx.json` sidecar, request queue, LRU. |
| V1.43 | Tile-strip frame navigator | `FrameStrip`; `MaterializedDataset.subset()`, `CropWorker`. |
| V1.44 | GUI overhaul + icon revamp | Top tab bar replaces sidebar; `icon_button.py` house convention. |
| V1.45 | CellTracker integration | Vendored tracking/metrics/fields; Track Objects tracking method. |
| V1.45 | Merged Analysis+Results, if-else, SPECIAL nodes, GraphRunner | Node-graph core: category/shape enums, `GraphRunner`, condition DSL. |
| V1.45 | Pipelines tab (GA3-style node graph) | Processing MVP → full Analysis/Results/Export wiring. |
| V1.45 | Interpolated Spatial Maps node | Generic griddata-interp node; SPECIAL→RESULTS re-home pattern. |
| V1.45 | Whole-frame review corner inset | Click-to-inspect zoomed per-cell crop in review dialog. |
| V1.45 | Review Objects node | Single-objects vs whole-frame review modes. |
| V1.45 | SerialTrack tracking method | Vendored headless subset; Incremental/Cumulative modes. |
| V1.45 | StarDist segmentation node | Vendored, optional dep; outline overlay mode. |
| V1.45 | Track Objects node refined | Distance/size/gap thresholds, moving reference centroid. |
| V1.46 | Pipelines viewer overhaul | Overlay tab bar, live per-frame streaming, track/vector overlays. |
| V1.46 | Node graph authoritative for recipe | Leaving Processing tab re-syncs `record.recipe` from the graph. |
| V1.46.3 | Viewer/plots pop-out | `PopOutWindow` reparents live widgets into standalone windows. |
| V1.46 | Track-persistence rework + terminal Dismiss | Per-track frame-count filter; Dismiss now a real terminal discard. |
| V1.46 | Silent skip / swallowed error diagnostics | Every guard bail-out now emits a status message. |
| V1.46 | Preview Crop + full preview coverage | `CroppedVolume`; crop-aware plots/nodes. |
| V1.46 | Fix preview plots & overlay tabs | Shared `_set_track_overlay_state`; async preview parity with Run. |
| V1.46 | Restore CellTracker segmentation speed | `filter_and_relabel` (~430x); TF single-thread cap reaffirmed. |
| V1.46 | Spatial maps: binned construction (superseded) | Reverted next plan. |
| V1.46 | Restore original CellTracker spatial-map construction | griddata + nan_to_num + blur, matches sibling repo. |
| V1.46 | Spatial Maps → interactive in-viewer tab | Shared tab for both spatial-map node types; template library. |
| V1.46 | StarDist multiprocessing | `mp_stardist.py`, ProcessPoolExecutor fan-out. |
| V1.46 | Streaming, resource-aware analysis | `LabelStackWriter`, `should_stream_analysis`, `plane_runner.py`. |
| V1.46 | Tracking off the GUI thread | `_TrackJob(AnalysisJob)` for both linker call sites. |
| V1.46 | Track Objects progress plumbing + speedups | `(fraction,message)` progress_cb convention; vectorized hot loops. |
| V1.46 | Dense-nuclei TIFF "tracks not computed" fix | Vectorized `compute_measurements` intensity extraction (~8x). |
| V1.47 | SerialTrack PTV plots & Analysis tab | `serialtrack_analysis.py` rebuilds fields from tracked rows. |
| V1.48 | Channel-wire pipeline model + M/T bugfix | Channels become graph wiring; `_processed_channels_for_m` bugfix. |
| V1.49 | Loop/iteration connector | Amber back-edge, parameter sweeps, scoped sub-`GraphRunner`. |
| V1.50 | Track displacement/motility measure basis | Net/Cumulative/Per-frame basis; per-frame outlier scrubbing. |
| V1.51 | DVC/ALDVC node | Clean-room ALDVC port as a Special node + panel. |
| V1.52 | SerialTrack field bounded interpolation | Fixed unbounded RBF extrapolation bug. |
| V1.53 | Checkpoint node | Freeze/resume via upstream-subgraph hash. |
| V1.54 | Multipoint stitching pipeline rebuild | Regime-aware `backend/stitch/` package; supersedes V1.1-V1.14. |
| V1.55 | DVC/ALDVC fidelity | Incremental accumulation, warm-start, multigrid seed. |
| V1.56 | Image Registration | Translation/ECC stabilization; Processing plugin + Special node. |
| V1.57 | Export cropped data from Import tab | XY ROI crop + T/M/Z crop export, reuses `ExportWorker`. |
| V1.58 | Tracking: LAP birth/death + mask-overlap linker | `solve_lap` — shared assignment primitive for all linkers. |
| V1.59 | Cropped resume from Checkpoint | Crop dropped from hash; registration state frozen/restored. |
| V1.59 | Pause-node crop preview on registered image | Register-then-crop convention established. |
| V1.60 | Registration late-frame robustness + ROI/feature models | Hold-last-good gating, template reference, ORB+RANSAC. |
| V1.61 | Pipelines overhaul master plan (R1,R2,R4,R5) | Merged scene, schema v4→v6, bridges removed. |
| V1.62 | Multi-input files + channels on node body (R3/R8/R9 phase 2) | Per-file input nodes bound by `exp_id`; focused-input resolution. |
| V1.63 | Deconvolution (proposed) | 4th registry; dimension-agnostic Richardson-Lucy engine. |
| V1.64 | Dynamic screen-size text scaling | `screen_scale()` folded into `ui_scale()`; QSS scaling pass. |
| V1.65 | 3D Mask Drawing node (Phase 1, DVC-on-object) | `special:mask3d`; `record._mask3d_by_m` storage contract. |
| V1.65 | PyVista 3D viewer | `PyVista3DViewer`, off-screen VTK, 4 render modes. |
| V1.66 | Streaming/memory-mapped loader | `StreamingDataset` drop-in; fixes multi-minute freezes/OOM. |
| V1.67 | DVC-on-object 3D render (Phase 2) | Voxel-first render (superseded by V1.68 surface-first). |
| V1.68 | DVC surface deformation metrics / MDM (Phase 3) | Marching-cubes surface + Stout 2016 MDM + UV-unwrap. |
| V1.68 | Frame/Object scope toggle | Per-edge scope lever; N-object crop generalization, wired for DVC. |
| V1.70 | Granule Separation (P0-P7) | 5-node bead→cluster→tessellate→mask→boundary chain; reuses mask3d/DVC/viz3d stack. |

---

## 13. Key reusable primitives index

Flat lookup of the most-reused functions/classes/dataclasses/record-keys, with file
paths, for fast reuse when building new features.

### Graph model & execution
- `GraphSlice.structural_edges()`/`structural_incoming()`/`structural_outgoing()`/
  `loop_edges()` — `pipeline_graph/model.py`. Use for ANY traversal that must ignore
  loop back-edges.
- `GraphRunner(sl, frozen=set())` — `pipeline_graph/executor.py`. Skip-already-computed
  nodes while keeping downstream gating correct.
- `recipe_for_node`, `apply_recipe`, `channel_sets`, `channel_recipes`,
  `edge_channels` — `pipeline_graph/executor.py`.
- `ProcessedFrameVolume` / `PinnedProcessedVolume` / `CroppedVolume` /
  `RegisteredFrameVolume` / `ObjectCropVolume` — `pipeline_graph/executor.py` (except
  the last, `backend/analysis/object_scope.py`-adjacent). All mirror the base volume's
  `get_frame`/shape/`channel_names` protocol — the standing "lazy wrapper" template.
- `partition_rows(cond, rows, group_by)` + `LENS_FRAME`/`LENS_OBJECT` —
  `pipeline_graph/conditions.py`. Per-object vs per-frame branching, generalizable
  beyond If/Else.
- `_processed_channels_for_m(record, m)` — `pages/pipelines_page.py`. The ONLY correct
  way to get genuinely per-M, all-T channel data; never read
  `record._raw_channels`/`processed_view()` directly for per-M correctness.
- `op_produces_objects(op_key)` — `pipeline_graph/registry_adapter.py`. Gates the
  Frame/Object scope lever.

### Tracking / linking
- `solve_lap(cost, no_match_cost)` — `backend/celltracker/tracking.py`. Jaqaman
  augmented-assignment; THE shared linker primitive (centroid/topology/fingerprint/
  overlap). Never call bare `linear_sum_assignment` for many-to-many frame linking.
- `link_objects_with_params(rows, params, pixel_size_um)` — `backend/object_tracker.py`.
  Canonical Qt-free param-translation entry point for any tracking call site.
- `scatter_to_grid_bounded`/`scatter_to_grid_multi_bounded` —
  `backend/serialtrack/regularization.py`. Bounded (griddata+blur) field display —
  prefer over RBF/thin-plate-spline for any display field.
- `backend/track_overlays.py` — Qt-free colormap/overlay/vector-drawing primitives.

### Streaming / performance
- `StreamingDataset` (+ `TiffMemmapSource`, `LazyVolumeSource`) —
  `backend/streaming_dataset.py`. Drop-in `MaterializedDataset`-surface streaming
  reader; default for ALL loads since V1.66.
- `recommended_cache_budget_bytes()`, `recommended_worker_count()`,
  `recommended_process_count()` — `utils/resources.py`. Shared RAM/core sizing.
- `fits_resident(volume)`, `should_stream_analysis(...)`,
  `materialize_channels_if_fits(...)` — `utils/resource_strategy.py`. The
  view-streams/compute-materializes-when-it-fits rule.
- `FrameProgress` / `total_frames_for(meta,...)` — `utils/progress.py`. Standard
  T×M×Z×C-accurate progress contract, ≤101 emissions.
- `JobRunner` (`compute/runner.py`) — key-coalescing off-thread job submission; the
  only QThreadPool user in the app.
- `LabelStackWriter` — `pipeline/storage.py`. Standard streaming label sink.
- `filter_and_relabel(mask, min_area, max_area)` — `backend/analysis/source_utils.py`.
  Vectorized O(pixels) area-filter + relabel.

### Volume / metadata access
- `volume.reopen()` (on any Lazy*Volume) — required for every new worker thread/process
  reading an nd2/TIFF file; nd2/TIFF handles are not thread-safe.
- `record._raw_volume.get_frame(c, m, t, z_mode='none')` /
  `.get_volume(c, m, t, z_start, z_end)` — the standard un-collapsed-Z read path,
  reused across DVC/registration/deconvolution/granule/PyVista.
- `_normalize_to_2d(frame)` — `widgets/multi_axis_viewer.py`. Coerce arbitrary loader
  output to displayable 2D grayscale.

### Object / mask primitives
- `record._mask3d_by_m = {m:{t:(Z,H,W) bool}}` and its granule analogue
  `record._granule_masks_by_m` (identical value shape by design) —
  `backend/analysis/mask3d.py`, `backend/analysis/granule_mask.py`.
- `object_scope.iter_objects(obj, voxel_size_um, ...)` /
  `iter_objects_3d_labels(...)` — `backend/analysis/object_scope.py`. Generic
  per-object crop-region extractor.
- `build_object_surface(mask, voxel_size_um, ...)` → `ObjectSurface` —
  `backend/viz3d/surface.py`. Turn any labeled 3D mask into a smoothed renderable
  surface (marching cubes + Taubin).
- `dvc_object_surface(result, mask, mask_voxel_size_um, ...)` → `SurfaceField` —
  `backend/viz3d/overlays.py`. Sample any DVC/derived scalar onto a surface.

### Dialogs / UI patterns
- `icon_button()`/`make_icon()`/`bind_toggle_icon()`/`ui_scale()`/`scaled(px)` —
  `widgets/icon_button.py`. Standing convention for all buttons/icons since V1.44.
- `PopOutWindow` — `widgets/popout_window.py`. Reparent-live-widget maximize/restore.
- `ConfigWizard`/`SavePreviewDialog` — `widgets/config_wizard.py`. Guided
  pick-then-diff-preview.
- `FrameStrip` — `widgets/frame_strip.py`. Single-paint-surface tile navigator.
- `TileLayoutWidget`/`compute_tile_layout()` — `widgets/tile_layout.py` /
  `backend/stitch/positions.py`. Shared geometry source of truth for any tile
  navigate/select/stitch-preview UI.

### Registries (the "registry → method → Special node → job → panel" template)
Established 4× — reuse this shape for any future dense/cross-channel analysis feature:
`core/dvc_registry.py` (V1.51), `core/registration_registry.py` (V1.56),
`core/deconvolution_registry.py` (V1.63, proposed), plus the SerialTrack vendored
subset (`backend/serialtrack/`, V1.45) as the pattern's origin.
