# ND2Studios Architecture

**Version:** V1.46

## V1.46 additions (Streaming, Resource-Aware Analysis)

- **`backend/analysis/plane_runner.py`** — `run_planes_to_labels`, the shared
  per-frame analysis driver. Routes each label frame to an in-RAM array (eager)
  or a streaming disk sink (lazy), keeping only measurement rows in RAM.
  `backend/analysis/source_utils.py` provides lazy-or-eager frame access
  (`source_shape`, `read_plane`).
- **`backend/analysis/mp_stardist.py`** — `run_stardist_multiprocess`, a
  process-parallel StarDist runner for many-core CPUs. StarDist/TF share a
  non-thread-safe model (so thread parallelism can't help); instead N worker
  processes (`spawn`) each load their own model and segment frames concurrently.
  The parent reads the lazy source one frame at a time, ships ndarrays to workers,
  and remains the sole writer of the `plane_runner` label sink; a bounded
  ~2×workers in-flight window preserves the streaming RAM bound. The StarDist
  node picks sequential vs. MP: GPU → 1 (MP would only contend over one card),
  else `n_processes` param or `resources.recommended_process_count()`
  (min(cores, free-RAM ÷ ~2 GB, 8)); `n_workers <= 1`/tiny stacks keep the
  thread path. Returns the same `PlaneRunOutput`, so output is identical to
  sequential — only where per-frame work runs changes.
- **`pipeline/storage.py:LabelStackWriter`** — incremental per-frame label sink
  (zarr `(1,H,W)` chunks, else memmap `.npy`). `AnalysisStage.commit_m` persists
  frame-by-frame and swaps results to the canonical readers.
- **`utils/resource_strategy.should_stream_analysis`** — adaptive eager/stream
  gate (dataset type / `StrategyDecision` / `MemoryMonitor` band) consumed by the
  Analysis, Results, and Batch compute paths, which now keep channels lazy.
- The four analysis pipelines and `results_engine.compute_measurements` iterate
  frames on demand; `AnalysisResult.label_masks` values may be disk-backed lazy
  readers. New flag `AnalysisPipeline.needs_full_stack`.

## V1.45 additions (Pipelines tab — GA3 node graph)

A new top-level **Pipelines** page fuses Recipe → Analysis → Results into one
node-graph editor modeled on NIS-Elements General Analysis 3. All three sub-tabs
are wired: **Processing** (branching DAG → recipe), **Analysis** (pipeline →
overlay + commit), and **Results** (preview an analysis result as a viewer
overlay + a sortable measurements table). The chain ends at the **Export** tab,
which surfaces every committed analysis result as a **"Pipeline Results"**
export source (measurements CSV / overlay frames / label-mask TIFF via
`results_engine`). The remaining gaps are multi-M analysis commit and Undo.

- **`pipeline_graph/` (new, Qt-free)** — the graph core, kept pure so it stays
  headless-testable (backend-purity rule extended to the graph layer):
  - `model.py` — `PortType` / `Stage` / `NodeRole` enums; `Port` / `Node` /
    `Edge` / `GraphSlice` / `Bridge` / `PipelineDoc` dataclasses with
    `to_dict`/`from_dict`; connection rules `can_connect` + `would_create_cycle`
    (DAG enforcement); `clone_node`.
  - `registry_adapter.py` — enumerates `PluginBase.get_plugins("enhancement")`
    **and** `AnalysisPipeline.get_pipelines()` into `NodeSpec`s at runtime
    (never hardcodes names; `enhancement_specs`/`analysis_specs`); `build_node`,
    `param_specs_for` (resolves `enhancement:` and `analysis:` op-keys),
    `default_params_for`, processing + analysis input/output node specs.
  - `executor.py` — `recipe_for_node` walks any node's single input edge back
    to the INPUT node, producing a `List[(plugin_name, params)]`; `apply_recipe`
    mirrors `RecipeWorker` / `EnhancedDataset.materialize_channel` using
    `PluginBase.get_plugin(...).execute(...)` (reimplemented, not imported, to
    avoid pulling the session/stage graph into the pure layer).
  - `io.py` — `save_pipeline` / `load_pipeline` ↔ `<name>.nd2s_pipeline.json`
    (`kind="nd2studios.pipeline"`, `schema_version=3`), mirroring
    `backend/recipes.py`. `_migrate_v1_to_v2` folds a v1 doc's separate Results
    slice into the merged Analysis slice on load; `_migrate_v2_to_v3` retargets
    the retired `special:validate_tracks` node to `special:track_objects`
    ("Track Objects").
  - **V1.45 merge (Phase 1):** `model.py` adds `NodeCategory`
    (PROCESSING/ANALYSIS/RESULTS/LOGIC/SPECIAL) + `ShapeKind`
    (RECT/TRIANGLE/HEXAGON) on `Node`; `PipelineDoc.merged` aliases the analysis
    slice (Analysis + Results now share one slice — results nodes keep
    `stage=RESULTS` for green coloring). `registry_adapter` adds `if_else_spec`
    (1 in / true+false out triangle), `special_specs` (track-objects / review /
    dismiss / export / send-to-results / pause hexagons; Dismiss red rect), and
    `merged_action_specs`. `executor.py` adds `GraphRunner` (a Qt-free Run
    state machine: topological walk with branch pruning + ready-when-live-preds-
    done), `topological_order`, and `evaluate_simple_condition` (the if-else
    aggregate gate over measurement rows).
  - **V1.45 merge (Phase 2):** `conditions.py` (Qt-free) is the if-else DSL — a
    `Condition` (ALL/ANY) of `ConditionBlock`s (each negatable), spanning
    results-number / object-population / timelapse families via a `BLOCK_KINDS`
    catalog (family + UI schema + evaluator per block); `evaluate_condition` is
    the Run gate and `describe_condition` the readout. The if-else node stores its
    tree in `params["condition"]`. `widgets/node_board/condition_builder_dialog.py`
    (`ConditionBuilderDialog`) edits it (double-click an If/Else node).
  - **V1.45 object lens:** the if-else `params["lens"]` chooses **Whole frame**
    (aggregate gate, prunes a branch — the original behavior) or **Each object**
    (per-object split). In object lens `partition_rows(cond, rows, group_by)`
    splits rows into pass/fail (reusing `evaluate_condition` per group — **Per
    track** groups by `(m_position, track_id)`, **Per row** one row each), both
    branches run, and the Run walk **scopes each downstream node to its branch's
    rows**: `_run_finish_node` records per-output-port rows and `_apply_branch_scope`
    re-points `_run_context["rows"]` to the live incoming branch port (so e.g.
    Dismiss discards only the branch's objects and Spatial Maps see only the kept).
    **Dismiss is a terminal discard:** `_discard_objects(drop_rows)` erases each
    object's `label_id` from the committed `label_masks`, drops its rows from the
    results table + frozen `_track_overlay_rows`, and rebuilds the overlay state —
    so the object, its track id and all its frames vanish from every viewer tab and
    never reach a downstream node (Run and preview share the helper). **Per-track
    persistence** (`track_persistence`, `object_lens_only`) is a single min/max
    frame-count test evaluated per track (`_track_frame_counts`) and is offered by
    `ConditionBuilderDialog` only when its `lens` kwarg is Each object. The Run's
    interactive nodes reuse the review dialogs — `widgets/track_validation_dialog.py`
    (`TrackValidationDialog`, the Review Objects **Single objects** mode, unit
    model that also reviews untracked objects; V1.45 each cropped panel matches
    CellTracker's Results (Cells) cell-detail level — a red `(255,80,80)`
    `find_boundaries` outline on the crop, per-frame detail text (frame · area ·
    ecc · intensity), and a per-cell metric **trace** (`MplCanvas`) with a
    current-frame marker, plus a top-bar Metric selector) and
    `widgets/whole_frame_review_dialog.py` (`WholeFrameReviewDialog`, the **Whole
    frame** mode: label ids on each object, **plain click selects a cell** into a
    top-right **corner cell-over-time inset** — a label-outlined crop centred on
    the cell that follows the T slider/playback (tracked cells animate across all
    their frames; untracked show one frame) with **Accept/Reject Track** actions —
    while **Alt/Ctrl-click toggles reject** and per-frame accept/reject + M selector
    + T slider remain) — `backend/object_tracker.link_objects` (object linking
    driven by the **Track Objects** node via `link_objects_with_params`; also run
    implicitly before timelapse conditions / Review). `link_objects` dispatches per
    (channel, m_position) group on the node's `method`: **Centroid
    (nearest-neighbor)** (`_link_group`, frame-to-frame Hungarian centroid linking
    with distance/size/track-length/frame-gap thresholds) or **SerialTrack
    (topology PTV)** (`_link_group_serialtrack`, scale/rotation-invariant topology
    matching via the vendored `backend/serialtrack`, fed the object centroids
    through its `track_coordinates` path; `st_mode` incremental/cumulative,
    `st_n_neighbors`), or the two **Cell-Tracker** linkers (`_link_group_celltracker`,
    bridging the row-dicts to/from the vendored `backend/celltracker` pandas
    trackers): **Topology (Hungarian)** (`track_timeseries`, rotation-invariant
    neighbor descriptor + distance; `ct_n_neighbors`, `ct_topo_weight`) and
    **Spatial Fingerprint** (`track_fingerprint`, position + log-area with gap
    filling; `ct_area_weight`, `ct_max_gap`). The `min_track_length` /
    `track_validation` post-pass is shared across all linkers.
    **Progress + diagnostics (V1.46):** `link_objects` accepts a backend-pure
    `progress_cb(fraction_0_1, message)`, splits the 0–1 range across its
    `(channel, m_position)` groups, and every `_link_group_*` reports per frame;
    `_TrackJob` (`pipelines_page`) adapts it onto its `ProgressReporter` and
    polls the cancel token in the callback, so a long track shows per-frame
    movement and is cancellable. The linkers log detection/frame counts and
    per-stage timing (topology tracker: link vs. features vs. assignment) and
    warn when the per-frame Hungarian cost matrix is huge (O(n³)). The vendored
    `celltracker.tracking` trackers group by frame once and assign `track_id`
    via a vectorized dict lookup (no per-row `df.apply`); `compute_topology_features`
    is vectorized (output unchanged).
    The **Cell-Tracker Metrics** special node (`backend/celltracker_bridge.augment_rows_with_metrics`
    → `celltracker/metrics.compute_spatial_metrics` + `compute_self_fold_change`)
    augments tracked rows with neighbor distance, local divergence/curl, and
    self-fold-change columns; the **Cell-Tracker Spatial Maps** and **Interpolated
    Spatial Maps** nodes now **open an interactive in-viewer "Spatial Maps" tab**
    (`widgets/spatial_maps_panel.py:SpatialMapsPanel`) instead of carrying their
    own parameters / writing files on Run. The panel is a faithful port of
    Cell-Tracker's spatial page (full control sidebar, background overlay,
    cell-mask mode, borders, quiver, scale bar, auto/global scaling, frame
    scrubbing) over `celltracker/fields.compute_spatial_fields` +
    `celltracker/metrics.compute_self_fold_change`, with the **Field** dropdown
    also offering any numeric measurement column (interpolated via
    `SpatialMapsPanel._interp_value_field`) — subsuming the old Interpolated node.
    **The construction matches the original Cell-Tracker repository:** density is a
    Gaussian-smoothed count of cells per grid bin, and every value-bearing field
    (`mean_area`, `intensity`, `fold_change`, `self_fold`, `velocity_*`, `speed`,
    `divergence`, `curl`, arbitrary columns) is a `scipy.interpolate.griddata`
    **linear interpolation** of the per-cell values across the grid — gaps outside
    the convex hull filled with the frame mean (`nan_to_num`, `nan=1.0` for
    `self_fold`, `0` for velocity), then Gaussian-smoothed. Scalar fields need ≥4
    cells (else all-NaN); velocity needs >3 cells tracked into the previous frame,
    and `divergence`/`curl` are `np.gradient` of that velocity grid. Arbitrary
    columns / `self_fold` fall back to `nearest` with fewer than four cells. The
    page feeds it via `build_tracked_df` (rows → CellTracker DataFrame) +
    per-M label/channel stacks (`_populate_spatial_panel`). The viewer is a
    `QStackedWidget` (image viewer ↔ panel); the **Spatial Maps** overlay tab
    swaps to the panel, and the viewer's maximize (`PopOutWindow`) toggles the
    panel between a compact top-dropdown-bar layout (docked) and the full vertical
    sidebar (maximized) via `set_compact`. An in-tab **Save** button exports
    PNG/TIFF. A node carries reusable **template names** (hidden `templates`
    param, edited via `SpatialTemplatePicker`); templates persist to a global JSON
    library (`backend/spatial_templates.py`) and auto-load when the node runs
    (importable across machines via `import_file`). Both nodes remain special ops
    carrying `NodeCategory.RESULTS` — they group under the green Results tab; the
    per-op category lives on `registry_adapter._SPECIAL_OPS`, and `io.py`'s
    `_normalize_special_categories` refreshes it for old saved graphs on load.
    `backend/results_engine` exporters (Export), and push to the
    top-level `ResultsPage` (Send to Results). Pause retains the `GraphRunner` so
    Run resumes from the pause point. **Run spans the whole file:** an analysis
    node loops every multipoint (`_advance_run_m`) — per-M analysis job →
    measurement job → accumulate rows tagged `m_position` — so the if-else and
    special nodes evaluate over the full dataset (Export writes every M, Review
    Objects opens per-M panels or one whole-frame viewer across all M, Track
    Objects re-links the accumulated rows). **Measurement selection:** a Compute Measurements node carries
    a metric set in `params["metrics"]`, edited via `MeasurementSelectDialog`
    (param-popup "Edit…" button); `compute_measurements(metrics=…)` skips the
    costly shape/intensity work and prunes columns to the selection. **Preview**
    of a results / logic / special node resolves the upstream analysis result
    (committed or the current frame's screened result) and shows the overlay +
    table; double-click previews, editor dialogs open from the popup's Edit
    button. **Multi-frame preview:** a results / Compute Measurements / if-else /
    special preview runs `_ResultsScreenMeasureJob` over the selected (M × T)
    planes — screening + measuring each — so the table aggregates all selected
    frames (rows tagged `m_position`/`frame`) and the overlay is per-plane;
    `_on_viewer_coords` re-previews on any plane-set change. **Validate in
    preview:** a Validate/Review node's popup "Validate tracked objects…" button
    runs `TrackValidationDialog` on the current M's selected-T stack. **Preview =
    scoped Run:** double-clicking a node walks the chain feeding it on the selected
    planes — `_start_preview_walk` shades every reachable node, runs
    `_ResultsScreenMeasureJob` (`_PV_SCREEN_KEY`, honoring the chain's Compute
    Measurements metric selection) for the analysis/results portion, then
    `_preview_walk_downstream` walks the logic/special nodes (gold→done, if-else
    branch, Validate/Review pop-ups when the trigger was deliberate). Subsequent
    frame navigation re-walks non-interactively (no pop-ups). **Export
    organization:** the Export node's *Frame organization* choice (M/T/Z fold vs
    split) drives `results_engine.export_organized` — TIFF multi-page stack per
    group, PNG/JPG single file or per-group subfolder; `EXPORT_ORG_MAP` maps the
    UI labels to fold keys.
- **`widgets/node_board/` (new, Qt)** — custom `QGraphicsScene` editor (no
  NodeGraphQt, per dependency conservatism): `NodeItem` (rounded body, **left**
  stage-accent lip, **top-edge input / bottom-edge output anchors** so flow is
  vertical, hover/selected Disconnect/Duplicate/Delete corner buttons with
  tooltips), `PortItem` (typed, colored per spec §6.2), `EdgeItem` (vertical
  Bézier wire in source-port color; hover-highlight + widened `shape()` +
  **double-click to disconnect** → `NodeScene.disconnect_edge`), `NodeScene`
  (one per `Stage`; type-checked drag-to-connect, cycle rejection,
  one-wire-per-input, fan-out, registry-driven add menu, `disconnect_edge`,
  `set_highlight`), `AddNodeDialog` (roomy catalog picker: a lengthy curated
  description + large synthetic raw→processed preview thumbnails), `ParamPopup`
  (frameless `Qt.Tool` window embedding the existing `widgets/common.py:
  ParamEditor`). Input (top) ports are mouse-transparent so a node can be dragged
  by its top anchor; connections start only from output (bottom) ports. A
  **previewed node** (sticky; set by double-click) and its connected component
  carry a gold outline (`Settings.ACCENT_GOLD`). **V1.45 merge:** `NodeItem` is
  shape-aware (rounded rect / upright triangle / hexagon) and colors per node
  **category** (analysis pink / results green / logic purple / special orange /
  Dismiss red) with a `set_run_state` (shaded / current-gold / done); `NodeScene`
  gained `set_run_states` / `clear_run_states`; `AddNodeDialog` groups its
  catalog into per-category tabs.
- **`pages/pipelines_page.py` (new)** — 50/50 `QSplitter`: node board (sub-tab
  selector + control bar with Add / Preview / Apply / Undo and a slim preview
  `QProgressBar` + one `QGraphicsView` swapping scenes) and a reused
  `MultiAxisViewer`. Preview via `_ProcessingPreviewJob(AnalysisJob)` →
  `MainWindow.job_runner` (coalesce key `"pipeline_preview"`, 300 ms debounce,
  `heavy_ops_blocked()`); the progress bar is driven by `JobRunner.job_progress`.
  The page is **stage-aware**: per-stage previewed node / highlight / preview
  target. **Processing** preview applies the linearized recipe and shows the
  processed image (`set_channels`); Apply commits the primary output's recipe to
  the record + best-effort `RecipeStage.commit()`. **V1.46 — node graph is
  authoritative:** the Processing graph, not a lingering `record.recipe`, defines
  what "processed" means. Leaving the Processing sub-tab (`_select_stage`)
  re-derives the committed recipe from the graph's primary output chain
  (`_graph_recipe` → `recipe_for_node`, `[]` when the Output is unwired / a direct
  Input→Output) and syncs the record (`_sync_committed_recipe_from_graph`:
  `record.recipe` / `recipe_normalized` / `_processed_channels` / rebuilt
  `_processed_view`), so an empty graph clears a stale recipe (e.g. a Background
  Subtract committed via config/session/legacy Recipe page) that the Analysis base
  image (`_show_base_image`) and `_extract_processed_frame` would otherwise still
  apply. The sync is in-memory only (no session-workspace persist — that stays the
  explicit Apply) and no-ops when the recipe is unchanged. Previews read the displayed
  M from `record._raw_volume.all_channels_as_lazy(m=…)` and pass the real
  `n_multipoints`, so the viewer navigates every M (recomputing per-M via
  `_on_viewer_coords` / `_channels_for_m` / `_preview_m`); param edits recompute
  only when the edited node is the previewed node or upstream of it; sub-tab
  switches `fitInView` the whole graph (`_frame_all_nodes`). **Analysis** preview runs the
  previewed pipeline on the current frame (`PipelinePreviewJob`,
  `"pipeline_analysis_preview"`) and overlays label masks via
  `set_frame_post_process` (reusing `analysis_page._overlay_labels`) over a
  processed base image; Apply runs `PipelineCommitJob`
  (`"pipeline_analysis_commit"`) on the current M's full stack into
  `record.analysis_results[name][m]`. **Results** preview overlays a chosen
  committed `AnalysisResult`'s masks (`_composite_results_overlay`) and computes
  its measurements off-thread (`_ResultsMeasureJob` →
  `results_engine.compute_measurements`, `"pipeline_results_preview"`) into a
  `QTableView` (reusing `results_page._MeasurementsModel`); the right pane is a
  vertical `QSplitter` of the viewer + table with an **image/table/split** mode
  bar. (`compute_measurements` is imported at module load on the main thread to
  avoid a skimage/scipy first-import race in the worker.)
  **V1.45 merge (Phase 1):** the page now has **two** sub-tabs — Processing and a
  merged **Analysis** (Stage.ANALYSIS holds analysis + results + logic + special
  nodes). The active right-pane mode is driven by the previewed node's category
  (`_merged_mode`: results node → results overlay + table; else analysis overlay),
  not a separate Results stage. The merged tab swaps **Apply → Run**: `_on_run`
  drives a `GraphRunner` stepper (`_run_pump` / `_run_finish_node`), executing
  analysis nodes (`_RUN_ANALYSIS_KEY`) and measurement nodes (`_RUN_RESULTS_KEY`)
  via the shared `JobRunner`, evaluating if-else conditions over the run-context
  measurement rows to prune a branch, and painting shaded → gold → done via
  `NodeScene.set_run_states`. **Pause** clears run states (editor mode); previews
  are suppressed while a Run is active.
- **V1.46 viewer overhaul:** the right pane wraps the viewer with an **overlay tab
  bar** (Image / Segmentation / Tracks / Vectors: cells / Vectors: field) and the
  bottom split is a **data `QTabWidget`** (Measurements + `MplCanvas` plot tabs:
  Cells/frame, Area, Tracks/frame, Track length). One dispatcher
  `_composite_pipeline_overlay(rgb,t,m)` paints the active layer over the cached
  base; switching a tab only invalidates the post-process cache (base never
  re-renders). Overlays resolve per `(m,t)` via `_overlay_result_for` (previewed
  plane → else committed `_run_results_by_m[m]` at frame `t`) — fixing the
  "overlay only on the Run-start frame" bug. **Live per-frame:** a per-frame
  channel (`ProgressReporter.frame` → `JobRunner.job_frame` →
  `plane_runner.run_planes_to_labels(frame_cb=…)` via `make_frame_cb`, injected by
  `PipelineCommitJob`) drives `_on_run_frame`, which jumps the viewer to each
  finished frame (`MultiAxisViewer.set_current_frame`), paints a live red outline,
  and extends the Cells/frame plot. **Tracks:** `_run_track_objects` builds a
  track colormap + long-track set, switches to the Tracks tab, and plays through
  the frames (colored cells build up); `backend/track_overlays.py` (Qt-free) does
  the solid track-color fill (`make_colored_overlay`), HSV colormap
  (`generate_track_colormap`), and arrowed vectors (`draw_cell_vectors` /
  `draw_grid_velocity` via `cv2.arrowedLine`).
- **V1.46.3 maximize/pop-out:** a `fa5s.expand` button on the viewer's overlay-tab
  row and in the data `QTabWidget` corner pops that panel out into a full window
  via `widgets/popout_window.py` `PopOutWindow` (a reusable `QDialog` that
  *reparents* the live widget — so overlay tabs, sliders, LUTs, zoom/pan and plot
  tabs are preserved — and restores it to its original splitter slot on close).
  `_toggle_popout` / `_popout_windows` / `_set_panel_visible` keep the
  image/table/split mode logic from hiding a popped-out panel.
- **V1.46 preview crop:** a checkable **Preview Crop** button (`_btn_preview_crop`)
  on the overlay-tab row arms the viewer's rubber-band crop tool
  (`MultiAxisViewer.set_crop_mode`); a drag (`crop_rect_selected`) or click
  (`canvas.clicked`) opens `_show_preview_crop_dialog` to confirm an
  `(x, y, w, h)` region stored in `self._preview_crop`. It scopes **preview only**
  (returned by `_crop_rect`, which yields `None` while `_run_active`) and is
  applied at the preview frame readers — `_read_planes_frames`,
  `_read_processed_planes`/`_extract_processed_frame`, `_materialize_channels_for_m`
  — via `_crop_frame` (slices the last two axes), so segmentation / tracking /
  measurement all run on the crop. The navigable base image is wrapped in
  `CroppedVolume` (`pipeline_graph/executor.py`, a `get_frame` proxy reporting
  crop-size `height`/`width`) by `_maybe_crop_volume` — including under
  `PinnedProcessedVolume` for the Processing preview — so the viewer shows the
  cropped region and overlays/masks/rows share one crop-space coordinate system.
  `_field_shape_for` and `_preview_metadata` report crop dims (Vectors: field,
  Spatial Maps). Toggling the button off (or a file change, in
  `load_from_experiment`) clears the crop; it is never serialized.
- **V1.46 cropped Run:** the Run button (`_on_run_button`) is a plain full-file
  Run with no crop, but with a preview crop set it drops a menu
  (`_show_run_menu`) — **Run full** vs **Run cropped region** — and shows a ▾
  (`_update_run_button`). `_start_run(cropped)` sets `_run_cropped`, so
  `_crop_rect()` returns the crop even while `_run_active`, and
  `_run_analysis_node` slices channels (`_crop_channel_for_run`) + crop metadata.
  Committed Run masks carry their geometry in `_run_results_crop`;
  `_overlay_result_for` / `_label_stack_for_m` reuse them only when
  `_run_results_crop == _crop_rect()`, so crop-sized and full-frame masks each
  paint only on a matching display.
- **V1.46 preview plots:** `_update_preview_plots()` builds the bottom-panel
  Cells/frame, Area, Tracks/frame and Track-length plots from the previewed
  measurement rows after each Results preview walk (`_PV_SCREEN_KEY` →
  `_preview_walk_downstream` → plots), so they follow the preview (selected
  frames + crop); previously these were Run-only.
- **V1.46 double-click resets the preview:** an Analysis-tab double-click
  (`_on_node_double_clicked`) calls `_reset_analysis_preview()` before promoting
  the node — it cancels the in-flight preview jobs (`_ANALYSIS_PREVIEW_KEY`,
  `_RESULTS_PREVIEW_KEY`, `_PV_SCREEN_KEY`, `_ANALYSIS_COMMIT_KEY`) via
  `JobRunner.cancel`, stops the debounce, and drops preview scratch
  (`_analysis_screen_results`, `_results_rows`, walk shading, plots, progress) so
  a new preview never mixes with or lands stale from a prior one. No-op during a
  Run; committed Apply/Run results survive.
- **Data flow:** Import `_raw_channels` → Processing INPUT node; each Processing
  OUTPUT node = a `Bridge(PROCESSING→ANALYSIS)` carrying an `EnhancedDataset`
  recipe view via `record.processed_view()`. Analysis INPUT reads
  `processed_view()`; each Analysis OUTPUT node = a `Bridge(ANALYSIS→RESULTS,
  BINARY)` and the run lands in `record.analysis_results`. Results action nodes
  pick a committed `AnalysisResult` by name (`analysis_result` param) and emit
  `Data`; each Results OUTPUT = a `Bridge(RESULTS→Export, DATA)`. (Multi-M
  analysis commit and Undo are deferred.)
- **Export integration (`pages/export_page.py`):** a **"Pipeline Results"**
  source enumerates `record.analysis_results` and exports measurements (CSV via
  `compute_measurements`), overlay frames (`export_overlay_frames`), and label
  masks (`export_label_masks_tiff`). Decoupled — reads the record, not the
  Pipelines page, so it also covers standalone Analysis-page results.
- **Edits:** `core/settings.py` (`PAGES` + `PAGE_PREREQS`), `core/main_window.py`
  (`page_classes`), `core/theme.py` (node-board / sub-tab / popup QSS),
  `pages/export_page.py` (Pipeline Results source),
  `backend/results_engine.py` (int32 label-mask TIFF fix).

## V1.44 additions (GUI Overhaul + Button Revamp)

- **`widgets/icon_button.py`** — `ui_scale()` / `scaled()` DPI helper +
  `qtawesome` icon factories (`make_icon`, `icon_button`, `tool_button`,
  `bind_toggle_icon`). The single source of DPI scaling and modern vector
  icons for the chrome (title bar, top tabs, play controls).
- **Top tab bar** replaces the left sidebar in `core/main_window.py`
  (`_build_top_tabs`); the `QStackedWidget` page model and `_navigate`
  are unchanged. New `#topTabBar` / `#topTabBtn` / `#sessionTabBtn` QSS.
- **Pinned metadata** at the bottom of the right `LutSidebar`
  (`set_metadata_text`, `#metaBox`), updated from
  `MultiAxisViewer._update_axis_labels`.
- New dependency: **`qtawesome`**.

## V1.46.1 — viewer overlay-style control + pixel-hover readout

- **`ZoomToolbar`** (`widgets/image_viewer.py`) gains, between **Pan** and the
  Zoom %, an **Overlay** button whose popup sets overlay `color` / `weight` /
  `multicolor` / `alpha` (state in `_overlay_style`, `overlay_style()` getter,
  `overlay_style_changed` signal; shown only when an overlay is active). To the
  right of the Zoom %: a **px/µm** toggle (`coord_mode_changed`) and a
  `lbl_hover` readout (`set_hover_text`).
- **Pixel hover:** both `ImageCanvas` (`mouseMoveEvent`/`leaveEvent`) and
  `GpuImageCanvas` (`_on_scene_moved`) emit a `hover(iy, ix)` signal (`-1,-1` on
  leave). `MultiAxisViewer._on_pixel_hover` builds the readout — position (image
  px or stage µm via `_pixel_to_stage`, matching `results_engine`
  `centroid_*_stage_um`) plus each enabled channel's intensity from
  `_plane_for_channel` (cached per `(m,t,z)`). Pages pass `stage_xy_um` into
  `set_volume` so the µm toggle has data.
- **Overlay style application:** `MultiAxisViewer.overlay_style()` is read by the
  Pipelines overlay painters; `analysis_page._overlay_labels` gained a
  `thickness` param (boundary dilation) so the weight control widens outlines.

## V1.46.2 — live-run frame sync, conditional tabs, richer if-else, review overlay

- **Frame-strip sync:** `MultiAxisViewer._on_strip_current` now calls
  `FrameStrip.set_current(value, emit=False)` after setting the (signal-blocked)
  slider, so programmatic nav (`set_current_frame`, used by live-run streaming)
  advances the visible strip selection — not just the slider.
- **Conditional overlay tabs:** `pipelines_page._update_overlay_tabs_available`
  (`QTabBar.setTabVisible`) shows Image always, Segmentation once a result exists
  / a run is live, and Tracks/Vectors only after Track Objects links; called at
  stage switch, run start, and `_build_track_overlay`. Plot tabs were already
  per-run (`_clear_plot_tabs` + lazy `_plot_canvas`).
  **Preview parity (V1.46):** the async preview-completion handler
  (`_PV_SCREEN_KEY`) also calls `_update_overlay_tabs_available()` after the
  downstream walk (so Segmentation/Spatial reveal once the async result lands,
  not just on node promotion), and the Run's track-overlay setup is factored into
  a shared `_set_track_overlay_state(rows)` that the preview Track Objects walk
  calls too — so Tracks/Vectors tabs + `_update_preview_plots` light up in preview
  exactly as in a Run. The handler resets `_track_colormap` / `_track_overlay_rows`
  when fresh results arrive so a graph without tracking clears stale track tabs.
- **If-else metrics:** `conditions._METRICS` extended with `track_length`,
  `delta_area_um2/px`, `neighbor_dist_mean/std`, `local_divergence`, `local_curl`,
  `self_fold` (read via `_metric_value` → `row.get`); the builder auto-discovers
  them. Cells/frame = `object_count` block; track counts = tracking-family blocks.
- **Review overlay:** `TrackValidationDialog(overlay_style=…)` renders the cropped
  cell through `analysis_page._overlay_labels` (segmentation format) with an
  Overlay control (Outline/Filled + color + weight); pipelines passes
  `self._overlay_style()`.

## V1.43 additions (Tile-Strip Frame Navigator)

- **`widgets/frame_strip.py`** — `FrameStrip`, a single-`paintEvent`
  strip of rectangular frame tiles (fills width, wraps to rows below a
  DPI-scaled min tile width). Selection / range / keyboard stepping /
  crop-request. One per axis in `MultiAxisViewer`, backed by a hidden
  `QSlider` (the index model) for zero-churn integration with existing
  playback / value code.
- **`MaterializedDataset.subset(m, t, z)`** + **`workers/crop_worker.py`**
  — "crop to selected frames" builds a new in-RAM dataset (`np.ix_`
  copy) off the GUI thread; the source is never mutated. A T crop keeps
  all M/Z (hierarchy rule). `file_panel.py` swaps `record._raw_volume`
  to the crop and offers "Revert to full data"; downstream tabs read the
  active dataset so they honor the crop.
- Per-axis metadata tooltips on the strip (T timestamps/Δt/elapsed, M
  pixel-size/resolution/FOV, Z step/height-in-range) via
  `MultiAxisViewer.set_frame_timestamps` + `_frame_meta_text`.

## V1.42 additions (Viewer Optimizations from Industry Comparison)

Five techniques borrowed from Bio-Formats, napari, and BigDataViewer
after a comparative pass on what popular microscopy viewers do.

- **Metadata sidecar (`.nd2idx.json`)** — `read_or_cache_nd2_metadata`
  in `backend/nd2_loader.py` is the Bio-Formats Memoizer analogue.
  First open writes the full extended metadata to a JSON sidecar;
  subsequent opens load it when fresh (mtime check) instead of
  walking ND2 chunks.  10×+ faster re-open on multi-GB files.
- **`LazyND2Channel` LRU read cache** — opt-in `cache_max_bytes`
  parameter (BDV `CellCache` pattern).  Disabled by default; future
  `LoadStrategy.LAZY_CACHED` path will set a budget.  Cached frames
  are flagged read-only.
- **`MultiAxisViewer.attach_pyramid(reader)`** — wires the existing
  `PyramidReader.pick_level_for_viewport` into the view path.  When
  zoomed out far enough that one screen pixel covers many image
  pixels, `_read_volume_plane` reads from the pyramid level whose
  width still beats the viewport — much less GPU upload on large
  mosaics.
- **`_invalidate_render_cache(*, lut_only=True)`** — napari shader-
  LUT decoupling.  GPU-canvas-active path makes LUT change a no-op
  (pyqtgraph re-applies levels each paint).  CPU-canvas path keeps
  existing cache entries and lets the pre-render worker overwrite
  them in-place rather than starting from scratch.
- **`utils/request_queue.py`** — cancel-first `SingleSlotMailbox`
  and `PriorityLanes` primitives (napari NAP-4 pattern).  Workers
  poll `token.cancelled` at checkpoints; new requests immediately
  invalidate in-flight peers.  Utility shipped; consumers wire it
  in incrementally.

## V1.41 additions (Pipeline Smoothness + Resource-Aware Loading)

- **`utils/progress.py`** — `FrameProgress` accountant + `total_frames_for`
  helper. The single progress contract: every long op counts real
  `T·M·Z·C` frames and emits integer percent only on change (1 % thresholds).
  Consumed by all exporters and the recipe / export / pre-render workers; the
  GUI-facing signal stays `BaseWorker.set_progress(int)`.
- **`utils/perf.py`** — process-wide perf instrumentation gated by
  `ND2_PERF_LOG=1`. `perf_log` decorator, `perf_block` context
  manager, `log_event` raw row emitter. Output appends to
  `~/.nd2studios/perf.csv`. Decorator is a no-op passthrough when
  disabled — zero call-site overhead in normal operation.
- **`utils/resource_strategy.py`** — pre-flight `choose_strategy`
  picks among `EAGER_FULL` / `EAGER_REDUCED` / `LAZY_CACHED` from
  RAM availability and the dataset footprint. Hard cap of
  `2× available` rejects loads via `StrategyError`. Test seam:
  `ND2_FAKE_RAM_GB` env override. **Wired into `LoadWorker` as a
  pre-flight gate (V1.41); the EAGER_REDUCED and LAZY_CACHED
  branches in `materialize_nd2` are deferred to V1.42** — today the
  budgeter rejects oversized loads but doesn't yet collapse Z or
  return lazy proxies.
- **`core/memory_monitor.py`** — `QObject` singleton polling
  `psutil.virtual_memory().percent` at 1.5 s. Banded signals with
  hysteresis: `warning` (80→75), `critical` (90→85, drops pre-render
  caches), `emergency` (95→90, modal warning + heavy-op block),
  `recovered` (back to NORMAL). Subscribed by `MainWindow`; emergency
  band sets `MainWindow.heavy_ops_blocked()` for pages to consult.
- **`utils/user_config.py`** — round-trips a small subset of
  `Settings` to `%APPDATA%/nd2studios/preferences.json` (or
  `~/.config/...` on POSIX). Loaded by `__main__` before UI build;
  saved on Performance dialog accept.
- **`compute/gpu/array.py`** — `should_dispatch_to_gpu` now consults
  free VRAM via a 200 ms-TTL-cached `memGetInfo()` and falls back to
  CPU with a `_vram_warn_once` log when needed × safety_factor > free.
- **`utils/storage.py`** — Windows storage probe via 16 MB read, JSON
  cached per drive letter. `log_default_storage_class` surfaces the
  verdict at startup and defers if no probeable file is present.
- **Streaming exporters** — `tiff_exporter.py` and `movie_exporter.py`
  now feed `tifffile.imwrite(data=iter, shape=..., dtype=...)` and
  `imageio.get_writer(...).append_data(...)` per frame/page. Peak
  RAM during export drops from full-stack-in-list to single frame.
- **`Settings.FORCED_LOAD_STRATEGY`, `EAGER_MAX_FRACTION`,
  `MEMORY_PRESSURE_*`** — runtime-overridable constants driving the
  new subsystems.
- **`widgets/video_player.py` deleted** — unused dead code. The real
  playback path is `MultiAxisViewer._axis_tick → _do_refresh`, and
  the cache-aware hot paths in `_do_refresh` (lines 987–1038)
  already implement what V1.41's plan attributed to a separate
  `FrameCache`.

## V1.40

## System Overview

ND2Studios is a PySide6 desktop application for the McGhee Lab. It loads
Nikon ND2 files (and TIFF stacks), surfaces their metadata, lets the user
build a linear processing recipe with a trial/accept/reject pattern, and
writes three kinds of output: per-channel Z-projection TIFF stacks,
multi-channel RGB composite TIFFs, and time-lapse movies (MP4 / GIF) with
configurable scale bar / timestamp / channel-label overlays.

It is intentionally narrower than CellTracker: there is no segmentation,
no tracking, and no cell-level analysis. ND2Studios is a *sibling* of
CellTracker — code that turned out to be reusable was copied verbatim
(ND2 I/O, enhancement plugins, image viewer, scale bar) rather than
imported, so the two packages can evolve independently.

## High-level layout

```
ND2Studios/
├── run.py                   # launcher
├── nd2studios/              # the package
│   ├── __main__.py          # Qt setup + theme + MainWindow
│   ├── core/                # app state and chrome
│   ├── widgets/             # Qt-aware reusable widgets
│   ├── backend/             # pure Python data layer (no Qt imports)
│   ├── plugins/enhancement/ # registered enhancement plugins
│   ├── workers/             # QThread workers (long-running, signal-heavy)
│   ├── compute/             # V1.37 QThreadPool-backed job runner + AnalysisPipeline adapters; V1.39 GPU dispatch shim (compute/gpu/); V1.40 thread/process parallel helpers (compute/parallel/)
│   ├── pipeline/            # V1.38 per-source workspace + stage commits (recipe / analysis); V1.39 pyramid stage
│   ├── utils/               # backend-pure helpers (profiling, resources) + one Qt boundary (threading) — see Module Breakdown
│   └── pages/               # the three GUI pages
├── CodeLog/                 # plans, changelog, architecture, refs
├── Research/                # literature reviews per method
├── hpc/                     # HPC integration (unused in V1.0)
├── profiling/               # V1.33 profiling harness + JSON baselines
└── scripts/                 # version_push.py
```

## Module Breakdown

### `nd2studios/__main__.py` — entry point
Sets `QT_FONT_DPI`, `SSL_CERT_FILE` (via `certifi`), instantiates
`QApplication` with the Fusion style + Dracula stylesheet, force-imports
`nd2studios.plugins.enhancement.builtin`,
`nd2studios.backend.analysis.nuclei_segmentation`,
`nd2studios.backend.analysis.tear_detection`,
`nd2studios.backend.analysis.histogram_threshold_pipeline`,
`nd2studios.backend.analysis.spots_pipeline`, and
`nd2studios.backend.analysis.manual_mask` (Mask Analysis pipeline) so both the plugin registry
and the analysis pipeline registry are fully populated before any page is
built, opens `MainWindow`.

### `core/`

| File | Purpose |
|---|---|
| `settings.py` | App constants (window dims, sidebar widths, animation duration, drop-shadow params), Dracula palette, `PAGES` list, `STATUS_ORDER`, `PAGE_PREREQS` |
| `theme.py` | Single QSS stylesheet adapted from CellTracker + PyDracula. Chrome rules (`#bgApp`, `#titleBarWidget`, `#navBtn`, `#sessionBtn`, `#toggleBtn`) coexist with form/button/list/dialog rules ported from CellTracker |
| `main_window.py` | Frameless `QMainWindow`. Builds: outer container with 10 px margin (room for drop shadow) → `#bgApp` card with `QGraphicsDropShadowEffect` → custom title bar (drag, double-click maximize, Min/Max/Close) → body `QSplitter(Qt.Horizontal)` of sidebar + content area (draggable handle). Sidebar collapses 60↔240 px via `QPropertyAnimation(InOutQuart)` on `minimumWidth`/`maximumWidth`; after expansion the max-width cap is removed so the splitter handle can grow the sidebar past 240 px. `_on_sidebar_anim_done()` calls `_reset_active_viewer_zoom()` to fit the image after every toggle. Content area has top bar (`titleLabel` + `StatusIndicator`), `QStackedWidget`, bottom bar (version label + `QProgressBar` + status text). Edge resize via 4 `CustomGrip`s. Navigation guards consult `Settings.PAGE_PREREQS` |
| `experiment_manager.py` | `ND2StudiosRecord` carries configs (`import`, `recipe`, `export`), metadata snapshot (`nd2_metadata`), `channel_display` (`{name: {enabled, color}}`), the recipe (`List[(plugin_name, params)]`) and `recipe_normalized` flag, frame counts + pixel size, plus *non-serialized* in-memory caches: `_raw_channels`, `_processed_channels`, `_frame_timestamps`. `ND2StudiosManager` is a `QObject` exposing `active_changed` / `status_changed`. `save_session()` writes JSON manifest + `_arrays.npz` companion (`raw_*`, `proc_*`, `frame_timestamps`); `load_session()` is the inverse. **V1.38 Phase 6**: new non-serialized field `_processed_view` holds the `EnhancedDataset` proxy after a Recipe-page release; `processed_view()` returns whichever processed-channels source is live (in-RAM dict → proxy → raw fallback) and `has_processed()` is True when either of the first two is available. Pages call these accessors so they don't have to branch on which mode is active |
| `plugin_registry.py` | `ParamSpec` (declarative parameter), `PluginBase` (decorator-based registry), `EnhancementPlugin` base for `(T,H,W)` array transforms |
| `analysis_registry.py` (V1.19) | `AnalysisResult` dataclass (`label_masks`, `measurements`, `summary`), `AnalysisPipeline` ABC with separate `_registry` (distinct from `PluginBase`). Pipelines take full multi-frame channel dicts and return structured results rather than transformed arrays |

### `widgets/` — Qt-aware reusable building blocks

| File | Purpose |
|---|---|
| `common.py` | `MplCanvas` (Dracula matplotlib figure), `ParamEditor` (auto-form from `List[ParamSpec]`), `StatusIndicator` (colored status badge) |
| `image_viewer.py` | `ImageCanvas` (zoom/pan), `ZoomToolbar`, `ImageViewer` (single- or multi-channel RGB compositing with per-channel color LUTs and enable/disable). Also exports `CHANNEL_COLORS`, `frame_to_uint8`, `apply_lut_color`, `composite_channels`. V1.30: `ImageCanvas` gained mutually-exclusive draw modes (`set_draw_mode("rect"|"ellipse"|"polygon"|None)`) and a `shape_drawn(mode, vertices)` signal; rectangle/ellipse are rubber-band drag, polygon samples vertices along a freehand path and auto-closes on release. The in-progress shape paints in cyan dashed lines to distinguish it from the yellow crop rubber-band. V1.36: still used by `export_preview_dialog`, the embedded `ImageViewer`, the recipe / stitch / export pages — the multi-axis viewer is the only consumer that swapped to the GPU canvas. |
| `gpu_image_canvas.py` (V1.36 Phase 4) | `GpuImageCanvas` — pyqtgraph-backed drop-in for `ImageCanvas` on the multi-axis viewer. Hosts a `pg.GraphicsLayoutWidget` + `ViewBox` with one `pg.ImageItem` per channel (`_ChannelLayer`); each layer owns a 256-entry colored LUT (black → channel RGB) plus `(lo, hi)` levels. `CompositionMode_Plus` makes Qt sum channels additively, replacing the per-frame CPU `apply_lut → multiply → clip → QImage` path. Public per-channel API: `configure_channels(names)`, `update_channel(c, plane)`, `set_channel_visible(c, on)`, `set_channel_color(c, rgb)`, `set_channel_levels(c, (lo, hi))`. A separate composite-fallback `ImageItem` accepts pre-composed RGB via `set_image(rgb)`/`set_grayscale(gray)` for callers that still produce RGB (post-process hook, etc.); per-channel and composite layers swap visibility atomically. Tool parity: a `_ToolOverlayItem(QGraphicsItem)` paints crop / draw (rect/ellipse/polygon) / vertex-edit overlays in scene = image-pixel coordinates; press/move/release is captured via an `eventFilter` on the viewport so tool drags don't fight ViewBox pan. Emits the same signals as the legacy canvas: `clicked`, `zoom_changed`, `pan_mode_changed`, `crop_rect_selected`, `shape_drawn`, `vertex_moved`, `edit_committed`. Zoom toolbar API (`reset_zoom`, `zoom_in`, `zoom_out`, `set_pan_mode`) routes through `ViewBox.autoRange()` and `scaleBy`. Coordinate transforms (`widget_to_image` / `image_to_widget`) compose through `viewbox.mapSceneToView` / `mapFromView`. Gamma is intentionally not applied on the GPU path in V1.36 — see Phase 4 plan for the follow-up. |
| `multi_axis_viewer.py` (V1.1, restructured V1.2, V1.17, hook V1.20, IO worker V1.34, pin/stats V1.35, GPU canvas V1.36, pyramid V1.39) | `ChannelChip` (compact toggle + color combo only — LUTs moved to the sidebar in V1.2); `MultiAxisViewer` with a `QSplitter(Qt.Horizontal)` root: left column (image canvas + zoom toolbar + M/T/Z sliders + chip strip) and right column (`LutSidebar`) — both panes are draggable. Each M/T/Z axis row includes a `QPushButton` play/pause toggle and a `QDoubleSpinBox` FPS control; per-axis `QTimer`s advance the slider on each tick (wraps to 0). `LutSidebar.collapse_changed` triggers `canvas.reset_zoom()`. M/T/Z sliders auto-hide when an axis is degenerate. Reads from a `LazyND2Volume` (M/Z scrolling, takes `stage_xy_um` for the tile overlay) or a flat `Dict[name, (T,H,W)]` (recipe / processed view). V1.17: slider changes check `FrameCache` first (instant display on hit); on a miss a 50/80 ms debounce timer fires `_do_refresh()`. After each frame compose, `PrefetchManager.request_neighbors()` queues ±5 T neighbors. Histogram samples cached per `(m, z_mode)` to skip resample on M revisits. V1.20: `set_frame_post_process(fn)` hook — `fn(rgb_uint8, t, m) -> rgb_uint8` applied after compositing before `canvas.set_image()`; pass `None` to clear. **V1.34 Phase 2**: `FrameCache.max_bytes` sourced from `recommended_cache_budget_bytes(reserve_fraction=0.6)` instead of a hard-coded 300 MB cap. A foreground `IOWorker` (started in `set_volume()` via `start_io_worker(volume)`, alongside the existing `PrefetchManager`) services cache misses off the GUI thread — `_compose_current_frame` now enqueues `PlaneRequest`s with priority 0, increments `_latest_request_id`, and calls `IOWorker.cancel_all()` to drop stale foreground reads; `_on_io_plane_ready(request_id, key, plane)` populates the cache and only redraws when the result matches the most recent gesture. `_teardown_io_worker()` mirrors the `PrefetchManager.stop()` teardown on `set_volume` / `set_channels` and bumps the gating counter so in-flight planes can't sneak a redraw after teardown. **V1.35 Phase 3**: `set_volume()` now picks `radius_t` adaptively from `detect().available_ram_gb` (2 / 4 / 8 in the < 2 / < 8 / ≥ 8 GB free tiers) and passes it to `PrefetchManager`; the V1.17 `n=5` override on `request_neighbors` is dropped so the manager's velocity-biased window takes effect. The viewer tracks `_pinned_keys: set[tuple]` and pin/unpin-diffs the channel keys that composited into each successful render so the on-screen planes are exempt from cache eviction; `set_volume`, `set_channels`, and `_teardown_io_worker`'s callers clear the pin set. `cache_stats_text()` returns a one-line summary (`"Cache: <used>/<budget> MB · hit <rate>% · evictions <n>"`) for the profiling harness and a future status-bar tooltip. **V1.36 Phase 4**: `__init__` picks `GpuImageCanvas` when `Settings.USE_GPU_DISPLAY` is True (default), falling back to `ImageCanvas` if construction raises or `ND2_DISABLE_GPU_DISPLAY=1` is set; `_use_gpu_canvas` gates the render branch. `_configure_gpu_canvas_channels(names)` pre-allocates one `pg.ImageItem` per channel on the canvas on every `set_volume` / `set_channels`. `_do_refresh()` calls `_render_current_frame_gpu()` first when the GPU canvas is active and no post-process hook is attached; that path walks each chip, pushes visibility / RGB color / `(lo, hi)` levels to the canvas, and uploads raw 2-D planes via `canvas.update_channel(c, frame_2d)` so the GPU does additive compositing. Cache misses still go through the V1.34 `IOWorker` via the same `PlaneRequest` queue and the same `_latest_request_id` gating used by the CPU branch. Pin/unpin and velocity-biased prefetch are triggered on every GPU refresh exactly like the CPU path. When `set_frame_post_process(fn)` is attached, the viewer falls back to `_compose_current_frame()` + `canvas.set_image(rgb)` so the hook still receives a `(H, W, 3) uint8` array; the GPU canvas's composite-fallback `ImageItem` activates automatically and the per-channel layers re-engage on the next hook-free refresh. **V1.39 Phase 7**: optional pyramid integration — `attach_pyramid(reader)` binds a `PyramidReader` and connects `viewbox.sigRangeChanged → _on_viewport_changed` so the level is recomputed on every pan/zoom. `_active_level=0` keeps the V1.38 read path (5-tuple cache keys, IOWorker, prefetcher); `_active_level>=1` switches the cache key to a 6-tuple `(level, c, m, t, z, z_mode)` (so pyramid planes coexist with source planes in the same cache without collision) and reads synchronously from the pyramid reader on miss (chunks are small enough that the IOWorker would only add overhead). Level changes clear pinning and bump `_request_counter` so any in-flight level-0 IOWorker plane arriving after the switch is dropped by `_on_io_plane_ready`'s gating |
| `lut_histogram.py` (V1.1) | `LutHistogramWidget` — log-scaled intensity histogram with draggable lo/hi handles, gamma slider, Auto / Reset buttons. Samples ≤ 32 frames so it stays cheap on long timeseries. Module-level `apply_lut(frame, lo, hi, gamma) -> uint8` |
| `lut_sidebar.py` (V1.2, expanded V1.3) | `LutSidebar` — collapsible right-side panel hosting two independently-collapsible sections: (1) a **Tiles** section with a `TileLayoutWidget` in navigate mode, and (2) a **LUTs** section with one `LutHistogramWidget` per channel. Sidebar width 300 / 28 px collapsed; animated on both `minimumWidth` and `maximumWidth` (animating both prevents `QHBoxLayout` from fighting the animation). Section headers (`_SectionHeader`) have ▾/▸ toggles. Tiles section auto-hides when ≤ 1 tile. Signals: `channel_contrast_changed(name, lo, hi, gamma)`, `tile_navigate_requested(m)`, `collapse_changed(bool)`. `update_swatch(name, rgb)` keeps the swatch in sync when the chip strip's color combo changes |
| `tile_layout.py` (V1.3, auto-sized V1.6) | `TileLayoutWidget` — interactive multipoint layout with two modes (`"navigate"` / `"select"`), `QPainter` rendering, hit-testing in cached widget rects, hover preview, tile-index labels, and an expand-to-modal `⤢` button. V1.6: `_adapt_minimum_size()` runs after every `set_tile_layout()` and sets `setMinimumSize(n_cols × 50 + margins, n_rows × 50 + header + margins)` so the host's `QScrollArea` can scroll when the natural size exceeds the viewport. Three signals: `navigate_requested(m)`, `selection_changed(set[int])`, `expand_requested`. `TileLayoutDialog` (V1.6: wraps the widget in its own `QScrollArea`) is the modal "expand" wrapper — auto-closes after click in navigate mode, stays open in select mode |
| `scale_bar.py` | Matplotlib scale-bar rendering helpers (used by `MplCanvas` consumers) |
| `spatial_maps_panel.py` (V1.46) | `SpatialMapsPanel` — the in-viewer **Spatial Maps** tab; a faithful port of Cell-Tracker's spatial page over `backend/celltracker/fields.compute_spatial_fields` + `metrics.compute_self_fold_change` (+ `backend/interp_maps.compute_interp_map` for arbitrary measurement columns). One set of control widgets is reparented between two layouts by `set_compact(bool)`: a compact top dropdown bar (docked viewer) and the full vertical collapsible sidebar (maximized viewer). Frame navigation uses the ND2Studios `FrameStrip` (NIS-style tile strip) + play/pause + FPS controls — same set as `MultiAxisViewer._make_axis_row` — not a plain slider. Scrubbing is debounced (a fast drag renders only the final frame) and a per-frame field cache (keyed by `(t, grid, sigma, channel, source)`, cleared on those param changes) makes revisits / playback instant; the tab opens with a fast single-frame colour-scale fit (`_autofit_scale_from_current`) instead of the all-frames pass, which is reserved for the "Global Scale" button. The canvas supports **zoom / pan** (Home/+/−/Pan buttons, scroll-zoom, drag-pan) via matplotlib data limits (`_zoom_about` / `_apply_view`, persisted in `_view` across redraws), and the colour bar is a **fixed axes pinned to the figure's right edge** so it stays at the viewer border when zoomed. Also: an optional multipoint selector (`m_change_requested`), `get_config`/`apply_config`, a template strip (Save / Load file / preset combo) backed by `backend/spatial_templates`, `set_node_templates(names)` to pre-load a node's templates, and an in-tab **Save** (current frame / all frames, PNG or raw-field TIFF). `SpatialTemplatePicker(QDialog)` edits which library templates a node carries (add/remove/reorder + Import JSON). Fed by `pipelines_page._populate_spatial_panel` via `celltracker_bridge.build_tracked_df` + per-M label/channel stacks |
| `video_player.py` | Play/Pause/FPS playback controls |
| `custom_grips.py` | Per-edge frameless-window resize widgets. Lean reimplementation of PyDracula's `CustomGrip` |
| `export_preview_dialog.py` (V1.32) | `ExportPreviewDialog(channels, colors, enabled, pixel_size_um, frame_timestamps_s, lut_settings, movie_options, title)` — modal preview shown after the user confirms movie or image-sequence export settings. Embeds an `ImageCanvas` + `ZoomToolbar` with a T scrubber, a ▶ Play / ⏸ Pause toggle, and an in-dialog FPS spin box; a second `QTimer` advances T at the chosen FPS so the user sees the movie actually play back with brightness / contrast / saturation / hue / fade applied live. Manually grabbing the T slider pauses playback. Exposes the five adjustment sliders (each a `_LabeledSlider`) and a "Show overlays in preview" toggle. A 30 ms single-shot debounce coalesces slider drags before recompositing; the play-tick path renders directly to keep the frame rate honest. `adjustments()` returns the user-chosen `ImageAdjustments` once the dialog is accepted; the dialog itself never runs the export. `closeEvent`/`done()` stop the playback timer so it can't outlive the dialog |

### `backend/` — pure-Python data layer (no Qt imports)

| File | Purpose |
|---|---|
| `nd2_loader.py` | `ND2Metadata` dataclass, `read_nd2_metadata`, `read_nd2_metadata_extended` (returns dict with full metadata: T/Z/C/P, pixel size, z step, channel names + colors + emission/excitation + exposures, objective + camera + microscope, binning, frame timestamps, **per-M stage XY/Z** — tries `f.experiment` XYPosLoop first, falls back to `frame_metadata()` per-M; `stage_layout_source` includes method suffix e.g. `"stage_xy:experiment"`, loops), `load_nd2_timeseries`, `LazyND2Channel` (numpy-protocol on-demand frame proxy with `__getitem__`, `materialize`, `crop`), `load_nd2_timeseries_lazy`, `z_project` |
| `nd2_volume.py` (V1.1) | `LazyND2Volume` exposing `(M, T, Z, H, W)`. `get_frame(c, m, t, z, z_mode, z_start, z_end)` returns a single (H, W) array on demand. `to_lazy_channel(c, m, z_mode, z_index, ...)` collapses to a V1.0 `LazyND2Channel` so the recipe pipeline keeps the `(T, H, W)` contract |
| `tiff_loader.py` (V1.17, multi-Z multi-C support V1.30) | `get_tiff_info`, `load_tiff_stack`, `LazyTIFFChannel` (lazy `(T,H,W)` view; `n_pages_per_t`/`page_within_t` params enable per-channel access; V1.17: caches `tifffile.TiffFile` handle on first read via `_ensure_open()`; V1.30: new `z_stride` param so Z accumulator reads `base_page + z * z_stride` — `z_stride=n_c` for ImageJ TZCYX hyperstacks, `z_stride=1` for single-channel multi-Z), `load_tiff_stack_lazy`, `read_imagej_tiff_metadata`, `load_imagej_tiff_channels(z_projection="max")` (V1.30: passes `n_z`/`z_stride=n_c`/`z_projection` so multi-channel multi-Z files actually project Z), `_SingleFileTIFFView` (V1.30: `get_frame` reads pages directly for multi-Z files honoring `z`/`z_mode`; `to_lazy_channel` builds proxies that honor `z_mode`/`z_index`/Z range; caches a `tifffile.TiffFile` for direct page reads), `LazyMultiFileTIFFVolume` (V1.30: `to_lazy_channel` propagates `z_mode`/`z_index`/`z_*`/`t_*` through to the underlying view), `assign_dimensions`, `extract_2d_timeseries` |
| `frame_cache.py` (V1.17, V1.34 adaptive budget, V1.35 pin / stats) | `FrameCache`: thread-safe LRU cache keyed by `(c, m, t, z, z_mode)` storing normalized `(H, W)` frames. `get`/`put`/`contains`/`clear` all lock-protected. Default budget 300 MB (in practice sized by `recommended_cache_budget_bytes`). **V1.35 Phase 3** adds: `CacheStats` (hits / misses / evictions / current_bytes / peak_bytes / `hit_rate`) updated under the same lock; `pin(key)` / `unpin(key)` / `is_pinned(key)` that exempt a key from eviction so the on-screen composite's planes can't be evicted by the prefetcher's neighbor writes; `set_max_bytes(n)` to resize on the fly; `current_bytes`/`max_bytes`/`__len__` accessors. Inserted arrays are flagged `writeable=False` so accidental in-place mutation downstream fails with `ValueError`. Eviction skips pinned entries and breaks out when only pins remain (over-budget but never evicting a displayed plane). `clear()` drops both entries and pins; hit/miss/eviction counters persist across `clear()` so a session-long hit rate stays meaningful |
| `normalization.py` | `normalize_frame_means` (frame-mean intensity normalization) |
| `recipes.py` | `build_recipe`, `save_recipe`, `load_recipe` for `.nd2s_recipe.json` (kind = `nd2studios.recipe`) |
| `spatial_templates.py` (V1.46) | Global JSON library of named Spatial Maps configurations under `~/.nd2studios/spatial_map_templates` (one `*.nd2s_spatialmap.json` per template). `list_templates` / `load_template` / `save_template` / `delete_template`; `read_template_file` parses a single-template or bundle file, `import_file` merges it into the library (cross-machine portability), `export_file` writes a bundle. Pure stdlib (no Qt/numpy). Consumed by `widgets/spatial_maps_panel.py` and the Spatial Maps nodes (which store template **names**) |
| `exporters/tiff_exporter.py` | `export_tiff_stack(stack, filepath, bit_depth, pixel_size_um, progress_cb)` (single-channel `(T,H,W)` or `(T,Z,H,W)`, used by label-mask export); `export_tiff_hyperstack(channels, enabled, filepath, bit_depth, pixel_size_um, progress_cb)` (V1.29 — multi-channel, writes a single `(T,Z,C,H,W)` ImageJ TZCYX hyperstack with `Labels=[<channel names>]`, `unit=um`, `spacing`, resolution; same file construction as `export_stitched_tiff`). BigTIFF auto-switch |
| `exporters/composite_exporter.py` | `export_rgb_composite_tiff(channels, colors, enabled, filepath, …, image_adjustments)`, `_composite_frame(…, image_adjustments)`, `CHANNEL_COLORS`. V1.32: `ImageAdjustments` dataclass (`brightness`, `contrast`, `saturation`, `hue`, `fade`); vectorised `apply_image_adjustments(rgb, adj)` for contrast / saturation / hue / fade on the final RGB; `_apply_brightness_to_gray(gray, brightness)` applies brightness as **multiplicative gain** (`gain = 1 + brightness/100`) inside the composite loop *before* the channel colour LUT is mixed in. The gain semantic keeps background pixels near zero (only signal scales visibly — additive offset would lift the noise floor as much as the peaks) and the per-channel placement keeps a red-only channel pure red rather than washing toward white. `ImageAdjustments.is_identity_post_composite()` lets the composite pass skip the post-RGB pass when only brightness is set |
| `exporters/movie_exporter.py` | `MovieOptions` dataclass + `export_movie(channels, colors, enabled, filepath, options, pixel_size_um, frame_timestamps_s, lut_settings, image_adjustments, progress_cb, status_cb)`. Uses `imageio` (libx264 for MP4, pillow for GIF). PIL-rendered overlays for scale bar, timestamp, channel labels |
| `exporters/image_sequence_exporter.py` (V1.32) | `format_frame_name(basename, t, m, z, n_t, n_m, n_z, extension)` — pure naming helper that only includes axis suffixes (`_T01`, `_M02`, `_Z03`) when the corresponding axis has more than one frame. `ImageSequenceRequest` dataclass; `export_image_sequence(request, progress_cb, status_cb)` writes one PNG per frame via PIL. Two iteration paths: in-memory channels (T-only, recipe-applied) or `LazyND2Volume` (M × T × Z, raw — no recipe). Reuses `_composite_frame` and `_draw_overlays` so the colour pipeline matches movie exports byte-for-byte |
| `analysis/manual_mask.py` (V1.30, V1.31 per-Z shapes, **V1.40 Custom Mask Creator**) | `ManualMaskPipeline` (registered name `"Mask Analysis"`; renamed from `"Manual Mask"` in V1.40) — rasterizes user-drawn shapes *and* auto-tiled rectangular strips into a `(T, H, W)` int32 label stack. **V1.40**: new `tiled_masks` hidden param (list of `{shape_type, direction, height_px, width_px, x_offset_px, start_offset_px}` dicts); `generate_tiled_label_boxes(tiled_masks, H, W)` returns `(y0,y1,x0,x1)` tuples for every *complete* strip (partial strips at the image boundary are excluded); `generate_tiled_mask(tiled_masks, H, W)` returns a `(H,W)` int32 overlay array for live preview. In `run()` tiled strips receive stable IDs 1..N across all frames so each strip is the same measurement object at every timepoint; hand-drawn shapes are assigned IDs N+1, N+2, … per frame. `rasterize_shapes()` gains a `label_offset` kwarg so the drawing overlay uses the same stable offset. `AnalysisPage` stores `_tiled_masks: List[Dict]` (persisted to `ND2StudiosRecord.tiled_mask_specs`), exposes a **Custom Mask Creator** `QGroupBox` (shape type, direction, strip height, optional custom width+x-offset, start offset, Generate/Clear buttons), and extends `_composite_overlay()` to paint tiled strips below drawn shapes in the live preview |
| `analysis/nuclei_segmentation.py` (V1.19, GPU flag V1.39) | `NucleiSegmentationPipeline` — Cellpose 3 per-frame 2D segmentation. Optional dep (`pip install cellpose`); raises `ImportError` with install message when absent. Uses `skimage.measure.regionprops` for per-object measurements. Registered via `@AnalysisPipeline.register`. **V1.39 Phase 7**: `models.Cellpose(gpu=is_gpu_enabled())` honours the runtime `compute.gpu` flag — Cellpose owns its own CUDA initialization; we only forward the boolean |
| `analysis/stardist_segmentation.py` (V1.45) | `StarDistSegmentationPipeline` — per-frame 2D StarDist instance segmentation (ported from CellTracker; its only segmentation method). Optional dep (`pip install stardist tensorflow csbdeep`); raises `ImportError` when absent. Mirrors the Cellpose node: `is_gpu_enabled()` → `disable_gpu`, per-frame via `plane_runner.run_planes_to_labels(n_workers=1)`, area filter + contiguous relabel in one `O(pixels)` pass via `source_utils.filter_and_relabel` (np.bincount + LUT — replaced the old per-object full-frame `mask[mask==label]` / `_relabel_contiguous` loops), `AnalysisResult(..., overlay_outline=params["outline"])`. When `outline` is on the result also sets `overlay_color=(255,50,50)` + `overlay_alpha=1.0` so `_overlay_labels` paints every `find_boundaries(mode="outer")` pixel in that exact colour at full opacity — byte-for-byte CellTracker's boundary rendering. Params: `channel_name`, `model_name` (`2D_versatile_fluo`/`_he`/`_paper_dsb2018`), `prob_thresh`, `nms_thresh`, `scale`, `min_area`/`max_area`, `outline`. Backend is the vendored `backend/celltracker/segmentation.py` (`get_stardist_model`, `segment_frame`, `_ensure_tf_threading` — TF pinned to a single inter/intra-op thread exactly as CellTracker's repo does, which is ~10× faster than multi-threaded for StarDist's tiled inference; Windows csbdeep-symlink patch; GPU flag honored by clearing `CUDA_VISIBLE_DEVICES` before the first TF import) |
| `analysis/tear_detection.py` (V1.19, GPU dispatch V1.39) | `TearDetectionPipeline` — classical dark/homogeneous region detection. Stage 1: tissue mask via Otsu on log-intensity + morphological cleanup (4× downsample). Stage 2: rank-normalised homogeneity score from local intensity (`uniform_filter`), local variance (`generic_filter`), local entropy (`skimage.filters.rank.entropy`). Stage 3: connected-component extraction with area filters. Stage 4: optional weak-stain disambiguation via counterstain channel. No DL dependencies. Measurements include `class` (`tear`/`weak_stain`/`unknown`), `homogeneity_score`, `solidity`, `eccentricity` in addition to the standard fields. **V1.39 Phase 7**: `gaussian` and `threshold_otsu` imports route through `compute.gpu.ops` instead of `skimage.filters`; same signatures, CPU pass-through when GPU mode is off |
| `analysis/histogram_threshold_pipeline.py` (V1.20) | `HistogramThresholdPipeline` — adapter that registers `histothresh/` into the `AnalysisPipeline` registry. Iterates T frames, calls `HistogramThresholdSegmenter.run(frame)` per slice, stacks `(T,H,W)` int32 label arrays, and assembles `AnalysisResult`. 16 `ParamSpec` parameters exposed in the Analysis tab UI. Sets `bit_depth_strict=False` so TIFF inputs that overflow the declared range warn rather than crash |
| `analysis/histothresh/` (V1.20) | Qt-free subpackage implementing the histogram threshold segmenter. `validation.py`: `validate_bit_depth()`, `infer_bit_depth()`, `BitDepthError` — rejects float inputs and values overflowing the declared bit-depth max. `histogram.py`: `Histogram` dataclass with `percentile()` / `cumulative()` helpers; `compute_histogram()` bins the full LUT range (one bin per integer value); four `suggest_threshold_*()` helpers (Otsu, Triangle, Minimum, Multi-Otsu). `thresholds.py`: `threshold_single()` (below/above/between/outside), `threshold_hysteresis()` (inverts image for `below` mode to reuse scikit-image's high-pass hysteresis), `threshold_percentile()`, `threshold_relative()`. `morphology.py`: `apply_spatial_constraints()` (opening → closing → hole-fill → small-object removal), `homogeneity_gate()`. `config.py`: `ThresholdConfig` plain dataclass with `__post_init__` validation of method/direction param combinations. `identifier.py`: `HistogramThresholdSegmenter` orchestrator returning `SegmentationResult` (mask, int32 labels, regions list, histogram, threshold_used dict, provenance dict) |
| `results_engine.py` (V1.22) | `compute_measurements(label_masks, channels, metadata, m_index)` → `List[Dict]` with area, centroids (px / µm / stage-absolute µm), perimeter, eccentricity, solidity, bbox, per-channel mean/std intensity. Uses `skimage.measure.regionprops`; no Qt. Also: `export_overlay_frames()` (uint8 RGB + cyan mask overlay, `imageio`), `export_label_masks_tiff()` (int32 stacks, `tifffile`) |
| `template.py` (V1.23) | `save_template()` / `write_template()` / `load_template()` for `.nd2st.json` pipeline templates (import, recipe, analysis, results configs without data). `TEMPLATE_EXTENSION = ".nd2st.json"` |
| `analysis/spots_pipeline.py` (V1.21) | `BrightDarkSpotsPipeline` — adapter that registers `spots/` into the `AnalysisPipeline` registry. Iterates T frames, calls `BrightDarkSpotsSegmenter.run(frame)` per slice, stacks `(T,H,W)` int32 label arrays, and assembles `AnalysisResult` with extra measurement columns (`diameter_px`, `contrast_score`, `circularity`, `polarity`). 11 `ParamSpec` parameters. `intensity_percentile=0.0` maps to gate disabled. Sets `bit_depth_strict=False` |
| `analysis/spots/` (V1.21, GPU dispatch V1.39) | Qt-free subpackage implementing the GA3-style scale-space spot detector. `validation.py`: re-exports `histothresh.validation` helpers + `validate_diameter()`. `scale_space.py`: `resolve_sigma()` (FWHM-matched sigma, Marr-Hildreth ratio), `log_response()` (σ²-normalised LoG, positive for bright), `dog_response()` (DoG band-pass, positive for bright). **V1.39 Phase 7**: `gaussian_filter` and `gaussian_laplace` imports route through `compute.gpu.ops` (which itself dispatches to `cupyx.scipy.ndimage` on GPU and falls back to `scipy.ndimage` on CPU); the DoG and LoG response functions are unchanged structurally. `detection.py`: `find_extrema()` (peak_local_max on normalised response, intensity-gate suppression, contrast gate), `h_transform_seeds()` (h-maxima / h-minima). `symmetry.py`: `SYMMETRY_FLOOR` dict, `circularity()`, `filter_by_symmetry()` (GA3 All/More/Medium/Less bins). `grow.py`: `grow_seeds_dilation()` (disk morphology), `grow_seeds_watershed()` (watershed-from-marker; negated image for bright). `config.py`: `SpotsConfig` dataclass with `__post_init__` validation. `identifier.py`: `BrightDarkSpotsSegmenter` orchestrator (12-step: validate → sigma → DoG/LoG → normalise → intensity mask → peak detection → label rasterisation → symmetry gate → grow → re-label → regionprops → provenance); `SpotsResult` dataclass |
| `exporters/stitch_exporter.py` (V1.1, layout rewritten V1.14, export format V1.15, metadata aligned V1.29, Z-preserving V1.30) | `StitchLayout` dataclass; `compute_tile_layout(stage_xy_um, pixel_size_um, tile_h, tile_w, m_indices)` converts stage XY µm to pixel offsets directly: `offset_x = round((sx - min_x) / pixel_size_um)`, `offset_y = round((max_y - sy) / pixel_size_um)` (Y flipped). Canvas = bounding box of all tile corners. Physical gaps between non-adjacent tiles appear as empty pixels. `source = "physical"`. Grid fallback when no stage XY data. Helper functions `_cluster_axis`, `_assign_index`, `_serpentine_layout` retained but not in the main path. `stitch_one_frame(tile_frames, layout, dtype)`, `export_stitched_tiff(volume, layout, m_indices, channel_indices, channel_colors, filepath, ...)` writes a `(T, Z, C, H, W)` ImageJ TZCYX hyperstack with `Labels=[<channel names>]`, `unit=um`, `spacing=pixel_size_um`, resolution. `z_mode="none"` on a multi-Z file keeps every Z plane (`Z = volume.n_zslices`); projection modes and single-Z files collapse to `Z=1`. File construction matches the Export page's `export_tiff_hyperstack` exactly so the two writers produce identical headers; re-importable into ND2Studios as a multi-channel multi-Z TIFF |

### `plugins/enhancement/builtin.py`
19 enhancement plugins (Normalize, CLAHE, GaussianBlur, MedianFilter,
BackgroundSubtract, GammaCorrection, BleachCorrection,
TemporalFoldCorrection, SpatialFlatness, TopHat, DoG, UnsharpMask,
BilateralDenoise, MorphGradient, LocalContrast, BlobSubtract,
NLMDenoise, WaveletDenoise, TVDenoise) registered via
`@PluginBase.register`. Each declares parameters via `ParamSpec` so
`ParamEditor` can auto-generate a form for it.

### `workers/`

| File | Purpose |
|---|---|
| `base_worker.py` | `BaseWorker(QThread)` with `progress(int)`, `status(str)`, `finished(object)`, `error(str)` signals and `cancel()` |
| `load_worker.py` | `LoadWorker` — opens an ND2 (extended metadata + lazy channels) or TIFF (single-channel TYX) file off the GUI thread |
| `recipe_worker.py` | `RecipeWorker` — applies an ordered recipe to each channel; lazy proxies are materialized once per channel; optional frame-mean normalization first |
| `export_worker.py` | `ExportRequest` dataclass + `ExportWorker` dispatching to `tiff_stack`, `tiff_zstack`, `rgb_composite`, `movie`, or `image_sequence` (V1.32) mode. `ExportRequest` carries an optional `image_adjustments: ImageAdjustments` shared across modes, plus `basename` / `iterate_volume` / `z_mode` / `z_view_index` for image-sequence jobs |
| `stitch_worker.py` (V1.1) | `StitchRequest` + `StitchWorker` wrapping `export_stitched_tiff` with progress / status signals |
| `analysis_worker.py` (V1.19, deprecated V1.37) | `AnalysisWorker(BaseWorker)` — runs any `AnalysisPipeline.run()` in a background thread; passes `progress_cb` and `cancelled_cb` (Qt-free lambda). **Deprecated** in V1.37: new analysis code submits `PipelinePreviewJob` / `PipelineCommitJob` to the project-wide `JobRunner` (see `compute/`). Kept for `BatchWorker` and any out-of-tree callers |
| `batch_worker.py` (V1.23) | `BatchWorker(BaseWorker)` — sequential per-file pipeline runner. Extra signal `file_done(int, int)`. For each file: calls `LoadWorker.run_task()` → `RecipeWorker.run_task()` → `pipeline.run()` → `results_engine.compute_measurements()` synchronously within the thread. Errors per file are caught and reported without aborting. Writes `batch_results.csv` to output dir at end |
| `prefetch_worker.py` (V1.17, V1.35 velocity-aware) | `PrefetchManager(QThread)` — single background thread that fills a `FrameCache` with neighbor T (or Z) frames. Opens its own volume handle inside `run()`. Queue is replaced on each `request_neighbors()` call. Emits `frame_ready(c, m, t, z, z_mode)` via queued signal. Only wired up for the ND2 volume path in `set_volume()`. **V1.35 Phase 3**: constructor accepts `radius_t` / `radius_z`; the manager tracks the previous focus + `time.perf_counter()` timestamp and derives a T-axis velocity from successive `request_neighbors` calls. Forward scrubs (≥ +1 plane/sec) bias the window to `[t - 2, t + radius_t]`; backward scrubs bias the opposite side; a stop falls back to a symmetric `radius_t // 2` window. 1.5 s TTL on the focus history resets the bias after an idle pause. Z-axis fallback (single-T volumes) stays symmetric ±`radius_z`. `cancel_all()` clears the queue without resetting velocity history (mid-scrub mouse release should keep the prediction) |
| `pre_render_worker.py` (V1.0 smooth-playback) | `PreRenderWorker(QThread)` — post-materialization compositor. After `LoadWorker` completes, this worker runs the same LUT+channel-multiply+additive-sum pipeline as `MultiAxisViewer._compose_current_frame()` on a background thread, depositing `(H, W, 3) uint8` numpy arrays into a shared `cache: dict[(m, t) → ndarray]` written by the worker and read by the GUI. Priority order: all T frames for `priority_m` first; emits `finished()` after that series so the viewer can start QPixmap pre-conversion before other M positions complete. Signals: `progress(int)`, `frame_cached(int, int)`, `finished()`, `error(str)`. Memory guard: restricts to `priority_m` when `n_m × n_t > 500` |

`MultiAxisViewer._do_refresh()` four-tier cache (V1.0 smooth-playback + overlay-perf-cache):
1. **GPU path** — pushes per-channel planes to `GpuImageCanvas`; skipped when `_frame_post_process` is set (GPU canvas has no CPU output buffer).
2. **QPixmap cache** (`_pixmap_cache[t]`) — zero numpy work; skipped when `_frame_post_process` is set.
3. **Render cache** (`_render_cache[(m, t)]`) — skips LUT+compose; skipped when `_frame_post_process` is set (base frames do not include overlay).
4. **Post-process cache** (`_pp_cache[(m, t, pp_version)]`) — overlay-composited result; only used when `_frame_post_process` is set. Invalidated by `invalidate_post_process_cache()` (bumps `_pp_version`, clears dict) whenever overlay content changes or base frames are rebuilt.
5. **Render cache + post-process** — if `_render_cache[(m, t)]` is present, copies base and runs the overlay callback, caching the result in `_pp_cache`.
6. **Live compose fallback** — always fills `_render_cache` on-demand and `_pp_cache` if a callback is active.

### `compute/` (V1.37 Phase 5)

Project-wide coordination layer for short, cancellable background work.
Complements (does not replace) the long-running `QThread` workers in
`workers/`. A single `JobRunner` instance lives on `MainWindow.job_runner`
and the Analysis page submits both preview and commit jobs through it.

| File | Purpose |
|---|---|
| `cancellation.py` | `CancellationToken` (`threading.Event`-backed) + `CancelledError`. Exposes `is_cancelled()` (matches `AnalysisPipeline.run(cancelled_cb=…)` contract) and `check()` (raises for cooperative unwind inside `AnalysisJob.run`) |
| `progress.py` | `ProgressReporter(QObject)` — `progress(key, fraction, message)` signal. `as_pipeline_progress_cb()` adapter returns a 0–100-int callable for unmodified `AnalysisPipeline` instances |
| `jobs.py` | `AnalysisJob` ABC (key + token + abstract `run(progress)`) + frozen `JobResult(key, ok, value, error)` dataclass emitted on `job_done` |
| `runner.py` | `JobRunner(QObject)` — `QThreadPool`-backed dispatcher sized by `recommended_worker_count()`. `submit(job)` coalesces by key (latest cancels in-flight predecessor). Public signals `job_done(JobResult)`, `job_cancelled(str)`, `job_progress(str, float, str)`. `shutdown(wait_ms)` drains the pool on `MainWindow.closeEvent`. Uses `_RunnerSignals` sink + `_Runnable(QRunnable)` wrapper that finalises by removing the job from the active-map and disconnecting the per-job `ProgressReporter` |
| `result_store.py` | `ResultStore` — `RLock`-guarded keyed dict for caching preview/commit outputs. Phase 6 will sit a Zarr store in front of it for bounded-RAM commits |
| `pipeline_jobs.py` | `PipelinePreviewJob` (runs an `AnalysisPipeline` on a `(1, H, W)` frame; key `"analysis_preview"`) and `PipelineCommitJob` (runs on a full `(T, H, W)` channel stack for one M position; key `"analysis_commit"`). Both pass `token.is_cancelled` as the pipeline's `cancelled_cb` and `progress.as_pipeline_progress_cb()` as `progress_cb`, so concrete pipelines (Histogram Threshold, Nuclei, Tear, Spots, Mask Analysis) need no changes. **V1.39 Phase 7**: also exports `BuildPyramidJob` which wraps `PyramidStage.build()` into the same coalescing dispatcher under key `"pyramid_build"` |
| `gpu/` (V1.39 Phase 7) | Optional GPU dispatch shim for the four analysis hotspots (`gaussian`, `gaussian_filter`, `gaussian_laplace`, `threshold_otsu`). `detect.py` reports `{available, reason, device_name, memory_gb, cucim}` via `gpu_status()` without raising; `log_gpu_status()` emits one startup info line. `array.py` owns the runtime `_USE_GPU` flag — `configure(True/False)` returns the *effective* value so a request on a CPU-only host stays False, the env override `ND2_DISABLE_GPU_ANALYSIS=1` pins it to False, and `should_dispatch_to_gpu(arr)` enforces a 256² array-size floor to keep tiny ops on CPU. `ops.py` exports the four scipy/skimage-compatible wrappers; each catches `OutOfMemoryError`, missing `cucim`, and driver-level errors, logs one warning per op name on the first failure, and falls back to the CPU implementation. The display GPU path (pyqtgraph in V1.36) and this analysis GPU path are intentionally independent — they can fail independently and the rest of the app keeps working |
| `parallel/` (V1.40 Phase Addendum) | Stdlib-only parallel-compute helpers for analysis and pipeline code. `shared_array.py` — `shared_ndarray(arr)` context manager exposes a numpy buffer to child processes via `multiprocessing.shared_memory` (yields `(name, shape, dtype_str)`) so per-task pickle cost stays out of the hot loop; `attach_shared(name, shape, dtype_str)` is the worker-side reconstructor (workers must `close()` but not `unlink()`). `process_map.py` — `process_map_planes(stack, fn_module, fn_name, indices, kwargs, n_workers)` drives a `ProcessPoolExecutor` for pure-Python CPU-bound per-plane functions; the worker function must be importable (module-level) because each child re-imports it. `thread_map.py` — `thread_map_planes(stack, fn, indices, kwargs, n_workers, progress_cb, cancelled_cb)` drives a `ThreadPoolExecutor` for C-extension work (scipy / scikit-image / numpy / tifffile / zarr all release the GIL). The decision rule is "threads for C-extension, processes for pure Python"; ND2Studios's analysis pipelines all fit the threads case |

`AnalysisPage` wiring:
- `param_editor.params_changed` and `viewer.coords_changed` both restart
  the 300 ms `_screen_debounce` timer; firing builds a `PipelinePreviewJob`
  and submits it. Auto-screen gates the param-driven path so the user can
  opt out.
- "Run Analysis" / per-M state machine builds a `PipelineCommitJob` per
  M. Cancel calls `runner.cancel("analysis_commit")` and clears the queue.
- The page connects to `job_done` once (per-page lambda dispatch in
  `_on_runner_done`) and filters by `result.key`.

### `pipeline/` (V1.38 Phase 6)

Per-source workspace + stage commits. Qt-free; sits next to `compute/`.
A single `Session` lives on `MainWindow.session` and is attached by the
Import page after a successful load. Recipe and Analysis pages flush
committed outputs to the workspace; the page-leave hook in
`MainWindow._navigate` then drops the in-memory copies so RAM is bounded
across the workflow.

| File | Purpose |
|---|---|
| `session.py` | `Session` owns one workspace dir keyed by `hash_source_file(path)` (sha-256 of size + mtime + first/last MiB; truncated to 16 hex chars). Default root `~/.nd2studios/workspace/sessions/`, overridable via `ND2STUDIOS_WORKSPACE`. `SessionManifest` + `StageRecord` dataclasses → `manifest.json`. `set_shape()` stamps `n_t`/`n_m`/`n_z`/dims/pixel_size after import. `record_stage()` / `is_committed()` / `archive()` for the Import page's "Start fresh" branch. `workspace_disabled()` honours `ND2STUDIOS_DISABLE_WORKSPACE=1` |
| `stage.py` | `PipelineStage` ABC. Subclasses implement `commit()` returning a `StageRecord`. Stages do not own threads; commits run on the GUI thread after a worker has produced the heavy data |
| `storage.py` | `write_label_stack(path, arr)` / `read_label_stack(path)` / `open_label_stack(path)`. Picks Zarr (chunked `(1, H, W)`, Blosc/zstd-3) when `import zarr` succeeds, falls back to NPZ otherwise. `HAS_ZARR` is exposed for callers |
| `stages/recipe_stage.py` | `RecipeStage` — parameter-only commit. Writes the accepted recipe to `recipe/recipe.json` and stamps the manifest. `EnhancedDataset` is a dict-like proxy over raw channels + recipe; `materialize_channel(name)` runs the plugin chain via `PluginBase` and caches per-channel results so peak RAM during re-derivation is one channel rather than `n_channels`. No `processed.zarr` is materialized by default — the recipe is small (JSON) and cheap to reapply. **V1.40 Phase Addendum**: `materialize_all()` fans out across channels with a `ThreadPoolExecutor` capped at `min(channels, recommended_worker_count())`; `materialize_channel()` guards the cache read/write with a `threading.Lock` so concurrent callers on the same channel don't both compute. `clear_cache()` is locked too. The per-channel work (skimage / opencv / scipy) releases the GIL so the speed-up is real |
| `stages/analysis_stage.py` | `AnalysisStage(session, pipeline_name)` — per-pipeline stage. `commit_m(m, AnalysisResult)` writes label masks (primary + secondary) to `analysis/<pipeline>/m_NNN/labels_<channel>.{zarr,npz}` and a `summary.json` per M with measurements + overlay defaults + serialised `volumetric_voxel_counts`. `commit()` stamps the manifest record after the queue drains. `committed_m_indices()` / `rehydrate_m(m)` / `open_label_lazy(m, channel)` are the read paths used by the Results page and the Import page's warm-start flow |
| `stages/pyramid_stage.py` (V1.39 Phase 7, parallelised V1.40) | `PyramidStage(session)` — multi-resolution display tier, keyed off the V1.38 `Session` hash. Builds a `pyramid.zarr` group under `<session_dir>/pyramid/` with one chunked `(M, T, Z, C, H_L, W_L)` ZarrArray per level, chunks `(1, 1, 1, 1, H_L, W_L)`. **Level 0 is the source ND2 itself** (read by `LazyND2Volume`); the pyramid stores levels 1..N only, capping disk overhead at ≈ 33 % of source size. `build(volume, progress_cb, cancel_cb)` is called from inside a `BuildPyramidJob` so it gets cancellation + progress + coalescing for free; a cancelled build leaves a partial zarr on disk and is not committed to the manifest. `PyramidReader(store_path)` is the read interface — `get_frame(level, c, m, t, z, z_mode)` mirrors `LazyND2Volume.get_frame()` so the viewer's dispatch site needs one extra argument, and `pick_level_for_viewport(screen_px, image_pixels_visible)` returns the smallest level whose width still ≥ the visible image-pixel extent. `default_pyramid_levels(h, w)` picks `log2(max_dim/256)` clamped to `[2, 6]`. Raises `PyramidUnavailable` when `zarr` is missing or the source is already ≤ 256 px. **V1.40 Phase Addendum**: the per-level M×T×Z×C nested loop runs in a `ThreadPoolExecutor` sized to `recommended_worker_count()`. Each worker holds a thread-local `volume.reopen()` handle (the `nd2` SDK is not thread-safe across handles); the handles are tracked in a lock-guarded registry and closed deterministically in the build's `finally` block since the executor recycles threads. Distinct `(m, t, z, c)` writes go to distinct zarr chunks (DirectoryStore files) so concurrent writes don't contend. Progress is reported through `_AtomicCounter` so the 0–100 bar stays monotonic across workers. Cancellation runs through a `threading.Event`; in-flight planes finish, the rest are abandoned, partial Zarr is left for the next build to overwrite |

`MainWindow` ↔ `pipeline/` integration:
- `attach_session_for(filepath)` (called from `ImportPage._on_confirm`)
  creates or reuses the workspace and stamps the manifest shape.
- `recipe_stage()` / `analysis_stage(name)` are convenience constructors
  the pages call instead of importing the stages directly.
- `_release_outgoing_stage(page_key)` (called from `_navigate` after
  `save_to_experiment`): leaving `recipe` after a committed recipe
  swaps `exp._processed_channels` for an `EnhancedDataset` proxy
  (`exp._processed_view`) and drops the dict; leaving `analysis`
  clears `label_masks` / `secondary_label_masks` on every result whose
  M position is committed.
- `ND2StudiosRecord.processed_view()` returns the in-RAM dict if
  present, else the `EnhancedDataset` proxy, else `_raw_channels` —
  the Export, Analysis, and Results pages call this instead of
  reading `_processed_channels` directly so they work transparently
  after a release.

**V1.39 Phase 7** adds three more `MainWindow` helpers and one slot
that sit alongside the V1.38 ones:

- `pyramid_stage()` — returns a `PyramidStage` for the active
  session, or `None` when no session is attached, when
  `ND2_DISABLE_PYRAMIDS=1`, or when `zarr` is missing. Mirrors
  the `recipe_stage()` / `analysis_stage(name)` convention.
- `start_pyramid_build(volume)` — submits a `BuildPyramidJob` for
  the active source (coalesce key `"pyramid_build"`; submitting
  again cancels the in-flight build). Reattaches the existing
  reader when a prior commit is on disk for this source hash —
  cross-launch resumption comes for free via the V1.38 `Session`
  key. Called from `ImportPage._kick_off_pyramid` after the
  workspace is attached, and again from the Performance dialog's
  "Rebuild pyramid for current file" button.
- `_attach_pyramid_reader_to_viewers(reader)` — pushes the reader
  into every page that exposes a `viewer` with an `attach_pyramid`
  method. Today only the multi-axis viewer qualifies; the scan is
  page-agnostic so future viewers benefit automatically.
- `_on_pyramid_job_done(result)` — connected once to
  `JobRunner.job_done`; filters by `result.key == "pyramid_build"`
  so analysis-page slots are unaffected. On success, instantiates
  a fresh `PyramidReader` and pushes it through
  `_attach_pyramid_reader_to_viewers`; on failure, surfaces the
  error in the status text.

The sidebar gains a new `⚙ Performance` button (next to New / Save /
Load) that opens `_open_performance_settings()` — a modal with two
checkboxes (GPU analysis, build pyramids) and a "Rebuild pyramid for
current file" button. OK pushes the values onto
`Settings.USE_GPU_ANALYSIS` / `Settings.BUILD_PYRAMIDS` and calls
`compute.gpu.configure()`; the dialog disables and tooltips
the GPU checkbox on CPU-only hosts and the pyramid checkbox when
`zarr` is missing.

### `pages/`

| File | Page | Responsibilities |
|---|---|---|
| `import_page.py` | Import | File browse, extended metadata table, **read-only** per-channel info rows, Z-mode + frame-stride config, multi-axis preview (M/T/Z sliders + per-channel toggle/color/LUT), Confirm Import (sets status `imported`), Stitch M… button → `StitchDialog`. **V1.38 Phase 6**: `_on_confirm` calls `MainWindow.attach_session_for(filepath)` to create or reuse the per-source workspace; if prior commits exist, prompts Resume / Start fresh. Resume rehydrates the recipe (promoting status to `preprocessed`) and per-pipeline `AnalysisResult`s without rerunning anything; Start fresh archives the prior workspace dir to `<hash>.archived-<timestamp>`. **V1.39 Phase 7**: `_kick_off_pyramid(exp)` runs after workspace attach — if `Settings.BUILD_PYRAMIDS` is on, `zarr` is installed, and the source has no committed pyramid, submits a `BuildPyramidJob`; if a pyramid is already on disk for this source hash, attaches the reader to the viewer without rebuilding. The build is fire-and-forget — the viewer reads from level 0 until it completes |
| `recipe_page.py` | Recipe | Plugin picker, parameter editor, trial/accept/reject pipeline, save/load `.nd2s_recipe.json`, side-by-side raw vs processed `MultiAxisViewer`s (T scrolling only on this page). **V1.38 Phase 6**: `_on_accept` also calls `_commit_recipe_stage()` to flush the canonical recipe to the workspace at every Accept; the actual `_processed_channels` release happens later in `MainWindow._navigate`'s page-leave hook so the right viewer stays live. `on_activated()` rehydrates from `exp._processed_view.materialize_all()` when the user returns from a downstream page |
| `export_page.py` | Export | Source selector (raw/processed), four tabs (TIFF / Composite / Movie / Image Sequence) each running the appropriate `ExportWorker` job. V1.32: clicking "Export Movie…" or "Preview & Export Image Sequence…" opens an `ExportPreviewDialog` (T scrubber + brightness / contrast / saturation / hue / fade sliders) before the worker is started. The Image Sequence tab adds a base-name line edit (auto-populated from the loaded filename) and an "Iterate all M / Z positions" checkbox (enabled only when the file actually has multi-M or unprojected multi-Z); filenames are `{basename}_T01[_M02][_Z03].png`. **V1.38 Phase 6**: `on_activated` and `_channels_to_export` use `exp.has_processed()` / `exp.processed_view()` so the "Processed" radio stays enabled after a Recipe → Export navigation that released RAM; per-channel materialization happens at write time |
| `analysis_page.py` (V1.19, layout V1.20, runner V1.37, workspace V1.38) | Analysis | Horizontal splitter: left panel (pipeline selector + `ParamEditor` + run row + results widget), right panel (`MultiAxisViewer`). Viewer wired to exp on `on_activated()` so the file is immediately visible for parameter tweaking. **V1.37 Phase 5**: both preview ("Screen frame" / Auto-screen) and commit ("Run Analysis") work submit through `MainWindow.job_runner` as `PipelinePreviewJob` / `PipelineCommitJob`. `params_changed` and viewer-coord changes restart a 300 ms debounce that fires a preview under key `"analysis_preview"` — fresh submissions cancel in-flight predecessors. Cancel calls `runner.cancel("analysis_commit")`. Results overlaid via `set_frame_post_process()`; results widget hidden until first commit completes. **V1.38 Phase 6**: `_on_finished` also calls `_commit_analysis_m(pipeline, m, result)` after every per-M completion, streaming label masks (primary + secondary) to the workspace as the run queue drains. `_finalize_analysis_stage()` stamps the manifest record once the queue is empty. `on_activated()` materializes any released `EnhancedDataset` proxy before handing channels to the viewer |
| `results_page.py` (V1.22, V1.38 rehydrate) | Results | Pipeline selector (populated from `exp.analysis_results` keys), "Compute Measurements" button triggers `results_engine.compute_measurements()` synchronously on the GUI thread. Measurements shown in a `QTableView` backed by a custom `QAbstractTableModel` with `QSortFilterProxyModel` for column sorting. Summary panel: n_objects, n_frames, mean/std area_um², dynamic per-channel mean intensity rows. Export row: CSV (`csv.DictWriter`), Label Masks TIFF (`export_label_masks_tiff`), Overlay Images TIFF/JPG (`export_overlay_frames`). Empty-state label shown until "Compute" is clicked. **V1.38 Phase 6**: `_on_compute` calls `_rehydrate_released_label_masks(pipeline, results_by_m)` before `compute_measurements` so released analyses (label masks dropped after a Recipe / Analysis page navigation) are silently refilled from the workspace |
| `batch_page.py` (V1.23) | Batch | Template path + browse/save controls; file queue (`QListWidget`, add files / add folder / clear / remove-selected); output directory picker; export-images checkbox + format combo; Run/Cancel buttons; progress bar + file counter + status label. Kicks off `BatchWorker`; on `finished` offers to open the output folder |
| `stitch_dialog.py` (V1.1) | dialog | Layout preview (matplotlib), per-tile / per-channel selectors, Z mode / Z slice picker, output path picker, kicks off `StitchWorker`. Launched from the Import page |

### `utils/` (V1.33, expanded V1.34)

| File | Purpose |
|---|---|
| `profiling.py` (V1.33) | Backend-pure (no Qt) helpers used by the `profiling/` harness. `Measurement` dataclass records wall / CPU ms, RSS before / after / peak, Python-allocation peak, and an `extra` dict; `measure(name)` context manager wraps a block of code with `time.perf_counter` + `tracemalloc` + `psutil`; `fps_from_durations(list)` summarises per-frame ms into mean / p50 / p99 FPS and latency. Reusable from worker tests and future optimization-phase scripts |
| `resources.py` (V1.34 Phase 2) | Backend-pure (no Qt). Single source of truth for host CPU / RAM. `SystemResources` (frozen dataclass: `total_ram_bytes`, `available_ram_bytes`, `cpu_count_physical`, `cpu_count_logical` + `*_gb` properties); `detect()` snapshots via `psutil`; `recommended_cache_budget_bytes(reserve_fraction=0.6)` reserves 60% of *available* RAM for OS / other workloads and hands the remainder to caches (floored at 64 MB); `recommended_worker_count()` returns physical cores capped at 8. Wired into `MultiAxisViewer.__init__` for the `FrameCache` budget and ready for Phase 3 / 5 / 6 / 7 |
| `threading.py` (V1.34 Phase 2; V1.40 Phase Addendum) | **Qt boundary** — only file under `utils/` that imports PySide6. `PlaneRequest` dataclass (priority + monotonic `request_id` + `(c, m, t, z, z_mode, z_start, z_end)`; `cache_key` property matches the `FrameCache` key shape); `IOWorker(QObject)` runs in a dedicated `QThread`, holds its own `volume.reopen()` handle (the `nd2` library's per-file state is not thread-safe), services a `PriorityQueue[PlaneRequest]` and emits `plane_ready(request_id, key, ndarray)` / `error(request_id, msg)` with queued-connection delivery to the GUI; planes are normalised to 2-D on the worker thread so the GUI slot stays trivial. `start_io_worker(volume) -> (worker, thread)` does the `moveToThread` + `thread.started.connect(worker.run_loop)` wiring. Used by `MultiAxisViewer` for the foreground cache-miss read path so the `PrefetchManager` can stay focused on speculative ±5 T neighbor reads. **V1.40**: `IOWorker.__init__` gains an optional `shared_queue` parameter; new `WorkerGroup` facade exposes a single `.submit` / `.cancel_all` / `.stop` surface over N readers draining one priority queue; `start_io_workers(volume, n)` spawns *n* workers (one per dedicated `QThread`) sharing one queue and returns `(WorkerGroup, [QThread])`. `n=1` is equivalent to `start_io_worker`; callers pick the count via `utils.storage.recommended_io_thread_count(path)` |
| `storage.py` (V1.40 Phase Addendum) | Backend-pure storage-class detection. `is_fast_storage(path)` returns `True` for NVMe / SSD-class storage by consulting `/sys/block/<dev>/queue/rotational` on Linux; conservative `False` on Windows / macOS and on any failure (the V1.34 single-reader default is the safe fallback). `recommended_io_thread_count(path)` returns 2 for NVMe-class storage and 1 elsewhere, capped at 2 because the queue-depth benefit plateaus past two file handles and the prefetcher's cache hit rate starves additional readers anyway |

### `profiling/` (V1.33 — top-level, not part of `nd2studios/`)

| File | Purpose |
|---|---|
| `harness/fixtures.py` | `TestFile` records, `DATA_DIR` (overridable via `PROFILING_TEST_DATA_DIR`), scrub / tab-switch counts, synthetic dimensions for the analysis scenario |
| `harness/scenario_load.py` | Times cold `LazyND2Volume(...)` / `LazyMultiFileTIFFVolume([...])` construction + first `get_frame(c=0, m=0, t=0, z=0)`. Skips files missing on disk |
| `harness/scenario_scrub.py` | Walks T / Z / M positions one-at-a-time via `volume.get_frame(...)` and reports `fps_from_durations` per axis. Paced at 16 ms per step to match a 60 Hz slider |
| `harness/scenario_tab_switch.py` | Forces `QT_QPA_PLATFORM=offscreen`, constructs `MainWindow`, force-promotes the active experiment status to bypass `PAGE_PREREQS`, then walks every key in `Settings.PAGES` 12 times via `_navigate()` with `app.processEvents()` between each |
| `harness/scenario_analysis.py` | Generates a deterministic 32 × 256 × 256 uint16 channel + a brighter "Foci" companion, then runs every registered `AnalysisPipeline` via its registry entry. `Nuclei Segmentation` failure (Cellpose missing) is recorded as an `error` field and the run continues |
| `harness/run_all.py` | Aggregator. Embeds platform / CPU / RAM info from `psutil`, runs all four scenarios, writes one `profiling/baselines/<label>.json`. Default label `phase_00_baseline` |
| `baselines/` | One JSON snapshot per phase (gitignored). Use this to diff before / after each optimization phase |
| `reports/` | Optional py-spy flamegraphs (gitignored). `README.md` documents the capture commands |
| `README.md` | How to run the harness, point it at real lab files, capture flamegraphs, and compare snapshots across phases |

## Data Flow

```
ND2 / TIFF on disk
   │
   │  ImportPage._on_browse        → single file → LoadWorker
   │  ImportPage._on_reconstruct   → ReconstructDialog (V1.28)
   │     ↳ user picks N files + chain_axis ∈ {T, M, Z, C}
   │     ↳ LoadWorker(filepaths=…, chain_axis=…) → LazyMultiFile{ND2,TIFF}Volume
   │       (axis-agnostic; bisect-routed; each file contributes its
   │        native chain-axis size)
   ▼
ND2StudiosRecord._raw_volume     (LazyND2Volume | LazyMultiFileND2Volume | LazyMultiFileTIFFVolume)
ND2StudiosRecord._raw_channels   (Dict[name, LazyND2Channel | MultiFileLazyChannel | LazyTIFFChannel])    ← initial M=0
ND2StudiosRecord.nd2_metadata    (full extended metadata dict)
   │
   │  Viewer scrolls M/T/Z; user toggles channels; user drags LUT.
   │  ImportPage Confirm → rebuilds _raw_channels via
   │  LazyND2Volume.all_channels_as_lazy(m, z_mode, z_index)
   ▼
ND2StudiosRecord._raw_channels   (Dict[name, LazyND2Channel])    ← chosen M/Z
   │
   │  RecipePage crop drag/click → _apply_crop()
   ▼
ND2StudiosRecord._original_raw_channels  (Dict[name, lazy|ndarray])  ← pre-crop snapshot
ND2StudiosRecord._raw_channels           (Dict[name, lazy|ndarray])  ← cropped view
ND2StudiosRecord.crop_rect               (x, y, w, h) | None
   │  Re-crop always applies to _original_raw_channels (non-destructive).
   │  Reset Crop restores _raw_channels ← _original_raw_channels.
   │  Snapshot persists on the record; survives tab navigation for all sources.
   │
   │  RecipePage trial/accept → RecipeWorker
   ▼
ND2StudiosRecord._processed_channels  (Dict[str, ndarray])
ND2StudiosRecord.recipe              (List[(plugin_name, params)])
   │
   │  ExportPage tab → ExportWorker → backend/exporters/
   ▼
.tif (single multi-channel TZCYX ImageJ hyperstack, V1.29)
.tif (RGB composite) | .mp4 / .gif on disk

Side branch (Import page → Stitch M…):
   _raw_volume → StitchDialog → StitchWorker → export_stitched_tiff()
   → multi-page TIFF (single-channel or RGB composite). The original
   ND2 is never touched.

Side branch (Analysis page — V1.19):
   _processed_channels (or _raw_channels) → AnalysisPage._on_run()
   → materialized ndarray → AnalysisWorker → AnalysisPipeline.run()
   → AnalysisResult {label_masks, measurements, summary}
   → exp.analysis_results[pipeline_name] = result
   → page overlay renderer + summary table
   → "Export Label Masks" → export_tiff_stack(..., bit_depth="passthrough")
   → "Export Measurements" → csv.DictWriter → .csv
```

## External Dependencies

| Package | Purpose | Notes |
|---|---|---|
| PySide6 | GUI framework | Qt 6 bindings |
| numpy, scipy, pandas | numerical core | core array ops |
| nd2 | ND2 I/O with full metadata | Talley Lambert's library |
| tifffile | TIFF stack export | per-page write, BigTIFF |
| scikit-image, opencv-python, PyWavelets | image processing | used by enhancement plugins |
| matplotlib | plots, scale bars | embedded via `MplCanvas` |
| imageio + imageio-ffmpeg | MP4 / GIF export | libx264 codec for MP4 |
| Pillow | overlay rendering | scale bar / timestamp / labels |
| certifi | SSL bundle | optional, for plugins that pull online resources |
| numba | JIT kernels | required by the vendored `backend/serialtrack` matcher (the Track Objects node's SerialTrack method); first call compiles + caches |

| cellpose (optional, V1.19) | Nuclei segmentation | `pip install cellpose`; detected at runtime via `importlib.util.find_spec`. Absent = friendly error dialog, not a crash |
| stardist + tensorflow + csbdeep (optional, V1.45) | StarDist segmentation node | `pip install stardist tensorflow csbdeep`; detected via `importlib.util.find_spec`. Absent = friendly `ImportError` in the node, not a crash. TF imported lazily on first model load; threading locked single inter/intra-op to avoid worker-thread deadlock |

TensorFlow / StarDist / csbdeep are **optional** dependencies — required only by
the StarDist segmentation node (V1.45). They are not in `requirements.txt`; the
node raises a friendly `ImportError` when they are missing. Object linking is
provided by the in-tree `backend/object_tracker` (centroid linker) plus a vendored,
headless subset of SerialTrack at `backend/serialtrack/` (topology PTV;
9 modules, no PySide6/h5py/tifffile-bound code) — see
`Research/serialtrack.md`. A second vendored subset, **CellTracker** at
`backend/celltracker/` (`tracking`, `metrics`, `fields`, `segmentation`; pure
numpy/scipy/pandas + lazy stardist/TF, no Qt), supplies the topology-Hungarian /
spatial-fingerprint linkers, per-cell spatial metrics + self-fold-change, Eulerian
field maps, and StarDist segmentation — adapted to/from ND2Studios' row-dicts by
`backend/celltracker_bridge.py` (analyses) and the StarDist analysis node. Cellpose
and StarDist are optional analysis dependencies; the rest of the app runs without
them.

## Design Decisions

1. **Sibling, not fork.** ND2Studios copies backend code from
   CellTracker rather than importing it. Keeps the two packages
   independent at the cost of a small duplication tax. If duplication
   passes ~30%, extract a shared `mcghee_lab_imaging` package.
2. **Backend purity.** Every module under `nd2studios/backend/` and
   `nd2studios/plugins/` is Qt-free. They take data + params + an
   optional `progress_cb`; nothing else. This makes them trivially
   testable and trivially callable from a future headless mode.
3. **`LazyND2Channel` everywhere on import.** ND2 files are routinely
   tens of GB. The viewer pulls one frame at a time. Materialization
   only happens when a worker needs a contiguous array (recipe
   application, export).
4. **Trial / Accept / Reject.** The committed recipe is the source of
   truth. A trial step runs `committed + trial` against the raw data
   and parks the result on `_processed_channels`; Accept appends the
   trial step to the recipe; Reject reverts by re-running just the
   committed recipe.
5. **PyDracula chrome ported into Python.** No `.ui` files. The
   collapsible animated sidebar with text labels and the custom
   frameless title bar are reimplemented as plain `QWidget`s, matching
   CellTracker's authoring style.
6. **Single dark theme.** Theme switching is out of scope for V1.0. The
   QSS is structured (palette via `Settings.*` constants, `#bgApp`
   wrapper) so a light theme can be added later without restructuring.
7. **`.nd2s` session = JSON manifest + `_arrays.npz`.** Mirrors
   CellTracker's `.cta`. Channels and frame timestamps are stored as
   compressed NPZ; everything else is human-readable JSON.
8. **Movies via `imageio`.** Avoids OpenCV's licensing complications
   for video encoding. `imageio-ffmpeg` is a binary distribution that
   ships its own `ffmpeg`.
9. **Thread-pool parallelism, no Dask / Numba (V1.40).** The Phase
   Addendum chose `concurrent.futures.ThreadPoolExecutor` over Dask
   for materialization and pyramid build because every ND2Studios
   hot kernel already sits on a GIL-releasing C extension (scipy /
   scikit-image / numpy / tifffile / zarr) and Dask would be a
   multi-MB dependency for one well-shaped loop. Numba is rejected
   on the same cost-benefit grounds — no remaining Python inner
   loop is hot enough to justify the JIT-cache footprint and the
   first-call compile pause. The pattern scales near-linearly with
   cores on the workloads that matter; revisit if 3.13t lands or
   if Dask is needed for a future feature.

## Known limitations / V1.1+ candidates

- TIFF loader only handles single-channel TYX. Multi-channel TIFFs
  need a `dim_order` picker in the Import page.
- No autosave. CellTracker's autosave is keyed to expensive detection
  runs; our recipes are fast enough that explicit Save/Load suffices.
- Theme switching is intentionally absent.
- Headless / HPC mode is wired into the directory tree (`hpc/`,
  `scripts/`) but not exposed on the GUI; `recipes.py` is already
  designed to be applied from a headless runner.
- Channel-aware multi-channel TIFF input (most multi-channel TIFFs
  follow `TCYX` or `TZCYX`).
