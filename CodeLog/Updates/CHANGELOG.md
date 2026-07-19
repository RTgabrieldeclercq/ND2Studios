# Changelog

All notable changes to ND2Studios will be documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/)

## [Unreleased] - 2026-07-17 (V1.79 — Bake DVC export into the Output node)

The V1.76 manual **Export** button in the DVC viewer is replaced by an **automatic
save baked into the pipeline's Output node**. Give the **DVC node** an output port,
wire it into an **Output node**, and on **Run** the Output node writes the full DVC
backend data (the `.nd2dvc` bundle — every multipoint / frame / object + the
quantitative displacement/strain fields) to a **folder named after that Output node**
under `<repo>/results/`, plus a `provenance.json` sidecar. Reload is unchanged (the
**Import DVC results** button + DVC Checkpoint node still read a `.nd2dvc`). Design:
`CodeLog/ClaudesPlan/V1.79_output_node_autosave.md`.

### Added

- **Output-node auto-save** (`pages/pipelines_page.py`): `_run_execute_node`'s OUTPUT
  branch now routes to a new `_run_output(node)`. When the node wired into the Output
  node is a **DVC** node (and DVC results exist), it gathers the bundle payload
  (`_gather_dvc_bundle_payload`, reused from V1.76), makes
  `Settings.PROJECT_DIR/results/<sanitized Output-node title>/`, and saves
  `<title>.nd2dvc` + `provenance.json` **off-thread** via `DVCExportWorker`, finishing
  the node from the worker's done slot (`_on_run_output_saved` / `_on_run_output_error`).
  Helpers: `_output_source_node` (the node feeding the Output node, DVC preferred),
  `_output_node_dir`, `_safe_output_name`. Non-DVC Output nodes remain the unchanged
  pass-through.

### Changed

- **DVC node gains an output port** (`pipeline_graph/registry_adapter.py`): the
  `SPECIAL_DVC_OP_KEY` spec's `output_types` is now `[PortType.ANY]` (was `[]`), so it
  connects to an Output node (ANY→BINARY via the `can_connect` wildcard), like
  Registration / Granule Volume Mask. *Existing* DVC nodes in a saved pipeline must be
  deleted and re-added to gain the port (ports are built at node creation).
- **`_run_execute_node`** (`pages/pipelines_page.py`): the combined
  `INPUT or OUTPUT → _run_finish_node` branch is split — INPUT still passes through,
  OUTPUT routes to `_run_output`.

### Removed

- **The manual Export button in the DVC viewer** (`widgets/dvc_panel.py`): the
  `export_requested` Signal and `btn_export` are gone (export is now automatic via the
  Output node). The page's `_ensure_dvc_panel` / `_ensure_dic_panel`
  `export_requested.connect(...)` wirings and the now-unused `_export_dvc_bundle` /
  `_on_dvc_export_finished` / `_on_dvc_export_error` handlers are removed. Import /
  reload is untouched.

### Notes / limits

- Auto-save is DVC-specific by design (the connected node must be a DVC node); other
  Output nodes are unaffected and the existing Save Data / Export nodes still cover
  analysis masks / measurements. Verified headless (connectivity + folder round-trip +
  the existing bundle tests); **live-GUI QA outstanding**.

## [V1.78] - 2026-07-14 (V1.78 — 2D DIC (pyALDIC) nodes)

Three new Pipelines nodes bring **2D Digital Image Correlation** to ND2Studios,
built on the optional open-source [`al-dic`](https://pypi.org/project/al-dic/)
package (pyALDIC — 2D Augmented-Lagrangian DIC, Yang & Bhattacharya 2019; the
same AL-DIC lineage as the 3D ALDVC node). `DIC (pyALDIC)` correlates the wired
channel's frames into a dense in-plane displacement + strain field series and
opens a dedicated **DIC** viewer tab; `DIC Mesh Region` and `DIC Mesh Refinement`
reproduce pyALDIC's README "mesh drawing" workflows (draw the ROI/mesh domain;
paint an adaptive-quadtree refinement brush). Maximum reuse: the node returns a
2D `DVCResult` rendered by the **existing** `DVCPanel`, and honors upstream Crop /
Registration / Exclude / channel wiring exactly like the DVC node. `al-dic` is an
**optional, lazily-imported** extra (`pip install al-dic`; never in
`requirements.txt`) — running the node without it shows a friendly message, not a
crash. Plan: `CodeLog/ClaudesPlan/V1.78_pyALDIC_DIC_nodes.md`; method review:
`Research/aldic_pyaldic_review.md`.

### Added

- **`special:dic` node — "DIC (pyALDIC)"** (`SPECIAL_DIC_OP_KEY`,
  `registry_adapter.py`) — orange `NodeCategory.SPECIAL`, `ShapeKind.HEXAGON`,
  `input_types=[IMAGE]` (rainbow `ch_in`/`ch_out`), terminal (`output_types=[]`).
  Params (`param_specs_for`): `tracking_mode` (cumulative→accumulative /
  incremental), `ref_frame` (visible when cumulative), `all_multipoints`,
  `downsample`, then `PyALDICMethod().get_params()` — `winsize`, `winstepsize`,
  `winsize_min`, `init_guess_mode`, `admm_max_iter`, `icgn_max_iter`, `mu`, `tol`,
  `disp_smoothness`, `strain_smoothness`, `compute_strain`.
- **`special:dic_roi` node — "DIC Mesh Region"** and **`special:dic_refine` —
  "DIC Mesh Refinement"** — pass-throughs (`IMAGE` in → `ANY` out) wired upstream
  of a DIC node. Region params: `mesh_preview_step`, hidden `roi_shapes`.
  Refinement params: `refine_mask_boundary`, `refine_roi_edge`, `refine_brush`,
  `min_element_size`, hidden `brush_shapes`.
- **`backend/dic/` package** (Qt-free): `method.py`
  (`@DVCMethod.register class PyALDICMethod`, `name="pyALDIC"`), `engine.py`
  (`run_pyaldic_series` / `run_pyaldic_pair`, `al_dic_available`, and the
  pyALDIC-`PipelineResult`→`DVCResult` **regular-grid adapter** — resamples the
  adaptive FE-mesh displacement via `scipy.griddata`, swapping the `(x,y)`/`[u,v]`
  convention to `DVCResult`'s `(y,x)`/`[dy,dx]`, displacement in pixels + µm via
  `voxel_size_um`), `roi.py` (`build_roi_mask` — replays a serializable
  rect/ellipse/circle/polygon/brush/invert/clear action list into a boolean mask,
  mirroring pyALDIC's `ROIController` Add/Cut semantics).
- **`DICMeshEditorDialog`** (`widgets/node_board/dic_mesh_editor_dialog.py`) — a
  modal editor over the wired channel's processed 2D frame reproducing pyALDIC's
  ROI-toolbar flow: Add/Cut × Rectangle/Circle/Polygon + Refine Brush (radius) +
  Invert/Clear/Undo, a live tinted overlay, and a mesh-node dot preview (ROI
  mode). Built on the reused `ImageCanvas`; returns an ordered shape list.
- **"DIC" overlay tab** in the Pipelines viewer (`pipelines_page.py`) — reuses
  `DVCPanel` (`_ensure_dic_panel` / `_populate_dic_panel`); gated on
  `self._dic_series_by_m`; independent of the DVC tab so both can hold results.

### Changed

- **`__main__.py`** force-imports `nd2studios.backend.dic.method` so
  `PyALDICMethod` registers at startup (next to the ALDVC force-import).
- **`pipeline_graph/__init__.py`** exports `SPECIAL_DIC_OP_KEY`,
  `SPECIAL_DIC_ROI_OP_KEY`, `SPECIAL_DIC_REFINE_OP_KEY`.
- **`pages/pipelines_page.py`** — `_RUN_DIC_KEY`; `_dic_*` state
  (`_dic_series_by_m` / `_dic_incr_by_m` / `_dic_bg_by_mt`, `_dic_roi_by_m`,
  `_dic_refine_by_m`); `_DICJob` (2D sibling of `_DVCJob` — projected +
  registered + cropped + excluded frame reads, whole-series `run_pyaldic_series`);
  `_run_dic` / `_finish_dic` / `_run_dic_region` / `_edit_dic_region`; Run
  dispatch, result routing, cancel + progress wiring, and popup "Edit…" wiring.

### Notes

- Optional dependency: `pip install al-dic` (pulls numba + PySide6≥6.6);
  intentionally **not** in `requirements.txt`.
- No `io.py` change — the new special ops round-trip automatically
  (`_normalize_special_categories` re-derives category from `special_specs()`).
- DIC does not yet participate in the DVC param-sweep Loop feature (deferred).

## [Unreleased] - 2026-07-14 (V1.77 — Prism channel-overlay node + view-only edges)

A new **Prism** node (`special:prism`) for the Pipelines node board, drawn as a
"cool-looking" **2.5D faceted gem** (new `ShapeKind.GEM`). It splices into any analysis
chain and, using the channel wired into its rainbow port, can **add / remove / replace**
the analysis channels flowing downstream, or **converge a channel into another node as a
view-only overlay**. Paired with a generic, node-agnostic **view-only edge** — a dotted
wire toggled by **clicking the wire** — whose channel is fed to the viewers only, never to
analysis, so it can never confuse an analysis pipeline or its downstream effects.

Flagship: `Granule Volume Mask` splits into (A) `DVC` (per-granule, correlating mCherry via
its rainbow wire, edge scope = Objects) and (B) `Granule Boundary → Prism(green) → DVC` on a
**view-only** edge, so both branches **reconverge into DVC**. DVC keeps correlating mCherry
while the green channel is drawn in the DVC 3-D object overlay, **auto-clipped to the shell
between the volume-mask boundary and the boundary-extraction boundary**
(`record._granule_bands_by_m` combined labels). DVC never runs on green → no correctness
risk. Design: `CodeLog/ClaudesPlan/V1.77_prism_channel_node.md`.

### Added

- **`special:prism` node** (`SPECIAL_PRISM_OP_KEY`, `registry_adapter.py`) — orange
  `NodeCategory.SPECIAL`, new `ShapeKind.GEM` silhouette (an upright faceted crystal: flat
  table/culet carrying the structural in/out ports, girdle mid-sides carrying the rainbow
  channel ports). Ports copy the Registration recipe (`input_types=[IMAGE]`,
  `output_types=[ANY]`) so it gains rainbow `ch_in`/`ch_out`. Params: `channel_op`
  (`add`/`remove`/`replace`, default `add`) and `overlay_render` (`Shell (iso)` /
  `Cloud (volume)` / `Cloud (MIP)`, default `Shell (iso)`). Auto-appears in the Add dialog /
  right-click menu and round-trips through `save_pipeline`/`load_pipeline` with **no `io.py`
  change**.
- **View-only (dotted) edges** — `edge.params["view_only"]` (`model.edge_view_only` /
  `set_edge_view_only`), toggled by **clicking the wire** (`edge_item.mousePressEvent` →
  `node_scene.toggle_edge_view_only`), rendered dotted (`edge_item._apply_pen`). Generic to
  every analysis node. Round-trips with no schema bump (like the V1.68 scope lever).
- **`ShapeKind.GEM`** (`model.py`) rendered by `node_item._paint_gem` — dark base + eight
  accent facets (brighter upper-right, darker lower-left) + diagonal gloss + facet seams;
  `_body_path`/`_is_shape`/`_height`/title-rect branches added.
- **`display_channel_sets(sl, all_names)`** (`executor.py`) — per-node view-only overlay
  channel propagation (a Prism source contributes its rainbow-injected channels; any other
  source its `channel_sets`), disjoint from `channel_sets`.
- **`pages/pipelines_page` Prism pipeline** — `_run_prism` (pass-through, publishes
  `record._prism_overlay_by_m`), `_resolve_prism_overlay` (overlay channel = injected;
  shell = union of granule bands when a Granule Boundary node is upstream),
  `_prism_boundary_upstream`, `_prism_shell_by_m`, `_prism_render_to_context_mode`; Run +
  preview dispatch branches. The DVC panel's `context_provider` is wrapped to clip the
  overlay channel to the shell and `DVCPanel.set_context_default` auto-selects it (respecting
  a user override). `record._prism_overlay_by_m` is reset at each Run start.

### Changed

- **`executor.channel_sets`** now (a) **skips view-only incoming edges** (they never enter
  analysis nor propagate downstream as analysis channels) and (b) applies a Prism's
  `channel_op` directive to its rainbow-injected vs structural channels. Every non-Prism
  node and every legacy graph is byte-for-byte the plain union as before.
- **`GraphSlice.structural_edge_into_port`** prefers the **analysis** wire over a coexisting
  view-only wire, so recipe/run linearization ignores an overlay wire on the same port.
- **`node_scene._try_connect` reconvergence** — normal nodes keep the legacy
  redrag-to-replace (one wire per input port); only a **DVC** input port accepts a second
  convergent wire, coexisting as a **view-only overlay**. The object-producing source (the
  granule/mask that scopes DVC per object) is kept as the analysis wire **regardless of the
  order the two branches were drawn** (a later object-producer demotes an existing
  non-object feed to view-only), so `Granule Mask → DVC` and `Prism → DVC` reconverge with
  no order-dependent mis-correlation.
- **`node_scene._apply_edge_scope_lever`** skips view-only edges — a view-only (overlay)
  wire never shows the Frame/Objects scope pill (the toggle would be meaningless), which
  also keeps the click-the-wire lever free to flip it back to an analysis wire.
- **`DVCPanel.set_context_default(channel, mode)`** (`dvc_panel.py`) — default the
  surrounding-channel overlay to a specific channel + render mode the first time it is
  offered (respects a later manual choice, including the first channel — the guard uses an
  explicit `is not None` check, not `x or -1`, so channel index 0 isn't misread).

## [Unreleased] - 2026-07-14 (V1.76 — DVC export / reload as a portable checkpoint node)

The **DVC viewer** gains an **Export** button that writes *every* DVC output — the
quantitative backend data included (`DVCResult` grid coords, displacement + strain
fields, q-factor, diagnostics) plus the per-frame image backdrops, 3-D object masks
and per-granule object bundles, for **all multipoints** — to a single portable
`.nd2dvc` file. Importing that file into **any** ND2Studios session (even a fresh one
with no ND2 loaded) drops a **DVC Checkpoint** input node on the board and re-renders
the whole field series in the same viewer **with no Run** — a reloadable checkpoint of
the DVC stage that also records a read-only **provenance** of how it was produced. The
3-D object view also gets a **Home** button. Design:
`CodeLog/ClaudesPlan/V1.76_dvc_export_reload.md`.

### Added

- **`nd2studios/backend/dvc_export.py`** (new, Qt-free) — the portable bundle format.
  `save_dvc_bundle(path, *, series_by_m, incr_by_m, bg_by_mt, whole_masks_by_m,
  obj_full_by_m, display_mask_by_m, meta, provenance)` writes an NPZ (zip) via
  `numpy.savez_compressed` through an open file handle (so the `.nd2dvc` extension is
  kept): one UTF-8 JSON `__manifest__` byte-array (structure + metadata + provenance,
  **no pickle**) plus the numpy arrays, **deduplicated by object identity**
  (`_ArrayRegistry` — a mask broadcast across frames is stored once). `load_dvc_bundle(
  path) -> dict` reconstructs the page's DVC stores (`series_by_m`, `incr_by_m`,
  `bg_by_mt`, `whole_masks_by_m`, `display_mask_by_m`, `obj_full_by_m`) + `meta` +
  `provenance`. `DVC_BUNDLE_EXTENSION = ".nd2dvc"`; `json_safe()` coerces a
  provenance/meta block to JSON. `DVCResult` (de)serialize via `_result_to_json` /
  `_result_from_json`.
- **`nd2studios/workers/dvc_export_worker.py`** (new) — `DVCExportWorker(path, payload)`
  and `DVCImportWorker(path)` (`BaseWorker` subclasses) run the `savez` / `np.load`
  off the GUI thread; `finished` carries the written path / the reconstructed bundle.
- **`special:dvc_checkpoint` node** (`SPECIAL_DVC_CHECKPOINT_OP_KEY`, `registry_adapter.py`)
  — a portable **DVC Checkpoint** input node, `NodeCategory.CHECKPOINT` (white,
  `ShapeKind.HEXAGON`), **no input/output ports** (terminal source, like the DVC node).
  Created only by the Import flow (excluded from the Add dialog); `dvc_checkpoint_spec()`
  helper; `params["bundle_path"]` + `params["provenance"]` round-trip through
  `save_pipeline`/`load_pipeline` with no `io.py` change.
- **`widgets/dvc_panel.DVCPanel`** — an **Export** button (`export_requested` signal;
  `fa5s.file-export`, top-right of the control row).
- **`widgets/node_board/node_scene.NodeScene.set_node_tooltip(node_id, text)`** — sets a
  per-node hover tooltip (used to surface the DVC Checkpoint's provenance).
- **`pages/pipelines_page`** — an **Import DVC results** toolbar button
  (`fa5s.file-import`, next to Load) and the reload pipeline:
  `_export_dvc_bundle` / `_gather_dvc_bundle_payload` / `_dvc_bundle_meta_for(m)` /
  `_dvc_provenance` / `_dvc_provenance_upstream`; `_import_dvc_bundle` /
  `_apply_imported_dvc_bundle` / `_make_dvc_checkpoint_node` /
  `_set_dvc_checkpoint_tooltip`; `_populate_dvc_panel_from_bundle(m)`;
  `_autoload_dvc_checkpoints()` (re-hydrates on pipeline load); `_run_dvc_checkpoint`
  (Run dispatch). New state: `_dvc_from_bundle`, `_dvc_bundle_meta`,
  `_dvc_bundle_whole_masks_by_m`, `_dvc_bundle_path`.

### Changed

- **`widgets/dvc_panel.DVCPanel._on_zoom_home`** — in the "3D Object" view the **Home**
  button now resets the embedded PyVista camera (`_view3d.reset_camera()`); other views
  keep the 2-D zoom-rectangle reset.
- **`widgets/viewer3d/pyvista_viewer`** — the 3-D control-bar reset button is now a
  **Home** affordance (`fa5s.home`, "Home — reset the 3-D view (fit + isometric)"),
  replacing the `fa5s.expand` "Reset camera" icon (same `reset_camera()` action).
- **`pages/pipelines_page._populate_dvc_panel`** — branches to
  `_populate_dvc_panel_from_bundle` when `_dvc_from_bundle` (a reloaded bundle has no
  live record to derive frame counts / pixel + voxel sizes / channel names from); a real
  DVC run clears the flag. The Add-node palette filters out `special:dvc_checkpoint`.

### Notes / limits

- On a pure reload (no ND2 loaded) the surrounding-channel **context** overlay is
  unavailable (it needs the raw volume); every other viewer capability — scalar fields,
  3-D object surface + MDM, 2-D unwrap, colour/global scale, quiver/histogram, figure
  export — works from the saved `DVCResult`s. Verified headless (round-trip +
  node save/load + a DVCPanel reload smoke across all views); **live-GUI QA outstanding**.

## [V1.75] - 2026-07-14 (V1.75 — Exclude node: ignore a region during analysis)

A new **Exclude** node (`special:exclude`) for the Pipelines node board. Wire an
object / mask / region producer into it — the flagship being **3D Mask Drawing**
(`special:mask3d`), also **Granule Volume Mask** (`special:granule_mask`) — and **every**
downstream analysis node ignores the voxels inside that region, across all Z, all
multipoints and all timepoints. It is the semantic **inverse** of the V1.68 Frame/Object
scope lever (which crops analysis *to* an object). Mechanism = the **Crop node** pattern
(V1.71): publish a region on the `record`, consume it at the shared image-read
chokepoints, so it is honoured with **no per-analysis-node changes**. The image itself is
untouched — Save Data / Export still write the full frame; only what analysis "sees" is
masked. Design: `CodeLog/ClaudesPlan/V1.75_exclude_node.md`.

### Added

- **`special:exclude` node** (`SPECIAL_EXCLUDE_OP_KEY`, `registry_adapter.py`) — a
  pass-through (`ANY → ANY`, `ShapeKind.HEXAGON`, `NodeCategory.SPECIAL`, rendered **red**
  in `node_scene._color_for_node` alongside Dismiss as a "removal" node). One param,
  `dilate_px` (int, 0–200, default 0): grow the ignored footprint by a safety margin.
  Auto-appears in the Add dialog / right-click menu and round-trips through
  `save_pipeline`/`load_pipeline` with no `io.py` change.
- **`pages/pipelines_page` exclusion pipeline** (all new, mirroring the Crop node):
  - `_resolve_run_exclusions()` — a **pre-Run pass** (called from `_on_run` before the
    walk, for both the full-run and checkpoint-resume paths) that builds
    `record._exclude_by_m = {m: (Z,H,W) bool}` (True = ignore) from every Exclude node's
    wired region. 3D-mask sources are **rasterized from the node's drawn shapes** up front
    (via the extracted `_build_mask3d_by_m`), so exclusion is **order-independent** — the
    Exclude branch is independent of the analysis branch, so the walk cannot be relied on
    to reach it first.
  - `_exclusion_source_masks` / `_union_frames_bool` / `_accumulate_granule_exclusion` /
    `_accumulate_mask` / `_merge_exclusion` — resolve + union the wired region(s) per
    multipoint (union over T; optional `binary_dilation`).
  - `_apply_exclusion_channels(record, out, m)` — zeroes the excluded voxels in each
    `(T,H,W)` analysis channel; the `(Z,H,W)` exclusion is projected over Z to an `(H,W)`
    footprint and sliced to the active `_crop_rect()` so it stays crop-aligned.
  - `_run_exclude(node)` — the in-walk handler (pass-through) that re-resolves + merges
    (catches a granule/analysis source that finished earlier in the same Run).

### Changed

- **`_build_mask3d_by_m(node, record, vol)`** extracted from `_run_mask3d_node`
  (behaviour identical) so the Exclude pre-pass can rasterize a drawn mask without running
  the mask node. `_run_mask3d_node` now calls it and derives the status dims from the built
  volumes.
- **Exclusion applied at the shared analysis reads:** `_processed_channels_for_m` (every
  `AnalysisPipeline` node — segmentation, spots, histogram, nuclei, StarDist, tear, mask
  analysis, measurements), `_materialize_channels_for_m` (Spatial Maps / validation),
  `_DVCJob._read_volume` (DVC — `_DVCJob` gains `exclude_by_m=`; zeroes the ignored voxels
  after the rect crop, aligned per-object), and **`_GranuleJob._read_registered_cropped`**
  (Bead Detection, `special:bead_detect` — reads the raw volume through the granule
  worker's own path, so it needs the same zeroing; the payload gains `exclude`). The rest
  of the granule chain (cluster → tessellate → mask → boundary) is covered transitively —
  no bead is detected in the excluded region, so nothing downstream includes it. Tracking
  is likewise transitive (no object detected in a zeroed region → none tracked).

### Bug Fixes

- **Exclude did not affect Bead Detection (`special:bead_detect`).** A
  `3D Mask Drawing → Exclude → Bead Detection` chain still detected beads inside the
  masked-out region (and clustered them), because the bead-detect worker
  (`_GranuleJob._read_registered_cropped`) reads its volume through its own path — not the
  analysis / DVC seams exclusion was wired into. It now zeroes the excluded voxels (aligned
  to the same registration + crop) so beads are never found there. Regression tests:
  `tests/exclude/test_bead_detect_exclusion.py`. Note: re-run the pipeline after wiring
  Exclude (adding it upstream invalidates the Bead Detection checkpoint, so detection
  recomputes).

### Notes / caveats

- **Global by design.** Any Exclude node with a resolved region affects the *whole* Run's
  analysis (matching "prevents *any* analysis node"). Deliberately **not** applied to Save
  Data / Export (they write the real image).
- **DVC** ignores the region by loss of texture (the field is discarded where voxels are
  zeroed); `backend/dvc/` is neither imported nor modified.
- **First-cut sources:** 3D Mask Drawing (order-independent) + Granule Volume Mask (read
  from `record._granule_masks_by_m` when present). Analysis-label / tracked-object sources,
  and a viewer overlay of the excluded footprint, are documented follow-ons.

### Bug Fixes

- **3D Mask Drawing — "Propagate across Z" now fills the whole stack from every drawn
  plane.** `backend/analysis/mask3d.build_mask_volume` with `PROPAGATE_INTERPOLATE` (the
  node's default "Interpolate between planes") only filled the planes *between* the first
  and last drawn slice, so two failures surfaced: (1) drawing on a **single** Z plane and
  hitting **Propagate** filled nothing (the interpolate loop `zip(drawn[:-1], drawn[1:])`
  is empty for one plane); (2) after propagating, drawing a **second** plane and
  re-propagating **shrank** the fill to the gap between the two drawn planes, emptying the
  ends. Now: a single drawn plane copy-extrudes through every Z (`len(drawn) < 2 → copy`),
  and multi-plane interpolate **extrapolates the ends** (copies the nearest drawn plane
  beyond the first / last slice) so all drawn planes propagate *together, throughout* the
  stack. Re-propagating after an edit recomputes from the current hand-drawn planes (the
  editor keeps them as the untagged propagation source and strips prior auto-fill), so an
  edit spreads through the whole volume. Fixes the editor's "Propagate now" preview and the
  Run-time rasterization identically. Regression tests: `tests/mask3d/test_build_mask_volume.py`.
- **3D Mask Drawing — "Propagate now" only propagated one of several shapes drawn on a
  plane.** `mask3d.volume_to_shapes` (which materializes the propagated volume back into
  editable polygons) defaulted to `max_regions=1`, tracing only the **largest** connected
  component per plane — so drawing several disjoint freeforms and propagating carried only
  one of them to the other planes (the drawn plane kept them all, hiding the loss). Now
  `volume_to_shapes` / `_plane_to_polygons` treat `max_regions <= 0` as **all** components
  (the new default), and `_propagate_now` requests all, so every shape drawn (and not
  cleared) propagates in one go. `threshold_seed_shapes` still caps seeds via its explicit
  `max_regions`. Regression tests cover a three-disjoint-shape plane.

## [Unreleased] - 2026-07-14 (V1.74 — Per-granule DVC surface selector + "All granules" composite)

Wiring a **Granule Volume Mask → DVC** edge on the **Objects** scope already ran one
surface-DVC solve *per granule* (V1.68 per-object scope + V1.70 granule masks), but the
DVC panel only ever rendered the **largest** granule — every other granule's field was
computed and discarded at the display layer. V1.74 closes that gap: the DVC tab gains a
per-granule **Object** selector plus an **"All granules"** entry that composites every
granule surface into one 3-D scene. The *run* path is unchanged (no change to scope
enumeration or the per-object `_DVCJob` loop). Design: `CodeLog/ClaudesPlan/V1.74_granule_dvc_object_selector.md`.

### Added

- **`backend/viz3d/overlays.merge_surface_fields(fields, offsets_um=None)`** — pure-numpy
  concatenation of per-object `SurfaceField`s into one composite mesh: vertices /
  `vertex_normals` / `u_par_vec` vstacked, `faces` re-indexed with a running vertex
  offset, `scalars` unioned (a field lacking a key contributes `NaN`, ignored by the
  viewer's clim), `mdm=None` / `interior=None`. `offsets_um` (world `(x,y,z)` µm per
  field, index-aligned before empties are dropped) places each granule's crop-local
  surface at its true relative position. Renders through the **existing**
  `_add_surface_field_overlay` (no viewer changes).
- **`widgets/dvc_panel.DVCPanel` per-granule selector (`cmb_object`)** — shown whenever
  per-object DVC data is present; "All granules (N)" + one entry per granule (largest
  first, `Granule {id} · {k}k vox`), defaulting to the largest (pre-V1.74 behavior).
  Selecting a granule swaps the active field series / backdrops / masks; "All granules"
  composites every granule surface in the **3-D Object** view (2-D Unwrap prompts for a
  single granule; the MDM readout shows a granule-count summary since a composite ⟨F⟩ is
  ill-defined). New `_MultiSurfaceFieldWorker` builds the composite off-thread
  (`with_metrics=False`) and emits on the same `done` signal as `_SurfaceFieldWorker`.
- **`set_data(objects=...)`** on `DVCPanel` — `{object_id: {"series","bg","increment",
  "mask","n_voxels","origin"}}` for the current M; `None` ⇒ no selector (legacy
  whole-frame / single-mask path unchanged).

### Changed

- **`pages/pipelines_page._store_dvc_objects`** now keeps a rich per-object store
  `self._dvc_obj_full_by_m = {m: {oid: {"series","bg","increment","mask","n_voxels",
  "origin"}}}` (per-object backgrounds / increments / masks were previously discarded);
  the existing `_dvc_obj_by_m` (status) and `_dvc_display_mask_by_m` (largest) and the
  largest-bundle return value are unchanged. `_populate_dvc_panel` passes
  `objects=self._dvc_obj_full_by_m[m]` to the panel; `_finish_dvc` resets the new store.
- **`_DVCJob.run`** (per-object path) now also returns `"object_origins"`
  `{oid: (z0,y0,x0)}` (full-frame raw-voxel crop origin) so the composite can offset
  each surface; whole-frame and single-object display paths are byte-identical.

### Notes / caveats

- The composite is one merged `SurfaceField` under a **shared** colour scale (good for
  cross-granule comparison); per-granule **MDM** and **2-D unwrap** require selecting a
  single granule.
- Verified **headless**: `merge_surface_fields` (face offset, per-field translation,
  NaN-filled scalar union, empty/single handling, real two-sphere composite),
  `_store_dvc_objects` full-store construction, and offscreen `DVCPanel` selector
  plumbing (`set_data(objects=)` default + `_on_object_changed` swap). All
  `tests/granule` + `tests/viz3d` + `tests/scope` pass (132). Live-GUI QA (wire granule
  mask → DVC(Objects), Run, switch granules, "All granules" composite) remains manual.

## [Unreleased] - 2026-07-14 (V1.73 — Dedicated viewers for the granule nodes)

Each of the five V1.70 granule nodes now has its own viewer, implemented as gated
overlay **modes** on the existing Pipelines-tab viewer (not new panels) — 2-D marks
drawn through the `set_frame_post_process` → `_composite_pipeline_overlay` hook, and
3-D through the existing in-tab `_btn_view3d` toggle fed by one new backend-pure
scene object. Design + recon: `CodeLog/ClaudesPlan/V1.73_granule_viewers.md`.

### Added

- **Five overlay tabs** on `_overlay_tabbar` (`pages/pipelines_page.py`), each gated
  on its node's store in `_update_overlay_tabs_available` and **auto-selected on Run**
  from the shared `_finish_granule` handler:
  - **Beads** — a crosshair (`cv2.MARKER_CROSS`) at each centroid on its Z-plane.
  - **Clusters** — centroid dots colored per granule via the new `granule_color(gid)`.
  - **Tessellation** — inter-centroid lines (Delaunay edges / Voronoi ridges) + dots.
  - **Granule Mask** — per-plane per-granule fill, categorical color per granule id.
  - **Boundary** — SOLID new-boundary contour (mask ∪ band) + DOTTED previous
    (node-4 mask) contour per granule.
  Painters `_paint_granule_{beads,clusters,tess,mask,boundary}` honor the per-Z
  filter (only when `z_view_mode == "none"`), invert the `_crop_rect()` offset the
  detect worker added, and never mutate the cached base. Helpers: `_granule_entry`,
  `_granule_display_points`, `_granule_point_labels`, `_granule_plane_labels`,
  `_blend_label_plane`, `_draw_dashed_contour`.
- **`backend/analysis/granule_types.granule_color(gid)` + `GRANULE_PALETTE`,
  `NOISE_COLOR`** — a deterministic, **set-independent** categorical color per granule
  id (unlike `track_overlays.generate_track_colormap`, which permutes by the id set),
  so the mask node's colors match the cluster node's. Palette copied verbatim from
  `analysis_page._LABEL_PALETTE` (a test asserts they stay equal).
- **`backend/viz3d/overlays.GranuleScene`** (dataclass) + builders
  `granule_points_scene` / `granule_mask_scene` / `granule_boundary_scene` and
  `granule_centroid_edges(points_zyx, mode)` (Delaunay/Voronoi edges; planar- and
  small-N-safe). Pure numpy: all `(z,y,x) voxel → (x,y,z) µm` reorders live here.
- **`PyVista3DViewer._add_granule_scene_overlay`** (+ dispatch in `_add_overlay_actor`,
  fed by `_feed_view3d_granule_overlay`): crosshair glyphs / per-granule colored point
  clouds, tessellation line network, one colored isosurface per label (assembled
  mask), and a solid new-boundary wireframe + dotted previous-boundary wireframe (VTK
  `SetLineStipplePattern`, with an opacity fallback).

### Notes / caveats

- **Cluster↔mask exact color identity** holds per id (both call `granule_color`), but
  the nodes use different id spaces (cluster: GMM/tessellation-final; mask: dense
  1-based), so a bead and its granule's mask can carry different ids → different
  palette entries. The cluster painter prefers the tessellation's final `point_labels`
  when available; exact cross-node identity across the dense re-key is a documented
  follow-up, not a guarantee.
- Numbered V1.73 because V1.71 (Save Data & Crop) and V1.72 (Split-by-axis export)
  were already taken; the plan doc keeps the recon's working name.
- Fixed a latent bug in `_granule_entry` (`array or ...` — ambiguous truth value)
  surfaced by the new painter tests.
- Verified headless: palette-drift guard, five 2-D painter pixel tests, and off-screen
  3-D render of all five scene types (`tests`-style smoke). Live-GUI QA (tab appears /
  auto-selects on Run, Z-scrub tracking, 2-D↔3-D color parity) remains manual.

## [Unreleased] - 2026-07-14 (V1.72 — Split-by-axis export from the Import tab)

A new main **"Export by axis…"** button on the Import tab (Page 1) writes the
current (cropped) dataset out as **multiple files split along the M / T / Z / C
axes**. Plan: `CodeLog/ClaudesPlan/V1.72_split_axis_export.md`.

### Added

- **`backend/exporters/split_exporter.py`** (backend, PySide6-free):
  - `SplitExportSpec` dataclass — `output_dir`, `basename`, per-axis flags
    `split_m` / `split_t` / `split_z` / `split_c` (True = one file per index on
    that axis; False = bundle every index inside each file), format flags
    `write_tiff` / `write_png` / `write_movie`, `z_mode`, `bit_depth`,
    `lut_mode` (`"auto"` percentile / `"manual"` viewer LUTs / `"full"` full
    dtype range).
  - `_effective_contrast(lut_mode, enabled_names, viewer_lut, dtype_max)` →
    `(composite_lut, tiff_bounds)` resolves the LUT mode into the
    `{name:(lo,hi,gamma)}` dict fed to the PNG/movie compositor and the
    `{name:(lo,hi)}` bounds fed to the TIFF rescale; `_dtype_max` reads the
    volume's integer full-scale value. `auto` → percentile (None); `manual` →
    the viewer window; `full` → `(0, dtype_max)`.
  - `export_split(volume, spec, colors, enabled, lut_settings, image_adjustments,
    movie_options, pixel_size_um, frame_timestamps_s, crop_rect, progress_cb,
    status_cb) -> List[str]` — iterates the cartesian product of the *split*
    axes' index groups (one output file per combination); each file spans the
    full range of the *keep* axes. Reuses `export_tiff_hyperstack`
    (ImageJ TZCYX per group), `_composite_frame` + `_draw_overlays` (one PNG per
    `(m,t,z)` frame), and `export_movie` (one clip per group, timeline =
    flattened `(m,t)`, M-major; kept Z max-projected for the 2-D frame). Kept M
    with >1 positions is flattened into the front of the T axis (ImageJ has no M
    dim). XY crop applied at read time; enabled channels + per-channel color/LUT
    honored so output matches the viewer. Split flags are normalized against the
    real dimensions (an axis with one index — or Z under a projection — can't be
    split) so no spurious `_Z1` / `_M1` suffixes appear.
  - `plan_split_export(n_m, n_t, n_z_eff, n_c_enabled, spec_like) -> dict` — pure
    file-count / page-shape / description helper driving the dialog's live summary.
  - `_axis_groups(n, split)` — `[[0],[1],…]` (split) vs `[[0,1,…]]` (keep).
- **`widgets/split_export_dialog.py`** — `SplitExportDialog`: a per-axis
  **Split / Keep** radio matrix (M/T/Z/C, with Z-split disabled under a projection
  and C-split disabled for single-channel data), independent **TIFF / PNG / Movie**
  checkboxes (TIFF bit-depth combo; movie format + fps), a **Contrast / LUT**
  radio group (Auto-scale each channel / Keep manual viewer LUTs / Full range),
  output folder + base name, a live count summary, and OK gated on a chosen
  folder + ≥1 format. Exposes `spec()` and `movie_options()`.
- **`widgets/file_panel.py`** — new primary **"Export by axis…"** button
  (`btn_split_export`, `fa5s.layer-group`) in the panel's Export group;
  `_open_split_export_dialog` gathers the volume + XY crop + enabled channels +
  color/LUT, opens the dialog, and runs an `ExportRequest(mode="split", …)`.

### Changed

- **`workers/export_worker.py`** — `ExportRequest` gains a `split_spec` field;
  `ExportWorker` dispatches a new `"split"` mode to `_export_split` (returns
  `"<n> file(s) → <dir>"`).
- **`backend/exporters/tiff_exporter.py`** — `export_tiff_hyperstack` gains an
  optional `lut_bounds: {name: (lo, hi)}` argument (before `progress_cb`). When a
  rescaling `bit_depth` (uint8/uint16) is chosen it uses those explicit per-channel
  bounds instead of the auto percentile stretch (channels absent from the dict
  still auto-stretch); `passthrough` ignores it and keeps raw data. Lets the split
  export bake a manual viewer window or full-range mapping into TIFFs. Existing
  callers are unaffected (all pass later args by keyword).
- **`widgets/file_panel.py`** — the previous V1.57 export button is relabeled
  **"Quick export…"** (`compactBtn`; current-M/Z view only); the new split export
  is the primary path.

## [Unreleased] - 2026-07-13 (V1.71 — Save Data & Crop pipeline-graph nodes)

Two new **special** pass-through pipeline-graph nodes. Plan:
`CodeLog/ClaudesPlan/V1.71_SaveData_and_Crop_nodes.md`.

### Added

- **Save Data node** (`special:save_data`, `SPECIAL_SAVE_DATA_OP_KEY`; HEXAGON,
  `PortType.ANY` in/out — a mid-pipeline pass-through). When a Run reaches it,
  `PipelinesPage._run_save_data` writes the dataset **as it is at that point**
  (registration + any upstream crop applied) to a chosen folder — one
  multi-channel ImageJ TZCYX TIFF hyperstack per multipoint (reusing
  `backend/exporters/tiff_exporter.export_tiff_hyperstack`, which accepts both
  `(T,H,W)` and `(T,Z,H,W)`) or a compressed `.npz`. Params:
  - `data` — **`Full Z-stack (raw voxels)`** (default): every Z plane as a
    `(T,Z,H,W)` hyperstack via the new `_zstack_channels_for_m` helper
    (`vol.get_volume` per `(c,m,t)`, registration applied per-Z, XY crop), no
    enhancement recipe (that is a 2-D projection operation);
    `Z-projection (recipe-processed)` = `_processed_channels_for_m` (recipe +
    registration + crop, `(T,H,W)`); `Z-projection (raw)` =
    `_materialize_channels_for_m` (registration + crop, no recipe).
  - `image_format` (`TIFF hyperstack` / `NPZ`), `bit_depth` (`passthrough` /
    `uint16` / `uint8`, TIFF only), `all_multipoints` (**default on** — one file
    per M, `pipeline_data_M01.tif`, …, so the full **M** axis is preserved; off =
    the viewed M only). Loaders without a `get_volume` (e.g. some TIFF sources)
    fall back to the raw Z-projection with a status note.
  - **Full metadata / architecture conservation.** Each file embeds the ND2's
    spatial + temporal calibration (`_save_data_calibration`): XY `pixel_size_um`
    (TIFF resolution), **`z_step_um`** (ImageJ `spacing`, i.e. the axial voxel
    depth), and the **frame interval** (ImageJ `finterval`, from the median of
    `frame_timestamps_s`). A sidecar **`pipeline_data_metadata.json`**
    (`_write_save_data_sidecar`) records the complete architecture
    (T, M, C, Z, H, W), channel names + display, calibration, the applied crop
    rect, and the M→file mapping — so nothing the TIFF/NPZ can't hold is lost.
    The crop is XY-only, so T/M/C/Z and pixel/Z/T calibration are all conserved
    across a crop (only H/W shrink); the sidecar records the crop origin.
- **`backend/exporters/tiff_exporter.export_tiff_hyperstack`** gained optional
  `z_step_um` and `finterval_s` kwargs. `z_step_um` now drives the ImageJ
  `spacing` (axial voxel depth) — previously `spacing` was wrongly set to the XY
  `pixel_size_um`, mis-calibrating the Z axis of any exported Z-stack; the XY
  fallback is kept only when no Z calibration is supplied. `finterval_s` writes
  the temporal calibration. Backward-compatible (both default `None`).
- **Crop node** (`special:crop`, `SPECIAL_CROP_OP_KEY`; HEXAGON, `PortType.ANY`
  in/out — wires upstream of analysis), **manual mode only for now**. Pick the
  rectangle via the param popup's **"Pick crop region…"** button
  (`_edit_crop_region`, reusing the existing `_show_preview_crop_dialog` with
  its x/y/w/h fields, jog pad and live cropped preview); stored in the hidden
  `rect` param `(x, y, w, h)` (raw-image px) by `_set_crop_region`. On a Run,
  `_run_crop` publishes `record._pipeline_crop` so the crop **applies to the rest
  of the pipeline** — every downstream node (analysis, tracking, export, Save
  Data) and the viewer read the cropped image. Params: `mode` (single-choice
  `Manual (draw / enter rectangle)` placeholder) + hidden `rect`.

### Changed

- **`PipelinesPage._crop_rect()` / `_effective_run_crop()`** now compose a third
  crop source — the manual Crop-node rect (`_pipeline_crop_rect()`, reading
  `record._pipeline_crop`) — intersected with the existing preview and
  registration common-region crops via `_intersect_crop_rects`. This is the whole
  "applies to the rest of the pipeline" mechanism: every downstream image read
  already funnels through `_crop_rect()` (`_processed_channels_for_m`,
  `_materialize_channels_for_m`, `_maybe_crop_volume`, `_crop_frame`).
- **`_run_crop`** updates `_run_results_crop = _crop_rect()` when it publishes the
  crop, so downstream analysis masks (computed at the new cropped geometry) match
  the overlay guard in `_overlay_result_for` / `_label_stack_for_m` and aren't
  suppressed as a geometry mismatch. Redraws the base image to show the crop.
- **Run start** (`_start_run`) resets `record._pipeline_crop = None` alongside
  `record._registration_crop`, so a Run only crops downstream if a Crop node runs
  in it again; the rect is never persisted with the pipeline geometry.
- **`pipeline_graph/__init__.py`** re-exports `SPECIAL_SAVE_DATA_OP_KEY` and
  `SPECIAL_CROP_OP_KEY`; both nodes appear on the Add-node dialog's **Special**
  tab automatically (spec-driven palette).

### Bug Fixes

- **Crop node could blank the viewer / crash the LUT histogram.** A persisted
  manual crop that fell out of bounds for the currently-displayed frame sliced a
  zero-size region; the LUT sampler (`LutHistogramWidget.set_data`) then hit
  `ValueError: zero-size array to reduction operation maximum` on navigating to
  the Pipelines page. Fixed in two layers: (1) `_pipeline_crop_rect()` and
  `_run_crop` now clamp the rect to the record's raw frame
  (`_raw_frame_shape`) so it can never slice empty; (2) `set_data` returns early
  when the sampled data is empty, and `_populate_lut_samples_from_volume` skips
  zero-size frames — so any empty-sample source (preview / registration /
  manual crop) degrades gracefully instead of crashing.

## [Unreleased] - 2026-07-13 (V1.70 — Granule Separation from bead point clouds)

Separate a 3-D point cloud of bead centroids into the individual hydrogel
**granules** they belong to, define each granule's boundary, voxelize it into a
per-Z mask, and extract a boundary band — a five-node pipeline-graph chain that
reuses the shipped V1.65–V1.68 mask / object / DVC-surface machinery for the back
half. Design + sub-plans: `CodeLog/ClaudesPlan/V1.70_P0…P7_*.md`.

### Added

- **Five new "special" pipeline-graph nodes** (`nd2studios/pipeline_graph/granule_ops.py`
  op keys; specs + params in `registry_adapter.py`; dispatch + handlers in
  `pages/pipelines_page.py`). Each computes per `(m, t)` on the currently-viewed
  timepoint (the reference frame, reused across T downstream) and publishes to a
  `record._granule_*_by_m` store mirroring `_mask3d_by_m`:
  - **Bead Detection** (`special:bead_detect`, IMAGE→DATA) — wraps
    `backend/serialtrack/detection.ParticleDetector` in the new pure module
    `backend/analysis/bead_detect.py::detect_beads(volume_zhw, voxel_size_um,
    params) -> ((N,3) (z,y,x) voxels, rows)`. The app's **first `centroid_z_px`
    writer**; the detector's native `(x,y,z)` order is flipped to `(z,y,x)` here
    (the only place). Publishes `record._granule_points_by_m`.
  - **Granule Clustering** (`special:granule_cluster`, DATA→DATA) —
    `backend/analysis/granule_cluster.py::cluster_granules(points_zyx,
    voxel_size_um, params) -> (labels, info)`. Full-covariance `GaussianMixture`
    (optional, `find_spec`-gated scikit-learn) with a **BIC sweep** over
    `k ∈ [n·(1−p%), n·(1+p%)]`; KMeans fallback. Publishes `_granule_labels_by_m`.
  - **Granule Tessellation** (`special:granule_tessellate`, DATA→ANY) —
    `backend/analysis/granule_tessellate.py::tessellate_granules(...) ->
    GranuleTessellation`. Two selectable modes on `scipy.spatial` — per-granule
    **alpha-shape** (concave hull) and global **Voronoi** — plus a region-adjacency
    **density-ratio merge** producing the final granule labels + boundaries.
  - **Granule Volume Mask** (`special:granule_mask`, ANY→ANY, **object-producing**) —
    `backend/analysis/granule_mask.py::build_granule_masks(tess, shape_zhw,
    voxel_size_um, params) -> ({gid:(Z,H,W) bool}, (Z,H,W) int32)`. Voxelizes the
    boundaries onto the confocal grid (`floor((z−z0)/dz)`), SDF-Gaussian smoothing
    (`smooth_sigma` µm). Publishes `_granule_masks_by_m[m][t]` =
    `{gid:(Z,H,W) bool, "_labels": (Z,H,W) int32}`.
  - **Granule Boundary Extraction** (`special:granule_boundary`, ANY→ANY) —
    `backend/analysis/granule_boundary.py::extract_boundary_bands(...)`. Outward
    band of N voxels into anything non-self (background + neighbour granules), by
    `binary_dilation` (default) or `distance_transform_edt` (anisotropy-correct).
    Publishes `_granule_bands_by_m`.
- **`backend/analysis/granule_types.py`** — shared P0 contracts: `GranuleBoundary`
  / `GranuleTessellation` dataclasses, `GRANULE_*_ATTR` record-key constants,
  `COMBINED_LABELS_KEY`, and `make_point_rows` / `points_from_rows` (the
  `centroid_z/y/x_px/_um` DATA-row schema).
- **`object_scope.iter_objects_3d_labels(labels_zhw, …)`** — enumerate a
  `(Z,H,W) int32` label volume as **one 3-D `ObjectRegion` per label id** (bbox via
  `find_objects`). The combined granule-label path (a plain `int` array would
  otherwise be misread as `(T,H,W)` by `iter_objects`).

### Changed

- **`registry_adapter.op_produces_objects`** now returns True for
  `special:granule_mask`, so the V1.68 **Frame/Object scope lever** appears on its
  outgoing edge.
- **`pages/pipelines_page.py::_dvc_scoped_object_regions`** now also discovers
  `record._granule_masks_by_m` (via `iter_objects_3d_labels` on the combined label
  volume, so touching granules stay distinct) — wiring **granule mask → DVC** with
  the edge set to *Objects* runs one DVC field + surface + MDM **per granule**,
  reusing the V1.67/V1.68 DVC-on-object render. This is how the wishlist's
  "Volume Viewer" (per-granule 3-D surface with displacement/strain painted on) is
  delivered — no new viewer, the existing PyVista `SurfaceField` path.

### Notes

- scikit-learn (clustering) is an optional, lazily-imported, `find_spec`-gated
  extra — **not** added to `requirements.txt` (cellpose / stardist / pyvista
  precedent). `scipy.spatial` (Voronoi/Delaunay/ConvexHull) was already installed;
  this feature is its first use.
- Backend tests: `tests/granule/` (bead detect, cluster, tessellate, mask,
  boundary). All modules are pure / Qt-free.
- Follow-on (not in this cut): a standalone label-coloured "View granule in 3-D"
  button; per-`(m,t)` granule tracking across the timelapse (first cut defines
  granules on the reference frame and reuses them across T).

## [Unreleased] - 2026-07-13 (V1.69 — 3D Mask Drawing: channel picker + registered image)

The 3D Mask Drawing editor now draws over the **channel wired into the node** and
its **registered (drift-corrected)** image, so the mask is defined on the same
object the DVC field is measured on.

### Changed

- **`Mask3DEditorDialog`** (`widgets/node_board/mask3d_editor_dialog.py`) — new
  **Source** panel with a **channel dropdown**: shows the wired channel's name and,
  when more than one channel is wired into the node, lets the user pick which to
  display/draw over (the drawn object mask is channel-independent, so only the
  background image changes). `get_volume(channel, t)` is now channel-aware and the
  render-block cache is keyed by `(channel, t)`.
- **`pages/pipelines_page.py`** — new `_mask3d_processed_volume(record, c_idx, m, t)`
  applies the pipeline's **registration** (per-Z drift correction, the same transform
  DVC uses in `_DVCJob._read_volume`) to the drawn-over volume; `_edit_mask3d` passes
  the node's wired channels + this registered volume to the editor. Registration is a
  within-frame pixel shift (frame size/coords unchanged), so the mask stays in the
  full/raw frame the rest of the pipeline uses — the registration **crop** is
  deliberately not applied to the mask (it would move the mask into the cropped frame
  and break per-object scoping's `region.bbox ∩ crop`).

### Added

- **"Propagate now" button in the 3D Mask Drawing editor** (Propagate-across-Z
  mode). Propagation was previously only applied implicitly at render/run time with
  no way to *see* or edit it. The button materializes the fill into editable
  polygons: **Copy to all Z** extrudes the drawn footprint through every plane;
  **Interpolate between planes** morphs the outline across the gaps between drawn
  planes (via `mask3d.volume_to_shapes` tracing the built mask volume). Auto-filled
  planes are tagged `src="propagate"` so re-running after drawing more planes strips
  the old fill and recomputes it, while hand-drawn / seeded planes are always kept.
  New `backend/analysis/mask3d.volume_to_shapes()` + `_plane_to_polygons()` (the
  contour→polygon tracer, factored out of `threshold_seed_shapes`).

### Bug Fixes

- **3D Mask editor status messages no longer vanish.** `_seed_threshold` /
  `_propagate_now` set their status *before* `_reload()`, whose `_update_status()`
  immediately overwrote it — so "Seeded N planes" / "Propagated to N planes" / the
  "draw first" guidance never showed. The status is now set after the reload.
- **Threshold seed + edit now stays inside the drawn area.** `Seed this Z` /
  `Seed all Z` previously ran Otsu over the whole plane and traced the largest
  bright component *anywhere*, so on a plane with a brighter neighbour the seed
  jumped to a spot that didn't overlap the manual mask at all.
  `backend/analysis/mask3d.threshold_seed_shapes` gains an `roi` argument (the Otsu
  level is computed from the ROI's pixels and the binary is intersected with it);
  `Mask3DEditorDialog._seed_threshold` builds that ROI from the manually-drawn
  object footprint (union across Z, lightly dilated) so seeding **refines the object
  within the drawn area** instead of snapping to the brightest blob. Falls back to
  whole-plane seeding when nothing is drawn yet (with a hint to draw first).

## [Unreleased] - 2026-07-13 (V1.68 — DVC surface deformation metrics + Frame/Object scope)

Two companion features. Plans:
`CodeLog/ClaudesPlan/V1.68_dvc_surface_mdm.md` and
`CodeLog/ClaudesPlan/V1.68_frame_object_scope_toggle.md`. Literature review:
`Research/mean_deformation_metrics.md`.

**A — DVC surface deformation metrics (Phase 3 of DVC-on-object).** Replaces the
V1.67 voxel object render as the *primary* object view with a **smoothed closed
surface mesh** carrying the DVC displacement, the **Mean Deformation Metrics**
(MDM) suite of Stout et al. 2016 (PNAS 113:2898), a 2-D cartographic unwrap, and a
surrounding-channel context overlay. **DVC computation is untouched** — `DVCResult`
is consumed read-only; `backend/dvc/` is neither imported nor modified.

**B — Frame / Object scope toggle.** A per-edge lever (a clickable pill on the
wire) switches downstream analysis between **whole frame** (default) and
**per object**; on "objects" the file is auto-cropped to each object (conserving
T/M/Z/C) and the downstream sub-pipeline runs once per object.

### Added

- **`backend/viz3d/mdm.py`** (pure numpy) — Mean Deformation Metrics.
  `mean_displacement_gradient(surface, u_vert)` = the discrete divergence-theorem
  surface integral `⟨∇u⟩ = (1/V) Σ_f (ū_f ⊗ n_f) A_f` (Stout Eq. 6);
  `deformation_metrics(grad_u, *, dim)` → `MDMResult` (`⟨F⟩=I+⟨∇u⟩`, `⟨J⟩=det⟨F⟩`,
  polar `⟨R⟩`/`⟨U⟩` via SVD + Kabsch, `⟨λ_i⟩`/`⟨N_i⟩`=eig(⟨U⟩),
  `cos⟨θ⟩=(tr⟨R⟩−1)/2`); `cumulative_rotation(thetas_deg, times_s)` = trapezoidal
  `⟨Θ⟩=∫|θ|dτ`. Recovers `⟨F⟩` to machine precision on any linear field (validated
  on the paper's stretch / rotation / shear canonical cases).
- **`backend/viz3d/surface.py`** (numpy/scipy/skimage; Qt- & VTK-free) —
  `build_object_surface(mask, voxel_size_um, *, smooth_iterations=10, ...)` →
  `ObjectSurface` (marching cubes + pure-numpy **Taubin λ|μ** smoothing, outward
  re-winding, divergence-theorem enclosed volume, area-weighted vertex normals, in
  physical µm); `sample_displacement_on_surface` / `sample_scalars_on_surface`
  (µm-aligned `RegularGridInterpolator`, reusing V1.67's alignment);
  `decompose_surface_displacement` → normal `u⊥` / tangential `u∥`.
- **`backend/viz3d/overlays.py`**: `SurfaceField` dataclass (mesh + per-vertex
  scalars incl. `u_perp`/`u_par` + `MDMResult` + optional interior `MaskedField`);
  `dvc_object_surface(result, mask, mask_voxel_size_um, *, smooth_iterations,
  with_interior, with_metrics)` — the new primary object adapter (orchestrates
  surface + sampling + decomposition + MDM); `UnwrapMap` + `unwrap_surface(surface,
  scalar, *, projection="mollweide"|"equirectangular")` — genus-0 spherical
  parameterization rasterized (seam-safe `griddata`) into a 2-D map + tangential
  `u∥` field for streamlines. Exported from `viz3d/__init__.py`.
- **`PyVista3DViewer._add_surface_field_overlay`** — renders a `SurfaceField` as
  `pv.PolyData` coloured by the per-vertex scalar (`u⊥` / signed strains →
  divergent `coolwarm` centred at 0, per Fig 4C; magnitudes → sequential), with
  the render mode styling the optional interior (`_add_masked_interior`).
  **`set_context_channel(volume, voxel_size_um, *, mode, color, opacity,
  iso_percentile)`** composites a *different* channel as a translucent
  volume/MIP/iso cloud around the object; wired through `_render_overlay_only`.
- **`DVCPanel`**: a **"2D Unwrap"** view + `u⊥`/`u∥` in the object colour picker; a
  **surrounding-channel** picker (channel + mode + opacity); a **Mean Deformation
  Metrics readout** (`⟨J⟩`, `⟨λ₁,λ₂,λ₃⟩`, `⟨θ⟩`, running `⟨Θ⟩`);
  `_ObjectFieldWorker`→**`_SurfaceFieldWorker`** now builds a `SurfaceField`
  off-thread (same generation-guard + coalescing) cached by
  `(incr, t, scalar, smooth_iters)`. `set_data(...)` gains `context_provider`,
  `context_channels`, `frame_times_s`, `surface_smooth_iterations`.
- **`backend/analysis/object_scope.py`** (pure numpy/scipy) — `ObjectRegion` +
  `iter_objects(obj, voxel_size_um, *, min_voxels, pad)`: a `(Z,H,W)` bool mask →
  3-D connected components (Z-scoped boxes); a `(T,H,W)` int label mask → one
  object per label with an XY box unioned over T (full Z preserved).
- **`ObjectCropVolume`** (`pipeline_graph/executor.py`, exported) — a lazy
  per-object crop (XY bbox + optional Z sub-stack) beside `CroppedVolume`,
  conserving T/M/channels; optional `mask_out` hard-clips outside the object mask
  (default off — bbox crop keeps the surrounding matrix DVC needs).
- **`model.py`**: `SCOPE_WHOLE`/`SCOPE_OBJECTS` + `edge_scope(edge)` /
  `set_edge_scope(edge, scope)` on `Edge.params["scope"]` (round-trips today — **no
  schema bump**; absent ⇒ whole-frame ⇒ every legacy graph is unchanged).
- **`registry_adapter.py`**: `node_produces_objects(node)` / `op_produces_objects`
  (mask3d / track / analysis nodes), and a `surface_smooth_iterations` (int,
  default 10) param on the 3D Mask Drawing node — the only surface knob on the node
  (all other DVC-on-object display choices live in the DVC panel).
- **Scope lever** on the node board: `EdgeItem.set_scope_lever(visible, objects)`
  draws a clickable **Frame ↔ Objects** pill at the wire midpoint (Objects tinted
  `ACCENT_ORANGE`), shown only on edges leaving an object-producing node;
  `NodeScene.toggle_edge_scope` / `edge_scope_changed` / `refresh_scope_levers`
  persist the choice to `Edge.params` and mark the graph dirty.
- **Per-object DVC** (`pages/pipelines_page.py`): when the mask→DVC edge scope is
  "objects", `_DVCJob(object_regions=...)` runs ALDVC once per object on its own
  crop (rect ∩ bbox + Z-range) — `method.run(...)` unchanged — and the DVC tab
  shows each object in its own crop frame (field + cropped mask aligned → one
  surface + MDM per object; the largest is shown, all are stored on
  `self._dvc_obj_by_m`).
- **Tests** (repo-root `tests/`): `tests/viz3d/test_mdm.py`,
  `test_surface.py`, `test_object_surface.py`; `tests/scope/` (new)
  `test_object_scope.py`, `test_edge_scope_model.py`, `test_dvc_object_job.py`.

### Changed

- **`pages/pipelines_page.py::_populate_dvc_panel`** passes the context provider
  (`vol.get_volume`), the channel list, `record._frame_timestamps` (for `⟨Θ⟩`), and
  the mask node's `surface_smooth_iterations` to `DVCPanel.set_data`; prefers the
  volume's real `z_step_um`; and, under per-object scope, feeds the display
  object's own cropped mask so the surface is that one object.
- **`DVCPanel`** primary object render is now the surface mesh (V1.67's
  `MaskedField`/`dvc_object_field` is demoted to the optional interior helper via
  `with_interior=True`).

### Bug Fixes

The whole-frame DVC path and every legacy graph are unchanged (gated on the
absent-⇒-whole-frame scope default). Hardening applied from an adversarial
self-review of the new code before ship:
- **DVC panel — no rebuild loop on a failed surface build.** A failed
  `_SurfaceFieldWorker` (`sf=None`) is recorded in `_obj_failed_keys`; the "2D
  Unwrap" / "3D Object" views show a "surface build failed" message instead of
  re-triggering the worker (which previously spun an unbounded background rebuild
  loop for that view).
- **DVC panel — a coalesced current-generation surface build is no longer
  orphaned** when a superseded (stale-generation) worker completes
  (`_on_surface_field_done` launches the pending build before dropping the stale
  result).
- **DVC panel — control labels track their widget's logical visibility**, not
  `QWidget.isVisible()` (which is `False` whenever the panel/ancestor is hidden and
  would leave labels permanently hidden after re-show).
- **Scope lever shown only where it is honored** — edges feeding a **DVC** node
  (its sole per-object consumer this build) — so toggling is never a silent no-op.
- **`sample_displacement_on_surface`** guards a degenerate DVC grid (returns zero
  displacement, matching the sibling scalar sampler) rather than propagating;
  `unwrap_surface` returns empty for a <3-vertex surface instead of a QhullError;
  per-object `_object_crop_mask` crops the mask's Z to the node's Z-range; and
  `ObjectCropVolume.get_frame` `mask_out` unions the object over Z for a
  projection. Removed a dead forward-Mollweide helper.

## [Unreleased] - 2026-07-12 (V1.67 — DVC-on-object 3D render)

Plan: `CodeLog/ClaudesPlan/V1.67_dvc_on_object_render.md`. **Phase 2** of the
DVC-on-object feature (Phase 1 = the 3D Mask Drawing node, V1.65): the
already-computed DVC displacement/strain field is re-rendered **onto the drawn 3-D
object mask** — surface boundary + interior, colored by a chosen scalar — as a new
**"3D Object"** subtab in the DVC viewer, reusing the PyVista 3-D viewer.

### Added

- **`backend/viz3d/overlays.dvc_object_field(result, mask, mask_voxel_size_um, *,
  z_offset_um=0.0, scalar_keys=None, max_box_voxels=4_000_000)`** → **`MaskedField`**
  — interpolates the sparse DVC subset grid onto the object mask's voxels
  (`scipy.interpolate.RegularGridInterpolator`, linear, extrapolating past the grid
  inset). Alignment is in physical **µm** (DVC nodes at `grid_coords ×
  voxel_size_um`, which folds in any XY downsample; mask voxels at `index ×
  mask_voxel_size_um`), so anisotropy and downsampling need no extra bookkeeping.
  Cropped to the object bounding box and strided to `max_box_voxels`. 2-D DIC /
  single-Z grids interpolate in-plane and broadcast over Z. `MaskedField` carries
  the cropped `mask (Z,H,W)`, per-name scalar volumes (NaN outside), `spacing`
  `(dz,dy,dx)` µm and world `origin_um (x,y,z)`. Exported from `viz3d/__init__.py`.
- **`PyVista3DViewer._add_masked_field_overlay`** — renders a `MaskedField`: the
  mask iso-contour as the **boundary surface** colored by the scalar, plus the
  **interior** per render mode — **Iso** (opaque surface), **Slices** (translucent
  shell + orthogonal interior slices), **Volume/MIP** (translucent shell + volume
  render of the field inside the object). New **overlay-only render path**
  (`_render_overlay_only` / `_rerender_current`) lets the object render with no raw
  image volume loaded; `set_overlay` / `set_render_mode` / `set_master_opacity` route
  through it.
- **"3D Object" subtab in `DVCPanel`** — added to `_VIEWS`; the matplotlib canvas is
  wrapped in a `QStackedWidget` with a lazily-built `PyVista3DViewer` swapped in
  place. A `Colour:` selector (`cmb_obj`, `_OBJECT_SCALARS`) picks the DVC scalar;
  `_render_object3d()` builds a `MaskedField` (cached by `(incr, t, scalar)`) and
  feeds the viewer. The interpolation runs **off the GUI thread**
  (`_ObjectFieldWorker(QThread)`, coalesced, with a generation guard so a superseded
  dataset's result is dropped) so view-switch / scrub / playback never freeze. The
  `Colour:` picker is **dimension-gated** (2-D DIC drops z-components), and the
  status reports the scalar *actually* rendered (a missing scalar falls back to
  displacement magnitude). No mask on a frame → a clear "add a 3D Mask Drawing node
  upstream" hint. `set_overlay(None)` clears the scene (no stale object left on
  screen).

### Changed

- **`DVCPanel.set_data(...)`** gains `masks` (`{t:(Z,H,W) bool}`) and
  `mask_voxel_size` (`(dz,dy,dx)` µm, raw) kwargs for the 3-D object view.
- **`pages/pipelines_page.py::_populate_dvc_panel`** passes
  `masks = record._mask3d_by_m[m]` and the raw `mask_voxel_size = (z_step, pixel,
  pixel)` to `DVCPanel.set_data`, so the DVC-on-object view aligns the mask with the
  (possibly downsampled) DVC grid.

### Bug Fixes

- **Registration / DVC / SerialTrack panels no longer crash on publish when their
  viewer tab is hidden.** Publishing a Registration result called
  `RegistrationPanel.set_data(...)` → `_render()` → `MplCanvas.draw()` while the
  panel was a not-yet-shown tab in the stacked viewer, so the figure had a zero
  size. The `imshow(..., aspect="equal")` before/after axes then made matplotlib's
  `apply_aspect` divide by the zero figure size and raise `'box_aspect' and
  'fig_aspect' must be positive`, propagating out of `_finish_register`. Added
  **`MplCanvas.safe_draw()`** (`widgets/common.py`) — the draw twin of the existing
  `safe_tight_layout()`: it skips `draw()` when the figure has a zero dimension and
  swallows any draw exception so a layout pass can never crash the GUI. Routed the
  `RegistrationPanel`, `DVCPanel`, and `SerialTrackPanel` `_render()` paths through
  `safe_draw()` (and switched `DVCPanel`'s two remaining unguarded
  `fig.tight_layout()` calls to `safe_tight_layout()`). The canvas repaints normally
  the moment its tab is shown at a real size.

## [Unreleased] - 2026-07-12 (V1.66 — streaming / memory-mapped loader)

Plan: `CodeLog/ClaudesPlan/V1.66_streaming_loader.md`. Files no longer freeze the
GUI while "materializing into RAM". Following Nikon NIS-Elements, **streaming is
now the overarching viewing setup**: every file **opens instantly and streams
frames on demand** (OS-paged memory-map / lazy reads) with a bounded LRU cache +
neighbour prefetch, instead of eagerly decoding the whole `(M,T,Z,H,W)` volume.
Measured on the provided 56 GB `GELS_TFM.tif` (T9·Z100·C2·4096²): open **~20 ms /
~30 MB RAM**, peak < 1 GB, versus eager's **~5 min / 56 GB** (which can't even
complete when the file exceeds RAM). Eager full-RAM loading remains available as
an opt-in (`FORCED_LOAD_STRATEGY = "eager_full"`) for small resident files.

### Added

- **`backend/streaming_dataset.py`** (pure numpy, Qt-free) — `StreamingDataset`,
  a drop-in `MaterializedDataset`-compatible reader (same `get_frame` /
  `get_volume` / `to_lazy_channel` / `all_channels_as_lazy` / `channel_array` /
  `subset` / `reopen` / `shape` / `n_*` API, so the viewer/pipelines are
  unchanged). Sources: `TiffMemmapSource` (`tifffile.memmap`, axis-aware, with a
  per-page fallback for compressed TIFF) and `LazyVolumeSource` (wraps any lazy
  volume with `get_volume`, e.g. `LazyND2Volume`). Features: byte-bounded
  thread-safe LRU frame cache, single-thread neighbour (`t±1`) prefetch, and
  **blocked Z-projection** (reduces a deep max/mean/min in ~256 MB Z-chunks so
  transient RAM stays bounded — numerically identical to a full projection).
- **`EAGER_MAX_BYTES_DEFAULT` (2 GB)** in `utils/resource_strategy.py` — an
  absolute cap on eager preallocation so large files always stream, independent
  of RAM %.
- **`tests/streaming/test_streaming_dataset.py`** — 11 headless cases.

### Changed

- **`utils/resource_strategy.choose_strategy`** — `STREAM_ALWAYS_DEFAULT = True`:
  streaming is the overarching viewing setup, so `choose_strategy` returns
  `LAZY_CACHED` for **every** file (the eager RAM-fraction / 2 GB cap logic
  remains for when streaming is disabled). The hard-cap OOM guard is exempt when
  streaming (it never materializes). `EAGER_FULL`/`EAGER_REDUCED` are opt-in via
  a forced override; all four load paths now pass that override.
- **`Settings.STREAM_ALWAYS = True`** and **`DISPLAY_PREVIEW_Z_PLANES = 32`** —
  universal streaming plus deep-Z display preview on by default (display is a
  fast bounded-Z projection; recipe/export/DVC stay exact).
- **Stream-for-view / materialize-for-analysis split** — viewing always streams,
  but heavy compute goes resident when it can: `should_stream_analysis` now
  materializes a streamed dataset that fits comfortably in RAM for fast in-RAM
  analysis (only a dataset too big to fit streams + spills labels to disk). New
  `resource_strategy.fits_resident()` (RAM-relative fit check, ignores the small
  view-open byte cap) + `materialize_channels_if_fits()`; `analysis_page` reads
  its input channels into RAM when they fit, else keeps them lazy. Memory
  pressure forces streaming regardless.
- **`workers/load_worker.py`** — `_load_tiff` and `_load_nd2` now honor
  `LAZY_CACHED`: single-file TIFF opens via `StreamingDataset.from_tiff` (memmap),
  ND2 wraps `LazyND2Volume` in `StreamingDataset.from_volume` (previously the
  lazy ND2 path had no working cache — `configure_cache` never existed on
  `LazyND2Volume`).
- **`workers/pre_render_worker.py`** — the bulk 500-frame RGB pre-render skips any
  dataset without an in-RAM `channels` dict (streaming), so the CPU-canvas path
  reads on demand instead of re-eager-loading.

### Notes

- **Multi-file ND2/TIFF stream too** — `LazyVolumeSource` stacks per-plane
  `get_frame` reads for composites that lack `get_volume`, so **all four** load
  paths (`_load_nd2`, `_load_nd2_multi`, `_load_tiff`, `_load_tiff_multi`) honor
  `LAZY_CACHED`. `tifffile` is already a core dependency (no new install).
- **Deep-Z fast display (opt-in)** — `StreamingDataset` can project a bounded set
  of evenly-spaced Z planes for the display path (`get_frame`), ~4× faster on the
  100-Z file, while `to_lazy_channel` / `get_volume` (recipe / export / DVC) stay
  exact. Gated by `Settings.DISPLAY_PREVIEW_Z_PLANES` (default **0 = exact**;
  set e.g. 32 to trade a little display fidelity for speed).

## [Unreleased] - 2026-07-12 (V1.65 — PyVista volumetric 3D viewer)

Plan: `CodeLog/ClaudesPlan/V1.65_pyvista_3d_viewer.md`. A reusable, **optional**
PyVista-backed 3-D viewer offered as a **2D ⇄ 3D toggle** in every Import-tab
panel and the Pipelines preview. It renders the raw `(Z,H,W)` volume from
`record._raw_volume` in four modes (volume / MIP / orthogonal slices /
isosurface) with M/T/Z navigation, time playback, and per-channel color/LUT
mirrored from the 2-D viewer. PyVista/VTK are a `find_spec`-gated extra (like
Cellpose/StarDist); when absent the toggle shows an install hint and the 2-D
view is untouched. Backend data-prep is pure numpy; all VTK work stays on the
GUI thread while volumes are built in a worker.

### Added

- **`backend/viz3d/`** (pure numpy, Qt- and PyVista-free). `prep.py`:
  `build_channel_volumes` / `channel_volume` / `to_uint8` (byte-identical to
  `lut_histogram.apply_lut`) / `Spacing` + `spacing_from_volume` (anisotropic
  `z_step_um` vs `pixel_size_um`) / `auto_contrast` / `color_rgb_float`.
  `overlays.py`: `dvc_field(DVCResult) → DVCField` (world-µm points/vectors +
  scalars: `disp_mag`, `u_x/u_y/u_z`, strain components, `eff_strain`,
  `qfactor`); `ptv_polylines(TrackData) → PtvTracks` (NaN-gapped world polylines,
  color by time/velocity, `max_tracks` cap); `label_volume`.
- **`widgets/viewer3d/`** — `PyVista3DViewer(QWidget)` embedding
  `pyvistaqt.QtInteractor`; public API mirrors `MultiAxisViewer` (`set_volume`,
  `set_channels`, `coords`, `channel_state`, `apply_channel_state`, `refresh`)
  plus `set_render_mode` / `set_z_range` / `set_master_opacity` / `set_overlay`
  / `screenshot`, and `coords_changed` / `channels_changed` signals.
  `Viewer3DDialog` (pop-out window), `Missing3DDeps` (install-hint placeholder),
  `deps.PYVISTA_AVAILABLE` / `PIP_COMMAND`.
- **`workers/volume3d_worker.py`** — `VolumeBuildWorker(BaseWorker)` builds
  LUT-mapped uint8 `(Z,H,W)` blocks off-thread (numpy only), returning
  `VolumeBuildResult(channels, spacing, m, t)` so stale scrubs are dropped.
- **`tests/viz3d/`** — 30 headless pytest cases covering `prep` + `overlays`
  (LUT parity, anisotropic spacing, Z-clamp, DVC axis reorder + strain scalars,
  PTV NaN-gap splitting + velocity coloring). All pass.

### Changed

- **`widgets/file_panel.py`** — the panel viewer is wrapped in a
  `QStackedWidget`; a header "3D" toggle (`_toggle_3d` / `_ensure_viewer3d` /
  `_feed_viewer3d`) swaps in a lazily-built `PyVista3DViewer`, fed from
  `record._raw_volume` with the 2-D viewer's `channel_state()`. `fit_viewer()`
  resets the 3-D camera when active. `self.viewer` still refers to the 2-D
  viewer, so PlayAll / zoom-reset / crop callers are unaffected.
- **`pages/pipelines_page.py`** — the preview viewer is wrapped in a
  `QStackedWidget`; a tab-row "3D" toggle (`_toggle_view3d` / `_ensure_view3d` /
  `_feed_view3d`) swaps in the 3-D viewer, fed from `_active_record()._raw_volume`
  at `viewer.coords()`. The 3-D viewer rides the existing
  `_toggle_popout("viewer")` maximize window for free.
- **`requirements.txt`** — documents the optional `pyvista` / `pyvistaqt` extra
  (gated, deliberately *not* pinned, matching the segmentation-backend policy).

### Fixed (post-testing, on the 56 GB `GELS_TFM.tif`)

- **3-D viewer no longer freezes/crashes on large or deep volumes.** It had built
  and rendered the *full-resolution* stack (`100×4096×4096×2` ≈ 3.4 B voxels — a
  ~6 GB float32 intermediate per channel, and a VTK grid that exhausted the GPU
  mapper and segfaulted, an uncatchable crash). It now builds a **downsampled,
  memory-bounded** render volume via `prep.build_render_volumes`: reads one plane
  at a time, Z-subsamples to ≤ `Settings.VIEW3D_Z_MAX`, XY-strides so the larger
  axis ≤ `VIEW3D_XY_MAX`, and caps `VIEW3D_MAX_VOXELS` per channel (~6–26 M
  voxels vs 1.7 B), keeping correct anisotropic spacing. Verified on the 56 GB
  file: bounded `(24,512,512)` render volume, ~one-plane transient RAM.
- **Plotter init is guarded** — a VTK initialization failure now falls back to
  an in-panel message instead of taking down the app.
- **No more "hole to the desktop."** A native OpenGL window (embedded
  `pyvistaqt.QtInteractor`) cannot composite into ND2Studios' frameless
  `WA_TranslucentBackground` main window — the VTK region showed through to the
  desktop and stole mouse input. The viewer now renders **off-screen**
  (`pyvista.Plotter(off_screen=True)`) and paints the result into a `QLabel`
  raster image (drag to orbit, wheel to zoom, isometric view + axes + bounding
  box), which composites correctly. It now needs only `pyvista` (not
  `pyvistaqt`), and `AA_ShareOpenGLContexts` is set at startup.
- **T-playback no longer crashes.** Playing the time axis spawned a fresh
  `VolumeBuildWorker` every timer tick while each still read gigabytes → QThreads
  piled up and one was destroyed mid-run (`QThread: Destroyed while thread is
  still running`). Builds are now **coalesced** (one worker at a time,
  latest-request-wins), playback **paces itself** to build speed (skips a tick
  while a build runs), `build_render_volumes` honors a `cancel_cb`, and
  `on_close` `wait()`s for the worker. The streaming frame cache is also **capped
  at 2 GB** (`recommended_cache_budget_bytes` had returned ~93 GB on a big-RAM
  host, which would balloon RAM).

### Added

- **Smooth 3-D time playback.** Pressing play now **prebuilds every timepoint's**
  downsampled volume once in the background (`TimeSeriesBuildWorker`, progress
  shown, cancellable, whole series RAM-capped ~2 GB), then plays from RAM with
  the camera **held steady** — no per-frame disk reads or camera snap.
  `_render_scene(reset_view=False)` swaps the cached frames; the prebuilt cache
  is render-mode-independent (switch volume / MIP / slices / iso without
  re-reading). The one-time prep reads the timepoints once (it needs the voxels),
  so smoothness is traded for an upfront read the first time you play a given
  M / Z-range / LUT.

### Notes

- DVC / PTV **"View in 3D"** result buttons are scaffolded in the backend
  (`viz3d.overlays`) and viewer (`set_overlay` supports `DVCField` / `PtvTracks`)
  but the `dvc_panel` / `serialtrack_panel` button wiring is a V1.66 follow-up
  (design doc §8.3–8.4). PTV renders space-time tubes until a `centroid_z_px`
  source exists.
- GUI/VTK paths need on-machine verification (no OpenGL in this build env); the
  numpy backend is unit-tested and an off-screen render smoke test is documented
  in the plan (§11).

## [Unreleased] - 2026-07-12 (V1.65 — 3D Mask Drawing node)

Plan: `CodeLog/ClaudesPlan/V1.65_mask_drawing_node.md`. **Phase 1** of the
DVC-on-object feature: a pipeline-graph node that draws a 3D object mask across the
Z-stack. Phase 2 (re-render an already-computed DVC field onto that mask — surface +
interior — as a new DVC subtab, with interpolation up to the mask resolution) is
deferred until requested and will consume the mask this node publishes.

### Added

- **3D Mask Drawing node** (`SPECIAL_MASK3D_OP_KEY = "special:mask3d"`,
  `pipeline_graph/registry_adapter.py`) — a HEXAGON Special node with an
  `IMAGE` input (rainbow channel wiring, so it draws over the wired channel's raw
  `(Z,H,W)` volume) and an `ANY` output — a pass-through in the image stream (like
  Registration) so it wires *in front of* a DVC node (`input → 3D Mask Drawing →
  DVC`); this both fixes connectivity (an `ANY` output pairs with DVC's `IMAGE`
  input, which a `BINARY` output would not) and orders the run so the mask is
  published before DVC's 3-D Object view reads it. *(V1.65 shipped this node with a
  `BINARY` output, which could not connect to DVC — fixed to `ANY` alongside V1.67.)* Registered via a `_SPECIAL_OPS` row + a `param_specs_for`
  branch; exported from `pipeline_graph/__init__.py`. Params: `mode`
  (`Manual (per-plane)` / `Propagate across Z` / `Threshold seed + edit`),
  `propagate` (`Copy to all Z` / `Interpolate between planes`, shown for Propagate),
  `apply_all_frames`, and a hidden `mask_shapes` slot holding the drawn geometry
  (`{str(m):{str(t):{z_key:[shape]}}}`, `z_key` an int Z index or `"all"` — the
  `manual_mask` shape schema, auto-serialized with the graph). Run builds a mask for
  every multipoint the user drew on (the store is authoritative), so there is no
  per-viewer-M scope toggle.
- **`backend/analysis/mask3d.py`** — pure (no PySide6) mask maths reused by the
  node's Run:
  - `build_mask_volume(shapes_by_z, Z, H, W, propagate)` → `(Z,H,W)` bool volume.
    `propagate="none"` fills only drawn planes; `"copy"` extrudes the union
    footprint through Z; `"interpolate"` morphs between consecutive drawn planes via
    a signed-distance-field blend (`_signed_distance`, `scipy.ndimage.distance_transform_edt`)
    so the surface between drawn slices is smooth and higher-resolution.
  - `rasterize_plane(shapes, H, W)` — union of `"add"` shapes minus `"sub"` (erase)
    shapes for one plane (wraps `manual_mask._rasterize`).
  - `threshold_seed_shapes(volume, threshold=0.0, …)` — per-Z Otsu/explicit
    threshold → largest components → `find_contours` → resampled editable polygons.
  - `mask_volume_bounds(volume)` — tight `(z0,z1,y0,y1,x0,x1)` box for consumers.
- **`widgets/node_board/mask3d_editor_dialog.py`** — `Mask3DEditorDialog`, a modal
  per-Z editor embedding the reusable `ImageCanvas` + a Z (and T) scrubber. Draw
  rect / ellipse / polygon (`ImageCanvas.set_draw_mode` / `shape_drawn`), `Erase`
  toggle (next shapes subtract), `Clear plane` / `Clear all`, threshold-seed
  (`Seed this Z` / `Seed all Z`), and a live mask overlay with a `Show 3D preview`
  toggle (renders the built-volume slice after propagation). All controls DPI-scaled
  via `scaled()`; tool buttons use `icon_button` (`fa5s.*` icons) with
  `setAutoDefault(False)`.

### Changed

- **`pages/pipelines_page.py`** — wired the node end-to-end: import
  `SPECIAL_MASK3D_OP_KEY`; new `self._mask3d_by_m` store; a "Draw 3D mask…" edit
  button (`_open_popup_for`) dispatched (`_on_popup_edit`) to `_edit_mask3d`
  (builds a per-frame `get_volume` closure over `record._raw_volume.get_volume`,
  opens the editor, persists the drawn shapes + mode/propagate back onto the node);
  Run dispatch (`_run_execute_node` → `_run_mask3d_node`, synchronous — rebuilds the
  mask store **fresh** from the drawn shapes each Run into a new per-record dict
  `{m:{t:(Z,H,W) bool}}` honoring `apply_all_frames`, publishes to
  `self._mask3d_by_m` + `record._mask3d_by_m`) and a preview-walk status branch
  (`_preview_execute_walk_node`). The fresh rebuild avoids stale frames/multipoints
  from earlier Runs and the per-record dict avoids multi-input m-index collisions.

## [Unreleased] - 2026-07-12 (V1.64 — dynamic screen-size text scaling)

Plan: `CodeLog/ClaudesPlan/V1.64_dynamic_text_scaling.md`. GUI text now grows
with the size of the monitor, and the controls that hold it grow in the same
ratio so enlarged text never overflows its button.

### Added

- **`screen_scale()`** (`widgets/icon_button.py`) — a clamped screen-size growth
  factor derived from the primary screen's available logical height vs a 1080p
  baseline: `1.0` at/below 1080p (grows by `SCREEN_SCALE_SLOPE = 0.5` of the
  excess height, capped at `SCREEN_SCALE_MAX = 1.5`; e.g. 1440p ≈ 1.17, 4K = 1.5).
- **`scaled_pt(base_pt)`** — scales a point size for the current screen
  (half-point steps).
- **`scale_qss(style, factor=None)`** — scales every `pt` font size and `px`
  dimension in a QSS/stylesheet string by `screen_scale()` (or an explicit
  factor). Returns the string unchanged at factor `1.0`, so baseline (1080p)
  renders are byte-identical to earlier versions. Used by `build_stylesheet()`
  and to wrap the app's inline `setStyleSheet(...)` calls.

### Changed

- **`ui_scale()`** (`widgets/icon_button.py`) now returns `dpi_factor ×
  screen_scale()` (was DPI-only), so every `scaled()` control grows in lockstep
  with the text — fixed-size text widgets (e.g. `QSpinBox`) therefore grow at
  least as fast as their font and never clip.
- **`build_stylesheet()`** (`core/theme.py`) passes the QSS template through
  `scale_qss()` before injecting arrow images, scaling all fonts and control
  dimensions (min-height, padding, border-radius, widths, arrow sizes) together.
- **Inline stylesheets scaled at the call site.** ~120 `setStyleSheet(...)`
  strings carrying an explicit `pt`/`px` font across `pages/`, `widgets/`, and
  `widgets/node_board/` (which override the global stylesheet) are now wrapped in
  `scale_qss(...)`; `QPainter`-drawn text (`QFont` point sizes in `frame_strip`,
  `tile_layout`, `tile_preview`, `node_item`, `whole_frame_review_dialog`, …) is
  scaled via `scaled_pt(...)`.
- **Window chrome grows with the display** (`core/main_window.py`): title-bar and
  bottom-bar heights, the bottom-bar status/memory/version labels, the progress
  bar, and the window minimum/opening size are scaled by `screen_scale()`.

### Bug Fixes

- **Panels no longer clip when text is scaled up.** ~125 `setFixed*` /
  `setMinimum*` / `setMaximum*` geometry calls across `pages/`, `widgets/`, and
  `widgets/results/` used raw pixel literals, so their containers stayed
  baseline-sized while the text grew and clipped (most visibly the **LUT
  histogram tab**: `LutSidebar.EXPANDED_WIDTH`, the histogram canvas max-height,
  spin-box widths, section-header heights, and the collapse-animation width
  endpoints). These pixel values (and pixel size constants) are now wrapped in
  `scaled()` so containers grow in lockstep with the text. No-op at the 1080p
  baseline; `setContentsMargins`/`setSpacing` and `QGraphicsScene`/matplotlib
  figure dimensions were left unchanged.

## [Unreleased] - 2026-07-10 (Pipelines overhaul — V1.62 Phase 2 (R3): one input node per loaded file)

Plan: `CodeLog/ClaudesPlan/V1.62_multi_input_files_and_channels.md`. Phase 2 = R3
(multi-input) + R8 (channels on the node body) + R9 (spacing). This pass is **R3**;
R8/R9 next.

### Added

- **One input node per loaded file (R3)** (`pages/pipelines_page.py`). Every file
  loaded in the Import tab now appears as its **own input node** in the Pipelines
  scene, **named after the file** (basename) and **bound to that file's record** by
  `exp_id` (stored in `node.params["source_exp_id"]`). New helpers `_loaded_files`
  (enumerates `main_window.pages["import"]._panels` records + the active record),
  `_loaded_records_by_id`, `_input_node_for` (walks structural inputs back to the
  INPUT node), and `_sync_input_nodes` (adopts the legacy universal input for the
  first file, creates a node per additional file, lays them out in a column).
  `_ensure_input_node` / `_refresh_input_node` now route to `_sync_input_nodes`,
  called from `_select_stage` / `on_activated` / `load_from_experiment`.
- **Per-file execution via a focused input.** `_active_record()` now resolves
  through a **focused input node** (`_focused_input_id`, set on double-click to the
  input feeding the previewed chain) → its bound record, falling back to the Import
  tab's active record. So previewing/running a file's chain uses **that file's**
  data; Run starts the `GraphRunner` at the focused input. All 52 existing
  `_active_record()` call sites follow the focused file through this single
  indirection, and the **single-file case is unchanged** (the sole input binds to
  the active record).

### Changed

- **Closing a file disables (keeps) its input node** (per the confirmed R3
  decision): `_sync_input_nodes` sets `node.enabled = False` on inputs whose file
  is no longer loaded (`NodeItem` already dims disabled nodes) and keeps them + their
  edges, so the pipeline structure survives; focus moves off a disabled input.

### Notes

- Deferred to R8/R9: channels rendered on the input-node body (currently the V1.48
  channel pills still attach to the primary input; secondary file inputs use the
  legacy all-channels behaviour). Multi-file *simultaneous* execution + per-input
  result viewers are Phase 4 (POV). Verified headless (offscreen mock Import page
  with 2 files: 2 bound/basename-titled inputs, `_active_record` follows focus,
  close-disables, single-file fallback; self-test + `MainWindow` boot green).

## [Unreleased] - 2026-07-10 (DVC viewer: clarify the Z slider is the correlation grid, not raw Z)

Investigated a report that a 100-Z-slice file showed "only 4 Z-stacks" on the DVC
panel's Z slider. **This is expected DVC behaviour, not a correlation bug:** the
displacement field is solved only at subset centers, so its Z extent is the grid
count ``Gz ≈ (Z − subset_size)/subset_spacing + 1``, not the raw slice count. At the
default ``subset=16, spacing=10`` a 100-slice stack yields ``Gz=9``; ``Gz=4``
corresponds to ``subset_spacing ≈ 25`` (near MATLAB's typical ``winstepsize``) or a
node Z-range of ~50 slices. Validated against FranckLab's MATLAB ALDVC
(``funIntegerSearch3.m`` grid = ``start:winstepsize:end`` per axis, inset ~subset/2;
``MeshSetUp3.m``) and end-to-end: ``run_aldvc`` recovers a known 3-D shift
``(1.5, 2.0, −1.0)`` vox as ``(1.499, 1.998, −0.999)`` (ZNCC 1.0, correct z,y,x
order). The ND2 loader reads Z by name from the SDK (``f.sizes["Z"]``), so the slice
count isn't mis-read.

### Changed

- **``widgets/dvc_panel.py``** — the in-frame Z slider is relabelled **"Grid Z:"**
  (was "Z:") with a tooltip on the label, slider, and value readout explaining it
  steps the correlation grid's Z-planes (subset centers), not the raw image
  Z-stack, and that lowering *Subset spacing* gives finer Z sampling. Module
  docstring updated to match. No behavioural/numeric change.

## [Unreleased] - 2026-07-08 (Cropped resume from a Checkpoint — V1.59)

Plan: `CodeLog/ClaudesPlan/V1.59_checkpoint_cropped_resume.md`.

### Bug Fixes

- **Checkpoint after a Registration node → "Run cropped region" went back to
  Registration, neglecting the checkpoint.** On the repro pipeline (Input →
  Registration[crop to common] → Checkpoint → DVC), resuming from the checkpoint
  with a preview crop re-ran the whole upstream instead of resuming. Three causes,
  all fixed:
  1. `_checkpoint_upstream_hash` folded the crop (`_crop_rect()`, incl. the
     registration common crop) into the validity hash, but `_on_run` clears
     `record._registration_crop` at run start *before* the resume check — so the
     hash always mismatched. **Fix:** the hash no longer folds in any crop (the
     registration crop is redundant — already implied by the reg node's params +
     file signature — and the preview crop is a downstream concern).
  2. The checkpoint never froze the **registration state** (transforms + common
     crop live on the record, cleared at run start), so a resume that skips the
     frozen Registration node lost the drift correction. **Fix:** `_run_checkpoint`
     freezes it and `_restore_checkpoint` → `_restore_registration_state`
     re-publishes it.
  3. **DVC ignored registration and crop** — `_DVCJob` read raw full-frame
     volumes. **Fix:** DVC now applies the drift transforms (per Z-slice) then
     crops (register → crop), correlating the registered, cropped region.

### Changed

- **`pages/pipelines_page.py`**
  - `_checkpoint_upstream_hash` drops the crop term entirely.
  - `_run_checkpoint` snapshot adds `reg_by_m` / `reg_interp_order` / `reg_crop` /
    `reg_bundle`, and stores `results_crop = _crop_rect()` (the geometry the frozen
    masks live in, incl. any registration crop).
  - New helpers `_effective_run_crop()`, `_crop_contains(outer, inner)`,
    `_crop_analysis_result(res, rect, origin)` (origin-aware mask slice),
    `_recrop_restored_checkpoint(target, stored)`, `_restore_registration_state(snap)`.
  - `_restore_checkpoint` re-publishes the frozen registration state, then re-scopes
    the frozen masks to the effective Run crop and **re-derives** rows / tracks from
    the sliced masks + cropped channels (`_ensure_run_rows`).
  - `_checkpoint_resume_target` applies the crop-containment guard **only when the
    checkpoint froze masks** (a registration-only checkpoint is always resumable).
  - `_DVCJob(rect, transforms_by_m, interp_order)`: `_read_volume` registers each
    Z-slice then crops; `_run_dvc` passes `_crop_rect()` + `_registration_by_m`.
  - Resume + DVC status messages note the crop / "registered".

## [Unreleased] - 2026-07-08 (Pipelines tab overhaul — V1.61 Phase 1: merge, navigation, submenu, rename)

Master plan: `CodeLog/ClaudesPlan/V1.61_pipelines_tab_overhaul.md` (14-requirement
overhaul, phased V1.61→V1.64). **Phase 1** = R1 (merge the Processing + Analysis
sub-tabs into one scene) + R2 (per-type add submenu) + R4 (input/output rename) +
R5 (click-drag pan). Verified headless (self-test + offscreen page/MainWindow +
v4→v6 load fold + reconnect); **needs interactive validation on a real ND2** (R1 touches
result-producing execution).

### Added

- **Processing + Analysis merged into ONE connected chain (R1)**
  (`pipeline_graph/model.py`, `io.py`, `pages/pipelines_page.py`). The two sub-tabs
  collapse into a single node canvas holding enhancement (cyan) + analysis (pink) +
  results (green) + logic (purple) + special (orange) + checkpoint (white) nodes,
  color-coded by category. The graph is now **one connected flow**: a single
  **universal input** node (for all loaded files) → processing → analysis →
  **output nodes at the ends** of analysis workflows. The intermediate
  Processing-OUTPUT / Analysis-INPUT **bridge nodes are gone** (they were two
  disconnected components joined only via `record.recipe`).
  - **Model:** `PipelineDoc.slice_for(PROCESSING|ANALYSIS)` both resolve to the
    merged (analysis) slice; schema → **6**. `io._migrate_to_connected` folds a
    saved two-slice doc into the merged slice, drops the bridge nodes, and
    **reconnects** the processing tail (or the universal input) directly to the
    analysis heads the old Analysis-INPUT fed — lossless for the common case.
  - **Page:** the sub-tab selector is hidden; `self._stage` **tracks the previewed
    node** (processing node → processed-image preview; anything else → analysis
    overlay); one scene aliased under both stage keys; stage-aware helpers
    `_node_group` / `_stage_output_nodes` / `_stage_input_node` and the new
    `_processing_tail_node` (deepest enhancement before analysis) drive recipe
    derivation — `_graph_recipe` / `_graph_channel_recipes` / `_apply_processing` /
    `_do_processing_preview` read the merged slice and anchor the recipe on the
    processing tail (no output node needed). `_ensure_input_node` creates only the
    one universal input.
  - **Unified Run** (`_on_run`) commits the processing recipe then walks the
    analysis graph starting at the **universal input**; the Run pump
    (`_run_execute_node`) **passes through** PROCESSING-stage nodes (they execute as
    the committed recipe, not as Run steps). Apply (always visible) commits the
    processing recipe. The catalog is `enhancement_specs()` + `merged_action_specs()`
    (output nodes added via "Add output node"); `_on_output_created` dispatches by
    stage.

### Bug Fixes

- **Base viewer blank on the merged tab.** After the merge, the right-pane image
  viewer could come up empty because the processing context relied on the
  debounced pinned-preview job and never set a base volume. `_select_stage` /
  `_refresh_active_view` now **always call `_show_base_image()`** so the (processed)
  base image is shown immediately on the merged tab; overlays / pinned processing
  previews layer on top. (Default context kept at Processing so the overlay-panel
  machinery doesn't run during MainWindow construction.)
- **The image viewer is never hidden.** Previously the `_viewer_stack`
  (`QStackedWidget`) **swapped the `MultiAxisViewer` out** for a specialized result
  panel (Registration / DVC / Spatial Maps / SerialTrack), and a table-only results
  view hid the viewer container — so the base image (and the Preview-Crop tool that
  draws on it) disappeared after running those nodes. Restructured: the
  `MultiAxisViewer` now lives **permanently** in a vertical splitter
  (`_viewer_split`) and is never hidden or swapped; the specialized panels live in a
  separate `_panel_stack` shown **below** the viewer (via `_set_active_panel`) only
  while their overlay tab is active, and hidden otherwise. `_set_results_view_mode`
  no longer hides the viewer in "table" mode. Preview Crop is therefore always
  accessible; arming it also hides the panel area so the crop is unobstructed.
- **`tight_layout` crash when a result panel renders while hidden.** A consequence
  of the above: the Registration / DVC / SerialTrack panels are populated the
  instant a node finishes, before the just-shown `_panel_stack` has been laid out,
  so their matplotlib figure had zero height and `fig.tight_layout()` raised
  `'box_aspect' and 'fig_aspect' must be positive` (seen from `_finish_register` →
  `_populate_registration_panel`). Added `MplCanvas.safe_tight_layout()`
  (`widgets/common.py`) — skips `tight_layout` on a zero-size figure and never lets
  a layout pass raise — and routed the three panels through it. The panel lays out
  correctly on its first real-size redraw.
- **Click-drag panning on the node board (R5)** (`pages/pipelines_page.py:_BoardView`).
  A plain left-drag on empty canvas now pans the view (closed-hand cursor) — the
  natural gesture for navigating a large graph. Holding **Ctrl or Shift** while
  dragging empty canvas keeps the rubber-band marquee select. A press on a node /
  port / wire still moves the node, starts a wire, cuts, or loops (unchanged).
  Panning is suppressed while a scene tool (scissors `_cut_mode` / loop
  `_loop_mode`) is armed, via a new `_BoardView._tool_active()` guard. Implemented
  with manual scrollbar translation in `mousePressEvent`/`mouseMoveEvent`/
  `mouseReleaseEvent` (`_panning`, `_pan_last`) rather than `ScrollHandDrag`, so
  item interaction stays crisp.

### Changed

- **Right-click "Add" menu is now grouped by node type (R2)**
  (`widgets/node_board/node_scene.py:contextMenuEvent`). The single flat "Add
  operation" submenu is replaced by one submenu **per `NodeCategory`**
  (Processing / Analysis / Results / Logic / Special / Channels),
  ordered by new `_CATEGORY_MENU_ORDER` / labeled by `_CATEGORY_MENU_LABEL`,
  grouping `self.action_specs` via `NodeSpec.effective_category()`. The
  **Checkpoint** node keeps its own (white) category color but is listed under
  the **Special** submenu (`_MENU_CATEGORY_ALIAS`). A
  single-category scene (e.g. Processing) shows just its one submenu; the merged
  scene fans out into per-type submenus so its ~15+ specs stay navigable. "Add
  output node" is unchanged.
- **Inline rename now covers INPUT nodes and triggers on double-click (R4)**
  (`widgets/node_board/node_item.py`). `_is_renamable()` now returns True for
  `NodeRole.INPUT` as well as `OUTPUT`. The rename editor opens on
  **double-click** (was single-click-arm on the name area, output-only);
  `mouseDoubleClickEvent` on a renamable node opens `_begin_name_edit()` and
  consumes the event, so double-click on input/output = rename while double-click
  on an ACTION node keeps promoting it to the previewed node. Enter / focus-out
  commits, Esc cancels; clicking empty canvas commits an open editor
  (`_BoardView` drops scene focus before panning). Removed the now-dead
  click-arm state (`_maybe_name_edit`, `_press_scene_pos`).

### Bug Fixes

- **Pipeline schema-version drift** (`pipeline_graph/model.py`). `PipelineDoc.
  schema_version` defaulted to `3` while `io.PIPELINE_VERSION` was `4` (drift from
  the V1.49 io bump), so a freshly-created doc serialized ahead of its default and
  failed the `scripts/_pipeline_graph_selftest.py` save/load round-trip (red on
  `HEAD`). The default now tracks `io.PIPELINE_VERSION` (both `6` for the merge) —
  round-trip green.
- **Stale self-test assertions** (`scripts/_pipeline_graph_selftest.py`) realigned
  to the current registry: the Track Objects node's `ct_min_iou` param and the
  Field Maps node's `templates` library editor (pre-existing WIP drift, unrelated
  to the overhaul). Self-test now fully green (+ v4→v6 fold/reconnect + v6 round-trip).

### Notes — why R1 is an executor merge (and how it was verified)

Processing and Analysis executed via **two different engines** — Processing commits
a linearized *recipe* (`_graph_recipe`→`recipe_for_node`→`record.recipe`); Analysis
runs the `GraphRunner` state machine — joined only through `record.recipe`, never by
a graph edge. The merge keeps both engines and stitches them at the page: one
scene/slice, `self._stage` follows the previewed node, PROCESSING nodes pass
through the Run pump (recipe, not a Run step), unified Run commits the recipe then
walks the analysis graph. **Verified headless:** `_pipeline_graph_selftest.py`
green; offscreen `PipelinesPage` (one aliased scene, combined 35-spec catalog,
recipe derivation across the merged slice, stage-aware input resolution, Run
pass-through); a synthetic **v4 file loads → folds → renders → derives its recipe**;
full offscreen `MainWindow` boot. **Still requires interactive validation on a real
ND2** — the graph-logic is verified, but result-producing execution (segmentation /
tracking / DVC over real pixels) should be exercised in the running app.

## [Unreleased] - 2026-07-08 (Pipelines crop/channel viewer fixes)

### Bug Fixes

- **Analysis viewer reverted to raw after Apply → switch to Analysis (worst under
  a preview crop).** Root cause: the Processing-preview result handler
  (`_on_runner_done`, `_PREVIEW_KEY` branch) was **not stage-gated**.
  `_apply_processing` ends with `_request_preview()`, and the crop preview also
  runs a Processing-preview job; both are debounced + async. When the user
  switched to the Analysis tab before the job finished, its result landed on the
  Analysis tab and **overwrote** the base image — which `_show_base_image` had
  correctly set to a fully-processed `ProcessedFrameVolume` — with the
  Processing-preview `PinnedProcessedVolume`, which serves the recipe only on the
  pinned plane and **raw everywhere else** (so the cropped Analysis view read
  raw). Fix: the `_PREVIEW_KEY` handler drops its result when
  `self._stage is not Stage.PROCESSING`, and `_select_stage` stops the preview
  debounce + cancels the in-flight `_PREVIEW_KEY` job when leaving Processing, so
  a stale Processing preview can never clobber the Analysis base. Diagnosed by
  driving the real page against `Bolus_Top_crop_tiff.tif` (2-ch, 33×734×745);
  the headless read path was already correct — the bug was purely the async
  cross-tab clobber. (Earlier `bypass_pyramid` + per-channel backdrop fixes remain
  for large-file / Spatial-Maps cases.)

## [Unreleased] - 2026-07-08 (Registration: late-frame robustness + region/feature estimation — V1.60)

Plan: `CodeLog/ClaudesPlan/V1.60_registration_roi_and_features.md`.
Research: `Research/image_registration.md`.

Fixes registration degrading at later timepoints of a timelapse, and adds
region-of-interest and feature-based estimation. Root cause of the late-frame bug
(reproduced on a synthetic series): the engine's confidence (NCC) already collapses
when a frame fails, but the default `min_confidence=0` applied the spurious shift
anyway — and `previous` mode then accumulated it (late-frame error reached 66–89 px;
now bounded to ~10 px with the new defaults).

### Added

- **Late-frame robustness** (`backend/registration/estimate.py`, `.../method.py`):
  - **Hold-last-good gating** — `estimate_series` now holds the last good transform
    (absolute modes) or skips the increment (`previous`) when `confidence <
    min_confidence`, instead of snapping to identity; returns a per-frame `gated`
    bool. `RigidRegistration` default `min_confidence` raised `0.0 → 0.2`.
  - **`reference="template"`** — a two-pass anchor (`_build_template`: rough-stabilize
    → average → register all frames to the template). Robust to cumulative drift and
    to a bleached/atypical single anchor frame. **New default** reference.
  - **`normalize="zscore"`** per-frame intensity normalization (counters photobleaching).
  - `estimate_translation` now passes `disambiguate=True` (large-shift wrap guard).
- **Region of interest (ROI)** — estimate the transform on a chosen sub-region, apply
  it full-frame (lock onto a static landmark; ignore moving cells / debris):
  - `estimate.roi_to_mask(roi, shape) -> (mask, bbox)`; `estimate_translation(mask=,
    bbox=)` (rectangle → subpixel crop; freeform → masked phase correlation);
    `ecc_align(mask=)` → ECC `inputMask`; `estimate_series(roi=)`.
  - Node: hidden `roi` `ParamSpec` + a **"Pick ROI…"** popup button
    (`pipelines_page._edit_registration_roi`) offering whole-frame / rectangle
    (x/y/w/h) / freeform (drawn on the viewer, `_on_registration_shape`). Freeform
    shapes rasterize via `backend/analysis/manual_mask.rasterize_shapes`.
- **Feature-based model** (`model="feature"`) — `estimate.estimate_features` (ORB
  keypoints + `match_descriptors` + RANSAC, Euclidean/Similarity/Affine) for large
  displacement / rotation / scale / partial overlap (re-mount, multi-round); inlier
  fraction is the confidence, too few inliers → identity fallback. New method params
  `feature_transform`, `min_inliers` (visible when `model=feature`).
- **Crop to common region** (translation) — node toggle `crop_to_common`: after
  registration, crop every frame of every multipoint to the **largest rectangle that
  is real (non-padded) data in all registered frames** (`estimate.common_translation_crop`),
  so frames are equal-size, recentred, and free of the black drift borders. A single
  region across all M (identical sizes). Published as `record._registration_crop` and
  composed into `pipelines_page._crop_rect()` (intersected with any preview crop), so it
  flows **everywhere downstream** — analysis (`_processed_channels_for_m`), export /
  validation / spatial maps (`_materialize_channels_for_m`), the viewer base
  (`_maybe_crop_volume`), the write-back, and the Registration panel — through the single
  existing crop chokepoint. Register → crop (full-frame first). Cleared on each Run start.
- **`tests/registration/test_robustness_roi_features.py`** — 15 tests: template+gating
  vs the old default on a degraded series, `previous` no-explode, ROI beats whole-frame
  + rect subpixel + freeform, feature rotation recovery / blank-field fallback / warp
  convention, and common-region crop (valid-region math, no-overlap → None, end-to-end
  equal-size border-free frames).

### Changed

- **`widgets/registration_panel.py`** — `set_data(gated=, min_confidence=)`; the drift
  plot shades **held (low-confidence)** frames and the meta line counts them, so
  late-frame degradation is visible at a glance.
- **`pages/pipelines_page.py`** — `_RegisterJob` bundle carries `gated`/`min_confidence`;
  the ROI spec rides through `node.params` into `method.run` (no job-signature change).

### Bug Fixes

- **Registered image played slowly in the viewer while the raw played fast.** The
  registered display is the lazy `RegisteredFrameVolume`, whose `get_frame` ran a
  full-resolution warp (`scipy.ndimage.shift` / `cv2.warp`) on **every** frame with no
  cache, so playback/scrubbing recomputed it each tick (the raw streams a
  pyramid/RAM plane). Fixed by adding a memory-budgeted LRU of warped display frames to
  `RegisteredFrameVolume` (`executor.py`, mirrors `ProcessedFrameVolume`; transforms are
  deterministic per `(m,t)` so caching is safe) — each frame is warped once, then loops
  are instant. Also, `_apply_registration_writeback` now drives the display through that
  cached lazy volume (`_show_base_image`) for lazy-volume files instead of pushing the
  full-res materialized stack via `set_channels`.

## [Unreleased] - 2026-07-07 (Pause-node crop preview on the registered image — V1.59)

Plan: `CodeLog/ClaudesPlan/V1.59_pause_crop_preview_registered.md`.

Makes the **Pause** node a troubleshooting checkpoint: when a Run pauses back to
editor mode, the **Preview Crop** tool now drives a *downstream preview* on the
cropped sub-region — and, when an **Image Registration** node ran upstream, that
preview (and the displayed base) is the drift-corrected image, so the user can
crop and check segmentation / tracking on the registered image before a full Run.
Preview-only: a resumed Run stays full-frame.

### Added

- **`backend/registration/estimate.py`** — `apply_frame(frame, transforms, t,
  interp_order=1)`: the single-plane analogue of `apply_series`; applies the
  `t`-th per-frame shift/warp to one 2-D frame (identity / absent / out-of-range
  `t` → unchanged).
- **`pipeline_graph/executor.py`** — `RegisteredFrameVolume`: lazy `get_frame`
  wrapper (`bypass_pyramid = True`) that applies the per-`(m, t)` registration
  transform on read (no-op for an M without a transform). Exported from
  `pipeline_graph/__init__.py`. Wrap **before** `CroppedVolume` → a crop of the
  registered image.

### Changed

- **`pages/pipelines_page.py`**
  - New helpers `_maybe_register_volume(record, vol)` and `_register_frame(record,
    frame, m, t)` (gated on `record._registration_by_m`).
  - Analysis/Results preview now applies registration then crops:
    `_extract_processed_frame` registers the lazy-volume frame before return;
    `_materialize_channels_for_m` registers the full stack before cropping.
  - `_show_base_image` wraps the base in `_maybe_register_volume` before
    `_maybe_crop_volume` (and prefers `_processed_channels` in the in-RAM
    fallback), so arming the crop no longer reverts the viewer to the
    un-registered image.
  - `_processed_channels_for_m` now registers the **full** frame first, then crops
    for a cropped Run (register → crop, was crop → register).
  - Pause status message mentions the crop-preview workflow.

## [Unreleased] - 2026-07-07 (Tracking: birth/death LAP + mask-overlap linker — V1.58)

Plan: `CodeLog/ClaudesPlan/V1.58_tracking_lap_overlap.md`.

Fixes the two post-StarDist tracking failure modes on dense monolayer nuclei: a
high `max_distance` linked slow cells to distant ones, and a low `max_distance`
chain-linked cells to their neighbors into propagating "currents". Both trace to
the per-frame Hungarian assignment having no birth/death ("no-match") option, so
it was forced to link the smaller side in full. Adds the Jaqaman-style LAP with
no-match nodes and a new mask-overlap (IoU) linker that consumes the StarDist
masks directly.

### Added

- **`backend/celltracker/tracking.py`**
  - `solve_lap(cost, no_match_cost)` — Jaqaman et al. (2008) augmented
    assignment with birth/death diagonals; a detection may stay unmatched at a
    fixed cost instead of being force-linked. Splits the gated cost matrix into
    connected components of finite-cost edges and solves each block
    (`_solve_lap_block`) so dense fields stay fast (exact — dummy nodes never
    couple components).
  - `track_overlap(df, masks, min_iou=0.1, max_gap=1, no_match_cost=None,
    progress_cb=None)` — links segmented objects by mask intersection-over-union
    (`cost = 1 − IoU`, pairs below `min_iou` forbidden) via `solve_lap`; keeps a
    track's last footprint for up to `max_gap` missed frames for overlap
    re-linking. Helper `_frame_footprints` extracts per-label pixel indices in
    one stable argsort over the foreground.
- **`backend/object_tracker.py`** — `METHOD_CT_OVERLAP = "Cell-Tracker: Mask
  Overlap (IoU)"` (added to `TRACKING_METHODS`); `_link_group_overlap` bridge;
  new `link_objects` args `ct_min_iou` and `label_masks`
  (`{(segmentation_channel, m_position): (T,H,W)}`), also on
  `link_objects_with_params`.
- **UI** — Track Objects node exposes the new method plus a `ct_min_iou`
  ("Min overlap (IoU)") knob (`pipeline_graph/registry_adapter.py`).
- **`tests/test_tracking_lap_overlap.py`** — birth/death LAP + overlap tests.

### Changed

- **`backend/celltracker/tracking.py`** — `link_frames` and `track_fingerprint`
  now route their assignment through `solve_lap` (birth/death) instead of a bare
  `linear_sum_assignment` + post-gate; out-of-gate pairs are marked `inf`.
  `link_frames` / `track_timeseries` / `track_fingerprint` gain an optional
  `no_match_cost` (defaults to `max_dist`). Gating is on raw distance, so
  topology only ranks reachable candidates.
- **`widgets/common.py`** — `ParamSpec.visible_when` values may be a single
  choice or a list/tuple/set of choices (row shows if the current choice is any
  of them). `ct_max_gap` is now shown for both the fingerprint and mask-overlap
  linkers.
- **`pages/pipelines_page.py`** — `_TrackJob` accepts `label_masks`;
  `_run_track_objects` builds `{(channel, m): (T,H,W)}` from the per-M analysis
  results (by reference) and passes it to the linker.

### Bug Fixes

- Tracking no longer force-links a slow cell to a distant one (high
  `max_distance`) or chain-links neighbors into "currents" (low `max_distance`);
  unmatched detections become births/deaths. On the reference monolayer clips,
  implausible one-frame steps (> nuclear spacing) dropped from ~861/489/2554
  (current topology linker) to ~7/3/31 with the overlap linker; the birth/death
  LAP alone roughly halves them on pure centroids.

## [Unreleased] - 2026-07-07 (Export cropped data from the Import tab — V1.57)

Plan: `CodeLog/ClaudesPlan/V1.57_import_tab_crop_export.md`.

Adds a self-contained **Export** control to each Import-tab `FilePanel`, so the
user can write the currently-loaded (cropped) dataset to disk without leaving the
first tab. Export honors **both** crops: the existing **T/M/Z tile-strip crop**
(already baked into `record._raw_volume` by `CropWorker`) and a **new spatial XY
ROI crop** drawn on the Import viewer. Works on any panel (primary and secondary
side-by-side panels), each against its own `record`.

### Added

- **`nd2studios/widgets/file_panel.py`** — new "Export" group in the controls
  sidebar:
  - **XY crop tool** — `btn_crop_xy` (checkable, `fa5s.crop-alt`) enables
    `MultiAxisViewer.set_crop_mode`; a rubber-band drag (`crop_rect_selected`) or
    click (`canvas.clicked`, guarded by the toggle) opens `_show_xy_crop_dialog`
    (spinbox confirm/edit, clamped to frame bounds). `_apply_xy_crop(x,y,w,h)`
    stores the rect in `self._xy_crop` and `record.crop_rect` and updates
    `lbl_xy_crop_status`; `btn_reset_xy` (`fa5s.undo`) / `_reset_xy_crop()` clears
    it. The rect is stored (not applied to the live viewer) and realized only in
    the exported file, so it composes with the T/M/Z crop and leaves M/T/Z
    browsing intact. Reset automatically on each fresh load.
  - **Export menu** — `btn_export` (`primaryBtn`, `fa5s.download`, disabled until a
    file loads) opens a `QMenu`: *TIFF hyperstack…*, *Movie (MP4)…*, *Movie
    (GIF)…*, *Image sequence (PNG)…*.
  - **Export dispatch** reuses `ExportWorker`/`ExportRequest` wholesale:
    `_export_source_channels()` builds the current M/Z channels via
    `volume.all_channels_as_lazy(...)` and applies the XY crop through each
    channel's `.crop(y0,y1,x0,x1)`; `_colors_enabled_lut()` reads colors/enabled/
    LUT from `viewer.channel_state()`. TIFF uses `tiff_zstack` (raw volume +
    `crop_rect`) for unprojected multi-Z stacks, else `tiff_stack` with pre-cropped
    channels. Movie and image-sequence materialize the cropped channels and open
    the shared `ExportPreviewDialog` (T-scrub + brightness/contrast/etc.) before
    dispatch. `_run_export`/`_on_export_done` route progress/status through the
    panel's existing `on_progress`/`on_status` callbacks; movie defaults to
    `MovieOptions(fps=10)` with scale bar / timestamp / channel-label overlays on.

## [Unreleased] - 2026-07-07 (Image registration — drift-correction processing node — V1.56)

Plan: `CodeLog/ClaudesPlan/V1.56_image_registration.md`.
Research: `Research/image_registration.md`.

Adds **image registration**: stabilize a `(T,H,W)` channel series onto a reference
frame (temporal drift correction), translation / rigid / affine. Ships two homes
sharing one pure, Qt-free engine — a per-channel `EnhancementPlugin` (the quick
recipe-step win) **and** a cross-channel `RegistrationMethod` + pipeline node
("register once on the reference channel, apply to all channels"). The node is a
**transform in the pipeline** (not a terminal viewer): wire it upstream of analysis /
tracking and its drift correction flows to every downstream node, so analyses run on
drift-free images.

### Added

- **`nd2studios/backend/registration/`** — new backend-pure package (no PySide6):
  - **`estimate.py`** — the estimation + resampling engine, reusing the stitcher's
    `_highpass`/`_hann2d`/`_ncc` primitives (`backend/stitch/register.py`):
    - `estimate_translation(reference, moving, upsample=20, highpass_sigma=2.0,
      window=True) -> (shift (row,col), ncc)` — sub-pixel phase correlation
      (`skimage.registration.phase_cross_correlation`, `normalization="phase"`).
    - `apply_shift(image, shift, order=1)` — `scipy.ndimage.shift`, dtype-preserving.
    - `ecc_align(reference, moving, model="euclidean", init_shift=None, iters=200,
      eps=1e-6, gauss=5, interp_order=1) -> (warp_matrix, cc, aligned)` —
      `cv2.findTransformECC`, seeded from a phase-correlation translation.
    - `apply_warp(image, warp_matrix, motion=None, output_shape=None, interp_order=1)`
      — resample through an ECC warp (`cv2.WARP_INVERSE_MAP`), dtype-preserving.
    - `stabilize(volume, model="translation", reference="previous", upsample=20,
      highpass_sigma=2.0, interp_order=1, min_confidence=0.0, progress_cb=None,
      cancelled_cb=None) -> (aligned (T,H,W), shifts (T,2), confidence (T,))` —
      the series driver. Reference modes `first`/`previous` (cumulative)/`mean`;
      models `translation`/`euclidean`/`affine`; confidence gating falls back to
      identity below `min_confidence`. `MODELS`/`REFERENCE_MODES` constants exported.
    - `estimate_series(series, model, reference, upsample, highpass_sigma,
      min_confidence, …) -> {"model","reference","shifts" (T,2),"warps" (T,2,3)|None,
      "confidence" (T,)}` — per-frame **absolute** effective transforms (composed ECC
      warps for cumulative mode) that align each frame onto the anchor.
    - `apply_series(series, transforms, interp_order=1) -> aligned (T,H,W)` — apply
      those transforms to any channel (the "apply to all channels" half of
      register-once).
- **`nd2studios/plugins/enhancement/registration.py`** — `RegistrationPlugin`
  (`@EnhancementPlugin.register`, name `"Registration (Drift Correction)"`): an
  `Image -> Image` Processing node auto-enumerated by
  `registry_adapter.enhancement_specs()`. Params `model`, `reference`, `upsample`,
  `highpass_sigma`, `interp_order`, `min_confidence`; `execute` delegates to
  `estimate.stabilize` and passes single frames / non-series through unchanged.
- **`nd2studios/core/registration_registry.py`** — `RegistrationMethod(ABC)` +
  `RegistrationResult` (per-frame `shifts_px`/`transforms`/`aligned`/`confidence`/
  `pixel_size_um`, `shifts_um()`), reusing `ParamSpec`. Mirrors `core/dvc_registry.py`.
- **`nd2studios/backend/registration/method.py`** — `RigidRegistration`
  (`@RegistrationMethod.register`, `"Rigid / Translation"`): estimates the transform on
  the reference series (`estimate_series`) and returns the bundle + aligned reference.
- **Registration node + viewer** — a Special pipeline-graph node
  (`SPECIAL_REGISTER_OP_KEY = "special:register"`, a HEXAGON with an `IMAGE` rainbow
  channel input **and an `ANY` structural output**, in
  `pipeline_graph/registry_adapter.py`) wired into `pages/pipelines_page.py`
  (`_RegisterJob` off-thread reads each channel's `(T,H,W)` via `get_frame`, registers
  the wired reference channel, applies the same transform to all channels;
  `_run_register`/`_finish_register`), with a new `widgets/registration_panel.py`
  "Registration" viewer tab (before/after playback + drift-vs-time plot + confidence).
  Params come from `RigidRegistration().get_params()` + node scope
  (`apply_to_all_channels`, `all_multipoints`).
- **Registration flows downstream** — the node is a pipeline transform, not a terminal
  sink: wire it upstream of analysis / tracking nodes. `_finish_register` publishes the
  per-multipoint per-frame transforms on the record (`record._registration_by_m`);
  `_processed_channels_for_m` applies them after the recipe
  (`_apply_registration_to_channels` → `estimate.apply_series`) so **every downstream
  analysis runs on the drift-corrected image, across all multipoints**. The image
  viewer is redrawn to the aligned base for the shown M. `_on_run` clears the transforms
  at the start of each Run (a graph without a Registration node is unaffected).
- **`tests/registration/`** — synthetic-ground-truth tests (13): known sub-pixel
  shift recovery (≤0.15 px @ upsample=20), apply/estimate round-trip, random-walk
  drift reduction for `first`/`previous`/`mean`, dtype preservation, register-once-
  apply-to-all, ECC euclidean rotation recovery, affine path smoke test.

### Changed

- **`nd2studios/__main__.py`** — force-import
  `nd2studios.plugins.enhancement.registration` and
  `nd2studios.backend.registration.method` at startup so the
  `@EnhancementPlugin.register` / `@RegistrationMethod.register` decorators fire.
- **`nd2studios/pipeline_graph/__init__.py`** — re-export `SPECIAL_REGISTER_OP_KEY`.
- **`nd2studios/pages/pipelines_page.py`** — `registration` overlay tab (keys/labels/
  `_on_overlay_tab_changed`/`_select_overlay_tab`/`_update_overlay_tabs_available`),
  `_RUN_REGISTER_KEY` runner wiring (done/cancel/progress), preview + Run dispatch for
  `SPECIAL_REGISTER_OP_KEY`, and `_reg_by_m` / `_registration_panel` state.

## [Unreleased] - 2026-07-07 (Multipoint stitching pipeline — regime-aware rebuild — V1.54)

Plan: `CodeLog/ClaudesPlan/V1.54_multipoint_stitch_pipeline.md`.
Research: `Research/multipoint_stitching.md`.

The multipoint (M) stitching **method** was replaced with a regime-aware
pipeline (per the *Build Spec: Multipoint TIFF Image Stitching Pipeline*).
"Stitching" is two problems: **overlapping** tiles are *registered* from image
content; **zero-overlap** tiles share no pixels and must be placed from **stage
coordinates** only. The pipeline inspects each dataset, decides the regime, and
routes accordingly (a registrar on zero-overlap data would lock onto a noise
peak). The **only** thing kept from the old stitcher is how metadata orients the
M frames (stage-XY reads + the Nikon sign/flip convention).

### Added

- **`nd2studios/backend/stitch/`** — new backend-pure package (no PySide6):
  - **`config.py`** — `StitchConfig` (regime, engine, `zero_overlap_tol`,
    `overlap_frac`, `pixel_size_um`, `axis_flip_x/axis_flip_y/swap_xy`,
    `align_channel`, `filter_sigma`, `max_shift_um`, `ncc_threshold`,
    `upsample_factor`, `per_timepoint_registration`, `blend`,
    `feather_width_px`, `fill_value`, `illumination_correction`, `z_mode`,
    output/pyramid/memory knobs) with `to_dict`/`from_dict`. Defaults
    (`axis_flip_x=True, axis_flip_y=False, swap_xy=False`) reproduce the exact
    pre-V1.54 placement.
  - **`positions.py`** — kept orientation math: `oriented_offsets_um`,
    `coordinate_offsets_px`, and `compute_tile_layout` / `StitchLayout`
    (backward-compatible with the preview widgets); `_cluster_axis`,
    `_assign_index`.
  - **`dataset.py`** — `Tile` / `Dataset` + `build_dataset(volume, stage_xy_um,
    m_indices, config)`: per-tile grid (row/col), oriented offsets, grid-shape
    inference, and overlap-fraction inference (`1 − step/tile_extent` per axis).
  - **`regime.py`** — `decide_regime(dataset, config)` → `"overlap"` /
    `"zero_overlap"` (explicit override, no-stage/single-tile → zero-overlap,
    else `overlap_frac > zero_overlap_tol`).
  - **`register.py`** — built-in phase-correlation engine:
    `refine_positions(dataset, ref_frames, seed, config)`. Per adjacent pair:
    high-pass + Hann window + `skimage.registration.phase_cross_correlation`
    (`upsample_factor`), NCC confidence + `ncc_threshold` rejection, `max_shift`
    bound; global optimization by weighted least squares Tikhonov-anchored to the
    coordinate seed (`(L+λI)p = λs + c`, row/col decoupled). Falls back to
    coordinates when no edge is trusted.
  - **`engines.py`** — `compute_positions(dataset, ref_frames, config, regime)`
    dispatch: `coordinate` / `phase_correlation` / `m2stitch` (grid; seeds
    `position_initial_guess` from coords, reads `y_pos`/`x_pos`) / `ashlar`
    (fully wired via `ashlar_engine.py`, never chosen by `auto`) / `auto`
    (m2stitch on a clean grid else phase-correlation). `seed_positions`.
  - **`ashlar_engine.py`** — `ashlar_positions(...)` + `ashlar_available()`:
    in-memory `Metadata`/`Reader` adapter → `EdgeAligner` → `aligner.positions`;
    auto-points `JAVA_HOME` at `jdk4py` so ashlar's `import jnius` succeeds
    without a system JDK (the JVM never starts — registration is pure Python).
  - **`compositor.py`** — `composite_frame(...)` with `feather` (border alpha
    ramp) / `average` / `max` / `none` blending; float accumulation cast back to
    the source dtype (uint16 → uint16); gap fill; `canvas_size`,
    `normalize_positions`.
  - **`illumination.py`** — `estimate_flatfield(tiles, config)` + `Flatfield`:
    `basic` (BaSiC/basicpy, gated — raises if unimportable), `builtin`
    (large-σ Gaussian flat-field fallback, always available), `supplied`.
    `basicpy_available()`.
  - **`writer.py`** — `write_ome_tiff(data, out_path, pixel_size_um,
    channel_names, config)`: pyramidal, tiled, BigTIFF-auto OME-TIFF (`TZCYX`,
    `PhysicalSizeX/Y` in µm, channel names, `subifds` pyramid levels).
  - **`qc.py`** — `write_qc(...)`: JSON + matplotlib-Agg PNG (regime, engine,
    per-tile positions, pairwise confidences, overlap, fallbacks).
  - **`pipeline.py`** — `run_stitch(volume, stage_xy_um, m_indices,
    channel_indices, config, out_path, meta, progress_cb) -> StitchResult`:
    build dataset → decide regime → register once on `align_channel` → optional
    illumination → composite every (T,Z,C) frame (disk-backed memmap when the
    mosaic exceeds `max_memory_gb`) → pyramidal OME-TIFF → QC. Register-once,
    apply-to-all-C/Z/T (spec §7); `per_timepoint_registration` re-registers per T.
- **`tiff_loader.read_ome_tiff_metadata(filepath)`** — reads level-0 OME-TIFF
  metadata (n_channels/T/Z, channel names + `PhysicalSizeX` via `ome-types`),
  ignoring pyramidal sub-resolutions. `_SingleFileTIFFView` gained an OME branch
  (multi-channel per-channel proxies; `_level0_pages()` sources frames from
  `series[0].levels[0]`), so a stitched OME-TIFF **reloads** through the normal
  single-file TIFF import (preserves the V1.16 round-trip guarantee).
- **`tests/test_stitch_pipeline.py`** — 12 synthetic-ground-truth tests (regime
  selection, exact zero-overlap placement, phase-corr + m2stitch position
  recovery ≤3 px under jitter, OME round-trip + real reload path, memmap path,
  built-in illumination, dtype preservation, ashlar gating).

### Changed

- **`backend/exporters/stitch_exporter.py`** is now a **compatibility shim**:
  re-exports `StitchLayout`, `compute_tile_layout`, `_cluster_axis`,
  `_assign_index` (from `backend/stitch/positions`), keeps `stitch_one_frame`,
  and turns `export_stitched_tiff` into a **deprecated** wrapper that writes via
  the new OME-TIFF writer (emits `DeprecationWarning`). All old import sites
  (preview widgets, `scripts/inspect_nd2.py`, `scripts/verify_v113_layout.py`)
  keep working.
- **`workers/stitch_worker.py`** — `StitchRequest` now carries `stage_xy_um` +
  `StitchConfig` + `m/channel indices` + `filepath` (no precomputed layout);
  `StitchWorker.run_task` calls `run_stitch` and exposes `.result`.
- **`pages/stitch_dialog.py`** — new **Stitching** group (regime / engine / blend
  / align channel / illumination combos); output is a pyramidal OME-TIFF
  (`.ome.tif`); the completion dialog reports regime/engine/canvas/QC. Tile
  selection + coordinate-placement preview unchanged.
- **Output format changed from ImageJ TZCYX hyperstack to pyramidal OME-TIFF.**
  Reload was updated (above) to keep round-tripping.

### Dependencies

- Installed **m2stitch 0.7.2** (grid overlap engine, MIST reimplementation) and
  **ashlar 1.20.0**. All engine deps are **optional gated extras** (like
  cellpose/stardist), never hard deps; `auto` uses only m2stitch + the built-in
  phase-correlation engine (no Java).
- **ashlar is now fully wired via an in-memory reader** (`backend/stitch/
  ashlar_engine.py`): ashlar's `EdgeAligner` registration is pure Python, so we
  feed it a `Metadata`/`Reader` built from our tiles + coordinate seed and read
  back `aligner.positions` — the JVM never starts. ashlar's `reg` module still
  needs a JDK to satisfy its unconditional `import jnius`; we install
  **`jdk4py`** (packaged Eclipse Temurin OpenJDK, pip wheel, no admin) and point
  `JAVA_HOME` at it at runtime when the user hasn't set one. Verified: ashlar
  recovers jittered grid positions to ~1.3 px. If neither `jdk4py` nor a
  `JAVA_HOME` JDK is present, the ashlar engine raises a clear install hint.
- **basicpy** is installed but not importable here
  (hyperactive/gradient_free_optimizers API drift) → BaSiC is gated with a
  friendly error and the `builtin` flat-field is the working default.
- **pandas 3.0.3 → 2.3.3** (m2stitch requires `pandas<3`). Verified the app and
  full test suite (184 tests) pass on 2.3.3.

## [Unreleased] - 2026-07-06 (Checkpoint node — freeze upstream, iterate downstream — V1.53)

Plan: `CodeLog/ClaudesPlan/V1.53_checkpoint_node.md`.

### Changed

- **Analysis preview is now a double-click-only, scoped "mini Run" (V1.49 rework).**
  Two behavior changes to how the Analysis tab preview works:
  - **Only a node double-click (re)generates the preview.** Navigation, parameter
    edits, frame-strip selection, crop changes, and the normalize toggle no longer
    recompute the analysis preview — they still refresh the (navigable) base image,
    but the segmentation/tracking/measurement recompute happens *only* on a
    deliberate double-click. Implemented with an `_analysis_preview_armed` flag set
    by `_on_node_double_clicked` and consumed in `_do_preview` (the analysis branch
    returns early when not armed). The Processing-tab live preview is unchanged.
  - **A double-click runs the pipeline *up to the clicked node* on the selected
    frames and fills every applicable tab.** All analysis double-clicks now route
    through the same scoped-Run walk the results preview used (`_start_preview_walk`)
    instead of the segmentation-only `_do_analysis_preview` path — so a single
    double-click produces Segmentation, Tracks, Vectors, Spatial Maps, the
    measurements table and the plots that the chain up to that node supports.
    Clicking the raw input (nothing to run "up to") falls back to the whole
    downstream chain via the new `_resolve_preview_run_target()`. The double-click
    also re-asserts the base image so the overlay always lands on a current frame.
    (`nd2studios/pages/pipelines_page.py`)

### Added

- **Checkpoint node** (white, in the Add-dialog "Checkpoint" tab). A pass-through
  special node in the merged Analysis graph that *freezes* everything computed
  upstream (segmentation / analysis label masks, measurement rows, tracks) when a
  Run reaches it. On a later Run, as long as the upstream graph is unchanged, the
  run **resumes from the checkpoint** with the frozen data restored — the
  expensive upstream work (StarDist, tracking) never re-runs, so the user can
  add / change / rewire downstream nodes and Run instantly.
  - **Auto invalidation via upstream hash.** `_checkpoint_upstream_hash` hashes
    the checkpoint's structural ancestors (op_key + params), the edges feeding
    them (structural + channel wiring), any touching loop-edge config, plus the
    Processing recipe / per-channel recipes / normalized flag and the Run crop.
    A Run resumes from a checkpoint only when its stored hash still matches;
    otherwise it re-runs from Input and re-freezes. Among several valid
    checkpoints, the deepest (furthest-downstream) one is chosen.
  - **Session RAM + per-node disk persistence.** Frozen masks / rows live in
    `PipelinesPage._checkpoint_store` (`node_id -> {"hash", "data"}`) for the
    session, released when the checkpoint is deleted (`_on_graph_changed` prunes
    orphans) and cleared on file change (`load_from_experiment`). Each checkpoint
    node has a **`persist_to_disk` bool param** (default off = session-only):
    when on, **saving the pipeline** writes *that* checkpoint's frozen data to a
    companion `<pipeline>.checkpoints/` cache (compressed NPZ of per-M label masks
    + an `index.json` manifest holding the rows, track state, overlay style,
    upstream hash and a source-file signature), and **loading the pipeline**
    restores it, so the checkpoint survives app restarts. Session-only checkpoints
    are never written, and toggling one back off removes its stale cache on the
    next save. A cache reloaded against a different file or an edited graph re-runs
    harmlessly because the source-file signature is folded into the upstream hash.
  - **Resume execution.** `GraphRunner.__init__(sl, frozen=<ancestor ids>)` marks
    each frozen node reached + done + dispatched and reaches its structural
    successors, so a topological walk skips the frozen ancestors while the
    checkpoint still runs and reconverging downstream nodes gate correctly.
    `frozen` empty ⇒ identical to the pre-V1.53 behavior.
  - **Visualization.** New `"cached"` node run-state (dimmed body + dashed white
    border) shades the frozen upstream nodes during a resumed Run;
    `NodeCategory.CHECKPOINT` maps to a new `Settings.ACCENT_WHITE`.
  - New `SPECIAL_CHECKPOINT_OP_KEY` / `NodeCategory.CHECKPOINT`; new Qt-free
    `pipeline_graph/checkpoint_io.py` (`save_checkpoints` / `load_checkpoints` /
    `checkpoints_dir_for`); page methods `_run_checkpoint`,
    `_checkpoint_ancestors`, `_checkpoint_upstream_hash`,
    `_checkpoint_resume_target`, `_restore_checkpoint`, `_record_signature`,
    `_save_checkpoint_cache`, `_load_checkpoint_cache`.
    (`nd2studios/pipeline_graph/{model,registry_adapter,executor,checkpoint_io,__init__}.py`,
    `nd2studios/core/settings.py`,
    `nd2studios/widgets/node_board/{node_scene,node_item,add_node_dialog}.py`,
    `nd2studios/pages/pipelines_page.py`)

## [Unreleased] - 2026-07-06 (SerialTrack PTV field: bounded interpolation — V1.52)

Plan: `CodeLog/ClaudesPlan/V1.52_serialtrack_bounded_field_interp.md`.

### Bug Fixes

- **SerialTrack PTV field vectors were enormous even though per-particle
  displacements were small (< 20 px).** The per-particle vectors from
  `particle_displacement` were correct; the *gridded field* was extrapolating
  without bound. `compute_field_bundle` and the panel defaulted `smoothness` to
  `1e-3` (> 0), and `scatter_to_grid` routes any `smoothness > 0` through
  `RBFInterpolator(kernel="thin_plate_spline", degree=1)`. Thin-plate-spline is
  a global interpolant whose kernel grows like `r²·log r`; the grid spans the
  particle **bounding box**, so its corners / inter-cluster gaps lie outside the
  convex hull, where the RBF extrapolated with no bound and no hull mask (a
  clustered repro with all displacements < 21 px produced field values of
  ±134 px). Fix: **wire back the original Cell-Tracker approach** for the
  field-*output* path — linear interpolation inside the convex hull, zero
  outside it (never extrapolated), plus an optional Gaussian blur. New
  `scatter_to_grid_bounded` / `scatter_to_grid_multi_bounded` in
  `serialtrack/regularization.py`; `compute_gridded_strain` now uses them and
  reinterprets `smoothness` as a **Gaussian σ in grid cells** (Cell-Tracker's
  `sigma`). The tracking global-step solvers (`solve_regularization`,
  `ADMMLSolver`) still use the RBF `scatter_to_grid` and are unaffected. Post-fix
  the field is bounded by the per-particle displacement range (±14 px).
  (`nd2studios/backend/serialtrack/regularization.py`,
  `nd2studios/backend/serialtrack/fields.py`,
  `nd2studios/backend/serialtrack_analysis.py`,
  `nd2studios/widgets/serialtrack_panel.py`)

### Changed

- **SerialTrack panel "Smooth" control** now represents a Gaussian smoothing σ
  in grid cells (decimals 4→1, step 0.001→0.5, default 0.001→1.0) with a tooltip
  describing the bounded (no-extrapolation) field model.

## [Unreleased] - 2026-07-06 (Digital Volume Correlation node — ALDVC — V1.51)

Adds **Digital Volume Correlation** (DVC) as a pipeline-graph analysis node — a
clean-room Python port of FranckLab's Augmented Lagrangian DVC (ALDVC;
Yang/Hazlett/Landauer/Franck 2020). Measures the dense displacement + strain
field between a reference and a deformed timepoint of a wired channel: true
**3D DVC** on confocal Z-stacks (reads full `(Z,H,W)` volumes), with a **2D DIC**
fallback when there is no Z. Brings forward + completes the `Add-DVC` branch's
Phase-0 scaffold (which only stubbed a single global FFT shift). Plan:
`CodeLog/ClaudesPlan/V1.51_dvc_aldvc_node.md`; literature review:
`Research/aldvc_literature_review.md`.

### Added

- **`backend/dvc/` engine** (new, pure numpy/scipy/scikit-image — no PySide6) —
  the full ALDVC pipeline: `mesh.py` (subset grid + DOF pack/unpack, layout
  matching SerialTrack's FD operator), `integer_search.py` (per-subset windowed
  FFT normalized-cross-correlation seed + parabolic subvoxel + q-factor),
  `icgn.py` (inverse-compositional Gauss-Newton subset solver — 6-DOF 2D /
  12-DOF 3D, cached reference Hessian, warp composition, ZNSSD objective,
  `map_coordinates` warping; optional μ/β penalty for the ADMM Subpb1),
  `outliers.py` (cc + normalized-median test + nearest-value inpaint),
  `global_step.py` (augmented-Lagrangian FD solve `(β·DᵀD+μI)û = β·Dᵀ(F+w_F) +
  μ(u+w_u)` **reusing** `serialtrack.regularization._build_gradient_operator`,
  plus F-coupling, β L-curve, Tikhonov, cached factorization), `admm.py` (the
  ADMM outer loop + scaled duals + convergence), `strain.py` (infinitesimal /
  Green-Lagrange / Almansi / Hencky measures with anisotropic-voxel scaling),
  `engine.py` (`run_aldvc` staged orchestration). Validated on synthetic known
  deformations to sub-voxel accuracy (2D rigid ≈0.003 vox, 2D affine ≈0.001 vox
  + strain, 3D rigid ≈0.1 vox).
- **`core/dvc_registry.py`** (brought forward from `Add-DVC`) — `DVCMethod` ABC
  registry + `DVCResult` (dense displacement/strain field container) + typed
  `DVCParams` (now incl. `search_radius`, `strain_smooth`); `backend/dvc/method.py`
  registers `ALDVCMethod` (force-imported in `__main__.py`), whose `get_params`
  is the single source of truth for the node's ParamSpec list.
- **DVC pipeline node** — a Special node `"DVC (ALDVC)"` (`SPECIAL_DVC_OP_KEY`,
  HEXAGON) in `registry_adapter._SPECIAL_OPS`, with an `IMAGE` input so it gets a
  rainbow channel port (wire a channel pill to choose the channel). `_SPECIAL_OPS`
  entries gained an optional 7th `input_types` override; `param_specs_for` exposes
  `ref_frame` / `def_frame` + the ALDVC engine knobs.
- **DVC viewer tab** (`widgets/dvc_panel.py`) — a "DVC" overlay tab that renders
  displacement magnitude/components, strain components, effective strain,
  divergence, curl, det(F), and von-Mises stress as heatmaps / contours / quiver
  over the reference backdrop, with a Z-slice slider for 3D. Reuses
  `serialtrack_analysis.scalar_field` by building a `FieldBundle` from the
  `DVCResult` — no DVC-specific field maths.
- **`pages/pipelines_page.py` wiring** — `_DVCJob(AnalysisJob)` runs the engine
  off the GUI thread; `_run_dvc` reads the reference/deformed `(Z,H,W)` volumes
  straight from `record._raw_volume.get_frame(z_mode="none")` (bypassing the
  Z-collapse) for the viewed multipoint; `_finish_dvc` stores the field, opens the
  DVC tab, completes the node. Progress/cancel/tab wiring mirrors Track Objects +
  the SerialTrack panel.
- **Parallelism** — `backend/dvc/parallel.py` fans the IC-GN sweep across a
  shared-memory `ProcessPoolExecutor` (reusing
  `compute/parallel/shared_array.py`); `local_icgn`/`run_admm`/`run_aldvc` take
  `n_workers` (node param, 0 = auto = cores−1) and parallelize by default on grids
  ≥ 64 subsets. Validated to match the serial result on synthetic affine 3D.
- **GPU (opt-in)** — the FFT integer-search seed is backend-agnostic: `_ncc_fft`
  runs the normalized cross-correlation on numpy/scipy (CPU) or, when the
  `use_gpu` node param is on and CuPy is present, on `cupy`/`cupyx.scipy.signal`
  with the volumes resident on the device (guarded, automatic CPU fallback). The
  IC-GN sweep stays on CPU cores.
- **All-multipoints sweep** — an `all_multipoints` node param runs one
  reference→deformed pair per multipoint; results are keyed by M and switchable in
  the DVC tab (`m_change_requested`). Default remains the single viewed M.
- **DVC viewer — Cell-Tracker scale controls** — Auto (2–98th-percentile fit,
  symmetric for divergent fields), manual **Min/Max**, and a **Global Scale**
  button that fixes the range over the whole 3-D field (so the colour scale holds
  while scrubbing Z) — the same model as `SpatialMapsPanel`.
- **DVC viewer — image-viewer zoom controls** — Home / + / − / Pan / zoom-%,
  scroll-to-zoom and drag-to-pan (same button set as the app's `ZoomToolbar`,
  driving the matplotlib axis limits), plus per-field render range applied to the
  heatmap / filled + line contours.
- **Full-timelapse field series + playable viewer** — the DVC node now computes a
  field for **every frame** (not a single pair), faithful to `main_ALDVC.m`'s
  `ImgSeqNum = 2..N` loop: `tracking_mode` = **cumulative** (each frame vs the
  fixed reference) or **incremental** (each frame vs the previous). The DVC tab
  gained a `FrameStrip` + play/pause + fps so the series plays like every other
  viewer; "Global Scale" now fits the colour range across the **whole series**.
  Node params `tracking_mode`, `z_start`/`z_end` (Z sub-range) and `downsample`
  (XY bin) replace the old single `def_frame` — the last two make deep/large
  stacks (e.g. 201×4096² ≈ 6.7 GB/volume) tractable.
- **True 3-D volume fetch** — new `LazyND2Volume.get_volume` /
  `MaterializedDataset.get_volume` return the full `(Z,H,W)` block (one dask
  slice / array view). `get_frame(z_mode="none")` only ever returned a single
  plane, so DVC now reads real volumes; `_DVCJob` reads them per-frame off the
  GUI thread (bounded memory) so a 121 GB file never lands in RAM whole.
- **Robustness** — `mesh.build_grid` guarantees ≥2 grid nodes per axis where the
  extent allows, and the global step uses a DVC-local finite-difference operator
  (`global_step._build_fd_operator`) + `strain` now guards singleton axes, so a
  thin (shallow-Z) grid can't overflow SerialTrack's operator or `np.gradient`.
  Verified end-to-end on the real 121 GB ND2 (2ch×9T×201Z×4096², cumulative +
  incremental, on a Z-cropped ÷16 sub-volume).

### Fidelity follow-up (V1.55 — restores ALDVC's large-deformation / series layer)

Plan: `CodeLog/ClaudesPlan/V1.55_dvc_aldvc_fidelity.md`. Closes the gaps between the
V1.51 core solver and FranckLab `main_ALDVC.m` for large motion + time series:

- **Incremental → cumulative accumulation** (`backend/dvc/tracking.py`) — incremental
  mode now composes the per-step increments into a cumulative field by Lagrangian
  point-tracking (`main_ALDVC.m` 631–706), the field ALDVC actually reports. The DVC
  tab gained a **"Show: Cumulative / Increment"** toggle (the job returns both series).
- **Cross-frame warm-start** — each frame seeds its IC-GN from the previous frame's
  field (`run_aldvc(u0_seed=…, use_fft_seed=False)`; ALDVC's `U0 =
  ResultDisp{ImgSeqNum-2}`), FFT-seeding only the first frame or when the new
  `newFFTSearch` param is set.
- **Multigrid integer seed** (`integer_search_multigrid`, `IntegerSearch3Multigrid`) —
  coarse-to-fine FFT search (new `seed_levels`, default 3) brackets large displacement:
  recovers an 18-vox shift at radius 6 (0.015 vox err) where single-scale fails (15.5);
  neutral on within-radius motion. `integer_search` gained a per-subset `u0_center`.
- New node params: `tracking_mode` (already), `seed_levels`, `newFFTSearch`. Verified
  end-to-end on the real 121 GB ND2 (warm-start + accumulation + dual series).

## [Unreleased] - 2026-07-06 (Loop / iteration connector — V1.49)

Plan: `CodeLog/ClaudesPlan/V1.49_loop_iteration_connector.md`.

### Bug Fixes

- **Double-clicking an analysis node now shows its preview overlay (regression fix).**
  The double-click preview reset (`_reset_analysis_preview`) clears the prior
  per-plane results *before* `_update_merged_view_mode` runs, so
  `_update_overlay_tabs_available` saw no segmentation results, **hid the
  Segmentation tab and fell back to the (overlay-less) Image tab**. The preview
  job then ran (progress bar flashed) and produced masks, but they were never
  displayed because the viewer was on Image — while the Run button worked because
  it explicitly re-selects the Segmentation tab. Fix: the analysis-preview and
  results-walk done-handlers now re-run `_update_overlay_tabs_available()` and
  re-select the Segmentation overlay once a result lands (if the viewer had
  fallen back to Image), mirroring the Run path. Also: the analysis-preview
  handler now surfaces the real pipeline error / an explicit "no result on the
  crop" message instead of a silent generic error.
  (`nd2studios/pages/pipelines_page.py`)

- **Analysis viewer reverted to raw images under a preview crop** (and, more
  generally, whenever the multi-resolution pyramid was engaged). The Pipelines
  viewer receives the BigDataViewer-style raw pyramid (`attach_pyramid`, pushed to
  every page's viewer). `MultiAxisViewer._read_volume_plane` short-circuits to
  `self._pyramid_reader.get_frame(level, …)` when `_choose_pyramid_level() > 0` —
  but the pyramid is built from the **raw** volume, so it bypassed the
  `ProcessedFrameVolume` recipe (and the `CroppedVolume` crop), showing raw,
  full-frame images. Swapping in a `CroppedVolume` made a pyramid level get picked,
  so the Analysis base image dropped back to raw (the Processing preview, which
  pins already-processed planes, looked fine). Fix: the processed / cropped preview
  wrappers (`ProcessedFrameVolume`, `PinnedProcessedVolume`, `CroppedVolume`) now
  declare `bypass_pyramid = True`, and `_choose_pyramid_level` returns 0 for any
  volume carrying that flag — so the viewer falls through to the wrapper's
  `get_frame` (which applies the per-channel recipe / crop). The raw pyramid still
  accelerates the raw volume everywhere else.
- **Double-clicking a node in the Analysis tab now refreshes the preview when a
  preview crop is active.** On a double-click with a crop set, `_on_node_double_clicked`
  re-asserts the cropped base image (`_show_base_image()`) after resetting the
  previous preview and before requesting the new one, so the freshly-previewed
  node's crop-sized overlay lands on a matching, freshly-rendered base (crisp, not
  stretched). No behavior change when no crop is active. (`nd2studios/pages/pipelines_page.py`)

- **Analysis preview now surfaces *why* nothing appeared.** The
  `_ANALYSIS_PREVIEW_KEY` done-handler previously collapsed any not-ok/empty
  result into a generic "Analysis preview error" with no detail and no log. It now
  shows (and logs) the real error message when the pipeline fails, and a distinct
  "produced no result on the previewed frame(s)/crop" message when the pipeline
  ran but returned nothing (e.g. StarDist finding no objects in a small crop) — so
  a "progress flashed but no overlay" case is diagnosable instead of silent.
  (`nd2studios/pages/pipelines_page.py`)

### Bug Fixes

- **Processing pipeline did not go through to Analysis after Apply, when a
  channel pill was wired but the main Input node wasn't** (V1.48 follow-up). In
  the two-layer channel model a channel pill feeds a process's **rainbow**
  (CHANNEL) port, while the process's **structural** image input comes from the
  main Input node. Users naturally wire `channel → process → Output` and skip the
  redundant `Input → process` structural wire — but `recipe_for_node` then raised
  "not connected back to an input", so **Apply committed nothing** (empty recipe,
  no `_processed_view`) and the Analysis tab showed raw. Likewise `GraphRunner`
  started only at the Input node and walked structural edges, so a channel-fed
  analysis node was **unreachable and never ran**. Fix: a channel wire into a
  rainbow port now counts as a valid image source — `recipe_for_node` terminates
  the chain at a channel-fed process (new `node_has_channel_input`), and
  `GraphRunner` treats channel-source pills as pre-completed roots so channel-fed
  processes are reachable and run. Wiring `channel → process → Output` now
  commits and flows to the analysis with **no** separate Input→process wire
  required (wiring it too still works).
- **Channel-specific processing did not carry through to the analysis** (V1.48
  follow-up). Processing and Analysis are separate node slices, each with their
  own channel pills. When an analysis node had **no** channel wired in the
  Analysis slice, `_analysis_channels_for` defaulted to the **first** channel —
  which was often a *raw* channel that the (channel-specific) Processing recipe
  never touched — so the analysis appeared to ignore the processing. Fix: an
  unwired analysis node now defaults to the channel(s) that were **processed**
  upstream (`record.recipe_by_channel`) before falling back to the first channel,
  so a channel-specific Processing recipe flows through to the analysis by
  default; explicit Analysis-slice wiring still wins. Also: the SerialTrack /
  Spatial-Maps backdrop readers now honor per-channel recipes
  (`apply_recipe(..., recipe_by_channel=…)`) instead of applying the single
  recipe to every channel. (The processing→analysis *data* path — base image,
  per-M Run channels, preview frames — was already per-channel-correct; this was
  purely the default channel *selection* + the two backdrops.)

### Added

- **Loop / iteration connector — a new back-edge wire type.** A loop wire exits a
  node's **bottom** output and returns to the **top** input of the same node or an
  upstream node, defining an *iterative loop region* re-run per iteration. It is
  drawn distinctly (amber, routed down the left gutter with a downward arrowhead)
  and is **invisible to every DAG codepath** — recipe linearization, channel
  propagation, topological order, `GraphRunner`, and the cycle check all ignore it
  — so a graph without loops behaves exactly as before.
  - **Model** (`pipeline_graph/model.py`): `Edge` gains `kind`
    (`"structural"`/`"loop"`) + `params` (loop config). New `STRUCTURAL_KIND` /
    `LOOP_KIND`, `is_loop_edge`, `can_connect_loop`, and structural-only
    `GraphSlice` queries (`structural_edges`, `structural_incoming/outgoing`,
    `structural_edge_into_port`, `loop_edges`). `predecessor`, `recipe_for_node`,
    `structural_chain`, `channel_sets`, `edge_channels`, `topological_order`, and
    `GraphRunner` now walk structural edges only.
  - **Pure loop core** (`pipeline_graph/loop.py`, new): `loop_region`,
    `loop_entry_map`, `iteration_plan` (parameter-sweep **grid**/`zip`, fixed
    `count`, `until` with `max_iterations` cap), `expand_axis`,
    `deduplicate_objects` (per-(m, channel, frame) **mask IoU** with a
    centroid-distance fallback), `combine_iterations`
    (`union_dedup`/`best`/`last`/`keep_all`), and `tracking_ratio`
    (coverage + longest continuous run).
  - **Stop conditions** (`pipeline_graph/conditions.py`): new
    `nondup_object_count` and `tracking_coverage` (≥ X% over ≥ K continuous
    frames) blocks, reusing the if-else `Condition` DSL for the "until" mode.
  - **Node board** (`widgets/node_board/`): `EdgeItem` loop routing/style +
    arrowhead + `set_loop_edge`; `NodeScene` loop-wire drag mode
    (`set_loop_mode`), `can_connect_loop`-gated creation (self-loop + upstream
    allowed, cycle check skipped, coexists with the structural feed),
    `loop_edge_edit_requested` signal, scissors cuts loop edges unchanged.
  - **Loop Settings dialog** (`widgets/node_board/loop_dialog.py`, new): opened on
    loop-wire create / double-click — iteration mode, swept-parameter axes (with a
    live iteration count), stop condition (`ConditionBuilderDialog`), combine rule
    (+ dedup metric/threshold / best metric), and multipoint scope.
  - **Execution** (`pages/pipelines_page.py`): a "Loop" toolbar toggle; when a Run
    reaches a loop-entry analysis node it iterates that node's per-M computation
    over the plan (applying param overrides, restored afterward), records each
    iteration's rows + label masks, evaluates the stop condition, then combines
    and publishes the merged result downstream. Multipoint scope: current M or all
    M. Params are restored on finish / cancel.
  - **Per-iteration plots** (`pages/pipelines_page.py`): a loop run overlays every
    iteration on the same analysis plots (Cells/frame line + Area step-histogram)
    in distinct colors with a **legend labelled by the swept parameters**
    (e.g. `#2  Scale=0.5`), plus a dashed **Combined** series — so you can see how
    each sweep value changed the result. `_update_analysis_plots_loop`,
    `_loop_iter_label`, `_iteration_colors`; series stashed at combine time and
    cleared on the next preview/run.
  - **Serialization** (`pipeline_graph/io.py`): `PIPELINE_VERSION` → 4; loop edges
    round-trip via `Edge.kind`/`params`; older graphs load as all-structural (no
    migration needed).

### Bug Fixes

- **Loop combine froze the GUI and ran far longer than the iterations themselves.**
  For a union+dedup loop the overlap de-duplication (`loop.deduplicate_objects`)
  ran on the **GUI thread** and compared every detection pair by allocating a
  **full-frame** boolean mask per pair — O(K²) full-image ops on a dense field
  (hundreds of nuclei × dozens of frames × several iterations), i.e.
  seconds-to-minutes of frozen UI at the end of a loop. Three fixes:
  1. **Bounding-box-gated, cropped IoU.** `deduplicate_objects` now pre-filters
     pairs by their `bbox_*` boxes (a cheap integer overlap test) and computes IoU
     only on the union bounding box; `_rebuild_merged` paints each merged object
     within its bbox slice — no full-frame allocations. A synthetic dense field
     (19,800 detections, 33 frames, 2048² masks) now de-dups in ~0.6 s (was
     minutes).
  2. **Off-thread combine.** The union-dedup combine runs in a background
     `_LoopCombineJob` (like `_TrackJob`), keyed `_RUN_LOOPCOMBINE_KEY`, so the
     window stays responsive; the GUI only does the light per-M result rebuild
     when it finishes (`_finalize_analysis_publish` / `_loop_finalize_union`).
     Cheap rules (best / last / keep-all) stay synchronous.
  3. **Per-iteration channel cache.** A current-multipoint loop caches the
     processed channels once (`_loop_ctx["chan_cache"]`) instead of
     re-materializing the volume and re-applying the recipe every iteration; the
     analysis (e.g. StarDist) still re-runs per iteration as intended.

- **A loop on the Track Objects node did nothing (appeared to "disappear").** The
  loop driver only hooked the analysis node, so a loop anchored on Track was never
  iterated — the node ran once and the loop had no effect (the loop *edge* itself
  persisted). **Fix:** the loop driver now also drives Track-Objects entries —
  `_run_track_objects` begins the loop (`_loop_maybe_begin`) and
  `_finish_track_objects` records each iteration, re-runs tracking with the next
  swept params, then combines. A new `_loop_entry_kind` gates loops to
  Analysis/Track entries (any other entry runs once with a status note instead of
  silently doing nothing); `_loop_start_next_iteration` dispatches the re-run by
  kind. Track sweeps have no masks, so union+dedup falls back to **Last**; the
  meaningful rules are **Keep all**, **Best (by tracking ratio)**, **Last**. The
  "until" stop condition on a Track sweep now evaluates on the iteration's
  `track_id` rows directly (so `tracking_coverage` works).

### Added

- **Per-iteration Tracks/frame + Track-length plots.** `_update_track_plots` now
  overlays every loop iteration as its own colored, legended series (labelled by
  the swept params) plus a dashed **Combined** line — mirroring the Cells/frame +
  Area behavior (`_update_track_plots_loop`). Fixes the Tracks/frame plot not
  updating for a parameter-sweep + Keep-all Track loop.
- **"Save all iterations" + viewer iteration selector.** A new **Save all
  iterations for viewing** checkbox in Loop Settings (`loop_dialog.py`;
  `default_loop_config()` gains `save_iterations`) retains each iteration's result.
  When set, a new **Iteration** dropdown appears beside the overlay tabs
  (`_iter_combo`) listing "Combined" + each "#k …"; selecting an entry re-points
  the viewer overlay, results table and plots to that iteration
  (`_loop_build_saved_store` / `_populate_iteration_selector` /
  `_on_iteration_selected`). Opt-in, since it holds every iteration's masks in RAM.
- **Loops now run on more nodes + an extensible loop-entry registry.** The loop
  driver is now data-driven (`_LOOP_KIND_BY_OP` / `_LOOP_RUN_BY_KIND` /
  `_LOOP_ROW_KINDS`) with a generic `_loop_step(node, publish)` helper, so wiring a
  new node into the loop is a small, uniform change. Newly loopable entries:
  **DVC**, **Registration** (sweep their params; the node's own tab shows the final
  "last" iteration), and **Cell-Tracker Metrics** (a row kind — full combine rules
  + per-iteration plots + the viewer selector). `_loop_begin_combine(force_sync=…)`
  keeps these off the union-dedup worker path; `_loop_entry_kind` lists the
  loopable set and other nodes run once with a status note. New developer guide:
  **`docs/DEVELOPING_LOOP_NODES.md`** documents the extension points + a wiring
  checklist.

## [Unreleased] - 2026-07-06 (Channel-wire pipeline + full M/T coverage — V1.48)

Plan: `CodeLog/ClaudesPlan/V1.48_channel_wiring_and_mt_coverage.md`.

### Bug Fixes

- **Analysis Run / measurement only ever processed one multipoint's pixels** (whichever M
  `record._raw_channels` held — M0 at import), while storing/tagging results under *every*
  M key — so a whole-file Run silently analyzed M0's image for all M. Root cause:
  `_run_analysis_node` captured `channels` once from the single-M `processed_view()`
  (`EnhancedDataset` wraps `record._raw_channels`, which is one multipoint) and
  `_advance_run_m` handed that same object to every per-M `PipelineCommitJob`; `m_index`
  was only a result *tag* (`compute/pipeline_jobs.py`), never used to re-read image data.
  The chained `_ResultsMeasureJob` likewise measured the single-M view. **Fix:** new
  `PipelinesPage._processed_channels_for_m(record, m)` reads that M's channels from the
  lazy volume (`all_channels_as_lazy(m=m)`) and applies the committed recipe;
  `_advance_run_m` builds channels per M and the measurement reuses the same M's channels
  (`_run_channels_m`). All multipoints (and all timepoints within each) are now truly
  analyzed. Peak RAM is unchanged (one M at a time).

### Added

- **Track displacement / motility — measure basis + per-frame outlier rejection (V1.50).**
  Plan: `CodeLog/ClaudesPlan/V1.50_track_displacement_basis_and_outlier_frame.md`.
  The if-else **"Track displacement / motility"** block (`pipeline_graph/conditions.py`)
  gains two params:
  - **Measure** (`basis`) — `Net (first→last)` (default, unchanged behavior),
    `Cumulative path` (the *accumulation* over the whole track = Σ frame-to-frame steps),
    or `Per-frame step` (a single frame-to-frame *instance*; reduced per track as the max
    step, so `max > value` means "any one step exceeds"). Reduced per track by
    `_track_metric_value` over `_track_points`, then run through the existing
    `aggregate`/`comparator`/`value` machinery.
  - **Per-frame exceed** (`outlier`) — `Reject object` (default, route the whole track) or
    `Reject frame only`. With `Per-frame step` + `Reject frame only`, `partition_rows` calls
    the new `scrub_outlier_frames`, which drops each outlier frame row (a frame whose step
    from the last *retained* frame exceeds the threshold) **before** routing — so a spike
    apex is removed while the rest of the track survives and routes normally; the dropped
    frame lands in neither branch. Frame-only rejection is inherently object-lens; a
    whole-frame if-else degrades to `Reject object`. New helpers `_basis_kind`,
    `_reject_frame_only`, `_track_points`, `_track_metric_value`, `_outlier_frame_row_ids`,
    and public `scrub_outlier_frames` (exported from `pipeline_graph`). Existing saved
    conditions default to `Net (first→last)` + `Reject object` (no behavior change).
- **Channel-wire pipeline model (V1.48).** "Which channel a process runs on" is now graph
  *wiring* instead of a per-node `channel_name` parameter:
  - **Channel source nodes** under the Input node — one small colored pill per loaded
    channel plus an **"All"** node — created / reconciled by
    `PipelinesPage._ensure_channel_nodes` (per file). Each emits a new `PortType.CHANNEL`
    payload.
  - **Rainbow ports** on every process node (enhancement / analysis): a `CHANNEL`-typed
    input on the left edge that accepts a channel wire, plus a `CHANNEL` output on the
    right ("both sides"). Wiring a channel **spawns a fresh free rainbow port** for the
    next channel (`NodeScene._sync_rainbow_ports`), so there is always one open.
  - **Automatic downstream propagation:** a channel wired into a node "follows" the
    standard (structural) wires downstream, and a channel wired into a *further* process
    joins the set there — `executor.channel_sets` (topological union of channel-edge
    sources + structural predecessors).
  - **Channel-colored wires:** a pure channel wire is drawn dashed in its channel color;
    every structural wire additionally shows one thin **channel-colored strand per channel
    flowing through it** (`executor.edge_channels`, `EdgeItem.set_channel_strands`).
  - **Per-channel processing "from the entry point":** the committed Processing recipe is
    now per-channel (`executor.channel_recipes`) — a channel gets only the enhancement
    steps at/after where it enters; **unwired channels stay raw**, in the committed
    processed view *and* the viewer (`EnhancedDataset` / `ProcessedFrameVolume` /
    `apply_recipe` gained a `recipe_by_channel` argument;
    `ND2StudiosRecord.recipe_by_channel`).
  - **Analysis on wired channels:** an analysis node segments the channel(s) wired into it
    (`_analysis_channels_for`), running once per channel and merging per-channel
    `label_masks` (new `_MultiChannelCommitJob` / multi-channel `_AnalysisPreviewJob` /
    `_ResultsScreenMeasureJob`), so the downstream `segmentation_channel` grouping used by
    tracking / results is preserved. A 2nd wired channel counterstains a
    counterstain-capable pipeline (tear detection).
  - **Scissors tool** in the control bar (by Preview): arm to cut any wire — channel or
    standard — by dragging a stroke across it or clicking it (`NodeScene.set_cut_mode`).

### Changed

- **Analysis node popups no longer show `channel_name` / `counterstain_channel`** — the
  channel is inferred from wiring (`registry_adapter.param_specs_for` strips them).
  `Cell-Tracker Metrics`'s `intensity_channel` is kept (it selects a *measurement column*,
  not a segmentation channel).
- **Backward compatibility:** a slice with **no** channel wiring behaves exactly as before
  (every process runs on all channels; analysis falls back to the old `channel_name` /
  first channel). Loading a pre-V1.48 graph migrates process nodes by adding rainbow ports
  (`NodeScene.ensure_rainbow_ports`) and shows the channel pills unwired, so the old graph
  keeps working until channels are wired.

## [Unreleased] - 2026-07-06 (SerialTrack PTV plots — V1.47, in progress)

### Bug Fixes

- **SerialTrack tracking aborted at frame 7, producing zero tracked objects** — root cause
  was a **missing scikit-learn** in the Python 3.12 environment. The POD-GPR ADMM warm
  start (`serialtrack/prediction.py::InitialGuessPredictor._predict_pod_gpr`, used for
  frames ≥7 when `Use previous results` is on) imports scikit-learn; without it the
  `ModuleNotFoundError` propagated out of the single `track_coordinates` call, so
  `object_tracker._link_group_serialtrack` raised (its `except ImportError` re-raised a
  clear `RuntimeError`) and **no `track_id` was written** — the run failed while the
  results table still showed the pre-tracking segmentation rows, and the SerialTrack tab
  correctly reported "no objects tracked" (log stopped at frame 6). Resolution:
  **scikit-learn 1.9.0 installed into Python 3.12**; the POD-GPR warm start now runs as
  intended and tracking completes across all frames (verified end-to-end: POD-GPR path
  active, no fallback). No code change — scikit-learn is a required dependency for the
  SerialTrack `Use previous results` path on 7+ frame sequences.

### Added

- **SerialTrack PTV analysis data layer** (`nd2studios/backend/serialtrack_analysis.py`,
  new, Qt-free). Rebuilds Particle-Tracking-Velocimetry analysis products from tracked
  measurement rows — because ND2Studios currently runs SerialTrack only to chain
  `track_id` and discards its displacement/strain fields. Public API:
  - `build_track_data(rows, m=None, *, pixel_size_um, z_step_um, time_step)` →
    `TrackData` (per-frame `coords`/`track_ids`, chained `trajectories` matrix
    `(N_tracks, n_frames, D)`). Dimensionality `D` auto-inferred (3 when rows carry
    `centroid_z_px`, else 2) — so 3D lights up for volumetric/imported centroids while
    today's 2D pipeline is unaffected.
  - `particle_displacement(td, t, mode)` — per-particle displacement vectors
    (`"cumulative"` vs a reference frame, or `"incremental"` frame-to-frame).
  - `compute_field_bundle(td, t, *, grid_step, smoothness, physical)` → `FieldBundle`
    with gridded `DisplacementField` + `StrainField` (via `serialtrack.fields.compute_gridded_strain`).
  - Derived fields: `displacement_magnitude`, `velocity_components`, `divergence`
    (dilatation), `curl` (2D scalar / 3D vorticity vector), `jacobian` (J = det(I+∂u/∂x)),
    `effective_strain` (Von-Mises strain), `compute_stress` (linear-isotropic Hooke's
    law) + `von_mises_from_sigma` (plane-strain in 2D), and a `scalar_field(fb, key)`
    dispatcher for the panel's Field combo.
  - Verified against an analytic 10 %-x-stretch deformation: `e_xx`/`div`/`det(F)`/`curl`
    match to regularization tolerance. Plan: `CodeLog/ClaudesPlan/V1.47_serialtrack_ptv_plots.md`.
- **SerialTrack PTV panel + viewer tab** (`nd2studios/widgets/serialtrack_panel.py`,
  new; wired in `nd2studios/pages/pipelines_page.py`). A new **"SerialTrack"** tab in
  the Pipelines image-viewer stack (sibling of the *Spatial Maps* tab), embedding
  `SerialTrackPanel` — a matplotlib (`MplCanvas`) PTV analysis view driven by a
  `FrameStrip` T-scrubber with play/pause + FPS. View modes:
  - **Trajectories** — particle paths up to the current frame, colored by time or net
    displacement, with XY / XZ / YZ projection for 3D data.
  - **Scalar field** — heatmap / filled-contour / line-contour of any of ~17 fields
    (displacement, velocity, strain εij, divergence, curl, det(F), effective strain,
    Von Mises + σij stress), over an optional background channel image, with colorbar,
    scale bar, per-field symmetric/divergent color scaling, and mid-Z slicing for 3D.
  - **Quiver** and **Heatmap + Quiver** — magnitude-colored displacement vector field
    (subsampled to an adjustable arrow density).
  - **Displacement histogram** (per-frame magnitude distribution, mean/median markers).
  - **Tracking dashboard** — detected/tracked bars + tracking-ratio line across frames.
  - Controls: mode (cumulative vs incremental), grid step, smoothness, colormap,
    stress material params (E, ν). Per-frame `FieldBundle` cache keeps playback smooth.
  - Wiring: overlay-tab key `"serialtrack"`, lazy `_ensure_serialtrack_panel()` /
    `_populate_serialtrack_panel()`, visible once tracked rows exist — mirrors the
    Spatial Maps tab exactly. Verified headless (offscreen Qt) across all 38 view/field
    combinations in 2D and 3D. (V1.47 phase 1.)

### Added

- **Preview-crop UX overhaul (Pipelines tab).** Several improvements to the
  preview-crop tool:
  - **Works with live Preview off.** Enabling/disabling the crop now refreshes
    the displayed base image even when the Preview toggle is off (`_on_crop_changed`
    → `_show_base_image(force=True)`), so you can set/clear a crop without live
    compute running. Arming the tool with Preview off also shows a base image to
    draw on if the viewer is empty.
  - **Undo arrow (↩) left of the Preview Crop button** (`_btn_crop_undo`) steps
    back through crop changes via an undo stack (`_crop_history`,
    `_on_crop_undo`, `_apply_crop_state`) — restores the previous crop, or
    no-crop. Reset on file change.
  - **More prominent crop rectangle.** The rubber-band rectangle now draws a dark
    halo under a bright magenta dashed line (both `ImageCanvas` and
    `GpuImageCanvas`) so it stands out over any image content.
  - **Live crop preview in the dialog.** The Preview Crop dialog shows a
    real-time thumbnail of the cropped region (`_composite_full_frame_rgb` →
    scaled `QLabel`) that updates as you edit X/Y/W/H or jog.
  - **Jog pad.** A symmetric 4-way cross of chevron-icon buttons (qtawesome, so
    they match the rest of the toolbar) around a configurable step (px) field in
    the middle; each arrow shifts the whole region by the step in that direction,
    clamped to the image. Equal-sized grid cells + a content-hugging group box
    keep the cross exactly symmetric, and the step field is sized wide enough
    (`scaled(58)`) that its value is fully visible (a too-narrow box previously
    hid the number and made the pad look broken); jog buttons are
    `autoDefault=False` so Enter accepts the dialog. Verified end-to-end (clicking
    ▶ shifts the region and updates the live preview).
    (`nd2studios/pages/pipelines_page.py`,
    `nd2studios/widgets/image_viewer.py`, `nd2studios/widgets/gpu_image_canvas.py`)

  A **skill** documenting ND2Studios button/icon/widget-layout conventions (icon
  helpers, `scaled()` DPI, QSS object names, symmetric grid pads, dialog layout,
  headless verification) is at `.claude/skills/nd2studios-widget-layout/SKILL.md`
  so future UI additions follow the correct pattern.

- **ADMM global solver routed into the SerialTrack tracking method**, plus the
  method's **complete tunable-parameter surface** on the Track Objects node. When
  *SerialTrack (topology PTV)* is the method, the param popup now exposes (all via
  `ParamSpec.visible_when`, hidden for the other methods):
  - `st_solver` — **Global solver** choice: **MLS** / **Regularization** (default)
    / **ADMM** (augmented-Lagrangian with automatic L-curve α — the paper's most
    faithful, costlier solver).
  - `st_loc_solver` — **Local matcher**: *Topology* / *Histogram then Topology*.
  - `st_n_neighbors` (Max neighbors) + `st_n_neighbors_min` (Min neighbors) — the
    topology-descriptor neighbor-count range (`n_neighbors_max` / `n_neighbors_min`).
  - `st_smoothness` — global smoothing strength (the α/µ knob).
  - `st_outlier_threshold` — Westerweel normalized-median-residual cutoff (0 = off).
  - `st_max_iter` + `st_iter_stop_threshold` — ADMM iteration budget + convergence
    tolerance.
  - `st_dist_missing` — ghost-particle cull distance ε_d.
  - `st_use_prev_results` — data-driven warm start (extrapolation, then POD-GPR
    from frame 7; the POD-GPR stage needs scikit-learn).

### Changed

- `backend/object_tracker.py`: `link_objects` / `link_objects_with_params` /
  `_link_group_serialtrack` gain the full `st_*` parameter set and map the GUI
  strings to SerialTrack's `GlobalSolver` / `LocalSolver` enums. `use_prev_results`
  raises a friendly "install scikit-learn" `RuntimeError` only if it actually
  reaches the POD-GPR stage (≥7 frames) without sklearn; short sequences work
  regardless. scikit-learn stays an **optional** extra (not in `requirements.txt`),
  unlike numba which the linking path always needs.

## [Unreleased] - 2026-07-03 (Pipeline Run: diagnose "runs but no results")

### Added

- **Preview Crop — restrict the pipeline preview to an XY sub-region.** A new
  checkable **Preview Crop** button in the Pipelines-page image-viewer overlay-tab
  row (`_btn_preview_crop`). Toggle it on, then either drag a rectangle on the
  image or click a point to enter the top-left corner + width/height in a dialog
  (`_show_preview_crop_dialog`, mirroring the Recipe-page crop dialog). Preview
  mode — Processing, Analysis and the Results walk — then runs on the *selected
  frames × the crop* instead of the whole frame, and the preview viewer shows the
  cropped region so segmentation masks, tracks, vectors, measurements and plots
  all live in one consistent crop-space coordinate system. Toggling the button
  off clears the crop; a full **Run** always processes the whole frame (crop is
  ignored while running) and the crop is not persisted to `.nd2s` (it resets when
  the active file changes). Implemented as a single choke-point crop applied in
  the preview frame readers (`_read_planes_frames`, `_extract_processed_frame` via
  `_read_processed_planes`, `_materialize_channels_for_m`) plus a new
  `CroppedVolume` (`nd2studios/pipeline_graph/executor.py`) wrapping the lazy base
  volume for display; `_field_shape_for` and preview job metadata report the crop
  dims, and `_overlay_result_for` / `_label_stack_for_m` skip the full-frame Run
  masks while a crop is active. New page helpers: `_crop_rect`, `_crop_frame`,
  `_maybe_crop_volume`, `_preview_metadata`, `_raw_frame_shape`, `_on_crop_changed`.
  (`nd2studios/pages/pipelines_page.py`, `nd2studios/pipeline_graph/executor.py`)

- **Preview mode now builds the bottom-panel plots.** Cells/frame, Area,
  Tracks/frame and Track length were previously generated only by a full Run.
  A new `_update_preview_plots()` rebuilds them from the previewed measurement
  rows (scoped to the selected frames + crop) after each Results preview walk, so
  the plots track the preview like the overlays and measurements table already
  did. (`nd2studios/pages/pipelines_page.py`)

### Changed

- **Track-persistence condition block is now a single per-track min/max test.**
  The `track_persistence` if-else block previously exposed *two* thresholds
  (`min_frames` **and** a `comparator`+`value` count of tracks) — a whole-frame
  population test that was meaningless in the per-track lens. It now has just
  **`mode`** (`At least` / `At most`) + **`frames`**: keep a track when it spans
  ≥ / ≤ N frames. Evaluated per track via a new `_track_frame_counts` (frame
  appearances per `track_id`, independent of the `track_length` column), so it
  works even when Cell-Tracker metrics never ran. The block is flagged
  `object_lens_only` and is offered only for an **Each object / Per track**
  if-else — the `ConditionBuilderDialog` now takes a `lens` kwarg and hides
  object-lens-only blocks from a whole-frame if-else's Add menu
  (`is_object_lens_only`). Old saved blocks migrate silently to "at least 2
  frames". Dead `_track_lengths` helper removed.
  (`nd2studios/pipeline_graph/conditions.py`,
  `nd2studios/widgets/node_board/condition_builder_dialog.py`,
  `nd2studios/pages/pipelines_page.py`)

- **Spatial maps: restore the original Cell-Tracker construction (griddata).**
  The spatial-field construction was reverted from the V1.46 binned
  Gaussian-weighted local-mean + cell-footprint method back to the original
  Cell-Tracker repository's method: density is a Gaussian-smoothed count of cells
  per grid bin, and every value-bearing field (`mean_area`, `intensity`,
  `fold_change`, `self_fold`, `velocity_*`, `speed`, `divergence`, `curl`, and
  arbitrary `col:` measurement columns) is a `scipy.interpolate.griddata` **linear
  interpolation** of the per-cell values over the grid — gaps filled with the
  frame mean (`nan_to_num`; `nan=1.0` for `self_fold`, `0` for velocity), then
  `gaussian_filter`-smoothed. Scalar fields need ≥4 cells; velocity needs >3 cells
  tracked into the previous frame; `divergence`/`curl` are `np.gradient` of the
  velocity grid. Data collection (Cell-Tracker tracking + StarDist segmentation)
  and every spatial-map functionality (grid step / σ, colour scale, all field
  types, cell-mask source, overlay, borders, quiver, scale bar, templates,
  zoom/pan, playback, in-tab Save) are unchanged. `fields.compute_spatial_fields`
  rewritten (binned helpers `_bin_sum_count` / `cell_footprint` /
  `_binned_mean_field` removed); `SpatialMapsPanel._bin_value_field` replaced by
  `_interp_value_field` (griddata, nearest fallback <4 cells) driving
  `_compute_self_fold_field` / `_interp_column`. The export bridge
  (`celltracker_bridge`) inherits the change via `compute_spatial_fields`.
  (`nd2studios/backend/celltracker/fields.py`,
  `nd2studios/widgets/spatial_maps_panel.py`)

- **The Run button is a full/cropped dropdown when a preview crop is active.**
  With no crop, Run behaves as before (one click → full-file Run). When a
  preview crop is set, the Run button shows a ▾ affordance and clicking it drops
  a menu with **Run full (uncropped) file** and **Run cropped region (w×h @ x,y)**.
  A cropped Run scopes the whole pipeline — analysis, measurement, overlays,
  plots and the run viewer — to the crop: `_crop_rect()` now also returns the
  crop while `_run_active` when the Run was launched cropped (`_run_cropped`),
  `_run_analysis_node` slices each channel via `_crop_channel_for_run` (lazy
  slice when the source supports it, else materialize-then-crop) and uses
  `_preview_metadata` for crop-sized dims. Overlay / Spatial-Maps mask reuse is
  now gated on a geometry match (`_run_results_crop` — the crop the committed
  masks were computed at — vs the current display crop) so crop-sized Run masks
  paint on a cropped display and full-frame masks paint on the full display, but
  never the mismatched combination. New: `_on_run_button`, `_show_run_menu`,
  `_start_run`, `_update_run_button`, `_crop_channel_for_run`.
  (`nd2studios/pages/pipelines_page.py`)

- **Double-clicking a node in the Analysis tab restarts the preview cleanly.**
  Double-click still promotes a node to the previewed (golden) node, but it now
  first cancels any preview analysis still computing and resets its scratch —
  per-plane overlay results (`_analysis_screen_results`), measurement rows,
  walk shading, plots and the progress bar — via the new
  `_reset_analysis_preview()`. So a slow analysis (e.g. StarDist) started for one
  node can't land stale on, or mix with, the node you just double-clicked. A full
  Run is untouched (it owns the runner + viewer), as are committed Apply/Run
  results. (`nd2studios/pages/pipelines_page.py`)

- **StarDist Segmentation runs multiple frames at once on multi-core CPUs.**
  StarDist/TensorFlow share a single, non-thread-safe model (TF pinned to one
  op-thread), so the node ran strictly one frame at a time — leaving a many-core
  workstation mostly idle. It now fans frames across separate **worker
  processes**, each with its own model + TF, via the new
  `run_stardist_multiprocess` (`nd2studios/backend/analysis/mp_stardist.py`,
  `spawn` start method). The parent reads frames from the lazy source one at a
  time and ships them to workers as ndarrays (the lazy reader is never opened in
  a child; the parent stays the single writer of the streaming/in-RAM label
  sink), and a bounded ~2×workers in-flight window keeps parent RAM bounded even
  for huge stacks. A new `n_processes` param (0 = auto) controls the count;
  auto is gated by `recommended_process_count()` (new in
  `nd2studios/utils/resources.py`) = min(physical cores, free-RAM ÷ ~2 GB/worker,
  8). The GPU flag forces sequential (MP just contends over one card), and
  `n_workers <= 1` / tiny stacks keep the existing thread path unchanged. Output
  (label masks + measurements) is identical to the sequential path — only where
  the per-frame work runs changes. Note: on native Windows, TF ≥ 2.11 has no GPU
  support, so StarDist is CPU-bound there and this is the primary throughput
  lever. (`nd2studios/backend/analysis/stardist_segmentation.py`)

- **Launcher auto-selects a capable Python interpreter (`run.py`).** On this
  machine the `py` launcher defaults to Python 3.14, which has no TensorFlow
  wheel — so `py run.py` launched the GUI on an interpreter where the StarDist
  Segmentation node can't work (raises "StarDist is not installed"), while a
  fully-provisioned Python 3.12 sat unused. `run.py` now probes the launching
  interpreter for the required stack (`PySide6`, `numpy`, `nd2`, `cv2`) and the
  preferred optional stack (`tensorflow`, `stardist`, `csbdeep`) via `find_spec`;
  if another installed interpreter (discovered from `$ND2STUDIOS_PYTHON`, the
  `py -0p` registry, or `PATH`) scores strictly higher, it re-launches itself
  there via `subprocess`, guarded against re-exec loops by an `ND2STUDIOS_REEXEC`
  env flag. When the current interpreter already has the full stack it runs in
  place with no probing (fast path). Set `ND2STUDIOS_PYTHON=<python.exe>` to force
  a specific interpreter. (`run.py`)

- **Tracking diagnostics.** `object_tracker.link_objects` and the vendored
  `celltracker.tracking` linkers now log (via `logging`) the detection count,
  frame count, mean detections/frame, and method on entry, and per-stage timing
  on exit — for the topology tracker, **link vs. topology-feature vs.
  final-assignment** seconds are reported separately so a slow run is
  attributable rather than mysterious. When the largest per-frame cost matrix
  exceeds 4 M entries, a one-time warning names the O(n³) Hungarian assignment as
  the inherent cost of a dense field. (`nd2studios/backend/celltracker/tracking.py`,
  `nd2studios/backend/object_tracker.py`)

### Changed

- **Faster Cell-Tracker topology / fingerprint tracking (result-identical).**
  The final `track_id` assignment is now a vectorized dict lookup over zipped
  numpy arrays instead of `df.apply(..., axis=1)` (which built a Series per row —
  tens of seconds at 100k+ detections); per-frame sub-frames are taken from a
  single `df.groupby("frame")` instead of a repeated boolean mask (was
  O(T² · cells)); and `compute_topology_features` is fully vectorized (the
  per-cell Python loop became array ops, guarded so the single-neighbor case
  stays bit-for-bit identical to the old loop — covered by
  `tests/test_tracking_topology.py`). Tracking output is unchanged; only the
  wall-clock improves. (`nd2studios/backend/celltracker/tracking.py`)

### Bug Fixes

- **Dismiss node never worked as a discard — now a true terminal discard.** The
  Dismiss node keyed off `track_validation == "rejected"`, a tag no row ever
  carries (Review/Validate *drops* rejected rows and only tags *accepted* ones).
  As a result a Run always fell to the else-branch and silently deleted the whole
  branch, while the preview path did nothing — so preview and Run disagreed and
  per-object if-else → Dismiss flows behaved unpredictably. Dismiss is now defined
  as a **terminal discard**: a new `_discard_objects(drop_rows)` erases every
  object routed to the node — its `label_id` pixels are zeroed out of the
  committed `label_masks[seg_channel][frame]`, and its rows are dropped from
  `_run_context["rows"]`, `_results_rows`, and the frozen `_track_overlay_rows` —
  then the track colormap is rebuilt and the table + viewer refreshed. The object,
  its track id and all its frames therefore vanish from every viewer tab and
  never reach a downstream node. Both `_run_dismiss` and the preview
  (`_preview_execute_walk_node`) call the same helper so the two paths agree.
  (`nd2studios/pages/pipelines_page.py`)

- **Preview plots and overlay tabs didn't appear after a preview (Analysis tab,
  results mode).** A preview computed fine but the bottom-panel plots and the
  Segmentation / Tracks / Vectors overlay tabs stayed hidden — silently, with no
  crop needed. The async preview-completion path (`_PV_SCREEN_KEY` handler) set
  `_analysis_screen_results` / `_results_rows` but never re-ran
  `_update_overlay_tabs_available()`, so tab visibility — computed at
  node-promotion time, *before* the async result exists — was never refreshed
  once results landed (the Run path calls it in `_advance_run_m`). Separately, the
  preview Track Objects walk (`_preview_execute_walk_node`) linked `track_id`s but
  never built `_track_colormap` / `_track_overlay_rows`, so `has_tracks` stayed
  False and the Tracks / Vectors tabs never revealed in preview (the Run built
  this in `_build_track_overlay`). **Fix:** the Run's track-overlay setup is
  factored into a shared `_set_track_overlay_state(rows)` now called by the
  preview walk too; the `_PV_SCREEN_KEY` handler resets stale track scratch when
  fresh results arrive and calls `_update_overlay_tabs_available()` after the walk
  + preview plots; `_reset_analysis_preview` clears the track scratch as well.
  Diagnostic logging was added (result/row counts in the handler, no-op reason in
  `_update_preview_plots`). (`nd2studios/pages/pipelines_page.py`)

- **The Track Objects node showed no progress and ran slowly (Cell-Tracker
  methods).** The tracking bar sat at 0 % until it jumped to done, and dense
  fields (thousands of StarDist nuclei) took a long time with no feedback. Two
  causes: (1) `_TrackJob.run` (`pages/pipelines_page.py`) reported only `0.0`
  then `1.0` around one opaque blocking call — the `progress_cb` that
  `track_timeseries` already accepted was never threaded through
  `link_objects_with_params → link_objects → _link_group_*`; and (2) the
  vendored trackers used a per-row `df.apply(..., axis=1)` for the final
  `track_id` column and a re-scanned `df[df["frame"] == f]` boolean mask on every
  iteration (O(T² · cells)). **Fix (works across all four methods — centroid,
  SerialTrack, Cell-Tracker topology, Cell-Tracker fingerprint):** a backend-pure
  `progress_cb(fraction_0_1, message)` is now threaded from `_TrackJob` down
  through `object_tracker.link_objects` (which splits the 0–1 range across
  `(channel, m_position)` groups) into each linker, which reports **per frame**
  (e.g. "Linking frame 42/120"). The GUI callback also polls the cancel token, so
  a long track is now cancellable mid-run instead of freezing the Stop button.
  (`nd2studios/backend/object_tracker.py`, `nd2studios/backend/celltracker/tracking.py`,
  `nd2studios/pages/pipelines_page.py`)

- **Analysis tab showed a stale committed recipe (e.g. a Background Subtract)
  even when the Processing node graph was empty.** On the Pipelines page the
  Processing node graph and the committed `record.recipe` were independent
  sources of truth: the Processing preview linearizes the *graph* (empty graph →
  raw), but the Analysis base image and every per-frame processed read use
  `record.recipe` / `record.recipe_normalized` directly (`_show_base_image` →
  `ProcessedFrameVolume`; `_extract_processed_frame` → `apply_recipe`).
  `record.recipe` is written only by an explicit Apply, the legacy Recipe page,
  config load/save, or session load — never by editing the node graph, and an
  empty graph never cleared it. So a recipe committed through any of those routes
  stayed live on the Analysis tab (and in exports) with nothing shown in the
  graph — the symptom being a background subtract "applied" with no node wired to
  the Output. The node graph is now **authoritative**: leaving the Processing
  sub-tab re-derives the committed recipe from the graph's primary output chain
  via new `_graph_recipe()` / `_sync_committed_recipe_from_graph()` (called from
  `_select_stage`), so an empty / disconnected graph clears the stale recipe and
  resets `record._processed_view`. Explicit Apply is unchanged (it still performs
  the durable session-workspace commit); the sync is a cheap in-memory
  reconciliation guarded to no-op when the effective recipe is unchanged.
  (`nd2studios/pages/pipelines_page.py`)

- **A pipeline Run could silently produce nothing.** When an analysis node's
  preconditions weren't met, `_run_analysis_node` (`nd2studios/pages/pipelines_page.py`)
  bailed out through one of four guard clauses — no active record, unresolvable
  action, unregistered pipeline, or unreadable/empty processed channels — each
  calling `_run_finish_node(node.id)` with **no status message and no log line**.
  The node animated shaded→gold→done, downstream nodes reported "no measured
  objects — skipped", and the run finished blank with zero clue as to why (the
  tell-tale `Run: '<pipeline>' done … → N objects.` line never appeared, since the
  per-M loop was never entered). Likewise, a **failed** analysis compute job for a
  multipoint was swallowed in the `_RUN_ANALYSIS_KEY` handler with a bare
  `_advance_run_m()`. Both paths now emit a specific reason to the status bar and
  the log via a new `_skip_analysis_node(node, reason)` helper (reasons: no file
  imported / node has no analysis action / no channels available / pipeline
  `<name>` not registered / could not read processed channels `<exc>` / processed
  view has no channels) and, for a job failure, `Run: '<pipeline>' failed on M<n>:
  <error>`. Diagnostics only — a working run is unchanged.
  (`nd2studios/pages/pipelines_page.py`)

## [Unreleased] - 2026-06-29 (Spatial maps: every field built like cell density)

### Bug Fixes

- **GUI froze ("not responding") at the segmentation→tracking transition.** The
  object linker ran on the **GUI thread** in two places during a Run: the per-M
  default tracking right after each multipoint's measurement
  (`_on_runner_result`/`_RUN_MEASURE_KEY`) and the Track Objects node
  (`_run_track_objects`). The Hungarian (and SerialTrack / Cell-Tracker) linkers
  cost grows with object count — seconds on a dense field of thousands of nuclei,
  minutes for SerialTrack — so the Qt event loop blocked and Windows showed the
  window as unresponsive. Both now run on a worker via a new `_TrackJob`
  (`AnalysisJob`) submitted to the `JobRunner`, with the run state machine
  resuming on the job result (`_RUN_TRACK_KEY` → accumulate + advance the M-loop;
  `_RUN_TRACKOBJ_KEY` → `_finish_track_objects` publishes rows, refreshes the
  table/plots/overlay, completes the node). The node gates the walk with
  `_run_pending` like the async analysis node; cancellation/progress key lists
  include the new keys. Measurement was already off-thread (`_ResultsMeasureJob`);
  tracking now matches. (`nd2studios/pages/pipelines_page.py`)

- **Segmentation ran ~2× slower than CellTracker's original (StarDist / Cellpose
  nodes).** The per-frame post-processing was `O(n_objects × pixels)`: both
  segmentation nodes applied the area filter with `mask[mask == region.label] = 0`
  per out-of-range object and then `_relabel_contiguous` with
  `out[mask == old_id] = new_id` per object — each a full-frame scan **per
  object** (~50 s/frame at 4096²). CellTracker's `segment_timeseries` has no such
  loop. Replaced by a shared `source_utils.filter_and_relabel`, which does the area
  filter **and** contiguous relabel in a single `O(pixels)` pass (`np.bincount` + a
  lookup table). Verified bit-identical to the old result; ~430× faster on a 2048²
  frame with ~6 k objects (12.8 s → 0.03 s).
  (`nd2studios/backend/analysis/{stardist,nuclei}_segmentation.py`,
  `nd2studios/backend/analysis/source_utils.py`)
  - **TF threading stays single inter/intra-op (CellTracker's way).** An interim
    attempt to "restore multi-threaded TF" was reverted — CellTracker's repository
    *explicitly* pins TF to one inter/intra-op thread, and multi-threaded TF is
    ~10× **slower** for StarDist here (its tiled `predict_instances` oversubscribes
    the cores on many small ops). `_ensure_tf_threading` keeps the cap; only the
    ND2Studios-specific GPU-flag handling (clearing `CUDA_VISIBLE_DEVICES` before
    the first TF import) is layered on top.
    (`nd2studios/backend/celltracker/segmentation.py`)

- **Measurement (and therefore tracking) stalled for minutes on large, dense
  frames.** `compute_measurements` extracted each object's per-channel intensity
  with a full-frame boolean scan, `frame_data[mask_frame == prop.label]`, run once
  *per object per channel* — O(n_objects × frame_pixels). On a 4096² frame with a
  few thousand nuclei this was ~50 s **per frame** (~8–9 min for a 10-frame stack),
  so on big files the measurement step that feeds tracking appeared to hang and no
  tracks were produced. Intensity stats are now computed once per channel in a
  single labelled pass (`scipy.ndimage.mean` / `standard_deviation` over the label
  image) and looked up per object — ~6 s/frame on the same data (≈8× faster),
  bit-identical values. (`nd2studios/backend/results_engine.py:compute_measurements`)

- **ImageJ TIFF channel name corrupted to its first letter.** When an ImageJ
  TIFF carries a single slice label, `tifffile` returns `Labels` as a bare
  string (e.g. `"H2B-iRFP670"`) rather than a list. `read_imagej_tiff_metadata`
  then sliced it as if it were a per-channel list (`labels[:n_c]` →
  `"H2B-iRFP670"[:1]` → `"H"`), so the channel loaded under a one-letter name.
  It now wraps a string label in a one-element list before slicing and coerces
  the result to `str`, so the full channel name survives.
  (`nd2studios/backend/tiff_loader.py:read_imagej_tiff_metadata`)

- **Every Cell-Tracker spatial-map field except cell density was computed wrong.**
  Density was correct — it bins each recorded cell's centroid into a grid cell
  (`density[int(y/step), int(x/step)] += 1`) and Gaussian-smooths, so the value at
  each node is built from the cells that are actually there. Every other field
  (`mean_area`, `intensity`, `fold_change`, `self_fold`, `velocity_y/x`, `speed`,
  `divergence`, `curl`, and arbitrary measurement columns) instead used
  `scipy.interpolate.griddata`, which triangulates the centroids, interpolates a
  value across the whole grid, and pads everything outside the convex hull with the
  single global frame mean (`nan_to_num(..., nan=np.nanmean(values))`) — a blanket
  field that invents values in empty space and washes out local structure.
  All value-bearing fields now generalise the density method to a
  **Gaussian-weighted local mean**: bin the per-cell value-sum and the cell count
  into grid bins, smooth both with the same Gaussian, divide. Density is exactly
  the denominator of that ratio, so the methods are now consistent. Velocity also
  no longer needs ≥4 tracked cells (was a griddata triangulation floor; binning
  works from one).
- **Spatial-map value fields spread into empty space and bridged the gaps between
  cells.** A normalised local mean (`sum / count`) does **not** fade away from
  cells the way the density *count* does — numerator and denominator decay together
  under the Gaussian, so the ratio holds the cell's value flat across the whole
  smoothing kernel (≈ `4·sigma` ≈ 8 grid bins ≈ 160 px by default) and fills the
  gaps between neighbours. So `mean_area`, `self_fold`, `speed`, `velocity_*`,
  `divergence`, `curl`, and arbitrary measurement columns showed values in regions
  with no objects (density / intensity / fold-change looked fine because a count
  self-masks and the channel-intensity path reflects the dark background). Every
  value field is now restricted to the **cell footprint** — the grid bins that
  actually contain an object, dilated by one bin for sub-grid coverage — via a new
  `cell_footprint` helper; outside it the field is `NaN` (transparent). Velocity is
  binned with `fill=0.0` (so its divergence / curl gradients stay finite) and then
  masked to its own footprint (only cells tracked into the previous frame).
  Density is deliberately left unmasked — it is a count that vanishes in empty
  space, so it masks itself. New helpers `_bin_sum_count` / `cell_footprint` /
  `_binned_mean_field` in `backend/celltracker/fields.py` (`compute_spatial_fields`
  rewritten to use them); the in-tab `self_fold` and arbitrary-column maps route
  through a shared `SpatialMapsPanel._bin_value_field`
  (`_compute_self_fold_field` / `_interp_column`,
  `nd2studios/widgets/spatial_maps_panel.py`). The gridded velocity overlay
  (`_paint_vectors_overlay`) drops the now-`NaN` dead space to 0 before drawing.
  This fixes both the in-viewer Spatial Maps tab and the node export path
  (`compute_field_for_frame`).

- **Esc closed the maximized viewer / plots window instead of docking it back.**
  Esc on a `QDialog` calls `reject()` → `hide()`, which never fires `closeEvent`,
  so the borrowed widget stayed inside the hidden dialog and the panel vanished.
  `PopOutWindow.reject()` now hands the widget back first (like `closeEvent`).
  (`nd2studios/widgets/popout_window.py`)

### Changed

- **Spatial Maps tab gains zoom / pan and a pinned intensity bar.** The map canvas
  now has the image-viewer-style **Home / + / − / Pan** controls (plus scroll-to-
  zoom and drag-to-pan) driving matplotlib data limits; the zoom view persists
  across frame changes and redraws so you can inspect a region while scrubbing.
  The colour (intensity) bar is drawn on a **fixed axes pinned to the figure's
  right edge**, so it stays attached to the viewer's right border at any zoom
  level. (`nd2studios/widgets/spatial_maps_panel.py`: `_build_zoom_toolbar`,
  `_zoom_about`, `_on_scroll`, `_on_canvas_press/motion/release`, `_apply_view`)
- **Spatial Maps tab uses the ND2Studios frame controls and is much faster to
  scrub.** The frame navigator is now the NIS-style `FrameStrip` + play/pause +
  FPS set (the same `MultiAxisViewer._make_axis_row` controls) instead of a plain
  Qt slider. Scrubbing is debounced (a fast drag computes/redraws only the final
  frame), a per-frame field cache (keyed by frame + grid/sigma/channel/source)
  makes revisits and playback instant, and the tab opens with a single-frame
  colour-scale fit rather than scanning every frame (the all-frames pass is now
  only the explicit "Global Scale" button). (`nd2studios/widgets/spatial_maps_panel.py`)
- **Spatial-map nodes now open an interactive in-viewer "Spatial Maps" tab.** Both
  **Cell-Tracker Spatial Maps** (`special:ct_fields`) and **Interpolated Spatial
  Maps** (`special:interp_map`) dropped their per-node parameters (field /
  intensity channel / grid / sigma / colormap / format / value column / …) and on
  preview *or* Run now bring up a new **Spatial Maps** overlay tab that faithfully
  reproduces Cell-Tracker's spatial page — full control sidebar (Data Source,
  Field, Grid, Scale, Display, Scale Bar), background-image overlay, cell-mask
  mode, cell borders, quiver, scale bar, auto/global colour scaling, frame
  scrubbing, plus interpolation of *any* numeric measurement column. The tab uses
  a **top dropdown bar** of section controls when the viewer is docked and the
  **full vertical Cell-Tracker sidebar** when the viewer is maximized (the
  `PopOutWindow` toggles `SpatialMapsPanel.set_compact`). An in-tab **Save** button
  (current frame / all frames, PNG or raw-field TIFF) replaces the old
  folder-export-on-Run. New widget `nd2studios/widgets/spatial_maps_panel.py`
  (`SpatialMapsPanel`, `SpatialTemplatePicker`); new helper
  `celltracker_bridge.build_tracked_df`; the viewer is now a `QStackedWidget`
  (image viewer ↔ panel) in `pages/pipelines_page.py`
  (`_open_spatial_maps_tab`, `_populate_spatial_panel`, `_ensure_spatial_panel`,
  `_label_stack_for_m`). Removed `_preview_field_maps` / `_preview_interp_maps` /
  `_run_ct_fields` / `_run_interp_maps` and the unused `export_field_maps` /
  `export_interp_maps` imports.
- **Spatial-map nodes carry reusable saved templates.** A Spatial Maps
  configuration can be saved from the tab ("Save template…") to a global JSON
  library (`~/.nd2studios/spatial_map_templates`, new backend
  `nd2studios/backend/spatial_templates.py`: `list/load/save/delete_template`,
  `import_file`, `export_file`). A node stores one or more template **names**
  (edited via the popup's **"Spatial map templates…"** button → new
  `SpatialTemplatePicker`); when the node runs they auto-load into the tab (first
  applied, the rest selectable as presets). For portability to another machine the
  picker/tab can **import a template JSON file** into the local library so the
  names resolve. (`nd2studios/pipeline_graph/registry_adapter.py`: both ops now
  expose only a hidden `templates` list param.)
- **Motion-vector overlays are now outgoing and object-anchored.** Per-cell arrows
  (Vectors: cells) put the **tail on the object's centroid** and the **head at the
  same track's position on the next frame** (t → t+1) on every frame (previously
  the arrow ran prev → current). The gridded velocity field (Vectors: field) now
  shows the same outgoing motion and is **masked to the regions around objects**
  (grid nodes farther than ~1.5 grid steps from any centroid are dropped) so no
  arrows appear in empty space. The last frame (no t+1) draws nothing.
  (`nd2studios/pages/pipelines_page.py`: `_paint_vectors_overlay`, new
  `_frame_centroids`)

## [Unreleased] - 2026-06-25 (Viewer overlays, pop-out restore, condition-block close)

### Bug Fixes

- **Image overlays not appearing on the Pipelines viewer.** On the GPU image
  canvas, an overlay is shown through the *composite* layer while the per-channel
  layers are hidden. `GpuImageCanvas.set_image` only hid those layers on the first
  mode transition, and `set_channel_visible` could re-show a base channel **over**
  the overlay while the canvas was in composite mode — leaving the base image
  visible and the overlay covered. `set_image`/`set_grayscale` now hide all
  channel layers every time, and `set_channel_visible` only reflects on-screen in
  per-channel mode (the visibility flag is reapplied when `update_channel`
  switches back). (`nd2studios/widgets/gpu_image_canvas.py`)
- **Overlay drawn twice on the first render of each frame.** The
  `MultiAxisViewer._do_refresh` live-compose fallback re-applied the frame
  post-process even though `_compose_current_frame` already applies it (and fills
  the pp caches), double-drawing the overlay. Removed the redundant second pass.
  (`nd2studios/widgets/multi_axis_viewer.py`)
- **Maximized viewer / plots window closed instead of docking back.** If the
  host's restore callback raised, the borrowed widget was left orphaned
  (parent `None`, in no layout) so the panel vanished. `PopOutWindow._hand_back`
  now runs the restore in a `try/finally`, and the Pipelines `_restore` forces the
  re-inserted panel visible and gives the splitter real sizes so it can't be
  collapsed to 0 px. (`nd2studios/widgets/popout_window.py`,
  `nd2studios/pages/pipelines_page.py`)

- **Vector overlays (Vectors: cells / field) blank after a Run.** Two causes:
  (1) the track reveal lands the viewer on **frame 0**, where the backward-looking
  vector overlays have no previous frame to draw from (segmentation / tracks are
  frame-local, so they still showed); (2) a downstream **Dismiss** empties
  `_results_rows`, which the vector / track overlays read. Fixes
  (`nd2studios/pages/pipelines_page.py`): the tracks / vectors overlays now read a
  **stable tracked-rows snapshot** (`_track_overlay_rows`, frozen when Track
  Objects runs) via `_overlay_rows()`, so downstream filtering no longer blanks
  them; and at a track's first frame the per-cell arrows and the velocity field
  look **forward** (t → t+1) so motion is shown at frame 0.

### Changed

- **If / Else condition builder: clear per-block close button.** Each condition
  block row now has an always-visible **✕** button (a text button, replacing the
  icon-font trash glyph that could fail to render) to remove that block.
  (`nd2studios/widgets/node_board/condition_builder_dialog.py`)

## [Unreleased] - 2026-06-24 (If / Else — object lens: per-object/per-track branching)

### Added

- **Object-lens branching for the If / Else node.** A new `lens` param toggles
  between **Whole frame** (current aggregate gate — routes the entire result down
  one branch) and **Each object** (evaluates the condition per object, sending
  passing objects to the TRUE branch and failing ones to FALSE — *both* branches
  run). A second `group_by` param (visible in object lens) chooses **Per track**
  (default — a whole tracked cell routes as a unit, e.g. "any frame-frame vector
  > 20 µm" excludes the entire cell; untracked objects fall back to per-row) or
  **Per row** (each object-frame routed independently).
  - This makes a downstream **Dismiss** drop only the failing objects (not the
    whole frame), and a downstream Spatial Maps / Export see only the kept ones.
  - `pipeline_graph/conditions.py`: `partition_rows(cond, rows, group_by)` splits
    rows into `(pass, fail)` by reusing `evaluate_condition` per group (so a
    metric block's any/all/mean aggregate means "across the cell's frames");
    `_partition_groups` groups by `(m_position, track_id)`. New constants
    `LENS_FRAME` / `LENS_OBJECT` / `GROUP_TRACK` / `GROUP_ROW`.
  - `pages/pipelines_page.py`: per-branch **row scoping** in the Run walk —
    `_apply_branch_scope` scopes each node to the rows carried on its live
    incoming branch port; `_run_finish_node(…, port_rows=…)` records per-output-
    port rows; an object-lens if-else finishes with no pruning and routes
    `{true: pass, false: fail}`. The preview walk mirrors this for the previewed
    branch. `GraphRunner.active_incoming(nid)` added (executor.py).
  - Note: in **Per row** mode a row with no measured velocity (a track's first
    frame) fails a speed condition (missing value → block false); **Per track**
    is the sensible default for motion conditions since a track always has a
    measured frame-frame vector.
- **µm/frame motion columns from Cell-Tracker Metrics.** `augment_rows_with_metrics`
  now also emits `speed_um`, `velocity_y_um`, `velocity_x_um` (= px columns ×
  `pixel_size_um`) so an If / Else can threshold motion directly in µm. Added to
  `METRIC_COLUMNS` and `TRACKING_METRICS`.

### Changed

- **If / Else node param popup** now shows the `lens` / `group_by` choices
  alongside the existing **Edit branch condition…** button. Old saved graphs
  (no `lens` param) default to **Whole frame**, i.e. unchanged behavior.

## [Unreleased] - 2026-06-23 (Interpolated Spatial Maps node + Results-category spatial maps)

### Added

- **Interpolated Spatial Maps node (Pipelines → Results).** A new generic
  spatial-map node that linearly interpolates **any** numeric measurement column
  (e.g. `area_px`, `mean_intensity_<channel>`, `speed`, or any Cell-Tracker
  Metrics column), sampled at each object's centroid, onto a regular grid — one
  heatmap per `(M, frame)` — and writes PNG / TIFF files. Optionally performs a
  per-track **temporal fill** first (linearly interpolating each track's value
  across the frames it spans to fill missing samples), so it pairs with an
  upstream Track Objects node.
  - New backend module `backend/interp_maps.py`:
    - `temporal_fill(rows, value_col)` — per-`(m_position, segmentation_channel,
      track_id)` `np.interp` of the value column over frames (no extrapolation;
      untracked rows untouched).
    - `compute_interp_map(rows, m, frame, field_shape, value_col, grid_step,
      sigma, method)` — `scipy.interpolate.griddata` of the centroid samples onto
      a `(H//grid_step, W//grid_step)` grid + `gaussian_filter`; falls back to
      `nearest` for fewer than four objects.
    - `export_interp_maps(rows, shapes_by_m, params, out_dir)` — mirrors
      `celltracker_bridge.export_field_maps`, reusing its `render_field_figure`
      for the PNG render.
  - `pipeline_graph/registry_adapter.py`: `SPECIAL_INTERP_MAP_OP_KEY =
    "special:interp_map"`, a `param_specs_for` branch (`value_column`,
    `interp_method` linear/nearest/cubic, `fill_time`, `grid_step`, `sigma`,
    `colormap`, `image_format`).
  - `pages/pipelines_page.py`: `_run_interp_maps` (Run), `_preview_interp_maps`
    + `_show_map_popup` (preview), and `_inject_value_columns` to fill the
    `value_column` choice from the measured rows' numeric columns (falling back
    to the standard shape columns + `mean_intensity_<channel>` per live channel).

### Changed

- **Spatial-map nodes are now in the Results category.** The existing
  "Spatial Field Maps" node was renamed **"Cell-Tracker Spatial Maps"** and moved
  from the orange **Special** category to the green **Results** category (joining
  the new Interpolated Spatial Maps node); both keep their special-op dispatch and
  custom Run/preview handlers and their hexagon silhouette — only color /
  Add-dialog grouping changed.
  - `registry_adapter._SPECIAL_OPS` tuples now carry an optional per-op
    `category` (default `SPECIAL`); `special_specs()` reads it.
  - `pipeline_graph/io.py`: `_normalize_special_categories(doc)` runs on load to
    re-derive saved special nodes' category from the current spec, so old graphs
    pick up the new color (cosmetic only — dispatch is op-key keyed, no
    `schema_version` bump).

## [Unreleased] - 2026-06-23 (If / Else — full Cell-Tracker tracking metrics)

### Added

- **Per-cell motion & crowding columns from the Cell-Tracker Metrics node.**
  `backend/celltracker_bridge.augment_rows_with_metrics` now also emits `speed`
  (px/frame), `velocity_y`, `velocity_x` (per-track centroid displacement from
  the previous frame) and `cell_density` (`1 / (π · neighbor_dist_mean²)`,
  cells/px²) — added in the new `_add_velocity_density` helper and appended to
  `METRIC_COLUMNS`. The vendored `compute_spatial_metrics` computed velocity
  internally but dropped it; these surface it as branchable per-row metrics.

### Changed

- **If / Else condition builder now exposes all tracking / Cell-Tracker
  results.** When an If / Else node is placed downstream of a Track Objects or
  Cell-Tracker Metrics node, its **Metric comparison** dropdown now lists track
  length, speed, velocity_x/y, cell density, neighbor distance, local divergence
  / curl and self-fold (plus every per-channel `mean_intensity_<ch>` and any
  other column already in the computed rows).
  - `pipeline_graph/conditions.py`: new public `TRACKING_METRICS` list (folded
    into `_METRICS`) and `metric_choices(extra)` helper that unions the static
    catalog with live columns.
  - `widgets/node_board/condition_builder_dialog.py`:
    `ConditionBuilderDialog(..., metrics=…)` injects the discovered columns into
    the `metric` choice editor (alongside the existing `channels` injection).
  - `pages/pipelines_page.py`: `_available_metric_columns(node)` gathers columns
    from `self._results_rows` and — via `_has_upstream_celltracker(node)`, an
    incoming-edge walk of the analysis slice — adds the Cell-Tracker columns even
    before a Run has produced rows. `_edit_if_else_condition` passes them in.

## [Unreleased] - 2026-06-21 (Whole-frame review — corner cell-over-time inset)

### Added

- **Corner "cell-over-time" inset in whole-frame object review.** In
  `WholeFrameReviewDialog` (Review Objects → `mode = "Whole frame"`), a plain
  **click selects a cell** and opens a zoomed inset in the canvas's top-right
  corner. The inset crops the composite frame centred on the cell, outlines its
  label (cyan, via `analysis_page._overlay_labels`), and **follows the T slider /
  playback** so the cell stays centred and same-scale while it evolves. Works for
  tracked cells (animates across all their frames; the title reads
  `Track N • T • area`) and untracked single objects (one frame; gap frames show
  `no object at T`). The selected cell is ringed on the main frame, and the inset
  is dismissed by clicking empty space or switching position.
  - New per-track index `_track_index[m][track_id] = {t: row}` and a cached
    `_last_rgb` composite; helpers `_selected_row_at_current`, `_crop_for_row`,
    `_crop_px_for_selection`, `_paint_corner_inset`.
- **Accept Track / Reject Track in whole-frame review.** Buttons appear while a
  cell is selected and decide every frame of the selected track at once (single
  row for untracked), reusing the existing `id(row)` decision model so
  `rejected_rows` / `accepted_rows` and the Run/preview apply paths are unchanged.

### Changed

- **Whole-frame review click semantics.** A plain click now *selects* a cell for
  the corner inset; **Alt-click (or Ctrl-click)** toggles reject (previously a
  plain click toggled reject). Per-frame **Accept Frame / Reject Frame** buttons
  are unchanged.

## [Unreleased] - 2026-06-20 (V1.46.3 Pipelines — maximize/pop-out viewer & plots)

### Added

- **Maximize buttons for the Pipelines image viewer and plots/data panel.** A
  small `fa5s.expand` button on the viewer's overlay-tab row and in the data
  `QTabWidget`'s top-right corner pops the panel out into a full, non-modal
  window; clicking again (icon swaps to `fa5s.compress`) docks it back. The
  live widget is **reparented**, not rebuilt, so all state is preserved both
  ways — overlay tabs (Image/Segmentation/Tracks/Vectors), sliders, channel
  LUTs, zoom/pan, and the plot/measurements tabs.
  - New reusable `widgets/popout_window.py` `PopOutWindow(QDialog)`: borrows a
    widget, hosts it maximized, and calls `on_restore(widget)` exactly once
    (title-bar close or programmatic `restore()`).
  - `pages/pipelines_page.py`: `_toggle_popout(kind)`, `_popout_windows` map,
    and `_set_panel_visible(kind, visible)` so the Results image/table/split
    mode logic never hides a panel that is currently popped out. Restore
    re-inserts at the original splitter slot and re-applies docked visibility.

## [Unreleased] - 2026-06-20 (V1.46.2 Live-run frame sync, conditional tabs, richer if-else, review overlay)

### Bug Fixes

- **Programmatic navigation now moves the selected frame in the strip.**
  `MultiAxisViewer._on_strip_current` updated only the (hidden) backing slider —
  with signals blocked the strip's highlight never moved. It now calls
  `FrameStrip.set_current` explicitly, so live-run per-frame streaming (and any
  `set_current_frame` call) visibly advances the T-strip selection as each frame
  is processed.

### Added

- **If-else conditions over all recorded data.** The "Metric comparison" block's
  field list (`conditions._METRICS`) now includes the tracking / Cell-Tracker /
  SerialTrack-derived columns: `track_length`, `delta_area_um2`, `delta_area_px`,
  `neighbor_dist_mean`, `neighbor_dist_std`, `local_divergence`, `local_curl`,
  `self_fold` (plus the existing area/shape/intensity). Cells-per-frame stays in
  the `object_count` block and track counts in the tracking-family blocks, so the
  condition builder can branch on any of them.
- **Review Objects uses the segmentation overlay format, with an overlay choice.**
  `TrackValidationDialog` renders each reviewed cell via the shared
  `analysis_page._overlay_labels` (same outline/fill engine as the segmentation
  overlay) and gained an **Overlay** control (Outline / Filled + color swatch +
  weight). It seeds from the viewer's overlay style (passed in via a new
  `overlay_style` arg) so the review matches what's shown on the image.

### Changed

- **Overlay tabs appear only when the current pipeline produced their data.**
  `pipelines_page._update_overlay_tabs_available` hides the **Tracks** / **Vectors**
  overlay tabs until a Track Objects node has linked, and the **Segmentation** tab
  until an analysis result exists (or a run is live); **Image** is always present.
  (Plot/data tabs were already gated — cleared at run start, re-created only for
  the data each run produces.) Uses `QTabBar.setTabVisible`; a hidden active tab
  falls back to Segmentation/Image.

## [Unreleased] - 2026-06-20 (V1.46.1 Viewer — overlay style control + pixel-hover readout)

### Added

- **Overlay style control** in the image viewer toolbar — a small "Overlay" button
  next to **Pan** (before the Zoom %) opens a popup to set overlay **color**
  (uniform), **weight** (outline / vector line thickness), **multicolor**
  (per-object/track palette) and **opacity**. Off by default ("Custom overlay
  style" unchecked) so overlays keep each result's built-in look until the user
  opts in. The button is shown only while an overlay is active.
  (`widgets/image_viewer.py` `ZoomToolbar`: `overlay_style()` +
  `overlay_style_changed`; `MultiAxisViewer.overlay_style()` passthrough,
  recompute-overlay-only on change.)
- **Pixel-hover readout** to the right of the Zoom % — shows the X/Y position and
  the intensity of every enabled channel under the cursor, with a **px / µm**
  toggle button switching between image pixels and absolute **stage** micrometers.
  Stage µm uses the same frame-center origin formula as
  `results_engine` `centroid_*_stage_um` (so a cell's hover value matches its
  measured stage centroid). New `hover` signal on both `ImageCanvas` and
  `GpuImageCanvas`; `MultiAxisViewer._on_pixel_hover` reads per-channel planes
  (cached per `(m, t, z)`), `_pixel_to_stage` converts.

### Changed

- The Pipelines overlay painters honor the overlay style: segmentation outlines
  recolor / thicken (`_overlay_labels` gains a `thickness` param, dilating the
  boundary); track-colored cells use a uniform color when multicolor is off;
  vector arrows take the chosen color + weight. (`pages/pipelines_page.py`,
  `pages/analysis_page.py` `_overlay_labels`.)
- `pages/pipelines_page.py` / `pages/analysis_page.py` now pass `stage_xy_um`
  (from `record.nd2_metadata`) into `MultiAxisViewer.set_volume`, so the hover
  stage toggle has data.

## [Unreleased] - 2026-06-20 (V1.46.0 Pipelines — live overlays, overlay/plot tabs, track displays)

> Overhauls the Pipelines viewer to match CellTracker: overlays on every frame,
> live per-frame updates during a run, tabbed overlays + plots, and track-colored /
> vector displays. See [CodeLog/ClaudesPlan/V1.46_live_overlays_and_tabs.md](../ClaudesPlan/V1.46_live_overlays_and_tabs.md).

### Bug Fixes

- **Analysis overlays now render on every frame after a Run** (previously only on the
  frame shown when Run was clicked). `_composite_*_overlay` resolved masks only from the
  previewed planes (`_analysis_screen_results`); a new `_overlay_result_for(m, t)` falls
  back to the committed full-stack result (`_run_results_by_m[m].label_masks[ch][t]`), and
  run finalization re-applies the overlay hook so all frames repaint.

### Added

- **Overlay tabs on the viewer** (`pages/pipelines_page.py`): Image / Segmentation /
  Tracks / Vectors: cells / Vectors: field. One dispatcher `_composite_pipeline_overlay`
  draws the active layer over the **cached** processed base; switching a tab only
  recomputes the overlay (`MultiAxisViewer.invalidate_post_process_cache`) — the base
  image is never re-rendered (low RAM / cheap). New `MultiAxisViewer.set_current_frame`.
- **Tabbed plots/data panel** (bottom of the right split, a `QTabWidget`): the
  Measurements table plus lazily-built `MplCanvas` tabs — **Cells/frame** + **Area**
  histogram after an analysis Run, **Tracks/frame** + **Track length** histogram after
  Track Objects.
- **Live per-frame streaming** (CellTracker's per-frame overlay + Live Stats): a new
  worker→GUI channel — `ProgressReporter.frame` / `report_frame` →
  `JobRunner.job_frame` → `plane_runner.run_planes_to_labels(frame_cb=…)` (via
  `make_frame_cb(params)`, injected by `PipelineCommitJob`). As each frame is segmented
  the viewer jumps to it, paints a live red outline, and the Cells/frame plot extends.
  Wired for all five analysis pipelines (StarDist, Cellpose, spots, tear, histogram).
- **Track-colored solid overlay + velocity vectors** — new Qt-free
  `backend/track_overlays.py`: `generate_track_colormap` (HSV-spread, from CellTracker),
  `make_colored_overlay` (solid per-track fill, from CellTracker), `draw_cell_vectors`
  and `draw_grid_velocity` (arrowheads via `cv2.arrowedLine`). After Track Objects, the
  viewer switches to the Tracks tab and **plays through the frames** so cells fill in with
  their track color, ending on frame 0; the Vectors tabs draw per-cell motion arrows and
  the gridded velocity field.

### Changed

- `compute/progress.py`, `compute/runner.py`, `compute/pipeline_jobs.py` — per-frame
  `frame` / `job_frame` streaming channel (connected + disconnected per job).
- `backend/analysis/plane_runner.py` — `run_planes_to_labels(..., frame_cb=None)` +
  `make_frame_cb(params)`; the 5 analysis pipelines forward it.
- `pages/pipelines_page.py` — right pane restructured (viewer wrapped with an overlay tab
  bar; bottom is a data `QTabWidget`); overlay callbacks unified through the dispatcher.
- No new pip dependency (OpenCV / matplotlib already present) and no schema bump.

## [Unreleased] - 2026-06-19 (V1.45.30 StarDist outline parity + CellTracker-style cell review)

### Changed

- **StarDist cell outlines now match CellTracker's rendering exactly.** When the
  StarDist Segmentation node renders boundaries as outlines, its result carries
  `overlay_color = (255, 50, 50)` and `overlay_alpha = 1.0`, and `_overlay_labels`
  draws **all** boundary pixels in that single colour at full opacity via
  `find_boundaries(mode="outer")` — byte-for-byte the colour/brightness CellTracker
  uses (`widgets/cell_overlay.draw_boundaries_on_painter`, `QColor(255, 50, 50)`).
  - `pages/analysis_page._overlay_labels` gained a single-colour outline path
    (paint every `mode="outer"` boundary pixel in `color`, no per-label
    attribution); the per-label palette outline path uses `mode="inner"` so
    boundary pixels stay object-side and attributable. `backend/results_engine`
    `_label_boundaries` likewise uses `mode="inner"` for its per-label export
    overlays (fixes outline export painting nothing).

### Added

- **Review Objects "Single objects" now mirrors CellTracker's Results (Cells)
  per-cell verification** (`widgets/track_validation_dialog.py`). Each review panel
  gained CellTracker's level of detail:
  - **Red cell-boundary outline** on the cropped cell — `(255, 80, 80)` via
    `find_boundaries`, matching CellTracker's cell-detail crop (replaces the old
    filled orange highlight).
  - **Per-frame cell-detail text** under the crop: `frame i/N · {area} px² ·
    ecc {e} · {channel} {intensity}`, updating as the shared T slider moves.
  - **Per-cell metric trace plot** (`MplCanvas`) over the unit's frames with a
    dashed marker at the current frame — the single-cell trace CellTracker shows.
  - A **Metric selector** in the top bar (Intensity per channel / Area /
    Eccentricity), derived from the columns present, driving the trace + info — the
    analogue of CellTracker's Results-page Metric combo.
  - The Accept/Reject validation flow, batch navigation, and `accepted_rows` /
    `rejected_rows` outputs are unchanged.

## [Unreleased] - 2026-06-19 (V1.45.29 Analysis — StarDist segmentation + outline overlays)

> Routes CellTracker's cell-boundary drawing (StarDist — its only segmentation
> method) into ND2Studios as a new analysis node, plus a boundary-outline display
> mode. See [CodeLog/ClaudesPlan/V1.45_stardist_segmentation.md](../ClaudesPlan/V1.45_stardist_segmentation.md).

### Added

- **"StarDist Segmentation" analysis node** — per-frame 2D instance segmentation of
  star-convex nuclei via StarDist, registered through `AnalysisPipeline` so it
  appears in the Pipelines **Analysis** tab and the Analysis page automatically
  (no graph/Add-dialog changes). Params: `channel_name`, `model_name`
  (`2D_versatile_fluo` / `2D_versatile_he` / `2D_paper_dsb2018`), `prob_thresh`
  (0.5), `nms_thresh` (0.3), `scale` (1.0 → no rescale), `min_area`/`max_area`
  filters, and `outline` (render boundaries as outlines). Its label masks feed the
  existing measurement → Track Objects (Centroid / SerialTrack / Cell-Tracker) →
  metrics / export chain — i.e. boundary drawing **and** tracking via StarDist.
  - **Optional, lazy dependency** (like Cellpose): `pip install stardist tensorflow
    csbdeep`. The module is always importable; `run()` raises a friendly
    `ImportError` when StarDist is missing. Not added to `requirements.txt`.
- **Vendored StarDist backend** at `nd2studios/backend/celltracker/segmentation.py`
  (`get_stardist_model`, `segment_frame`, `segment_timeseries`, plus the
  single-threaded TF config and the Windows csbdeep-symlink patch). Qt-free; TF is
  imported lazily, the GPU flag is honored by clearing `CUDA_VISIBLE_DEVICES`
  before the first TF import.
- **Boundary-outline display mode** — `AnalysisResult.overlay_outline`. When set,
  label masks render as boundary outlines (via `skimage.segmentation.find_boundaries`,
  `mode="inner"`) instead of filled regions, across the analysis-page overlay, the
  pipeline preview overlay, and the `results_engine` export overlays.

### Changed

- `core/analysis_registry.py`: `AnalysisResult` gains `overlay_outline: bool = False`
  (backward-compatible).
- `pages/analysis_page.py`: `_overlay_labels(..., outline=False)` paints only
  boundary pixels when set; `_composite_overlay` passes `result.overlay_outline`.
- `pages/pipelines_page.py`: the preview overlays pass `res.overlay_outline`;
  `_run_export` sources `outline = any(r.overlay_outline …)` and forwards it.
- `backend/results_engine.py`: new `_label_boundaries` helper; `outline` param on
  `export_overlay_frames`, `export_label_masks_as_overlay`, `export_organized`,
  `_render_export_rgb`, `_label_to_rgb` (paint boundary pixels only when set).
- `nd2studios/__main__.py`: force-import the new analysis module at startup.
- No new pip dependency (StarDist is optional) and no schema bump.

## [Unreleased] - 2026-06-19 (V1.45.28 Pipelines — CellTracker integration)

> Routes the parts of the sibling **CellTracker** backend that ND2Studios lacks
> into cell-tracking pipelines, **without** re-integrating what already exists
> (Cellpose segmentation, the centroid linker, SerialTrack, `compute_measurements`).
> See [CodeLog/ClaudesPlan/V1.45_celltracker_integration.md](../ClaudesPlan/V1.45_celltracker_integration.md).

### Added

- **Two CellTracker tracking methods on the "Track Objects" node**, selectable
  from the `method` dropdown alongside *Centroid* and *SerialTrack*:
  - **"Cell-Tracker: Topology (Hungarian)"** — blends a rotation-invariant
    neighbor distance/angle descriptor with distance cost (`track_timeseries`).
    Params (shown via `visible_when`): `ct_n_neighbors` (default 5),
    `ct_topo_weight` (0–1, default 0.3).
  - **"Cell-Tracker: Spatial Fingerprint"** — position + log-area cost with
    multi-frame gap filling (`track_fingerprint`). Params: `ct_area_weight`
    (0–1, default 0.3), `ct_max_gap` (default 3).
  - Both reuse the shared `max_distance` / `distance_unit` / `min_track_length`
    knobs and annotate rows with the same `track_id` / `track_length` /
    `track_validation`, so Review / if-else / Export are unchanged. CellTracker's
    `track_serialtrack` is **not** routed (hardcoded dev path; already covered by
    the vendored `serialtrack/`).
- **"Cell-Tracker Metrics" node** (orange special, hexagon). Augments tracked
  rows with per-cell `neighbor_dist_mean` / `neighbor_dist_std` /
  `local_divergence` / `local_curl` (`compute_spatial_metrics`) and, when an
  `intensity_channel` is chosen, `self_fold` (each cell vs. its own time-average,
  `compute_self_fold_change`). Params: `n_neighbors` (default 6),
  `intensity_channel` (live-injected; "None" skips self-fold). Velocity/fold
  metrics need `track_id` from an upstream Track Objects node; neighbor distance
  does not.
- **"Spatial Field Maps" node** (orange special, hexagon). Renders Eulerian
  gridded heatmaps (`compute_spatial_fields`: density, mean_area, intensity,
  fold_change, self_fold, speed, velocity_x/y, divergence, curl) per (M, T) and
  writes them to a chosen folder on **Run** — PNG (colormap + colorbar, quiver
  overlay for *speed*) or raw float TIFF. **Preview** double-click renders the
  selected (M, T) field in a popup `MplCanvas`. Params: `field`,
  `intensity_channel`, `grid_step` (default 20), `sigma` (default 2.0),
  `colormap`, `image_format`.
- **Vendored headless CellTracker subset** at `nd2studios/backend/celltracker/`
  (`tracking.py`, `metrics.py`, `fields.py` + `__init__.py`) — pure
  numpy/scipy/pandas (no Qt, no StarDist/TensorFlow). Copied from
  `Cell-Tracker/CellTracker/backend`; StarDist segmentation and basic per-cell
  measurement are intentionally **not** copied (ND2Studios has Cellpose +
  `results_engine`).

### Changed

- `backend/object_tracker.py`: `link_objects(...)` / `link_objects_with_params`
  gain `ct_n_neighbors` / `ct_topo_weight` / `ct_area_weight` / `ct_max_gap`;
  `link_objects` dispatches the two new methods per (channel, m_position) group to
  the new `_link_group_celltracker` (builds a CellTracker DataFrame from the
  row-dicts, runs the chosen linker, remaps per-call track ids onto the shared
  global counter so ids never collide across groups). The shared
  `min_track_length` / `track_validation` post-pass covers all linkers.
- New `backend/celltracker_bridge.py` (Qt-free) adapts ND2Studios row-dicts ↔
  CellTracker DataFrames: `augment_rows_with_metrics`, `compute_field_for_frame`,
  `draw_field_on_ax` / `render_field_figure`, `export_field_maps`. Field
  rendering uses matplotlib's object-oriented Agg API so it never disturbs the
  GUI's global `QtAgg` backend.
- `pipeline_graph/registry_adapter.py`: new `SPECIAL_CT_METRICS_OP_KEY` /
  `SPECIAL_CT_FIELDS_OP_KEY` ops + their param specs; the Track Objects method
  tooltip and `ct_*` params. `pages/pipelines_page.py`: `_run_ct_metrics`,
  `_run_ct_fields`, `_preview_field_maps`, `_show_field_popup`, `_field_shape_for`
  handlers (Run + Preview); channel injection now also fills `intensity_channel`
  and runs regardless of stage (the new specials are RESULTS-stage).
- No new pip dependencies and no schema bump (only new op-key strings + params;
  the `Node`/`Edge` model is unchanged).

## [Unreleased] - 2026-06-19 (V1.45.27 Pipelines — node param popup stays on-page)

### Bug Fixes

- **Node parameter popups no longer bleed off the page.** The `ParamPopup` was
  pinned to a fixed 280×320 and simply `move()`d to the node's top-right corner,
  so nodes near the right/bottom edge pushed it past the window (and off-screen).
  It now **sizes to its editor content** (capped to the page; the scroll area
  absorbs any overflow) and is **clamped fully inside the top-level window**
  (intersected with the screen's available area). New helpers
  `ParamPopup._resize_to_content`, `_move_within_page`, `_page_rect`; horizontal
  scrolling is disabled so rows compress to the clamped width.

## [Unreleased] - 2026-06-19 (V1.45.26 Pipelines — SerialTrack tracking option)

### Added

- **"SerialTrack (topology PTV)" method on the "Track Objects" node.** A second
  linker alongside *Centroid (nearest-neighbor)*, selectable from the node's
  `method` dropdown. SerialTrack (Yang et al., *SoftwareX* 19 (2022) 101204) is a
  scale/rotation-invariant topology matcher — robust where the centroid linker
  loses tracks under large or rotational inter-frame motion. It runs on the
  object centroids the analysis pipelines already produce (SerialTrack's
  `track_coordinates` path — no image re-detection) and annotates rows with the
  same `track_id` / `track_length` / `track_validation`, so Review / if-else /
  Export are unchanged.
  - New SerialTrack-only params, shown only when SerialTrack is selected (via
    `ParamSpec.visible_when`): `st_mode` (**Incremental** — link each frame to the
    previous, default; **Cumulative** — link every frame to the first) and
    `st_n_neighbors` (topology-descriptor size, default 25).
  - The centroid-only knobs (`max_size_diff`, `max_frame_gap`) are now hidden when
    SerialTrack is selected; `max_distance` (→ SerialTrack field-of-search),
    `distance_unit`, and `min_track_length` remain shared.
- **Vendored headless SerialTrack** at `nd2studios/backend/serialtrack/` (9
  modules: `config, tracking, detection, matching, outliers, regularization,
  prediction, fields, trajectories` + `__init__.py`). The upstream
  `io/results/run` GUI/IO modules are deliberately omitted, keeping backend purity
  (no PySide6 under `backend/`). See [Research/serialtrack.md](../../Research/serialtrack.md).

### Changed

- `backend/object_tracker.py`: `link_objects(...)` and `link_objects_with_params`
  gain `st_mode` / `st_n_neighbors`; `link_objects` now dispatches per
  (channel, m_position) group to `_link_group_serialtrack` (new) for the
  SerialTrack method, else the existing centroid `_link_group`. The
  `min_track_length` / `track_validation` post-pass is shared by both linkers.
- `requirements.txt`: add **numba** (JIT kernels for the SerialTrack matcher;
  the only new hard dependency on the linking path).

## [Unreleased] - 2026-06-19 (V1.45.25 Pipelines — "Review Objects" two modes)

### Added

- **"Review Objects" node** (renamed from *Review Object*) with a `mode` choice:
  - **Single objects** — the cropped-panel accept/reject dialog
    (`TrackValidationDialog`), now **generalized to untracked objects**: when a
    Track Objects node feeds it, each *track* is one review unit; with no
    tracking, each *object* (one label on one frame) is its own unit.
  - **Whole frame** — a new `WholeFrameReviewDialog`: a single viewer that
    composites the current `(M, T)` frame and draws each object's **label id in
    white at its centroid**. Step **T** with the slider, switch **M** with the
    position dropdown (Z collapsed at load). **Click an object to toggle reject**
    (white → red), or **Accept Frame / Reject Frame** to decide every object on
    the frame at once. Image channels are fetched lazily per position.

### Changed

- **`TrackValidationDialog` refactored to a generic unit model** (`_ReviewUnit`
  + `_build_units`, new `include_untracked` arg). Decisions are now exposed as
  **row references** — `rejected_rows` / `accepted_rows` — with
  `rejected_track_ids` / `accepted_track_ids` kept as derived back-compat (the
  Results page is unchanged). Window title is now "Review Objects".
- **`pipelines_page` review handlers** route on the node's mode:
  `_run_review_objects` → `_run_open_validation(include_untracked=True)` (per-M
  panels) or `_run_review_whole_frame` (one whole-frame viewer across all M);
  `_preview_validate` honors the mode too and no longer force-tracks — tracked
  rows review as tracks, untracked as single objects. New
  `_apply_review_decisions` applies accept/reject by row identity.

### Migration

- `io._migrate_v2_to_v3` also retitles old `Review Object` nodes to
  `Review Objects` and seeds the default `mode` param.

## [Unreleased] - 2026-06-19 (V1.45.24 Pipelines — "Track Objects" node)

### Added

- **"Track Objects" pipeline node** (the renamed, repurposed *Validate Tracked
  Objects* special node). It no longer pops up an accept/reject dialog — it now
  *performs* the object tracking with configurable thresholds (accept/reject
  still lives in the unchanged **Review Object** node). Params:
  - `method` (choice) — the linking method, named for what it is:
    **"Centroid (nearest-neighbor)"**. The node is built to grow more methods
    (overlap / IoU, Kalman, …) as new choices; today there is one.
  - `max_distance` + `distance_unit` (pixels / µm) — the centroid-displacement
    limit beyond which an object is no longer the same track (µm uses the ND2
    pixel size).
  - `max_size_diff` (0–1) — max fractional area change between linked
    detections, `|Δarea| / max(area)`; `1.0` disables the gate.
  - `min_track_length` (frames) and `max_frame_gap` (missed frames a track may
    bridge before it is retired).
  - Wired into Run (`PipelinesPage._run_track_objects`) and the preview walk; the
    node re-links the accumulated rows and is authoritative over the default
    per-M tracking done at measurement time.

### Changed

- **`object_tracker.link_objects` refined for frame-to-frame centroid tracking.**
  Each track's reference centroid + area now **updates every frame** (a moving
  reference that follows the object) instead of the old bounding-box-overlap
  gate. New args `max_size_diff_frac`, `max_frame_gap`, and `method`; the linker
  carries unmatched tracks forward up to `max_frame_gap` frames so a track can
  re-link across a detection gap. New helper `link_objects_with_params(rows,
  params, pixel_size_um=None)` maps the node's param dict (incl. µm→px distance
  conversion) onto `link_objects`. New module constants `METHOD_CENTROID` /
  `TRACKING_METHODS`.

### Migration

- **Pipeline schema bumped to v3.** Saved graphs containing the old
  `special:validate_tracks` node migrate on load (`io._migrate_v2_to_v3`) to
  `special:track_objects` ("Track Objects") with default tracking params.

## [Unreleased] - 2026-06-19 (V1.45.23 Pipelines — load a new file refreshes the tab)

### Bug Fixes

- **Loading a new file on top of an old one now transfers to the Pipelines tab.**
  The page only refreshed on page-navigation (`on_activated`); loading a file
  while the Pipelines tab was showing left it on the previous file's data. The
  page now implements `load_from_experiment(exp)` (called on every active-record
  change), which drops stale per-file preview/overlay state, refreshes the source
  node's channel count, and — when Pipelines is the visible page — rewires the
  viewer + re-runs the preview against the new record. The built pipeline graph
  (nodes / wires / params) is preserved so it carries over to the new file. New
  helpers: `PipelinesPage.load_from_experiment`, `_refresh_active_view`,
  `_refresh_input_node`.

## [Unreleased] - 2026-06-19 (V1.45.22 Pipelines — Export frame organization)

### Added

- **Export node "Frame organization" dropdown.** A single choice controls how the
  exported frames map to files — which of **M / T / Z** *fold* into a file
  (become its pages/stack) vs *split* into separate files:
  - **Per frame** — one image per M, T, Z.
  - **M folds into T** — one file per timepoint, containing all multipoints.
  - **T folds into M** — one file per multipoint, containing all its timepoints.
  - **All frames merged** — one file with everything.
  - Plus the Z-axis variants (Z folds / M·Z / T·Z / M·T) for a fully general
    formula (Z is collapsed at load today, so Z-fold options coincide with their
    non-Z siblings until per-Z masks exist).
- **`results_engine.export_organized`** — the cross-axis exporter. TIFF writes a
  multi-page stack per group; PNG/JPG write a single file or a per-group subfolder
  of numbered frames. Honors the Export node's content (objects / frames) +
  burn-in-overlay; runs on the full-file **Run**. Replaces the old per-M-only
  export path.

## [Unreleased] - 2026-06-19 (V1.45.21 Pipelines — preview walks the graph like Run)

### Changed

- **Preview now executes the graph like a scoped Run.** Double-clicking a node
  (the deliberate trigger) walks the chain feeding it on the selected (M × T)
  planes: every reachable node is **shaded**, the executed chain turns **gold →
  normal**, the if-else **branches**, and interactive nodes (**Validate /
  Review**) **pop up before continuing downstream**. Plain frame navigation
  afterwards re-walks **non-interactively** (updates the table/overlay/shading,
  no pop-ups). Nodes off the executed chain (other branches, downstream of the
  previewed node) stay shaded.
- **The Compute Measurements metric picker now takes effect.** Previously the
  Compute Measurements node never ran as a step in preview, so its metric
  selection had nothing to act on. The preview walk's screen+measure now honors
  the chain's Compute Measurements **metrics** selection (only the ticked metrics
  are computed / shown in the table).

### Notes

- `_ResultsScreenMeasureJob` is reused for the walk's analysis+measure portion
  (`_PV_SCREEN_KEY`); the logic/special nodes then walk synchronously
  (`_preview_walk_downstream`). Export / Send / Pause are no-ops in preview
  (they run on the full-file **Run**).

## [Unreleased] - 2026-06-19 (V1.45.20 Pipelines — multi-frame results preview + validate-in-preview)

### Bug Fixes

- **The results table now spans every selected frame.** A results / Compute
  Measurements / if-else / special preview screens the upstream analysis on the
  **current frame or every frame of an M × T selection** and measures each, so
  the table aggregates all selected planes (rows tagged `m_position` + `frame`)
  and the overlay paints each plane. Navigating to a different frame or changing
  the frame selection now re-runs the preview and updates the table
  (`_ResultsScreenMeasureJob`; `_on_viewer_coords` re-previews on any plane-set
  change).
- **Validate tracked objects works in preview.** A Validate / Review node's param
  popup has a *Validate tracked objects…* button that opens `TrackValidationDialog`
  on the current multipoint's selected-T stack (objects track across the selected
  timepoints); accept/reject is applied back onto the preview rows.

### Deferred (next)

- **Export frame organization** — a single *Export frame organization* dropdown
  on the Export node: per-frame; **M folds into T** (one file per T, all M
  inside); **T folds into M** (one file per M, all T inside); **all frames
  merged**; generalized over Z. Needs a cross-axis stacking exporter — landing
  next.

## [Unreleased] - 2026-06-19 (V1.45.19 Pipelines — measurement picker + previewable results/logic/special nodes)

### Added

- **Compute Measurements metric picker.** A Compute Measurements node now has a
  *Select measurements…* button (param popup) that opens a grouped checklist
  (`MeasurementSelectDialog`) of every quantity — Size / Change / Position /
  Shape / Bounding box / Intensity. Only the ticked metrics are computed, so
  asking for a couple runs much faster than the full set. The selection is stored
  in the node's `params["metrics"]`.
- **`compute_measurements(..., metrics=...)`** — selective computation: skips the
  costly shape props and per-channel intensity unless requested, and prunes the
  output columns to the selection (identity columns always kept). The analysis is
  **not** re-run — measurements come from the existing label masks. `metrics=None`
  computes everything (back-compat).

### Changed / Bug Fixes

- **Previewing a results / logic / special node now works** (the table appears).
  Double-clicking any node in the merged tab previews it: results / if-else /
  special nodes resolve the upstream analysis result (committed, or the current
  frame's screened result) and show the overlay + measurements table. Previewing
  an **If/Else** node also reports which branch it would take on the current
  frame's objects.
- **Editing moved off double-click.** Double-click now always previews; the
  If/Else condition builder and the measurement picker open from the param
  popup's **Edit…** button (`ParamPopup` gained an optional action button +
  `edit_requested` signal). `_merged_mode` treats logic + special nodes as
  results-mode so their table/overlay show.

### Deferred

- **Export parameters** (TIFF/JPG/PNG + frame-folding: all merged, M-in-T,
  T-in-M) — needs a cross-multipoint stacking exporter; coming next (the fold
  semantics will be confirmed first).

## [Unreleased] - 2026-06-18 (V1.45.18 Pipelines — condition builder + interactive Run nodes [Phase 2])

Phase 2 of the merged Analysis tab: the full if-else **condition builder** and
the **interactive special nodes** now run for real.

### Added

- **If-else condition builder.** Double-clicking an If/Else node opens
  `ConditionBuilderDialog`: stack condition **blocks**, combine them with
  **ALL (and) / ANY (or)**, and negate any block (**NOT**), with a live readout.
  Blocks span all three families — **results-number** (metric vs value with
  any/all/mean/median/min/max/count aggregation), **object-population** (object
  count, empty field, marker-positive count/fraction, co-localization /
  double-positive), and **timelapse / tracking** (track count, persistence,
  object-count change over time, track displacement / motility). The condition is
  stored in the node's `params["condition"]` and the Run gate evaluates it via
  `evaluate_condition` over the live measurement rows.
- **Interactive special nodes during a Run:**
  - **Validate Tracked Objects** / **Review Objects** open `TrackValidationDialog`
    on the current measurements; accepted/rejected `track_id`s filter the stream.
  - **Dismiss** drops rejected-track rows (or the whole set on a false branch).
  - **Export Objects / Frames** writes to a chosen folder via
    `results_engine.export_overlay_frames` / `export_label_masks_tiff` (honoring
    the node's content / image-format / burn-in-overlay params).
  - **Send to Results** pushes the rows (+ masks/channels) into the top-level
    Results page and navigates there.
- **Object tracking in the Run**: measurement nodes run `object_tracker.link_objects`
  so `track_id` / `track_length` are available to timelapse conditions and the
  Validate node.
- **Resume after Pause**: a Pause node halts the Run into editor mode (all nodes
  un-shaded) but retains the runner — clicking **Run** resumes from the pause
  point (restarts instead if the graph's node set changed).

### Added (graph core)

- `pipeline_graph/conditions.py` (Qt-free): `Condition` / `ConditionBlock` tree
  with `to_dict`/`from_dict`, a `BLOCK_KINDS` catalog (family + UI schema +
  evaluator per block), `evaluate_condition`, `describe_condition`, `families`,
  `make_block`, `default_condition`. The if-else node seeds a default condition.
- `widgets/node_board/condition_builder_dialog.py` — the builder UI.

### Bug Fixes

- **Run now processes the WHOLE file, not just the current frame.** The analysis
  step previously ran on the single multipoint the viewer was on. Run now loops
  **every multipoint** — each M's analysis job is followed by a measurement job
  on a worker thread, accumulating per-M results + rows (tagged with
  `m_position`, tracked per-M) before the if-else / special nodes run. Downstream
  nodes operate on the full dataset: **Export** writes every M (`pipeline_mNNN_…`),
  **Send to Results** sends all rows, and **Validate / Review** open once per M.
  Verified: a 3-multipoint run issues 3 analysis + 3 measure jobs and aggregates
  rows across M0–M2 before branching.
- **If-else / special nodes now work directly after an analysis node** (no
  Compute Measurements node required). Previously the if-else evaluated over an
  empty measurement set unless a `results:` node had run, so an
  `analysis → if-else → (Export / Dismiss)` graph always took the false branch
  and Export never fired. The Run now **auto-derives measurements from the
  upstream analysis result** (`_ensure_run_rows` → `compute_measurements` +
  `link_objects`, cached per result) whenever an if-else condition or a special
  node (Validate / Dismiss / Send) needs rows.

## [Unreleased] - 2026-06-17 (V1.45.17 Pipelines — merged Analysis tab + branching Run [Phase 1])

Merges the Analysis and Results sub-tabs into one **Analysis** tab and adds
branching control flow, special action nodes, and a graph-walking **Run**.
This is **Phase 1** (structure + visuals + a working Run skeleton); the full
condition-block builder and interactive special-node dialogs land in Phase 2.

### Added

- **Merged Analysis sub-tab.** The Pipelines page now has two sub-tabs —
  **Processing** and **Analysis** — and the Analysis scene holds analysis
  pipelines, results ops, branch nodes, and special nodes together, each painted
  by **category**: analysis = pink, results = green, logic (if-else) = purple,
  special = orange (Dismiss = red). The previewed node's category drives the
  viewer (analysis mask overlay vs. results overlay + measurements table).
- **If/Else branch node** (`logic:if_else`, purple **triangle**): one input at
  the apex, `true` (bottom-left) and `false` (bottom-right) outputs. An
  *aggregate gate* — its condition is evaluated over the upstream measurement
  rows and the whole downstream pipeline flows down one branch
  (`evaluate_simple_condition`; Phase-1 params: `metric`/`aggregate`/
  `comparator`/`value`).
- **Special action nodes** (orange **hexagons**; Dismiss is a red rectangle):
  Validate Tracked Objects, Review Object, Dismiss, Export Objects/Frames, Send
  to Results, Pause. Phase 1 registers and wires them; Pause halts a Run and
  returns to editor mode. (Interactive dialogs + exporters are Phase 2.)
- **Run** (replaces Apply on the merged tab). A Qt-free `GraphRunner` walks the
  graph in topological order with **branch pruning**; the page paints each node
  **shaded** (pending / un-taken branch) → **gold** (executing) → **normal**
  (done), parking on background analysis / measurement jobs. **Pause** un-shades
  all nodes (editor mode).
- **Add dialog category tabs.** The node picker groups its catalog into
  Processing / Analysis / Results / Logic / Special tabs.
- Graph core: `NodeCategory` + `ShapeKind` (on `Node`); `if_else_spec()`,
  `special_specs()`, `merged_action_specs()`; `GraphRunner`,
  `topological_order()`, `evaluate_simple_condition()` in `executor.py`.

### Changed

- **`PipelineDoc` schema → v2.** Analysis + Results share one `GraphSlice`
  (`PipelineDoc.merged` aliases `analysis`); results nodes keep `stage=RESULTS`
  (so they color green) but live in the analysis slice. `load_pipeline` migrates
  v1 docs (`_migrate_v1_to_v2` folds the results slice into the merged slice).
- `NodeItem` is now shape-aware (rounded rect / triangle / hexagon) and colors by
  node category; added `set_run_state`. `NodeScene` gained
  `set_run_states` / `clear_run_states`.
- `_pipeline_graph_selftest.py` extended: category/shape on built nodes,
  `GraphRunner` branch-pruning walk, `evaluate_simple_condition`, io v2
  round-trip + v1→v2 migration.

### Bug Fixes

- **The if-else / special nodes now accept any upstream node.** Their ports were
  strictly `DATA`, so wiring an analysis node (`BINARY`) or the processed input
  (`IMAGE`) into the triangle's top vertex was rejected (red flash) — you could
  only connect a Compute Measurements node. Added a wildcard `PortType.ANY`
  (`can_connect` treats it as compatible with any type) and gave the if-else and
  special nodes ANY ports, so any upstream can feed them and either branch can
  drive any downstream node. Typed mismatches among regular nodes (e.g. IMAGE →
  BINARY input) are still rejected.

## [Unreleased] - 2026-06-14 (V1.45.16 Pipelines — analysis overlay auto-refresh)

### Bug Fixes

- **The Analysis preview now auto-refreshes.** A screened result only called
  `invalidate_post_process_cache()`, which clears the overlay cache but does not
  repaint — so the overlay appeared on a tab switch (where `set_volume` repaints)
  but not when navigating frames, editing params, or changing the previewed node.
  The `_ANALYSIS_PREVIEW_KEY` handler now calls `MultiAxisViewer.refresh()` after
  storing the screen results, so the overlay repaints for the just-screened
  plane(s). Verified on the real ND2: navigating (0,1)→(0,4) re-screens and
  repaints.
- **`ProcessedFrameVolume` caches processed planes** (bounded LRU), so the extra
  repaint — and repeated reads of the same Analysis base frame — don't re-run the
  recipe each time.

## [Unreleased] - 2026-06-14 (V1.45.15 Pipelines — multi-axis selection, processed Analysis base, Apply stays put)

### Bug Fixes

- **`set_volume` was wiping the frame-strip selection**, so the first preview
  dropped the user's multi-frame selection — multi-axis (M + T) selections only
  processed the current M, and browsing already-processed frames re-ran the
  pipeline. `_show_preview_volume` now **saves and restores** the M/T/Z tile
  selection around `set_volume` (new `MultiAxisViewer.set_axis_selection`), so a
  selection survives the volume swap and keeps driving the preview.
- **Multi-axis selection now previews the M × T product.** `_selected_planes`
  takes the cartesian product of the M and T strip selections (each defaulting to
  the current position), so selecting positions *and* timepoints processes every
  combination (verified: M{0,2} × T{3,6} → all four planes processed, including
  other-M planes).
- **Browsing already-processed frames no longer re-runs the pipeline.**
  `_on_viewer_coords` re-previews only when the *set of planes to preview*
  actually changes, so arrowing through a selection (all already processed) is a
  no-op; with no selection it still follows the single current frame.
- **Apply stays on the Processing tab and shows the progress bar.** It no longer
  auto-switches to Analysis; it records the recipe (applied lazily to the whole
  file) and refreshes the Processing preview. The **Analysis sub-tab base now
  shows the processed image** (re-added `ProcessedFrameVolume`, applied per
  displayed frame) instead of raw — so a committed Background Subtract is visible
  in Analysis (verified: analysis base diff-from-raw ≈ 52).

## [Unreleased] - 2026-06-14 (V1.45.14 Pipelines — multi-frame selection preview + Apply → Analysis)

### Added / Changed

- **Multi-frame selection is previewed all at once.** Selecting a range of
  frames on the T strip (Shift-click) now runs every selected frame through the
  golden pipeline — Processing shows each selected plane processed (over raw)
  and Analysis overlays each selected plane's masks. `PinnedProcessedVolume`
  generalized from a single pinned plane to a **set of planes**
  (`set_planes`), `_ProcessingPreviewJob` / new `_AnalysisPreviewJob` process
  the whole selected set (progress per plane), and the Analysis screen result is
  now `{(m, t): AnalysisResult}`. New `MultiAxisViewer.selection_changed`
  signal + `axis_selection()` let the page re-preview when the selection
  changes; with no selection it falls back to the single current frame.
- **Apply (Processing) commits the recipe to the entire file and hands off to
  Analysis.** The recipe is dataset-agnostic and applied lazily to every M/T/Z
  wherever the processed data is read (Export's `EnhancedDataset`; per-frame in
  Analysis via `record.recipe`), so it scales to multi-GB files without
  materializing the stack. After Apply the page **switches to the Analysis
  sub-tab**, whose input now reads the processed output.
- Verified on the real 17 GB ND2: selecting T frames {2,5,8} processes exactly
  those planes (mean|Δ|≈50 each) and leaves frame 3 raw; the analysis preview
  produces results for the selected planes {(0,1),(0,4),(0,6)}; Apply records
  the recipe and switches to Analysis; progress fires; boot clean.

## [Unreleased] - 2026-06-14 (V1.45.13 Pipelines — processed plane follows the current frame)

### Changed

- **The processing preview's single processed plane now follows the user.**
  V1.45.12 pinned it to the frame where you started and left it there; the
  intended behavior is an isolated processed plane laid over the raw volume that
  **transitions to whatever frame you click / select** while only ever
  processing that one frame. `PinnedProcessedVolume.set_pin` now moves the
  processed `(M, T)` **in place**, and `_on_viewer_coords` re-processes the
  current plane (debounced, background job → progress bar) on every navigation /
  frame-strip selection. The update is applied via a new lightweight
  `MultiAxisViewer.refresh()` (re-render only) instead of `set_volume`, so the
  frame selection, LUTs and slider positions are preserved — no churn. Net: at
  any instant exactly one frame is processed, it follows you, and every other
  frame stays raw.
  ([executor.py](nd2studios/pipeline_graph/executor.py),
  [multi_axis_viewer.py](nd2studios/widgets/multi_axis_viewer.py) `refresh()`,
  [pipelines_page.py](nd2studios/pages/pipelines_page.py))
- Verified on the real 17 GB ND2: preview at (0,3) → processed (mean|Δ|≈52),
  navigating to (0,7) re-uses the same volume object (selection preserved),
  (0,7) becomes processed and (0,3) reverts to raw; progress events fire.

## [Unreleased] - 2026-06-14 (V1.45.12 Pipelines — pinned single-frame preview + progress restored)

### Bug Fixes

- **Processing preview now pins to one plane (every other frame stays raw), and
  the progress bar is back.** V1.45.11 scoped the recipe with a *live* viewer-
  coords callable, so the displayed plane was always the current one → every
  frame the user navigated to looked processed ("all frames are background-
  subtracted"); and the switch to instant `set_volume` had removed the progress
  bar. Now the previewed plane is **pinned**: `_do_processing_preview` reads the
  current `(M, T)` plane's raw channels, processes them in a background
  `_ProcessingPreviewJob` (progress bar restored), and shows the result via a
  new `PinnedProcessedVolume` that returns the cached processed frames **only**
  for that pinned `(M, T)` and the raw plane for every other read. So exactly
  one frame is under the pipeline; navigating / Shift-selecting / strip tiles /
  pre-render all read raw. The pin re-sets (fresh job) on a recipe or
  previewed-node change, not on plain navigation. The Analysis/Results base is
  now the plain raw navigable volume with the single-frame overlay on top.
  Removed the now-dead `ProcessedFrameVolume` live-scoping path
  ([executor.py](nd2studios/pipeline_graph/executor.py),
  [pipelines_page.py](nd2studios/pages/pipelines_page.py)).
- Verified on the real 17 GB ND2: pin (0,3) processed (mean|Δ|≈52), (0,5) and
  (40,7) raw (Δ=0), and the preview progress bar fires.

## [Unreleased] - 2026-06-14 (V1.45.11 Pipelines — preview scoped to the current plane)

### Bug Fixes

- **The recipe preview now processes only the `(M, T)` plane the user is on.**
  V1.45.10's `ProcessedFrameVolume` applied the recipe inside *every*
  `get_frame`, so the whole volume read back processed — the frame strip, range
  selection (Shift-click), LUT sampling and background pre-render all ran the
  recipe, i.e. "all frames in T and M were background-subtracted." The wrapper
  now takes a ``current_mt`` (a live ``() -> (m, t)`` callable = the viewer's
  coords); it recipe-processes a plane **only** when it matches, and returns the
  raw plane for every other read. So exactly the displayed plane is "under the
  pipeline"; it follows navigation (the provider is read live, no syncing), and
  selecting / strip-thumbnailing / pre-rendering other frames leaves them raw
  ([executor.py](nd2studios/pipeline_graph/executor.py),
  [pipelines_page.py](nd2studios/pages/pipelines_page.py) `_viewer_mt`).
  Verified on the real 17 GB ND2: plane (0,3) processed (mean|Δ|≈52), (0,5) and
  (1,3) raw (Δ=0); navigating the provider to (0,5) flips which single plane is
  processed.

## [Unreleased] - 2026-06-14 (V1.45.10 Pipelines — on-demand single-frame preview)

### Changed

- **Pipeline previews now process only the displayed frame, on demand.**
  Previously the Processing preview (and the Analysis/Results base image)
  materialized and ran the recipe over the *whole* M's T-stack per refresh —
  i.e. "applied to all frames" — which on a multi-GB file (e.g. M=68, T=10,
  C=4, 1024²) is both heavy and not what a probe should do. New
  `pipeline_graph.ProcessedFrameVolume` wraps the lazy `record._raw_volume` and
  applies the recipe to a single plane inside `get_frame`; the page hands it to
  `MultiAxisViewer.set_volume`, so the viewer reads (and the recipe processes)
  **only the frame on screen** while every M/T/Z stays navigable — no
  full-stack materialization. The Analysis/Results base shows the raw lazy
  volume directly (or a `ProcessedFrameVolume` when a recipe is committed), with
  the single-frame overlay on top. Navigation no longer recomputes a whole
  stack — the viewer just reads the next plane. Small in-RAM files with no lazy
  volume keep the previous background-materialize fallback.
  ([executor.py](nd2studios/pipeline_graph/executor.py),
  [pipelines_page.py](nd2studios/pages/pipelines_page.py))
- Verified against the real 17 GB ND2: the previewed plane is processed
  on-demand, navigating to M40/T7 reads one frame (no materialization), and the
  Analysis overlay stays isolated to the screened frame.

## [Unreleased] - 2026-06-14 (V1.45.9 Pipelines — single-frame analysis preview)

### Bug Fixes

- **Analysis overlay is now isolated to the previewed frame.**
  `_composite_analysis_overlay` painted the committed full-stack result
  (`_analysis_results_per_m`) on every non-screened frame, so after an Apply the
  overlay appeared on *all* frames and live param/node edits only updated the
  single current frame (looking like it "wasn't refreshing"). The Analysis
  preview now paints **only the live single-frame screen result** on the exact
  ``(m, t)`` the viewer is on, and follows the viewer as you navigate / edit;
  the committed full-stack result is reviewed in the **Results** sub-tab. An
  Apply re-screens the current frame so its overlay reflects the committed
  result ([pipelines_page.py](nd2studios/pages/pipelines_page.py)).

## [Unreleased] - 2026-06-14 (V1.45.8 Pipelines — preview navigation & refresh)

### Changed

- **Preview viewer now exposes the whole M axis.** The Processing / Analysis /
  Results previews read the displayed M's channels from `record._raw_volume`
  (`all_channels_as_lazy(m=…)`, mirroring the Recipe page) and call
  `set_channels` with the real `n_multipoints`, so the M slider navigates every
  position instead of being pinned to one M. A new `_channels_for_m` /
  `_record_n_multipoints` and a `_preview_m` tracker drive this
  ([pipelines_page.py](nd2studios/pages/pipelines_page.py)).
- **Preview refreshes when the frame changes.** `_on_viewer_coords` recomputes
  on an **M** change for every stage (the new M's processed stack / overlay /
  table); Analysis additionally re-screens its overlay on any T/Z change and
  refreshes its base image on M change. (Processing needs no T recompute — it
  holds the full T-stack.)
- **Preview refreshes on the right edits.** Editing a node's params now triggers
  a recompute only when that node is the **previewed (golden) node or upstream
  of it** (`_on_popup_params_changed` consults `_upstream_subgraph`); editing a
  downstream / unrelated node no longer recomputes. Double-clicking a new golden
  node still refreshes (unchanged).
- **Switching sub-tabs centers on the whole graph.** `_frame_all_nodes`
  `fitInView`s the scene's items bounding rect (with margin, capped at 1:1) on
  sub-tab switch and page entry, instead of leaving the view on the input node.

## [Unreleased] - 2026-06-14 (V1.45.7 Export — pipeline results sources)

Completes the GA3 chain: a committed analysis/pipeline result is now an
**export source**. The Pipelines tab's Processing → Analysis → Results flow
ends at the Export tab.

### Added

- **"Pipeline Results" export source** ([export_page.py](nd2studios/pages/export_page.py)).
  A 4th entry in the Export type selector, enabled whenever
  `record.analysis_results` holds a committed `AnalysisResult` (written by the
  Pipelines tab's Analysis Apply *or* the standalone Analysis page — fully
  decoupled). Its panel picks a committed result and exports, via
  `results_engine`:
  - **Measurements (CSV)** — `compute_measurements` across every committed M
    (ordered-union-of-keys `DictWriter`; adds `m_position` for multi-M).
  - **Overlay Frames** — `export_overlay_frames` (per-object colored masks over
    the processed composite; scale-bar toggle; tiff/jpg).
  - **Label Masks (TIFF)** — `export_label_masks_tiff`.
  `on_activated` refreshes the selector and disables the source until a result
  exists; the stacked-panel index mapping handles the new entry.

### Bug Fixes

- **`export_label_masks_tiff` no longer fails on Windows** — it wrote int32
  label stacks with `imagej=True`, which the ImageJ TIFF format rejects
  (`data type 'l'`). Now writes a plain multipage int32 TIFF (preserves exact
  label IDs, still opens in ImageJ). Also fixes the Results page's label-mask
  export ([results_engine.py](nd2studios/backend/results_engine.py)).

## [Unreleased] - 2026-06-14 (V1.45.6 Pipelines — Results sub-tab preview)

### Added

- **Results sub-tab wired for preview.** Previewing a Results node overlays its
  chosen analysis result's label masks on the image viewer **and** computes a
  sortable per-object measurements table, with an **image / table / split** mode
  bar (right pane is now a draggable vertical `QSplitter` of the viewer + a
  `QTableView`). The overlay reuses the Analysis overlay path
  (`set_frame_post_process` + `analysis_page._overlay_labels`); the table reuses
  `results_page._MeasurementsModel` and runs `results_engine.compute_measurements`
  off-thread (`_ResultsMeasureJob`, key `"pipeline_results_preview"`), with a
  processed *base image* underneath (`_submit_base_preview`). A summary line shows
  object count / frames / mean area.
- **Results nodes** from new registry helpers (`results_specs` → *Compute
  Measurements*, *Summary*; `results_input_spec` / `results_output_spec`;
  `param_specs_for` synthesizes an `analysis_result` **choice** populated at
  pop-up time from `record.analysis_results`). Action nodes are `Binary → Data`;
  the output node bridges `Results → Export` (deferred consumer). The Add-node
  catalog gains curated Results descriptions.
- **Apply (Results)** recomputes the previewed node's measurements and retains
  the source pipeline on `record.results_config` (Export source-selector wiring
  deferred).

### Bug Fixes

- **Initialized `PipelinesPage._analysis_scratch`** (referenced by the V1.46
  label-streaming `_inject_label_streaming`/`_clear_analysis_scratch` but never
  set in `__init__`, so analysis Apply would `AttributeError`); the scratch dir
  is now cleaned up in `on_close`.
- **Pre-import `compute_measurements` on the main thread** (module import of
  `pipelines_page`) so the Results measure job never first-imports
  skimage/scipy on a worker thread — that raced into a
  `scipy.spatial.distance` circular-import error.

## [Unreleased] - 2026-06-14 (V1.46 Streaming, Resource-Aware Analysis)

Analysis / Results / Batch no longer populate RAM with the whole dataset or full
label-mask stacks on memory-constrained machines. They adopt a Fiji/NIS-Elements
"virtual stack + streaming sink" model — read frames on demand, spill masks to
disk per-frame, keep only measurement rows in RAM — chosen **adaptively** from
the resource-aware load strategy. Plan:
[CodeLog/ClaudesPlan/V1.46_streaming_resource_aware_analysis.md](CodeLog/ClaudesPlan/V1.46_streaming_resource_aware_analysis.md).

### Added

- **`LabelStackWriter`** in [pipeline/storage.py](nd2studios/pipeline/storage.py)
  — incremental, per-frame label-stack sink (Zarr `(1,H,W)` chunks, else a
  memory-mapped `.npy`); `read_label_stack`/`open_label_stack` learned `.npy`.
- **`nd2studios/backend/analysis/plane_runner.py`** (new) — `run_planes_to_labels`
  routes per-frame label output to an in-RAM array (eager) or a `LabelStackWriter`
  (streamed) + `make_label_writers`; **`source_utils.py`** (`source_shape`,
  `read_plane`).
- **`should_stream_analysis(volume, *, decision, monitor, force)`** in
  [utils/resource_strategy.py](nd2studios/utils/resource_strategy.py) — adaptive
  eager/stream gate (dataset type → strategy → memory band).
- **`AnalysisPipeline.needs_full_stack`** flag (opt-out for temporal pipelines).
- `tests/test_streaming_analysis.py` (9 tests: gate, writer round-trip incl.
  forced-memmap, pipeline parity, measurements parity).

### Changed

- **The four analysis pipelines** (histogram, tear, spots, nuclei) stop
  `np.asarray`-ing the whole stack — they read frames lazily and stream label
  output via `run_planes_to_labels`. Spots/Nuclei stay sequential (segmenter not
  thread-safe). `AnalysisResult.label_masks` may now be a disk-backed lazy reader.
- **`compute_measurements`** ([results_engine.py](nd2studios/backend/results_engine.py))
  reads one frame per channel at a time instead of materializing all channels up
  front.
- **`AnalysisStage.commit_m`** persists masks frame-by-frame via `LabelStackWriter`
  (no whole-stack `np.ascontiguousarray`) and swaps the result to the persistent
  canonical readers.
- **Analysis / Results / Batch call sites** keep channels lazy and inject the
  streaming flag + a temp-scratch sink factory when the gate says to stream
  ([analysis_page.py](nd2studios/pages/analysis_page.py),
  [results_page.py](nd2studios/pages/results_page.py),
  [batch_worker.py](nd2studios/workers/batch_worker.py)); scratch is removed after
  the masks are committed/measured.

## [Unreleased] - 2026-06-10 (V1.45 Pipelines tab — Phase 1: Processing MVP)

### Added (V1.45.6 Pipelines node-board refinements)

- **Rename output nodes by clicking the name.** Clicking an OUTPUT node's name
  (on either the Processing or Analysis sub-tab) opens an inline `QLineEdit`
  editor over the title; Enter/click-away commits, Esc cancels. A drag still
  moves the node (the edit only triggers on a click that didn't move, armed in
  `NodeItem.mousePressEvent` / fired in `mouseReleaseEvent`). Committing calls
  the new `NodeScene.rename_node` → `node_renamed(node_id, title)` signal, which
  `PipelinesPage._on_node_renamed` uses to keep the node's **bridge name in
  sync** so the rename is reflected wherever the bridge is referenced (Analysis
  input / Export source labels). New `_NameLineEdit` (Escape-aware) +
  `NodeItem._begin/_commit/_cancel_name_edit`; QSS `#nodeNameEdit`
  ([node_item.py](nd2studios/widgets/node_board/node_item.py),
  [node_scene.py](nd2studios/widgets/node_board/node_scene.py),
  [pipelines_page.py](nd2studios/pages/pipelines_page.py),
  [theme.py](nd2studios/core/theme.py)). *(Renamable OUTPUT nodes consume the
  double-click for editing instead of preview-promotion; the primary Output is
  the default preview target regardless.)*

### Added

- **New "Pipelines" page** — a GA3-style node-graph editor that fuses the
  Recipe → Analysis → Results flow into one connected board. Registered in
  `Settings.PAGES` (after Results, before Batch) with an `"imported"` prereq in
  `Settings.PAGE_PREREQS`; wired into `MainWindow._build_content_area`'s
  `page_classes`. Layout is a draggable 50/50 horizontal `QSplitter`: a node
  board on the left, a reused `MultiAxisViewer` on the right
  ([pipelines_page.py](nd2studios/pages/pipelines_page.py)).
  **Phase 1 wires the Processing sub-tab end-to-end**; Analysis and Results
  render with a "coming soon" overlay (execution lands in V1.46+).
- **Qt-free graph core** `nd2studios/pipeline_graph/` (no PySide6, headless-
  testable): `model.py` (`PortType`/`Stage`/`NodeRole` enums; `Port`/`Node`/
  `Edge`/`GraphSlice`/`Bridge`/`PipelineDoc` dataclasses; `can_connect`,
  `would_create_cycle`, `clone_node`), `registry_adapter.py` (enumerates
  `PluginBase.get_plugins("enhancement")` into node specs — names read at
  runtime, never hardcoded; `build_node`, `param_specs_for`), `executor.py`
  (`recipe_for_node` linearizes any node's unique chain back to the input into a
  `List[(plugin_name, params)]`; `apply_recipe` mirrors `RecipeWorker` /
  `EnhancedDataset`), and `io.py` (`save_pipeline`/`load_pipeline` →
  `<name>.nd2s_pipeline.json`, `kind="nd2studios.pipeline"`, `schema_version=1`,
  mirroring `backend/recipes.py`).
- **Custom node editor** `nd2studios/widgets/node_board/` built on
  `QGraphicsScene` (no new dependency, per the project's dependency
  conservatism): `NodeItem` (rounded body + stage-accent lip + hover/selected
  **Disconnect / Duplicate / Delete** corner buttons with tooltips), `PortItem`
  (typed ports colored per spec §6.2, reject flash), `EdgeItem` (cubic-Bézier
  wire colored by source port type), `NodeScene` (type-checked drag-to-connect,
  DAG-enforced via `would_create_cycle`, one-wire-per-input, fan-out on outputs,
  right-click "Add operation" menu generated from the registry), and `ParamPopup`
  (frameless `Qt.Tool` window embedding the existing `widgets/common.py:
  ParamEditor`).
- **Branching-DAG Processing execution.** Each Processing output node is the
  unique path back to the input ⇒ its own recipe ⇒ its own `EnhancedDataset`.
  Live **Preview** (ON/OFF) runs `apply_recipe` on the selected node via a thin
  `_ProcessingPreviewJob(AnalysisJob)` submitted to `MainWindow.job_runner`
  (coalesce key `"pipeline_preview"`, ~300 ms debounce, `heavy_ops_blocked()`
  honored). **Apply** commits the primary output's recipe to `record.recipe` /
  `recipe_normalized`, sets `record._processed_view = EnhancedDataset(...)`,
  advances status to `preprocessed`, and best-effort commits via
  `MainWindow.recipe_stage()` so Export's "Processed Image" reflects the graph.
- **Pipeline serialization** — Save / Load buttons round-trip the full
  `PipelineDoc` (all three slices, node params/positions/enabled, edges, bridge
  registry) to `.nd2s_pipeline.json`.
- **Theme** — QSS for `#nodeBoard`, sub-tab selector, control-bar tool buttons,
  and the param popup frame ([theme.py](nd2studios/core/theme.py)).

### Notes

- Deferred to follow-ups: Results sub-tab execution, Export source-selector
  bridges, Undo/redo, batch-over-files, and current-frame / pyramid preview
  (Phase 1 materializes the full stack, matching the Recipe page).

### Added (V1.45.5 Pipelines — Analysis sub-tab)

- **Analysis sub-tab wired end-to-end.** Nodes come from
  `AnalysisPipeline.get_pipelines()` via new registry-adapter helpers
  (`analysis_specs`, `analysis_input_spec`, `analysis_output_spec`,
  `analysis_pipeline_name_for_op_key`; `param_specs_for` now resolves
  `"analysis:<name>"` op-keys). Each pipeline node is `Image → Binary` (label
  masks); the output node bridges `Analysis → Results` (`PortType.BINARY`).
  Names are read from the registry at runtime — never hardcoded.
- **Live overlay preview.** Selecting/previewing a pipeline node screens it on
  the current frame via `PipelinePreviewJob` (key `"pipeline_analysis_preview"`)
  and overlays the label masks with `MultiAxisViewer.set_frame_post_process` +
  `invalidate_post_process_cache` — the same path the Analysis page uses,
  reusing its `_overlay_labels`. A processed *base image* is shown underneath
  (`_submit_base_preview`, key `"pipeline_preview"`), and scrubbing T/M/Z
  re-screens via `coords_changed`. Channel `choice` params get live channel
  names injected on pop-up open, mirroring `analysis_page._load_params`.
- **Apply runs the pipeline on the current M's full stack** via
  `PipelineCommitJob` (key `"pipeline_analysis_commit"`), writing the
  `AnalysisResult` into `record.analysis_results[pipeline_name][m]` and showing
  the committed overlay (multi-M sequential commit deferred).
- **Page generalized to be stage-aware:** per-stage previewed node, highlight,
  preview-target and input-node creation; preview dispatches to processing vs
  analysis; the catalog **Add** dialog gains curated analysis descriptions and
  an "overlay shown live in the viewer" note for analysis nodes.

### Changed (V1.45.4 Pipelines node-board refinements)

- **Golden highlight is now upstream-only.** Previewing a node golds the node and
  the chain feeding *into* it (the pipeline that actually produces what the
  viewer shows) — nodes and bridges *downstream* of the previewed node no longer
  get the gold outline. `PipelinesPage._connected_subgraph` (undirected
  component) is replaced by `_upstream_subgraph`, which walks incoming edges back
  toward the input ([pipelines_page.py](nd2studios/pages/pipelines_page.py)).

### Bug Fixes (V1.45.3 Pipelines node-board refinements)

- **Old golden outline lingered when the previewed node changed.** The
  "previewed" node's thick (3 px) gold border strokes ~1.5 px *outside* the node
  body, but `NodeItem.boundingRect` returned the exact body rect — and an
  `update()` only repaints `boundingRect`, so the outer ring of the old outline
  was never cleared, leaving a gold ghost. `boundingRect` now pads by the widest
  pen, and a tight `shape()` override keeps mouse hit-testing on the visible body
  ([node_item.py](nd2studios/widgets/node_board/node_item.py)).

### Changed (V1.45.3 Pipelines node-board refinements)

- **Bridges route as rounded-corner orthogonal elbows.** When two connected
  nodes are offset laterally, the wire now drops out of the output, runs
  horizontally across the **gap between the nodes** at the midpoint, then drops
  into the input — with both bends drawn as **rounded corners** (radius clamped
  to half the shorter adjacent segment). Routing through the inter-node gap keeps
  the wire from ever overlapping either node body; when the anchors line up
  vertically it stays a straight drop with no corners
  ([edge_item.py](nd2studios/widgets/node_board/edge_item.py)).

### Changed (V1.45.2 Pipelines node-board refinements)

- **Sticky "previewed" node with a golden outline.** The viewer now tracks an
  explicit *previewed* node (no longer the current selection): it and every node
  + wire in its connected component are drawn with a gold outline
  (`Settings.ACCENT_GOLD = #ffc83d`; the previewed node thicker). **Double-click**
  a node to make it the previewed node; a **single click** only opens its
  parameter pop-up; clicking off the canvas (or selecting another node) leaves
  the preview and highlight unchanged. The previewed node falls back to the
  primary Output (else the Input) until one is chosen. New `NodeItem.set_highlighted`,
  `EdgeItem.set_highlighted`, `NodeScene.set_highlight`, and page helpers
  `_update_preview_highlight` / `_connected_subgraph`
  ([node_item.py](nd2studios/widgets/node_board/node_item.py),
  [edge_item.py](nd2studios/widgets/node_board/edge_item.py),
  [node_scene.py](nd2studios/widgets/node_board/node_scene.py),
  [pipelines_page.py](nd2studios/pages/pipelines_page.py)). *(Double-click no
  longer renames Input/Output nodes — that gesture is now the preview promotion.)*
- **Drag a node from any top anchor.** Input (top) ports are now
  mouse-transparent (`setAcceptedMouseButtons(NoButton)`), so pressing a top
  anchor falls through to the node body and moves the whole node; they remain
  valid connection *drop* targets. Output (bottom) ports still start a
  connection drag ([port_item.py](nd2studios/widgets/node_board/port_item.py)).
- **Bigger, richer Add dialog.** The catalog dialog is enlarged
  (≥ 900×640) with larger 210 px thumbnails; each node now shows a **lengthy**
  curated explanation (a full paragraph plus per-parameter notes) instead of the
  one-line plugin description, and the **example image is larger (240²) and far
  more comprehensive** — cell blobs across many sizes, large diffuse debris, a
  fine textured background, spatial illumination heterogeneity (vignette × a
  diagonal gradient), read noise, and scattered hot pixels — so every operation
  shows a visible effect ([add_node_dialog.py](nd2studios/widgets/node_board/add_node_dialog.py)).

### Changed (V1.45.1 Pipelines node-board refinements)

- **Vertical node flow.** Connection anchors moved from the left/right edges to
  the **top** (inputs) and **bottom** (outputs) edges so pipelines read
  top→bottom, and the stage-accent **lip** moved to the node's **left** edge
  (the top/bottom edges are now reserved for ports). `NodeItem` height is fixed
  (`_title_h + _body_pad`) instead of growing with the port count, and ports are
  spread evenly along each edge via `_port_x` ([node_item.py](nd2studios/widgets/node_board/node_item.py)).
- **Wires are now vertical Bézier curves** — `EdgeItem._rebuild` pushes the
  control points out along *y* (leave outputs downward, enter inputs from above)
  ([edge_item.py](nd2studios/widgets/node_board/edge_item.py)).
- **Double-click a wire to disconnect it.** `EdgeItem` is now interactive:
  hover thickens it + shows a "Double-click to disconnect" tooltip, the hit area
  is widened with a `QPainterPathStroker` (`shape()`), and a double-click calls
  the new `NodeScene.disconnect_edge(edge_id)` (removes the edge + emits
  `graph_changed`). The transient `"__temp__"` drag wire stays inert
  ([edge_item.py](nd2studios/widgets/node_board/edge_item.py),
  [node_scene.py](nd2studios/widgets/node_board/node_scene.py)).

### Added (V1.45.1 Pipelines node-board refinements)

- **Add button + node-catalog dialog.** A new **Add** button on the Processing
  control bar opens `AddNodeDialog`
  ([add_node_dialog.py](nd2studios/widgets/node_board/add_node_dialog.py)): a list
  of every node type with, per selection, a concise plain-language description
  (the plugin's `description` plus its tunable parameter labels) and a live
  **raw → processed** thumbnail pair — a synthetic microscopy-like image
  (`_synthetic_raw`: blobs + uneven illumination + noise) run through the real
  enhancement plugin at its default params, separated by an arrow. Double-click a
  row or press **Add node** to drop it; the page positions the node at the
  visible-canvas center with a small per-add cascade (`_new_node_pos`). The Add
  button is disabled on stages that don't allow adding (Analysis/Results).
- **Preview progress bar.** A slim `QProgressBar` sits next to the Preview /
  Apply / Undo buttons and animates while the pipeline is applied to the preview
  image — driven by the shared `JobRunner.job_progress` signal filtered to the
  `"pipeline_preview"` key. Shown on preview submit, advanced per channel/step,
  hidden on completion / preview-off / stage switch
  ([pipelines_page.py](nd2studios/pages/pipelines_page.py)).
- **Theme** — QSS for the preview progress bar and the add-node dialog
  (list, preview frame, thumbnails) ([theme.py](nd2studios/core/theme.py)).

## [Unreleased] - 2026-06-10 (V1.44 GUI Fixes — round 4)

### Bug Fixes

- **Processed image blank in the Analysis tab after running a recipe.**
  `_try_build_processed_volume` built the processed `MaterializedDataset` with
  4-D `(M, T, H, W)` channels, but the dataset (and its `get_frame`) require
  5-D `(M, T, Z, H, W)`. The Analysis viewer therefore mis-indexed the volume
  (`channels[m, t]` returned `(H, W)` instead of `(Z, H, W)`), rendering blank.
  Now a singleton Z axis is inserted and `n_zslices=1` is set
  ([recipe_page.py](nd2studios/pages/recipe_page.py)). (The Recipe tab looked
  fine because it displays the `(T,H,W)` channel dict directly, not the volume.)
- **"Force Lazy (cached)" had no effect.** The single-file ND2 load path
  computed the strategy decision but always called `materialize_nd2` (eager),
  ignoring both the decision and the user's override — so forcing lazy still
  materialized the whole file into RAM (and could OOM). `LoadWorker._load_nd2`
  now passes the forced override to `choose_strategy` and, when the strategy is
  `LAZY_CACHED`, opens a `LazyND2Volume` (on-demand frame reads) instead of
  materializing ([load_worker.py](nd2studios/workers/load_worker.py)). The
  forced override is read from `Settings.FORCED_LOAD_STRATEGY` via
  `_forced_strategy_override()`.

## [Unreleased] - 2026-06-10 (V1.44 GUI Fixes — round 3)

### Bug Fixes

- **Recipe / analysis progress bars now update per-frame.** `RecipeWorker`
  passed `progress_cb=None` to each plugin's `execute`, so the bar froze for the
  entire duration of a step. It now passes a per-frame callback that maps the
  plugin's 0–100 onto that step's T-frame block via `FrameProgress.set_done`,
  so the bar advances frame-by-frame through reads *and* steps
  ([recipe_worker.py](nd2studios/workers/recipe_worker.py)). (Analysis already
  forwarded a per-frame `progress_cb`; verified.)
- **Processed image / overlay now render on the cache-cold path.** The
  `_do_refresh` live-compose fallback skipped the post-process hook, so on the
  Analysis tab's `set_channels` / cold-cache path the processed image + overlay
  could fail to appear until the pre-render cache warmed. The fallback now
  applies the post-process overlay before painting
  ([multi_axis_viewer.py](nd2studios/widgets/multi_axis_viewer.py)).

### Changed

- **Shared control set now spans both raw and processed.** On the Recipe page
  the processed viewer's controls (zoom toolbar, frame strips, FPS, play,
  channels) are detached via `MultiAxisViewer.take_control_widgets()` into a
  single bar centered below *both* canvases, instead of sitting under only the
  processed viewer. The two canvases live in their own splitter that
  auto-equalizes their image widths ([recipe_page.py](nd2studios/pages/recipe_page.py)).

## [Unreleased] - 2026-06-10 (V1.44 GUI Fixes — round 2)

### Bug Fixes

- **Invisible combo/spin arrows** — `QComboBox` dropdown arrows and
  `QSpinBox`/`QDoubleSpinBox` up/down arrows showed nothing (CSS border-triangle
  / empty rules). Now rendered as crisp qtawesome chevrons to cached PNGs and
  injected into the stylesheet via `theme.build_stylesheet()` (called from
  `__main__`); arrows are visible and DPI-scaled.
- **Raw/processed separator sizing** — the splitter no longer mis-sizes; it
  auto-equalizes the two *images* (accounting for the shared LUT sidebar width),
  re-running on resize and on sidebar collapse (`_RecipeColumn.equalize`).
- **Slow/jerky zoom** — re-enabled smooth mouse-wheel zoom toward the cursor on
  the legacy `ImageCanvas` (was disabled); small 1.12×/notch steps.

### Changed

- **Single Play All** — removed the per-viewer Play All button and the old
  import-toolbar Play All. Added a universal `PlayAllBanner`
  ([play_all_banner.py](nd2studios/widgets/play_all_banner.py)): a centered
  play/pause + **FPS** banner spanning the viewer area, shown only when 2+
  files are loaded, that plays every viewer's T axis together.
- **"+ Add File" moved** from the import top toolbar into each file panel's left
  controls ([file_panel.py](nd2studios/widgets/file_panel.py) `on_add_file`).
- **Total-frame label** now matches the frame-counter font size and is
  vertically centered to it.
- **Axis order** T (top), M, Z (bottom).
- **Recipe raw + processed share one control set + one LUT sidebar.** The
  processed viewer is the master (frame strips, FPS, play, zoom, pan, channels,
  LUT+metadata sidebar); the raw viewer's controls/sidebar are hidden and it
  mirrors the master's coordinates, channel state, and zoom/pan
  (`MultiAxisViewer.set_controls_visible` / `mirror_from`; canvas
  `set_zoom_level`). Coordinate mirroring is emit-safe so it doesn't loop with
  the page's existing M-sync.

## [Unreleased] - 2026-06-10 (V1.44 GUI Fixes + Tile-Strip / LUT Polish)

Follow-up fixes after the GUI overhaul landed.

### Bug Fixes

- **GpuImageCanvas crash on navigation** — added `GpuImageCanvas.set_pixmap_direct`
  ([gpu_image_canvas.py](nd2studios/widgets/gpu_image_canvas.py)). The
  post-process overlay / playback cache hot-path called it on the GPU canvas
  (analysis page), which only the legacy canvas had → `AttributeError`. The GPU
  canvas now converts the pixmap to an RGB array and routes it through its
  composite layer.
- **Maximize button showed two icons** — `_toggle_max_restore`
  ([main_window.py](nd2studios/core/main_window.py)) set a text glyph on top of
  the qtawesome icon. It now swaps the icon (`window-maximize` ⇄ `window-restore`)
  and sets no text.

### Changed

- **Removed the per-page header bar** (page title + data-status badge) and the
  redundant "Import" section-header label; the top tab bar already labels the
  page. The data-status indicator is gone
  ([main_window.py](nd2studios/core/main_window.py),
  [import_page.py](nd2studios/pages/import_page.py)).
- **Axis rows reordered to T (top), M, Z (bottom)** in the multi-axis viewer.
- **Frame stepping is now smooth via arrow keys / click-drag.** The tile strip
  supports slider-style **left-drag scrubbing**
  ([frame_strip.py](nd2studios/widgets/frame_strip.py)), and strip current
  changes take a direct fast-refresh path (`_on_strip_current`) instead of the
  debounced slider cascade — M/Z stepping (was 80/50 ms debounce) is now as
  smooth as FPS playback ([multi_axis_viewer.py](nd2studios/widgets/multi_axis_viewer.py)).

### Added

- **"Play All" button** below the per-axis play/pause controls (same icon
  style). Sweeps every frame across T·M·Z (honoring per-axis tile selections)
  via `_set_play_all` / `_play_all_tick`
  ([multi_axis_viewer.py](nd2studios/widgets/multi_axis_viewer.py)).
- **`nd2studios/widgets/collapsible_sidebar.py`** (new) — `CollapsibleSidebar`
  wraps a panel with a collapse toggle matching the right LUT sidebar. The
  Recipe and Analysis left control panels are now collapsible.
- **Single right LUT+metadata sidebar on the Recipe page.** The raw viewer's
  sidebar is hidden; the processed viewer's LUT changes mirror onto the raw
  viewer, **scaled by data range** (`MultiAxisViewer.set_channel_contrast` /
  `lut_effective_max`) so a different processed bit depth (e.g. 16-bit processed
  vs 12-bit raw) still compares at matched intensities
  ([recipe_page.py](nd2studios/pages/recipe_page.py)).

## [Unreleased] - 2026-06-09 (V1.44 GUI Overhaul + Button Revamp)

Navigation moves from a left sidebar to a top tab bar; buttons become modern,
DPI-scaling qtawesome vector icons; file metadata is pinned at the bottom of the
right panel. Plan:
[CodeLog/ClaudesPlan/V1.44_gui_overhaul_buttons.md](CodeLog/ClaudesPlan/V1.44_gui_overhaul_buttons.md).

### Added

- **`nd2studios/widgets/icon_button.py`** (new) — DPI-aware icon button system.
  `ui_scale()` / `scaled(px)` derive a scale factor from the primary screen's
  logical DPI; `make_icon()` builds recolored `qtawesome` vector icons on the
  Dracula palette; `icon_button()` / `tool_button()` are DPI-scaled button
  factories; `bind_toggle_icon()` swaps a checkable button's icon on toggle
  (play ⇄ pause). qtawesome import is guarded (degrades to text if absent).
- **`qtawesome`** dependency (added to `requirements.txt`).
- **Top tab bar** ([main_window.py](nd2studios/core/main_window.py)
  `_build_top_tabs`) — horizontal `#topTabBar` under the title bar with page
  tabs (icon + text, accent underline when active) and session actions on the
  right. New QSS `#topTabBar` / `#topTabBtn` / `#sessionTabBtn` in
  [theme.py](nd2studios/core/theme.py).
- **Pinned metadata block** — `LutSidebar.set_metadata_text()` adds an
  always-visible `#metaBox` at the bottom of the right panel
  ([lut_sidebar.py](nd2studios/widgets/lut_sidebar.py)), fed by
  `MultiAxisViewer._update_axis_labels` with the current T/M/Z metadata.

### Changed

- **Navigation migrated from the left sidebar to the top tab bar.** The
  collapsible sidebar, its animation, and the `_sidebar*` / `_toggle_sidebar` /
  `_on_sidebar_anim_done` machinery were removed from
  [main_window.py](nd2studios/core/main_window.py). Page switching still flows
  through `_navigate` → `QStackedWidget`.
- **Title-bar Min/Max/Close** and **per-axis play buttons** now use qtawesome
  icons (`fa5s.window-minimize/maximize/times`, `fa5s.play`⇄`fa5s.pause`).
  `MultiAxisViewer._set_axis_playing` no longer swaps button text.

## [Unreleased] - 2026-06-09 (V1.43 Tile-Strip Frame Navigator)

A Nikon NIS-Elements-style rectangular tile strip replaces the per-axis slider,
adding frame selection, range/keyboard navigation, play-only-selected, and a
crop-to-selection that builds a new in-RAM dataset without touching the
original file. Plan:
[CodeLog/ClaudesPlan/V1.43_tile_strip_navigator.md](CodeLog/ClaudesPlan/V1.43_tile_strip_navigator.md).

### Added

- **`nd2studios/widgets/frame_strip.py`** (new) — `FrameStrip(QWidget)`, a
  single-`paintEvent` strip of rectangular frame tiles that fill the viewer
  width and **wrap to additional rows** below a DPI-scaled minimum tile width
  (≥ half a cursor). Current frame highlighted in the theme accent; selection in
  a translucent accent. Click selects; **Shift-click** selects an inclusive
  range from the current frame; **Ctrl-click** toggles a tile. Keyboard:
  `←`/`→` ±1, `Shift` ±5, `Ctrl` ±10. Right-click → Crop / Clear. Signals:
  `current_changed(int)`, `selection_changed(frozenset)`,
  `crop_requested(frozenset)`; per-frame tooltip via an injected `meta_fn`.
- **`MaterializedDataset.subset(*, m, t, z, progress_cb)`**
  ([materialized_dataset.py](nd2studios/backend/materialized_dataset.py)) —
  `np.ix_` advanced-indexing copy of the selected M/T/Z indices (H/W preserved).
  Returns a **new** dataset; the source is never mutated. Out-of-range/empty
  selections fall back to keeping the whole axis.
- **`nd2studios/workers/crop_worker.py`** (new) — `CropWorker(BaseWorker)` runs
  `subset` off the GUI thread with frame-accurate progress.
- **`MultiAxisViewer.set_frame_timestamps()`** and per-axis metadata tooltips
  (T: time imaged / Δt / total elapsed; M: pixel size / resolution / field of
  view; Z: µm step / height within Z range). New
  `crop_to_selection_requested(axis, frozenset)` signal.
- **"Revert to full data"** control on the import file panel
  ([file_panel.py](nd2studios/widgets/file_panel.py)) — restores the full
  dataset after a crop.

### Changed

- **`MultiAxisViewer`** ([multi_axis_viewer.py](nd2studios/widgets/multi_axis_viewer.py))
  — each axis's `QSlider` is now a *hidden backing model* parented to a visible
  `FrameStrip`; the two sync bidirectionally so all existing playback / value
  code is unchanged. `_configure_slider` drives the strip's count + visibility.
  Playback (`_axis_tick` → new `_next_playback_value`) loops **only the selected
  frames** when a tile selection exists. A **T crop automatically keeps all M and
  Z** (`subset(t=…)`); M-only / Z-only crops constrain just that axis.

## [Unreleased] - 2026-06-09 (V1.41 Frame-Accurate Progress)

Every long-running operation now reports progress against the file's *real*
total frame count (`T·M·Z·C`) instead of a T-only or phase-based estimate, so a
multi-position / multi-Z file no longer jumps to 100 % after the first position.
Part of the all-around QoL initiative
([CodeLog/ClaudesPlan/V1.41_pipeline_smoothness_resource_aware.md](CodeLog/ClaudesPlan/V1.41_pipeline_smoothness_resource_aware.md)).

### Added

- **`nd2studios/utils/progress.py`** (new) — `FrameProgress(total_frames,
  emit_cb, *, lo=0, hi=100)` progress accountant. `.advance(n=1)` counts
  processed frames and emits an integer percent **only when it changes** (1 %
  thresholds → at most 101 emissions regardless of frame count, so the GUI is
  never flooded). Optional output band `(lo, hi)` lets a worker reserve a slice
  of the bar for one phase. Helper `total_frames_for(meta, *, channels=...,
  z_collapsed=..., include_channels=...)` derives `T·M·Z·C` from an
  `ND2Metadata`-like object.

### Changed

- **Exporters now count real pages/frames** instead of `(t+1)/n_t*100`:
  [tiff_exporter.py](nd2studios/backend/exporters/tiff_exporter.py) counts
  `T·Z` (stack) / `T·Z·C` (hyperstack) pages;
  [movie_exporter.py](nd2studios/backend/exporters/movie_exporter.py) and
  [composite_exporter.py](nd2studios/backend/exporters/composite_exporter.py)
  count `T` composites (their input is already single-position, Z-projected);
  [image_sequence_exporter.py](nd2studios/backend/exporters/image_sequence_exporter.py)
  counts `T` (channels path) / `M·T·Z` (volume path) via `FrameProgress`.
- **`RecipeWorker`** ([recipe_worker.py](nd2studios/workers/recipe_worker.py))
  — replaced phase accounting with frame-equivalent counting: total =
  `n_channels · T · (read + normalize? + n_steps)`. Frame reads advance one at a
  time (so the bar tracks slow disk I/O), each recipe step credits a full
  `T`-block on completion.
- **`ExportWorker._export_tiff_zstack`**
  ([export_worker.py](nd2studios/workers/export_worker.py)) — the Z-stack read
  loop now counts every `C·T·Z` plane in the 0–80 % band (was T-only and assumed
  a single position) before the hyperstack write fills 80–100 %.
- **`PreRenderWorker`** ([pre_render_worker.py](nd2studios/workers/pre_render_worker.py))
  — converted its `M·T` progress to `FrameProgress` for consistent 1 % throttling.

### Bug Fixes

- **Export page no longer freezes the GUI when changing the TIFF z-projection
  mode at export time.** `_on_export_tiff`
  ([export_page.py](nd2studios/pages/export_page.py)) kept the re-projected
  channels lazy instead of calling `.materialize()` on the GUI thread; the heavy
  per-frame read now happens inside `ExportWorker` via `np.asarray`.

## [Unreleased] - 2026-06-09 (V1.42 Viewer Optimizations from Industry Comparison)

Five techniques borrowed from popular microscopy viewers (Bio-Formats
Memoizer, napari, BigDataViewer) after a comparative research pass.

### Added

- **`read_or_cache_nd2_metadata(filepath)`** in
  [nd2studios/backend/nd2_loader.py](nd2studios/backend/nd2_loader.py)
  — Bio-Formats Memoizer-style sidecar.  On first open, writes
  `<file>.nd2idx.json` next to the ND2 with the full result of
  `read_nd2_metadata_extended`.  Subsequent opens load the sidecar
  when its mtime is ≥ the source ND2's, skipping the multi-second
  chunk scan.  Stale or wrong-version sidecars are silently
  regenerated.  Wired into [load_worker.py](nd2studios/workers/load_worker.py)
  `_load_nd2`.

- **`LazyND2Channel` LRU read cache** — BigDataViewer `CellCache`
  pattern.  New `cache_max_bytes` constructor arg (default `0` =
  disabled) plus `configure_cache()` / `clear_cache()` methods.
  When enabled, `_read_frame` consults an `OrderedDict[t_local]`
  before hitting nd2/Dask; LRU eviction respects the byte budget.
  Read-only flag set on cached arrays so callers can't accidentally
  mutate them.

- **`MultiAxisViewer.attach_pyramid(reader)`** +
  **`_choose_pyramid_level()`** — BigDataViewer mipmap pattern.
  Stores a `PyramidReader`; `_read_volume_plane` short-circuits to
  `reader.get_frame(level, …)` when the canvas's viewport zoom would
  benefit from a coarser level.  Uses the existing
  [PyramidReader.pick_level_for_viewport](nd2studios/pipeline/stages/pyramid_stage.py)
  helper that was already there but not wired into the view path.
  Effect: zoom-out on large mosaics reads small arrays from the
  zarr pyramid instead of pushing the full level-0 plane to the
  GPU canvas.

- **`nd2studios/utils/request_queue.py`** (new) — cancel-first
  `SingleSlotMailbox` and `PriorityLanes` primitives borrowed from
  napari's NAP-4 async slicer.  Mailbox holds one in-flight
  `RequestToken`; submitting a new one cancels the previous.
  Workers cooperate by polling `token.cancelled` at known
  checkpoints.  `PriorityLanes` adds a background lane that's
  cancelled wholesale when foreground work needs the device.
  Utility ships in this release; consumers (the existing pre-render
  worker, future async slice loader) opt in incrementally.

- **CodeLog plan** —
  `CodeLog/ClaudesPlan/V1.42_viewer_optimizations.md`.

### Changed

- **`MultiAxisViewer._invalidate_render_cache(*, lut_only=False)`**
  — V1.42 napari shader-LUT pattern.  When `lut_only=True`:
  - On GPU canvas (`USE_GPU_DISPLAY=True`): completely no-op.
    `GpuImageCanvas` re-applies LUT/levels as a pyqtgraph uniform
    on every paint, so the existing per-(m,t) cache is never read
    and there is nothing to invalidate.
  - On CPU canvas: keep the existing cache dicts; restart the
    pre-render worker which overwrites entries in-place as it
    visits them.  User sees brief stale-LUT frames during
    scrubbing instead of dropping all the way to the slow
    live-compose path while the worker rebuilds.
  Bumps `_pp_version` so overlay caches re-render on demand.
  `_rebuild_render_cache_after_lut` (the 250 ms debounce target)
  now calls `_invalidate_render_cache(lut_only=True)`; channel
  enable / color / Z-mode changes keep the full
  `_invalidate_render_cache()` semantics.

- **`LazyND2Channel.crop`** — propagates new LRU fields to the
  cropped view so `_read_frame` overrides keep working.

### Architecture

See [CodeLog/Architecture/ARCHITECTURE.md](CodeLog/Architecture/ARCHITECTURE.md)
V1.42 section for how the five additions fit into the existing
viewer / loader / pyramid pipeline.

## [Unreleased] - 2026-06-09 (V1.41 Pipeline Smoothness + Resource-Aware Loading)

### Added

- **`nd2studios/utils/perf.py`** — lightweight CSV perf instrumentation
  gated by `ND2_PERF_LOG=1`. Exports `perf_log` decorator, `perf_block`
  context manager, and `log_event` for ad-hoc rows. When disabled the
  decorator returns the original function unchanged so there is zero
  call-site overhead in normal operation. Output appends to
  `~/.nd2studios/perf.csv` with `(timestamp_ns, event, duration_us,
  frame_idx, cache_hit, extra)` columns. Annotated the real playback
  hot paths in `MultiAxisViewer._do_refresh`, `_axis_tick`, and
  `_compose_current_frame` so cache hit / live-compose timings can be
  diff'd between baseline and after each optimisation.

- **`nd2studios/utils/resource_strategy.py`** — pre-flight
  `choose_strategy(meta, path)` that picks between `EAGER_FULL`,
  `EAGER_REDUCED`, and `LAZY_CACHED` based on
  `psutil.virtual_memory().available` and the dataset footprint.
  `StrategyDecision` dataclass exposes the inputs and the human-readable
  reason. Refuses loads whose projected footprint exceeds 2× available
  RAM via `StrategyError`. Test seam: `ND2_FAKE_RAM_GB` env var
  overrides the available-RAM reading.

- **`nd2studios/core/memory_monitor.py`** — banded
  `MemoryMonitor(QObject)` singleton polling `virtual_memory().percent`
  on a `QTimer`. Signals `warning` (80%), `critical` (90%, drops the
  pre-render cache), `emergency` (95%, modal + heavy-op block), and
  `recovered` (back to NORMAL). Hysteresis at 75/85/90 to prevent
  signal oscillation across jittering boundaries.

- **`nd2studios/utils/user_config.py`** — JSON persistence for the
  tracked subset of `Settings` (GPU flags, pyramid flag, eager
  fraction, memory pressure thresholds, forced load strategy). Path:
  `%APPDATA%\nd2studios\preferences.json` on Windows or
  `~/.config/nd2studios/preferences.json` elsewhere. Loaded by
  `__main__` before any UI construction; saved by the Performance
  dialog on Accept.

- **`Settings.EAGER_MAX_FRACTION`, `MEMORY_RESERVE_OVERHEAD`,
  `MEMORY_PRESSURE_*`, `MEMORY_MONITOR_INTERVAL_MS`,
  `FORCED_LOAD_STRATEGY`** — new module-level constants on
  `nd2studios/core/settings.py` driving the resource-aware load /
  memory monitor subsystems. Overridable via the Performance dialog
  and persisted by `user_config`.

- **`LoadWorker.strategy_chosen = Signal(object)`** — emitted between
  metadata read and materialize so the GUI can show the chosen
  `StrategyDecision`. The decision is also returned under the
  `"strategy"` key on the finished payload.

- **Memory gauge in bottom bar** — `_memory_gauge` `QLabel` in
  `MainWindow._build_content_area` shows `RAM 9.6 / 31.7 GB (30%)`
  and changes colour through yellow / orange / red at the pressure
  thresholds.

- **Performance dialog tabs** — `_open_performance_settings` rebuilt
  as a `QTabWidget` with GPU, Memory, Storage, and Diagnostics tabs.
  Memory tab exposes a `LoadStrategy` override dropdown (Auto / Force
  Eager / Force Eager Z-collapsed / Force Lazy). Diagnostics tab has
  a "Save snapshot…" button that writes a JSON capturing the current
  detection state.

- **Streaming TIFF + movie exports** — `export_tiff_stack`,
  `export_tiff_hyperstack`, and `export_movie` now stream pages /
  frames via `tifffile.imwrite(data=iter, shape=..., dtype=...)` and
  `imageio.get_writer(...).append_data(...)`. RAM peak during export
  drops from full-stack/list to single-frame.
  `_percentile_bounds_subsample` computes per-channel percentile
  bounds from a 1M-pixel subsample so rescale modes no longer require
  the full-stack `astype(np.float32)` copy.

- **Proactive GPU VRAM guard** — `should_dispatch_to_gpu(array, *,
  op="", safety_factor=2.5)` now consults a 200 ms-cached
  `cp.cuda.runtime.memGetInfo()` and falls back to CPU before
  dispatch when the array (× safety factor) doesn't fit. Per-op
  warn-once via `_vram_warn_once` mirrors the existing `_WARNED`
  pattern from `ops.py`.

- **Windows storage probe** — `windows_is_fast_storage(path)` reads
  16 MB from the target file, classifies ≥400 MB/s as "fast", and
  caches the verdict per drive letter in
  `%APPDATA%\nd2studios\storage_probe.json`. Wired through
  `is_fast_storage` so existing callers benefit automatically.

- **Startup diagnostic sequence** — `__main__` now logs RAM/CPU
  (`log_system_resources()`) and default storage class
  (`log_default_storage_class()`) immediately after the GPU status
  line, then auto-disables `BUILD_PYRAMIDS` on <8 GB-available
  machines and `USE_GPU_DISPLAY` on <2 GB-VRAM cards so first-launch
  defaults match the host.

- **CodeLog plan doc** — `CodeLog/ClaudesPlan/V1.41_pipeline_smoothness_resource_aware.md`.

### Changed

- **`MultiAxisViewer._do_refresh` cache-hit marking** — sets
  `_last_cache_hit` at each fast-path branch so the V1.41 perf log
  records hit/miss explicitly. Pure metadata; no functional change.

- **`movie_exporter.export_movie`** — pre-rendered frame list replaced
  with a per-frame generator feeding the codec writer. MP4 path now
  downscales one frame at a time after computing scale parameters
  from the first frame only. GIF / TIF / fallback codecs use
  `imageio.get_writer(...).append_data(...)` instead of `mimsave`.

- **`export_tiff_hyperstack`** — the `(T, Z, C, H, W)` `np.stack`
  intermediate is gone; pages flow through
  `tifffile.imwrite(data=_pages(), shape=..., imagej=True)` directly.
  BigTIFF threshold is now computed from the projected output
  footprint rather than the source ndarray nbytes.

- **`compute.gpu.ops`** — every `should_dispatch_to_gpu(image)` call
  now passes `op="<name>"` so VRAM-guard warnings name the op that
  fell back.

- **Performance dialog** — tabbed redesign; OK now also persists the
  tracked subset of `Settings` via `save_user_preferences()`.

### Removed

- **`nd2studios/widgets/video_player.py`** — `VideoPlayer` was never
  instantiated anywhere in the codebase. The real playback path lives
  in `MultiAxisViewer._axis_tick`. Deleted along with its 145 lines of
  dead code; perf instrumentation now annotates the actual hot path.

### Notes

- The `EAGER_REDUCED` and `LAZY_CACHED` strategy branches in
  `materialize_nd2` / `materialize_from_volume` are **deferred** to
  V1.42. Today the strategy budgeter is a pre-flight gate: it logs
  the chosen strategy, emits the signal, and raises `StrategyError`
  when the load definitely won't fit. Files that fit eager but would
  have benefited from Z-collapse still take the eager path until the
  branch lands.

- **WSL2 integration** is also deferred per the V1.41 plan. The two
  cuCIM-only ops (`gaussian` and `threshold_otsu` in tear detection)
  continue to fall back to CPU on Windows.

- The `MainWindow.heavy_ops_blocked()` flag set by the emergency-band
  modal is a soft block — it's a public accessor that pages can
  consult before launching new heavy workers. The check isn't wired
  into every button click site yet; V1.42 will add per-entrypoint
  guards.

## [Unreleased] - 2026-05-27 (V1.40 Mask Analysis — Custom Mask Creator)

### Added

- **Custom Mask Creator panel** (`nd2studios/pages/analysis_page.py`) — new
  `QGroupBox` visible whenever the Mask Analysis pipeline is selected.
  Controls: shape type (Rectangle Strip), direction (Vertical / Horizontal),
  strip height in px, optional custom width + X offset, start offset in px.
  **Generate Masks** computes all complete strips from the spec and stores
  them in `self._tiled_masks`; a count label shows "N strips per frame".
  **Clear Tiled Masks** resets the state. Both buttons invalidate the raster
  cache and trigger an immediate overlay refresh so the user sees the yellow
  bands before running the pipeline.

- **`generate_tiled_label_boxes(tiled_masks, H, W)`**
  (`nd2studios/backend/analysis/manual_mask.py`) — pure-Python helper that
  converts a list of tiled-mask spec dicts into a list of
  `(y0, y1, x0, x1)` bounding boxes, one per complete strip. Strips that
  would extend past the image boundary are excluded. Supports
  `direction="vertical"` (tiles in Y) and `direction="horizontal"` (tiles
  in X). Spec keys: `shape_type`, `direction`, `height_px`, `width_px`
  (`None` = full image width), `x_offset_px`, `start_offset_px`.

- **`generate_tiled_mask(tiled_masks, H, W)`**
  (`nd2studios/backend/analysis/manual_mask.py`) — returns a `(H, W)` int32
  array with label IDs 1..N for tiled strips; convenience wrapper for the
  live-overlay preview path.

- **`ND2StudiosRecord.tiled_mask_specs`**
  (`nd2studios/core/experiment_manager.py`) — new serialized field
  (`List[Dict[str, Any]]`) that persists Custom Mask Creator specs across
  session save/load. Deserialized from `"tiled_mask_specs"` key in the .nd2s
  manifest with no migration needed (defaults to empty list).

### Changed

- **"Manual Mask" pipeline renamed to "Mask Analysis"**
  (`nd2studios/backend/analysis/manual_mask.py`,
  `nd2studios/pages/analysis_page.py`) — `ManualMaskPipeline.name` is now
  `"Mask Analysis"`; `MANUAL_MASK_PIPELINE_NAME` constant updated to match.
  All runtime references use the constant so no hardcoded strings remain.

- **`ManualMaskPipeline.run()`** now applies tiled strips (label IDs 1..N,
  same in every frame) before hand-drawn shapes (offset IDs N+1, N+2, …).
  This keeps each tiled region a stable object across all timepoints so
  per-frame ΔArea and delta columns in the Results tab are meaningful.

- **`rasterize_shapes(frame, shapes, H, W, label_offset=0)`** — new optional
  `label_offset` parameter so the drawing overlay assigns IDs starting at
  `label_offset + 1`, preventing collision with tiled-strip label IDs.

- **`_composite_overlay()`** (Analysis page) extended: when Mask Analysis is
  active and `_tiled_masks` is non-empty, tiled strips are painted first into
  the live mask, then drawn shapes on top with the correct offset.

## [Unreleased] - 2026-05-26 (V1.0 macro-recorder)

### Added

- **`nd2studios/backend/macro_engine.py`** — NEW backend module (no Qt). Defines
  `MacroAction` dataclass (`action_type`, `label`, `params: dict`, `enabled: bool`,
  `timestamp: float`) and `MacroRecorder` (start / pause / resume / cancel /
  finish). `save_macro(name, actions, path)` and `load_macro(path)` serialize to
  `.nd2s_macro.json`. `list_macros(directory)` enumerates macro files.

- **`nd2studios/widgets/macro_dialog.py`** — Non-modal `MacroDialog(QDialog)` with
  three panels managed by a `QStackedWidget`:
  - **Manage** — create (with name) or open an existing `.nd2s_macro.json`.
  - **Record** — live feed of recorded actions with animated ● REC indicator;
    Pause/Resume toggle, Cancel, and Finish buttons. Finish prompts for a save path.
  - **Edit** — drag-to-reorder list of `_ActionRowWidget` blocks. Each block has
    an enabled checkbox, label, **Edit** button (opens param key-value editor), and
    Delete (✕) button. **Run Macro** button replays all enabled actions on the
    currently loaded file. **Save As…** persists edits.

- **`MainWindow.macro_action_recorded = Signal(dict)`** — class-level PySide6
  signal emitted whenever a macro action is captured. Consumed by `MacroDialog`
  to update its live-record list without polling.

- **`MainWindow.macro_recorder: MacroRecorder`** — per-session recorder instance
  initialized in `__init__`.

- **`MainWindow._open_macro()`**, **`record_macro_action(action)`**,
  **`replay_macro_action(action)`** — sidebar button handler, recording helper
  (emits `macro_action_recorded`), and action dispatcher for replay (routes to
  `recipe._replay_add_step`, `analysis._replay_run`, or `export._replay_export`).

- **Sidebar Macro button** (`🎬  Macro`) added to the bottom control group in
  `MainWindow._build_sidebar()`. Opens / raises the `MacroDialog`.

- **Recipe page recording hooks** (`nd2studios/pages/recipe_page.py`):
  - `_record(action_type, label, **params)` helper forwards to
    `main_window.record_macro_action`.
  - `_on_accept()` records `recipe_add_step` with `plugin`, `plugin_params`,
    `normalized`.
  - `_on_remove_last()` records `recipe_remove_last`.
  - `_on_clear()` records `recipe_clear`.
  - `_replay_add_step(action)` sets `combo_plugin`, applies params, calls
    `_on_trial()` for replay.

- **Analysis page recording hook** (`nd2studios/pages/analysis_page.py`):
  - `_record_macro_action(action_type, label, **params)` helper.
  - `_on_run()` records `analysis_run` with `pipeline` and `pipeline_params`.
  - `_replay_run(action)` sets pipeline + params, calls `_on_run()` for replay.

- **Export page recording hooks** (`nd2studios/pages/export_page.py`):
  - `_record_macro(action_type, label, **params)` helper.
  - `_on_export_tiff()` records `export_tiff` (bit_depth).
  - `_on_export_composite()` records `export_composite`.
  - `_on_export_movie()` records `export_movie` (fps, format, scale_bar flags, etc.).
  - `_replay_export(action)` restores widget state and triggers the matching export slot.

- **Batch page Macro section** (`nd2studios/pages/batch_page.py`):
  - New `QGroupBox("Macro")` with macro file picker + "Run macro instead of
    template" checkbox.
  - `_on_run_macro_batch()` — iterates the file list; for each file, loads it via
    the Import page then calls `main_window.replay_macro_action()` for every
    enabled action.
  - `save_to_experiment` / `load_from_experiment` persist `macro_path` and
    `use_macro` in `exp.batch_config`.

---

## [Unreleased] - 2026-05-26 (V1.0 tracking-shape-filter)

### Added

- **`nd2studios/backend/results_engine.py`** — `circularity` field added to
  every measurement row: `4π·area/perimeter²` (range 0–1; 1 = perfect circle;
  0 if perimeter is zero).  Appears in the Results table under the Shape column
  group.

### Changed

- **`nd2studios/backend/object_tracker.py`** — `link_objects()` gains two new
  shape-filter parameters:
  - `min_circularity: float = 0.0` — objects whose circularity falls below this
    value are excluded from tracking; they keep `track_id = None`.
  - `max_eccentricity: float = 1.0` — objects whose eccentricity exceeds this
    value are excluded from tracking.
  Both filters default to disabled (pass everything).

  **Bug fix** — bounding-box overlap check added to `_link_group`.  Before
  accepting a Hungarian-assignment pair, the linker now verifies that the two
  objects' bounding boxes actually intersect spatially.  Pairs with no bbox
  overlap receive a cost above `max_displacement_px` and are therefore never
  linked, preventing unrelated objects that happen to be within the centroid
  displacement threshold from being incorrectly joined into the same track.
  The `_bbox()` helper extracts `bbox_min/max_row/col` from a measurement row
  and expands degenerate (zero-size) bboxes to a 1-px centroid-centred region
  so the check is non-blocking for edge cases.

- **`nd2studios/pages/results_page.py`** — Object Tracking row gains two new
  `QDoubleSpinBox` controls:
  - *Min. circularity* (0.00–1.00, step 0.05, default 0.00 = disabled).
  - *Max. eccentricity* (0.00–1.00, step 0.05, default 1.00 = disabled).
  Both values are forwarded to `link_objects()` on every Compute Measurements
  click.  The `circularity` column is added to the Shape group in
  `_COLUMN_GROUPS` and to `_DEFAULT_COLUMNS`.

---

## [Unreleased] - 2026-05-26 (V1.0 export-tab-restructure)

### Added

- **`nd2studios/backend/exporters/tracked_objects_exporter.py`** — NEW backend
  module.  `export_tracked_objects(measurements, label_masks, channels,
  channel_display, output_dir, objects_per_m, with_image, with_mask_overlay,
  fmt, progress_cb)` renders each tracked object as a 3× crop (matching the
  validation dialog), tiles `objects_per_m` crops side-by-side per output file,
  and writes `tracked_objects_M000.tif`, `tracked_objects_M001.tif`, … (or a
  numbered PNG series for multi-frame PNG output).  Returns the list of written
  paths.

### Changed

- **`nd2studios/core/settings.py`** — Export page moved to the last position in
  `PAGES`; new order is Import → Recipe → Analysis → Results → Batch → Export.

- **`nd2studios/pages/export_page.py`** — complete UI restructure:
  - The `QGroupBox("Source")` with `rb_raw` / `rb_proc` radio buttons is
    replaced by a single `combo_export_type` (`QComboBox`) at the top of the
    page with three options: **Raw Image**, **Processed Image**, and **Tracked
    Objects**.  Selecting Raw or Processed shows the existing image-export tab
    widget; selecting Tracked Objects shows the new tracked-objects panel.
  - The page body is now a `QStackedWidget` (index 0: image tabs, index 1:
    tracked objects panel).
  - **Tracked Objects panel** — new panel with pipeline selector
    (`combo_tracked_pipeline`, populated on page activation from available
    segmentation channels in `ResultsPage._measurements`), *Objects per frame*
    spinbox (1–100, default 1), *Include image channels* checkbox, *Overlay
    mask highlight* checkbox, format selector (TIFF / PNG), and an
    *Export Tracked Objects…* button that calls `export_tracked_objects`.
  - `on_activated()` now enables/disables the *Processed Image* combo option
    (instead of the old radio button) and refreshes the pipeline selector.
  - `_channels_and_state()` now reads `combo_export_type.currentText()` instead
    of checking `rb_proc.isChecked()`.

---

## [Unreleased] - 2026-05-26 (V1.0 tracked-object-validation-v3)

### Added

- **`nd2studios/widgets/track_validation_dialog.py`** — **Accept All** and
  **Reject All** buttons in the bottom bar.  Each applies its decision to
  every undecided panel in the current batch, then triggers the 400 ms
  visual-confirmation delay before loading the next batch.

### Changed

- **`nd2studios/backend/object_tracker.py`** — `link_objects()` gains a
  `min_track_length: int = 2` parameter.  Tracks whose frame count falls
  below this threshold receive `track_id = None` and are excluded from
  validation.  The threshold is clamped to ≥ 1 internally.

- **`nd2studios/pages/results_page.py`** — the tracking controls are now
  presented as a dedicated **Object Tracking** parameter row (below the
  main controls row) with two labelled spinboxes:
  - *Link distance (px)* — max centroid displacement between frames
    (default 100; was `Max disp. (px)` in the controls row).
  - *Min. track length (frames)* — minimum consecutive frames for a valid
    track (default 2).
  Both values are forwarded to `link_objects()` on every Compute Measurements
  click.

---

## [Unreleased] - 2026-05-26 (V1.0 tracked-object-validation-v2)

### Changed

- **`nd2studios/widgets/track_validation_dialog.py`** — complete redesign of
  the validation dialog:
  - **3× crop** — `_CROP_FACTOR = 3.0`; crop now spans center ± 1.5× object
    bbox half-extent (previously 2×, i.e., ± 1× half-extent).
  - **1/2/4-up mode selector** — three toggle buttons at the top-right
    ("1", "2", "4") switch between single, side-by-side, and 2×2 quartet
    layouts.  Mode can be changed at any time; `_batch_start` is
    realigned to a clean batch boundary for the new mode size.
  - **`_TrackPanel` inner widget** — each slot in the grid is an independent
    `_TrackPanel(QWidget)` with its own `ImageCanvas`, track-info label,
    and per-track **Accept** / **Reject** buttons.  After a decision the
    panel's border turns green (accepted) or red (rejected) and its buttons
    are disabled.
  - **Batch auto-advance** — once every active panel in the current batch has
    been decided, the dialog waits 400 ms (so the user can see the final
    colours) then loads the next batch.  The dialog closes (`exec()` returns)
    when all tracked objects have been reviewed.
  - **Shared T controls with "▶ Play All / ⏸ Pause All"** — a single T
    slider, T spinbox, and FPS spinbox drive all visible panels in sync.
    Play / Pause button renamed "Play All" / "Pause All".
  - Previously-made decisions are preserved when switching view mode.

---

## [Unreleased] - 2026-05-26 (V1.0 tracked-object-validation)

### Added

- **`nd2studios/backend/object_tracker.py`** (NEW) — `link_objects(rows,
  max_displacement_px=100.0)` runs a frame-to-frame Hungarian linker
  (`scipy.optimize.linear_sum_assignment`) on the measurement list returned by
  `compute_measurements()`.  Groups rows by `(segmentation_channel,
  m_position)`, matches objects between consecutive frames whose centroid
  Euclidean distance is ≤ `max_displacement_px`, and adds three columns to
  every dict: `track_id` (int or None), `track_length` (int), and
  `track_validation` ("unvalidated" / None).  Tracks spanning only one frame
  receive `track_id = None`.

- **`nd2studios/widgets/track_validation_dialog.py`** (NEW) —
  `TrackValidationDialog(QDialog)` opens at 90 % of primary screen size and
  iterates through unvalidated tracked objects one at a time.  For each track
  it pre-renders all T frames as a cropped composite (2× the union bounding box,
  clamped to FOV) with a yellow-orange mask highlight.  Controls: T slider,
  play/pause toggle, FPS spinbox (0.5–60).  Buttons: **Accept** (keeps all
  rows for that track), **Reject** (all corresponding rows removed from the
  table), **Close** (exits early).  After every decision the dialog
  auto-advances to the next unvalidated track.

- **`nd2studios/pages/results_page.py`** — "Tracking" column group added to
  the column selector sidebar with columns `track_id`, `track_length`, and
  `track_validation`; all three are on by default.

- **`nd2studios/pages/results_page.py`** — `Max disp. (px)` `QSpinBox`
  (range 1–9999, default 100) in the top controls row; its value is forwarded
  to `link_objects` on every Compute Measurements click.

- **`nd2studios/pages/results_page.py`** — "Validate Tracked Objects"
  `QPushButton` in the export row; enabled only after at least one
  multi-frame track is found.  Clicking opens `TrackValidationDialog`; on
  close, rejected tracks' rows are removed from `self._measurements` and the
  table refreshes.

- **`CodeLog/ClaudesPlan/V1.0_tracked_object_validation.md`** — permanent
  record of design decisions, algorithm choice, and V1 scope limitations.

---

## [Unreleased] - 2026-05-26 (V1.0 recipe-page-fixes)

### Added

- **`nd2studios/pages/recipe_page.py`** — `▶ Play All` / `⏸ Pause` toggle button
  in the trial-controls area, wired to `viewer_proc.set_t_playing()`.  Enabled
  after a trial completes; reset on Reject / Clear All.  Mirrors the `>` play
  button already in the slider row but gives it a dedicated, prominently-placed
  control.

### Changed

- **`nd2studios/pages/analysis_page.py`** — `on_activated` and
  `_reload_for_selected_record` now prefer `exp._processed_channels` over
  `exp._raw_volume` when a recipe has been accepted.  The Analysis viewer
  therefore shows processed data automatically upon tab entry, matching what the
  analysis pipeline will actually receive.  Raw data remains intact on the record.

### Bug Fixes

- **`nd2studios/pages/recipe_page.py`** — `_on_trial_done_primary`: the global
  progress bar was left at 100 % after a trial completed because `RecipeWorker`
  emits `progress(100)` on finish but nothing reset it.  Now calls
  `main_window.set_progress(0)` and sets a "Trial complete — Accept or Reject."
  status message.
- **`nd2studios/pages/recipe_page.py`** — `_on_error`: also resets the progress
  bar to 0 and sets a "Recipe failed." status message on worker errors.
- **`nd2studios/widgets/multi_axis_viewer.py`** — `set_channels`: 2-D `(H, W)`
  single-frame channels were treated as `T = H` frames, configuring the slider
  with `H` positions and making `data[t]` return a 1-D row rather than a 2-D
  frame (renders nothing).  Fixed: 2-D inputs are now treated as `T = 1`.
- **`nd2studios/widgets/multi_axis_viewer.py`** — `set_channels`: the T slider
  was always reset to frame 0 when new processed data arrived (e.g. after a
  trial on a different M position).  Now preserves the current T position when
  the new data has at least that many frames, so the user stays on the frame
  they were inspecting.
- **`nd2studios/widgets/multi_axis_viewer.py`** — `_compose_current_frame` and
  `_render_current_frame_gpu`: when a channel array is 2-D `(H, W)`, index it
  as a single frame instead of silently swallowing an IndexError and rendering
  nothing.

## [Unreleased] - 2026-05-25 (V1.0 results-column-selector)

### Added

- **`nd2studios/widgets/config_wizard.py`** — new module.
  `ConfigWizard(QDialog)`: two-step guided wizard (Analysis + Results nav
  sidebar + `QStackedWidget`). Analysis panel has a `QComboBox` for pipeline
  selection and a `ParamEditor`; Results panel has the same flat checkbox groups
  as the results sidebar, live-synced to `results_page._col_checkboxes`. Cancel
  restores original checkbox states; Accept applies all changes to the live page.
  `apply_config(cfg)` pre-populates the wizard from a saved dict.
  `get_config()` returns a serialisable dict.
- **`nd2studios/widgets/config_wizard.py`** — `SavePreviewDialog(QDialog)`:
  read-only diff view (Analysis / Results nav) showing lines colour-coded green
  (`+`), red (`−`), orange (`→`). Browse button lets the user change the save
  path before confirming.
- **`nd2studios/widgets/config_wizard.py`** — `build_config_from_pages`,
  `apply_config_to_pages`, `build_config_diff` helpers; `CONFIG_EXTENSION =
  ".nd2s_cfg"` constant.
- **`nd2studios/pages/results_page.py`** — `_COLUMN_GROUPS` catalogue: 6 static
  groups (Identity, Size, Change Δ, Position, Shape, Bounding box) with all
  column keys and display labels; `_DEFAULT_COLUMNS` set of keys shown on first
  open.
- **`nd2studios/pages/results_page.py`** — `ResultsPage._build_columns_sidebar()`:
  permanent left panel (flat `QVBoxLayout`, every checkbox and section-header
  label at `_CB_HEIGHT = 24` px) with an initially-hidden Intensity group. "All"
  / "None" buttons at the bottom.
- **`nd2studios/pages/results_page.py`** — `ResultsPage._sync_intensity_columns(rows)`:
  dynamically registers `mean_intensity_*` / `std_intensity_*` checkboxes into
  the Intensity group on first `_on_compute` call; shows the group on first add.
- **`nd2studios/core/main_window.py`** — `_configure()`: opens `ConfigWizard`
  (pre-populated from `self._cfg_data` if a config file has been loaded/saved).
- **`nd2studios/core/main_window.py`** — `_save_config()`: if no baseline config
  exists, opens `ConfigWizard` then prompts for a file path; otherwise opens
  `SavePreviewDialog` with a diff against the baseline and writes the new config
  on Accept. Updates `self._cfg_path` and `self._cfg_data`.
- **`nd2studios/core/main_window.py`** — `_load_config()`: opens a `QFileDialog`
  for `*.nd2s_cfg`, calls `apply_config_to_pages` to push all settings into the
  live page widgets, and records the loaded config as the new baseline.
- **`nd2studios/core/main_window.py`** — `self._cfg_path: Optional[str]` and
  `self._cfg_data: Optional[dict]` instance variables tracking the active config
  file path and its last loaded/saved content.

### Changed

- **`nd2studios/core/main_window.py`** — sidebar bottom buttons: removed "New
  Session"; replaced "Save Session" / "Load Session" with "Save" / "Load" (config
  file ops); added "Configure" button opening `ConfigWizard`.
- **`nd2studios/pages/results_page.py`** — column sidebar layout changed from
  `QGroupBox`-per-group to a flat `QVBoxLayout` with section-header `QLabel`s
  and `QFrame` separators; all rows fixed to `_CB_HEIGHT = 24` px for uniform
  spacing. Removed `_ResultsConfigDialog` and the "Configure…" button (replaced
  by the main-window "Configure" sidebar button).
- **`nd2studios/pages/results_page.py`** — `QTableView.setAlternatingRowColors`
  changed from `True` → `False`; explicit stylesheet sets uniform
  `background-color: BG_PRIMARY`, `color: FG_PRIMARY`, `border-bottom` grid
  lines using `BORDER_COLOR`, and `BG_HOVER` for selected rows.
- **`nd2studios/pages/results_page.py`** — `_update_table(rows)` now filters row
  dicts to only the columns whose sidebar checkbox is checked before constructing
  `_MeasurementsModel`; live toggling re-renders without recomputing measurements.
- **`nd2studios/pages/results_page.py`** — `save_to_experiment` / `load_from_experiment`
  persist `selected_columns` (list of checked column keys) in `exp.results_config`
  alongside the existing `pipeline` and `image_format` entries.
- **`nd2studios/pages/results_page.py`** — splitter restructured from 2-pane
  (table | summary) to 3-pane (columns sidebar | table | summary).

---

## [Unreleased] - 2026-05-24 (V1.0 multifile-analysis-perf)

### Added

- **`nd2studios/pages/analysis_page.py`** — file-selector `QComboBox` row (hidden
  when only one confirmed record exists); populated from
  `MainWindow.get_confirmed_records()` in `on_activated()`. Selecting a different
  file calls `_reload_for_selected_record()` which rewires the viewer and restores
  stored analysis results for that record. `_active_exp()` now returns
  `self._selected_rec` if set, so all downstream logic (job submission, overlay,
  export) automatically uses the selected file.

### Changed

- **`nd2studios/pages/recipe_page.py`** — secondary recipe workers now start
  **after** the primary trial finishes (serialized via `_on_trial_done_primary`)
  rather than simultaneously, eliminating I/O contention on large files.
- **`nd2studios/pages/recipe_page.py`** — `_rebuild_columns()` now returns `bool`;
  `on_activated()` only calls `_populate_viewers_from_records()` when columns
  actually changed, preventing redundant `set_channels()` calls on every tab
  switch.

---

## [Unreleased] - 2026-05-24 (V1.0 multifile-recipe-playal)

### Added

- **`nd2studios/widgets/multi_axis_viewer.py`** — `MultiAxisViewer.set_t_playing(playing, fps=None)`:
  public method to drive T-axis playback from external callers; sets FPS if
  provided, then toggles `_t_play.setChecked(playing)`.
- **`nd2studios/pages/import_page.py`** — "> Play All" toggle button and FPS
  `QDoubleSpinBox` (0.1–60 Hz, default 10 Hz) in the toolbar; `_on_play_all_toggled`
  calls `panel.viewer.set_t_playing()` on every open panel.
- **`nd2studios/pages/import_page.py`** — `get_confirmed_records()`: returns list
  of `ND2StudiosRecord` instances whose `status` is in
  `{imported, preprocessed, analyzed, exported}`.
- **`nd2studios/core/main_window.py`** — `get_confirmed_records()`: thin wrapper
  that queries the ImportPage; falls back to active record if Import page is
  not yet built.
- **`nd2studios/pages/recipe_page.py`** — `_RecipeColumn` helper class
  (per-confirmed-file: `record`, `viewer_raw`, `viewer_proc`, `widget`, `worker`).
  `_rebuild_columns()` repopulates the viewer area on every `on_activated()` call:
  1 file → horizontal Raw | Processed; >1 files → vertical Raw / Processed per
  column, columns side-by-side. Trial / Accept / Reject / Remove Last / Clear all
  operate on all columns in parallel. Crop still applies to primary file only.

### Changed

- **`nd2studios/widgets/image_viewer.py`** — zoom-out button label changed from
  `"−"` (Unicode minus, invisible on Windows) to `"-"`.
- **`nd2studios/widgets/multi_axis_viewer.py`** — play button initial label
  changed from `"▶"` to `">"`, pause label from `"⏸"` to `"||"`, resume
  label from `"▶"` to `">"` (all were invisible on Windows).
- **`nd2studios/pages/recipe_page.py`** — `viewer_raw` and `viewer_proc` are
  now read-only properties pointing to the primary `_RecipeColumn`; all direct
  viewer references inside the class use `self._columns[...]` or the properties.

---

## [Unreleased] - 2026-05-24 (V1.0 multifile-sidebyside)

### Added

- **`nd2studios/widgets/file_panel.py`** — new `FilePanel` widget that
  bundles a collapsible controls sidebar (220 px ↔ 0 px, animated) and a
  `MultiAxisViewer` for a single file. Signals: `panel_close_requested`,
  `loaded`. Public helper `fit_viewer()` calls `canvas.reset_zoom()` and is
  wired to both the inner splitter's `splitterMoved` and the sidebar
  animation's `finished` signal so images autofit on every resize event.
  Module-level `release_record_arrays(rec)` replaces the former
  `_release_experiment_arrays` helper in `import_page.py`.
- **`ChannelInfoRow`** moved from `import_page.py` into `file_panel.py`
  (same implementation, now shared).

### Changed

- **`nd2studios/pages/import_page.py`** — rewritten as a multi-panel
  container.  Hosts a toolbar ("+Add File" button) and a horizontal
  `QSplitter` of `FilePanel` instances.  The primary panel (index 0) shares
  its `record` attribute with `exp_manager.active`; secondary panels have
  standalone records.  `load_from_experiment` / `save_to_experiment` /
  `on_activated` delegate to `self._panels[0]` for full backward
  compatibility.  `reset_all_viewers()` and `viewer` property added for
  `MainWindow` integration.
- **`nd2studios/core/main_window.py`** —
  `_reset_active_viewer_zoom` now checks for `reset_all_viewers()` first
  (ImportPage multi-panel path) and falls back to `viewer.canvas.reset_zoom()`
  for single-viewer pages.
- **`nd2studios/core/theme.py`** — added QSS rules for `#filePanelHeader`,
  `QLabel#filePanelTitle`, `QPushButton#filePanelCollapseBtn`,
  `QPushButton#filePanelCloseBtn`, `#filePanelControls`.

---

## [Unreleased] - 2026-05-23 (V1.0 stitch-black-fix)

### Bug Fixes

- **`export_stitched_tiff`** (`nd2studios/backend/exporters/stitch_exporter.py`):
  Removed bare `except Exception: pass` handler wrapping every
  `volume.get_frame()` call in the tile-reading loop. Previously any failure
  (wrong channel index, IO error, dtype mismatch) silently produced a zero tile,
  causing the entire stitch output to be black with no error shown to the user.
  Exceptions now propagate through `BaseWorker.run()` and surface as the
  existing "Stitch failed" error dialog. Added `if f.ndim != 2: f = f.squeeze()`
  after `get_frame` for edge-case shape robustness.
- **`resolutionunit` in all TIFF exporters**: `resolutionunit="MICROMETER"` is
  not a valid TIFF spec value and was silently ignored by `tifffile`
  v2026.5.15. Changed to `resolutionunit=None` in:
  - `stitch_exporter.py` (`tifffile.memmap` call)
  - `tiff_exporter.py` (`TiffWriter.write` and `tifffile.imwrite` calls)
  - `composite_exporter.py` (`TiffWriter.write` call)

---

## [Unreleased] - 2026-05-23 (V1.0 overlay-perf-cache)

### Changed

- **`MultiAxisViewer._do_refresh`** (`nd2studios/widgets/multi_axis_viewer.py`):
  adds two new fast-path tiers for the post-process / overlay path, which
  previously bypassed all existing cache tiers:
  1. **Post-process cache hit** (`_pp_cache[(m, t, pp_version)]`): zero numpy
     work if the overlay-composited frame was already computed at the current
     overlay version — analogous to the QPixmap cache for base frames.
  2. **Render cache + post-process**: if `_render_cache[(m, t)]` exists, skips
     the full LUT+channel-compose (~20–50 ms for 2K multi-channel) and only
     pays for the overlay callback (~10–30 ms rasterize + blend).
- **`MultiAxisViewer._compose_current_frame`**: fills `_render_cache[(m, t)]`
  on-demand when the entry is absent (benefits GPU-canvas users who never run
  `PreRenderWorker`); also stores the post-processed result in `_pp_cache`
  after the live-compose fallback so subsequent refreshes hit the cache.
- **`MultiAxisViewer.set_frame_post_process`**: now calls
  `invalidate_post_process_cache()` before `_do_refresh()` so stale overlay
  composites are never shown after the callback changes.
- **`MultiAxisViewer._invalidate_render_cache` /
  `_start_pre_render_worker`**: both clear `_pp_cache` alongside the existing
  render/pixmap cache clears — LUT or chip changes rebuild the base, so the
  derived overlay cache must be flushed too.
- **`AnalysisPage._composite_overlay`** (`nd2studios/pages/analysis_page.py`)
  Manual Mask branch: caches the rasterized `int32` mask in
  `_raster_cache[(m, t, z_slot, raster_version)]` so repeated calls for the
  same frame and shape state skip `rasterize_shapes()`. Cache is bounded at
  200 entries.
- **`AnalysisPage._on_shape_drawn` / `_on_clear_current_frame` /
  `_on_vertex_moved` / `_on_apply_expand`**: each calls new
  `_invalidate_mask_cache()` before `_update_overlay()` to ensure the raster
  cache and the viewer's `_pp_cache` are flushed whenever shapes change.

### Added

- **`MultiAxisViewer.invalidate_post_process_cache()`**: public method that
  increments `_pp_version` and clears `_pp_cache`. Called by `AnalysisPage`
  when shape state changes so that all overlay-composite entries for the old
  shapes are evicted without touching the base render cache.
- **`AnalysisPage._invalidate_mask_cache()`**: helper that bumps
  `_raster_version`, clears `_raster_cache`, and delegates to
  `viewer.invalidate_post_process_cache()`.

---

## [Unreleased] - 2026-05-23 (V1.0 smooth-playback)

### Added

- **`PreRenderWorker`** (`nd2studios/workers/pre_render_worker.py`): background
  `QThread` that pre-composes all `(M, T)` frames to `(H, W, 3) uint8` numpy
  arrays off the GUI thread after dataset materialization. Priority order:
  current M first so the visible time-series is cached before other M positions.
  Signals: `progress(int)`, `frame_cached(int, int)`, `finished()`, `error(str)`.
  Memory guard: limits cache to `priority_m` only when `n_m × n_t > 500` frames.
- **`ImageCanvas.set_pixmap_direct(pixmap)`** (`nd2studios/widgets/image_viewer.py`):
  displays a pre-built `QPixmap` without creating a `QImage`; the hot path
  during cached playback where per-frame cost is now just a `paintEvent` repaint.

### Changed

- **T-slider debounce** (`multi_axis_viewer.py` `_t_debounce`): 50ms → 5ms.
  The old 50ms was designed for the V1.40 lazy-loading + IOWorker stack; with
  all data in RAM it only added artificial latency.
- **FPS spinbox range** (`_make_axis_row`): 0.1–30 fps → 0.1–60 fps.
- **`_set_axis_playing`**: T-axis playback timer floor drops to 16ms (60fps)
  once the pre-render cache is ready; otherwise keeps 50ms (20fps) to avoid
  overloading the live compose path.
- **`_axis_tick`**: blocks `QSlider.valueChanged` during playback and calls
  `_do_refresh()` directly, eliminating the 5ms debounce overhead per tick.
- **`_do_refresh`**: adds two fast-paths before live compose:
  1. QPixmap cache path (zero numpy work, pure paintEvent swap).
  2. numpy composite cache path (skips LUT+compose; ~1ms vs ~50ms at 2048²).
- **`_on_lut_contrast_changed` / `_on_chip_state`**: now call
  `_invalidate_render_cache()` so pre-rendered frames are discarded and the
  worker restarts with the updated LUT / color settings.

## [Unreleased] - 2026-05-22 (V1.40)

### Added

- **Phase Addendum — Multi-Core Parallelism** (`nd2studios/utils/storage.py`,
  `nd2studios/utils/threading.py`,
  `nd2studios/compute/parallel/__init__.py`,
  `nd2studios/compute/parallel/shared_array.py`,
  `nd2studios/compute/parallel/process_map.py`,
  `nd2studios/compute/parallel/thread_map.py`,
  `nd2studios/backend/analysis/tear_detection.py`,
  `nd2studios/pipeline/stages/recipe_stage.py`,
  `nd2studios/pipeline/stages/pyramid_stage.py`):

  Applies `CodeLog/ClaudesPlan/08_addendum_parallelism.md` to the
  V1.34/V1.38/V1.39 codebase. The addendum closes the parallelism
  gaps the original 7-phase plan left on the table — most importantly
  the sequential M×T×Z×C loop in `PyramidStage.build`. The work is
  scoped to ND2Studios: where the addendum prescribes Dask or Numba
  (neither is a project dependency), we substitute the GIL-releasing
  `ThreadPoolExecutor` pattern with thread-local file handles.

  - New `nd2studios/utils/storage.py`:
    - `is_fast_storage(path)` — Linux sysfs lookup for the backing
      device's `queue/rotational` flag. Returns `True` for NVMe /
      SSD-class storage, `False` on rotational disks and on
      Windows/macOS (conservative; the V1.34 single-reader default
      is the safe fallback).
    - `recommended_io_thread_count(path)` — returns 2 for NVMe-class
      storage and 1 elsewhere. Caps at 2; beyond that the queue
      depth benefit plateaus and the prefetcher's cache hit rate
      starves the readers anyway.
  - `nd2studios/utils/threading.py` additions:
    - `IOWorker.__init__(shared_queue=None)` — when a shared
      `PriorityQueue` is supplied, the worker drains the caller's
      queue instead of allocating its own. Single-worker callers
      keep the V1.34 shape.
    - `WorkerGroup` — facade over N `IOWorker` instances sharing one
      priority queue. Exposes `.submit`, `.cancel_all`, `.stop`;
      callers connect to the individual workers' `plane_ready` and
      `error` signals as before.
    - `start_io_workers(volume, n)` — spawns *n* `IOWorker`
      instances on dedicated `QThread`s sharing one queue and
      returns `(WorkerGroup, [QThread])`. `n=1` is equivalent to
      `start_io_worker`. Per-thread `volume.reopen()` keeps each
      worker on its own ND2/TIFF handle (the `nd2` SDK is not
      thread-safe across file handles).
  - New `nd2studios/compute/parallel/` subpackage:
    - `shared_array.py` — `shared_ndarray(arr)` context manager
      exposes a numpy array to worker processes via
      `multiprocessing.shared_memory`, yielding `(name, shape,
      dtype_str)`. Cleans up the block on exit. `attach_shared(...)`
      is the worker-side reconstructor.
    - `process_map.py` — `process_map_planes(stack, fn_module,
      fn_name, indices, kwargs=, n_workers=)` runs a pure-Python
      CPU-bound per-plane function across a `ProcessPoolExecutor`
      backed by shared memory. The worker function must be
      importable (module-level); use this when the work is
      GIL-bound at the Python level.
    - `thread_map.py` — `thread_map_planes(stack, fn, indices=,
      kwargs=, n_workers=, progress_cb=, cancelled_cb=)` runs a
      callable across a `ThreadPoolExecutor`. The right primitive
      for ND2Studios analysis pipelines, all of which sit on
      scipy / scikit-image / numpy C extensions that release the
      GIL.
  - `TearDetectionPipeline.run` — the per-T loop is now fanned out
    across a `ThreadPoolExecutor` sized to
    `recommended_worker_count()`. Per-frame outputs (label mask +
    measurement rows) are collected and stitched in T-order at the
    end so the output `label_stack` shape and the row order match
    the V1.19 sequential path byte-for-byte. The `cancelled_cb`
    contract is preserved.
  - `EnhancedDataset.materialize_all` — channels are independent
    (different keys, separate output buffers, GIL-releasing plugin
    work) so the recipe is now applied across them in parallel via
    a `ThreadPoolExecutor` capped at `min(channel_count,
    recommended_worker_count())`. `materialize_channel` gains a
    `threading.Lock` around the cache read/write so two concurrent
    materializations on the same channel don't both compute and
    race the final write. `clear_cache` is held under the same lock.
  - `PyramidStage.build` — the sequential `for m for t for z for c`
    nest is converted to a per-level `ThreadPoolExecutor` whose
    tasks share a thread-local `volume.reopen()` handle (the `nd2`
    SDK's per-file state is not thread-safe). Each task writes to a
    unique `(m, t, z, c)` zarr chunk — distinct files in a
    DirectoryStore, no contention. Progress is reported through a
    new `_AtomicCounter` so the bar stays monotonic across N
    worker threads. Cancellation is honored within ~1 s. Reader
    handles are closed deterministically after each level finishes
    (the executor recycles worker threads, so per-thread destructors
    are not available; we track them in a registry and close them
    in a `finally`).

### Changed

- `nd2studios/utils/threading.py:IOWorker` — gains an optional
  `shared_queue` constructor parameter. Default `None` preserves
  the V1.34 owns-its-own-queue behavior.

### Notes

- No new third-party dependencies. The addendum's Dask / Numba
  patterns require packages ND2Studios does not ship; the
  thread-pool path used here achieves the same near-linear scaling
  on the workloads that matter (analysis per-T, pyramid build per
  chunk, recipe per channel) because every hot kernel sits on a
  GIL-releasing C extension.
- The new patterns are opt-in: nothing in the V1.34/V1.38/V1.39
  call sites changes. Tear detection, recipe materialization, and
  pyramid building light up the parallel path automatically the
  next time they run.

## [Unreleased] - 2026-05-22 (V1.39)

### Added

- **Phase 7 — Optional GPU Acceleration & Multi-Resolution Pyramids**
  (`nd2studios/compute/gpu/`,
  `nd2studios/pipeline/stages/pyramid_stage.py`,
  `nd2studios/compute/pipeline_jobs.py`,
  `nd2studios/core/main_window.py`,
  `nd2studios/core/settings.py`,
  `nd2studios/pages/import_page.py`,
  `nd2studios/widgets/multi_axis_viewer.py`,
  `nd2studios/backend/analysis/spots/scale_space.py`,
  `nd2studios/backend/analysis/tear_detection.py`,
  `nd2studios/backend/analysis/nuclei_segmentation.py`,
  `nd2studios/__main__.py`):

  Up to V1.38 every analysis op ran on CPU and every viewer
  zoom-out painted a full-resolution plane to the texture. Phase 7
  introduces *two* independently-toggleable optional speedups:

  1. **GPU dispatch for hot analysis ops.** A new
     `nd2studios.compute.gpu` package wraps the four ops the V1.33
     profiling baseline flagged as analysis hotspots — `gaussian`,
     `gaussian_filter`, `gaussian_laplace`, `threshold_otsu` — so
     they route through CuPy + `cucim.skimage.filters` when GPU
     mode is on, and through scipy / scikit-image when it is off.
     Hosts without CuPy installed run exactly as in V1.38 with no
     warnings; hosts with CuPy fall back gracefully on
     `cupy.cuda.memory.OutOfMemoryError`, missing `cucim`, or any
     other GPU runtime failure (one log warning per op the first
     time it falls back, then silent).
  2. **Multi-resolution display pyramids.** Each imported file now
     gets a `pyramid.zarr` artifact in its V1.38 workspace; the
     viewer reads from a downsampled level when the visible image
     extent overflows the viewport, so a 4096² source at "fit to
     window" zoom uploads a 1024² plane instead of the full one.
     Level 0 is the source ND2 itself (read by the existing
     `LazyND2Volume`) — the pyramid stores levels 1..N only,
     keeping disk overhead to ≈ 33 % of source size rather than
     ≈ 133 %. The build runs as a Phase 5 background `AnalysisJob`,
     so the GUI stays responsive at level 0 while the pyramid is
     materialising. Pyramids require `zarr` (already a soft V1.38
     dep); without it the viewer pins to level 0 and the
     Performance dialog explains why.

  - New `nd2studios.compute.gpu` package (Qt-free, scipy/skimage
    re-exports on CPU-only hosts):

    - `detect.py` — `gpu_status()` returns
      `{available, reason, device_name, memory_gb, cucim}` without
      raising; `log_gpu_status()` emits a single info line at
      startup.
    - `array.py` — module-level `_USE_GPU` bool driven by
      `configure(use_gpu)` (returns the *effective* value so a
      requested-True on a CPU-only host stays False). Reads via
      `is_gpu_enabled()`. Env override `ND2_DISABLE_GPU_ANALYSIS=1`
      pins the flag to False regardless of the GUI checkbox. A
      256² array-size floor (`should_dispatch_to_gpu`) prevents
      tiny-array GPU dispatch where transfer cost dominates.
    - `ops.py` — the four shape-compatible wrappers. Every function
      catches the broad set of GPU failures (`OutOfMemoryError`,
      `cucim` ImportError, NVML/driver runtime errors) and falls
      back to the CPU implementation, logging once per op name.

  - New `BuildPyramidJob` in `compute/pipeline_jobs.py` — wraps
    `PyramidStage.build()` into the Phase 5 `JobRunner` so the build
    is cancellable, progress-reported, and coalesced on the
    `"pyramid_build"` key. Submitting a new build cancels any
    in-flight one.

  - New `PyramidStage` + `PyramidReader` in
    `pipeline/stages/pyramid_stage.py`. `PyramidStage` mirrors
    `RecipeStage` / `AnalysisStage`: Qt-free, owns a per-level
    Zarr array `(M, T, Z, C, H_L, W_L)` chunked
    `(1, 1, 1, 1, H_L, W_L)`. `default_pyramid_levels(h, w)` picks
    a level count from source dimensions, clamped to `[2, 6]`.
    `PyramidReader.get_frame(level, c, m, t, z, z_mode)` mirrors
    `LazyND2Volume.get_frame()` with one extra slot so the
    viewer's read site dispatches by level with no other branch.
    `pick_level_for_viewport(screen_px, image_pixels_visible)`
    returns the smallest level whose width still ≥ the visible
    extent. Raises `PyramidUnavailable` when `zarr` is missing or
    the source is too small to benefit (≤ 256 px); callers catch
    and degrade to level 0.

  - `MainWindow` gains three Phase-7 hooks and one slot:
    - `pyramid_stage()` — returns a `PyramidStage` for the active
      session, or `None` when no session is attached, when
      `ND2_DISABLE_PYRAMIDS=1`, or when `zarr` is missing.
    - `start_pyramid_build(volume)` — submits a `BuildPyramidJob`
      and surfaces the status; reattaches the reader if the
      pyramid is already committed for this source hash.
    - `_attach_pyramid_reader_to_viewers(reader)` — pushes a
      `PyramidReader` into every page exposing a `viewer` that
      implements `attach_pyramid`. Today only the multi-axis
      viewer does, but the scan is page-agnostic.
    - `_on_pyramid_job_done(result)` — connected to
      `JobRunner.job_done`; filters by `result.key == "pyramid_build"`
      so analysis-preview/commit slots are unaffected. Updates
      the status text and reattaches the freshly-built reader.

  - `MainWindow._open_performance_settings()` — modal dialog with
    two checkboxes ("Use GPU acceleration (NVIDIA RTX A4000, 16.0 GB)"
    and "Build multi-resolution pyramids on import") plus a
    "Rebuild pyramid for current file" button. The GPU checkbox is
    disabled with a "GPU not available: <reason>" tooltip on
    CPU-only hosts; the pyramid checkbox is disabled and tooltipped
    when `zarr` is missing. OK pushes the values onto
    `Settings.USE_GPU_ANALYSIS` / `Settings.BUILD_PYRAMIDS` and
    calls `compute.gpu.configure()`. A new sidebar `⚙ Performance`
    button next to New / Save / Load opens the dialog.

  - `MultiAxisViewer` extensions:
    - `attach_pyramid(reader)` / `detach_pyramid()` — bind or unbind
      a `PyramidReader`. Attach connects
      `viewbox.sigRangeChanged → _on_viewport_changed` so pyramid
      level recomputes on every pan/zoom.
    - `_active_level: int` — 0 means "read from
      `LazyND2Volume`/IOWorker (V1.38 path)"; ≥ 1 means
      "synchronously read from `_pyramid_reader.get_frame(level, …)`".
    - `_frame_cache_key(c)` — returns the V1.38 5-tuple
      `(c, m, t, z, z_mode)` at level 0 and a 6-tuple
      `(level, c, m, t, z, z_mode)` at level ≥ 1, so pyramid
      planes coexist in the cache with level-0 planes without
      colliding. The V1.38 prefetcher and IOWorker continue
      reading and writing 5-tuples — they remain level-0-only by
      design (pyramid planes are small enough to read
      synchronously without their help).
    - `_render_current_frame_gpu()` and `_compose_current_frame()`
      now call `_frame_cache_key(c)` and, on a level-≥-1 miss,
      take the synchronous pyramid path via `_read_pyramid_plane(c)`.
      Level-0 misses still flow through the V1.34 IOWorker and
      V1.35 prefetcher unchanged. The level-change handler bumps
      the existing `_request_counter` so any in-flight level-0 IO
      worker plane arriving after the switch is silently dropped
      by `_on_io_plane_ready`, exactly as a stale slider would be.

  - `ImportPage._on_confirm`'s `_attach_workspace_and_maybe_resume`
    path now calls a new `_kick_off_pyramid(exp)` helper after the
    workspace is attached. The helper:
    - no-ops when `Settings.BUILD_PYRAMIDS` is False, when no raw
      volume is bound to the experiment, or when
      `MainWindow.pyramid_stage()` returns `None`;
    - submits a `BuildPyramidJob` when the source has no committed
      pyramid yet; or
    - reattaches the existing `PyramidReader` to the viewer when a
      prior pyramid is already on disk for this source hash.

  - Hot analysis backends switched to GPU-aware shims:
    - `backend/analysis/spots/scale_space.py` — imports
      `gaussian_filter` and `gaussian_laplace` from
      `compute.gpu.ops` instead of `scipy.ndimage`. Same
      signatures; CPU pass-through when GPU is off.
    - `backend/analysis/tear_detection.py` — imports `gaussian`
      and `threshold_otsu` from `compute.gpu.ops` instead of
      `skimage.filters`.
    - `backend/analysis/nuclei_segmentation.py` —
      `models.Cellpose(gpu=is_gpu_enabled())`. Cellpose owns its
      own CUDA setup; we just forward the flag. Failure to import
      the GPU module on a CPU-only host leaves
      `use_gpu = False` and Cellpose runs on CPU as before.

  - `__main__.py` — after `QApplication` instantiation calls
    `log_gpu_status()` (one info line summarising availability)
    and `configure(Settings.USE_GPU_ANALYSIS)` (defaulting to
    False). Both are wrapped in a broad `try/except` so a
    misbehaving GPU runtime cannot block startup.

  - `Settings` additions:
    - `USE_GPU_ANALYSIS = False` — off by default; user opts in
      via the Performance dialog.
    - `GPU_ANALYSIS_ENV_DISABLE = "ND2_DISABLE_GPU_ANALYSIS"` —
      env override that pins the flag to False.
    - `BUILD_PYRAMIDS = True` — on by default; the build is a
      background job and the GUI stays responsive while it runs.
    - `PYRAMID_ENV_DISABLE = "ND2_DISABLE_PYRAMIDS"` — env
      override that suppresses both build submission and reader
      attachment, pinning the viewer to level 0 (used by the
      profiling harness for like-for-like measurements).

### Changed

- `nd2studios/pipeline/__init__.py` re-exports `PyramidStage`,
  `PyramidReader`, `PyramidUnavailable`, `default_pyramid_levels`.
- `nd2studios/compute/__init__.py` re-exports `BuildPyramidJob`.
- The frame cache (V1.35) now coexists with level-tagged 6-tuple
  keys at level ≥ 1 alongside the legacy 5-tuple keys at level 0;
  the cache itself is unchanged — it has always treated keys as
  opaque tuples — but the viewer's pin/unpin bookkeeping clears
  on level transitions so pyramid planes do not displace source
  planes when the user zooms back in.

### Notes

- Pyramids are an optional speedup, not a correctness change.
  Analysis pipelines always receive full-resolution channel data;
  the pyramid is read by the multi-axis viewer only.
- GPU mode is opt-in even when CuPy is installed, so analysis
  outputs stay byte-identical to V1.38 unless the user explicitly
  flips the Performance dialog checkbox.
- Optional dependencies (not added to `requirements.txt` as hard
  pins because `version_push.py` regenerates that file from
  `pip freeze`): `cupy-cuda12x` (or matching CUDA version),
  `cucim-cu12`. `zarr` is shared with V1.38; `cellpose` continues
  to be optional via the existing `find_spec` guard.

## [Unreleased] - 2026-05-22 (V1.38)

### Added

- **Phase 6 — Pipeline Workspace with Per-Source Stage Commits**
  (`nd2studios/pipeline/`, `nd2studios/core/main_window.py`,
  `nd2studios/core/experiment_manager.py`,
  `nd2studios/pages/import_page.py`, `nd2studios/pages/recipe_page.py`,
  `nd2studios/pages/analysis_page.py`,
  `nd2studios/pages/results_page.py`,
  `nd2studios/pages/export_page.py`):

  Up to V1.37 every stage of a session lived in RAM at once: raw
  channels, processed channels (full `(T, H, W)` per channel),
  per-pipeline label masks `(T, H, W)` int32 per channel × per M.
  On a 200-frame 2048² 3-channel ND2 with two committed analyses,
  that's >5 GiB of int32 labels alone — and there was no on-disk
  copy, so a navigation that "should have" freed RAM didn't, and
  re-opening the same file later re-ran every stage from scratch.

  Phase 6 introduces an automatic per-source-file workspace and
  two stage adapters that flush committed outputs to disk so the
  in-RAM copies can be released without losing the work.

  - New `nd2studios.pipeline` package (Qt-free; sits next to
    `compute/`):

    - `Session` (`session.py`) — owns one workspace directory per
      source file, keyed by `hash_source_file(path)` (first MiB +
      last MiB + size + mtime; sha-256 truncated to 16 hex chars).
      Default root `~/.nd2studios/workspace/sessions/`, overridable
      via `ND2STUDIOS_WORKSPACE`. `SessionManifest` + `StageRecord`
      are plain dataclasses serialized to `manifest.json` alongside
      the artifacts. `archive(suffix)` renames the session dir to
      `<hash>.archived-<timestamp>/` for the Import page's "Start
      fresh" branch.
    - `PipelineStage` ABC (`stage.py`) — Qt-free; subclasses
      implement `commit()` returning a `StageRecord`. Stages do not
      own threads or timers; commits run on the GUI thread after a
      worker has produced the heavy data, and the I/O is small
      (JSON manifest + one label-stack write per M).
    - `storage.py` — `write_label_stack(path, arr)` /
      `read_label_stack(path)` / `open_label_stack(path)` with a
      Zarr-or-NPZ fallback. `zarr` is optional: when present, label
      stacks are chunked `(1, H, W)` and zstd-3 compressed; when
      absent, NPZ is used (already in the dependency surface via
      `ND2StudiosManager.save_session`). `HAS_ZARR` exposes the
      decision to callers.
    - `RecipeStage` (`stages/recipe_stage.py`) — parameter-only
      commit: writes the accepted recipe to `recipe/recipe.json`
      and stamps the manifest. No `processed.zarr` is materialized
      by default — `EnhancedDataset` lazily reapplies the recipe
      on a per-channel basis when the Export / Analysis / Results
      pages ask for processed channels after the Recipe page has
      released them from RAM.
    - `EnhancedDataset` (`stages/recipe_stage.py`) — dict-like
      proxy over raw channels + recipe; `materialize_channel(name)`
      runs the plugin chain via `PluginBase` (matching
      `RecipeWorker`'s logic) and caches per-channel results.
      Peak RAM during a re-derivation is one channel rather than
      `n_channels`.
    - `AnalysisStage` (`stages/analysis_stage.py`) — per-pipeline
      stage. `commit_m(m, AnalysisResult)` streams one M's label
      masks (primary + secondary) to
      `analysis/<pipeline>/m_NNN/labels_<channel>.{zarr,npz}` and
      writes a small `summary.json` per M with measurements +
      overlay defaults + serialised
      `volumetric_voxel_counts`. `rehydrate_m(m)` reads them back
      as a fresh `AnalysisResult`; `open_label_lazy(m, channel)`
      hands out a zarr `Array` so the Results page can stream
      slices for a measurements computation.

  - `MainWindow.__init__` now also owns `self.session: Optional[Session]`,
    plus three helpers:

    - `attach_session_for(filepath)` — called by the Import page
      after a successful load; creates or reuses the workspace and
      stamps the manifest's shape (`n_t`, `n_m`, `n_z`,
      `n_channels`, dims, pixel size, channel names).
    - `recipe_stage()` / `analysis_stage(name)` — convenience
      constructors used by the pages so the touchpoints don't need
      to import `pipeline.stages`.
    - `_release_outgoing_stage(page_key)` — called from
      `_navigate` after `save_to_experiment`. When the outgoing
      page is `recipe` and the workspace has a committed recipe,
      swaps `exp._processed_channels` for an `EnhancedDataset`
      proxy (`exp._processed_view`) and drops the dict. When the
      outgoing page is `analysis`, clears `label_masks` /
      `secondary_label_masks` on every `AnalysisResult` whose M
      position is committed to the workspace.

  - `ND2StudiosRecord` gains two accessors that hide the dict /
    proxy duality from consumers:

    - `processed_view()` — returns the in-RAM `_processed_channels`
      if present, else `_processed_view` (the `EnhancedDataset`),
      else `_raw_channels`. Dict-like in all cases.
    - `has_processed()` — True when either source is available.
    - New non-serialized field `_processed_view`.

  - `ImportPage._on_confirm` attaches the workspace via
    `MainWindow.attach_session_for(filepath)`. If prior commits are
    found, a single `QMessageBox.question` offers
    **Resume / Start fresh**. Resume re-populates the active
    `ND2StudiosRecord` with the prior recipe (promoting status to
    `preprocessed`) and per-pipeline `AnalysisResult`s — no rerun
    needed. Start fresh archives the prior session dir.

  - `RecipePage._on_accept` now calls `_commit_recipe_stage()`
    after appending the new step; the workspace receives the
    canonical recipe at every Accept. The actual `_processed_channels`
    release is gated on the page-leave hook in `MainWindow._navigate`
    so the right-side preview stays live while the user is still
    on the Recipe page. `on_activated()` rehydrates from the
    `EnhancedDataset` proxy (running `materialize_all()`) when the
    user returns from a downstream page.

  - `AnalysisPage._on_finished` now also calls
    `_commit_analysis_m(pipeline, m, result)` on every M-completion,
    so multi-M runs stream label masks to disk as they go rather
    than holding all M's in RAM until the queue drains. The
    multi-M state machine, the Phase 5 commit / cancel keys, and
    the per-M results dict are unchanged. `_finalize_analysis_stage()`
    stamps the manifest record once the queue is empty.
    `on_activated()` runs the same EnhancedDataset materialize as
    the Recipe page so the multi-axis viewer sees a normal dict.

  - `ResultsPage._on_compute` calls
    `_rehydrate_released_label_masks(pipeline, results_by_m)`
    before `compute_measurements`. Released results have empty
    `label_masks` but populated `measurements` / `summary` —
    rehydrate refills the arrays from the workspace (Zarr or NPZ)
    so the Results page works identically whether the masks were
    released or not.

  - `ExportPage.on_activated` and `_channels_to_export` use
    `has_processed()` / `processed_view()` so the "Processed"
    radio stays enabled after a Recipe → Export navigation that
    released RAM. Per-channel materialization happens at write
    time, one channel at a time, so peak RAM during a movie export
    is unchanged from V1.37.

- **Workspace escape hatches.** Setting
  `ND2STUDIOS_DISABLE_WORKSPACE=1` makes
  `MainWindow.attach_session_for` a no-op and the release hooks
  short-circuit, restoring V1.37 behaviour exactly. Setting
  `ND2STUDIOS_WORKSPACE=/path/to/dir` scopes the workspace root
  (used by the profiling harness and tests).

### Notes

- `.nd2s` save/load is unchanged. The workspace is automatic and
  independent: a `.nd2s` saved on machine A and loaded on machine
  B without the source file (or with a workspace miss) behaves
  exactly as in V1.37.
- `zarr` is treated as an optional install. When absent, label
  stacks fall back to NPZ; no other code path notices.
- No `requirements.txt` bump is mandatory; `pip install zarr`
  is recommended for users with large analyses (multi-GiB label
  stacks compress 5–20× with zstd-3).
- Long-running QThread workers (`RecipeWorker`, `BatchWorker`,
  `LoadWorker`, `PrefetchWorker`, `IOWorker`, `StitchWorker`) are
  unchanged. `BatchWorker` does not write to the workspace yet —
  see Phase 6 plan, "Out of scope".

## [Unreleased] - 2026-05-22 (V1.37)

### Added

- **Phase 5 — Background Analysis with Cancellation: project-wide
  `JobRunner`, parameter-driven previews, cooperative cancellation
  tokens** (`nd2studios/compute/`, `nd2studios/core/main_window.py`,
  `nd2studios/pages/analysis_page.py`):

  Up to V1.36 the Analysis page's "Screen frame" preview re-ran only
  when the user moved the T/M/Z slider — dragging a Histogram
  Threshold cutoff or Spots `min_sigma` left the overlay stale until
  the next viewer-coord change. Cancellation used a plain bool flag
  on `BaseWorker`, with no memory barrier between threads, and there
  was no coordinated drain on app exit. Phase 5 layers a small,
  additive coordination module on top of the existing
  `BaseWorker` / `AnalysisWorker` plumbing without ripping it out.

  - New `nd2studios.compute` package:

    - `CancellationToken` (`cancellation.py`) — `threading.Event`-backed,
      with both `is_cancelled()` (matches the existing
      `AnalysisPipeline.run(cancelled_cb=...)` contract) and
      `check()` (raises `CancelledError` for cooperative unwind).
    - `ProgressReporter` (`progress.py`) — `QObject` with a single
      `progress(key, fraction, message)` signal. The
      `as_pipeline_progress_cb()` adapter hands back a 0–100 int
      callable so existing pipelines stay verbatim.
    - `AnalysisJob` ABC + frozen `JobResult` dataclass (`jobs.py`).
      Subclasses fill in `run(progress)`; the token gives them
      cooperative cancellation.
    - `JobRunner` (`runner.py`) — `QThreadPool`-backed dispatcher
      sized by `recommended_worker_count()`. `submit(job)` coalesces
      by key: a fresh submission cancels any in-flight job sharing
      the same key. Public signals `job_done(JobResult)`,
      `job_cancelled(str)`, `job_progress(str, float, str)` are
      auto-connected to the GUI thread. `shutdown(wait_ms)` drains
      the pool on app close.
    - `ResultStore` (`result_store.py`) — `RLock`-guarded keyed dict
      for caching preview/commit outputs. Phase 6 will sit a Zarr
      store in front of it for bounded-RAM commits.
    - `PipelinePreviewJob` / `PipelineCommitJob` (`pipeline_jobs.py`)
      — adapters that bridge any `AnalysisPipeline` to the runner.
      Preview runs the pipeline on a `(1, H, W)` frame; Commit runs
      it on a full `(T, H, W)` stack for one M position. Both pass
      `token.is_cancelled` as `cancelled_cb` and
      `progress.as_pipeline_progress_cb()` as `progress_cb`, so no
      concrete pipeline (Histogram Threshold, Nuclei, Tear, Spots,
      Manual Mask) needs to change.

  - `MainWindow.__init__` now constructs `self.job_runner = JobRunner(self)`
    before any page is built; `closeEvent` calls
    `job_runner.shutdown(wait_ms=3000)` and `page.on_close()` so the
    app drains workers cleanly and no longer prints
    `QThread: Destroyed while thread is still running` on exit.

  - `AnalysisPage` routes both tiers through the runner:

    - `param_editor.params_changed` now restarts the 300 ms screen
      debounce, so dragging a threshold slider re-screens the
      current frame automatically (gated by the Auto-screen
      checkbox, same as viewer-coord changes).
    - `_screen_current_frame()` builds a `PipelinePreviewJob` under
      key `"analysis_preview"` and submits to the runner instead of
      starting an ad-hoc `AnalysisWorker` and connecting per-launch
      lambdas. The runner's coalescing replaces the hand-rolled
      `_cancel_screen()` dance.
    - `_start_next_m_run()` builds a `PipelineCommitJob` under key
      `"analysis_commit"` and submits, preserving the existing
      multi-M state machine (`_run_queue`, `_results_per_m`,
      `_on_finished`, `_on_error`).
    - Cancel now calls `runner.cancel("analysis_commit")` and the
      progress bar advances via `job_progress` instead of a
      per-worker `progress` signal.
    - New `on_close()` stops the debounce timer so it cannot fire
      during the runner's drain window.

- **`nd2studios.workers.analysis_worker.AnalysisWorker` marked
  deprecated** in its docstring — the class is retained because no
  out-of-tree caller has been audited and `RecipeWorker` /
  `BatchWorker` still use the `BaseWorker` lifecycle. New analysis
  code should submit through the runner.

### Notes

- Long-running QThread workers (`RecipeWorker`, `BatchWorker`,
  `LoadWorker`, `PrefetchWorker`, `IOWorker`, `StitchWorker`) are
  **unchanged** in V1.37. Each owns a file-handle or recipe-pipeline
  lifecycle that fits the long-lived `QThread` model better than a
  short pool job. A later phase will revisit.
- No `.nd2s` schema migration; no recipe format change; no new
  external dependency (`psutil` was pinned in Phase 2).

## [Unreleased] - 2026-05-22 (V1.36)

### Added

- **Phase 4 — pyqtgraph GPU-Accelerated Display: per-channel `ImageItem`
  layers, additive composition, GPU LUT + levels** (`nd2studios/widgets/gpu_image_canvas.py`,
  `nd2studios/widgets/multi_axis_viewer.py`, `nd2studios/core/settings.py`,
  `nd2studios/__main__.py`):

  Up to V1.35, the multi-axis viewer's render path was per-pixel CPU
  work on the GUI thread: `apply_lut(frame, lo, hi, gamma)` →
  multiply by per-channel RGB color → accumulate float32 → clip to
  uint8 → `QImage` → `QPixmap`. With Phase 3 delivering cache hits
  sub-millisecond, this composite step had become the dominant cost
  for LUT drag, channel toggle, and slider scrub.

  - New `nd2studios.widgets.gpu_image_canvas.GpuImageCanvas` hosts a
    `pyqtgraph.GraphicsLayoutWidget` + `ViewBox` with one
    `pg.ImageItem` per channel. Each `_ChannelLayer` owns a 256-entry
    colored LUT (black → channel color) plus `(lo, hi)` levels;
    `CompositionMode_Plus` makes Qt sum the channels additively.
    `update_channel(c, plane)`, `set_channel_visible(c, on)`,
    `set_channel_color(c, rgb)`, and `set_channel_levels(c, (lo, hi))`
    are the per-channel hot-path API. A composite-fallback
    `ImageItem` accepts pre-composed RGB arrays via `set_image(rgb)`
    for callers that already produce RGB (export preview, the
    `set_frame_post_process` hook).

  - Tool parity with the legacy `ImageCanvas`: `crop_rect_selected`,
    `shape_drawn`, `vertex_moved`, `edit_committed`, `clicked`,
    `zoom_changed`, `pan_mode_changed` signals are emitted with the
    same payloads. `set_crop_mode`, `set_draw_mode("rect"|"ellipse"|
    "polygon"|None)`, `set_edit_vertices([(iy, ix), ...])`, and
    `set_pan_mode(on)` route through a `_ToolOverlayItem`
    (`QGraphicsItem`) that paints crop/draw/edit overlays in scene
    coordinates. Press/move/release goes through a viewport
    `eventFilter`; the active tool short-circuits ViewBox pan so
    drags don't fight. `widget_to_image` / `image_to_widget` reuse
    pyqtgraph's `viewbox.mapSceneToView` / `mapFromView` so
    coordinate math stays correct under pan/zoom.

  - `MultiAxisViewer.__init__` defensively constructs
    `GpuImageCanvas` when `Settings.USE_GPU_DISPLAY` is True (default)
    and falls back to the legacy `ImageCanvas` if construction
    raises, mirroring the same defensive pattern used elsewhere in
    the codebase. The env override `ND2_DISABLE_GPU_DISPLAY=1`
    forces the legacy path without code edits.

  - `MultiAxisViewer._do_refresh` branches on `_use_gpu_canvas`. The
    new `_render_current_frame_gpu` walks each chip, sets visibility
    / color / levels on the GPU canvas, and pushes raw 2-D planes
    via `canvas.update_channel`. Cache misses are dispatched to the
    V1.34 `IOWorker` exactly as in the CPU branch — the IO
    contract did not change. When a post-process hook is attached
    (a CPU operation that expects `(H, W, 3) uint8`) the viewer
    falls back to `_compose_current_frame` + `canvas.set_image`; the
    GPU canvas's composite-fallback layer is engaged automatically.

  - Pin/unpin bookkeeping ([Phase 3 pinning] from V1.35) carries
    over to the GPU branch — on-screen planes are still exempt from
    LRU eviction. Velocity-biased prefetch is also triggered on
    every GPU refresh.

  - `__main__` configures pyqtgraph globally before any canvas is
    constructed: `imageAxisOrder="row-major"` (numpy `(H, W)` displays
    upright without transpose), `useOpenGL=False` (safer on
    Windows/RDP; can be flipped after benchmarking), `antialias=False`,
    `background=Settings.BG_SECONDARY`. The block is wrapped in a
    try/except so an absent pyqtgraph still launches the legacy path.

### Changed

- **`Settings`**: adds `USE_GPU_DISPLAY = True` and
  `GPU_DISPLAY_ENV_DISABLE = "ND2_DISABLE_GPU_DISPLAY"` so the new
  display backend is opt-out via env, not a code edit.

### Known limitations

- The GPU path applies linear `(lo, hi)` levels only; per-channel
  `gamma` from `LutHistogramWidget` is silently ignored on the GPU
  branch in V1.36. A 256-entry LUT per `(color × gamma)` pair is a
  cheap follow-up but doesn't ship in this version — the McGhee Lab
  uses gamma rarely. Sessions with a non-1.0 gamma will still
  display, just without the gamma curve until the GPU LUT generator
  is extended.
- Other display surfaces — `export_preview_dialog`, the recipe page's
  embedded `ImageViewer`, the stitch dialog — continue to use the
  legacy `ImageCanvas`. The multi-axis viewer (the hot path) is
  where the win is and where Phase 4 lands; other surfaces migrate
  incrementally in later versions.

## [Unreleased] - 2026-05-22 (V1.35)

### Added

- **Phase 3 — Frame Cache & Prefetcher: pinning, stats, velocity-aware lookahead**
  (`nd2studios/backend/frame_cache.py`, `nd2studios/workers/prefetch_worker.py`,
  `nd2studios/widgets/multi_axis_viewer.py`):

  V1.17 shipped a byte-budgeted `FrameCache` and a `PrefetchManager`;
  V1.34 grew the cache budget from a hard-coded 300 MB to a fraction
  of available RAM. Three Phase-3-specific gaps remained:
  (1) the prefetcher's ±5 T window was symmetric regardless of which
  direction the user was scrubbing; (2) the cache evicted strictly by
  LRU, so the on-screen plane could be displaced by its own neighbors
  on a tight budget; (3) there was no way to ask the cache how it was
  doing.

  - `FrameCache` gains a `CacheStats` dataclass (hits, misses,
    evictions, current_bytes, peak_bytes, hit_rate) updated under the
    existing thread lock and readable from any thread; `pin(key)` /
    `unpin(key)` / `is_pinned(key)` that exempt a key from eviction;
    `set_max_bytes(n)` to resize the budget on the fly;
    `current_bytes` / `max_bytes` / `__len__` accessors. Every cached
    array is now marked `writeable=False` on insert so an accidental
    in-place mutation downstream fails loudly with `ValueError`
    instead of silently corrupting the next hit. `clear()` drops both
    entries and pins. Eviction skips pinned entries and breaks out of
    the loop when only pins remain — accepting a temporary over-budget
    state rather than evicting a displayed plane (the trade Phase 3
    deliberately makes).

  - `PrefetchManager` gains a velocity-biased T-axis window. The
    manager tracks the previous `(m, t, z, z_mode)` and its
    `time.perf_counter()` timestamp and derives a T-axis velocity in
    planes/sec from successive `request_neighbors` calls. Forward
    scrubs (≥ +1 planes/sec) widen the window to `[t - 2, t + radius_t]`;
    backward scrubs widen the opposite side; a stop falls back to a
    symmetric `radius_t // 2` window. A 1.5 s TTL on the previous-focus
    timestamp resets the bias after an idle pause. `radius_t` and
    `radius_z` are now constructor parameters (defaults stay at 5 for
    back-compat with V1.17 callers and unit tests).

  - `MultiAxisViewer` (1) picks `radius_t` adaptively from
    `nd2studios.utils.resources.detect().available_ram_gb` (2 below
    2 GB free, 4 below 8 GB, 8 above) and passes it to
    `PrefetchManager`; (2) tracks `_pinned_keys: set[tuple]` and
    pin/unpin-diffs the displayed channel keys around every
    successful `_compose_current_frame` so the on-screen composite's
    contributing planes are exempt from eviction; (3) clears
    `_pinned_keys` on `set_volume`, `set_channels`, and inside
    `_teardown_io_worker`'s callers; (4) gains a `cache_stats_text()`
    helper returning `"Cache: <used>/<budget> MB · hit <rate>% ·
    evictions <n>"` for the V1.33 profiling harness and a future
    status-bar tooltip; (5) drops the V1.17 `n=5` override on
    `PrefetchManager.request_neighbors` so the manager's adaptive
    radius takes effect.

  Plan: `CodeLog/ClaudesPlan/V1.35_phase3_frame_cache_prefetch.md`.

## [Unreleased] - 2026-05-22 (V1.34)

### Added

- **Phase 2 — Lazy Loading & Threading: foreground I/O worker + adaptive frame cache**
  (`nd2studios/utils/resources.py`, `nd2studios/utils/threading.py`,
  `nd2studios/widgets/multi_axis_viewer.py`):

  V1.33 profiling confirmed that despite the V1.0 worker model and the
  V1.17 prefetcher, one synchronous read path still lived on the GUI
  thread: every cache miss inside
  :meth:`MultiAxisViewer._compose_current_frame` called
  ``volume.get_frame(...)`` inline, so every fresh slider movement
  blocked the Qt event loop for ``n_channels`` disk reads + Z-projection
  + decode. The :class:`PrefetchManager` only filled the cache for
  *neighbors* of the previous coords, not the plane the user was about
  to display. This phase closes that gap and also lifts the
  hard-coded 300 MB cache budget the V1.17 :class:`FrameCache` shipped
  with.

  - New :mod:`nd2studios.utils.resources` (backend-pure, no Qt) —
    :class:`SystemResources` dataclass + :func:`detect`,
    :func:`recommended_cache_budget_bytes` (default reserves 60% of
    *available* RAM, hands 40% to caches, floored at 64 MB),
    :func:`recommended_worker_count` (physical cores capped at 8).
    Single source of truth for "available RAM" / "physical core count"
    used by Phase 2 cache sizing and ready for Phase 3 / 5 / 6 / 7.

  - New :mod:`nd2studios.utils.threading` (Qt boundary) —
    :class:`PlaneRequest` dataclass with :class:`~queue.PriorityQueue`-
    friendly ordering (priority + monotonic ID), :class:`IOWorker`
    (a :class:`QObject` moved to a dedicated :class:`QThread` with
    ``plane_ready(request_id, key, ndarray)`` / ``error(request_id,
    msg)`` signals), and :func:`start_io_worker(volume)` that wires
    a fresh thread + ``volume.reopen()`` handle. Handle ownership
    matches the existing :class:`PrefetchManager` convention — one
    ``nd2.ND2File`` per thread, never shared. Normalization to 2-D
    runs on the worker thread so the GUI slot stays trivial.

  - :class:`MultiAxisViewer` now (1) sizes its :class:`FrameCache` from
    :func:`recommended_cache_budget_bytes` instead of the V1.17 300 MB
    constant — on a 32 GB workstation with ~16 GB free the cache grows
    to ≈ 6.4 GB, twenty times the old cap, trading the RAM the user
    has spare for fewer scrub misses; (2) starts an :class:`IOWorker`
    alongside the :class:`PrefetchManager` on every ``set_volume()``,
    via the same volume ``reopen()`` factory; (3) replaces the inline
    ``volume.get_frame(...)`` cache-miss path in
    ``_compose_current_frame`` with a :meth:`IOWorker.submit` enqueue
    + ``cancel_all()`` of any stale foreground requests + bump of
    ``_latest_request_id``; (4) gains ``_on_io_plane_ready`` /
    ``_on_io_error`` / ``_teardown_io_worker`` slots. Stale
    ``plane_ready`` results still populate the cache (so an
    out-of-order arrival is not wasted work) but only redraw the
    canvas when ``request_id == self._latest_request_id``.
    ``set_channels`` (Recipe page path) and ``set_volume(None)`` both
    teardown the worker the same way they already teardown the
    prefetcher; ``_teardown_io_worker`` bumps the gating counter so
    in-flight planes arriving after teardown can't sneak a redraw.

  - The previous-frame stays on the canvas during the brief window
    where the worker is still reading — no flash, no synthetic
    placeholder. Slider input remains responsive across the whole
    gesture; the *displayed* frame may lag by one composite, which is
    the trade Phase 2 deliberately makes.

  Plan: `CodeLog/ClaudesPlan/V1.34_phase2_lazy_loading_threading.md`.

## [Unreleased] - 2026-05-22 (V1.33)

### Added

- **Phase 1 — Profiling & Measurement Baseline harness**
  (`nd2studios/utils/profiling.py`, `profiling/`):

  Backend-pure profiling helpers and a `profiling/` directory of
  driver scripts that capture a reproducible performance snapshot of
  the running application. No production code paths are modified;
  everything new is opt-in and lives outside `nd2studios/` except for
  a Qt-free helper module.

  - New :mod:`nd2studios.utils.profiling` exposes :class:`Measurement`
    (wall / CPU ms, RSS before / after / peak, Python-allocation peak,
    extra dict) and a :func:`measure` context manager built on
    :mod:`time.perf_counter`, :mod:`tracemalloc`, and :mod:`psutil`.
    :func:`fps_from_durations` turns per-frame ms into mean / p50 /
    p99 FPS and latency stats. Backend-pure — no PySide6 import.

  - New `profiling/harness/` package with one scenario per category:
    `scenario_load.py` (cold open + first frame on
    :class:`LazyND2Volume` / :class:`LazyMultiFileTIFFVolume`),
    `scenario_scrub.py` (per-frame `get_frame()` latency walking T / Z
    / M), `scenario_tab_switch.py` (offscreen Qt — instantiates
    :class:`MainWindow`, walks every key in :data:`Settings.PAGES`
    via ``_navigate()`` 12× and times each), and
    `scenario_analysis.py` (runs every registered
    :class:`AnalysisPipeline` on a deterministic 32 × 256 × 256
    synthetic channel so the scenario succeeds without lab data).

  - `profiling/harness/run_all.py` aggregates all four scenarios and
    writes a single ``profiling/baselines/<label>.json`` snapshot
    keyed by a CLI-provided label so subsequent phases can re-run and
    diff. Snapshot embeds platform / CPU / RAM via :mod:`psutil`.

  - Test files are opt-in via the ``PROFILING_TEST_DATA_DIR``
    environment variable (fallback ``~/profiling_data``). Scenarios
    skip cleanly when expected ND2 / TIFF files are missing so a
    clean checkout can still produce a snapshot (tab-switch and
    analysis scenarios both run without lab data).

  - `profiling/README.md` documents how to run the harness, drop in
    real lab files, and capture optional py-spy flamegraphs into
    `profiling/reports/`. Generated JSON / SVG / .prof outputs are
    gitignored; `.gitkeep` files preserve the directories.

  Plan: `CodeLog/ClaudesPlan/V1.33_phase1_profiling_baseline.md`.

## [Unreleased] - 2026-05-21 (V1.32)

### Bug Fixes

- **Stitched tile positions were doubly wrong**
  (`backend/exporters/stitch_exporter.py`):

  Two stacked bugs in `compute_tile_layout`'s physical-layout branch.
  (1) `offsets = list(reversed(offsets))` reversed the per-tile pixel
  offsets while leaving the tile image list in natural `m_indices`
  order; downstream zips (`stitch_one_frame`, `export_stitched_tiff`,
  the preview widgets in `tile_layout` / `tile_preview`) then paired
  M=k's pixels with M=last-k's canvas slot, point-mirroring every M
  across the canvas. (2) The stage-XY-to-pixel math assumed image-style
  conventions (smallest X → leftmost, largest Y → top), but this
  scope's Nikon stage uses the opposite sign on both axes, leaving
  tiles mirrored on X and Y even after fix (1). Removed the reverse
  and negate `sx`/`sy` up-front so the existing `min_x` / `max_y`
  arithmetic resolves to the correct quadrant.

### Added

- **Export preview dialog with brightness / contrast / saturation / hue / fade**
  (`backend/exporters/composite_exporter.py`,
  `backend/exporters/movie_exporter.py`,
  `backend/exporters/image_sequence_exporter.py`,
  `widgets/export_preview_dialog.py`,
  `workers/export_worker.py`, `pages/export_page.py`):

  New :class:`ImageAdjustments` dataclass (`brightness`, `contrast`,
  `saturation`, `hue`, `fade`) and :func:`apply_image_adjustments`
  vectorised helper in `composite_exporter`. **Brightness is applied
  per-channel** in the channel's grayscale (new `_apply_brightness_to_gray`)
  *before* the colour LUT is mixed in, so brightening a red-only channel
  stays pure red rather than washing toward white. Brightness is
  implemented as **multiplicative gain** (`gain = 1 + brightness/100`)
  rather than additive offset, so the background floor stays at zero
  and only above-floor signal scales visibly — the additive version
  lifted the noise floor and washed images out. Contrast / saturation /
  hue / fade still operate on the final RGB via `apply_image_adjustments`.
  `_composite_frame`, `export_rgb_composite_tiff`, and `export_movie` learn
  an `image_adjustments=` kw-only argument (default `None` = identity, no
  behaviour change for existing callers). `ImageAdjustments` gains
  `is_identity_post_composite()` so the composite pass can skip the
  post-composite RGB pass when only brightness is set.

- **Image-sequence exporter**
  (`backend/exporters/image_sequence_exporter.py`,
  `backend/exporters/__init__.py`, `workers/export_worker.py`,
  `pages/export_page.py`):

  New `"image_sequence"` export mode that writes one PNG per frame.
  :func:`format_frame_name` builds filenames as
  ``{basename}_TXX[_MXX][_ZXX].png`` — only axes with more than one
  frame contribute a suffix, and the digit width tracks the maximum
  index per axis. Two iteration paths:

  - Default: iterate ``T`` over the current channels (recipe-applied,
    same data the movie tab uses).
  - "Iterate all M / Z positions" checkbox (only enabled when the
    loaded file has multi-M or unprojected multi-Z): walk `(M, T, Z)`
    directly off the `LazyND2Volume` — raw pixels, recipe *not*
    applied, but channel LUT / colour / image-adjustments and overlays
    still apply.

- **`ExportPreviewDialog` with live playback**
  (`widgets/export_preview_dialog.py`):

  Modal preview that pops up after the user clicks "Export Movie…" or
  "Preview & Export Image Sequence…". Shows the composited frame in an
  embedded :class:`ImageCanvas` with a T scrubber, a ▶ Play / ⏸ Pause
  toggle and a per-dialog FPS spin box that drive a `QTimer` to auto-
  advance T — so the user actually sees the movie play back live, with
  brightness / contrast / saturation / hue / fade applied in real time
  on every frame. Manually grabbing the T slider pauses playback;
  closing the dialog stops the timer. The "Show overlays in preview"
  toggle hides scale bar / timestamp / channel labels when the user
  wants to see the raw composite. The dialog only collects values —
  the actual export is kicked off by the caller through the existing
  :class:`ExportWorker` once the dialog is accepted.

- **Export page — Image Sequence tab** (`pages/export_page.py`):

  New fourth tab beside Movie. Provides a "Base name" line edit (auto-
  populated from the loaded filename), the iterate-all-axes checkbox,
  and a live "Will write N PNG files…" summary. The Movie tab is
  unchanged except that "Export Movie…" now opens
  :class:`ExportPreviewDialog` before invoking the worker.

## [Unreleased] - 2026-05-20 (V1.30)

### Added

- **Manual Mask analysis pipeline + canvas drawing tools**
  (`backend/analysis/manual_mask.py`, `widgets/image_viewer.py`,
  `widgets/multi_axis_viewer.py`, `widgets/common.py`,
  `pages/analysis_page.py`, `core/experiment_manager.py`):

  New `ManualMaskPipeline` registered under the name "Manual Mask".
  The Analysis tab shows a Drawing tools group whenever Manual Mask is
  selected, with three exclusive shape buttons (`Rectangle`, `Ellipse`,
  `Polygon`) and three frame ops (`Clear frame`,
  `Copy from previous frame`, `Apply to all frames`).

  - `ImageCanvas` gains `set_draw_mode(mode)` and a `shape_drawn(str, list)`
    signal. Three modes — `"rect"` and `"ellipse"` are rubber-band drag;
    `"polygon"` is freehand drag that samples vertices ~ every 2 image
    pixels and auto-closes on release. Pan / crop / draw remain mutually
    exclusive.
  - `MultiAxisViewer` delegates `set_draw_mode` and re-emits `shape_drawn`.
  - `ParamEditor` learns a new `"hidden"` `ParamSpec` type for
    internal-state slots that should round-trip via params but never
    render a widget. Used for the manual-pipeline's `frame_shapes` slot.
  - `AnalysisPage` keeps `_manual_shapes: Dict[m, Dict[t, List[Shape]]]`
    in memory, paints a live yellow overlay rasterized on each
    `paintEvent`, and injects the per-M shape dict into params at
    `_start_next_m_run` so the pipeline stays pure.
  - `ND2StudiosRecord.manual_mask_shapes` persists drawings to `.nd2s`
    (int keys are stringified to JSON and converted back on load).
  - `backend.analysis.manual_mask.rasterize_shapes()` is the shared
    skimage-based rasterizer used by both the pipeline and the live overlay.

- **Mask editing for Manual Mask shapes — vertex drag + uniform expand**
  (`widgets/image_viewer.py`, `widgets/multi_axis_viewer.py`,
  `pages/analysis_page.py`, `backend/analysis/manual_mask.py`):

  The Drawing tools panel gains an **Edit shape** toggle, a **Shape**
  selector (when multiple shapes exist on a frame), and an **Expand by
  (px)** spinbox + Apply button.

  - Toggling Edit shape converts the selected shape into a polygon with
    24 evenly-spaced vertices (sampled by arc length) — rectangles trace
    their perimeter, ellipses sample the parametric outline, polygons
    are resampled. Draggable handles appear on the canvas; clicking and
    dragging a handle deforms the mask locally and updates the live
    overlay in real time.
  - Apply expand grows (positive) or shrinks (negative) the shape
    uniformly via `scipy.ndimage.binary_dilation`/`erosion` on the
    rasterized mask, then re-extracts the outline with
    `skimage.measure.find_contours` and resamples to 24 vertices. Works
    correctly for arbitrary polygons including concave outlines.
    Falls back to the original vertices if erosion eats the entire mask.
  - Edit mode is mutually exclusive with pan / crop / draw tools.
    Drawing a new shape, changing frames, clearing the frame,
    copy-from-prev, apply-to-all, or switching pipelines all exit edit
    mode cleanly.
  - New canvas signals: `vertex_moved(int, float, float)`,
    `edit_committed()`. `ImageCanvas.set_edit_vertices(vertices)`
    enters/leaves edit mode with a list of `(iy, ix)` handle positions.
  - New backend helpers in `backend/analysis/manual_mask.py`:
    `shape_to_editable_polygon`, `expand_polygon_uniformly`,
    `_resample_closed_polygon`, plus the constant
    `DEFAULT_EDIT_VERTEX_COUNT = 24`.

- **ΔArea per frame in `compute_measurements`**
  (`backend/results_engine.py`):

  Results tab now reports `delta_area_px` and `delta_area_um2` for every
  pipeline (manual, threshold, nuclei, spots). After the regionprops pass
  rows are sorted by `(segmentation_channel, label_id, frame)` and
  consecutive areas are differenced; the first row of each track gets
  `None`. The behaviour is exact for single-object and manually-drawn
  masks; for instance segmentations whose label ids are not stable across
  frames it is "label k between consecutive frames", which is documented
  in the column comment.

- **Per-Z mask drawing + volume / ΔVolume metrics**
  (`backend/analysis/manual_mask.py`, `core/analysis_registry.py`,
  `backend/results_engine.py`, `pages/analysis_page.py`,
  `pages/results_page.py`, `core/experiment_manager.py`):

  Manual Mask shapes are now keyed by Z slot in addition to (M, T).
  Each frame holds `{z_key: [shapes]}` where `z_key` is either an
  integer Z index or the string `"all"` meaning "applies uniformly to
  every Z slice". Routing of new shapes is automatic:
  - **Projected viewing** (z_mode = max / mean / min, or single-Z
    files): new shapes go under the `"all"` slot. The pipeline
    replicates each "all" shape across every Z slice when computing
    volume — i.e., the uniform-mask assumption.
  - **Per-Z viewing** (z_mode = "none" on a multi-Z file): new shapes
    go under the current integer Z index. Each Z slice can carry its
    own distinct mask.

  Results tab gains two new columns for every pipeline:
  - `volume_um3` — true 3D voxel count × `pixel_size_um²` × `z_step_um`
    when the pipeline supplies `AnalysisResult.volumetric_voxel_counts`
    (Manual Mask does this whenever any per-Z shape exists). Otherwise
    falls back to `area_um2 × n_zslices × z_step_um`, which is the
    correct interpretation for projected masks under the uniform-Z
    assumption.
  - `delta_volume_um3` — per-track frame-to-frame volume change,
    computed alongside the existing `delta_area_*` columns.

  Other changes:
  - `AnalysisResult.volumetric_voxel_counts: Optional[Dict[ch, Dict[(t, label_id), int]]]`
    is the new optional pipeline output.
  - Drawing status bar reports the current Z target
    (`Frame T=5, Z=3, M=0: 1 at this Z + 2 all-Z`).
  - Edit-shape combo lists each shape with its slot
    (`Shape 1 (all-Z)`, `Shape 2 (Z=3)`); vertex drag and uniform
    expand operate on the chosen slot.
  - Live overlay only renders shapes visible at the current Z (every
    `"all"` shape plus current-Z shapes), so navigating Z in per-Z
    mode reveals slice-specific masks.
  - `ND2StudiosRecord.manual_mask_shapes` schema upgraded; V1.30
    sessions auto-migrate on load (the bare list value becomes
    `{"all": [shapes]}`).
  - Helper `_voxel_counts_for_frame` rasterizes each Z slice in turn
    while honoring label-order overwrites, so the reported
    `volume_um3` is consistent with the visible 2D `area_um2` reported
    by regionprops on the union mask.

### Bug Fixes

- **Single-file TIFF loader is now Z-aware** (`backend/tiff_loader.py`,
  `workers/load_worker.py`):

  Before, the loader silently collapsed Z to 1 on re-import of any
  multi-channel multi-Z ImageJ TIFF (including stitched outputs).
  `load_imagej_tiff_channels` built per-channel `LazyTIFFChannel`
  proxies with `n_pages_per_t=n_z*n_c, page_within_t=c_idx` but no
  `n_z` — so every T read page `z=0` only. The existing multi-Z code
  in `LazyTIFFChannel._read_frame` also used `base_page + z` (stride
  1), which is wrong for ImageJ TZCYX layout (Z stride per channel =
  `n_c`).

  - `LazyTIFFChannel` gains a `z_stride: int = 1` parameter; the Z
    accumulator now reads `base_page + z * z_stride`.
  - `load_imagej_tiff_channels(z_projection="max")` passes `n_z`,
    `z_stride=n_c`, and the requested projection through to each
    per-channel proxy. Per-channel `(T, H, W)` views are projected
    across Z correctly.
  - `_SingleFileTIFFView` caches an internal `tifffile.TiffFile`
    handle (`_ensure_tiff`), reads pages directly for multi-Z files
    via `page = t*n_z*n_c + z*n_c + c` in `get_frame`, and builds
    `to_lazy_channel` proxies that honor `z_mode`/`z_index`/Z range.
  - `LazyMultiFileTIFFVolume.to_lazy_channel` was dropping
    `z_mode`/`z_index`/`z_*`/`t_*` for chain `C`/`M`/`Z`; now passes
    them through to the underlying view.
  - `LoadWorker._load_tiff` always builds a
    `LazyMultiFileTIFFVolume([filepath], chain_axis="Z")` and reads
    `n_zslices`, `n_timepoints`, `n_channels` etc. from it. Returns
    `volume=...` so the Import page's multi-axis viewer can scroll Z.

  A stitched TIFF written with `z_mode="none"` now re-imports with its
  full Z stack intact and scrolls in the multi-axis viewer just like
  the source ND2.

- **Stitched TIFF now preserves Z stacks when `z_mode = "none"`**
  (`backend/exporters/stitch_exporter.py`, `pages/stitch_dialog.py`):

  `export_stitched_tiff` always pre-allocated `(T, 1, C, H, W)` and read a
  single Z slice via `volume.get_frame(z=z_index, z_mode=z_mode)`,
  collapsing Z even when the user selected `z_mode="none"` in the Stitch
  dialog (which is meant to keep every Z plane). The exporter now
  computes `preserve_z = (z_mode == "none" and volume.n_zslices > 1)`,
  allocates `(T, Z, C, H, W)` accordingly, and adds an inner Z loop that
  reads `volume.get_frame(z=z_out, z_mode="none")` for each plane.
  BigTIFF threshold updated to include the Z dimension. Projection modes
  (`max`/`mean`/`min`) and single-Z files still produce `(T, 1, C, H, W)`.

  Stitch dialog cleanup: the "Z slice (when 'none')" `QSpinBox` is
  removed — it had no effect under projection modes and is bypassed by
  the new Z-preserving path. Replaced by a hint label that describes
  what each Z mode does for the current file (`n_zslices` planes
  preserved vs collapsed via X-projection). `combo_z.currentTextChanged`
  is wired so the size estimate in `_refresh_summary` reflects the
  ×`n_z` blow-up when Z is preserved.

## [Unreleased] - 2026-05-20 (V1.29)

### Changed

- **Z-Projection TIFF export now matches Stitched TIFF file construction**
  (`backend/exporters/tiff_exporter.py`,
  `backend/exporters/stitch_exporter.py`,
  `backend/exporters/__init__.py`,
  `workers/export_worker.py`,
  `pages/export_page.py`):

  Both writers now produce a single multi-channel ImageJ TZCYX hyperstack
  per export instead of diverging shapes. New
  `export_tiff_hyperstack(channels, enabled, filepath, bit_depth,
  pixel_size_um, progress_cb)` accepts the channel dict directly, promotes
  `(T, H, W)` inputs to `(T, 1, H, W)`, validates shape consistency across
  channels, applies the bit-depth selector (per-channel 0.5–99.5
  percentile stretch for `uint8`/`uint16`), and writes one
  `(T, Z, C, H, W)` array via `tifffile.imwrite(imagej=True, ...)` with
  `Labels=[<channel names>]`, `unit="um"`, `spacing=pixel_size_um`,
  resolution, and `resolutionunit="MICROMETER"`. BigTIFF auto-switch
  above 3.9 GB.

  `ExportWorker._export_tiff_stack` and `_export_tiff_zstack` are
  rewritten to call the new writer once with the full channel dict —
  the per-channel filename-suffix loop (`_<channel>.tif`) is gone. Both
  return the single output path; the export-complete dialog now shows
  one filename instead of a semicolon-joined list.

  `export_stitched_tiff` adds `spacing=pixel_size_um` to its ImageJ
  metadata so the two exporters declare resolution identically.

  `export_tiff_stack` is kept (still used by `analysis_page.py` for
  label-mask export — single-channel, doesn't need ImageJ wrapping).

  Reload compatibility: `tiff_loader.read_imagej_tiff_metadata` +
  `load_imagej_tiff_channels` already understand `(T, Z, C, H, W)`
  ImageJ hyperstacks with `Labels`, so files round-trip into the Import
  page exactly the same regardless of which exporter wrote them.

## [Unreleased] - 2026-05-20 (V1.34)

### Bug Fixes

- **Recipe Trial no longer appears to freeze on large reconstructed
  imports** (`workers/recipe_worker.py`):
  `RecipeWorker.run_task` used to call `data.materialize()` in one
  shot before running any plugin step.  For TIFF reconstructions
  whose member files are full-canvas stitched outputs the materialised
  array can reach hundreds of MB to a few GB — disk reads of that
  size are slow but not hung, but with no progress emitted between
  the "Trial running…" status and the first plugin step the user
  saw nothing for tens of seconds and assumed Trial was stuck.

  The worker now reads lazy proxies (`LazyND2Channel`,
  `MultiFileLazyChannel`, `LazyTIFFChannel`, plain ndarrays) frame by
  frame via `data[t]` indexing, emitting per-frame `set_status`
  ("Channel Cy5: read 12/35") and `set_progress` so the GUI shows
  steady advancement during the load phase.  `cancelled` is polled
  every frame so the Trial Cancel button now interrupts the read
  immediately instead of waiting for materialization to finish.

  One-shot `.materialize()` is kept as a fallback for proxies that
  don't expose `__getitem__` or whose per-frame shape can't be
  stacked (rare; the codebase's lazy proxies all support indexing).
  Output is byte-identical to the previous path.

## [Unreleased] - 2026-05-20 (V1.33)

### Bug Fixes

- **MP4 export of stitched-panorama time-lapses now plays in any
  media player** (`backend/exporters/movie_exporter.py`):
  `export_movie` previously handed libx264 the full canvas at its
  native resolution.  Stitched panoramas easily exceed H.264 Level
  6.2's frame-MB / DPB limits — a 6144×11264 canvas hit
  `frame MB size (384x704) > level limit (139264)` and
  `DPB size > level limit (2 frames, 696320 mbs)`, producing an MP4
  whose stream rate ffprobe couldn't even estimate.

  The exporter now caps the longest output edge at 3840 px via a new
  `_downscale_for_mp4` helper (PIL LANCZOS resize, dimensions
  rounded to even for yuv420p chroma subsampling) before feeding the
  frames to ffmpeg.  Overlays are drawn at full canvas resolution and
  resized along with the frame, so scale bars, timestamps, and
  channel labels keep their relative position and visual weight.
  libx264 is now invoked with explicit `-level 6.2 -refs 2 -bf 0` so
  the encoder's DPB stays inside the level budget at the capped size.

  GIF / TIFF / other codec paths are unchanged — only the MP4 branch
  applies the cap, since H.264 is the only one with a hard frame-MB
  ceiling.  A round-trip test with a synthetic (4, 11264, 6144, 3)
  RGB stack now produces `h264 High yuv420p 2094×3840 10fps` that
  decodes without warnings.

## [Unreleased] - 2026-05-20 (V1.32)

### Bug Fixes

- **Stitch export no longer allocates the entire TZCYX hyperstack in RAM**
  (`backend/exporters/stitch_exporter.py`):
  `export_stitched_tiff` previously called
  `np.zeros((n_t, n_z, n_c, canvas_h, canvas_w), dtype=...)` and only
  then walked the volume to fill it.  For a typical multi-file T-chain
  composite this could blow past available RAM — a `(T=35, Z=10, C=2,
  11264, 6144)` uint16 export OOM-killed at 90.2 GiB before any pixels
  were read.

  The exporter now builds the output via `tifffile.memmap(...,
  imagej=True, bigtiff=...)` which allocates the file on disk and
  returns an ndarray view backed by `mmap`.  The per-(T, Z, C) loop
  reads the M tiles for one frame, stitches the canvas (≈ 132 MB at
  the sizes above), and assigns it into the memmapped view; the OS
  pages canvas frames in/out as needed.  Peak working set drops from
  ~90 GiB to roughly one canvas frame plus the M tile reads for that
  frame.  Output remains a valid Fiji-readable ImageJ TZCYX hyperstack
  with the same channel labels, resolution, and pixel-size metadata
  as before.

## [Unreleased] - 2026-05-20 (V1.31)

### Added

- **Drag-reorderable assembly blocks in the Reconstruct dialog**
  (`pages/reconstruct_dialog.py`):
  The dialog now shows three rows of colored blocks below the file
  table — one row per axis (T, M, Z) — and the chain-axis row is
  drag-and-drop reorderable.  Each block represents one source slice
  on that axis; blocks are colored per source file via an
  HSV-distributed `file_color_palette(n)`.  Hovering a block surfaces
  its full filename, source-frame index, and assembled output index
  ("Output T=2") via tooltip.  Switching the chain-axis combo flips
  drag enable to the correct row; non-chain rows are read-only
  context with a neutral gray fill.

  Implemented as a new `AxisBlocksWidget(QListWidget)` with
  `setFlow(LeftToRight) + setWrapping(True)` and Qt's built-in
  `InternalMove` drag mode.  Each item carries
  `(file_idx, local_idx)` in `Qt.UserRole`; the widget exposes
  `mapping()` to read back the current display order.

- **Per-slice chain mapping in the multi-file backend**
  (`backend/nd2_volume.py`, `backend/tiff_loader.py`,
  `backend/nd2_loader.py`, `workers/load_worker.py`,
  `pages/import_page.py`):
  Both `LazyMultiFileND2Volume` and `LazyMultiFileTIFFVolume` accept
  an optional `chain_mapping: list[tuple[int, int]]` where entry *i*
  is the `(file_idx, local_idx_in_file)` for output slice *i*.  When
  omitted the natural `(0,0), (0,1), …, (1,0), …` ordering is used
  (V1.28 behavior; regression-safe).  `_lookup_chain(idx)` is now a
  direct list index instead of a bisect on cumulative sizes, so
  arbitrary interleavings work in O(1).  A new
  `_validate_chain_mapping()` rejects out-of-range pairs with a clear
  message naming the bad entry.

  `read_nd2_metadata_extended_multi(paths, chain_axis, chain_mapping=None)`
  walks the mapping when assembling per-axis overrides, so
  `frame_timestamps_s`, `stage_xy_um`, `stage_z_um`, and concatenated
  channel name / exposure / emission / excitation / color lists all
  reflect the user's chosen order.  The reader now probes every file
  (not just file 0) so mappings that pull slices from any file get
  the right per-slice metadata.

  `LazyND2Volume.file_z_coords_um()` returns Z coords in mapped
  output order (the previous "raw file order" call moved to a new
  private `_raw_file_z_coords_um()` helper).

  `reopen()` round-trips the mapping so the prefetch worker's reader
  factory rebuilds the composite exactly.

  `LoadWorker` accepts `chain_mapping` and threads it into both the
  metadata reader and the composite-volume constructor.

  `ImportPage._on_reconstruct` reads `dialog.chain_mapping()` and
  forwards it; the rest of the app sees the same composite-volume
  surface as before.

## [Unreleased] - 2026-05-20 (V1.28)

### Added

- **Universal multi-file reconstruction with chosen chain axis**
  (`backend/nd2_volume.py`, `backend/nd2_loader.py`,
  `backend/tiff_loader.py`, `workers/load_worker.py`,
  `pages/reconstruct_dialog.py` (new), `pages/import_page.py`):

  The Import page gains a **Reconstruct from multiple files…** button
  that opens a modal where the user picks N ND2 *or* N TIFF files
  (mixed formats rejected), sees each file's `(T, M, Z, C, H × W,
  dtype)` in a table, and explicitly chooses the chain axis
  (`T` / `M` / `Z` / `C`).  Filename heuristics (`Count####`,
  `ZStack####`, `T####`, `Z####`, `M####`, `Pos####`, `Ch####`, etc.)
  preselect the axis; the user can override.  The dialog disables
  Load when any non-chain axis disagrees across files and names the
  offending axis(es).

  `LazyMultiFileND2Volume` (originally V1.27, Z-only) now takes a
  `chain_axis: str = "Z"` argument and validates that every axis
  *except* the chain axis matches across files.  A cumulative-offset
  table over the chain axis routes each `(c, m, t, z)` request to its
  owning file via `bisect.bisect_right`; each file contributes its
  *native* size on the chain axis (sum across files = combined size).
  `MultiFileLazyZChannel` is renamed `MultiFileLazyChannel` (the old
  name is kept as an alias) and now carries `z_index` so chain-`T`
  with `z_mode='none'` pins a Z plane correctly.

  `read_nd2_metadata_extended_multi(filepaths, chain_axis="Z")`
  overrides only the chain-axis fields on top of file 0's metadata:
  Z → `n_zslices`/`z_step_um`/`stage_z_um`; T → `n_timepoints` and
  concatenated `frame_timestamps_s`; M → `n_multipoints` and
  concatenated `stage_xy_um`; C → `n_channels` and concatenated
  channel name / exposure / emission / excitation / color lists.

  New `LazyMultiFileTIFFVolume` and `read_tiff_meta_fast` provide the
  same surface for TIFF reconstructions, reusing `LazyTIFFChannel`,
  `read_imagej_tiff_metadata`, and `get_tiff_info` underneath.

  `LoadWorker` accepts `filepaths=` and `chain_axis=` parameters.
  New `_load_nd2_multi(...)` and `_load_tiff_multi(...)` branches
  handle the multi-file paths and emit `source_type="nd2_multi"` /
  `"tiff_multi"` alongside the composite volume.

### Changed

- **Browse reverts to single-file** (`pages/import_page.py`):
  V1.27 piggybacked multi-file Z-stack loading onto the Browse
  button; V1.28 moves that into the Reconstruct dialog (general
  axis control, explicit user choice).  Browse uses
  `QFileDialog.getOpenFileName` again — one file at a time.

  Motivating new dataset:
  `E:\ELISA_3D\20260515_141025_848\Count0000X_..._Seq000X.nd2` ×5,
  five files of `(P=66, Z=10, C=2, 1024×1024)` that together form a
  five-timepoint series.  Reconstruct… → Add files → axis `T`
  produces a `(M=66, T=5, Z=10, C=2)` composite volume.

  The V1.27 Z-chain regression (`F:\ELISA_in_LLS_Density_Test\…`,
  76 ZStack files) continues to work via Reconstruct… → axis `Z`.

## [Unreleased] - 2026-05-07 (V1.26)

### Changed

- **Results page — all-M measurement aggregation**
  (`pages/results_page.py`):
  "Compute Measurements" now iterates over every M position present in
  `exp.analysis_results[pipeline_name]` (a `Dict[int, AnalysisResult]` since
  V1.25) rather than reading only the currently viewed M.  For multi-M results
  each measurement row gains an `m_position` column.  A new helper
  `_channels_for_m(exp, m)` resolves the correct `(T, H, W)` channel arrays
  per M (using `LazyND2Volume` when available, falling back to flat channels).
  `_on_export_masks` and `_on_export_images` also iterate all M positions;
  for multi-M results each position is written to a `M00/`, `M01/`, …
  sub-directory so frames from different positions don't overwrite each other.

- **Batch worker — per-M analysis across full file**
  (`workers/batch_worker.py`):
  `_process_one` now probes each ND2 file for `n_multipoints` via
  `LazyND2Volume`.  When a file has more than one multipoint position the inner
  loop runs analysis, recipe application, measurement computation, and (if
  requested) image export independently for each M position.  Measurement rows
  gain a `m_position` column for multi-M files.  Image export sub-directories
  are suffixed `_M00`, `_M01`, … per position.  Single-M files (and non-ND2
  TIFF files) are unaffected — the loop runs once at M=0 exactly as before.

## [Unreleased] - 2026-05-07 (V1.25)

### Changed

- **Analysis page — whole-file run, per-M overlays, overlay toggle**
  (`pages/analysis_page.py`):

  - **Run Analysis now covers all M positions.**  Pressing "Run Analysis" builds
    a queue of every multipoint position (`vol.n_multipoints`) and processes
    them sequentially in chained `AnalysisWorker` threads.  Single-M files and
    TIFF/processed-channel files behave identically to before (queue length 1).
    The progress bar is scaled across all M positions
    (`(n_done × 100 + worker_p) / total`), and the status label shows
    "M 2/5: Running…" for multi-M files.

  - **Overlays appear on all analyzed frames.**  Results are stored in
    `self._results_per_m: Dict[int, AnalysisResult]` keyed by M index.
    `_composite_overlay` looks up `_results_per_m.get(m)` so the binary/label
    overlay is rendered for every M that has completed, not just the one that
    was visible when Run was pressed.  Partial results appear as each M
    finishes.  `exp.analysis_results[pipeline_name]` now stores
    `Dict[int, AnalysisResult]`; `_restore_results_from_exp` handles backward-
    compatible restore of the old single-`AnalysisResult` format.

  - **"Hide Overlays" toggle button** added to the top-right of the viewer
    panel (outside all pipeline group boxes).  The button is checkable; when
    active it sets `_overlay_visible = False`, changing its label to
    "Show Overlays" and suppressing both full-run and screen-frame overlays
    until toggled back.

  - **New helpers:** `_build_channels_for_m(exp, m, params)` extracts the
    per-channel `(T, H, W)` arrays for one M position (factored out of
    `_on_run`); `_start_next_m_run()` pops the queue and launches the next
    worker; `_on_m_progress(p)` scales per-worker progress;
    `_restore_results_from_exp(exp, name)` centralises result restore on
    tab-switch and pipeline-change.

  - **Screening unchanged:** "Screen frame" and "Auto-screen" remain
    single-frame only.  Export functions operate on `self._result` (last
    completed M result).

## [Unreleased] - 2026-05-07 (V1.24)

### Changed

- **Bright / Dark Spots pipeline — background mask overlay**
  (`backend/analysis/spots/config.py`, `backend/analysis/spots/identifier.py`,
  `backend/analysis/spots_pipeline.py`, `core/analysis_registry.py`,
  `pages/analysis_page.py`):
  New optional parameter `background_mask` (bool, default `False`).  When
  enabled, the pipeline produces a secondary binary mask covering all pixels
  **not** covered by detected spots.  The background mask is overlaid in neutral
  gray (RGB 160, 160, 160, alpha 0.25) on the viewer — rendered beneath the
  spot overlay so detected regions remain clearly visible.  Per-frame background
  intensity statistics (`background_mean_intensity`, `background_std_intensity`)
  are appended to every measurement row in the CSV export so background signal
  can be compared to spot signal across frames.
  - `SpotsConfig.background_mask: bool = False` controls the feature.
  - `SpotsResult.background_labels: np.ndarray | None` carries the int32 inverse
    mask for a single frame (1 = background, 0 = spot region).
  - `AnalysisResult` gains three new fields: `secondary_label_masks`,
    `secondary_overlay_color`, `secondary_overlay_alpha`.  These are pipeline-
    agnostic; any future pipeline can produce secondary overlays via the same
    mechanism.
  - `_composite_overlay` in `AnalysisPage` iterates `secondary_label_masks` after
    compositing the primary spots mask, so both screen-frame and full-run
    overlays support secondary masks.

## [Unreleased] - 2026-05-07 (V1.23)

### Added

- **Batch page** (`pages/batch_page.py`, `workers/batch_worker.py`,
  `backend/template.py`): New "⚡ Batch" tab (always accessible, no import
  required) for running a saved pipeline template over many files and
  aggregating all per-object measurements into a single CSV.
  - **Pipeline template format** (`.nd2st.json`): JSON file capturing
    `import` (z_mode, frame_stride), `recipe` (ordered plugin steps),
    `recipe_normalized`, `analysis` (pipeline_name + params), and `results`
    (image_format) without any file-specific data.
    `backend/template.py` provides `save_template()`, `write_template()`,
    `load_template()`; constant `TEMPLATE_EXTENSION = ".nd2st.json"`.
  - **"Save current session as template"** button captures the active
    experiment's `import_config`, `recipe`, `analysis_config`, and
    `results_config` into a named `.nd2st.json` file.
  - **File queue**: add individual files or an entire folder (auto-scanned for
    `.nd2` / `.tif` / `.tiff`); duplicate-safe; per-item tooltip shows full path.
  - **BatchWorker** (`workers/batch_worker.py`): sequential per-file runner
    that calls `LoadWorker.run_task()` → `RecipeWorker.run_task()` →
    `pipeline.run()` → `results_engine.compute_measurements()` entirely
    within the worker thread (no nested `QThread`). Extra signal
    `file_done(int, int)` fires after each file; error in one file is caught
    and reported without aborting the rest. Aggregate CSV is written to
    `<output_dir>/batch_results.csv` with `source_file` as the first column.
  - **Optional per-file overlay image export** via
    `results_engine.export_overlay_frames()`; images land in
    `<output_dir>/<basename>/frame_NNNN.{tiff,jpg}`.
  - Batch config (`template_path`, `output_dir`, `image_format`,
    `export_images`) stored in `ND2StudiosRecord.batch_config` and
    round-trips through `.nd2s` session save/load.

## [Unreleased] - 2026-05-07 (V1.22)

### Added

- **Results page** (`pages/results_page.py`, `backend/results_engine.py`):
  New "📊 Results" tab (prereq: file imported) for computing extended
  per-object measurements from analysis binaries and exporting to CSV /
  overlay images / label-mask TIFFs.
  - **`backend/results_engine.py`** (pure, no Qt): `compute_measurements()`
    takes `label_masks`, `channels`, `metadata`, `m_index` and returns a
    `List[Dict]` with columns:
    `segmentation_channel`, `frame`, `label_id`, `area_px`, `area_um2`,
    `centroid_y/x_px`, `centroid_y/x_um`,
    `centroid_x/y_stage_um` (when `stage_xy_um[m_index]` present in metadata),
    `perimeter`, `eccentricity`, `solidity`, `bbox_*`,
    `mean_intensity_{ch}` / `std_intensity_{ch}` for every channel.
    Uses `skimage.measure.regionprops` (no new dependencies).
    Also exports: `export_overlay_frames()` (uint8 RGB composite + cyan mask
    overlay, written with `imageio`); `export_label_masks_tiff()` (int32
    stacks via `tifffile`).
  - **ResultsPage UI**: pipeline selector (`analysis_results` keys),
    "Compute Measurements" button, sortable `QTableView` backed by a custom
    `QAbstractTableModel` with proxy sort, summary panel (n_objects, n_frames,
    mean/std area_um2, per-channel mean intensity), export row
    (CSV / Label Masks TIFF / Overlay Images TIFF or JPG).
  - Results config (`pipeline`, `image_format`) stored in
    `ND2StudiosRecord.results_config` and round-trips through session save/load.
- **`ND2StudiosRecord`** gains two new serialized fields: `results_config`
  and `batch_config` (both `Dict[str, Any]`, persisted to `.nd2s` manifest).

## [Unreleased] - 2026-05-06 (V1.21)

### Added

- **Bright / Dark Spots analysis pipeline**
  (`backend/analysis/spots/`, `backend/analysis/spots_pipeline.py`):
  New `AnalysisPipeline` entry in the General Analysis tab. Reproduces the
  behaviour of Nikon NIS-Elements **General Analysis 3 → Bright Spots / Dark
  Spots** operators. Detection pipeline: (1) Difference-of-Gaussians (DoG) or
  Laplacian-of-Gaussian (LoG) band-pass response at the FWHM-matched scale
  derived from `typical_diameter_um`; (2) normalised contrast gate (response
  normalised to [-1, 1], threshold in [0, 1] is bit-depth-independent);
  (3) LUT-histogram intensity gate reusing `histothresh.histogram.percentile()`
  — bright: centroid pixel ≥ P(intensity_percentile); dark: pixel ≤ P;
  (4) circularity-based symmetry gate with four bins matching GA3's
  "All / More / Medium / Less objects" setting (floors: 0.0, 0.45, 0.65, 0.80);
  (5) two output modes — `circular` (rasterised disk of radius
  `typical_diameter/2`) and `region` (watershed-from-point-seed bounded by the
  intensity gate mask); (6) optional post-gate grow step (dilation or
  watershed). Measurements include the standard fields plus `diameter_px`,
  `contrast_score`, `circularity`, and `polarity`. No new external dependencies.
  Parameters: `channel_name`, `polarity`, `typical_diameter_um`, `contrast`,
  `symmetry`, `intensity_percentile`, `bit_depth`, `grow_radius_um`,
  `grow_method`, `output_mode`, `kernel` (11 total).
  Core logic (`validation.py`, `scale_space.py`, `detection.py`,
  `symmetry.py`, `grow.py`, `config.py`, `identifier.py`) is Qt-free.

- **Test suite** (`tests/spots/`): 102 pytest tests covering validation,
  sigma/FWHM resolution, DoG/LoG response signs, `peak_local_max` with
  intensity gate, h-transform seeds, circularity formula, symmetry gate
  monotonicity, dilation radius, watershed expansion, end-to-end
  `SpotsResult` contract, provenance integrity, synthetic precision/recall
  grid (5 diameters × 3 SNRs × 3 background types), blob_log parity,
  bright/dark polarity symmetry, and determinism (idempotence + hash
  sensitivity). All green, ~3 s total.

## [Unreleased] - 2026-05-05 (V1.20)

### Added

- **Histogram Threshold Segmenter analysis pipeline**
  (`backend/analysis/histothresh/`, `backend/analysis/histogram_threshold_pipeline.py`):
  New `AnalysisPipeline` entry in the General Analysis tab. Identifies image
  regions by intensity threshold using four methods — `hysteresis` (two-level
  strict/permissive), `single` (hard cut), `percentile` (resolved from the
  image histogram), and `relative` (fraction of within-frame median) — and
  four directions: `below`, `above`, `between`, `outside`. Spatial cleanup
  via morphological opening/closing, small-hole fill, and min-area filter.
  Operates on raw integer counts in the declared bit-depth range (default
  12-bit, 0–4095); rejects inputs whose values overflow the expected max
  (configurable strict/warn mode). Stain-agnostic; designed as a drop-in
  exclusion-mask producer alongside Tear Detection and Nuclei Segmentation.
  Parameters: `channel_name`, `method`, `direction`, `bit_depth`, `strict`,
  `permissive`, `low`, `high`, `percentile_low`, `percentile_high`,
  `fraction_low`, `fraction_high`, `min_area`, `opening_radius`,
  `closing_radius`, `min_hole_size` (16 total).
  Core logic (`validation.py`, `histogram.py`, `thresholds.py`,
  `morphology.py`, `config.py`, `identifier.py`) is Qt-free and independently
  testable.

- **Test suite** (`tests/histothresh/`): 26 pytest tests covering bit-depth
  validation, histogram computation, all threshold methods and directions,
  morphological filters, and end-to-end identifier runs (2-D, percentile,
  determinism, provenance). Zero warnings.

### Changed

- **Analysis tab: live file preview before running analysis**
  (`pages/analysis_page.py`, `widgets/multi_axis_viewer.py`):
  The Analysis page now embeds a full `MultiAxisViewer` (T/M/Z sliders,
  channel toggle + color chips, collapsible LUT histogram sidebar) so the
  file is visible as soon as the user enters the tab — no need to press
  "Run" first. Layout changed from vertical stack + bottom splitter to a
  horizontal splitter: controls + results on the left (~38 %), viewer on
  the right (~62 %). After analysis, a semi-transparent label-mask overlay
  is applied on the viewer for each frame via the new
  `MultiAxisViewer.set_frame_post_process()` hook (replaces the old
  `ImageCanvas` + standalone frame slider). Navigating the T/M/Z sliders
  while results are active updates the overlay automatically.
  The old `_render_frame()` / `_on_frame_changed()` methods and the
  dedicated `ImageCanvas` / `ZoomToolbar` / frame-slider widgets were removed.

- **`MultiAxisViewer.set_frame_post_process(fn)`** (`widgets/multi_axis_viewer.py`):
  New public method. Accepts a callable `fn(rgb_uint8, t, m) -> rgb_uint8`
  applied after channel compositing but before `canvas.set_image()`. Pass
  `None` to clear. Used by `AnalysisPage` to overlay label masks; the
  shared widget itself has no analysis-specific knowledge.

- **Analysis tab: frame screening** (`pages/analysis_page.py`):
  New "Screen frame" button runs the selected pipeline on the single frame
  currently visible in the viewer and overlays the label mask immediately —
  no need to run the full multi-frame analysis. A "Auto-screen" checkbox
  enables automatic re-screening on every T/M/Z slider move (300 ms
  debounce). Screen results take priority for their specific frame; the
  full-analysis overlay (from "Run Analysis") covers all other frames.
  Stale screen completions from superseded slider moves are silently dropped
  via an `expected_t` check. The screen worker is always cancelled before a
  full analysis run or a pipeline change.

## [Unreleased] - 2026-05-05 (V1.19)

### Added

- **`TearDetectionPipeline`** (`backend/analysis/tear_detection.py`):
  Classical Phase-1 baseline for detecting dark / homogeneous tear regions.
  Four-stage pipeline: (1) tissue mask via Otsu on log-intensity at 4×
  downsample + Gaussian blur + morphological closing/fill/opening;
  (2) rank-normalised homogeneity score =
  `(1−norm_intensity) × (1−norm_variance) × (1−norm_entropy)` computed with
  `scipy.ndimage.uniform_filter`, `generic_filter(np.var)`, and
  `skimage.filters.rank.entropy`; (3) connected-component extraction with
  `min_area_um2` / `max_area_um2` filters; (4) optional weak-stain
  disambiguation using a second counterstain channel (15th-percentile tissue
  floor). All normalisation uses rank-based methods for cross-batch robustness.
  Params: `channel_name`, `counterstain_channel` (default `"None"`),
  `window_size_px` (default 20), `homogeneity_threshold` (default 0.55),
  `min_area_um2` (default 50), `max_area_um2` (default 0 = no limit).
  Measurements include: `frame`, `label_id`, `area_px`, `area_um2`,
  `centroid_y`, `centroid_x`, `mean_intensity`, `class`
  (`tear` / `weak_stain` / `unknown`), `homogeneity_score`, `solidity`,
  `eccentricity`. No additional ML dependencies.

- **Dynamic CSV fieldnames** (`pages/analysis_page.py`):
  The measurements CSV export now derives its column list from
  `measurements[0].keys()` rather than a hardcoded 7-field list, so all
  pipeline-specific measurement fields are exported automatically.

- **`counterstain_channel` param injection** (`pages/analysis_page.py`):
  `_load_params()` now injects live channel names into any `choice` param
  named `counterstain_channel` (prepending `"None"` as the first option),
  in addition to the existing `channel_name` injection.



- **General Analysis tab (Page 4)**
  (`core/analysis_registry.py`, `workers/analysis_worker.py`,
  `backend/analysis/nuclei_segmentation.py`, `pages/analysis_page.py`):
  New 🔬 Analysis page accessible from the sidebar after import. Hosts an
  extensible `AnalysisPipeline` framework (separate registry from the
  enhancement plugin system) with a `ParamEditor`-based parameter form,
  `AnalysisWorker` background runner, frame-by-frame label overlay preview,
  summary stats panel, per-frame table, and export buttons (int32 label TIFF
  via `export_tiff_stack(bit_depth="passthrough")` and measurements CSV via
  `csv.DictWriter`).

- **`AnalysisPipeline` ABC** (`core/analysis_registry.py`):
  `AnalysisPipeline.register` decorator, `get_pipelines()` / `get_pipeline()`,
  `get_params() -> List[ParamSpec]`, `run(channels, metadata, params,
  progress_cb, cancelled_cb) -> AnalysisResult`. `AnalysisResult` holds
  `label_masks` (`Dict[str, (T,H,W) int32]`), `measurements`
  (`List[Dict]`), `summary` (`Dict`). Registry is separate from
  `PluginBase._registry`.

- **`AnalysisWorker(BaseWorker)`** (`workers/analysis_worker.py`):
  Thin wrapper — calls `pipeline.run()` with `cancelled_cb=lambda: self.cancelled`
  so the backend stays Qt-free.

- **`NucleiSegmentationPipeline`** (`backend/analysis/nuclei_segmentation.py`):
  Per-frame 2D nuclei segmentation using Cellpose 3 (`model_type=nuclei` or
  `cyto3`, `gpu=False`). Params: `channel_name`, `model_type`, `diameter`
  (0 = auto), `min_area`, `max_area`. Each frame is 1st–99.9th-percentile
  normalized before `model.eval()`. `skimage.measure.regionprops` computes
  per-object `area_px`, `area_um2`, `centroid_y/x`, `mean_intensity`.
  `CELLPOSE_AVAILABLE` flag checked at import; `ImportError` with install
  instructions raised at `run()` time if absent.

- **`ND2StudiosRecord.analysis_config` and `.analysis_results`** fields
  (`core/experiment_manager.py`): in-memory only, not serialized to `.nd2s`.

## [Unreleased] - 2026-05-04 (rev 3)

### Bug Fixes

- **Reset Crop failing after tab navigation for TIFF files**
  (`pages/recipe_page.py`, `core/experiment_manager.py`): The pre-crop channel
  snapshot (`_original_raw_channels`) was a page-local attribute on
  `RecipePage`, so it was cleared every time the user navigated away and back.
  For ND2 sources the snapshot was reconstructed from `_raw_volume` on
  re-entry; for TIFF sources (no volume) the Reset Crop button was enabled but
  did nothing. Fixed by moving the snapshot to `ND2StudiosRecord` as
  `_original_raw_channels` (in-memory only, not serialized), where it persists
  for the lifetime of the session. Removed the `_raw_volume` reconstruction
  block from `on_activated()` — it is no longer needed.

- **Z mode not propagating from Import tab to Recipe/Export tabs**
  (`pages/import_page.py` — `save_to_experiment()`): `save_to_experiment()` now
  rebuilds `exp._raw_channels` from the raw volume (via
  `LazyND2Volume.all_channels_as_lazy()`) whenever it runs, using the current
  `combo_zproj` selection, M index, and Z index. Previously, changes to the Z
  mode combo made after clicking "Confirm Import" were saved to
  `exp.z_view_mode` but never reflected in the lazy channel proxies that Recipe
  and Export tabs read. Now any navigation away from the Import tab (or a
  session save) atomically flushes the current Z mode into the channel proxies.
  `_on_confirm()` simplified accordingly — the explicit `all_channels_as_lazy`
  block it previously owned is now fully delegated to `save_to_experiment()`.

### Added

- **Z stack TIFF export when Z mode = "none"**
  (`backend/exporters/tiff_exporter.py`, `workers/export_worker.py`,
  `pages/export_page.py`):
  - `export_tiff_stack()` now accepts `(T, Z, H, W)` arrays in addition to
    `(T, H, W)`. A 4-D stack is written as an ImageJ TZYX hyperstack (axes tag,
    pixel-size metadata, bigtiff threshold) so Fiji/napari can re-open it with
    all Z planes intact.
  - New `ExportRequest` fields: `raw_volume: Optional[Any]` and `m_index: int`.
  - New `ExportWorker._export_tiff_zstack()` method: reads every `(T, Z)` frame
    combination from the raw `LazyND2Volume` on the worker thread, assembles
    `(T, Z, H, W)` per channel, and calls `export_tiff_stack()`. Progress is
    reported in two phases: 0–50 % reading, 50–100 % writing.
  - `ExportPage.on_activated()` detects the Z stack case (`z_view_mode == "none"`
    and `n_zslices > 1`) and relabels the TIFF button to "Export Z Stack TIFF…"
    and appends a Z-slices note to the summary label.
  - `ExportPage._on_export_tiff()` forks: when a Z stack is applicable it builds
    a `tiff_zstack` `ExportRequest`; otherwise it follows the existing
    `tiff_stack` path unchanged.

### Bug Fixes (rev 2)

- **Z stack not recognized in Recipe tab raw viewer**
  (`pages/recipe_page.py`): Added `_refresh_raw_viewer()` helper. When
  `z_view_mode == "none"`, `n_zslices > 1`, a raw volume is available, and no
  crop is active, the raw viewer is now wired to the volume via `set_volume()`
  so the Z slider appears and Z scrolling works. When a crop is active, the
  viewer falls back to `set_channels()` with the cropped proxies (crop takes
  priority because the volume holds the full-size data).

- **Crop state survives tab navigation in Recipe page**
  (`pages/recipe_page.py` — `on_activated()`): `on_activated()` now reads
  `exp.crop_rect` instead of blindly resetting the crop UI to "No crop applied".
  If a crop is active, the status label and Reset button are restored, and the
  pre-crop channel reference is rebuilt from the volume so the Reset Crop button
  works correctly after navigating away and back.

- **Z stack export ignored crop → memory allocation failure**
  (`core/experiment_manager.py`, `workers/export_worker.py`,
  `pages/export_page.py`, `pages/recipe_page.py`):
  - `ND2StudiosRecord` gains a `crop_rect: Optional[Tuple[int,int,int,int]]`
    field (x, y, w, h — in-session, not serialized to .nd2s).
  - `RecipePage._apply_crop()` now writes `exp.crop_rect`; `_reset_crop()`
    clears it.
  - `ExportRequest` gains a `crop_rect` field passed to the Z stack worker.
  - `ExportWorker._export_tiff_zstack()` slices each frame
    `frame[cy:cy+h, cx:cx+w]` when `crop_rect` is set, so the exported
    hyperstack matches the cropped dimensions instead of the full raw volume.

## [V1.17] - 2026-05-03

### Added

#### Draggable sidebars, auto fit-to-window, and axis play buttons

- **`core/main_window.py`** — Body layout replaced from `QHBoxLayout` to a
  `QSplitter(Qt.Horizontal)` so the nav sidebar is draggable. `setFixedWidth` on
  the sidebar removed; `setMinimumWidth(SIDEBAR_COLLAPSED_WIDTH)` retained so
  the collapsed floor is enforced. Toggle animation unchanged (animates
  `minimumWidth`/`maximumWidth` on the sidebar widget, which the splitter
  respects). New `_on_sidebar_anim_done()` slot: after expansion removes the
  max-width cap (`setMaximumWidth(16777215)`) so the user can drag wider than
  240 px; also calls `_reset_active_viewer_zoom()`. New `_reset_active_viewer_zoom()`
  helper calls `canvas.reset_zoom()` on the active page's viewer (via
  `getattr(page, "viewer", None)`).

- **`widgets/multi_axis_viewer.py`** — Root layout replaced from `QHBoxLayout`
  to a `QVBoxLayout` wrapping a `QSplitter(Qt.Horizontal)`, stored as
  `self._viewer_splitter`. Left column (image + sliders + chips) and
  `LutSidebar` are the two splitter children (stretch factors 1 and 0).
  `LutSidebar.collapse_changed` connected to `canvas.reset_zoom()` so toggling
  the LUT panel fits the image to the newly sized canvas.
  Each M/T/Z axis row now also contains a `QPushButton("▶")` (checkable, 28 × 22 px)
  and a `QDoubleSpinBox` (0.1–30 fps, default 5 fps, suffix " fps", 72 px wide).
  `_make_axis_row()` returns five items `(row, slider, info, play_btn, fps_spin)`.
  Per-axis `QTimer` instances (`_m_timer`, `_t_timer`, `_z_timer`) drive
  `_axis_tick(slider)` which advances the slider mod its maximum (wraps to 0).
  `_set_axis_playing(axis, playing)` starts/stops the timer and sets the button
  text to ▶/⏸. `_configure_slider()` gains optional `play_btn` and `fps_spin`
  keyword args that control visibility and stop playback when an axis is hidden.
  Play buttons are stopped at the start of `set_volume()` and `set_channels()`.

- **`pages/import_page.py`** — Outer layout changed from `QHBoxLayout` to a
  `QVBoxLayout` wrapping a `QSplitter(Qt.Horizontal)`. Left metadata panel
  `setFixedWidth(420)` replaced with `setMinimumWidth(200)`; default sizes set
  to `[420, 900]`.

#### Frame-switching smoothness: prefetch cache + debounce

- **`backend/frame_cache.py`** (new) — `FrameCache`: thread-safe LRU cache
  keyed by `(c, m, t, z, z_mode)` storing normalized `(H, W)` numpy arrays.
  Backed by `collections.OrderedDict`; evicts oldest entries when total frame
  bytes exceed `max_bytes` (default 300 MB).  All public methods (`get`,
  `put`, `contains`, `clear`) protected by `threading.Lock`.

- **`workers/prefetch_worker.py`** (new) — `PrefetchManager(QThread)`:
  single background thread that fills `FrameCache` with ±`n` (default 5)
  nearest-neighbor T frames (falls back to Z when `n_timepoints == 1`).
  Opens its own `LazyND2Volume(filepath)` inside `run()` — never shares the
  main-thread file handle.  Queue is replaced (not appended) on each call to
  `request_neighbors(m, t, z, t_range, z_range, z_mode, n)` so a new slider
  position supersedes the previous prefetch plan.  Emits
  `frame_ready(c, m, t, z, z_mode)` as a queued signal to the main thread
  after each cache fill.

- **`backend/tiff_loader.py`** — `LazyTIFFChannel._read_frame` previously
  called `tifffile.imread(path, key=page)` on every frame, opening and
  closing the file each time.  Now adds `self._tiff: tifffile.TiffFile`
  attribute, `_ensure_open()` (lazy first-use open), and replaces the read
  body with `self._tiff.pages[page].asarray()`.  Adds `close()` and
  `__del__` mirroring `LazyND2Volume`.  The `crop()` path is unaffected —
  crop views call the parent's `_read_frame` via closure and inherit the
  cached handle.

- **`widgets/multi_axis_viewer.py`** — Added to `__init__`: three
  single-shot debounce `QTimer`s (`_t_debounce` 50 ms, `_z_debounce` 50 ms,
  `_m_debounce` 80 ms) separate from the existing playback timers.
  `FrameCache` instance (`_frame_cache`, 300 MB), `PrefetchManager`
  reference (`_prefetch`), and histogram sample cache (`_hist_cache` keyed
  by `(m, z_mode)`).  Slider handlers now check `_all_channels_cached()`
  first — a full cache hit calls `_do_refresh()` immediately with zero disk
  I/O; a miss starts the debounce timer.  `_on_m_changed` no longer calls
  `_populate_lut_samples_from_volume` directly; `_do_m_refresh()` (fired by
  `_m_debounce`) skips the resample if `(m, z_mode)` is in `_hist_cache`.
  `_compose_current_frame()` (volume path) now checks the cache first for
  each channel, falls back to `volume.get_frame()` on a miss, and stores the
  normalized 2D result; after compositing it calls
  `prefetch.request_neighbors(...)`.  `_on_prefetch_ready(c, m, t, z,
  z_mode)` re-renders if the arrived frame completes the current view.
  `set_volume()` and `set_channels()` stop and destroy the old
  `PrefetchManager` and clear all caches; `set_volume()` creates a new
  `PrefetchManager` for the ND2 path.

## [V1.16] - 2026-05-03

### Bug Fixes

#### Stitched TIFF multi-channel axis fix

- **`backend/exporters/stitch_exporter.py`** — `export_stitched_tiff()` now
  builds a `(T, Z=1, C, H, W)` array and writes it with `tifffile.imwrite(
  imagej=True)`.  tifffile's `imagej_shape()` for `ndim=5` returns
  `(frames, slices, channels) = (shape[0], shape[1], shape[2])`, correctly
  mapping T→frames and C→channels.  A 4D `(T, C, H, W)` array is silently
  mis-read as `(T, Z, H, W)` → `channels=1`; axes/metadata hints are
  ignored because tifffile re-derives dimensions from the array shape at
  write time.  Channel names are now copied from `volume.channel_names`
  into the IJMetadata `Labels` field so Fiji shows the original nd2 channel
  names on the C slider.

- **`backend/tiff_loader.py`** — `LazyTIFFChannel` gains two new constructor
  params `n_pages_per_t` and `page_within_t` (both default to 1/0, preserving
  backward compatibility).  `_read_frame` now computes
  `page = t * n_pages_per_t + page_within_t` so any channel from a
  multi-channel TIFF can be addressed lazily without loading adjacent pages.
  `crop()` copies the new attributes.  Added `read_imagej_tiff_metadata(filepath)`
  (reads `tif.imagej_metadata` to recover `n_t`, `n_z`, `n_c`, channel
  `Labels`, and `pixel_size_um` from the `XResolution` tag) and
  `load_imagej_tiff_channels(filepath, ij_info, ...)` (builds one
  `LazyTIFFChannel` per channel using the page-stride formula).

- **`workers/load_worker.py`** — `_load_tiff()` now calls
  `read_imagej_tiff_metadata` first.  If `n_channels > 1` it uses
  `load_imagej_tiff_channels` to build the proper
  `{channel_name: LazyTIFFChannel}` dict and sets `n_timepoints` and
  `pixel_size_um` correctly from ImageJ metadata; otherwise it falls back to
  the original single-channel TYX path.

## [V1.15] - 2026-05-02

### Bug Fixes

#### TIFF bit-depth preservation (display + export)

- **`widgets/lut_histogram.py`** — `LutHistogramWidget` now tracks
  whether contrast has been explicitly set (`_contrast_explicitly_set`
  flag, defaulting to `False`).  `set_contrast()` sets the flag; the
  auto-percentile step in `set_data()` is skipped only when the flag is
  set.  Previously the condition `_hi == _dtype_max` failed for any
  dtype whose natural maximum differs from the widget's 65535 initial
  default (e.g. float32 data), leaving the LUT at 0–65535 and causing
  narrow-range images (12-bit data in a uint16 container) to display as
  black.

- **`backend/exporters/stitch_exporter.py`** — `export_stitched_tiff()`
  now writes every selected channel as a separate `minisblack` page in
  the volume's original dtype (e.g. `uint16`).  The previous
  `_percentile_uint8` / RGB composite path has been removed; multi-
  channel exports produce an ImageJ-compatible grayscale hyperstack.
  `nbytes_per_frame` bigtiff threshold now accounts for `n_channels`.
  Unused import of `_percentile_uint8` removed.

- **`pages/stitch_dialog.py`** — `StitchRequest.rgb` is now always
  `False`; multi-channel exports no longer trigger the (removed) 8-bit
  RGB composite path.

- **`workers/load_worker.py`** — TIFF metadata dict now includes a
  `"dtype"` key so the Import page metadata table shows the correct
  bit depth instead of an empty cell.

## [V1.14] - 2026-05-02

### Changed

#### Physical tile layout + f.experiment primary XY source

- **`backend/nd2_loader.py`** — `read_nd2_metadata_extended()` now tries
  `f.experiment` (XYPosLoop) as the primary source for per-M stage XY
  positions before falling back to `frame_metadata()` per-M. The
  experiment loop stores the *planned* stage positions in structured
  metadata, making it more reliable than per-frame readback (which
  depends on correct flat-index arithmetic over `dim_order`). New inner
  helper `_read_xy_from_experiment()`. `stage_layout_source` now includes
  the method suffix, e.g. `"stage_xy:experiment"` or
  `"partial:61/121:frame_metadata"`.

- **`backend/exporters/stitch_exporter.py`** — `compute_tile_layout()`
  now uses pure physical coordinates (V1.14 algorithm):
  - `offset_x = round((stage_x − min_stage_x) / pixel_size_um)`
  - `offset_y = round((max_stage_y − stage_y) / pixel_size_um)` (Y flipped)
  - Canvas = bounding box of all tile corners.
  - Physical gaps between non-adjacent tiles appear as empty (black)
    pixels, faithfully representing the scan footprint as if the file
    were one reconstituted image.
  - `StitchLayout.source = "physical"`.
  - `_cluster_axis`, `_assign_index`, `_serpentine_layout` helpers
    retained but no longer called by the main layout path.

- **`pages/import_page.py`** — `_tile_layout_source_label()` updated for
  the new `"stage_xy:*"` source format and the `"physical"` layout label.
  Diagnostic now shows cols × rows and canvas pixel size.

## [V1.13] - 2026-04-30

### Changed

#### Row-flush, center-aligned tile layout
- **`backend/exporters/stitch_exporter.py`** — V1.12's pure physical
  layout placed tiles at exact stage-XY pixel offsets. That's
  faithful to the metadata, but for the user's Well3 file the X
  step within each row is **2 tile widths** (3519 µm vs 1759.5 µm
  tile width), so consecutive tiles in a row leave a tile-sized
  empty column between them. The user reported this as "unintentional
  horizontal spacing between each tile that is supposed to be
  connected to one another."
- New algorithm:
  1. Cluster Y values into rows (using V1.6's bimodal-jump
     `_cluster_axis()` with a half-tile-height cap).
  2. Within each row, sort tiles by `(stage X asc, M index asc)`.
  3. Place tiles **flush** within their row at columns `(0, 1, …)
     × tile_w`.
  4. **Center each row** in a canvas as wide as the widest row.
- Y still flipped: highest stage Y → row 0 (top of image).
- `StitchLayout.source = "row_flush"` for the new path; `"physical"`
  is preserved as a recognized label for backward compat.

### Why
- The user's hex-staggered acquisitions (and similar irregular ROIs)
  have legitimate X gaps between tiles in the metadata. V1.12
  honored those gaps; V1.13 collapses them so tiles touch within a
  row while preserving the row structure (Y) and the overall ROI
  shape (via centering).
- For revisits at identical (X, Y) — Well2's case — the algorithm
  groups them into the same row cluster, sorts by `(X, M)`, and
  produces a 4×17 layout where each row holds the four visits in
  M order. Same outcome the user wanted earlier.
- For dense regular mosaics where every column is populated in
  every row, the row-flush placement is identical to the V1.12
  physical placement — no regression for "normal" tile scans.

### Tile-source diagnostic
- `import_page.py`'s "Tile source" row now reads (e.g.)
  `"61 multipoints from stage XY · row-flush 7 × 13 layout
  (rows centered)"` for full-metadata files and
  `"61 of 121 multipoints have stage XY · row-flush 7 × 13 layout
  (rows centered)"` for partial-metadata files.

### Documentation
- V1.13 plan: `CodeLog/ClaudesPlan/V1.13_row_flush_layout.md`.

## [V1.12] - 2026-04-30

### Changed

#### Pure physical tile layout (replaces clustering + serpentine)
- **`backend/exporters/stitch_exporter.py`** —
  `compute_tile_layout()` rewritten to use **exact stage-XY pixel
  positions** rather than clustered grid centroids. The user's
  request:
  > "Load all M files, record their centroid, identify tile
  > landscape based on mapping of centroids, then, establish tile
  > size via image size, then generate the tile layout. Some M
  > files are diagonal from each other when going from row to row,
  > and sometimes M files are uneven, and I need your tile layout
  > to be accurate in relation to the metadata."
- New algorithm (the entire stage-XY path is now four lines):
  ```python
  px = ((xs - xs.min()) / pixel_size_um).round().astype(int)
  py = ((ys.max() - ys) / pixel_size_um).round().astype(int)  # Y flipped
  canvas_w = int(px.max() + tile_w)
  canvas_h = int(py.max() + tile_h)
  ```
- All the V1.6 + V1.7 + V1.8 + V1.10 cluster / fallback machinery
  (bimodal-tolerance clustering, horizontal expansion, serpentine
  with three trigger reasons, fill-ratio detection) is **no longer
  invoked**. Helper functions (`_cluster_axis`, `_assign_index`,
  `_serpentine_layout`) remain in the file as orphans for forward
  compatibility, but the active code path doesn't call them.
- **`StitchLayout.source` gains `"physical"`** as the canonical
  stage-XY label. Old labels (`"stage_xy"`, `"stage_xy_expanded"`,
  `"serpentine"` with reasons) are no longer emitted by
  `compute_tile_layout`. They remain valid string values on the
  dataclass for backward compatibility with serialized state.

### Why
- Diagonal scans, uneven spacing, and sparse / irregular ROIs were
  getting snapped to discrete grids that misrepresented the
  acquisition. The user explicitly asked for "accurate in relation
  to the metadata" — pure physical placement preserves whatever the
  stage XY data says, no fabrication, no smoothing.
- Revisits at the same (x, y) now visually overlap (multiple tiles
  draw at the same pixel offset). This is faithful to the metadata;
  click-to-navigate returns the topmost M, and the M slider can step
  through the others.

### Kept

- **V1.11 partial-metadata truncation:** when `len(stage_xy_um) <
  len(m_indices)`, narrow `m_indices` to those with valid metadata.
- **V1.9 frame-shape normalization** in the viewer (RGB / Z / N-D
  input frames coerced to 2D before composition).
- **V1.5 grid fallback** (sqrt row-major) when there's no stage XY
  data at all.

### Documentation
- V1.12 plan: `CodeLog/ClaudesPlan/V1.12_pure_physical_layout.md`.
- `import_page.py`'s "Tile source" diagnostic simplified:
  `"61 of 121 multipoints from stage XY · physical layout 13314 × 13315 px"`
  for partial-metadata files;
  `"68 of 68 multipoints from stage XY · physical layout 3072 × 16384 px"`
  for full-metadata files.

## [V1.11] - 2026-04-30

### Bug Fixes

#### Files where `f.sizes['P']` overstates the metadata count
- **Diagnosis (via `scripts/inspect_nd2.py` against the user's
  Well3_…Cy5,TD_Seq0000.nd2):** `f.sizes` reports `P=121` but only
  61 multipoints have valid `frame_metadata` (M=61..120 return
  `None`). 24 GiB on disk → all 12342 frames are written, so the
  *image data* is intact; only the per-frame *metadata* is missing
  for the trailing 60 multipoints. Most likely: acquisition was
  stopped after position 61 but the file's `sizes` field still
  reports the planned 121.
- The diagnostic itself produced a clean V1.10 serpentine layout
  (5 × 13, 61 unique offsets) for the 61 valid positions because it
  passed `m_indices = None` (defaulting to `range(len(stage_xy))`).
- **The GUI was wrong** because it passes `m_indices = range(121)`
  while `stage_xy_um` only has 61 entries. The first guard in
  `compute_tile_layout` — `if len(stage_xy_um) >= len(m_indices)` —
  evaluated false (61 < 121), so the function fell through to
  `grid_fallback`, producing an 11 × 11 sqrt grid with 121 cells and
  no spatial structure. That's the layout the user saw.
- **Fix in `compute_tile_layout`:** when `len(stage_xy_um) <
  len(m_indices)`, truncate `m_indices` to
  `m_indices[:len(stage_xy_um)]`. Then proceed with the existing
  cluster + serpentine path. The truncation assumes missing
  positions are contiguous at the end (the typical
  "acquisition stopped early" case).
- **Trigger reason added:** `serpentine_reason = "partial_metadata"`
  takes precedence over `"revisits"` and `"sparse"`. The user wants
  to know first that the file is incomplete; layout shape is
  secondary information.

### Changed
- `import_page.py`'s "Tile source" diagnostic gains the partial-
  metadata case:
  `"61 of 121 multipoints have stage XY metadata · serpentine 5 × 13
   layout (M=0 top-left)"`.

### Documentation
- V1.11 plan: `CodeLog/ClaudesPlan/V1.11_partial_metadata_layout.md`.

## [V1.10] - 2026-04-30

### Bug Fixes

#### Sparse / irregular ROI scans now use serpentine
- **`backend/exporters/stitch_exporter.py`** — V1.8 only switched to
  serpentine when an ND2 had **revisits** (`max_dup > 1`). It missed
  acquisitions with **gaps** — irregular / ROI-shaped scans that
  visit only a subset of cells in a regular grid.
- Diagnosed against the user's `Stitch_test.tif` (12 × 7 grid, only
  33 / 84 cells populated ≈ 39 % — a clearly irregular ROI scan):
  V1.8 took the physical-layout path and stitched each tile at its
  spatial position with all the empty cells preserved as black
  background. The user wants those gaps gone.
- Added `fill_ratio = n_populated_cells / (n_phys_cols * n_phys_rows)`
  and a new trigger: serpentine fires when **either**
  `max_dup > 1` (revisits) **or** `fill_ratio < 0.9` (sparse).
- New `StitchLayout.serpentine_reason` field (`"revisits"` |
  `"sparse"` | `""`) so the diagnostic can explain *why* serpentine
  fired. New `StitchLayout.fill_ratio` field for completeness.

### Changed

#### Diagnostic clarity
- "Tile source" row in the Import metadata table now distinguishes:
  - Revisit: `"68 multipoints → 17 unique cells × 4 visits → serpentine 4 × 17 layout (M=0 top-left)"`
  - Sparse: `"33 multipoints in an irregular ROI (39% of grid populated) → serpentine 5 × 7 layout (M=0 top-left)"`
  - Dense: `"68 of 68 from stage XY"` (unchanged).

### Documentation
- V1.10 plan: `CodeLog/ClaudesPlan/V1.10_sparse_grid_detection.md`.

## [V1.9] - 2026-04-30

### Bug Fixes

#### `ValueError: too many values to unpack (expected 2)` on RGB TIFF load
- **`widgets/multi_axis_viewer.py`** — `_compose_current_frame()`
  assumed every channel frame was 2D `(H, W)` and called
  `h, w = sample.shape`. That broke when a frame was 3D — for
  example, `tifffile.imread()` returns `(H, W, 3)` for RGB pages, so
  loading an RGB stitch output (`Stitch_test.tif`) crashed the viewer
  immediately on open.
- New `_normalize_to_2d(frame)` static helper coerces any reasonable
  loader output into 2D:
  - `(H, W)` — unchanged.
  - `(H, W, 3)` / `(H, W, 4)` — converts RGB / RGBA → luminance via
    Rec. 601 (`0.299·R + 0.587·G + 0.114·B`).
  - `(Z, H, W)` / `(T, H, W)` — takes the first slice along axis 0.
  - `(T, Z, H, W)` and deeper — recursively slices `[0]` until 2D.
  - Trivial dims (`(1, H, W, 1)` etc.) — squeezed first.
  - 0D / 1D — returns `None` (frame can't be displayed).
- Applied in both branches of `_compose_current_frame` (volume and
  flat-channels) and in `_populate_lut_samples_from_volume()` — so
  the histogram seeding doesn't crash on RGB inputs either.

### Documentation
- V1.9 plan: `CodeLog/ClaudesPlan/V1.9_frame_shape_normalization.md`.

## [V1.8] - 2026-04-30

### Changed

#### Serpentine layout for revisit-style acquisitions
- **`backend/exporters/stitch_exporter.py`**: when V1.7's duplicate
  detection fires (`max_dup > 1`), the layout now switches to a
  **serpentine (boustrophedon)** path keyed on M index instead of
  V1.7's horizontal expansion. M=0 lands at top-left; even rows go
  left-to-right, odd rows right-to-left. Matches McGhee-Lab scan
  conventions and what the user described:
  > "the tile layout should be a serpentine pattern from the top
  > being 1, going from left to right and right to left the next row"
- New `_serpentine_layout(n_tiles, tile_h, tile_w, n_rows_hint)`
  helper. `n_rows_hint` defaults to `n_phys_rows` from clustering
  (17 for the user's file); `n_cols = ceil(N / n_rows)` (4 in their
  case → 68 multipoints in a clean 4 × 17 layout).
- `StitchLayout.source` gains the value `"serpentine"`.
- The V1.7 horizontal-expansion path (`"stage_xy_expanded"`) stays in
  the code for forward compatibility but is no longer reached from
  `compute_tile_layout`. Files with revisits use serpentine; files
  with all-unique positions still use the V1.7 physical
  (`"stage_xy"`) layout.

### Bug Fixes

- **Y orientation:** V1.7 left M=0 at the bottom of the layout because
  of the "highest stage Y → row 0" flip. The user's scan starts at
  the top, so V1.8's serpentine path puts M=0 at row 0 unambiguously.
  (Genuine mosaics still use the Y-flipped physical layout — that's
  correct for them.)

### Documentation
- V1.8 plan: `CodeLog/ClaudesPlan/V1.8_serpentine_layout.md`.
- `import_page.py`'s "Tile source" diagnostic gains a serpentine
  variant: `"68 multipoints → 17 unique cells × 4 visits → serpentine
  4 × 17 layout (M=0 top-left)"`.

## [V1.7] - 2026-04-30

### Bug Fixes

#### Duplicate-position multipoints (visit cycles in M)
- **Diagnosis (via `scripts/inspect_nd2.py` against the user's 127.6 GiB
  68-position file):** the file is `dim_order = TPZCYX, T=80, P=68, Z=3,
  C=4`, with **only 17 unique stage XY positions** in the 68 multipoints.
  Each physical cell is revisited 4 times across the M axis (M=0, 17,
  34, 51 are all at the same (x, y); ditto for every other phys cell).
  The scan is a 17-position zigzag/checkerboard alternating between
  2 stage X positions, repeated 4 times. V1.6 correctly clustered to
  2 cols × 17 rows but stacked all 4 M's in each cell.
- **Fix in `compute_tile_layout()`:** after clustering, count how many
  M's land in each `(col_idx, row_idx)`. If `max_dup > 1`, expand each
  physical column into `max_dup` sub-columns. Within each cell, M's
  are sorted by index and placed in sub-columns 0, 1, …, max_dup−1.
  Every M now gets its own visible slot. Empty sub-cells appear as
  background, surfacing scan patterns like the user's checkerboard.

### Added

#### Richer tile-source diagnostic
- New fields on `StitchLayout`: `n_phys_cells`, `n_visits_per_cell`,
  `n_phys_cols`, `n_phys_rows`. Source label gains a third value
  `"stage_xy_expanded"` when duplicate-position expansion fired.
- `import_page.py`'s "Tile source" metadata row now reads (e.g.):
  `"68 multipoints → 17 unique cells × 4 visits → 8 × 17 layout"` for
  acquisitions with revisits, or `"68 of 68 from stage XY"` when every
  M is unique. `_tile_source_summary()` helper consolidates the V1.4
  metadata-reader status with the V1.7 layout decision.

#### ND2 metadata diagnostic script
- New `scripts/inspect_nd2.py`: prints `f.sizes`, `dim_order`, voxel
  size, channel info, all per-M stage XY positions, gap statistics,
  the V1.6 clustering result, and the final layout decision against
  any ND2 file. Used to diagnose this issue end-to-end.

### Documentation
- V1.7 plan: `CodeLog/ClaudesPlan/V1.7_duplicate_position_expansion.md`.

## [V1.6] - 2026-04-30

### Bug Fixes

#### Adaptive clustering tolerance (horizontal stacking on heavy-overlap scans)
- **`backend/exporters/stitch_exporter.py`** — `_cluster_axis()`
  rewritten. V1.5 used a fixed `tolerance = 30 % × tile-size-in-µm`,
  which for typical Nikon tile-scan acquisitions (e.g. 2048 px tile at
  0.5 µm/px → 1024 µm tile) yielded ~307 µm. With a typical step of
  100–200 µm between adjacent columns, all real columns merged into
  1–2 clusters and tiles overlapped horizontally even after the V1.5
  flush-grid rewrite.
- New algorithm derives the tolerance from the **gap structure** of
  the data:
  - Compute the sorted positive gaps between sorted unique values.
  - Bimodal detection: find the largest ratio between consecutive
    sorted gaps. If it's ≥ 3×, that's the jitter→step jump; tolerance
    is the midpoint between the largest jitter gap and the smallest
    step gap.
  - Otherwise (uniform gaps → regular grid, no jitter): tolerance =
    half the smallest gap.
- `cluster_tolerance_frac` (default `0.5`) is repurposed as a
  **sanity cap** — the tolerance never exceeds half the tile size in
  µm, which protects degenerate single-row scans from spurious
  cluster boundaries while staying out of the way for realistic
  acquisitions.

### Changed

#### Auto-sized scrollable tile widget
- **`widgets/tile_layout.py`** — V1.5 set `setFixedHeight(240)` on
  the widget; with 17 rows letterbox-fitted into 240 px each tile
  rendered ~10 px tall and labels stacked. New behaviour:
  - `_TARGET_TILE_PX = 50` defines a comfortable per-tile pixel size
    for 2-digit M labels.
  - After every `set_tile_layout()` the widget calls
    `_adapt_minimum_size()` which sets
    `setMinimumSize(n_cols × 50 + margins, n_rows × 50 + header +
    margins)`. The host (`LutSidebar` or `TileLayoutDialog`) then
    scrolls if that exceeds the visible viewport.
  - Letterbox painting is unchanged — the widget still scales the
    flush grid into whatever rect it actually receives, but the rect
    is now ≥ the readable minimum.
- `TileLayoutDialog` wraps its child widget in a `QScrollArea`
  (horizontal + vertical scrollbars `AsNeeded`) so even very large
  mosaics stay readable inside the modal.
- **`widgets/lut_sidebar.py`** — drop the explicit
  `tile_widget.setFixedHeight(...)` (the widget handles its own
  sizing now) and switch the body's horizontal scroll policy from
  `Qt.ScrollBarAlwaysOff` to `Qt.ScrollBarAsNeeded` so wide-format
  mosaics stay readable.

### Documentation
- V1.6 plan: `CodeLog/ClaudesPlan/V1.6_adaptive_clustering_scrollable_layout.md`.

## [V1.5] - 2026-04-30

### Changed

#### Flush-grid tile layout
- **`backend/exporters/stitch_exporter.py`** —
  `compute_tile_layout()` rewritten. The V1.4 algorithm converted
  stage XY directly to physical pixels, which caused real per-pixel
  overlap whenever the acquisition itself had tile overlap (typical
  Nikon multi-position scans use 10–30 % overlap). In the layout
  widget that overlap was large enough that tile rectangles literally
  stacked and the M-index labels piled up.
  - New algorithm: cluster stage X and Y into discrete column / row
    centroids (sweep-line clustering with a per-axis tolerance of
    `30 % × tile-size-in-µm`, exposed as `cluster_tolerance_frac`).
  - Each M is assigned a `(col_idx, row_idx)`; offsets become
    `(row_idx × tile_h, col_idx × tile_w)` — flush borders, no
    overlap, no gaps.
  - Y axis flipped so the highest stage Y becomes row 0 (top of
    image), matching how microscope stages and image arrays differ
    in Y orientation.
  - Canvas dimensions are exactly `(n_cols × tile_w) × (n_rows ×
    tile_h)`, which combines naturally with the tile widget's
    existing letterbox `min(avail/canvas)` scaling: the panel "fits
    the max horizontal / vertical M scans" by construction.
- New helpers `_cluster_axis(values, tolerance)` and
  `_assign_index(value, centroids)` in the same module.
- The grid fallback (sqrt row-major) is unchanged. The `source`
  field on `StitchLayout` still reads `"stage_xy"` for the new
  cluster-based path so the V1.4 "Tile source" diagnostic in the
  Import metadata table keeps working.

The same flush-grid layout drives the stitch export too — the
exported TIFF is a tidy contact sheet of tiles. Real overlap
blending is intentionally out of scope for V1.5.

### Documentation
- V1.5 plan: `CodeLog/ClaudesPlan/V1.5_flush_grid_tile_layout.md`.

## [V1.4] - 2026-04-30

### Bug Fixes

#### Per-M stage XY (tile-layout overlap)
- **`backend/nd2_loader.py`** — `read_nd2_metadata_extended()` was
  iterating `for t in range(n_timepoints)` and treating `t` as a
  multipoint index when calling `f.frame_metadata(t)`. The nd2 library
  actually takes a *flat sequence index* (M × T × Z × C); for typical
  ND2 dim orders the first `n_timepoints` flat indices stay within
  `M=0`, which produced duplicate stage XYs and caused
  `compute_tile_layout()` to draw multiple M tiles at the same
  visual position. On the user's 68-tile file the layout widget
  showed ~24 visible positions with 2–3 tile-index labels piled on
  each one.
- Replaced the per-T loop with a per-M loop that computes the flat
  index for `(M=m, T=0, Z=0, C=0)` from the file's `dim_order` and
  `sizes` (`_flat()` helper). One `frame_metadata` call per M instead
  of per frame.
- Frame timestamps still iterate per-T (movie overlays need
  per-frame granularity), but they now use the correct flat-index
  math too: `(M=0, T=t, Z=0, C=0)`.

### Added

#### Tile source diagnostic
- New `stage_layout_source` field returned by
  `read_nd2_metadata_extended()`. Values:
  - `"stage_xy"` — every M reported a usable stage XY.
  - `"partial:N/M"` — only some M positions had usable XY data.
  - `"missing"` — no XY data; tile layout will use grid fallback.
- Surfaced in the Import metadata table as a "Tile source" row, e.g.
  `68 of 68 from stage XY` or `grid fallback (no stage XY metadata)`.
  Lets the user verify at a glance whether the layout came from the
  acquisition or from `compute_tile_layout`'s grid fallback.

### Changed

- **`pages/import_page.py`** — metadata table now renders the new
  "Tile source" row when present; visible whenever there's more than
  one M position or the diagnostic was set.

### Documentation
- V1.4 plan: `CodeLog/ClaudesPlan/V1.4_per_m_stage_positions.md`.

## [V1.3] - 2026-04-30

### Added

#### Interactive tile layout (sidebar nav + stitch selection)
- **`widgets/tile_layout.py`**: new `TileLayoutWidget` with two
  interaction modes (`"navigate"` and `"select"`). Renders the M-tile
  arrangement using the same `compute_tile_layout()` math the
  stitcher uses, so what you see is what you'd export. Hit-tested
  click handler picks the topmost tile under the cursor; hover paints
  a faint outline so the user previews the click target before
  committing. Shows a tile-index label inside each rect when there's
  pixel room for it (≥ 18 px). Three signals: `navigate_requested(m)`,
  `selection_changed(set[int])`, `expand_requested`.
- **`TileLayoutDialog`** (same module): modal "expand" wrapper. In
  navigate mode the dialog auto-closes after the user clicks (one-shot
  "find that tile"); in select mode it stays open and forwards
  selection changes continuously.

#### Tiles section in `LutSidebar`
- New independently-collapsible "Tiles" header above the LUTs section
  in the right sidebar. Hosts a `TileLayoutWidget` in `"navigate"`
  mode at the sidebar-friendly default size (240 × 240 px) with an
  expand-to-modal button.
- `LutSidebar.set_tile_layout(stage_xy_um, pixel_size_um, tile_h,
  tile_w, n_multipoints, current_m)` configures or hides the section
  (auto-hidden when ≤ 1 tile).
- `LutSidebar.set_current_m(m)` keeps the highlight in sync with M
  slider drags.
- `LutSidebar.tile_navigate_requested(m)` signal forwarded from the
  embedded widget (and from the modal when the user expands).
- The whole-sidebar header is now "Panel" (not "LUTs") since it
  hosts both Tiles and LUTs.
- Sidebar default width grew from 280 → 300 px to give the tile
  widget breathing room.

### Changed

- **`widgets/multi_axis_viewer.py`**: removed the V1.2 corner
  `TilePreviewWidget` overlay and its `_reposition_overlay()` /
  `resizeEvent` plumbing. Tile state now flows through
  `LutSidebar.set_tile_layout(...)`. New
  `_on_tile_navigate_requested(m)` slot just sets the M slider —
  reuses the existing `_on_m_changed` path so LUT re-sample,
  highlight sync, and frame redraw all fire without duplication.
  The `show_tile_preview` constructor argument is preserved for
  compatibility but now gates whether the *sidebar* tile section
  populates rather than the dropped corner overlay.
- **`pages/stitch_dialog.py`**: replaced the matplotlib
  `MplCanvas` layout preview *and* the row of per-tile `QCheckBox`es
  with a single `TileLayoutWidget` in `"select"` mode. Click any
  tile in the canvas to toggle inclusion. "Select all" / "Select
  none" buttons remain; the live summary shows
  `N/M tiles · canvas WxH · source` and updates as you click.
  The widget's expand button opens a `TileLayoutDialog` in select
  mode; selection changes mirror back to the dialog's small widget
  in real time. Min size grew to 680 × 720 to fit the bigger preview.

### Removed
- `widgets/tile_preview.py` — superseded by
  `widgets/tile_layout.py` (no remaining imports).

### Documentation
- V1.3 plan: `CodeLog/ClaudesPlan/V1.3_interactive_tile_layout.md`.
- Architecture updated to describe the new tile widget, the sidebar
  Tiles section, and the StitchDialog refactor.

## [V1.2] - 2026-04-30

### Added

#### Collapsible LUT sidebar
- **`widgets/lut_sidebar.py`**: new `LutSidebar` (`QFrame`) hosting one
  `LutHistogramWidget` per channel, plus a per-channel header row
  with a small color swatch and the channel name. Expanded width
  280 px; collapsed width 28 px (just the toggle button strip).
  Animated via `QPropertyAnimation` on both `minimumWidth` and
  `maximumWidth` (260 ms `InOutQuart`) — animating both bounds is
  required so `QHBoxLayout` doesn't fight the animation. Header has
  a "▶ / ◀" toggle button. Forwards every per-channel
  `contrast_changed` as a single `channel_contrast_changed(name, lo,
  hi, gamma)` signal so the viewer subscribes once. `update_swatch
  (name, rgb)` keeps the per-channel header swatches in sync when the
  user changes a channel's color in the chip strip.

#### Tile-layout preview overlay
- **`widgets/tile_preview.py`**: new `TilePreviewWidget` — small
  `QFrame` painted with `QPainter` (160 × 120 px). Reuses
  `compute_tile_layout()` from the stitch exporter so what's drawn
  matches what'd be exported. Highlights the currently displayed M
  in `ACCENT_PURPLE`. Title text reads "Tile layout (N tiles ·
  click to stitch)". Click → `stitch_requested` signal. Calls
  `is_renderable()` to auto-hide when M ≤ 1 or stage XY is missing.

### Changed

- **`widgets/multi_axis_viewer.py`**: restructured from a vertical
  stack to a horizontal split. Left column: image canvas + zoom
  toolbar + M/T/Z sliders + a compact `ChannelChip` strip
  (toggle + color combo only — LUTs moved out). Right column: the
  collapsible `LutSidebar`. The new `TilePreviewWidget` is parented
  onto the image canvas and absolutely positioned in the bottom-right
  corner with an 8 px margin; `resizeEvent` repositions it. `set_volume`
  gains a `stage_xy_um` parameter so the overlay can render. New
  `stitch_requested` signal is forwarded from the tile-preview.
  `ChannelChip` replaces the V1.1 `ChannelControlRow` (which embedded
  the LUT inline). Channel-state read-out (`channel_state()`) and
  apply (`apply_channel_state()`) keep the same shape so the
  experiment record contract is unchanged.
- **`pages/import_page.py`**: passes `stage_xy_um` from
  `nd2_metadata` into `viewer.set_volume(...)` so the tile preview
  has data to render. Connects `viewer.stitch_requested` to
  `_on_stitch` so clicking the corner tile preview opens
  `StitchDialog` with no extra UI.

### Documentation
- V1.2 plan: `CodeLog/ClaudesPlan/V1.2_lut_sidebar_tile_preview.md`.
- Architecture updated to describe the new widget split, the
  `LutSidebar` collapse animation pattern, and the tile-preview
  overlay.

## [V1.1] - 2026-04-30

### Added

#### M / T / Z scrolling in the viewer
- **`backend/nd2_volume.py`**: new `LazyND2Volume` class exposing
  `(M, T, Z, H, W)` lazy reads against an ND2 file. Holds a single
  `nd2.ND2File` handle internally; `get_frame(c, m, t, z, z_mode,
  z_start, z_end)` returns one (H, W) array per call. Z projection is
  applied per-frame on demand.
- `LazyND2Volume.to_lazy_channel(c, m, z_mode, z_index, ...)` collapses
  the volume to a V1.0-compatible `LazyND2Channel` so the recipe /
  export pipeline keeps its `(T, H, W)` contract — plugins did not
  need to change. `all_channels_as_lazy()` is a convenience that
  returns an `OrderedDict` of channel-name → lazy proxy.
- **`workers/load_worker.py`**: also returns `volume: LazyND2Volume`
  in the loaded payload so the Import page can wire up the new viewer.

#### Live channel toggle / color / LUT in the viewer
- **`widgets/multi_axis_viewer.py`**: new `MultiAxisViewer` widget
  with M / T / Z sliders (auto-hidden when an axis is degenerate)
  and a per-channel control strip (`ChannelControlRow`) holding an
  enable checkbox, color combobox, and embedded LUT histogram. Any
  change recomposes the displayed image instantly. The viewer
  accepts either a `LazyND2Volume` (M/Z navigation enabled) or a
  flat `Dict[name, (T, H, W)]` (recipe / processed view).
- Image rendering goes through `apply_lut(frame, lo, hi, gamma)` per
  channel rather than the old auto-percentile path.

#### Manual LUT histogram per channel
- **`widgets/lut_histogram.py`**: new `LutHistogramWidget` showing a
  log-scaled intensity histogram with two draggable vertical handles
  (`lo`, `hi`), a `γ` slider (0.20..5.00), and `Auto` / `Reset`
  buttons. Histogram is computed by sampling at most 32 frames so it
  stays cheap on large timeseries. Emits
  `contrast_changed(lo, hi, gamma)`. Exports a module-level
  `apply_lut(frame, lo, hi, gamma) -> uint8` helper used by the
  multi-axis viewer.

#### M-position stitching
- **`backend/exporters/stitch_exporter.py`**: `StitchLayout` dataclass,
  `compute_tile_layout()` (places tiles using ND2 stage XY positions
  divided by the file's pixel size; falls back to row-major grid when
  positions are missing or all identical),
  `stitch_one_frame(tile_frames, layout)` for one timepoint, and
  `export_stitched_tiff(volume, layout, m_indices, channel_indices,
  channel_colors, filepath, z_mode, z_index, rgb, pixel_size_um,
  progress_cb)` for the full timelapse. Single-channel output writes
  a grayscale TIFF; multi-channel writes an RGB composite using
  per-channel percentile contrast.
- **`workers/stitch_worker.py`**: `StitchRequest` + `StitchWorker`
  wrapping `export_stitched_tiff` with progress signals.
- **`pages/stitch_dialog.py`**: `StitchDialog` launched from a new
  "Stitch M…" button on the Import page. Shows a live layout preview
  (matplotlib canvas with tile rectangles), per-tile checkboxes,
  per-channel toggle + color combo, Z mode picker, and an output
  path picker. Cancels cleanly; never modifies the source file.

### Changed
- **`core/experiment_manager.py`**: `ND2StudiosRecord` gains
  `m_index`, `z_view_mode`, `z_view_index` (round-tripped via JSON),
  `n_multipoints`, `n_zslices`, and a non-serialized `_raw_volume`
  field. `channel_display` entries now also carry `lut_lo`, `lut_hi`,
  `lut_gamma` so contrast settings persist across save/load.
- **`pages/import_page.py`**: replaced the `ChannelRow` widget with
  the read-only `ChannelInfoRow` (exposure / em / ex display only).
  The right-side preview is now a `MultiAxisViewer`. Added a
  "Stitch M…" button next to "Confirm Import" — enabled when the
  loaded file has more than one M position. Confirm Import now
  rebuilds per-channel proxies via
  `LazyND2Volume.all_channels_as_lazy(m, z_mode, z_index)` so the
  pipeline operates on whatever (M, Z mode) the user is viewing.
- **`pages/recipe_page.py`**: both side-by-side viewers are now
  `MultiAxisViewer`s. Removed the `_channel_view_state()` helper —
  the new viewer pulls channel display state directly from the
  experiment record.

### Documentation
- V1.1 plan: `CodeLog/ClaudesPlan/V1.1_mz_scrolling_lut_stitching.md`.
- Architecture updated to describe the new modules and data flow.

## [V1.0] - 2026-04-30

### Added

#### Project scaffold
- VSClaude project layout (`CodeLog/`, `Research/`, `hpc/`, `scripts/`)
- `CLAUDE.md` customized for ND2Studios with project conventions, key
  technical decisions, and dependencies
- V1.0 plan document at `CodeLog/ClaudesPlan/V1.0_initial_scaffold.md`
- Top-level `run.py` launcher that prepends the project root to
  `sys.path` and calls `nd2studios.__main__.main()`

#### Core (`nd2studios/core/`)
- `settings.py`: Dracula palette, app dimensions, animated sidebar
  config (60↔240 px, 280 ms `InOutQuart`), 3-page list (`import`,
  `recipe`, `export`), status ladder (`new → imported → preprocessed →
  ready_to_export`) and per-page prerequisites
- `theme.py`: Dracula dark stylesheet adapted from CellTracker, with
  additional rules for `#bgApp` rounded card, `#titleBarWidget` custom
  frameless title bar, `#navBtn` collapsible sidebar buttons with text
  labels, `#sessionBtn`, and `#toggleBtn` (hamburger)
- `main_window.py`: frameless `QMainWindow` with:
  - PyDracula-style custom title bar (drag-to-move, double-click to
    maximize, Min/Max/Close buttons)
  - 4 `CustomGrip` resize handles + a `QSizeGrip` corner
  - Animated collapsible sidebar via `QPropertyAnimation` on
    `minimumWidth` and `maximumWidth`
  - Drop shadow on `#bgApp` (blur 17, alpha 150)
  - Sidebar navigation buttons in a `QButtonGroup` (exclusive)
  - Session buttons (New / Save / Load) at the sidebar bottom
  - Top bar (page title + `StatusIndicator`) and bottom bar (version +
    `QProgressBar` + status text)
  - `QStackedWidget` page switching with prerequisite-gated navigation
- `experiment_manager.py`: `ND2StudiosRecord` (configs, metadata, recipe,
  channel display, raw + processed channel caches, frame timestamps)
  and `ND2StudiosManager` (`active_changed`, `status_changed` signals,
  `.nd2s` JSON manifest + `_arrays.npz` companion save/load)
- `plugin_registry.py`: `PluginBase` with `@register` decorator,
  `ParamSpec` for declarative parameter UIs, `EnhancementPlugin` base
  (input/output `(T, H, W)` numpy)

#### Backend (`nd2studios/backend/`)
- `nd2_loader.py`: copied from CellTracker — `ND2Metadata` dataclass,
  `read_nd2_metadata`, `load_nd2_timeseries`, `LazyND2Channel` (numpy
  protocol proxy with `shape`, `dtype`, `__getitem__`, `materialize`,
  `crop`), `load_nd2_timeseries_lazy`, `z_project`. Extended with
  `read_nd2_metadata_extended()` returning a dict with: dim_order,
  sizes, dtype, height, width, T/Z/C/P counts, pixel size, z step,
  voxel size, channel names + colors + emission/excitation + exposures,
  objective name/magnification/NA/immersion, binning, camera name,
  microscope name, per-frame timestamps, per-frame stage XY/Z, loops
- `tiff_loader.py`: copied from CellTracker — `LazyTIFFChannel`,
  `load_tiff_stack_lazy`, `assign_dimensions`, `extract_2d_timeseries`
- `normalization.py`: copied from CellTracker — frame-mean normalization
- `recipes.py`: copied from CellTracker — recipe save/load with the
  extension changed to `.nd2s_recipe.json` and the kind tag set to
  `nd2studios.recipe`
- `exporters/tiff_exporter.py`: `export_tiff_stack(stack, filepath,
  bit_depth, pixel_size_um, progress_cb)` with passthrough / uint8 /
  uint16 modes (using 0.5–99.5 percentile bounds), TIFF resolution
  tags, BigTIFF when > 3.9 GB
- `exporters/composite_exporter.py`:
  `CHANNEL_COLORS`, `_composite_frame()` (additive RGB blending),
  `export_rgb_composite_tiff()`
- `exporters/movie_exporter.py`: `MovieOptions` dataclass (FPS, scale
  bar config, timestamp config, channel-label config), `export_movie()`
  using `imageio` for MP4 (libx264) and GIF; PIL-drawn overlays —
  scale bar (configurable corner / length in µm / color / thickness),
  timestamp (corner / color / synthetic-dt fallback when ND2 timestamps
  are unavailable), channel labels with color swatches

#### Plugins (`nd2studios/plugins/enhancement/`)
- `builtin.py`: 19 enhancement plugins copied verbatim from CellTracker
  (Normalize, CLAHE, GaussianBlur, MedianFilter, BackgroundSubtract,
  GammaCorrection, BleachCorrection, TemporalFoldCorrection,
  SpatialFlatness, TopHat, DoG, UnsharpMask, BilateralDenoise,
  MorphGradient, LocalContrast, BlobSubtract, NLMDenoise,
  WaveletDenoise, TVDenoise) — registered via `@PluginBase.register`

#### Widgets (`nd2studios/widgets/`)
- `common.py`: `MplCanvas` (Dracula-styled matplotlib figure with
  per-axis dark theming), `ParamEditor` (auto-generates a form from
  `List[ParamSpec]` with float/int/bool/choice/str support and a
  `params_changed(dict)` signal), `StatusIndicator` (colored badge with
  the ND2Studios 5-status palette: new / imported / preprocessed /
  ready_to_export / error)
- `image_viewer.py`: copied from CellTracker — `CHANNEL_COLORS`,
  `frame_to_uint8`, `apply_lut_color`, `composite_channels`,
  `ImageCanvas` (zoom/pan), `ZoomToolbar`, `ImageViewer` (multi-channel
  RGB compositing, T slider, auto-contrast)
- `scale_bar.py`: copied from CellTracker — matplotlib scale bar
  rendering helpers
- `video_player.py`: copied from CellTracker — Play/Pause/FPS playback
  controls
- `custom_grips.py`: PySide6 frameless-window edge resize handles —
  reimplemented from PyDracula's `CustomGrip` as a leaner
  `QWidget`-based class with per-edge cursor, transparent background,
  and `_resize_parent(delta)` that respects `minimumWidth`/`Height`

#### Workers (`nd2studios/workers/`)
- `base_worker.py`: `BaseWorker(QThread)` with `progress(int)`,
  `status(str)`, `finished(object)`, `error(str)` signals and `cancel()`
- `load_worker.py`: `LoadWorker(filepath, z_projection, z_start, z_end,
  t_start, t_end, t_stride)` returning a dict with metadata, lazy
  channels, channel names, and frame timestamps; supports `.nd2`
  (extended metadata) and `.tif`/`.tiff` (single-channel TYX)
- `recipe_worker.py`: `RecipeWorker(channels, recipe, normalized)`
  applies the ordered list of `(plugin_name, params)` plugins to each
  channel; lazy proxies are materialized once per channel; optional
  frame-mean normalization runs first
- `export_worker.py`: `ExportRequest` dataclass + `ExportWorker`
  dispatching to `tiff_stack` / `rgb_composite` / `movie` modes; per
  enabled channel for TIFF mode

#### Pages (`nd2studios/pages/`)
- `import_page.py`: file browse, full extended-metadata table (file,
  dimensions, frame size, dtype, pixel size, z step, objective, camera,
  microscope, binning, acquisition span, mean dt, stage XY range,
  loops), scrollable channel-row list (enable + color combo + per-channel
  exposure / emission / excitation), Z-projection mode (max/mean/min/none),
  frame stride, image preview via multi-channel `ImageViewer`, Confirm
  Import button
- `recipe_page.py`: plugin combo + description label, auto-generated
  parameter form, frame-mean normalization checkbox, Trial / Accept /
  Reject / Remove Last / Clear All buttons, recipe list view, Save / Load
  recipe (`.nd2s_recipe.json`), side-by-side raw vs processed previews
- `export_page.py`: source selector (raw / processed), three tabs:
  TIFF stack (bit depth picker), RGB composite (one click), Movie
  (FPS, format, scale bar, timestamp, channel labels — with per-corner
  position pickers); each tab kicks off `ExportWorker`

### Documentation
- `CodeLog/Architecture/ARCHITECTURE.md` populated with module
  breakdown, data flow, dependencies, and design decisions
- This changelog
