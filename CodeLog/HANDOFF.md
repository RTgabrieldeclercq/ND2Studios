# Session handoff — Granule separation feature suite

> Context transfer for a new chat. Read this first, then the plan docs + CHANGELOG it
> points to. Everything below is on branch `Version-1.45` as **uncommitted working-tree
> changes** (nothing committed this session). Date: 2026-07-14.

## The task (from `wishlist.md`)

1. ✅ Plan a granule-separation feature, partition into concurrent sub-plans, save to
   `CodeLog/ClaudesPlan/`, then compact all ClaudesPlan files (with Sonnet) into a
   code-history ledger.
2. ✅ Implement the feature (5 nodes).
3. ✅ Follow-up requests (registration-first, crop-awareness, freeze fix, per-node viewers).
4. ⏸️ **DEFERRED by the user — do NOT do:** compare our workflows to Matt Pocock's
   skills repo (`https://github.com/mattpocock/skills`, wishlist.md lines 33–35).

## What shipped (to the working tree)

### V1.70 — Granule separation nodes (5 pipeline-graph "special" nodes)
Chain: **Bead Detect → Granule Cluster → Granule Tessellation → Granule Volume Mask →
Granule Boundary** (also wireable into DVC with the Frame/Object lever for per-granule DVC).

- **Backend (pure, Qt-free)** in `nd2studios/backend/analysis/`:
  `bead_detect.py` (wraps `serialtrack/detection.ParticleDetector`; only `(x,y,z)→(z,y,x)`
  flip; first `centroid_z_px` writer), `granule_cluster.py` (full-cov GMM + BIC sweep over
  relaxed count; optional `find_spec`-gated scikit-learn), `granule_tessellate.py`
  (per-granule alpha-shape OR global Voronoi + RAG density-ratio merge → final labels),
  `granule_mask.py` (voxelize + SDF-Gaussian smooth → per-granule `(Z,H,W) bool` + combined
  `(Z,H,W) int32`, **dense 1-based labels**), `granule_boundary.py` (outward N-voxel band,
  dilation or EDT). Shared contracts + `granule_color`/palette in `granule_types.py`.
- **Op keys:** `pipeline_graph/granule_ops.py` (exported via `pipeline_graph/__init__.py`).
- **Integration:** `registry_adapter.py` (`_SPECIAL_OPS` rows, `param_specs_for` branches,
  `op_produces_objects` incl. mask), `object_scope.py` (`iter_objects_3d_labels` for the
  `(Z,H,W) int32` combined volume), `pages/pipelines_page.py` (dispatch + handlers).
- **Threading:** all five computes run **off the GUI thread** via `_GranuleJob(AnalysisJob)`
  submitted to the JobRunner (key `_RUN_GRANULE_KEY`), gated by `_run_pending`, resumed in
  `_finish_granule`. (This fixed a reported GUI-freeze — they were synchronous initially.)
- **Registration/crop:** bead detect reads the **registered** volume and confines detection
  to `_crop_rect()` (the same crop DVC uses), reporting full-frame coords so masks + DVC
  `bbox∩crop` stay consistent.
- Stores (per `(m,t)`, mirroring `record._mask3d_by_m`):
  `record._granule_{points,labels,tess,masks,bands}_by_m`. Granules are defined on the
  currently-viewed **reference timepoint** and reused across T.

### V1.73 — Per-node viewers (numbered 1.73 because 1.71/1.72 were taken)
Gated overlay **modes** on the Pipelines `_overlay_tabbar` (not new panels), auto-selected
on Run from `_finish_granule`:
- 2-D painters `_paint_granule_{beads,clusters,tess,mask,boundary}` in `pipelines_page.py`
  (crosshairs / colored dots / tessellation lines / per-plane colored masks /
  solid-new+dotted-previous contours). Honor per-Z filter (only when `z_view_mode=="none"`)
  and invert the `_crop_rect()` offset.
- 3-D via existing `_btn_view3d` → `_feed_view3d_granule_overlay` → pure-numpy
  `backend/viz3d/overlays.GranuleScene` (+ `granule_centroid_edges`) →
  `widgets/viewer3d/pyvista_viewer._add_granule_scene_overlay`.
- Categorical color = `granule_types.granule_color(gid)` (set-independent; palette is a
  verbatim copy of `analysis_page._LABEL_PALETTE`).

### Also done
- `CodeLog/Architecture/CODE_HISTORY.md` — Sonnet-compacted ledger of all ~114 ClaudesPlan
  docs (13 thematic sections). Originals untouched.
- Sub-plans: `CodeLog/ClaudesPlan/V1.70_P0..P7_*.md`, `V1.73_granule_viewers.md`.
- Docs: `CHANGELOG.md` V1.70 + V1.73 blocks; `ARCHITECTURE.md` granule sections.

## Verification status — IMPORTANT
Verified **headless only**: `py_compile`; `tests/granule/` (41) + `tests/viz3d/` (98 total)
pass; off-screen pyvista render of all 5 scene types; 5 painter pixel tests; palette-drift
guard. **NOT driven in the live GUI against a real ND2.** That end-to-end QA is the top
outstanding item (see below).

## Suggested next steps
1. **Live-GUI QA** (the real gap): `python run.py`, import the test ND2 (see memory
   [Test ND2 File]), wire detect→cluster→tessellate→mask→boundary, Run each; confirm tabs
   appear/auto-select, Z-scrub tracks the plane, 2-D↔3-D colors match, granule→DVC works.
2. **Deferred follow-ups:** cluster↔mask exact color identity across id spaces (GMM/final
   vs dense 1-based); per-`(m,t)` granule tracking across T; a standalone label-colored
   granule 3-D viewer; generic per-object execution for non-DVC downstream nodes.
3. Commit when ready (not done this session; user hasn't asked).

## Gotchas learned this session
- `_granule_entry` must not use `array or default` (ambiguous truth on numpy) — fixed.
- Granule combined labels are **1-based** (0=background); a 0-based id would vanish — fixed
  in `granule_mask.build_granule_masks`.
- `_select_overlay_tab` sets the tab under `blockSignals`, so `_finish_granule` must
  explicitly re-arm `set_frame_post_process(self._composite_pipeline_overlay)` to repaint.
- Env: interpreter auto-pinned to Python 3.12; run headless GUI/pyvista tests with
  `QT_QPA_PLATFORM=offscreen` and `PYTHONPATH=<repo root>`.
