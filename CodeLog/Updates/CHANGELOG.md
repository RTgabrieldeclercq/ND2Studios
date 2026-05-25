# Changelog

All notable changes to ND2Studios will be documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/)

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
