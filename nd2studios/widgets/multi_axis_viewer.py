"""
``MultiAxisViewer`` (V1.2 layout) — image canvas with M / T / Z sliders,
a compact channel chip strip below the sliders, an embedded
:class:`TilePreviewWidget` overlaid on the image canvas, and a
collapsible :class:`LutSidebar` on the right edge holding per-channel
histogram controls.

V1.2 changes vs V1.1:

- LUT histograms moved out of the inline channel rows and into the
  collapsible :class:`LutSidebar` so the image gets more vertical
  room. Toggle / color combos remain inline as small ``ChannelChip``
  widgets — those are clicked frequently and shouldn't be hidden.
- New tile-layout overlay (bottom-right corner of the image canvas)
  for files with multiple M positions and stage XY data. Click it to
  open the stitch dialog.

Image compositing is unchanged from V1.1: per-channel ``apply_lut`` →
multiply by per-channel RGB color → additive sum → clip to uint8.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel,
    QPushButton, QSlider, QSplitter, QVBoxLayout, QWidget,
)

from nd2studios.backend.frame_cache import FrameCache
from nd2studios.backend.nd2_volume import LazyND2Volume
from nd2studios.core.settings import Settings
from nd2studios.utils.resources import detect, recommended_cache_budget_bytes
from nd2studios.utils.threading import IOWorker, PlaneRequest, start_io_worker
from nd2studios.widgets.image_viewer import (
    CHANNEL_COLORS, ImageCanvas, ZoomToolbar,
)
from nd2studios.widgets.lut_histogram import LutHistogramWidget, apply_lut
from nd2studios.widgets.lut_sidebar import LutSidebar
from nd2studios.workers.prefetch_worker import PrefetchManager


def _gpu_display_enabled() -> bool:
    """Return True if the pyqtgraph GPU canvas should be used.

    Honors :attr:`Settings.USE_GPU_DISPLAY` and the
    ``ND2_DISABLE_GPU_DISPLAY=1`` env override. Centralized so the
    legacy-fallback path stays a single line in __init__.
    """
    if not getattr(Settings, "USE_GPU_DISPLAY", False):
        return False
    if os.environ.get(
        getattr(Settings, "GPU_DISPLAY_ENV_DISABLE", "ND2_DISABLE_GPU_DISPLAY"),
        "",
    ) == "1":
        return False
    return True


class ChannelChip(QWidget):
    """Compact toggle + color combo for one channel.

    Lives in the chip strip below the sliders. The LUT histogram lives
    in :class:`LutSidebar` instead.
    """

    state_changed = Signal()

    def __init__(self, name: str, color_default: str = "gray",
                 enabled: bool = True, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.name = name
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        self.cb = QCheckBox(name)
        self.cb.setChecked(enabled)
        self.cb.stateChanged.connect(lambda _s: self.state_changed.emit())
        layout.addWidget(self.cb)

        self.combo_color = QComboBox()
        self.combo_color.addItems(list(CHANNEL_COLORS.keys()))
        if color_default in CHANNEL_COLORS:
            self.combo_color.setCurrentText(color_default)
        self.combo_color.setFixedWidth(86)
        self.combo_color.currentTextChanged.connect(
            lambda _t: self.state_changed.emit())
        layout.addWidget(self.combo_color)

    @property
    def enabled(self) -> bool:
        return self.cb.isChecked()

    @property
    def color_name(self) -> str:
        return self.combo_color.currentText()

    @property
    def color_rgb(self) -> Tuple[int, int, int]:
        return CHANNEL_COLORS.get(self.color_name, (255, 255, 255))


class _SingleFrameSeries:
    """Shim matching ``LutHistogramWidget.set_data``'s expected protocol
    when we hand it a list of pre-flattened pixel samples."""

    def __init__(self, samples: List[np.ndarray], frame_shape):
        self._samples = samples
        self._frame_shape = frame_shape

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._samples[int(idx)]


class MultiAxisViewer(QWidget):
    """Image viewer with M/T/Z sliders, chip strip, LUT sidebar, tile preview."""

    # M, T, Z (the page persists these on the experiment record).
    coords_changed = Signal(int, int, int)
    # Any channel toggle / color / LUT change.
    channels_changed = Signal()
    # Click on the corner tile preview overlay.
    stitch_requested = Signal()
    # Crop rect selected via drag on the canvas (x, y, w, h in image pixels).
    crop_rect_selected = Signal(int, int, int, int)
    # Manual-mask shape drawn on the canvas. See ImageCanvas.shape_drawn.
    shape_drawn = Signal(str, list)
    # Vertex-edit signals — see ImageCanvas.
    vertex_moved = Signal(int, float, float)
    edit_committed = Signal()

    def __init__(self, parent: Optional[QWidget] = None,
                 show_tile_preview: bool = True):
        super().__init__(parent)
        # ``show_tile_preview`` is kept as an init arg for backwards
        # compatibility with V1.2 callers, but the V1.3 tile widget
        # lives in the LutSidebar — this flag now just gates whether
        # the sidebar tile section is populated.
        self._enable_tile_section = show_tile_preview

        self._volume: Optional[LazyND2Volume] = None
        self._channels: Dict[str, Any] = {}
        self._chip_strip: List[ChannelChip] = []
        self._z_mode: str = "max"
        self._z_index: int = 0
        self._stage_xy_um: List[Tuple[float, float]] = []

        self._m: int = 0
        self._t: int = 0
        self._z: int = 0

        self._m_timer = QTimer(self)
        self._t_timer = QTimer(self)
        self._z_timer = QTimer(self)

        # Debounce timers — separate from the playback timers above.
        self._t_debounce = QTimer(self)
        self._t_debounce.setSingleShot(True)
        self._t_debounce.setInterval(50)
        self._t_debounce.timeout.connect(self._do_refresh)

        self._z_debounce = QTimer(self)
        self._z_debounce.setSingleShot(True)
        self._z_debounce.setInterval(50)
        self._z_debounce.timeout.connect(self._do_refresh)

        self._m_debounce = QTimer(self)
        self._m_debounce.setSingleShot(True)
        self._m_debounce.setInterval(80)
        self._m_debounce.timeout.connect(self._do_m_refresh)

        # V1.34 Phase 2: size the LRU adaptively from available RAM.
        # On a 32 GB workstation with ~16 GB free this comes out to
        # ≈ 6.4 GB (vs the V1.0 hard-coded 300 MB), trading the RAM
        # the user has spare for fewer cache misses on scrub.
        self._frame_cache: FrameCache = FrameCache(
            max_bytes=recommended_cache_budget_bytes(reserve_fraction=0.6),
        )
        self._prefetch: Optional[PrefetchManager] = None
        # V1.34 Phase 2: foreground IO worker — serves cache misses
        # without freezing the GUI thread. See utils/threading.py.
        self._io_worker: Optional[IOWorker] = None
        self._io_thread = None
        self._request_counter: int = 0
        self._latest_request_id: int = 0
        # V1.35 Phase 3: keys exempt from cache eviction while on screen.
        # Re-derived from the set of channel planes that composited into
        # the most recent successful render in _compose_current_frame.
        self._pinned_keys: set = set()
        self._hist_cache: dict = {}  # (m, z_mode) -> True
        self._frame_post_process: Optional[Callable] = None

        # V1.39 Phase 7: optional multi-resolution pyramid.
        # ``_active_level == 0`` → read from the existing
        # :class:`LazyND2Volume`/IOWorker path (V1.38 behaviour).
        # ``_active_level >= 1`` → read synchronously from
        # ``_pyramid_reader.get_frame(level, ...)``; cache keys are
        # extended to 6-tuples ``(level, c, m, t, z, z_mode)`` so
        # pyramid planes coexist with level-0 planes in the cache
        # without colliding. The level is recomputed on every
        # viewport change (``_on_viewport_changed``).
        self._pyramid_reader: Optional[object] = None
        self._active_level: int = 0

        self._build_ui()

    # ── UI construction ──
    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self._viewer_splitter = QSplitter(Qt.Horizontal)
        self._viewer_splitter.setChildrenCollapsible(False)

        # Left column — image + sliders + chips.
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        # V1.36 Phase 4: prefer the pyqtgraph GPU canvas. Falls back
        # to the legacy QLabel-based ``ImageCanvas`` if pyqtgraph
        # isn't importable, if construction raises, or if the env
        # override is set. ``_use_gpu_canvas`` gates the render path
        # below (per-channel GPU push vs. CPU RGB composite).
        self._use_gpu_canvas: bool = False
        self.canvas: QWidget
        if _gpu_display_enabled():
            try:
                from nd2studios.widgets.gpu_image_canvas import GpuImageCanvas
                self.canvas = GpuImageCanvas()
                self._use_gpu_canvas = True
            except Exception:
                # GL drivers, headless display, missing pyqtgraph —
                # any of these should drop us cleanly back to the
                # legacy canvas rather than crash the viewer.
                self.canvas = ImageCanvas()
                self._use_gpu_canvas = False
        else:
            self.canvas = ImageCanvas()
        self.canvas.crop_rect_selected.connect(self.crop_rect_selected)
        self.canvas.shape_drawn.connect(self.shape_drawn)
        self.canvas.vertex_moved.connect(self.vertex_moved)
        self.canvas.edit_committed.connect(self.edit_committed)
        left_layout.addWidget(self.canvas, stretch=1)

        # The V1.2 corner tile-preview overlay is gone in V1.3 — the
        # tile layout now lives in the LutSidebar with full click
        # navigation and an expand-to-modal action.
        self.tile_preview = None

        zoom_row = QHBoxLayout()
        zoom_row.setContentsMargins(4, 0, 4, 0)
        self.zoom_toolbar = ZoomToolbar(self.canvas)
        zoom_row.addWidget(self.zoom_toolbar)
        zoom_row.addStretch(1)
        left_layout.addLayout(zoom_row)

        # M / T / Z sliders (each row includes a ▶/⏸ play button + fps spinbox).
        slider_box = QFrame()
        slider_box.setObjectName("contentArea")
        slider_layout = QVBoxLayout(slider_box)
        slider_layout.setContentsMargins(8, 4, 8, 4)
        slider_layout.setSpacing(2)
        self._m_row, self.m_slider, self.m_label, self._m_play, self._m_fps = \
            self._make_axis_row("M")
        self._t_row, self.t_slider, self.t_label, self._t_play, self._t_fps = \
            self._make_axis_row("T")
        self._z_row, self.z_slider, self.z_label, self._z_play, self._z_fps = \
            self._make_axis_row("Z")
        self.m_slider.valueChanged.connect(self._on_m_changed)
        self.t_slider.valueChanged.connect(self._on_t_changed)
        self.z_slider.valueChanged.connect(self._on_z_changed)
        self._m_play.toggled.connect(lambda on: self._set_axis_playing("m", on))
        self._t_play.toggled.connect(lambda on: self._set_axis_playing("t", on))
        self._z_play.toggled.connect(lambda on: self._set_axis_playing("z", on))
        self._m_timer.timeout.connect(lambda: self._axis_tick(self.m_slider))
        self._t_timer.timeout.connect(lambda: self._axis_tick(self.t_slider))
        self._z_timer.timeout.connect(lambda: self._axis_tick(self.z_slider))
        slider_layout.addLayout(self._m_row)
        slider_layout.addLayout(self._t_row)
        slider_layout.addLayout(self._z_row)
        left_layout.addWidget(slider_box)

        # Chip strip — compact toggle + color, one row across.
        chip_box = QFrame()
        chip_box.setObjectName("contentArea")
        self._chip_layout = QHBoxLayout(chip_box)
        self._chip_layout.setContentsMargins(8, 2, 8, 4)
        self._chip_layout.setSpacing(8)
        self._chip_layout.addStretch(1)
        left_layout.addWidget(chip_box)

        self._viewer_splitter.addWidget(left)

        # Right column — collapsible LUT sidebar (now also hosts the
        # interactive tile layout for navigation).
        self.lut_sidebar = LutSidebar()
        self.lut_sidebar.channel_contrast_changed.connect(
            self._on_lut_contrast_changed)
        self.lut_sidebar.tile_navigate_requested.connect(
            self._on_tile_navigate_requested)
        self.lut_sidebar.collapse_changed.connect(
            lambda _: self.canvas.reset_zoom())
        self._viewer_splitter.addWidget(self.lut_sidebar)
        self._viewer_splitter.setStretchFactor(0, 1)
        self._viewer_splitter.setStretchFactor(1, 0)

        outer_layout.addWidget(self._viewer_splitter, stretch=1)

    def _make_axis_row(self, label: str):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        lbl = QLabel(label + ":")
        lbl.setFixedWidth(20)
        row.addWidget(lbl)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 0)
        row.addWidget(slider, stretch=1)
        info = QLabel("0/0")
        info.setMinimumWidth(60)
        row.addWidget(info)
        play_btn = QPushButton("▶")
        play_btn.setCheckable(True)
        play_btn.setFixedSize(28, 22)
        play_btn.setToolTip(f"Play / pause {label} axis")
        row.addWidget(play_btn)
        fps_spin = QDoubleSpinBox()
        fps_spin.setRange(0.1, 30.0)
        fps_spin.setValue(5.0)
        fps_spin.setSingleStep(0.5)
        fps_spin.setSuffix(" fps")
        fps_spin.setFixedWidth(72)
        fps_spin.setToolTip("Playback speed")
        row.addWidget(fps_spin)
        return row, slider, info, play_btn, fps_spin

    def set_frame_post_process(self, fn: Optional[Callable]) -> None:
        """Set a callable applied to the composited frame before display.

        fn(rgb_uint8: np.ndarray, t: int, m: int) -> np.ndarray
        Pass None to remove any active hook.
        """
        self._frame_post_process = fn
        self._do_refresh()

    # ── Crop tool ──
    def set_crop_mode(self, enabled: bool) -> None:
        """Enable or disable the crop selection tool on the image canvas."""
        self.canvas.set_crop_mode(enabled)

    # ── Manual-mask drawing ──
    def set_draw_mode(self, mode: Optional[str]) -> None:
        """Set the shape-drawing tool on the canvas. See ImageCanvas.set_draw_mode."""
        self.canvas.set_draw_mode(mode)

    def set_edit_vertices(
        self, vertices: Optional[List[Tuple[float, float]]]
    ) -> None:
        """Enter vertex-edit mode on the canvas with draggable handles."""
        self.canvas.set_edit_vertices(vertices)

    # ── Population ──
    def set_volume(self,
                   volume: Optional[LazyND2Volume],
                   channel_display: Optional[Dict[str, Dict[str, Any]]] = None,
                   z_mode: str = "max",
                   z_index: int = 0,
                   m: int = 0, t: int = 0, z: int = 0,
                   stage_xy_um: Optional[List[Tuple[float, float]]] = None) -> None:
        """Wire up to a LazyND2Volume so M/T/Z scrolling reads from disk."""
        if self._prefetch is not None:
            self._prefetch.stop()
            self._prefetch = None
        self._teardown_io_worker()
        self._frame_cache.clear()
        self._pinned_keys.clear()
        self._hist_cache.clear()

        self._volume = volume
        self._channels = {}
        self._z_mode = z_mode
        self._z_index = z_index
        self._m = m
        self._t = t
        self._z = z
        self._stage_xy_um = list(stage_xy_um or [])

        for btn in (self._m_play, self._t_play, self._z_play):
            if btn.isChecked():
                btn.setChecked(False)

        if volume is None:
            self._populate_chip_strip([], channel_display or {})
            self.lut_sidebar.rebuild([], channel_display or {})
            self.lut_sidebar.set_tile_layout(
                stage_xy_um=[], pixel_size_um=1.0,
                tile_h=0, tile_w=0, n_multipoints=0,
            )
            self._configure_gpu_canvas_channels([])
            return

        # Slider configuration.
        self._configure_slider(self.m_slider, self.m_label, self._m_row,
                                volume.n_multipoints, self._m,
                                visible=volume.n_multipoints > 1,
                                play_btn=self._m_play, fps_spin=self._m_fps)
        self._configure_slider(self.t_slider, self.t_label, self._t_row,
                                volume.n_timepoints, self._t, visible=True,
                                play_btn=self._t_play, fps_spin=self._t_fps)
        z_visible = volume.n_zslices > 1 and z_mode == "none"
        self._configure_slider(self.z_slider, self.z_label, self._z_row,
                                volume.n_zslices, self._z, visible=z_visible,
                                play_btn=self._z_play, fps_spin=self._z_fps)
        self._update_axis_labels()

        names = list(volume.channel_names)
        self._populate_chip_strip(names, channel_display or {})
        self.lut_sidebar.rebuild(names, channel_display or {})
        # Apply saved LUT params (if any) and seed histograms from samples.
        for name in names:
            cd = (channel_display or {}).get(name, {})
            lut = self.lut_sidebar.lut_for(name)
            if lut is None:
                continue
            if "lut_lo" in cd and "lut_hi" in cd:
                lut.set_contrast(float(cd["lut_lo"]), float(cd["lut_hi"]),
                                  float(cd.get("lut_gamma", 1.0)))
        self._populate_lut_samples_from_volume()

        # V1.36 Phase 4: hand the GPU canvas an empty channel-layer
        # stack sized to this volume. Per-channel ImageItems get
        # populated on the first ``_render_current_frame_gpu`` call.
        self._configure_gpu_canvas_channels(names)

        # Tile section in the sidebar.
        if self._enable_tile_section:
            self.lut_sidebar.set_tile_layout(
                stage_xy_um=self._stage_xy_um,
                pixel_size_um=volume.pixel_size_um,
                tile_h=volume.height, tile_w=volume.width,
                n_multipoints=volume.n_multipoints,
                current_m=self._m,
            )
        else:
            self.lut_sidebar.set_tile_layout(
                stage_xy_um=[], pixel_size_um=1.0,
                tile_h=0, tile_w=0, n_multipoints=0,
            )

        # Start background prefetch for the ND2 volume path.
        if volume is not None:
            # Volume types expose ``reopen()`` so the prefetch thread can
            # get its own handle without leaking the source layout (single
            # file vs. multi-file Z-stack composite).
            def _factory(_v=volume):
                return _v.reopen()

            # V1.35 Phase 3: scale T-axis lookahead with available RAM.
            # Three-tier table matches the framework-agnostic Phase 3
            # doc. A laptop with 2 GB free can't store 16+ neighbors
            # for a multi-channel scrub; the workstation can.
            avail_gb = detect().available_ram_gb
            if avail_gb < 2.0:
                radius_t = 2
            elif avail_gb < 8.0:
                radius_t = 4
            else:
                radius_t = 8

            self._prefetch = PrefetchManager(
                reader_factory=_factory,
                cache=self._frame_cache,
                channels=list(range(volume.n_channels)),
                parent=self,
                radius_t=radius_t,
            )
            self._prefetch.frame_ready.connect(self._on_prefetch_ready)
            self._prefetch.start()
            # V1.34 Phase 2: foreground IO worker — same reopen() handle
            # convention as the prefetcher (per-thread handle; nd2 file
            # state is not thread-safe). Serves cache misses for the
            # plane the user is *about* to display, so the GUI thread
            # never blocks on disk inside _compose_current_frame.
            self._io_worker, self._io_thread = start_io_worker(volume)
            self._io_worker.plane_ready.connect(self._on_io_plane_ready)
            self._io_worker.error.connect(self._on_io_error)
            # Mark the initial M/Z-mode histogram as already sampled.
            self._hist_cache[(self._m, self._z_mode)] = True

        self._refresh()

    def set_channels(self,
                     channels: Dict[str, Any],
                     channel_display: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Wire up to a flat dict of (T, H, W) channel arrays.

        Used on the Recipe page where M is fixed and Z has been
        collapsed already.
        """
        if self._prefetch is not None:
            self._prefetch.stop()
            self._prefetch = None
        self._teardown_io_worker()
        self._frame_cache.clear()
        self._pinned_keys.clear()
        self._hist_cache.clear()

        for btn in (self._m_play, self._t_play, self._z_play):
            if btn.isChecked():
                btn.setChecked(False)

        self._volume = None
        self._channels = dict(channels)
        if not channels:
            self._populate_chip_strip([], channel_display or {})
            self.lut_sidebar.rebuild([], channel_display or {})
            self.lut_sidebar.set_tile_layout(
                stage_xy_um=[], pixel_size_um=1.0,
                tile_h=0, tile_w=0, n_multipoints=0,
            )
            self._configure_gpu_canvas_channels([])
            return

        sample = next(iter(channels.values()))
        n = sample.shape[0]
        # Hide M / Z sliders entirely.
        self._configure_slider(self.m_slider, self.m_label, self._m_row,
                                1, 0, visible=False,
                                play_btn=self._m_play, fps_spin=self._m_fps)
        self._configure_slider(self.z_slider, self.z_label, self._z_row,
                                1, 0, visible=False,
                                play_btn=self._z_play, fps_spin=self._z_fps)
        self._configure_slider(self.t_slider, self.t_label, self._t_row,
                                n, 0, visible=True,
                                play_btn=self._t_play, fps_spin=self._t_fps)
        self._t = 0
        self._update_axis_labels()

        names = list(channels.keys())
        self._populate_chip_strip(names, channel_display or {})
        self.lut_sidebar.rebuild(names, channel_display or {})
        # V1.36 Phase 4: pre-create GPU layers for the recipe-page
        # flat-channel path. The GPU canvas pushes a per-channel
        # plane per chip rather than a precomposited RGB.
        self._configure_gpu_canvas_channels(names)
        for name in names:
            data = channels.get(name)
            cd = (channel_display or {}).get(name, {})
            lut = self.lut_sidebar.lut_for(name)
            if lut is None:
                continue
            if data is not None:
                lut.set_data(data)
            if "lut_lo" in cd and "lut_hi" in cd:
                lut.set_contrast(float(cd["lut_lo"]), float(cd["lut_hi"]),
                                  float(cd.get("lut_gamma", 1.0)))

        # Recipe page uses the flat-channel path; no tile section.
        self.lut_sidebar.set_tile_layout(
            stage_xy_um=[], pixel_size_um=1.0,
            tile_h=0, tile_w=0, n_multipoints=0,
        )
        self._refresh()

    def _configure_slider(self, slider: QSlider, label: QLabel, row,
                           total: int, value: int, visible: bool,
                           play_btn: Optional[QPushButton] = None,
                           fps_spin: Optional[QDoubleSpinBox] = None) -> None:
        slider.blockSignals(True)
        slider.setRange(0, max(0, total - 1))
        slider.setValue(min(value, max(0, total - 1)))
        slider.blockSignals(False)
        # First widget in the row is the axis letter label.
        prefix = row.itemAt(0).widget()
        if prefix is not None:
            prefix.setVisible(visible)
        slider.setVisible(visible)
        label.setVisible(visible)
        if play_btn is not None:
            play_btn.setVisible(visible)
            if not visible and play_btn.isChecked():
                play_btn.setChecked(False)
        if fps_spin is not None:
            fps_spin.setVisible(visible)

    def _populate_chip_strip(self,
                              names: List[str],
                              display: Dict[str, Dict[str, Any]]) -> None:
        for chip in self._chip_strip:
            chip.setParent(None)
            chip.deleteLater()
        self._chip_strip.clear()

        cycle = ["green", "red", "cyan", "magenta", "yellow", "blue", "orange"]
        for i, name in enumerate(names):
            cd = display.get(name, {})
            chip = ChannelChip(
                name,
                color_default=cd.get("color", cycle[i % len(cycle)]),
                enabled=bool(cd.get("enabled", True)),
            )
            chip.state_changed.connect(self._on_chip_state)
            # Insert before the trailing stretch.
            self._chip_layout.insertWidget(
                self._chip_layout.count() - 1, chip)
            self._chip_strip.append(chip)

    def _populate_lut_samples_from_volume(self) -> None:
        """Sample 8 frames at the current M to seed each LUT histogram."""
        if self._volume is None:
            return
        for c, name in enumerate(self._volume.channel_names):
            lut = self.lut_sidebar.lut_for(name)
            if lut is None:
                continue
            n_t = self._volume.n_timepoints
            idxs = np.linspace(0, max(0, n_t - 1), max(1, min(8, n_t)),
                                dtype=int)
            samples: List[np.ndarray] = []
            last_shape: Optional[Tuple[int, int]] = None
            for t in idxs:
                try:
                    f = self._volume.get_frame(c=c, m=self._m, t=int(t),
                                                z=self._z, z_mode=self._z_mode)
                except Exception:
                    continue
                f_2d = self._normalize_to_2d(f)
                if f_2d is None:
                    continue
                samples.append(f_2d.ravel())
                last_shape = f_2d.shape
            if not samples or last_shape is None:
                continue
            lut.set_data(_SingleFrameSeries(samples, last_shape),
                         dtype=self._volume.dtype)

    # ── Slider handlers ──
    def _on_m_changed(self, v: int) -> None:
        self._m = int(v)
        self._update_axis_labels()
        # Keep the sidebar tile widget in sync with slider drags.
        self.lut_sidebar.set_current_m(self._m)
        self.coords_changed.emit(self._m, self._t, self._z)
        if self._all_channels_cached():
            self._do_refresh()
        else:
            self._m_debounce.start()

    def _do_m_refresh(self) -> None:
        """Timer-delayed M refresh: resample histogram only on first visit."""
        hist_key = (self._m, self._z_mode)
        if hist_key not in self._hist_cache:
            self._populate_lut_samples_from_volume()
            self._hist_cache[hist_key] = True
        self._do_refresh()

    def _on_tile_navigate_requested(self, m: int) -> None:
        """User clicked a tile in the sidebar (or expand modal) → jump."""
        # Setting the slider value triggers ``_on_m_changed`` which does
        # the rest (LUT re-sample, frame redraw, coords_changed signal).
        self.m_slider.setValue(int(m))

    def _on_t_changed(self, v: int) -> None:
        self._t = int(v)
        self._update_axis_labels()
        self.coords_changed.emit(self._m, self._t, self._z)
        if self._all_channels_cached():
            self._do_refresh()
        else:
            self._t_debounce.start()

    def _on_z_changed(self, v: int) -> None:
        self._z = int(v)
        self._update_axis_labels()
        self.coords_changed.emit(self._m, self._t, self._z)
        if self._all_channels_cached():
            self._do_refresh()
        else:
            self._z_debounce.start()

    def _set_axis_playing(self, axis: str, playing: bool) -> None:
        timer = {"m": self._m_timer, "t": self._t_timer, "z": self._z_timer}[axis]
        fps_spin = {"m": self._m_fps, "t": self._t_fps, "z": self._z_fps}[axis]
        play_btn = {"m": self._m_play, "t": self._t_play, "z": self._z_play}[axis]
        if playing:
            interval = max(50, int(1000 / fps_spin.value()))
            timer.start(interval)
            play_btn.setText("⏸")
        else:
            timer.stop()
            play_btn.setText("▶")

    def _axis_tick(self, slider: QSlider) -> None:
        if slider.maximum() <= 0:
            return
        slider.setValue((slider.value() + 1) % (slider.maximum() + 1))

    def _on_chip_state(self) -> None:
        # Keep the LUT sidebar swatches in sync when colors change.
        for chip in self._chip_strip:
            self.lut_sidebar.update_swatch(chip.name, chip.color_rgb)
        self._do_refresh()
        self.channels_changed.emit()

    def _on_lut_contrast_changed(self, _name: str, _lo: float,
                                  _hi: float, _gamma: float) -> None:
        self._do_refresh()
        self.channels_changed.emit()

    def _on_prefetch_ready(self, c: int, m: int, t: int,
                           z: int, z_mode: str) -> None:
        """Re-render when the newly cached frame completes the current view."""
        if (m == self._m and t == self._t and z == self._z
                and z_mode == self._z_mode and self._all_channels_cached()):
            self._do_refresh()

    def _on_io_plane_ready(self, request_id: int, key: tuple,
                           plane: np.ndarray) -> None:
        """Receive a plane from the V1.34 foreground :class:`IOWorker`.

        Always caches the plane (stale results may still help a future
        slider re-visit), but only redraws when the result matches the
        most recent slider gesture. The IO worker normalizes to 2D on
        its own thread so the GUI slot stays trivial.
        """
        try:
            self._frame_cache.put(key, plane)
        except Exception:
            pass
        if request_id == self._latest_request_id:
            self._do_refresh()

    def _on_io_error(self, request_id: int, message: str) -> None:
        """Swallow IO worker errors silently for now.

        We deliberately do not surface a modal here: an isolated read
        failure during fast scrubbing is recoverable (the next gesture
        re-requests), and a permanent failure will manifest on every
        subsequent plane so the user notices anyway. Future work can
        wire this into the status bar.
        """
        _ = request_id, message

    def _teardown_io_worker(self) -> None:
        """Stop the foreground IO worker + its thread, if running.

        Called whenever the volume changes (``set_volume`` /
        ``set_channels``). Mirrors the ``PrefetchManager.stop()``
        teardown so neither worker outlives the file it was opened
        for. Bounded ``wait(2000)`` so a stuck disk read can't hang
        app shutdown.
        """
        if self._io_worker is not None:
            try:
                self._io_worker.stop()
            except Exception:
                pass
            self._io_worker = None
        if self._io_thread is not None:
            try:
                self._io_thread.quit()
                self._io_thread.wait(2000)
            except Exception:
                pass
            self._io_thread = None
        # Bump the gating counter so any in-flight plane_ready arriving
        # after teardown gets dropped by _on_io_plane_ready.
        self._latest_request_id = self._request_counter + 1
        self._request_counter = self._latest_request_id

    # ── V1.39 Phase 7 — multi-resolution pyramid support ──
    def attach_pyramid(self, reader) -> None:
        """Bind a :class:`PyramidReader` to this viewer.

        After attach, viewport changes choose the smallest pyramid
        level whose width matches (or exceeds) the visible screen
        extent. Passing ``None`` detaches; ``_active_level`` resets to
        0 so the next refresh reads from the source volume.
        """
        self._pyramid_reader = reader
        self._active_level = 0
        # Connect once. The pyqtgraph ViewBox has a sigRangeChanged
        # signal that fires on every pan/zoom — we use that to drive
        # level selection. The legacy CPU canvas does not expose this
        # signal, so the level always stays at 0 there (which is fine
        # — the legacy path predates pyramids).
        if reader is not None:
            vb = self._viewbox_if_any()
            if vb is not None:
                try:
                    vb.sigRangeChanged.connect(self._on_viewport_changed)
                except Exception:  # noqa: BLE001 — defensive; already-connected re-call is harmless
                    pass
        # Immediate re-render so the right level is picked on attach.
        self._on_viewport_changed()
        self._do_refresh()

    def detach_pyramid(self) -> None:
        """Drop the pyramid binding; next refresh reads from level 0."""
        self._pyramid_reader = None
        self._active_level = 0

    def _viewbox_if_any(self):
        """Return the pyqtgraph :class:`ViewBox` of the active canvas, if any."""
        vb = getattr(self.canvas, "_viewbox", None)
        if vb is not None:
            return vb
        # Legacy canvas: no ViewBox.
        return None

    def _on_viewport_changed(self, *_args) -> None:
        """Pick the best pyramid level for the current viewport.

        Connected to the GPU canvas's :class:`ViewBox.sigRangeChanged`
        signal. Cheap: compares the current image-pixel extent of the
        view rectangle against the widget's screen-pixel size and asks
        the reader which level wins. When the chosen level changes,
        clear pinning, bump the request counter, and request a redraw.
        """
        if self._pyramid_reader is None:
            return
        vb = self._viewbox_if_any()
        if vb is None:
            return
        try:
            view_rect = vb.viewRect()
            screen_w = max(1, vb.width())
            image_w = max(1, int(round(view_rect.width())))
        except Exception:  # noqa: BLE001 — never fail rendering on a probe
            return
        try:
            level = int(self._pyramid_reader.pick_level_for_viewport(
                viewport_screen_px=int(screen_w),
                image_pixels_visible=int(image_w),
            ))
        except Exception:  # noqa: BLE001
            level = 0
        if level == self._active_level:
            return
        # Level change: invalidate pin set (old keys live at a
        # different level), bump the gating counter so in-flight IO
        # worker frames at the old level are dropped on arrival, and
        # force a redraw.
        for old_key in list(self._pinned_keys):
            self._frame_cache.unpin(old_key)
        self._pinned_keys.clear()
        self._request_counter += 1
        self._latest_request_id = self._request_counter
        self._active_level = level
        self._do_refresh()

    def _frame_cache_key(self, c_idx: int) -> tuple:
        """Cache key for the current ``(c, m, t, z, z_mode)`` and active level.

        Level 0 keeps the V1.38 5-tuple shape so the prefetcher and
        IO worker — both of which speak the 5-tuple convention —
        continue working unchanged. Level ≥ 1 uses a 6-tuple that
        cannot collide with the 5-tuples.
        """
        if self._active_level <= 0:
            return (c_idx, self._m, self._t, self._z, self._z_mode)
        return (
            self._active_level, c_idx, self._m, self._t, self._z, self._z_mode,
        )

    def _read_pyramid_plane(self, c_idx: int) -> Optional[np.ndarray]:
        """Synchronously read a single plane from the pyramid.

        Only called when ``_active_level >= 1``. Pyramid planes are
        small (a level-2 plane of a 2048² source is 512×512 ≈ 0.5 MB
        decompressed), so the read is fast enough to do on the GUI
        thread without a worker.
        """
        if self._pyramid_reader is None or self._volume is None:
            return None
        try:
            plane = self._pyramid_reader.get_frame(
                level=self._active_level,
                c=c_idx, m=self._m, t=self._t, z=self._z,
                z_mode=self._z_mode,
            )
        except Exception:  # noqa: BLE001 — fall back to source on any error
            return None
        return self._normalize_to_2d(plane)

    # ── GPU canvas helpers (V1.36 Phase 4) ──
    def _configure_gpu_canvas_channels(self, names: List[str]) -> None:
        """Tell the GPU canvas which channels to allocate :class:`ImageItem`
        layers for. No-op on the legacy CPU canvas — that path
        recomposes a single RGB array per refresh and doesn't need
        per-channel pre-allocation."""
        if not self._use_gpu_canvas:
            return
        try:
            self.canvas.configure_channels(list(names))  # type: ignore[attr-defined]
        except Exception:
            # If the GPU canvas misbehaves, we silently keep going.
            # The render branch below catches Per-call exceptions too.
            pass

    # ── Drawing ──
    def _do_refresh(self) -> None:
        # V1.36 Phase 4: when the GPU canvas is active, push planes
        # per channel and let the GPU additively composite + apply
        # LUT. We only fall back to the legacy RGB composite when a
        # post-process hook is attached (which expects a uint8 RGB
        # array) or when the GPU branch raises.
        if (self._use_gpu_canvas
                and self._frame_post_process is None
                and hasattr(self.canvas, "update_channel")):
            try:
                if self._render_current_frame_gpu():
                    return
                # If the GPU branch returns False (e.g. no enabled
                # channels with cached planes yet) we fall through to
                # the legacy path so the canvas still shows *some*
                # representation — typically a blank frame.
            except Exception:
                # GPU path raised — fall through to the safe CPU
                # composite path rather than show nothing.
                pass
        composite = self._compose_current_frame()
        if composite is None:
            return
        self.canvas.set_image(composite)

    # Backward-compat alias used by set_volume / set_channels / apply_channel_state.
    _refresh = _do_refresh

    # ── GPU-mode render (V1.36 Phase 4) ──
    def _render_current_frame_gpu(self) -> bool:
        """Per-channel GPU render path.

        Returns True if at least one channel was successfully pushed
        to the canvas (the caller skips the CPU composite). Returns
        False if no cached planes were available — the caller may
        then fall back to the legacy path to display *something*.
        Issues any missing-plane reads through the V1.34 IO worker
        exactly like ``_compose_current_frame`` does, so cache misses
        don't block the GUI thread.
        """
        sources_for_pin: List[Tuple[Any, tuple]] = []
        missing_channels: List[int] = []
        any_pushed = False

        if self._volume is not None:
            for c_idx, name in enumerate(self._volume.channel_names):
                chip = next(
                    (c for c in self._chip_strip if c.name == name), None)
                if chip is None:
                    self.canvas.set_channel_visible(c_idx, False)  # type: ignore[attr-defined]
                    continue

                # Channel visibility + color + (linear) levels — these
                # are the GPU-only changes the chip / LUT widgets used
                # to trigger via a full recompose.
                self.canvas.set_channel_visible(c_idx, chip.enabled)  # type: ignore[attr-defined]
                self.canvas.set_channel_color(c_idx, chip.color_rgb)  # type: ignore[attr-defined]
                lut = self.lut_sidebar.lut_for(name)
                if lut is not None:
                    lo, hi, _gamma = lut.get_contrast()
                    self.canvas.set_channel_levels(c_idx, (lo, hi))  # type: ignore[attr-defined]

                if not chip.enabled:
                    continue

                key = self._frame_cache_key(c_idx)
                frame_2d = self._frame_cache.get(key)
                if frame_2d is None:
                    # Level ≥ 1: synchronous pyramid read (planes
                    # are small; we don't pay for IOWorker dispatch).
                    if self._active_level >= 1:
                        plane = self._read_pyramid_plane(c_idx)
                        if plane is not None:
                            self._frame_cache.put(key, plane)
                            frame_2d = plane
                    if frame_2d is None:
                        missing_channels.append(c_idx)
                        continue
                self.canvas.update_channel(c_idx, frame_2d)  # type: ignore[attr-defined]
                sources_for_pin.append((chip, key))
                any_pushed = True

            # Level ≥ 1 reads are synchronous (above); the IO worker
            # only services level-0 misses where the source decode
            # dominates frame time. At higher levels every miss is
            # already resolved by the time we reach this branch.
            if (missing_channels and self._io_worker is not None
                    and self._active_level == 0):
                self._request_counter += 1
                self._latest_request_id = self._request_counter
                self._io_worker.cancel_all()
                for c_idx in missing_channels:
                    self._io_worker.submit(PlaneRequest(
                        priority=0,
                        request_id=self._latest_request_id,
                        c=c_idx, m=self._m, t=self._t, z=self._z,
                        z_mode=self._z_mode,
                    ))
        else:
            # Recipe-page flat-channel path — same as the CPU branch,
            # but we push planes per chip rather than precompositing.
            for c_idx, chip in enumerate(self._chip_strip):
                self.canvas.set_channel_visible(c_idx, chip.enabled)  # type: ignore[attr-defined]
                self.canvas.set_channel_color(c_idx, chip.color_rgb)  # type: ignore[attr-defined]
                lut = self.lut_sidebar.lut_for(chip.name)
                if lut is not None:
                    lo, hi, _gamma = lut.get_contrast()
                    self.canvas.set_channel_levels(c_idx, (lo, hi))  # type: ignore[attr-defined]
                if not chip.enabled:
                    continue
                data = self._channels.get(chip.name)
                if data is None:
                    continue
                try:
                    frame = np.asarray(data[self._t])
                except Exception:
                    continue
                frame_2d = self._normalize_to_2d(frame)
                if frame_2d is None:
                    continue
                self.canvas.update_channel(c_idx, frame_2d)  # type: ignore[attr-defined]
                any_pushed = True

        # Pinning bookkeeping mirrors the CPU branch — keep on-screen
        # planes resident on a tight cache budget.
        new_pinned = {key for _chip, key in sources_for_pin if key is not None}
        for old_key in self._pinned_keys - new_pinned:
            self._frame_cache.unpin(old_key)
        for new_key in new_pinned - self._pinned_keys:
            self._frame_cache.pin(new_key)
        self._pinned_keys = new_pinned

        # Velocity-biased prefetch (Phase 3) is independent of the
        # render path; trigger it on every GPU refresh too.
        if self._prefetch is not None and self._volume is not None:
            self._prefetch.request_neighbors(
                m=self._m, t=self._t, z=self._z,
                t_range=(0, self._volume.n_timepoints - 1),
                z_range=(0, self._volume.n_zslices - 1),
                z_mode=self._z_mode,
            )

        return any_pushed

    def _all_channels_cached(self) -> bool:
        """Return True if every enabled channel for current coords is cached.

        V1.39 Phase 7: honours :attr:`_active_level` via
        :meth:`_frame_cache_key`. At level 0 this is the V1.38 5-tuple
        lookup; at level ≥ 1 it checks the pyramid-level cache slot.
        """
        if self._volume is None:
            return False
        for c_idx, name in enumerate(self._volume.channel_names):
            chip = next((c for c in self._chip_strip if c.name == name), None)
            if chip is None or not chip.enabled:
                continue
            key = self._frame_cache_key(c_idx)
            if not self._frame_cache.contains(key):
                return False
        return True

    @staticmethod
    def _normalize_to_2d(frame) -> Optional[np.ndarray]:
        """Coerce any reasonable frame shape into 2D `(H, W)`.

        Handles the cases that real-world loaders produce:

        - ``(H, W)``                              → unchanged
        - ``(H, W, 3) | (H, W, 4)``  RGB / RGBA   → luminance (Rec. 601)
        - ``(Z, H, W) | (T, H, W)``               → first slice along axis 0
        - ``(T, Z, H, W)`` and deeper             → recursively slice [0]
        - ``(1, H, W, 1)`` etc.                   → squeezed first

        Returns ``None`` when the frame can't be reduced to 2D (e.g. 1D
        or 0D arrays).
        """
        if frame is None:
            return None
        a = np.squeeze(np.asarray(frame))
        if a.ndim == 2:
            return a
        if a.ndim == 3:
            if a.shape[-1] in (3, 4):
                # RGB / RGBA → Rec. 601 luminance
                r, g, b = a[..., 0], a[..., 1], a[..., 2]
                lum = 0.299 * r.astype(np.float32) \
                      + 0.587 * g.astype(np.float32) \
                      + 0.114 * b.astype(np.float32)
                return lum.astype(a.dtype if np.issubdtype(a.dtype, np.integer)
                                  else np.float32)
            return a[0]
        while a.ndim > 2:
            a = a[0]
        return a if a.ndim == 2 else None

    def _compose_current_frame(self) -> Optional[np.ndarray]:
        # V1.35 Phase 3: track keys alongside (chip, plane) so we can
        # pin the displayed planes against eviction.
        sources: List[Tuple[ChannelChip, np.ndarray, Optional[tuple]]] = []
        missing_channels: List[int] = []
        if self._volume is not None:
            for c_idx, name in enumerate(self._volume.channel_names):
                chip = next((c for c in self._chip_strip if c.name == name), None)
                if chip is None or not chip.enabled:
                    continue
                key = self._frame_cache_key(c_idx)
                frame_2d = self._frame_cache.get(key)
                if frame_2d is None:
                    # V1.39 Phase 7: level ≥ 1 — read pyramid plane
                    # synchronously (cheap; small chunks).
                    if self._active_level >= 1:
                        plane = self._read_pyramid_plane(c_idx)
                        if plane is not None:
                            self._frame_cache.put(key, plane)
                            frame_2d = plane
                    if frame_2d is None:
                        # V1.34 Phase 2: level-0 misses go through the
                        # IO worker so the GUI does not block on disk.
                        missing_channels.append(c_idx)
                        continue
                sources.append((chip, frame_2d, key))

            if (missing_channels and self._io_worker is not None
                    and self._active_level == 0):
                self._request_counter += 1
                self._latest_request_id = self._request_counter
                # Drop any in-flight foreground requests the user has
                # already moved past — the prefetcher is the one that
                # speculatively warms neighbors, the IO worker is only
                # for the plane(s) the viewer wants *right now*.
                self._io_worker.cancel_all()
                for c_idx in missing_channels:
                    self._io_worker.submit(PlaneRequest(
                        priority=0,
                        request_id=self._latest_request_id,
                        c=c_idx, m=self._m, t=self._t, z=self._z,
                        z_mode=self._z_mode,
                    ))
        else:
            # Recipe-page in-RAM channels — no cache, no pin needed.
            for chip in self._chip_strip:
                if not chip.enabled:
                    continue
                data = self._channels.get(chip.name)
                if data is None:
                    continue
                try:
                    frame = np.asarray(data[self._t])
                except Exception:
                    continue
                frame_2d = self._normalize_to_2d(frame)
                if frame_2d is None:
                    continue
                sources.append((chip, frame_2d, None))

        if not sources:
            return None

        # V1.35 Phase 3: pin the keys that composited into this render
        # and unpin any previously-pinned keys we no longer use. Cache
        # eviction (driven by prefetcher writes) will skip these so the
        # on-screen plane stays resident even on a tight budget.
        new_pinned = {key for _chip, _frame, key in sources if key is not None}
        for old_key in self._pinned_keys - new_pinned:
            self._frame_cache.unpin(old_key)
        for new_key in new_pinned - self._pinned_keys:
            self._frame_cache.pin(new_key)
        self._pinned_keys = new_pinned

        sample = sources[0][1]
        h, w = sample.shape
        composite = np.zeros((h, w, 3), dtype=np.float32)
        for chip, frame, _key in sources:
            lut = self.lut_sidebar.lut_for(chip.name)
            if lut is None:
                continue
            lo, hi, gamma = lut.get_contrast()
            mapped = apply_lut(frame, lo, hi, gamma).astype(np.float32)
            r, g, b = chip.color_rgb
            composite[..., 0] += mapped * (r / 255.0)
            composite[..., 1] += mapped * (g / 255.0)
            composite[..., 2] += mapped * (b / 255.0)

        if self._prefetch is not None and self._volume is not None:
            # V1.35 Phase 3: drop the V1.17 ``n=5`` override so the
            # manager applies its velocity-biased ``radius_t`` window.
            self._prefetch.request_neighbors(
                m=self._m, t=self._t, z=self._z,
                t_range=(0, self._volume.n_timepoints - 1),
                z_range=(0, self._volume.n_zslices - 1),
                z_mode=self._z_mode,
            )

        result = np.clip(composite, 0, 255).astype(np.uint8)
        if self._frame_post_process is not None:
            result = self._frame_post_process(result, self._t, self._m)
        return result

    def _update_axis_labels(self) -> None:
        if self._volume is not None:
            self.m_label.setText(f"{self._m + 1}/{self._volume.n_multipoints}")
            self.t_label.setText(f"{self._t + 1}/{self._volume.n_timepoints}")
            self.z_label.setText(f"{self._z + 1}/{self._volume.n_zslices}")
        else:
            n_t = self.t_slider.maximum() + 1 if self.t_slider.maximum() >= 0 else 0
            self.t_label.setText(f"{self._t + 1}/{n_t}")

    # ── Read-out / round-tripping ──
    def channel_state(self) -> Dict[str, Dict[str, Any]]:
        """Snapshot every chip + LUT state for the experiment record."""
        out: Dict[str, Dict[str, Any]] = {}
        for chip in self._chip_strip:
            lut = self.lut_sidebar.lut_for(chip.name)
            lo, hi, g = lut.get_contrast() if lut else (0.0, 65535.0, 1.0)
            out[chip.name] = {
                "enabled": chip.enabled,
                "color": chip.color_name,
                "lut_lo": lo, "lut_hi": hi, "lut_gamma": g,
            }
        return out

    def coords(self) -> Tuple[int, int, int]:
        return self._m, self._t, self._z

    def cache_stats_text(self) -> str:
        """One-line summary of the frame cache state.

        Format: ``Cache: <used>/<budget> MB · hit <rate>% · evictions <n>``.
        Used by the V1.33 profiling harness and intended for a future
        status-bar tooltip; safe to call from the GUI thread (the cache
        snapshots stats under its own lock).
        """
        s = self._frame_cache.stats
        used_mb = s.current_bytes / (1024 ** 2)
        budget_mb = self._frame_cache.max_bytes / (1024 ** 2)
        return (
            f"Cache: {used_mb:,.0f}/{budget_mb:,.0f} MB · "
            f"hit {s.hit_rate * 100:.1f}% · "
            f"evictions {s.evictions}"
        )

    def apply_channel_state(self, state: Dict[str, Dict[str, Any]]) -> None:
        for chip in self._chip_strip:
            cfg = state.get(chip.name)
            if not cfg:
                continue
            chip.cb.blockSignals(True)
            chip.cb.setChecked(bool(cfg.get("enabled", True)))
            chip.cb.blockSignals(False)
            color = cfg.get("color")
            if color and color in CHANNEL_COLORS:
                chip.combo_color.blockSignals(True)
                chip.combo_color.setCurrentText(color)
                chip.combo_color.blockSignals(False)
            lut = self.lut_sidebar.lut_for(chip.name)
            if lut is not None and "lut_lo" in cfg and "lut_hi" in cfg:
                lut.set_contrast(
                    float(cfg["lut_lo"]),
                    float(cfg["lut_hi"]),
                    float(cfg.get("lut_gamma", 1.0)),
                )
        # Re-sync sidebar swatches to chip colors after state apply.
        for chip in self._chip_strip:
            self.lut_sidebar.update_swatch(chip.name, chip.color_rgb)
        self._refresh()
