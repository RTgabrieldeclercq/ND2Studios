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
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QSlider, QSplitter, QVBoxLayout,
    QWidget,
)

from nd2studios.backend.materialized_dataset import MaterializedDataset
from nd2studios.core.settings import Settings
from nd2studios.widgets.frame_strip import FrameStrip
from nd2studios.widgets.icon_button import bind_toggle_icon, icon_button
from nd2studios.widgets.image_viewer import (
    CHANNEL_COLORS, ImageCanvas, ZoomToolbar,
)
from nd2studios.widgets.lut_histogram import apply_lut
from nd2studios.widgets.lut_sidebar import LutSidebar
from nd2studios.workers.pre_render_worker import PreRenderWorker
from nd2studios.utils.perf import perf_log, perf_block


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
    # Frame-strip tile selection changed: (axis "m"/"t"/"z", frozenset[int]).
    selection_changed = Signal(str, object)
    # Any channel toggle / color / LUT change.
    channels_changed = Signal()
    # Click on the corner tile preview overlay.
    stitch_requested = Signal()
    # Crop rect selected via drag on the canvas (x, y, w, h in image pixels).
    crop_rect_selected = Signal(int, int, int, int)
    # V1.43 — tile-strip "crop to selected frames" request: (axis, frozenset).
    crop_to_selection_requested = Signal(str, object)
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

        # V1.41: ``_volume`` holds a :class:`MaterializedDataset` (or any
        # object exposing the same surface). All pixel data is resident
        # in RAM after :class:`LoadWorker` finishes; the viewer reads
        # planes by direct ndarray indexing. The IOWorker / FrameCache /
        # PrefetchManager stack from V1.34-V1.40 is gone — there is
        # nothing to async, cache, or speculatively warm.
        self._volume: Optional[MaterializedDataset] = None
        self._channels: Dict[str, Any] = {}
        self._chip_strip: List[ChannelChip] = []
        self._z_mode: str = "max"
        self._z_index: int = 0
        self._stage_xy_um: List[Tuple[float, float]] = []
        # V1.43 — per-frame acquisition timestamps (seconds) for the T-axis
        # tile-strip metadata tooltip. ``None`` until the host supplies them.
        self._frame_timestamps = None

        self._m: int = 0
        self._t: int = 0
        self._z: int = 0

        self._m_timer = QTimer(self)
        self._t_timer = QTimer(self)
        self._z_timer = QTimer(self)

        # T-slider debounce: 5ms coalesces rapid drag events without
        # adding perceptible latency. The old 50ms was designed for the
        # lazy-loading + IOWorker stack (V1.40 and earlier); with all
        # data in RAM this overhead is gone.
        self._t_debounce = QTimer(self)
        self._t_debounce.setSingleShot(True)
        self._t_debounce.setInterval(5)
        self._t_debounce.timeout.connect(self._do_refresh)

        self._z_debounce = QTimer(self)
        self._z_debounce.setSingleShot(True)
        self._z_debounce.setInterval(50)
        self._z_debounce.timeout.connect(self._do_refresh)

        self._m_debounce = QTimer(self)
        self._m_debounce.setSingleShot(True)
        self._m_debounce.setInterval(80)
        self._m_debounce.timeout.connect(self._do_m_refresh)

        # Debounce for LUT/color changes: refresh the current frame instantly,
        # but defer the full render-cache rebuild until the user stops adjusting.
        self._lut_rebuild_timer = QTimer(self)
        self._lut_rebuild_timer.setSingleShot(True)
        self._lut_rebuild_timer.setInterval(250)
        self._lut_rebuild_timer.timeout.connect(self._rebuild_render_cache_after_lut)

        self._hist_cache: dict = {}  # (m, z_mode) -> True

        # V1.42 — optional multi-resolution pyramid (BigDataViewer-style).
        # ``_attach_pyramid_reader_to_viewers`` in :class:`MainWindow`
        # calls :meth:`attach_pyramid` after the build worker finishes;
        # we then prefer a coarse level when zoomed out so the GPU /
        # CPU compose pays for fewer pixels.
        self._pyramid_reader = None  # set by attach_pyramid()
        self._frame_post_process: Optional[Callable] = None

        # Pre-render frame cache.
        # _render_cache: shared dict written by PreRenderWorker, read here.
        # _pixmap_cache: main-thread QPixmap conversion of render_cache for
        # the current M — frame display becomes a pure paintEvent swap.
        self._render_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._pixmap_cache: Dict[int, QPixmap] = {}  # T → QPixmap for current M
        self._cache_m: int = -1      # which M the pixmap cache covers
        self._cache_ready: bool = False
        self._pre_render_worker: Optional[PreRenderWorker] = None

        # Post-process overlay cache.
        # Keyed by (m, t, pp_version); invalidated whenever the overlay
        # content changes (new shape, vertex edit, LUT change).  Avoids
        # re-running the full LUT+compose on every refresh when an overlay
        # is active — the same gain the render_cache gives for base frames.
        self._pp_cache: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._pp_pixmap_cache: Dict[Tuple[int, int, int], QPixmap] = {}
        self._pp_version: int = 0

        self._pixmap_build_timer = QTimer(self)
        self._pixmap_build_timer.setSingleShot(True)
        self._pixmap_build_timer.setInterval(0)  # fire on next event-loop tick
        self._pixmap_build_timer.timeout.connect(self._build_next_pixmap_batch)
        self._pixmap_build_t_idx: int = 0

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

        self._zoom_toolbar_widget = self.zoom_toolbar  # for set_controls_visible

        # Pixel-hover readout + overlay-style control wiring (V1.46).
        self.canvas.hover.connect(self._on_pixel_hover)
        self.zoom_toolbar.overlay_style_changed.connect(self._on_overlay_style_changed)
        self.zoom_toolbar.coord_mode_changed.connect(self._on_coord_mode_changed)
        self._hover_stage: bool = False
        self._last_hover: Optional[Tuple[float, float]] = None
        self._hover_plane_cache: Dict[str, np.ndarray] = {}
        self._hover_plane_key: Optional[tuple] = None

        # M / T / Z sliders (each row includes a ▶/⏸ play button + fps spinbox).
        slider_box = QFrame()
        self._slider_box = slider_box
        slider_box.setObjectName("contentArea")
        slider_layout = QVBoxLayout(slider_box)
        slider_layout.setContentsMargins(8, 4, 8, 4)
        slider_layout.setSpacing(2)
        self._m_row, self.m_slider, self.m_label, self._m_total, self._m_play, self._m_fps, self._m_strip = \
            self._make_axis_row("M")
        self._t_row, self.t_slider, self.t_label, self._t_total, self._t_play, self._t_fps, self._t_strip = \
            self._make_axis_row("T")
        self._z_row, self.z_slider, self.z_label, self._z_total, self._z_play, self._z_fps, self._z_strip = \
            self._make_axis_row("Z")
        self.m_slider.valueChanged.connect(self._on_m_changed)
        self.t_slider.valueChanged.connect(self._on_t_changed)
        self.z_slider.valueChanged.connect(self._on_z_changed)
        # V1.43 — per-axis tile-strip selection + crop wiring.
        self._m_strip.selection_changed.connect(
            lambda s: self._on_strip_selection("m", s))
        self._t_strip.selection_changed.connect(
            lambda s: self._on_strip_selection("t", s))
        self._z_strip.selection_changed.connect(
            lambda s: self._on_strip_selection("z", s))
        self._m_strip.crop_requested.connect(
            lambda s: self._on_strip_crop("m", s))
        self._t_strip.crop_requested.connect(
            lambda s: self._on_strip_crop("t", s))
        self._z_strip.crop_requested.connect(
            lambda s: self._on_strip_crop("z", s))
        # V1.44 — strip current changes (click / drag / arrow keys) take a
        # direct fast-refresh path instead of the debounced slider cascade.
        self._m_strip.current_changed.connect(lambda v: self._on_strip_current("m", v))
        self._t_strip.current_changed.connect(lambda v: self._on_strip_current("t", v))
        self._z_strip.current_changed.connect(lambda v: self._on_strip_current("z", v))
        # Per-axis selections (frozenset of indices). T drives M/Z inheritance.
        self._axis_selection = {"m": frozenset(), "t": frozenset(), "z": frozenset()}
        self._t_strip.set_meta_fn(lambda i: self._frame_meta_text("t", i))
        self._m_strip.set_meta_fn(lambda i: self._frame_meta_text("m", i))
        self._z_strip.set_meta_fn(lambda i: self._frame_meta_text("z", i))
        self.m_label.editingFinished.connect(lambda: self._on_axis_label_edited("m"))
        self.t_label.editingFinished.connect(lambda: self._on_axis_label_edited("t"))
        self.z_label.editingFinished.connect(lambda: self._on_axis_label_edited("z"))
        self._m_play.toggled.connect(lambda on: self._set_axis_playing("m", on))
        self._t_play.toggled.connect(lambda on: self._set_axis_playing("t", on))
        self._z_play.toggled.connect(lambda on: self._set_axis_playing("z", on))
        self._m_timer.timeout.connect(lambda: self._axis_tick(self.m_slider))
        self._t_timer.timeout.connect(lambda: self._axis_tick(self.t_slider))
        self._z_timer.timeout.connect(lambda: self._axis_tick(self.z_slider))
        # V1.44 — axis order is T (top), M, Z (bottom). (The "Play All" control
        # is no longer per-viewer; it's a universal banner across the whole
        # viewer area — see PlayAllBanner — shown only with multiple files.)
        slider_layout.addLayout(self._t_row)
        slider_layout.addLayout(self._m_row)
        slider_layout.addLayout(self._z_row)
        left_layout.addWidget(slider_box)

        # Chip strip — compact toggle + color, one row across.
        chip_box = QFrame()
        self._chip_box = chip_box
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
        row.setSpacing(4)
        lbl = QLabel(label + ":")
        lbl.setFixedWidth(20)
        lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        row.addWidget(lbl)
        # V1.43 — NIS-Elements-style rectangular tile strip is the visible
        # control. A hidden QSlider remains the source of truth for the
        # current index so all existing playback / value code keeps working.
        strip = FrameStrip(label)
        row.addWidget(strip, stretch=1)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 0)
        slider.setParent(strip)
        slider.hide()
        # slider → strip keeps the tiles current when the slider is driven by
        # other code. The strip → refresh direction is wired in __init__ via
        # _on_strip_current (V1.44), which does a *direct* fast refresh and
        # bypasses the per-axis debounce so arrow/drag stepping is as smooth as
        # FPS playback (M/Z debounce was 80/50 ms and made arrows feel choppy).
        slider.valueChanged.connect(lambda v: strip.set_current(v, emit=False))
        # Editable current-frame number (user can type to jump).
        info_edit = QLineEdit("1")
        info_edit.setObjectName("axisFrameEdit")
        info_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_edit.setFixedWidth(36)
        info_edit.setFixedHeight(26)
        info_edit.setToolTip("Current frame — type a number and press Enter to jump")
        row.addWidget(info_edit)
        # Static total — same font size as the counter and vertically centred
        # to its height so "/N" reads cleanly next to the current frame (V1.44).
        info_total = QLabel("/1")
        info_total.setFixedWidth(30)
        info_total.setFixedHeight(26)
        info_total.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        info_total.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 10pt 'Helvetica Neue';")
        row.addWidget(info_total)
        play_btn = icon_button("fa5s.play", f"Play / pause {label} axis",
                               checkable=True, object_name="playBtn",
                               button_px=26, icon_px=12)
        bind_toggle_icon(play_btn, "fa5s.play", "fa5s.pause",
                         color_checked=Settings.ACCENT_GREEN)
        row.addWidget(play_btn)
        fps_spin = QDoubleSpinBox()
        fps_spin.setRange(0.1, 60.0)
        fps_spin.setValue(5.0)
        fps_spin.setSingleStep(0.5)
        fps_spin.setSuffix(" fps")
        fps_spin.setFixedWidth(72)
        fps_spin.setToolTip("Playback speed")
        fps_spin.setSizePolicy(fps_spin.sizePolicy().horizontalPolicy(),
                               QSizePolicy.Policy.Fixed)
        row.addWidget(fps_spin)
        row.setAlignment(info_edit, Qt.AlignmentFlag.AlignTop)
        row.setAlignment(info_total, Qt.AlignmentFlag.AlignTop)
        row.setAlignment(play_btn, Qt.AlignmentFlag.AlignTop)
        row.setAlignment(fps_spin, Qt.AlignmentFlag.AlignTop)
        return row, slider, info_edit, info_total, play_btn, fps_spin, strip

    def refresh(self) -> None:
        """Re-render the current frame, dropping caches so the volume is re-read.

        Use when a lazy volume's backing data changes *in place* (e.g. the
        Pipelines preview re-points which plane is recipe-processed) — unlike
        :meth:`set_volume` this preserves the frame selection, LUTs and slider
        positions.
        """
        self._render_cache.clear()
        self._pixmap_cache.clear()
        self._pp_cache.clear()
        self._pp_pixmap_cache.clear()
        self._cache_m = -1
        self._cache_ready = False
        self._refresh()

    def set_current_frame(self, m: Optional[int] = None,
                          t: Optional[int] = None) -> None:
        """Programmatically navigate to ``(m, t)`` — used for live run streaming.

        Reuses the immediate strip-step path (no debounce) so the display updates
        right away as each analysed frame arrives. Out-of-range values are clamped.
        """
        if m is not None and int(m) != self._m:
            self._on_strip_current("m", int(m))
        if t is not None and int(t) != self._t:
            self._on_strip_current("t", int(t))

    def invalidate_post_process_cache(self) -> None:
        """Discard cached overlay composites so the next refresh recomputes them.

        Call this whenever the overlay *content* changes (new shape drawn,
        vertex moved, analysis result received) but the base frame and LUT
        have not changed.  LUT / chip changes should use
        _invalidate_render_cache() which also clears this cache.
        """
        self._pp_version += 1
        self._pp_cache.clear()
        self._pp_pixmap_cache.clear()

    def set_frame_post_process(self, fn: Optional[Callable]) -> None:
        """Set a callable applied to the composited frame before display.

        fn(rgb_uint8: np.ndarray, t: int, m: int) -> np.ndarray
        Pass None to remove any active hook.
        """
        self._frame_post_process = fn
        # The overlay-style control only applies when an overlay is active.
        self.zoom_toolbar.set_overlay_button_visible(fn is not None)
        self.invalidate_post_process_cache()
        self._do_refresh()

    # ── Overlay style + pixel-hover readout (V1.46) ───────────────────────
    def overlay_style(self) -> Dict[str, Any]:
        """The current overlay-style settings (color / weight / multicolor /
        alpha / enabled) — read by the page overlay painters."""
        return self.zoom_toolbar.overlay_style()

    def _on_overlay_style_changed(self, _style: Dict[str, Any]) -> None:
        # Overlay content (not the base) changed → recompute only the overlay.
        self.invalidate_post_process_cache()
        self._do_refresh()

    def _on_coord_mode_changed(self, stage: bool) -> None:
        self._hover_stage = bool(stage)
        if self._last_hover is not None:
            self._on_pixel_hover(*self._last_hover)

    def _plane_for_channel(self, name: str) -> Optional[np.ndarray]:
        """Raw 2-D intensity plane for ``name`` at the current (m, t, z),
        cached per coordinate so repeated hovers are O(1)."""
        key = (self._m, self._t, self._z, self._z_mode)
        if self._hover_plane_key != key:
            self._hover_plane_cache = {}
            self._hover_plane_key = key
        if name in self._hover_plane_cache:
            return self._hover_plane_cache[name]
        plane: Optional[np.ndarray] = None
        if self._volume is not None:
            try:
                c_idx = list(self._volume.channel_names).index(name)
                plane = self._read_volume_plane(c_idx, name)
            except (ValueError, Exception):  # noqa: BLE001
                plane = None
        else:
            data = self._channels.get(name)
            if data is not None:
                try:
                    arr = np.asarray(data)
                    plane = arr if arr.ndim == 2 else arr[min(self._t, arr.shape[0] - 1)]
                except Exception:  # noqa: BLE001
                    plane = None
        if plane is not None:
            plane = np.asarray(plane)
            if plane.ndim != 2:
                plane = None
        self._hover_plane_cache[name] = plane
        return plane

    def _pixel_to_stage(self, row: int, col: int) -> Optional[Tuple[float, float]]:
        """Image pixel → absolute stage µm, matching ``results_engine``'s
        centroid_*_stage_um formula (origin at the frame center)."""
        if self._volume is None or not self._stage_xy_um:
            return None
        if self._m >= len(self._stage_xy_um):
            return None
        sx, sy = self._stage_xy_um[self._m]
        px = float(getattr(self._volume, "pixel_size_um", 0.0) or 0.0)
        if px <= 0:
            return None
        h = float(self._volume.height)
        w = float(self._volume.width)
        stage_x = sx + (col - w / 2.0) * px
        stage_y = sy + (row - h / 2.0) * px
        return stage_x, stage_y

    def _on_pixel_hover(self, iy: float, ix: float) -> None:
        """Update the toolbar readout with position + per-channel intensity."""
        if iy < 0 or ix < 0:
            self._last_hover = None
            self.zoom_toolbar.set_hover_text("")
            return
        row, col = int(iy), int(ix)
        self._last_hover = (float(iy), float(ix))

        if self._hover_stage:
            stage = self._pixel_to_stage(row, col)
            if stage is not None:
                pos = f"X {stage[0]:.1f}µm  Y {stage[1]:.1f}µm"
            else:
                pos = f"X {col}  Y {row} (px)"   # no stage data → fall back
        else:
            pos = f"X {col}  Y {row}"

        parts = [pos]
        for chip in self._chip_strip:
            if not getattr(chip, "enabled", False):
                continue
            plane = self._plane_for_channel(chip.name)
            if plane is None or not (0 <= row < plane.shape[0] and 0 <= col < plane.shape[1]):
                continue
            val = plane[row, col]
            parts.append(f"{chip.name} {int(val)}")
        self.zoom_toolbar.set_hover_text("   ".join(parts))

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

    def set_channel_contrast(self, name: str, lo: float, hi: float,
                             gamma: float = 1.0) -> None:
        """Programmatically set a channel's LUT contrast and refresh.

        Used by the Recipe page (V1.44) to mirror the processed viewer's LUT
        onto the raw viewer so the two can be compared at matched intensities.
        """
        lut = self.lut_sidebar.lut_for(name)
        if lut is None:
            return
        lut.set_contrast(float(lo), float(hi), float(gamma))
        self._on_lut_contrast_changed(name, float(lo), float(hi), float(gamma))

    def lut_effective_max(self, name: str) -> float:
        """Effective intensity max for a channel's LUT — the actual data max
        (histogram top edge) when known, else the dtype max. Lets callers map
        contrast between viewers of different bit depths by percentage."""
        lut = self.lut_sidebar.lut_for(name)
        if lut is None:
            return 65535.0
        edges = getattr(lut, "_hist_edges", None)
        if edges is not None and len(edges) > 0 and float(edges[-1]) > 0:
            return float(edges[-1])
        return float(getattr(lut, "_dtype_max", 65535.0))

    def take_control_widgets(self) -> list:
        """Detach this viewer's control widgets (zoom toolbar, frame-strip box,
        channel-chip box) and return them so a host can place them in a shared
        bar spanning multiple viewers (Recipe page single control set across raw
        + processed, V1.44). The widgets stay wired to this viewer's signals, so
        they keep driving it from their new parent."""
        widgets = []
        for w in (getattr(self, "_zoom_toolbar_widget", None),
                  getattr(self, "_slider_box", None),
                  getattr(self, "_chip_box", None)):
            if w is not None:
                w.setParent(None)
                widgets.append(w)
        return widgets

    def set_controls_visible(self, visible: bool) -> None:
        """Show/hide this viewer's own navigation controls (frame strips, zoom
        toolbar, channel chips). Used on the Recipe page (V1.44) so the raw and
        processed viewers share a single visible control set — the hidden
        viewer is driven externally via :meth:`mirror_from`."""
        visible = bool(visible)
        for w in (getattr(self, "_slider_box", None),
                  getattr(self, "_zoom_toolbar_widget", None),
                  getattr(self, "_chip_box", None)):
            if w is not None:
                w.setVisible(visible)

    def mirror_from(self, master: "MultiAxisViewer") -> None:
        """Slave this viewer to ``master``: mirror coordinates, channel state,
        and zoom/pan. The slave's own controls should be hidden first via
        :meth:`set_controls_visible(False)`."""
        master.coords_changed.connect(self._apply_external_coords)
        master.channels_changed.connect(
            lambda: self.apply_channel_state(master.channel_state()))
        master.canvas.zoom_changed.connect(
            lambda _z: self._apply_external_view(master))

    def _apply_external_coords(self, m: int, t: int, z: int) -> None:
        # Emit-safe: update sliders with signals blocked and refresh directly,
        # so mirroring does NOT re-fire coords_changed (which would loop with a
        # host page that also syncs the two viewers, e.g. RecipePage M-sync).
        changed = False
        for slider, val, attr in ((self.m_slider, m, "_m"),
                                  (self.t_slider, t, "_t"),
                                  (self.z_slider, z, "_z")):
            if 0 <= val <= slider.maximum() and val != slider.value():
                slider.blockSignals(True)
                slider.setValue(val)
                slider.blockSignals(False)
                setattr(self, attr, val)
                strip = self._strip_for(slider)
                if strip is not None:
                    strip.set_current(val, emit=False)
                changed = True
        if changed:
            self._update_axis_labels()
            self._do_refresh()

    def _apply_external_view(self, master: "MultiAxisViewer") -> None:
        """Match the master canvas's zoom/pan."""
        mc, sc = master.canvas, self.canvas
        if hasattr(sc, "set_zoom_level"):
            pan = (getattr(mc, "_pan_x", 0.0), getattr(mc, "_pan_y", 0.0))
            sc.set_zoom_level(float(getattr(mc, "_zoom", 1.0)), pan)

    def set_frame_timestamps(self, timestamps) -> None:
        """Supply per-frame acquisition timestamps (seconds) for the T-axis
        tile metadata tooltip. Pass ``None`` to clear."""
        self._frame_timestamps = timestamps

    # ── Population ──
    def set_volume(self,
                   volume: Optional[MaterializedDataset],
                   channel_display: Optional[Dict[str, Dict[str, Any]]] = None,
                   z_mode: str = "max",
                   z_index: int = 0,
                   m: int = 0, t: int = 0, z: int = 0,
                   stage_xy_um: Optional[List[Tuple[float, float]]] = None) -> None:
        """Wire up to a :class:`MaterializedDataset` for M/T scrolling.

        V1.41: ``volume`` is now an in-RAM dataset, not a lazy file
        handle. Frame reads are direct ndarray indexing — no cache,
        no IOWorker, no prefetcher.
        """
        self._cancel_pre_render_worker()
        self._hist_cache.clear()
        self._render_cache.clear()
        self._pixmap_cache.clear()
        self._cache_m = -1
        self._cache_ready = False

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
            self._cancel_pre_render_worker()
            self._render_cache.clear()
            self._pixmap_cache.clear()
            self._cache_m = -1
            self._cache_ready = False
            self._populate_chip_strip([], channel_display or {})
            self.lut_sidebar.rebuild([], channel_display or {})
            self.lut_sidebar.set_tile_layout(
                stage_xy_um=[], pixel_size_um=1.0,
                tile_h=0, tile_w=0, n_multipoints=0,
            )
            self._configure_gpu_canvas_channels([])
            return

        # Slider configuration.
        self._configure_slider(self.m_slider, self.m_label, self._m_total,
                                self._m_row,
                                volume.n_multipoints, self._m,
                                visible=volume.n_multipoints > 1,
                                play_btn=self._m_play, fps_spin=self._m_fps)
        self._configure_slider(self.t_slider, self.t_label, self._t_total,
                                self._t_row,
                                volume.n_timepoints, self._t, visible=True,
                                play_btn=self._t_play, fps_spin=self._t_fps)
        n_z = int(getattr(volume, "n_zslices", 1))
        z_visible = (z_mode == "none") and n_z > 1
        self._configure_slider(self.z_slider, self.z_label, self._z_total,
                                self._z_row,
                                n_z, max(0, min(z, n_z - 1)), visible=z_visible,
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

        self._hist_cache[(self._m, self._z_mode)] = True
        self._refresh()
        self._start_pre_render_worker()

    def set_channels(self,
                     channels: Dict[str, Any],
                     channel_display: Optional[Dict[str, Dict[str, Any]]] = None,
                     n_multipoints: int = 1,
                     m: int = 0) -> None:
        """Wire up to a flat dict of (T, H, W) channel arrays.

        Used on the Recipe page processed viewer. Pass ``n_multipoints`` and
        ``m`` to enable M-slider navigation when the file has multiple
        positions — the page drives actual data refresh on M changes via
        coords_changed.
        """
        self._hist_cache.clear()
        self._cancel_pre_render_worker()
        self._render_cache.clear()
        self._pixmap_cache.clear()
        self._cache_m = -1
        self._cache_ready = False

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
        sample_arr = np.asarray(sample)
        # Guard: a 2D (H, W) single-frame channel must be treated as T=1.
        # Without this, n = H and the T slider would have H positions where
        # each "frame" would be a 1-D row — nothing would render.
        if sample_arr.ndim == 2:
            n = 1
        else:
            n = sample_arr.shape[0]
        n_multipoints = max(1, int(n_multipoints))
        self._m = max(0, min(int(m), n_multipoints - 1))
        # Preserve the current T position when reloading the same or
        # compatible data (e.g. after M-slider change with same T count).
        # Reset to 0 only when the new T count is smaller.
        prev_t = self._t
        new_t = min(prev_t, max(0, n - 1))
        self._configure_slider(self.m_slider, self.m_label, self._m_total,
                                self._m_row,
                                n_multipoints, self._m,
                                visible=n_multipoints > 1,
                                play_btn=self._m_play, fps_spin=self._m_fps)
        self._configure_slider(self.z_slider, self.z_label, self._z_total,
                                self._z_row,
                                1, 0, visible=False,
                                play_btn=self._z_play, fps_spin=self._z_fps)
        self._configure_slider(self.t_slider, self.t_label, self._t_total,
                                self._t_row,
                                n, new_t, visible=True,
                                play_btn=self._t_play, fps_spin=self._t_fps)
        self._t = new_t
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

    def _configure_slider(self, slider: QSlider, label: QLineEdit,
                           total_label: QLabel, row,
                           total: int, value: int, visible: bool,
                           play_btn: Optional[QPushButton] = None,
                           fps_spin: Optional[QDoubleSpinBox] = None) -> None:
        slider.blockSignals(True)
        slider.setRange(0, max(0, total - 1))
        slider.setValue(min(value, max(0, total - 1)))
        slider.blockSignals(False)
        # Drive the visible tile strip (the QSlider stays a hidden model).
        strip = self._strip_for(slider)
        if strip is not None:
            strip.set_count(total)
            strip.set_current(min(value, max(0, total - 1)), emit=False)
            strip.clear_selection(emit=False)
            strip.setVisible(visible)
            axis = ("m" if slider is self.m_slider
                    else "t" if slider is self.t_slider else "z")
            self._axis_selection[axis] = frozenset()
        # First widget in the row is the axis letter label.
        prefix = row.itemAt(0).widget()
        if prefix is not None:
            prefix.setVisible(visible)
        label.setVisible(visible)
        total_label.setVisible(visible)
        if play_btn is not None:
            play_btn.setVisible(visible)
            if not visible and play_btn.isChecked():
                play_btn.setChecked(False)
        if fps_spin is not None:
            fps_spin.setVisible(visible)

    def set_m(self, m: int, *, emit: bool = True) -> None:
        """Jump to M position without triggering a data-fetch cascade.

        Used by the Recipe page to keep the processed and raw viewer M
        sliders in sync without recursively firing coords_changed on
        both sides.
        """
        n_m = (self._volume.n_multipoints if self._volume is not None
               else self.m_slider.maximum() + 1)
        m = max(0, min(int(m), n_m - 1))
        self.m_slider.blockSignals(not emit)
        self.m_slider.setValue(m)
        self.m_slider.blockSignals(False)
        self._m = m
        self._update_axis_labels()
        if emit:
            self.coords_changed.emit(self._m, self._t, self._z)

    def _populate_chip_strip(self,
                              names: List[str],
                              display: Dict[str, Dict[str, Any]]) -> None:
        for chip in self._chip_strip:
            chip.setParent(None)
            chip.deleteLater()
        self._chip_strip.clear()

        cycle = ["green", "red", "cyan", "magenta", "yellow", "blue", "orange"]
        _named_colors = {"gray", "green", "red", "blue", "cyan", "magenta",
                         "yellow", "orange", "white"}
        for i, name in enumerate(names):
            cd = display.get(name, {})
            saved = cd.get("color", "")
            if not saved:
                # Use the channel name as the color if it's a recognized color
                # (e.g. "red","green","blue" channels from an RGB overlay TIFF).
                saved = name.lower() if name.lower() in _named_colors else cycle[i % len(cycle)]
            chip = ChannelChip(
                name,
                color_default=saved,
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
    # V1.41: all data is in RAM so refresh is essentially free; the
    # debounce timers (50–80 ms) only coalesce 60+ Hz drag events into
    # one render per tick, which is what pyqtgraph wants anyway.
    def _on_m_changed(self, v: int) -> None:
        self._m = int(v)
        self._update_axis_labels()
        # Keep the sidebar tile widget in sync with slider drags.
        self.lut_sidebar.set_current_m(self._m)
        self.coords_changed.emit(self._m, self._t, self._z)
        self._m_debounce.start()

    def _do_m_refresh(self) -> None:
        """Timer-delayed M refresh: resample histogram only on first visit."""
        # Pixmap cache is keyed by T for a single M — clear it on M change.
        # The numpy render_cache retains all M positions (reusable if LUT unchanged).
        if self._m != self._cache_m:
            self._pixmap_cache.clear()
            self._pixmap_build_timer.stop()
            # If this M's frames are already in the render cache (pre-rendered
            # while another M was active), start QPixmap conversion for it.
            n_t = self._volume.n_timepoints if self._volume else 0
            if n_t > 0 and all((self._m, t) in self._render_cache for t in range(n_t)):
                self._cache_m = self._m
                self._cache_ready = True
                self._pixmap_build_t_idx = 0
                self._pixmap_build_timer.start()

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
        self._t_debounce.start()

    def _on_z_changed(self, v: int) -> None:
        self._z = int(v)
        # Render caches are keyed by (m, t) without Z — stale for a new Z position.
        self._render_cache.clear()
        self._pixmap_cache.clear()
        self._pp_cache.clear()
        self._pp_pixmap_cache.clear()
        self._cache_m = -1
        self._cache_ready = False
        self._update_axis_labels()
        self.coords_changed.emit(self._m, self._t, self._z)
        self._z_debounce.start()

    def _set_axis_playing(self, axis: str, playing: bool) -> None:
        timer = {"m": self._m_timer, "t": self._t_timer, "z": self._z_timer}[axis]
        fps_spin = {"m": self._m_fps, "t": self._t_fps, "z": self._z_fps}[axis]
        play_btn = {"m": self._m_play, "t": self._t_play, "z": self._z_play}[axis]
        if playing:
            # When T-axis cache is ready allow 60fps (16ms floor);
            # otherwise keep 50ms to avoid overloading the live compose path.
            if axis == "t" and self._cache_ready:
                min_interval = 16
            else:
                min_interval = 50
            interval = max(min_interval, int(1000 / fps_spin.value()))
            timer.start(interval)
        else:
            timer.stop()
        # The play/pause icon is driven by bind_toggle_icon on the button's
        # toggled signal (V1.44) — no text to swap here.

    def _on_strip_current(self, axis: str, value: int) -> None:
        """Direct fast-refresh when the user steps a strip (click/drag/arrows).

        Mirrors :meth:`_axis_tick`: update the backing slider with signals
        blocked (no debounce), sync coords, refresh immediately — so M/Z
        stepping is as smooth as FPS playback rather than waiting on the
        80/50 ms debounce timers.
        """
        slider = {"m": self.m_slider, "t": self.t_slider, "z": self.z_slider}[axis]
        value = max(0, min(int(value), slider.maximum()))
        slider.blockSignals(True)
        slider.setValue(value)
        slider.blockSignals(False)
        # The slider→strip sync is bypassed when signals are blocked, so move the
        # strip's highlighted (selected) tile explicitly — this lets programmatic
        # navigation (e.g. live-run frame streaming) visibly advance the strip.
        strip = self._strip_for(slider)
        if strip is not None:
            strip.set_current(value, emit=False)
        z_changed = (axis == "z" and value != self._z)
        if axis == "t":
            self._t = value
        elif axis == "m":
            self._m = value
            self.lut_sidebar.set_current_m(self._m)
        else:
            self._z = value
        self._last_frame_idx = value
        if z_changed:
            # Z caches are keyed by (m, t) without Z — stale for a new Z plane.
            self._render_cache.clear()
            self._pixmap_cache.clear()
            self._pp_cache.clear()
            self._pp_pixmap_cache.clear()
            self._cache_m = -1
            self._cache_ready = False
        self._update_axis_labels()
        self.coords_changed.emit(self._m, self._t, self._z)
        self._do_refresh()

    # ── V1.43 tile-strip helpers ──
    def _strip_for(self, slider: QSlider) -> Optional[FrameStrip]:
        if slider is self.m_slider:
            return self._m_strip
        if slider is self.t_slider:
            return self._t_strip
        if slider is self.z_slider:
            return self._z_strip
        return None

    def _next_playback_value(self, slider: QSlider) -> int:
        """Next frame for playback. When the axis has a tile selection, loop
        only through the selected frames; otherwise advance linearly."""
        axis = ("m" if slider is self.m_slider
                else "t" if slider is self.t_slider
                else "z")
        sel = sorted(self._axis_selection.get(axis, frozenset()))
        cur = slider.value()
        if len(sel) >= 2:
            # Step to the next selected index after the current one (wrap).
            for v in sel:
                if v > cur:
                    return v
            return sel[0]
        return (cur + 1) % (slider.maximum() + 1)

    def _on_strip_selection(self, axis: str, sel: frozenset) -> None:
        self._axis_selection[axis] = sel
        self.selection_changed.emit(axis, sel)

    def axis_selection(self, axis: str) -> frozenset:
        """Selected tile indices for ``axis`` ("m"/"t"/"z"); empty = none."""
        return self._axis_selection.get(axis, frozenset())

    def set_axis_selection(self, axis: str, indices) -> None:
        """Restore a tile selection for ``axis`` (does not emit selection_changed).

        Used to re-apply a selection after :meth:`set_volume` (which clears it as
        part of reconfiguring the strips), so the Pipelines preview can keep the
        user's multi-frame selection across a volume swap.
        """
        strip = {"m": self._m_strip, "t": self._t_strip,
                 "z": self._z_strip}.get(axis)
        if strip is None:
            return
        idx = [int(i) for i in indices]
        strip.set_selection(idx, emit=False)
        self._axis_selection[axis] = frozenset(idx)

    def _on_strip_crop(self, axis: str, sel: frozenset) -> None:
        self.crop_to_selection_requested.emit(axis, sel)

    def _frame_meta_text(self, axis: str, idx: int) -> str:
        """Per-axis metadata tooltip for tile ``idx`` (T/M/Z)."""
        vol = self._volume
        if axis == "t":
            lines = [f"T {idx + 1}"]
            ts = self._frame_timestamps
            if ts is not None and 0 <= idx < len(ts):
                t0 = float(ts[0])
                lines.append(f"Time imaged: {self._fmt_seconds(float(ts[idx]) - t0)}")
                if idx > 0:
                    lines.append(
                        f"Δ to previous: {float(ts[idx]) - float(ts[idx - 1]):.2f} s")
                lines.append(
                    f"Elapsed total: {self._fmt_seconds(float(ts[-1]) - t0)}")
            return "\n".join(lines)
        if axis == "m":
            px = getattr(vol, "pixel_size_um", 1.0) if vol is not None else 1.0
            h = getattr(vol, "height", 0) if vol is not None else 0
            w = getattr(vol, "width", 0) if vol is not None else 0
            lines = [f"M {idx + 1}", f"Pixel size: {px:.4g} µm/px",
                     f"Resolution: {w}×{h} px",
                     f"Field of view: {w * px:.1f}×{h * px:.1f} µm"]
            return "\n".join(lines)
        # Z
        step = getattr(vol, "z_step_um", 1.0) if vol is not None else 1.0
        n_z = getattr(vol, "n_zslices", 1) if vol is not None else 1
        total = max(0.0, step * (n_z - 1))
        height = step * idx
        lines = [f"Z {idx + 1}", f"Step: {step:.4g} µm",
                 f"Height: {height:.2f} µm / {total:.2f} µm range"]
        return "\n".join(lines)

    @staticmethod
    def _fmt_seconds(s: float) -> str:
        s = max(0.0, s)
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = s % 60
        if h > 0:
            return f"{h:d}:{m:02d}:{sec:05.2f}"
        return f"{m:02d}:{sec:05.2f}"

    @perf_log("viewer._axis_tick")
    def _axis_tick(self, slider: QSlider) -> None:
        if slider.maximum() <= 0:
            return
        next_val = self._next_playback_value(slider)
        self._last_frame_idx = next_val
        # Block valueChanged so the debounce timer is not started — the
        # playback timer is the clock and we call _do_refresh directly below.
        slider.blockSignals(True)
        slider.setValue(next_val)
        slider.blockSignals(False)
        strip = self._strip_for(slider)
        if strip is not None:
            strip.set_current(next_val, emit=False)
        # Manually sync the internal coordinate and axis label.
        if slider is self.t_slider:
            self._t = next_val
        elif slider is self.m_slider:
            self._m = next_val
        elif slider is self.z_slider:
            self._z = next_val
        self._update_axis_labels()
        self.coords_changed.emit(self._m, self._t, self._z)
        self._do_refresh()

    def _on_chip_state(self) -> None:
        # Keep the LUT sidebar swatches in sync when colors change.
        for chip in self._chip_strip:
            self.lut_sidebar.update_swatch(chip.name, chip.color_rgb)
        # Color or enable/disable change alters the composite — invalidate
        # any pre-rendered frames so they don't show stale colors.
        self._invalidate_render_cache()
        self._do_refresh()
        self.channels_changed.emit()

    def _on_lut_contrast_changed(self, _name: str, _lo: float,
                                  _hi: float, _gamma: float) -> None:
        # Fast path: flag the pre-render worker to stop writing stale frames,
        # evict only the current frame from the caches, then re-compose it
        # immediately from RAM. The debounce timer handles the full cache
        # rebuild + worker restart 250 ms after the user stops adjusting.
        if self._pre_render_worker is not None:
            self._pre_render_worker.cancel()
        key = (self._m, self._t)
        self._render_cache.pop(key, None)
        self._pixmap_cache.pop(self._t, None)
        self._pp_cache.clear()
        self._pp_pixmap_cache.clear()
        self._cache_m = -1
        self._cache_ready = False
        self._do_refresh()
        self._lut_rebuild_timer.start()
        self.channels_changed.emit()

    def _rebuild_render_cache_after_lut(self) -> None:
        """Cache rebuild triggered 250 ms after the last LUT change.

        V1.42 — uses ``lut_only=True`` so the rebuild reuses the
        existing cache slots in-place (the worker overwrites entries
        as it visits them) rather than emptying the dict first.  The
        user sees stale-LUT frames briefly during scrubbing instead
        of falling all the way back to the slow live-compose path
        while the worker warms up.
        """
        self._invalidate_render_cache(lut_only=True)

    # ── Pre-render cache management ──

    def _start_pre_render_worker(self) -> None:
        """Cancel any running pre-render and start a fresh one.

        Only runs on the CPU path with a loaded volume.  The GPU canvas
        handles compositing GPU-side; the recipe-page set_channels path
        has no M axis and is always fast enough for live compose.
        """
        self._cancel_pre_render_worker()
        if self._volume is None or self._use_gpu_canvas:
            return

        # Snapshot LUT and chip state on the main thread — the worker
        # runs on a background thread and cannot touch Qt objects.
        lut_snapshot: Dict[str, Tuple[float, float, float]] = {}
        chip_snapshot: Dict[str, Tuple[bool, Tuple[int, int, int]]] = {}
        for chip in self._chip_strip:
            chip_snapshot[chip.name] = (chip.enabled, chip.color_rgb)
            lut = self.lut_sidebar.lut_for(chip.name)
            if lut is not None:
                lut_snapshot[chip.name] = lut.get_contrast()

        self._render_cache.clear()
        self._pixmap_cache.clear()
        self._pp_cache.clear()
        self._pp_pixmap_cache.clear()
        self._cache_m = -1
        self._cache_ready = False

        worker = PreRenderWorker(
            volume=self._volume,
            lut_snapshot=lut_snapshot,
            chip_snapshot=chip_snapshot,
            priority_m=self._m,
            cache=self._render_cache,
            z_mode=self._z_mode,
            z_index=self._z,
        )
        worker.frame_cached.connect(self._on_frame_cached)
        worker.finished.connect(self._on_pre_render_finished)
        self._pre_render_worker = worker
        worker.start()

    def _cancel_pre_render_worker(self) -> None:
        """Stop any running pre-render worker and wait up to 200ms for exit."""
        if self._pre_render_worker is not None and \
                self._pre_render_worker.isRunning():
            self._pre_render_worker.cancel()
            self._pre_render_worker.wait(200)
        self._pre_render_worker = None
        self._pixmap_build_timer.stop()

    def _invalidate_render_cache(self, *, lut_only: bool = False) -> None:
        """Drop pre-render caches and restart the worker.

        Parameters
        ----------
        lut_only : bool
            Default ``False`` — full invalidation: every dict is emptied,
            ``_cache_m`` is reset, the QPixmap build timer is stopped.
            Used when channels change shape (enable / disable / color
            swap, z-mode change, dataset reload).

            ``True`` — V1.42 LUT-only fast path.  On the GPU canvas this
            is a no-op: ``GpuImageCanvas`` applies LUT/levels as a GPU
            uniform on every paint, so the stale cache is never read.
            On the CPU canvas the existing dicts are kept and the
            worker is restarted to overwrite entries as it visits them.
            The user sees brief stale-LUT frames during scrub instead
            of falling all the way through to the slow live-compose
            path while the worker warms up.  ``_pp_version`` is bumped
            so the overlay caches re-render on demand.
        """
        if lut_only:
            self._pp_version += 1
            if self._use_gpu_canvas:
                # GPU canvas re-applies LUT every paint — the CPU caches
                # are not being read by the render path, so leave them
                # untouched and skip the worker churn.
                return
            # CPU canvas — restart the worker but keep the existing cache
            # dicts. ``PreRenderWorker._render`` writes ``cache[(m, t)] =
            # composite`` which overwrites stale entries atomically.
            self._cancel_pre_render_worker()
            self._start_pre_render_worker()
            return

        self._render_cache.clear()
        self._pixmap_cache.clear()
        self._pp_cache.clear()
        self._pp_pixmap_cache.clear()
        self._cache_m = -1
        self._cache_ready = False
        self._pixmap_build_timer.stop()
        self._start_pre_render_worker()

    def _on_frame_cached(self, m: int, t: int) -> None:
        """Called via queued connection when the worker finishes one frame.

        Refreshes the display if the newly cached frame is the one currently
        being shown — snaps to the cached version without waiting for the
        full priority-M series to finish.
        """
        if m == self._m and t == self._t and not self._cache_ready:
            self._do_refresh()

    def _on_pre_render_finished(self) -> None:
        """Priority-M series is fully cached; activate the fast display path."""
        self._cache_ready = True
        # Tighten the T playback timer to 60fps now that frame display is
        # essentially free (pixmap swap vs. full numpy compose).
        if self._t_timer.isActive():
            fps = self._t_fps.value()
            self._t_timer.setInterval(max(16, int(1000 / fps)))
        # Schedule QPixmap pre-conversion for the current M.
        self._cache_m = self._m
        self._pixmap_build_t_idx = 0
        self._pixmap_build_timer.start()

    def _build_next_pixmap_batch(self, batch_size: int = 5) -> None:
        """Convert the next batch of numpy composites to QPixmap objects.

        Runs on the main thread in small increments so the event loop stays
        responsive between batches.  Each QPixmap.fromImage call costs ~1–3ms
        at 2048×2048; batching 5 = ~5–15ms per event-loop spin, imperceptible.
        """
        m = self._cache_m
        if m < 0 or self._volume is None:
            return
        n_t = self._volume.n_timepoints
        built = 0
        while self._pixmap_build_t_idx < n_t and built < batch_size:
            t = self._pixmap_build_t_idx
            key = (m, t)
            if key in self._render_cache and t not in self._pixmap_cache:
                rgb = self._render_cache[key]
                h, w = rgb.shape[0], rgb.shape[1]
                rgb_c = np.ascontiguousarray(rgb)
                qimg = QImage(rgb_c.data, w, h, w * 3, QImage.Format.Format_RGB888)
                # fromImage makes a deep copy so rgb_c can be released after.
                self._pixmap_cache[t] = QPixmap.fromImage(qimg)
            self._pixmap_build_t_idx += 1
            built += 1
        if self._pixmap_build_t_idx < n_t:
            self._pixmap_build_timer.start()  # schedule next batch

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
    @perf_log("viewer._do_refresh")
    def _do_refresh(self) -> None:
        # Track which branch handled the frame so perf logs can split
        # cache-hit / cache-miss rates without us having to instrument
        # each branch separately.
        self._last_frame_idx = self._t
        self._last_cache_hit = False

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
                    # GPU canvas bypasses the CPU cache by design — mark
                    # as a hit so playback timing isn't classified as
                    # the slow live-compose path.
                    self._last_cache_hit = True
                    return
                # If the GPU branch returns False (e.g. no enabled
                # channels with cached planes yet) we fall through to
                # the legacy path so the canvas still shows *some*
                # representation — typically a blank frame.
            except Exception:
                # GPU path raised — fall through to the safe CPU
                # composite path rather than show nothing.
                pass

        # QPixmap cache fast-path — zero numpy work, pure paintEvent swap.
        # Only active on the volume path (not recipe-page set_channels).
        if (self._volume is not None
                and self._frame_post_process is None
                and self._cache_m == self._m
                and self._t in self._pixmap_cache):
            self.canvas.set_pixmap_direct(self._pixmap_cache[self._t])
            self._last_cache_hit = True
            return

        # numpy composite cache fast-path — skip LUT+compose, still needs
        # QImage→QPixmap but that's ~1ms vs ~50ms for 2048×2048 live compose.
        key = (self._m, self._t)
        if (self._volume is not None
                and self._frame_post_process is None
                and key in self._render_cache):
            self.canvas.set_image(self._render_cache[key])
            self._last_cache_hit = True
            return

        # Post-process overlay fast-path: QPixmap already built for this
        # (m, t, overlay-version) — zero numpy work, pure paintEvent swap.
        pp_key = (self._m, self._t, self._pp_version)
        if self._frame_post_process is not None:
            if pp_key in self._pp_pixmap_cache:
                self.canvas.set_pixmap_direct(self._pp_pixmap_cache[pp_key])
                self._last_cache_hit = True
                return
            if pp_key in self._pp_cache:
                pm = self._numpy_to_pixmap(self._pp_cache[pp_key])
                self._pp_pixmap_cache[pp_key] = pm
                self.canvas.set_pixmap_direct(pm)
                self._last_cache_hit = True
                return

        # Render cache + post-process: skip LUT/channel compose, only pay
        # for the overlay callback (rasterize + blend, ~10–30 ms vs ~50 ms
        # for the full live compose on multi-channel 2K images).
        key = (self._m, self._t)
        if (self._volume is not None
                and self._frame_post_process is not None
                and key in self._render_cache):
            processed = self._frame_post_process(
                self._render_cache[key].copy(), self._t, self._m
            )
            self._pp_cache[pp_key] = processed
            pm = self._numpy_to_pixmap(processed)
            self._pp_pixmap_cache[pp_key] = pm
            self.canvas.set_pixmap_direct(pm)
            self._last_cache_hit = True  # render_cache hit, overlay reblend
            return

        # Live compose fallback (GPU off, no cache, or recipe-page path).
        # ``_compose_current_frame`` already applies the post-process overlay
        # internally (and fills the pp caches) when a hook is set, so the
        # returned composite is final — applying it again here would double-draw
        # the overlay on the first render of each frame.
        composite = self._compose_current_frame()
        if composite is None:
            return
        self.canvas.set_image(composite)
        # _last_cache_hit stays False — this is the slow path.

    # Backward-compat alias used by set_volume / set_channels / apply_channel_state.
    _refresh = _do_refresh

    # ── GPU-mode render (V1.36 Phase 4) ──
    def _render_current_frame_gpu(self) -> bool:
        """Per-channel GPU render path.

        V1.41: every plane comes from the in-RAM :class:`MaterializedDataset`,
        so there is no cache miss path, no IOWorker dispatch, no
        pyramid level swap. The whole function is "for each enabled
        channel, push its (H, W) plane to the GPU canvas and let
        pyqtgraph composite + LUT."
        """
        any_pushed = False

        if self._volume is not None:
            for c_idx, name in enumerate(self._volume.channel_names):
                chip = next(
                    (c for c in self._chip_strip if c.name == name), None)
                if chip is None:
                    self.canvas.set_channel_visible(c_idx, False)  # type: ignore[attr-defined]
                    continue

                # Channel visibility + color + levels — GPU-side per-tick.
                self.canvas.set_channel_visible(c_idx, chip.enabled)  # type: ignore[attr-defined]
                self.canvas.set_channel_color(c_idx, chip.color_rgb)  # type: ignore[attr-defined]
                lut = self.lut_sidebar.lut_for(name)
                if lut is not None:
                    lo, hi, _gamma = lut.get_contrast()
                    self.canvas.set_channel_levels(c_idx, (lo, hi))  # type: ignore[attr-defined]

                if not chip.enabled:
                    continue

                frame_2d = self._read_volume_plane(c_idx, name)
                if frame_2d is None:
                    continue
                self.canvas.update_channel(c_idx, frame_2d)  # type: ignore[attr-defined]
                any_pushed = True
        else:
            # Recipe-page flat-channel path — same as the volume branch
            # but channels are indexed by name from an in-RAM dict that
            # was already (T, H, W).
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
                    arr = np.asarray(data)
                    if arr.ndim == 2:
                        frame = arr
                    else:
                        t_idx = min(self._t, arr.shape[0] - 1)
                        frame = arr[t_idx]
                except Exception:
                    continue
                frame_2d = self._normalize_to_2d(frame)
                if frame_2d is None:
                    continue
                self.canvas.update_channel(c_idx, frame_2d)  # type: ignore[attr-defined]
                any_pushed = True

        return any_pushed

    # ── V1.42 pyramid plumbing ──
    def attach_pyramid(self, reader) -> None:
        """Adopt a :class:`PyramidReader` for zoom-aware reads.

        Called by :meth:`MainWindow._attach_pyramid_reader_to_viewers`
        after the pyramid build worker finishes.  When the user zooms
        out far enough that one screen pixel spans multiple source
        pixels, :meth:`_choose_pyramid_level` returns ``L > 0`` and the
        plane is read from the pyramid instead of the in-RAM level-0
        ndarray — much smaller transfer to the GPU canvas.
        """
        self._pyramid_reader = reader

    def _choose_pyramid_level(self) -> int:
        """Return the pyramid level appropriate for the current zoom.

        Returns 0 when no pyramid is attached, when the volume is
        absent (recipe-page path), or when the canvas doesn't expose
        a ``ViewBox``.  The actual level pick uses the canvas's
        viewport rect via :meth:`PyramidReader.pick_level_for_viewport`.
        """
        reader = self._pyramid_reader
        if reader is None or self._volume is None:
            return 0
        # Both canvases expose ``width()``; the GPU canvas additionally
        # has a pyqtgraph ViewBox for the source-pixel rect.
        try:
            viewport_screen_px = max(1, int(self.canvas.width()))
        except Exception:  # noqa: BLE001
            return 0
        # Image-pixels-visible: the ViewBox extent in source units.
        # On the legacy QLabel canvas we don't have a precise rect, so
        # use the full image width as a conservative upper bound (this
        # only over-picks level 0, which is the safe direction).
        image_pixels_visible = int(getattr(self._volume, "width", 0))
        view_box = getattr(self.canvas, "view_box", None)
        if view_box is not None:
            try:
                rect = view_box.viewRect()
                image_pixels_visible = max(1, int(round(rect.width())))
            except Exception:  # noqa: BLE001
                pass
        if image_pixels_visible <= 0:
            return 0
        try:
            return int(reader.pick_level_for_viewport(
                viewport_screen_px, image_pixels_visible,
            ))
        except Exception:  # noqa: BLE001
            return 0

    def _read_volume_plane(self, c_idx: int, name: str) -> Optional[np.ndarray]:
        """Fetch the current (m, t) plane for channel ``c_idx`` from RAM.

        Fast path: if the dataset is a MaterializedDataset we index the
        per-channel (M, T, Z, H, W) array directly and apply Z projection
        or slice selection in-place. Otherwise we fall back to the
        LazyND2Volume.get_frame interface for backwards compatibility.

        V1.42 — when a pyramid is attached and zoom-out chose a level
        ``>0``, read the plane from the pyramid instead of level 0.
        """
        volume = self._volume
        if volume is None:
            return None

        # V1.42 pyramid short-circuit: read from the coarsest level
        # whose pixels still beat screen pixels, transferring far less
        # data to the GPU canvas when zoomed out.
        level = self._choose_pyramid_level()
        if level > 0 and self._pyramid_reader is not None:
            try:
                return np.asarray(self._pyramid_reader.get_frame(
                    level, c_idx, self._m, self._t, self._z,
                    z_mode=self._z_mode,
                ))
            except Exception:  # noqa: BLE001
                # On any read failure, fall through to level 0.
                pass

        # Fast path — MaterializedDataset has a ``channels`` dict.
        channels = getattr(volume, "channels", None)
        if channels is not None:
            arr = channels.get(name)
            if arr is None:
                return None
            try:
                zstack = arr[self._m, self._t]  # (Z, H, W)
            except IndexError:
                return None
            n_z = zstack.shape[0]
            if n_z <= 1:
                return zstack[0]
            if self._z_mode == "none":
                zi = max(0, min(self._z, n_z - 1))
                return zstack[zi]
            if self._z_mode == "max":
                return zstack.max(axis=0)
            if self._z_mode == "min":
                return zstack.min(axis=0)
            # mean
            return zstack.mean(axis=0).astype(zstack.dtype)
        # Compatibility path — call the LazyND2Volume API.
        try:
            plane = volume.get_frame(
                c=c_idx, m=self._m, t=self._t,
                z=self._z, z_mode=self._z_mode,
            )
        except Exception:
            return None
        return self._normalize_to_2d(plane)

    @staticmethod
    def _numpy_to_pixmap(rgb: np.ndarray) -> QPixmap:
        """Convert a (H, W, 3) uint8 array to a QPixmap (deep copy, safe to cache)."""
        h, w = rgb.shape[0], rgb.shape[1]
        rgb_c = np.ascontiguousarray(rgb)
        qimg = QImage(rgb_c.data, w, h, w * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

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

    @perf_log("viewer._compose_current_frame")
    def _compose_current_frame(self) -> Optional[np.ndarray]:
        """CPU-side RGB composite, used when the GPU canvas is off or
        when a frame post-process hook needs a uint8 RGB array.

        V1.41: every plane is read directly from the in-RAM dataset
        (or the recipe-page channel dict). No cache, no IOWorker, no
        prefetcher — those layers paid for themselves only when the
        underlying reads were slow.
        """
        sources: List[Tuple[ChannelChip, np.ndarray]] = []
        if self._volume is not None:
            for c_idx, name in enumerate(self._volume.channel_names):
                chip = next((c for c in self._chip_strip if c.name == name), None)
                if chip is None or not chip.enabled:
                    continue
                plane = self._read_volume_plane(c_idx, name)
                if plane is None:
                    continue
                frame_2d = self._normalize_to_2d(plane)
                if frame_2d is None:
                    continue
                sources.append((chip, frame_2d))
        else:
            # Recipe-page in-RAM channels.
            for chip in self._chip_strip:
                if not chip.enabled:
                    continue
                data = self._channels.get(chip.name)
                if data is None:
                    continue
                try:
                    arr = np.asarray(data)
                    if arr.ndim == 2:
                        # Single-frame (H, W) channel — no T axis.
                        frame = arr
                    else:
                        t_idx = min(self._t, arr.shape[0] - 1)
                        frame = arr[t_idx]
                except Exception:
                    continue
                frame_2d = self._normalize_to_2d(frame)
                if frame_2d is None:
                    continue
                sources.append((chip, frame_2d))

        if not sources:
            return None

        sample = sources[0][1]
        h, w = sample.shape
        composite = np.zeros((h, w, 3), dtype=np.float32)
        for chip, frame in sources:
            lut = self.lut_sidebar.lut_for(chip.name)
            if lut is None:
                continue
            lo, hi, gamma = lut.get_contrast()
            mapped = apply_lut(frame, lo, hi, gamma).astype(np.float32)
            r, g, b = chip.color_rgb
            composite[..., 0] += mapped * (r / 255.0)
            composite[..., 1] += mapped * (g / 255.0)
            composite[..., 2] += mapped * (b / 255.0)

        result = np.clip(composite, 0, 255).astype(np.uint8)

        # Fill render cache on-demand so the post-process fast-path in
        # _do_refresh can use it on subsequent refreshes.  This also
        # benefits the GPU-canvas path, which skips PreRenderWorker but
        # still needs a cached base when an overlay is active.
        key = (self._m, self._t)
        if self._volume is not None and key not in self._render_cache:
            self._render_cache[key] = result

        if self._frame_post_process is not None:
            processed = self._frame_post_process(result.copy(), self._t, self._m)
            pp_key = (self._m, self._t, self._pp_version)
            self._pp_cache[pp_key] = processed
            self._pp_pixmap_cache[pp_key] = self._numpy_to_pixmap(processed)
            return processed

        return result

    def _update_axis_labels(self) -> None:
        if self._volume is not None:
            self.m_label.setText(str(self._m + 1))
            self._m_total.setText(f"/{self._volume.n_multipoints}")
            self.t_label.setText(str(self._t + 1))
            self._t_total.setText(f"/{self._volume.n_timepoints}")
            self.z_label.setText(str(self._z + 1))
            self._z_total.setText(f"/{getattr(self._volume, 'n_zslices', 1)}")
        else:
            n_t = self.t_slider.maximum() + 1 if self.t_slider.maximum() >= 0 else 0
            self.t_label.setText(str(self._t + 1))
            self._t_total.setText(f"/{n_t}")
            n_m = self.m_slider.maximum() + 1 if self.m_slider.maximum() >= 0 else 1
            self.m_label.setText(str(self._m + 1))
            self._m_total.setText(f"/{n_m}")
        # V1.44 — mirror the current frame's metadata into the pinned panel.
        sidebar = getattr(self, "lut_sidebar", None)
        if sidebar is not None and hasattr(sidebar, "set_metadata_text"):
            parts = [self._frame_meta_text("t", self._t),
                     self._frame_meta_text("m", self._m)]
            if self._volume is not None and getattr(self._volume, "n_zslices", 1) > 1:
                parts.append(self._frame_meta_text("z", self._z))
            sidebar.set_metadata_text("\n".join(parts))

    def _on_axis_label_edited(self, axis: str) -> None:
        """Parse a manually entered frame number and jump the slider."""
        label_map = {"m": self.m_label, "t": self.t_label, "z": self.z_label}
        slider_map = {"m": self.m_slider, "t": self.t_slider, "z": self.z_slider}
        label = label_map[axis]
        slider = slider_map[axis]
        try:
            frame = int(label.text().strip())
        except ValueError:
            self._update_axis_labels()
            return
        value = max(0, min(frame - 1, slider.maximum()))
        label.clearFocus()
        slider.setValue(value)

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
        """One-line summary of the in-RAM dataset footprint.

        V1.41: there is no cache anymore; the dataset itself is the
        cache. We report how many bytes the materialized channels are
        holding — which is the relevant number for the user.
        """
        if self._volume is None:
            return "Volume: not loaded"
        nbytes_fn = getattr(self._volume, "nbytes", None)
        if callable(nbytes_fn):
            mb = nbytes_fn() / (1024 ** 2)
        else:
            mb = 0.0
        z_mode = getattr(self._volume, "z_mode", "?")
        return f"In-RAM dataset: {mb:,.0f} MB · z_mode={z_mode}"

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

    def swap_channels_for_m(self, channels: Dict[str, Any], m: int) -> None:
        """Swap displayed channels to pre-computed data for M position ``m``.

        Unlike :meth:`set_channels`, this does NOT stop play timers, reset
        the T position, or rebuild the chip/LUT strip.  For recipe-page
        instant M navigation only — caller must ensure ``channels`` is the
        correct pre-computed result for this M.
        """
        self._channels = dict(channels)
        self._m = m
        self._update_axis_labels()
        self._do_refresh()

    def set_t_playing(self, playing: bool, fps: Optional[float] = None) -> None:
        """Start or stop T-axis playback; optionally set FPS first."""
        if fps is not None:
            self._t_fps.setValue(fps)
        if self._t_play.isChecked() != playing:
            self._t_play.setChecked(playing)
