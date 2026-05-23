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

from nd2studios.backend.materialized_dataset import MaterializedDataset
from nd2studios.core.settings import Settings
from nd2studios.widgets.image_viewer import (
    CHANNEL_COLORS, ImageCanvas, ZoomToolbar,
)
from nd2studios.widgets.lut_histogram import apply_lut
from nd2studios.widgets.lut_sidebar import LutSidebar


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

        self._m: int = 0
        self._t: int = 0
        self._z: int = 0

        self._m_timer = QTimer(self)
        self._t_timer = QTimer(self)
        self._z_timer = QTimer(self)

        # Debounce timers coalesce fast drag events (60 Hz+) into ~50 ms
        # refresh ticks. With in-RAM data the refresh itself is cheap;
        # the debounce mainly avoids redundant LUT recomputation and
        # pyqtgraph texture uploads.
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

        self._hist_cache: dict = {}  # (m, z_mode) -> True
        self._frame_post_process: Optional[Callable] = None

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
        # V1.41: Z is collapsed at load time; the Z slider is always
        # hidden. The user re-loads to switch Z mode.
        self._configure_slider(self.z_slider, self.z_label, self._z_row,
                                1, 0, visible=False,
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

    def set_channels(self,
                     channels: Dict[str, Any],
                     channel_display: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Wire up to a flat dict of (T, H, W) channel arrays.

        Used on the Recipe page where M is fixed and Z has been
        collapsed already.
        """
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
        self._update_axis_labels()
        self.coords_changed.emit(self._m, self._t, self._z)
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
                    frame = np.asarray(data[self._t])
                except Exception:
                    continue
                frame_2d = self._normalize_to_2d(frame)
                if frame_2d is None:
                    continue
                self.canvas.update_channel(c_idx, frame_2d)  # type: ignore[attr-defined]
                any_pushed = True

        return any_pushed

    def _read_volume_plane(self, c_idx: int, name: str) -> Optional[np.ndarray]:
        """Fetch the current (m, t) plane for channel ``c_idx`` from RAM.

        Fast path: if the dataset is a MaterializedDataset we index the
        per-channel (M, T, H, W) array directly — zero copy. Otherwise
        we fall back to the LazyND2Volume.get_frame interface for
        backwards compatibility with any callers still wiring a lazy
        volume in.
        """
        volume = self._volume
        if volume is None:
            return None
        # Fast path — MaterializedDataset has a ``channels`` dict.
        channels = getattr(volume, "channels", None)
        if channels is not None:
            arr = channels.get(name)
            if arr is None:
                return None
            try:
                return arr[self._m, self._t]
            except IndexError:
                return None
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
                    frame = np.asarray(data[self._t])
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
        if self._frame_post_process is not None:
            result = self._frame_post_process(result, self._t, self._m)
        return result

    def _update_axis_labels(self) -> None:
        if self._volume is not None:
            self.m_label.setText(f"{self._m + 1}/{self._volume.n_multipoints}")
            self.t_label.setText(f"{self._t + 1}/{self._volume.n_timepoints}")
            # V1.41: Z is collapsed at load; the label is informational only.
            self.z_label.setText(
                f"{self._z + 1}/{getattr(self._volume, 'n_zslices', 1)}"
            )
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
