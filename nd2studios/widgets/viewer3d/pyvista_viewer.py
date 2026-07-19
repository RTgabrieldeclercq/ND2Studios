"""``PyVista3DViewer`` — off-screen volumetric 3-D viewer (V1.65 / V1.66).

Renders the raw ``(Z,H,W)`` image volume for the current ``(channel, M, T)`` in
one of four modes (volume / MIP / ortho slices / isosurface), with M/T/Z
navigation, time playback, and per-channel color/LUT from
``record.channel_display``. It can also overlay DVC fields and PTV tracks.

**Why off-screen.** ND2Studios' main window is frameless + ``WA_TranslucentBackground``
with a drop-shadow effect. A native OpenGL window (an embedded
``pyvistaqt.QtInteractor``) cannot composite into such a window — it punches a
"hole" straight through to the desktop and steals mouse input. So we render VTK
**off-screen** (``pyvista.Plotter(off_screen=True)``) and paint the result into a
plain ``QLabel``, which composites correctly. Orbit / zoom are driven by mouse
events that re-render off-screen.

Design constraints honored: no module-level PyVista/VTK import (the class imports
even when the extra is absent → :class:`Missing3DDeps` placeholder); volumes are
built off-thread (``VolumeBuildWorker`` → numpy) and downsampled to a GPU-safe
size; every VTK call happens on the GUI thread. Public API mirrors
``MultiAxisViewer`` so it is a drop-in alternate behind a 2D/3D toggle.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QHBoxLayout, QLabel, QPushButton, QSlider,
    QSpinBox, QVBoxLayout, QWidget,
)

from nd2studios.core.settings import Settings
from nd2studios.widgets.icon_button import icon_button, scale_qss, scaled
from nd2studios.widgets.viewer3d.deps import PYVISTA_AVAILABLE
from nd2studios.widgets.viewer3d.placeholder import Missing3DDeps
from nd2studios.workers.volume3d_worker import (
    TimeSeriesBuildWorker, VolumeBuildResult, VolumeBuildWorker,
)

MODE_VOLUME = "volume"
MODE_MIP = "mip"
MODE_SLICES = "slices"
MODE_ISO = "isosurface"
RENDER_MODES = (MODE_VOLUME, MODE_MIP, MODE_SLICES, MODE_ISO)
_MODE_LABELS = {MODE_VOLUME: "Volume", MODE_MIP: "MIP",
                MODE_SLICES: "Slices", MODE_ISO: "Iso"}

# V1.68 — surface scalars that are **signed** (a divergent colormap centred at 0
# reads outward vs inward, per Stout et al. Fig 4C); everything else (magnitudes,
# q-factor) uses a sequential map.
_DIVERGENT_SURFACE_SCALARS = {
    "u_perp", "u_x", "u_y", "u_z",
    "e_xx", "e_yy", "e_zz", "e_xy", "e_xz", "e_yz",
}


class _Canvas(QLabel):
    """Displays the off-screen render; forwards mouse orbit / wheel zoom."""

    def __init__(self, viewer: "PyVista3DViewer"):
        super().__init__()
        self._viewer = viewer
        self._last = None
        self.setMinimumSize(scaled(240), scaled(240))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"background: {Settings.BG_PRIMARY};")
        self.setText("")

    def _pt(self, event) -> Tuple[float, float]:
        pos = event.position() if hasattr(event, "position") else event.pos()
        return float(pos.x()), float(pos.y())

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._last = self._pt(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._last is None:
            return
        x, y = self._pt(event)
        self._viewer._orbit(x - self._last[0], y - self._last[1])
        self._last = (x, y)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._last = None
        self._viewer._re_render()

    def wheelEvent(self, event) -> None:  # noqa: N802
        d = event.angleDelta().y()
        self._viewer._zoom(1.15 if d > 0 else 1.0 / 1.15)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._viewer._on_canvas_resized(self.width(), self.height())


class PyVista3DViewer(QWidget):
    """Off-screen volumetric 3-D viewer with a control bar and image canvas."""

    coords_changed = Signal(int, int, int)   # (m, t, z)
    channels_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._available = PYVISTA_AVAILABLE

        self._volume: Any = None
        self._channel_display: Dict[str, Dict[str, Any]] = {}
        self._channel_names: List[str] = []
        self._m = 0
        self._t = 0
        self._z = 0
        self._z_start = 0
        self._z_end: Optional[int] = None
        self._n_multipoints = 1
        self._n_timepoints = 1
        self._n_zslices = 1
        self._mode = MODE_VOLUME
        self._master_opacity = 0.5
        self._overlay: Any = None
        self._overlay_scalar: Optional[str] = None
        self._timestamps = None

        self._plotter = None
        self._have_scene = False
        self._busy = False
        self._last_img = None
        self._build_worker: Optional[VolumeBuildWorker] = None
        self._pending_build: Optional[Tuple] = None
        # Smooth 3-D time playback: prebuild every timepoint's downsampled
        # volume once, then play from RAM (no per-frame disk reads).
        self._prebuilt: Dict[int, VolumeBuildResult] = {}
        self._prebuild_sig: Optional[Tuple] = None
        self._prebuild_worker: Optional[TimeSeriesBuildWorker] = None
        self._playing = False
        self._cache: Dict[Tuple, VolumeBuildResult] = {}
        self._cache_order: List[Tuple] = []
        self._cache_limit = 8
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_playback)

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(scaled(2))

        self._controls = self._build_controls()
        root.addWidget(self._controls)

        if self._available:
            self._canvas = _Canvas(self)
            root.addWidget(self._canvas, stretch=1)
        else:
            self._canvas = None
            root.addWidget(Missing3DDeps(self), stretch=1)
            self._set_controls_enabled(False)

        self._status = QLabel("")
        self._status.setStyleSheet(scale_qss(
            f"color: {Settings.FG_SECONDARY}; font: 8pt;"))
        root.addWidget(self._status)

    def _build_controls(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("pipelineControlBar")
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(scaled(6), scaled(4), scaled(6), scaled(4))
        hl.setSpacing(scaled(6))

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_buttons: Dict[str, QPushButton] = {}
        for mode in RENDER_MODES:
            btn = QPushButton(_MODE_LABELS[mode])
            btn.setObjectName("toggleBtn")
            btn.setCheckable(True)
            btn.setChecked(mode == self._mode)
            btn.clicked.connect(lambda _c=False, m=mode: self.set_render_mode(m))
            self._mode_group.addButton(btn)
            self._mode_buttons[mode] = btn
            hl.addWidget(btn)

        hl.addSpacing(scaled(8))

        self._m_label = QLabel("M")
        self._m_spin = QSpinBox()
        self._m_spin.setFixedWidth(scaled(58))
        self._m_spin.valueChanged.connect(self._on_m_changed)
        hl.addWidget(self._m_label)
        hl.addWidget(self._m_spin)

        self._t_play = icon_button("fa5s.play", "Play — animate over time (T)",
                                   object_name="playBtn", checkable=True,
                                   button_px=28, icon_px=12)
        self._t_play.toggled.connect(self._on_play_toggled)
        self._t_slider = QSlider(Qt.Orientation.Horizontal)
        self._t_slider.setMinimum(0)
        self._t_slider.valueChanged.connect(self._on_t_changed)
        self._t_label = QLabel("T 0")
        hl.addWidget(self._t_play)
        hl.addWidget(self._t_slider, stretch=1)
        hl.addWidget(self._t_label)

        hl.addSpacing(scaled(8))
        hl.addWidget(QLabel("Z"))
        self._z0_spin = QSpinBox()
        self._z0_spin.setFixedWidth(scaled(58))
        self._z0_spin.valueChanged.connect(self._on_zrange_changed)
        self._z1_spin = QSpinBox()
        self._z1_spin.setFixedWidth(scaled(58))
        self._z1_spin.valueChanged.connect(self._on_zrange_changed)
        hl.addWidget(self._z0_spin)
        hl.addWidget(QLabel("–"))
        hl.addWidget(self._z1_spin)

        hl.addSpacing(scaled(8))
        hl.addWidget(QLabel("Opacity"))
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setMinimum(0)
        self._opacity_slider.setMaximum(100)
        self._opacity_slider.setValue(int(self._master_opacity * 100))
        self._opacity_slider.setFixedWidth(scaled(90))
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        hl.addWidget(self._opacity_slider)

        self._chan_row = QHBoxLayout()
        self._chan_row.setSpacing(scaled(6))
        self._chan_checks: Dict[str, QCheckBox] = {}

        hl.addSpacing(scaled(8))
        # Home — reset the 3-D camera to the fit + isometric "home" view. (Was a
        # fa5s.expand "Reset camera" button; relabelled to a Home affordance in V1.76
        # so the 3-D object view has a Home button matching the 2-D field views.)
        self._btn_reset = icon_button("fa5s.home",
                                      "Home — reset the 3-D view (fit + isometric)",
                                      object_name="pipelineToolBtn",
                                      button_px=28, icon_px=13)
        self._btn_reset.clicked.connect(self.reset_camera)
        self._btn_shot = icon_button("fa5s.camera", "Save a screenshot (PNG)",
                                     object_name="pipelineToolBtn",
                                     button_px=28, icon_px=13)
        self._btn_shot.clicked.connect(self._on_screenshot)
        hl.addWidget(self._btn_reset)
        hl.addWidget(self._btn_shot)

        outer = QWidget()
        ov = QVBoxLayout(outer)
        ov.setContentsMargins(0, 0, 0, 0)
        ov.setSpacing(scaled(2))
        ov.addWidget(bar)
        chan_holder = QWidget()
        chan_holder.setLayout(self._chan_row)
        ov.addWidget(chan_holder)
        return outer

    def _ensure_plotter(self):
        """Lazily create the OFF-SCREEN PyVista plotter (main thread only)."""
        if self._plotter is not None:
            return self._plotter
        os.environ.setdefault("QT_API", "pyside6")
        import pyvista as pv

        try:
            pv.set_plot_theme("dark")
        except Exception:  # noqa: BLE001
            pass
        w = max(320, self._canvas.width() if self._canvas else 800)
        h = max(240, self._canvas.height() if self._canvas else 600)
        plotter = pv.Plotter(off_screen=True, window_size=[int(w), int(h)])
        try:
            plotter.set_background(Settings.BG_PRIMARY)
        except Exception:  # noqa: BLE001
            pass
        self._plotter = plotter
        return plotter

    # ── public API (mirrors MultiAxisViewer) ───────────────────────────────
    def set_volume(self, volume: Any, *,
                   channel_display: Optional[Dict[str, Dict[str, Any]]] = None,
                   z_mode: str = "max", z_index: int = 0,
                   m: int = 0, t: int = 0, z: int = 0,
                   stage_xy_um: Optional[list] = None) -> None:
        self._volume = volume
        self._channel_display = dict(channel_display or {})
        self._channel_names = list(getattr(volume, "channel_names", []) or [])
        self._n_multipoints = int(getattr(volume, "n_multipoints", 1) or 1)
        self._n_timepoints = int(getattr(volume, "n_timepoints", 1) or 1)
        self._n_zslices = int(getattr(volume, "n_zslices", 1) or 1)
        self._m = max(0, min(int(m), self._n_multipoints - 1))
        self._t = max(0, min(int(t), self._n_timepoints - 1))
        self._z = max(0, min(int(z), self._n_zslices - 1))
        self._z_start = 0
        self._z_end = self._n_zslices
        self._cache.clear()
        self._cache_order.clear()
        self._sync_controls_to_state()
        self._rebuild()

    def set_channels(self, channels: Dict[str, Any], *,
                     channel_display: Optional[Dict[str, Dict[str, Any]]] = None,
                     n_multipoints: int = 1, m: int = 0) -> None:
        self._volume = None
        self._channel_display = dict(channel_display or {})
        self._have_scene = False
        self._set_status("Loaded data has no Z axis — 3-D view needs a raw volume.")
        if self._canvas is not None:
            self._canvas.setText("No Z axis to render in 3-D.")

    def set_frame_timestamps(self, timestamps: Any) -> None:
        self._timestamps = timestamps

    def coords(self) -> Tuple[int, int, int]:
        return (self._m, self._t, self._z)

    def set_current_frame(self, m: Optional[int] = None,
                          t: Optional[int] = None) -> None:
        if m is not None:
            self._m = max(0, min(int(m), self._n_multipoints - 1))
        if t is not None:
            self._t = max(0, min(int(t), self._n_timepoints - 1))
        self._sync_controls_to_state()
        self._rebuild()

    def channel_state(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._channel_display)

    def apply_channel_state(self, state: Dict[str, Dict[str, Any]]) -> None:
        self._channel_display = dict(state or {})
        self._populate_channel_row()
        self._cache.clear()
        self._cache_order.clear()
        self._rebuild()

    def refresh(self) -> None:
        self._rebuild()

    def reset_camera(self) -> None:
        if not self._available or self._plotter is None:
            return
        try:
            self._plotter.reset_camera()
            self._plotter.view_isometric()
            self._re_render()
        except Exception:  # noqa: BLE001
            pass

    # ── 3-D specific ────────────────────────────────────────────────────────
    def set_render_mode(self, mode: str) -> None:
        if mode not in RENDER_MODES:
            return
        self._mode = mode
        btn = self._mode_buttons.get(mode)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)
        self._rerender_current()

    def set_z_range(self, z_start: int, z_end: int) -> None:
        self._z_start = max(0, int(z_start))
        self._z_end = max(self._z_start + 1, int(z_end))
        self._cache.clear()
        self._cache_order.clear()
        self._rebuild()

    def set_master_opacity(self, alpha: float) -> None:
        self._master_opacity = max(0.0, min(1.0, float(alpha)))
        self._rerender_current()

    def set_overlay(self, overlay: Any, scalar: Optional[str] = None) -> None:
        self._overlay = overlay
        self._overlay_scalar = scalar
        self._rerender_current()

    def _rerender_current(self) -> None:
        """Redraw with the current volume + overlay, or the overlay alone when no
        raw image volume is loaded (the DVC-object / result 'View in 3D' path).

        When there is no volume and the overlay was cleared (``set_overlay(None)``),
        still route through :meth:`_render_overlay_only` if a scene exists so the
        stale object is cleared — the overlay-add is then a no-op, leaving a blank
        (axes-only) scene rather than the previous render frozen on screen."""
        if self._volume is not None:
            self._render_from_cache_or_build()
        elif self._overlay is not None or self._have_scene:
            self._render_overlay_only()

    def _render_overlay_only(self) -> None:
        """Render a scene containing only the current overlay (no channel volume).

        Used when the viewer is driven purely by ``set_overlay`` (e.g. the DVC
        panel's '3D Object' subtab), so there is no ``VolumeBuildResult`` to feed
        ``_render_scene``."""
        if not self._available:
            return
        try:
            p = self._ensure_plotter()
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"3-D backend init failed: {type(exc).__name__}")
            return
        try:
            p.clear()
        except Exception:  # noqa: BLE001
            pass
        if self._overlay is not None:
            try:
                self._add_overlay_actor()
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"Overlay render failed: {exc}")
        try:
            self._add_context_actor()          # V1.68 surrounding-channel cloud
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Context render failed: {exc}")
        for fn in (p.add_axes, lambda: p.add_bounding_box(color="gray")):
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass
        try:
            p.reset_camera()
            p.view_isometric()
        except Exception:  # noqa: BLE001
            pass
        self._have_scene = True
        self._re_render()

    def screenshot(self, path: str) -> None:
        if self._available and self._plotter is not None:
            try:
                self._plotter.screenshot(path)
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"Screenshot failed: {exc}")

    # ── build pipeline ──────────────────────────────────────────────────────
    def _cache_key(self) -> Tuple:
        lut_sig = tuple(sorted(
            (n, c.get("enabled", True), c.get("color", ""),
             c.get("lut_lo", None), c.get("lut_hi", None), c.get("lut_gamma", 1.0))
            for n, c in self._channel_display.items()
        ))
        return (self._m, self._t, self._z_start, self._z_end, lut_sig)

    def _rebuild(self) -> None:
        if not self._available or self._volume is None:
            return
        self._render_from_cache_or_build()

    def _render_from_cache_or_build(self) -> None:
        if not self._available or self._volume is None:
            return
        cached = self._cache.get(self._cache_key())
        if cached is not None:
            self._render_scene(cached)
            return
        self._request_build()

    def _request_build(self) -> None:
        # Coalesce rapid requests (e.g. T playback): remember the LATEST request
        # and run exactly ONE build worker at a time. Spawning a worker per tick
        # while the previous still reads gigabytes piled up QThreads and crashed
        # ("QThread: Destroyed while thread is still running").
        self._pending_build = (self._m, self._t, self._z_start, self._z_end,
                               dict(self._channel_display))
        if self._build_worker is not None:
            try:
                if self._build_worker.isRunning():
                    return   # the current build's completion will pick this up
            except RuntimeError:
                pass
        self._launch_pending_build()

    def _launch_pending_build(self) -> None:
        if self._pending_build is None or self._volume is None:
            return
        m, t, z0, z1, cd = self._pending_build
        self._pending_build = None
        self._set_status("Building 3-D volume…")
        worker = VolumeBuildWorker(self._volume, cd, m, t, z_start=z0, z_end=z1)
        worker.finished.connect(self._on_volumes_built)
        worker.error.connect(self._on_build_error)
        worker.status.connect(self._set_status)
        self._build_worker = worker
        worker.start()

    def _on_volumes_built(self, result: Optional[VolumeBuildResult]) -> None:
        if result is not None and (result.m, result.t) == (self._m, self._t):
            key = (result.m, result.t, self._z_start, self._z_end,
                   self._cache_key()[-1])
            self._cache[key] = result
            self._cache_order.append(key)
            while len(self._cache_order) > self._cache_limit:
                old = self._cache_order.pop(0)
                self._cache.pop(old, None)
            self._render_scene(result)
        # Run the next coalesced request (latest T during playback), if any.
        if self._pending_build is not None:
            self._launch_pending_build()

    def _on_build_error(self, msg: str) -> None:
        self._set_status(f"3-D build error: {msg.splitlines()[0] if msg else '?'}")
        if self._pending_build is not None:
            self._launch_pending_build()

    # ── rendering (GUI thread, off-screen) ──────────────────────────────────
    def _render_scene(self, result: VolumeBuildResult,
                      reset_view: bool = True) -> None:
        if not self._available:
            return
        try:
            p = self._ensure_plotter()
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"3-D backend init failed: {type(exc).__name__}")
            return

        try:
            p.clear()
        except Exception:  # noqa: BLE001
            pass

        n = 0
        try:
            for cv in result.channels:
                self._add_channel_actor(cv, result.spacing)
                n += 1
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Render failed ({self._mode}): {exc}")

        if self._overlay is not None:
            try:
                self._add_overlay_actor()
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"Overlay render failed: {exc}")

        try:
            p.add_axes()
        except Exception:  # noqa: BLE001
            pass
        try:
            p.add_bounding_box(color="gray")
        except Exception:  # noqa: BLE001
            pass
        if reset_view:
            try:
                p.reset_camera()
                p.view_isometric()
            except Exception:  # noqa: BLE001
                pass

        self._have_scene = True
        self._re_render()

        if not n:
            self._set_status("No enabled channels to render in 3-D.")
        elif self._mode in (MODE_VOLUME, MODE_MIP):
            z, h, w = result.channels[0].data.shape
            self._set_status(
                f"3-D {self._mode}: {z}×{h}×{w} (downsampled), {n} channel(s) "
                "— drag to rotate, wheel to zoom.")
        else:
            self._set_status(
                f"3-D {self._mode}: {n} channel(s) — drag to rotate.")

    def _re_render(self) -> None:
        """Re-screenshot the current off-screen scene into the canvas label."""
        if not self._available or self._plotter is None or self._canvas is None:
            return
        if self._busy or not self._have_scene:
            return
        self._busy = True
        try:
            import numpy as np
            w = max(160, self._canvas.width())
            h = max(160, self._canvas.height())
            try:
                self._plotter.window_size = [int(w), int(h)]
            except Exception:  # noqa: BLE001
                pass
            img = self._plotter.screenshot(return_img=True)
            arr = np.ascontiguousarray(np.asarray(img)[:, :, :3], dtype=np.uint8)
            self._last_img = arr  # keep alive for QImage
            ih, iw = arr.shape[:2]
            qimg = QImage(arr.data, iw, ih, 3 * iw, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self._canvas.setPixmap(pix.scaled(
                self._canvas.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"3-D display failed: {type(exc).__name__}: {exc}")
        finally:
            self._busy = False

    def _make_grid(self, arr, spacing):
        import numpy as np
        import pyvista as pv

        arr_xyz = np.ascontiguousarray(arr.transpose(2, 1, 0))  # (nx, ny, nz)
        try:
            grid = pv.ImageData()
        except AttributeError:  # pyvista < 0.44
            grid = pv.UniformGrid()
        grid.dimensions = arr_xyz.shape
        grid.spacing = (float(spacing.dx), float(spacing.dy), float(spacing.dz))
        grid.point_data["intensity"] = arr_xyz.flatten(order="F")
        return grid

    def _channel_cmap(self, color):
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap.from_list(
            "ch", [(0.0, 0.0, 0.0), (color[0], color[1], color[2])])

    def _add_channel_actor(self, cv, spacing) -> None:
        p = self._plotter
        grid = self._make_grid(cv.data, spacing)
        cmap = self._channel_cmap(cv.color)
        alpha = self._master_opacity

        if self._mode in (MODE_VOLUME, MODE_MIP):
            blending = "maximum" if self._mode == MODE_MIP else "additive"
            p.add_volume(grid, scalars="intensity", cmap=cmap,
                         opacity=[0.0, alpha], blending=blending,
                         show_scalar_bar=False)
        elif self._mode == MODE_SLICES:
            slices = grid.slice_orthogonal()
            p.add_mesh(slices, scalars="intensity", cmap=cmap,
                       opacity=1.0, show_scalar_bar=False)
        elif self._mode == MODE_ISO:
            import numpy as np
            vals = cv.data[cv.data > 0]
            level = float(np.percentile(vals, 60)) if vals.size else 1.0
            surf = grid.contour([level], scalars="intensity")
            if surf.n_points > 0:
                p.add_mesh(surf, color=cv.color, opacity=max(0.2, alpha),
                           show_scalar_bar=False)

    def _add_overlay_actor(self) -> None:
        from nd2studios.backend.viz3d.overlays import (
            DVCField, GranuleScene, MaskedField, PtvTracks, SurfaceField,
        )
        if isinstance(self._overlay, SurfaceField):
            self._add_surface_field_overlay(self._overlay)
        elif isinstance(self._overlay, MaskedField):
            self._add_masked_field_overlay(self._overlay)
        elif isinstance(self._overlay, DVCField):
            self._add_dvc_overlay(self._overlay)
        elif isinstance(self._overlay, PtvTracks):
            self._add_ptv_overlay(self._overlay)
        elif isinstance(self._overlay, GranuleScene):
            self._add_granule_scene_overlay(self._overlay)

    # ── Granule node scenes (V1.73) ──────────────────────────────────────────
    def _binary_surface(self, mask_zhw, spacing, origin):
        """Marching-cubes surface of a ``(Z,H,W)`` bool mask as a ``pv.PolyData`` in
        world ``(x,y,z)`` µm (``ImageData`` spacing = ``(dx,dy,dz)``). ``None`` on
        an empty mask / contour failure. Pads by 1 so edge-touching objects close."""
        import numpy as np
        import pyvista as pv
        m = np.asarray(mask_zhw)
        if m.ndim == 2:
            m = m[None, ...]
        if not m.any():
            return None
        m = np.pad(m.astype(np.uint8), 1)     # close surfaces at the volume border
        dz, dy, dx = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
        ox, oy, oz = (float(origin[0]), float(origin[1]), float(origin[2]))
        m_xyz = np.ascontiguousarray(m.transpose(2, 1, 0))   # (nx, ny, nz)
        try:
            grid = pv.ImageData()
        except AttributeError:               # pyvista < 0.44
            grid = pv.UniformGrid()
        grid.dimensions = m_xyz.shape
        grid.spacing = (dx, dy, dz)
        grid.origin = (ox - dx, oy - dy, oz - dz)   # undo the 1-voxel pad offset
        grid["m"] = m_xyz.flatten(order="F").astype(float)
        try:
            surf = grid.contour([0.5], scalars="m")
        except Exception:  # noqa: BLE001
            return None
        return surf if surf.n_points > 0 else None

    def _add_label_isosurfaces(self, labels_zhw, spacing, origin, *,
                               style="surface", line_width=2.0,
                               opacity=0.92) -> bool:
        """One colored isosurface per label id (>0), via :func:`granule_color`."""
        import numpy as np
        from nd2studios.backend.analysis.granule_types import granule_color
        lv = np.asarray(labels_zhw)
        if lv.ndim == 2:
            lv = lv[None, ...]
        drew = False
        for gid in (int(v) for v in np.unique(lv) if int(v) > 0):
            surf = self._binary_surface(lv == gid, spacing, origin)
            if surf is None:
                continue
            col = tuple(c / 255.0 for c in granule_color(gid))
            kw = dict(color=col, show_scalar_bar=False)
            if style == "wireframe":
                kw.update(style="wireframe", line_width=line_width)
            else:
                kw.update(opacity=opacity, smooth_shading=True)
            try:
                self._plotter.add_mesh(surf, **kw)
                drew = True
            except Exception:  # noqa: BLE001
                continue
        return drew

    def _add_dotted_boundary(self, labels_zhw, spacing, origin) -> bool:
        """Per-label feature-edge wireframe rendered dotted (VTK line stipple; falls
        back to faint gray when the VTK build lacks stippling)."""
        import numpy as np
        lv = np.asarray(labels_zhw)
        if lv.ndim == 2:
            lv = lv[None, ...]
        drew = False
        for gid in (int(v) for v in np.unique(lv) if int(v) > 0):
            surf = self._binary_surface(lv == gid, spacing, origin)
            if surf is None:
                continue
            try:
                edges = surf.extract_feature_edges()
                actor = self._plotter.add_mesh(edges, color=(0.6, 0.6, 0.6),
                                               line_width=1.5, show_scalar_bar=False)
            except Exception:  # noqa: BLE001
                continue
            try:
                prop = actor.GetProperty()
                if hasattr(prop, "SetLineStipplePattern"):
                    prop.SetLineStipplePattern(0xF0F0)
                    prop.SetLineStippleRepeatFactor(1)
                else:                       # no stipple in this VTK — fade instead
                    prop.SetOpacity(0.5)
            except Exception:  # noqa: BLE001
                pass
            drew = True
        return drew

    def _add_granule_scene_overlay(self, scene) -> None:
        """Render a :class:`GranuleScene` (V1.71): point crosshairs / per-granule
        colored dots, inter-centroid tessellation edges, assembled label
        isosurfaces (mask node), and new-solid + previous-dotted boundary
        wireframes (boundary node). Geometry is already world ``(x,y,z)`` µm."""
        import numpy as np
        import pyvista as pv
        from nd2studios.backend.analysis.granule_types import granule_color

        p = self._plotter
        drew = False

        pts = np.asarray(getattr(scene, "points_um", np.zeros((0, 3))))
        mode = str(getattr(scene, "draw_points_as", "none"))
        if pts.ndim == 2 and pts.shape[0] and mode != "none":
            span = float(np.ptp(pts, axis=0).max()) if pts.shape[0] > 1 else 1.0
            gsize = max(span * 0.02, 1e-3)
            labels = getattr(scene, "point_labels", None)
            if mode == "crosshair":
                col = tuple(c / 255.0 for c in getattr(scene, "point_color",
                                                       (255, 255, 0)))
                cross = pv.PolyData(
                    np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0],
                              [0, 0, -1], [0, 0, 1]], float),
                    lines=np.array([2, 0, 1, 2, 2, 3, 2, 4, 5]))
                try:
                    glyphs = pv.PolyData(pts).glyph(geom=cross, scale=False,
                                                    orient=False, factor=gsize)
                    p.add_mesh(glyphs, color=col, line_width=2.0,
                               show_scalar_bar=False)
                except Exception:  # noqa: BLE001 — fall back to spheres
                    p.add_points(pts, color=col, render_points_as_spheres=True,
                                 point_size=8.0)
                drew = True
            elif labels is not None and np.asarray(labels).size == pts.shape[0]:
                labels = np.asarray(labels).astype(int)
                for gid in sorted(set(int(v) for v in labels)):
                    sub = pts[labels == gid]
                    if not sub.shape[0]:
                        continue
                    col = tuple(c / 255.0 for c in granule_color(gid))
                    p.add_points(sub, color=col, render_points_as_spheres=True,
                                 point_size=10.0)
                drew = True
            else:
                p.add_points(pts, render_points_as_spheres=True, point_size=10.0)
                drew = True

        edges = getattr(scene, "edges", None)
        if edges is not None and pts.shape[0] and np.asarray(edges).size:
            e = np.asarray(edges).reshape(-1, 2)
            e = e[(e[:, 0] < pts.shape[0]) & (e[:, 1] < pts.shape[0])]
            if e.shape[0]:
                lines = np.hstack([np.full((e.shape[0], 1), 2, int), e]).ravel()
                try:
                    net = pv.PolyData(pts, lines=lines)
                    p.add_mesh(net, color=(0.0, 1.0, 1.0), line_width=1.5,
                               show_scalar_bar=False)
                    drew = True
                except Exception:  # noqa: BLE001
                    pass

        lv = getattr(scene, "label_volume", None)
        if lv is not None:
            drew = self._add_label_isosurfaces(
                lv, scene.spacing, scene.origin_um, style="surface") or drew

        newb = getattr(scene, "boundary_label_volume", None)
        if newb is not None:
            drew = self._add_label_isosurfaces(
                newb, scene.spacing, scene.origin_um, style="wireframe",
                line_width=2.5) or drew
        prevb = getattr(scene, "prev_label_volume", None)
        if prevb is not None:
            drew = self._add_dotted_boundary(
                prevb, scene.spacing, scene.origin_um) or drew

        if not drew:
            self._set_status("Granule view: nothing to show — Run the node first.")

    def _add_masked_field_overlay(self, mf) -> None:
        """Render a DVC field interpolated onto a 3-D object mask (Phase 2).

        The object's **boundary surface** (mask iso-contour) is colored by the
        selected scalar; the render **mode** picks how the **interior** is shown:
        Iso → opaque surface only; Slices → translucent shell + orthogonal interior
        slices; Volume/MIP → translucent shell + a volume render of the field
        inside the object. So "surface boundaries + inside the object" is one view.
        """
        import numpy as np
        import pyvista as pv

        p = self._plotter
        if getattr(mf, "is_empty", False):
            self._set_status("No object mask — draw a 3D mask (Mask node), then Run.")
            return
        name = self._overlay_scalar or mf.default_scalar
        scal = mf.scalars.get(name)
        if scal is None and mf.scalars:
            name, scal = next(iter(mf.scalars.items()))

        mask = np.ascontiguousarray(mf.mask.astype(np.uint8))
        dz, dy, dx = (float(mf.spacing[0]), float(mf.spacing[1]), float(mf.spacing[2]))
        ox, oy, oz = (float(mf.origin_um[0]), float(mf.origin_um[1]),
                      float(mf.origin_um[2]))
        mask_xyz = np.ascontiguousarray(mask.transpose(2, 1, 0))   # (nx, ny, nz)
        try:
            grid = pv.ImageData()
        except AttributeError:      # pyvista < 0.44
            grid = pv.UniformGrid()
        grid.dimensions = mask_xyz.shape
        grid.spacing = (dx, dy, dz)
        grid.origin = (ox, oy, oz)
        grid["mask"] = mask_xyz.flatten(order="F").astype(float)

        clim = None
        if scal is not None:
            finite = scal[np.isfinite(scal)]
            if finite.size:
                lo = float(np.percentile(finite, 2))
                hi = float(np.percentile(finite, 98))
                clim = (lo, hi if hi > lo else lo + 1e-6)
            fill = clim[0] if clim else 0.0
            s_xyz = np.ascontiguousarray(np.nan_to_num(scal, nan=fill).transpose(2, 1, 0))
            grid[name] = s_xyz.flatten(order="F")

        # Boundary surface (mask iso-contour), colored by the scalar it carries.
        surf = None
        try:
            surf = grid.contour([0.5], scalars="mask")
        except Exception:  # noqa: BLE001
            surf = None
        surf_opaque = self._mode == MODE_ISO
        surf_op = 1.0 if surf_opaque else max(0.12, 0.35 * self._master_opacity + 0.1)
        if surf is not None and surf.n_points > 0:
            if scal is not None:
                p.add_mesh(surf, scalars=name, cmap="viridis", clim=clim,
                           opacity=surf_op, smooth_shading=True,
                           show_scalar_bar=surf_opaque)
            else:
                p.add_mesh(surf, color="lightgray", opacity=surf_op)

        # Interior.
        if self._mode == MODE_SLICES and scal is not None:
            try:
                obj = grid.threshold(0.5, scalars="mask")
                p.add_mesh(obj.slice_orthogonal(), scalars=name, cmap="viridis",
                           clim=clim, show_scalar_bar=True)
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"Interior slices failed: {exc}")
        elif self._mode in (MODE_VOLUME, MODE_MIP) and scal is not None:
            # Field outside the object is already at clim-low (fill) → transparent
            # under the low-end-suppressing opacity ramp; the shell frames it.
            blending = "maximum" if self._mode == MODE_MIP else "composite"
            try:
                p.add_volume(grid, scalars=name, cmap="viridis", clim=clim,
                             opacity=[0.0, 0.0, 0.3, 0.6, 0.85], blending=blending,
                             show_scalar_bar=True)
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"Interior volume failed: {exc}")

        if surf is None or surf.n_points == 0:
            self._set_status("Object surface empty (mask has no boundary here).")
        else:
            self._set_status(
                f"3-D object · {name} · {mf.n_object_voxels} voxels · "
                f"mode={self._mode} — drag to rotate, wheel to zoom.")

    def _add_surface_field_overlay(self, sf) -> None:
        """Render a DVC field on a **smoothed closed surface mesh** (V1.68 — the
        primary DVC-on-object view; Stout et al. 2016).

        The surface (built by ``backend.viz3d.surface`` — marching cubes + Taubin)
        is drawn directly as ``pv.PolyData`` (no VTK contouring here) and coloured
        by the selected per-vertex scalar: ``u⊥`` and signed strains use a
        divergent map centred at 0 (outward vs inward, Fig 4C); magnitudes use a
        sequential map. The render **mode** styles the optional interior
        (``sf.interior`` — a voxel :class:`MaskedField`): Iso = opaque surface only;
        Slices = translucent shell + interior slices; Volume/MIP = shell + volume.
        """
        import numpy as np
        import pyvista as pv

        p = self._plotter
        if getattr(sf, "is_empty", False):
            self._set_status(
                "No object surface — draw a 3D mask (Mask node), then Run.")
            return
        name = self._overlay_scalar or sf.default_scalar
        scal = sf.scalars.get(name)
        if scal is None and sf.scalars:
            name, scal = next(iter(sf.scalars.items()))

        verts = np.ascontiguousarray(np.asarray(sf.vertices_um, dtype=float))
        faces = np.asarray(sf.faces, dtype=np.int64)
        if verts.shape[0] == 0 or faces.shape[0] == 0:
            self._set_status("Object surface empty (mask has no boundary here).")
            return
        face_arr = np.empty((faces.shape[0], 4), dtype=np.int64)
        face_arr[:, 0] = 3
        face_arr[:, 1:] = faces
        mesh = pv.PolyData(verts, face_arr.ravel())

        divergent = name in _DIVERGENT_SURFACE_SCALARS
        cmap = "coolwarm" if divergent else "viridis"
        clim = None
        if scal is not None:
            arr = np.asarray(scal, dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                if divergent:
                    a = float(np.percentile(np.abs(finite), 98)) or 1e-6
                    clim = (-a, a)
                else:
                    lo = float(np.percentile(finite, 2))
                    hi = float(np.percentile(finite, 98))
                    clim = (lo, hi if hi > lo else lo + 1e-6)
            mesh[name] = np.nan_to_num(arr, nan=(0.0 if divergent else
                                                 (clim[0] if clim else 0.0)))

        surf_opaque = self._mode == MODE_ISO
        surf_op = 1.0 if surf_opaque else max(0.15, 0.4 * self._master_opacity + 0.1)
        try:
            if scal is not None:
                p.add_mesh(mesh, scalars=name, cmap=cmap, clim=clim,
                           opacity=surf_op, smooth_shading=True,
                           show_scalar_bar=True)
            else:
                p.add_mesh(mesh, color="lightgray", opacity=surf_op,
                           smooth_shading=True)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Surface render failed: {exc}")
            return

        interior = getattr(sf, "interior", None)
        if interior is not None and self._mode in (MODE_SLICES, MODE_VOLUME,
                                                    MODE_MIP):
            self._add_masked_interior(interior, name, clim)

        mdm = getattr(sf, "mdm", None)
        jinfo = f" · ⟨J⟩={mdm.J:.3f}" if mdm is not None else ""
        self._set_status(
            f"3-D object surface · {name} · {sf.n_vertices} verts{jinfo} · "
            f"mode={self._mode} — drag to rotate, wheel to zoom.")

    def _add_masked_interior(self, mf, name: str, clim) -> None:
        """Render only the **interior** of a :class:`MaskedField` (no extra shell).

        Used by :meth:`_add_surface_field_overlay` for the Slices / Volume / MIP
        interior under the coloured surface mesh (the mesh already supplies the
        boundary), so the two do not draw competing shells.
        """
        import numpy as np
        import pyvista as pv

        p = self._plotter
        scal = mf.scalars.get(name)
        if scal is None and mf.scalars:
            _n, scal = next(iter(mf.scalars.items()))
        if scal is None or getattr(mf, "is_empty", False):
            return
        mask = np.ascontiguousarray(mf.mask.astype(np.uint8))
        dz, dy, dx = (float(mf.spacing[0]), float(mf.spacing[1]), float(mf.spacing[2]))
        ox, oy, oz = (float(mf.origin_um[0]), float(mf.origin_um[1]),
                      float(mf.origin_um[2]))
        try:
            grid = pv.ImageData()
        except AttributeError:      # pyvista < 0.44
            grid = pv.UniformGrid()
        grid.dimensions = np.ascontiguousarray(mask.transpose(2, 1, 0)).shape
        grid.spacing = (dx, dy, dz)
        grid.origin = (ox, oy, oz)
        grid["mask"] = mask.transpose(2, 1, 0).flatten(order="F").astype(float)
        fill = clim[0] if clim else 0.0
        s_xyz = np.ascontiguousarray(np.nan_to_num(scal, nan=fill).transpose(2, 1, 0))
        grid[name] = s_xyz.flatten(order="F")
        try:
            if self._mode == MODE_SLICES:
                obj = grid.threshold(0.5, scalars="mask")
                p.add_mesh(obj.slice_orthogonal(), scalars=name, cmap="viridis",
                           clim=clim, show_scalar_bar=False)
            else:
                blending = "maximum" if self._mode == MODE_MIP else "composite"
                p.add_volume(grid, scalars=name, cmap="viridis", clim=clim,
                             opacity=[0.0, 0.0, 0.3, 0.6, 0.85], blending=blending,
                             show_scalar_bar=False)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Interior render failed: {exc}")

    def set_context_channel(self, volume: Any,
                            voxel_size_um: Optional[Tuple[float, float, float]] = None,
                            *, mode: str = "none", color=(0.6, 0.6, 0.6),
                            opacity: float = 0.25,
                            iso_percentile: float = 70.0) -> None:
        """Set a **surrounding-channel context** cloud around the object (V1.68).

        Renders a *different* raw channel (``(Z, H, W)``) as a translucent
        ``volume`` / ``mip`` cloud or an ``iso``-surface shell, composited with the
        coloured object surface in one scene (the "surrounding factors" overlay
        the user asked for). ``mode="none"`` (or ``volume=None``) clears it.
        ``voxel_size_um`` ``(dz, dy, dx)`` places it in the object's world µm frame.
        """
        import numpy as np
        arr = None if (volume is None or mode == "none") else np.asarray(volume)
        self._context_volume = arr
        self._context_voxel = (tuple(float(v) for v in voxel_size_um)
                               if voxel_size_um else (1.0, 1.0, 1.0))
        self._context_mode = str(mode)
        self._context_color = color
        self._context_opacity = max(0.0, min(1.0, float(opacity)))
        self._context_iso_pct = float(iso_percentile)
        self._rerender_current()

    def _add_context_actor(self) -> None:
        """Composite the surrounding-channel context volume (if any) into the scene."""
        vol = getattr(self, "_context_volume", None)
        mode = getattr(self, "_context_mode", "none")
        if vol is None or mode == "none" or not self._available:
            return
        import numpy as np
        import pyvista as pv

        p = self._plotter
        arr = np.asarray(vol, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        # Stride to a render budget so a full channel volume stays interactive.
        budget = 8_000_000
        stride = 1
        while (arr[::stride, ::stride, ::stride].size > budget) and stride < 16:
            stride += 1
        arr = np.ascontiguousarray(arr[::stride, ::stride, ::stride])
        dz, dy, dx = (float(v) * stride for v in self._context_voxel)
        color = self._context_color
        op = getattr(self, "_context_opacity", 0.25)
        try:
            grid = pv.ImageData()
        except AttributeError:
            grid = pv.UniformGrid()
        arr_xyz = np.ascontiguousarray(arr.transpose(2, 1, 0))
        grid.dimensions = arr_xyz.shape
        grid.spacing = (dx, dy, dz)
        grid.origin = (0.0, 0.0, 0.0)
        grid["ctx"] = arr_xyz.flatten(order="F")
        try:
            cmap = self._channel_cmap(color)
            if mode in ("volume", "mip"):
                blending = "maximum" if mode == "mip" else "additive"
                p.add_volume(grid, scalars="ctx", cmap=cmap, opacity=[0.0, op],
                             blending=blending, show_scalar_bar=False)
            elif mode == "iso":
                vals = arr[arr > 0]
                level = (float(np.percentile(vals, self._context_iso_pct))
                         if vals.size else 1.0)
                surf = grid.contour([level], scalars="ctx")
                if surf.n_points > 0:
                    p.add_mesh(surf, color=color, opacity=max(0.12, op),
                               show_scalar_bar=False)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Context channel render failed: {exc}")

    def _add_dvc_overlay(self, field) -> None:
        import numpy as np
        import pyvista as pv

        p = self._plotter
        if field.n_points == 0:
            return
        scalar_name = self._overlay_scalar or field.default_scalar
        scal = field.scalars.get(scalar_name)
        if scal is None and field.scalars:
            scalar_name, scal = next(iter(field.scalars.items()))
        cloud = pv.PolyData(np.ascontiguousarray(field.points_um, dtype=float))
        cloud["displacement"] = np.ascontiguousarray(field.vectors_um, dtype=float)
        if scal is not None:
            cloud[scalar_name] = np.ascontiguousarray(scal, dtype=float)
        mags = np.linalg.norm(field.vectors_um, axis=1)
        finite = mags[np.isfinite(mags)]
        ref = float(np.nanmax(finite)) if finite.size else 1.0
        factor = 1.0 if ref <= 0 else (
            0.05 * float(np.linalg.norm(np.ptp(field.points_um, axis=0))) / ref)
        try:
            glyphs = cloud.glyph(orient="displacement", scale=False, factor=factor)
            p.add_mesh(glyphs, scalars=scalar_name if scal is not None else None,
                       cmap="viridis", show_scalar_bar=True)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"DVC glyphs failed: {exc}")

    def _add_ptv_overlay(self, tracks) -> None:
        import numpy as np
        import pyvista as pv

        p = self._plotter
        if tracks.n_segments == 0:
            return
        for pts, scal in zip(tracks.segments, tracks.seg_scalars):
            if pts.shape[0] < 2:
                continue
            poly = pv.lines_from_points(np.ascontiguousarray(pts, dtype=float))
            poly["scalar"] = np.ascontiguousarray(scal, dtype=float)
            try:
                tube = poly.tube(radius=max(1e-6, 0.004 * _extent(pts)))
                p.add_mesh(tube, scalars="scalar", cmap="plasma",
                           show_scalar_bar=False)
            except Exception:  # noqa: BLE001
                p.add_mesh(poly, scalars="scalar", cmap="plasma",
                           show_scalar_bar=False)

    # ── camera interaction (off-screen orbit / zoom) ─────────────────────────
    def _orbit(self, dx: float, dy: float) -> None:
        if not self._have_scene or self._plotter is None:
            return
        try:
            cam = self._plotter.camera
            cam.Azimuth(-dx * 0.4)
            cam.Elevation(dy * 0.4)
            cam.OrthogonalizeViewUp()
            self._re_render()
        except Exception:  # noqa: BLE001
            pass

    def _zoom(self, factor: float) -> None:
        if not self._have_scene or self._plotter is None:
            return
        try:
            self._plotter.camera.Zoom(float(factor))
            self._re_render()
        except Exception:  # noqa: BLE001
            pass

    def _on_canvas_resized(self, w: int, h: int) -> None:
        if self._have_scene:
            self._re_render()

    # ── playback ─────────────────────────────────────────────────────────────
    def _on_play_toggled(self, playing: bool) -> None:
        if not playing:
            self._playing = False
            self._play_timer.stop()
            return
        if self._volume is None or self._n_timepoints <= 1:
            self._t_play.setChecked(False)
            return
        self._playing = True
        sig = self._playback_sig()
        if (self._prebuild_sig == sig
                and len(self._prebuilt) >= self._n_timepoints):
            self._set_status("Playing (cached)…")
            self._play_timer.start(60)     # frames served from RAM
        else:
            self._start_prebuild(sig)

    def _playback_sig(self) -> Tuple:
        # Prebuilt volumes are raw voxel data — independent of render MODE (mode
        # only changes how they're drawn). Invalidate only on M / Z-range / LUT.
        return (self._m, self._z_start, self._z_end, self._cache_key()[-1])

    def _start_prebuild(self, sig: Tuple) -> None:
        """Prebuild every timepoint's downsampled volume in the background so
        playback can run from RAM. Reads the series once (with progress)."""
        if self._prebuild_worker is not None:
            try:
                if self._prebuild_worker.isRunning():
                    self._prebuild_worker.cancel()
            except RuntimeError:
                pass
        self._prebuilt = {}
        self._prebuild_sig = sig
        n_ch = max(1, sum(1 for c in self._channel_display.values()
                          if c.get("enabled", True))
                   or len(self._channel_names) or 1)
        # Keep the whole prebuilt series under ~2 GB of RAM (shrink per-frame
        # voxels when there are many timepoints).
        budget = max(2_000_000, (2 * 1024 ** 3) // max(1, self._n_timepoints * n_ch))
        self._set_status("Preparing smooth playback (reading timepoints)…")
        worker = TimeSeriesBuildWorker(
            self._volume, self._channel_display, self._m,
            list(range(self._n_timepoints)),
            z_start=self._z_start, z_end=self._z_end,
            max_voxels_per_frame=budget)
        worker.frame_ready.connect(self._on_prebuilt_frame)
        worker.status.connect(self._set_status)
        worker.finished.connect(self._on_prebuild_done)
        self._prebuild_worker = worker
        worker.start()

    def _on_prebuilt_frame(self, t: int, result: object) -> None:
        if isinstance(result, VolumeBuildResult):
            self._prebuilt[int(t)] = result
            # Show the first frame right away for feedback (before the timer runs).
            if int(t) == self._t and not self._play_timer.isActive():
                self._render_scene(result)

    def _on_prebuild_done(self) -> None:
        if not self._playing:
            return
        if self._prebuilt:
            self._set_status(f"Playing {len(self._prebuilt)} cached frames…")
            self._play_timer.start(60)
        else:
            self._playing = False
            self._set_status("Playback preparation cancelled.")

    def _advance_playback(self) -> None:
        # Smooth playback: advance T and render the PREBUILT frame from RAM,
        # keeping the camera fixed (no disk reads, no camera snap).
        if self._n_timepoints <= 1 or not self._prebuilt:
            return
        self._t = (self._t + 1) % self._n_timepoints
        self._t_slider.blockSignals(True)
        self._t_slider.setValue(self._t)
        self._t_slider.blockSignals(False)
        self._t_label.setText(f"T {self._t}")
        result = self._prebuilt.get(self._t)
        if result is not None:
            self._render_scene(result, reset_view=False)

    # ── control callbacks ─────────────────────────────────────────────────────
    def _on_m_changed(self, value: int) -> None:
        self._m = max(0, min(int(value), self._n_multipoints - 1))
        self.coords_changed.emit(self._m, self._t, self._z)
        self._render_from_cache_or_build()

    def _on_t_changed(self, value: int) -> None:
        self._t = max(0, min(int(value), self._n_timepoints - 1))
        self._t_label.setText(f"T {self._t}")
        self.coords_changed.emit(self._m, self._t, self._z)
        self._render_from_cache_or_build()

    def _on_zrange_changed(self, _value: int) -> None:
        z0 = int(self._z0_spin.value())
        z1 = int(self._z1_spin.value())
        if z1 <= z0:
            z1 = z0 + 1
        self._z_start, self._z_end = z0, z1
        self._cache.clear()
        self._cache_order.clear()
        self._rebuild()

    def _on_opacity_changed(self, value: int) -> None:
        self.set_master_opacity(value / 100.0)

    def _on_screenshot(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save 3-D screenshot", "view3d.png", "PNG image (*.png)")
        if path:
            self.screenshot(path)

    # ── helpers ───────────────────────────────────────────────────────────────
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._available:
            return
        if self._have_scene:
            self._re_render()
        elif self._volume is not None:
            self._render_from_cache_or_build()

    def _sync_controls_to_state(self) -> None:
        self._m_spin.blockSignals(True)
        self._m_spin.setMaximum(max(0, self._n_multipoints - 1))
        self._m_spin.setValue(self._m)
        self._m_spin.blockSignals(False)
        self._m_label.setVisible(self._n_multipoints > 1)
        self._m_spin.setVisible(self._n_multipoints > 1)

        self._t_slider.blockSignals(True)
        self._t_slider.setMaximum(max(0, self._n_timepoints - 1))
        self._t_slider.setValue(self._t)
        self._t_slider.blockSignals(False)
        self._t_label.setText(f"T {self._t}")
        has_t = self._n_timepoints > 1
        self._t_slider.setVisible(has_t)
        self._t_play.setVisible(has_t)
        self._t_label.setVisible(has_t)

        for spin, default in ((self._z0_spin, 0),
                              (self._z1_spin, self._n_zslices)):
            spin.blockSignals(True)
            spin.setMaximum(self._n_zslices)
            spin.setValue(default)
            spin.blockSignals(False)

        self._populate_channel_row()

    def _populate_channel_row(self) -> None:
        while self._chan_row.count():
            item = self._chan_row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._chan_checks.clear()
        for name in self._channel_names:
            conf = self._channel_display.get(name, {})
            cb = QCheckBox(name)
            cb.setChecked(bool(conf.get("enabled", True)))
            cb.toggled.connect(lambda checked, n=name: self._on_channel_toggled(n, checked))
            self._chan_checks[name] = cb
            self._chan_row.addWidget(cb)
        self._chan_row.addStretch(1)

    def _on_channel_toggled(self, name: str, checked: bool) -> None:
        conf = dict(self._channel_display.get(name, {}))
        conf["enabled"] = bool(checked)
        self._channel_display[name] = conf
        self._cache.clear()
        self._cache_order.clear()
        self.channels_changed.emit()
        self._rebuild()

    def _set_controls_enabled(self, enabled: bool) -> None:
        for w in (getattr(self, "_mode_buttons", {}) or {}).values():
            w.setEnabled(enabled)

    def _set_status(self, msg: str) -> None:
        if hasattr(self, "_status"):
            self._status.setText(msg or "")

    def on_close(self) -> None:
        self._playing = False
        self._play_timer.stop()
        self._pending_build = None
        # Wait for both workers to actually stop before teardown, else Qt warns
        # /crashes ("QThread: Destroyed while thread is still running").
        for w in (self._build_worker, self._prebuild_worker):
            if w is not None:
                try:
                    w.cancel()
                    w.wait(4000)
                except RuntimeError:
                    pass
        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:  # noqa: BLE001
                pass


def _extent(pts) -> float:
    import numpy as np
    if pts.shape[0] == 0:
        return 1.0
    span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    return span if span > 0 else 1.0
