"""In-viewer SerialTrack PTV panel.

``SerialTrackPanel`` is the "SerialTrack" tab in the Pipelines image viewer stack
(sibling of :class:`~nd2studios.widgets.spatial_maps_panel.SpatialMapsPanel`). It
turns tracked measurement rows into Particle-Tracking-Velocimetry plots using the
Qt-free data layer :mod:`nd2studios.backend.serialtrack_analysis`:

* **Trajectories** — particle paths up to the current frame, colored by time or
  cumulative displacement, with XY / XZ / YZ projection for 3D data.
* **Scalar field** — a gridded map of any displacement / velocity / strain /
  stress quantity, rendered as heatmap, filled contour, or line contour, with an
  optional background image and scale bar.
* **Quiver** — the gridded displacement/velocity vector field (magnitude-colored),
  standalone or overlaid on the scalar heatmap.
* **Histogram** — per-frame displacement-magnitude distribution.
* **Dashboard** — detected / tracked counts and tracking ratio across all frames.

The panel is fed via :meth:`set_data`; a ``FrameStrip`` scrubs frames and a
per-frame ``FieldBundle`` cache keeps playback smooth. See
``CodeLog/ClaudesPlan/V1.47_serialtrack_ptv_plots.md``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QPushButton, QSizePolicy,
)

from nd2studios.core.settings import Settings
from nd2studios.widgets.common import MplCanvas
from nd2studios.widgets.frame_strip import FrameStrip
from nd2studios.widgets.icon_button import scale_qss, scaled
from nd2studios.widgets.image_viewer import frame_to_uint8
from nd2studios.widgets.scale_bar import draw_scale_bar
from nd2studios.backend import serialtrack_analysis as sta

CMAPS_SEQ = ["viridis", "plasma", "inferno", "magma", "hot", "turbo"]
CMAPS_DIV = ["coolwarm", "bwr", "seismic", "PiYG", "PRGn", "RdBu_r"]
ALL_CMAPS = CMAPS_SEQ + CMAPS_DIV

# View modes shown in the "View" combo.
_VIEWS = [
    ("trajectories", "Trajectories"),
    ("field", "Scalar field"),
    ("quiver", "Quiver"),
    ("heatmap_quiver", "Heatmap + Quiver"),
    ("histogram", "Displacement histogram"),
    ("dashboard", "Tracking dashboard"),
]

# Scalar-field keys → (label, is-divergent-scale). Divergent fields default to a
# symmetric color scale about zero with a diverging colormap.
_FIELDS = [
    ("disp_mag", "Displacement |u|", False),
    ("u_x", "Displacement uₓ", True),
    ("u_y", "Displacement u_y", True),
    ("u_z", "Displacement u_z", True),
    ("v_x", "Velocity vₓ", True),
    ("v_y", "Velocity v_y", True),
    ("e_xx", "Strain εₓₓ", True),
    ("e_yy", "Strain ε_yy", True),
    ("e_xy", "Strain εₓ_y", True),
    ("eff_strain", "Effective strain", False),
    ("div", "Divergence (dilatation)", True),
    ("curl", "Curl (vorticity)", True),
    ("detF", "Jacobian det(F)", False),
    ("von_mises", "Von Mises stress", False),
    ("s_xx", "Stress σₓₓ", True),
    ("s_yy", "Stress σ_yy", True),
    ("s_xy", "Stress σₓ_y", True),
]
_DIVERGENT = {k for k, _l, d in _FIELDS if d}
_STRESS_KEYS = {"von_mises", "s_xx", "s_yy", "s_xy", "s_yz", "s_xz", "s_zz"}


class SerialTrackPanel(QWidget):
    """Embeddable SerialTrack PTV analysis view."""

    m_change_requested = Signal(int)   # user picked a different multipoint
    status_message = Signal(str)       # surfaces hints to the page status bar

    def __init__(self, parent=None):
        super().__init__(parent)
        # ── data state (set via set_data) ──
        self._td: Optional[sta.TrackData] = None
        self._channels: Optional[Dict[str, np.ndarray]] = None
        self._field_shape: Tuple[int, int] = (0, 0)
        self._n_frames = 1
        self._m = 0
        self._n_multipoints = 1
        self._pixel_size: Optional[float] = None
        self._t = 0
        # per-frame FieldBundle cache, keyed by (t, compute-signature)
        self._fb_cache: Dict[Tuple, Optional[sta.FieldBundle]] = {}
        self._applying = False

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_frame)

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # Control bar (two rows to stay compact).
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        self.cmb_view = self._combo([lbl for _k, lbl in _VIEWS], self._on_view_changed)
        self.cmb_field = self._combo([lbl for _k, lbl, _d in _FIELDS],
                                     self._recompute_render)
        self.cmb_render = self._combo(["Heatmap", "Filled contour", "Line contour"],
                                      self._render)
        self.cmb_proj = self._combo(["XY", "XZ", "YZ"], self._render)
        self.cmb_color_by = self._combo(["Time", "Displacement"], self._render)
        self.cmb_cmap = self._combo(ALL_CMAPS, self._render)
        row1.addWidget(QLabel("View:"))
        row1.addWidget(self.cmb_view)
        row1.addWidget(QLabel("Field:"))
        row1.addWidget(self.cmb_field)
        row1.addWidget(QLabel("Render:"))
        row1.addWidget(self.cmb_render)
        row1.addWidget(QLabel("Proj:"))
        row1.addWidget(self.cmb_proj)
        row1.addWidget(QLabel("Color by:"))
        row1.addWidget(self.cmb_color_by)
        row1.addWidget(QLabel("Cmap:"))
        row1.addWidget(self.cmb_cmap)
        row1.addStretch(1)
        root.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(6)
        self.cmb_mode = self._combo(["Cumulative", "Incremental"], self._on_compute_changed)
        self.spn_grid = QSpinBox()
        self.spn_grid.setRange(4, 256)
        self.spn_grid.setValue(24)
        self.spn_grid.setSuffix(" px")
        self.spn_grid.valueChanged.connect(self._on_compute_changed)
        self.spn_smooth = QDoubleSpinBox()
        self.spn_smooth.setDecimals(1)
        self.spn_smooth.setRange(0.0, 10.0)
        self.spn_smooth.setSingleStep(0.5)
        self.spn_smooth.setValue(1.0)
        self.spn_smooth.setToolTip(
            "Gaussian smoothing σ, in grid cells (Cell-Tracker model). "
            "The field is linearly interpolated inside the tracked-particle "
            "hull and zero outside it — never extrapolated.")
        self.spn_smooth.valueChanged.connect(self._on_compute_changed)
        self.spn_arrows = QSpinBox()
        self.spn_arrows.setRange(4, 64)
        self.spn_arrows.setValue(24)
        self.spn_arrows.valueChanged.connect(self._render)
        self.chk_bg = QCheckBox("Background")
        self.chk_bg.setChecked(True)
        self.chk_bg.toggled.connect(self._render)
        self.cmb_channel = self._combo([], self._render)
        self.chk_scalebar = QCheckBox("Scale bar")
        self.chk_scalebar.toggled.connect(self._render)
        # Stress material params (shown only for stress fields).
        self.spn_E = QDoubleSpinBox()
        self.spn_E.setRange(0.0, 1e9)
        self.spn_E.setValue(1000.0)
        self.spn_E.setPrefix("E=")
        self.spn_E.valueChanged.connect(self._render)
        self.spn_nu = QDoubleSpinBox()
        self.spn_nu.setRange(0.0, 0.49)
        self.spn_nu.setSingleStep(0.01)
        self.spn_nu.setValue(0.3)
        self.spn_nu.setPrefix("ν=")
        self.spn_nu.valueChanged.connect(self._render)
        row2.addWidget(QLabel("Mode:"))
        row2.addWidget(self.cmb_mode)
        row2.addWidget(QLabel("Grid:"))
        row2.addWidget(self.spn_grid)
        row2.addWidget(QLabel("Smooth:"))
        row2.addWidget(self.spn_smooth)
        row2.addWidget(QLabel("Arrows:"))
        row2.addWidget(self.spn_arrows)
        row2.addWidget(self.chk_bg)
        row2.addWidget(self.cmb_channel)
        row2.addWidget(self.chk_scalebar)
        row2.addWidget(self.spn_E)
        row2.addWidget(self.spn_nu)
        row2.addStretch(1)
        root.addLayout(row2)

        # Canvas.
        self.canvas = MplCanvas(self, width=7, height=5)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        root.addWidget(self.canvas, stretch=1)

        # Frame transport.
        trow = QHBoxLayout()
        trow.setSpacing(6)
        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(scaled(32))
        self.btn_play.clicked.connect(self._toggle_play)
        self.spn_fps = QSpinBox()
        self.spn_fps.setRange(1, 60)
        self.spn_fps.setValue(8)
        self.spn_fps.setSuffix(" fps")
        self.lbl_frame = QLabel("T 0/0")
        self.lbl_frame.setStyleSheet(scale_qss(f"color: {Settings.FG_SECONDARY}; font: 9pt;"))
        self.strip = FrameStrip()
        self.strip.current_changed.connect(self._on_frame_changed)
        trow.addWidget(self.btn_play)
        trow.addWidget(self.spn_fps)
        trow.addWidget(self.lbl_frame)
        trow.addWidget(self.strip, stretch=1)
        root.addLayout(trow)

        self._sync_control_visibility()

    def _combo(self, items: List[str], slot) -> QComboBox:
        cb = QComboBox()
        cb.addItems(items)
        cb.currentIndexChanged.connect(slot)
        return cb

    # ── public API ──────────────────────────────────────────────────────────
    def set_data(
        self, rows: List[Dict[str, Any]], channels: Optional[Dict[str, np.ndarray]],
        field_shape: Tuple[int, int], pixel_size: Optional[float], n_frames: int,
        *, m: int = 0, n_multipoints: int = 1,
        z_step_um: Optional[float] = None, time_step: float = 1.0,
    ) -> None:
        """Feed the panel one multipoint's tracked rows + display context."""
        self._m = int(m)
        self._n_multipoints = int(max(1, n_multipoints))
        self._field_shape = tuple(field_shape)
        self._pixel_size = pixel_size
        self._n_frames = int(max(1, n_frames))
        self._channels = channels or None
        self._td = sta.build_track_data(
            rows, m=self._m, pixel_size_um=pixel_size,
            z_step_um=z_step_um, time_step=time_step)
        self._fb_cache.clear()

        # Channel combo (background source).
        self._applying = True
        self.cmb_channel.clear()
        if self._channels:
            self.cmb_channel.addItems(list(self._channels.keys()))
        # Projection combo only useful for 3D.
        self.cmb_proj.setEnabled(self._td.ndim == 3)
        self._applying = False

        self.strip.set_count(self._n_frames)
        self._t = min(self._t, self._n_frames - 1)
        self.strip.set_current(self._t, emit=False)
        self._sync_control_visibility()
        self._render()

    def set_compact(self, compact: bool) -> None:
        """Kept for API parity with SpatialMapsPanel (single layout here)."""
        # This panel uses one responsive layout; nothing to reflow.
        return

    def set_node_templates(self, names: List[str]) -> None:
        """API parity with SpatialMapsPanel; templates not yet implemented."""
        return

    # ── control callbacks ─────────────────────────────────────────────────────
    def _view_key(self) -> str:
        return _VIEWS[self.cmb_view.currentIndex()][0]

    def _field_key(self) -> str:
        return _FIELDS[self.cmb_field.currentIndex()][0]

    def _on_view_changed(self, *_):
        self._sync_control_visibility()
        self._render()

    def _on_compute_changed(self, *_):
        if self._applying:
            return
        self._fb_cache.clear()
        self._render()

    def _recompute_render(self, *_):
        self._sync_control_visibility()
        self._render()

    def _sync_control_visibility(self) -> None:
        view = self._view_key()
        is_field = view in ("field", "heatmap_quiver")
        is_quiver = view in ("quiver", "heatmap_quiver")
        is_traj = view == "trajectories"
        self.cmb_field.setVisible(is_field)
        self.cmb_render.setVisible(view == "field")
        self.cmb_proj.setVisible(is_traj and (self._td is None or self._td.ndim == 3))
        self.cmb_color_by.setVisible(is_traj)
        self.cmb_cmap.setVisible(view in ("field", "heatmap_quiver", "quiver",
                                          "trajectories"))
        self.spn_grid.setVisible(is_field or is_quiver)
        self.spn_smooth.setVisible(is_field or is_quiver)
        self.spn_arrows.setVisible(is_quiver)
        show_bg = view in ("field", "quiver", "heatmap_quiver", "trajectories")
        self.chk_bg.setVisible(show_bg)
        self.cmb_channel.setVisible(show_bg)
        self.chk_scalebar.setVisible(show_bg)
        is_stress = is_field and self._field_key() in _STRESS_KEYS
        self.spn_E.setVisible(is_stress)
        self.spn_nu.setVisible(is_stress)

    def _on_frame_changed(self, t: int) -> None:
        self._t = int(t)
        self.lbl_frame.setText(f"T {self._t + 1}/{self._n_frames}")
        self._render()

    def _toggle_play(self) -> None:
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.btn_play.setText("▶")
        else:
            self._play_timer.start(int(1000 / max(1, self.spn_fps.value())))
            self.btn_play.setText("⏸")

    def _advance_frame(self) -> None:
        nxt = (self._t + 1) % self._n_frames
        self.strip.set_current(nxt, emit=True)

    # ── compute ────────────────────────────────────────────────────────────────
    def _mode(self) -> str:
        return "incremental" if self.cmb_mode.currentIndex() == 1 else "cumulative"

    def _bundle(self, t: int) -> Optional[sta.FieldBundle]:
        if self._td is None:
            return None
        gstep = float(self.spn_grid.value())
        smooth = float(self.spn_smooth.value())
        mode = self._mode()
        key = (int(t), mode, gstep, smooth)
        if key not in self._fb_cache:
            try:
                self._fb_cache[key] = sta.compute_field_bundle(
                    self._td, t, mode=mode, grid_step=gstep, smoothness=smooth)
            except Exception as exc:  # noqa: BLE001 — degrade gracefully
                self.status_message.emit(f"SerialTrack field error: {exc}")
                self._fb_cache[key] = None
        return self._fb_cache[key]

    def _stress_kw(self) -> Dict[str, float]:
        return {"youngs_modulus": float(self.spn_E.value()),
                "poisson_ratio": float(self.spn_nu.value())}

    def _bg_frame(self, t: int) -> Optional[np.ndarray]:
        if not (self.chk_bg.isChecked() and self._channels):
            return None
        name = self.cmb_channel.currentText()
        arr = self._channels.get(name)
        if arr is None or arr.ndim < 3:
            return None
        fi = min(int(t), arr.shape[0] - 1)
        try:
            return frame_to_uint8(np.asarray(arr[fi]))
        except Exception:  # noqa: BLE001
            return None

    # ── rendering ───────────────────────────────────────────────────────────
    def _render(self, *_):
        if self._applying:
            return
        self.canvas.clear()
        ax = self.canvas.add_subplot(111)
        if self._td is None or self._td.n_tracks == 0:
            ax.text(0.5, 0.5, "No tracked particles.\nRun Track Objects first.",
                    ha="center", va="center", color=Settings.FG_SECONDARY,
                    transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas.safe_draw()
            return
        view = self._view_key()
        try:
            if view == "trajectories":
                self._render_trajectories(ax)
            elif view == "field":
                self._render_scalar(ax, quiver=False)
            elif view == "heatmap_quiver":
                self._render_scalar(ax, quiver=True)
            elif view == "quiver":
                self._render_quiver(ax, background=True)
            elif view == "histogram":
                self._render_histogram(ax)
            elif view == "dashboard":
                self._render_dashboard(ax)
        except Exception as exc:  # noqa: BLE001 — never crash the GUI on a bad frame
            ax.clear()
            ax.text(0.5, 0.5, f"Render error:\n{exc}", ha="center", va="center",
                    color=Settings.ACCENT_RED, transform=ax.transAxes, fontsize=8)
            ax.set_axis_off()
        self.canvas.safe_tight_layout()
        self.canvas.safe_draw()

    def _draw_background(self, ax) -> None:
        H, W = self._field_shape
        bg = self._bg_frame(self._t)
        if bg is not None:
            ax.imshow(bg, cmap="gray", extent=[0, W, H, 0], zorder=0,
                      aspect="equal")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)  # image convention: y increases downward
        if self.chk_scalebar.isChecked() and self._pixel_size:
            draw_scale_bar(ax, pixel_size_um=self._pixel_size)

    def _grid_extent(self, fb: sta.FieldBundle) -> List[float]:
        gy, gx = fb.grids[0], fb.grids[1]
        return [float(gx.min()), float(gx.max()), float(gy.max()), float(gy.min())]

    def _render_trajectories(self, ax) -> None:
        td = self._td
        fi = td.frame_index(self._t)
        if fi < 0:
            fi = td.n_frames - 1
        self._draw_background(ax)
        traj = td.trajectories[:, : fi + 1, :]  # (N, fi+1, D)
        # Axis pair for the chosen projection.
        proj = self.cmb_proj.currentText() if td.ndim == 3 else "XY"
        ai, aj = {"XY": (1, 0), "XZ": (1, 2), "YZ": (0, 2)}[proj]
        color_by = self.cmb_color_by.currentText()
        cmap = self.cmb_cmap.currentText()
        n = min(traj.shape[0], 800)
        import matplotlib as mpl
        cmo = mpl.colormaps[cmap]
        # Per-track color: by time uses frame fraction; by displacement uses net |u|.
        if color_by == "Displacement":
            net = np.linalg.norm(
                np.nan_to_num(traj[:, -1, :] - traj[:, 0, :]), axis=1)
            vmax = float(np.nanmax(net)) if net.size else 1.0
            vmax = vmax if vmax > 0 else 1.0
        for k in range(n):
            path = traj[k]
            valid = np.isfinite(path).all(axis=1)
            if valid.sum() < 2:
                continue
            xs = path[valid, ai]
            ys = path[valid, aj]
            if color_by == "Displacement":
                col = cmo(float(net[k]) / vmax)
            else:
                col = cmo(k / max(n - 1, 1))
            ax.plot(xs, ys, "-", color=col, lw=0.8, alpha=0.8, zorder=2)
            ax.plot(xs[-1], ys[-1], "o", color=col, ms=2.5, zorder=3)
        ax.set_title(f"Trajectories · T{self._t + 1} · {n} tracks",
                     color=Settings.FG_PRIMARY, fontsize=9)
        if proj == "XY":
            ax.set_aspect("equal")

    def _render_scalar(self, ax, quiver: bool) -> None:
        fb = self._bundle(self._t)
        if fb is None:
            ax.text(0.5, 0.5, "Not enough tracked particles for a field.",
                    ha="center", va="center", color=Settings.FG_SECONDARY,
                    transform=ax.transAxes)
            ax.set_axis_off()
            return
        key = self._field_key()
        data = sta.scalar_field(fb, key, **self._stress_kw())
        data = np.asarray(data, dtype=float)
        # 3D fields: show the mid-Z slice.
        if data.ndim == 3:
            data = data[:, :, data.shape[2] // 2]
        cmap = self.cmb_cmap.currentText()
        if key in _DIVERGENT and cmap in CMAPS_SEQ:
            cmap = "coolwarm"
        vmax = float(np.nanmax(np.abs(data))) if np.isfinite(data).any() else 1.0
        if key in _DIVERGENT:
            vmin, vmaxr = -vmax, vmax
        else:
            vmin = float(np.nanmin(data)) if np.isfinite(data).any() else 0.0
            vmaxr = float(np.nanmax(data)) if np.isfinite(data).any() else 1.0
        extent = self._grid_extent(fb)
        self._draw_background(ax)
        render = self.cmb_render.currentText()
        gy, gx = fb.grids[0], fb.grids[1]
        if render == "Filled contour":
            im = ax.contourf(gx, gy, data, levels=20, cmap=cmap,
                             vmin=vmin, vmax=vmaxr, alpha=0.85, zorder=1)
        elif render == "Line contour":
            im = ax.contour(gx, gy, data, levels=12, cmap=cmap, zorder=1)
        else:
            im = ax.imshow(data, cmap=cmap, extent=extent, origin="upper",
                           vmin=vmin, vmax=vmaxr, alpha=0.85, zorder=1,
                           aspect="equal")
        cbar = self.canvas.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors=Settings.FG_SECONDARY, labelsize=7)
        label = dict((k, l) for k, l, _d in _FIELDS)[key]
        ax.set_title(f"{label} · T{self._t + 1}", color=Settings.FG_PRIMARY,
                     fontsize=9)
        if quiver:
            self._quiver_on(ax, fb)

    def _render_quiver(self, ax, background: bool) -> None:
        fb = self._bundle(self._t)
        if fb is None:
            ax.text(0.5, 0.5, "Not enough tracked particles for a field.",
                    ha="center", va="center", color=Settings.FG_SECONDARY,
                    transform=ax.transAxes)
            ax.set_axis_off()
            return
        if background:
            self._draw_background(ax)
        self._quiver_on(ax, fb)
        ax.set_title(f"Displacement field · T{self._t + 1}",
                     color=Settings.FG_PRIMARY, fontsize=9)

    def _quiver_on(self, ax, fb: sta.FieldBundle) -> None:
        gy, gx = fb.grids[0], fb.grids[1]
        u = fb.disp_field.components  # (D, ny, nx[, nz])
        uy, ux = u[0], u[1]
        if uy.ndim == 3:  # 3D → mid-Z slice for the 2D quiver
            zc = uy.shape[2] // 2
            uy, ux = uy[:, :, zc], ux[:, :, zc]
            gy, gx = gy[:, :, zc], gx[:, :, zc]
        # Subsample to ~arrows count along the longer axis.
        target = int(self.spn_arrows.value())
        step = max(1, max(gy.shape) // target)
        sl = (slice(None, None, step), slice(None, None, step))
        X, Y = gx[sl], gy[sl]
        U, V = ux[sl], uy[sl]  # U = x-comp, V = y-comp
        mag = np.sqrt(U ** 2 + V ** 2)
        cmap = self.cmb_cmap.currentText()
        if cmap in CMAPS_DIV:
            cmap = "viridis"
        q = ax.quiver(X, Y, U, V, mag, cmap=cmap, angles="xy",
                      scale_units="xy", scale=1.0, width=0.003, zorder=4)
        return q

    def _render_histogram(self, ax) -> None:
        coords, disp = sta.particle_displacement(
            self._td, self._t, mode=self._mode())
        if len(disp) == 0:
            ax.text(0.5, 0.5, "No displacement at this frame.",
                    ha="center", va="center", color=Settings.FG_SECONDARY,
                    transform=ax.transAxes)
            ax.set_axis_off()
            return
        mag = np.linalg.norm(disp, axis=1)
        if self._pixel_size:
            mag = mag * self._pixel_size
            unit = "µm"
        else:
            unit = "px"
        ax.hist(mag, bins=40, color=Settings.ACCENT_CYAN, alpha=0.85)
        ax.axvline(float(np.mean(mag)), color=Settings.ACCENT_PINK, lw=1.2,
                   label=f"mean {np.mean(mag):.2f} {unit}")
        ax.axvline(float(np.median(mag)), color=Settings.ACCENT_YELLOW, lw=1.0,
                   ls="--", label=f"median {np.median(mag):.2f} {unit}")
        ax.set_xlabel(f"Displacement magnitude ({unit})", fontsize=8)
        ax.set_ylabel("Particles", fontsize=8)
        ax.set_title(f"Displacement distribution · T{self._t + 1} · "
                     f"n={len(mag)}", color=Settings.FG_PRIMARY, fontsize=9)
        ax.legend(fontsize=7, facecolor=Settings.BG_TERTIARY,
                  edgecolor=Settings.BORDER_COLOR, labelcolor=Settings.FG_SECONDARY)

    def _render_dashboard(self, ax) -> None:
        td = self._td
        frames = td.frames
        detected = [len(td.coords[f]) for f in frames]
        tracked = [int(np.sum(td.track_ids[f] >= 0)) for f in frames]
        ratio = [(tr / de if de else 0.0) for tr, de in zip(tracked, detected)]
        x = list(range(len(frames)))
        ax.bar([i - 0.2 for i in x], detected, width=0.4,
               color=Settings.BORDER_COLOR, label="Detected")
        ax.bar([i + 0.2 for i in x], tracked, width=0.4,
               color=Settings.ACCENT_GREEN, label="Tracked")
        ax.set_xlabel("Frame", fontsize=8)
        ax.set_ylabel("Particles", fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(x, ratio, "-o", color=Settings.ACCENT_CYAN, ms=3,
                 label="Track ratio")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Tracking ratio", color=Settings.ACCENT_CYAN, fontsize=8)
        ax2.tick_params(colors=Settings.FG_SECONDARY, labelsize=7)
        ax.axvline(td.frame_index(self._t), color=Settings.ACCENT_YELLOW,
                   lw=1.0, ls=":", zorder=0)
        ax.set_title("Tracking dashboard", color=Settings.FG_PRIMARY, fontsize=9)
        ax.legend(fontsize=7, loc="upper left", facecolor=Settings.BG_TERTIARY,
                  edgecolor=Settings.BORDER_COLOR, labelcolor=Settings.FG_SECONDARY)
