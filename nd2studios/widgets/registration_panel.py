"""In-viewer image-registration result panel (V1.56).

``RegistrationPanel`` is the "Registration" tab in the Pipelines image-viewer
stack (sibling of :class:`~nd2studios.widgets.dvc_panel.DVCPanel`). A registration
node estimates a per-frame transform on the wired reference channel and applies it
to every channel; this panel lets the user inspect the result:

* **Before / After** — the raw vs stabilized reference-channel frame, side by side,
  playable through the timelapse with a :class:`FrameStrip`.
* **Shifts vs time** — the recovered per-frame ``(dx, dy)`` drift (µm if a pixel
  size is known) + the registration confidence, with a marker on the current frame.
* **Write aligned → processed channels** — push the stabilized channels into the
  record's processed view so downstream nodes / exports use them.

Backend-pure maths lives in :mod:`nd2studios.backend.registration.estimate`; this
widget only visualizes the :class:`~nd2studios.core.registration_registry.
RegistrationResult` the node produced.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QSpinBox,
    QSizePolicy,
)

from nd2studios.core.settings import Settings
from nd2studios.widgets.common import MplCanvas
from nd2studios.widgets.frame_strip import FrameStrip
from nd2studios.widgets.icon_button import scaled, scale_qss
from nd2studios.widgets.image_viewer import frame_to_uint8

_VIEWS = [("compare", "Before / After"), ("shifts", "Shifts vs time")]


class RegistrationPanel(QWidget):
    """Embeddable, playable registration before/after + drift-plot viewer."""

    m_change_requested = Signal(int)
    status_message = Signal(str)
    write_requested = Signal(int)   # emit the multipoint whose aligned channels to write

    def __init__(self, parent=None):
        super().__init__(parent)
        self._raw: Optional[np.ndarray] = None       # (T,H,W) reference channel raw
        self._aligned: Optional[np.ndarray] = None   # (T,H,W) reference channel aligned
        self._shifts: Optional[np.ndarray] = None    # (T,2) row/col
        self._confidence: Optional[np.ndarray] = None
        self._gated: Optional[np.ndarray] = None      # (T,) bool — held frames
        self._min_confidence: float = 0.0
        self._pixel_size: Optional[float] = None
        self._t = 0
        self._n_frames = 1
        self._m = 0
        self._n_multipoints = 1
        self._channel = ""
        self._model = "translation"
        self._ref_mode = "first"
        self._n_channels = 0
        self._applying = False
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_frame)
        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(scaled(4), scaled(4), scaled(4), scaled(4))
        root.setSpacing(scaled(4))

        row = QHBoxLayout()
        row.setSpacing(scaled(6))
        self.cmb_view = QComboBox()
        self.cmb_view.addItems([lbl for _k, lbl in _VIEWS])
        self.cmb_view.currentIndexChanged.connect(self._render)
        row.addWidget(QLabel("View:"))
        row.addWidget(self.cmb_view)
        self.lbl_meta = QLabel("")
        self.lbl_meta.setStyleSheet(scale_qss(f"color: {Settings.FG_SECONDARY}; font: 9pt;"))
        row.addWidget(self.lbl_meta)
        row.addStretch(1)
        self.btn_write = QPushButton("Write aligned → processed channels")
        self.btn_write.setObjectName("compactBtn")
        self.btn_write.setToolTip(
            "Replace the processed channels with the stabilized ones for the "
            "current multipoint, so downstream nodes and exports use the aligned "
            "data. (Register once on the reference channel, apply to all.)")
        self.btn_write.clicked.connect(self._on_write_clicked)
        row.addWidget(self.btn_write)
        root.addLayout(row)

        self.canvas = MplCanvas(self, width=7, height=5)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        root.addWidget(self.canvas, stretch=1)

        # ── multipoint selector + frame transport ──
        trow = QHBoxLayout()
        trow.setSpacing(scaled(6))
        self.lbl_m = QLabel("M:")
        self.cmb_m = QComboBox()
        self.cmb_m.currentIndexChanged.connect(self._on_m_changed)
        self.btn_play = QPushButton("▶")
        self.btn_play.setObjectName("compactBtn")
        self.btn_play.setFixedWidth(scaled(32))
        self.btn_play.setToolTip("Play / pause through the timelapse.")
        self.btn_play.clicked.connect(self._toggle_play)
        self.spn_fps = QSpinBox()
        self.spn_fps.setRange(1, 60)
        self.spn_fps.setValue(6)
        self.spn_fps.setSuffix(" fps")
        self.lbl_frame = QLabel("T 0/0")
        self.lbl_frame.setStyleSheet(scale_qss(f"color: {Settings.FG_SECONDARY}; font: 9pt;"))
        self.strip = FrameStrip()
        self.strip.current_changed.connect(self._on_frame_changed)
        trow.addWidget(self.lbl_m)
        trow.addWidget(self.cmb_m)
        trow.addWidget(self.btn_play)
        trow.addWidget(self.spn_fps)
        trow.addWidget(self.lbl_frame)
        trow.addWidget(self.strip, stretch=1)
        root.addLayout(trow)

    # ── public API ──────────────────────────────────────────────────────
    def set_data(
        self,
        raw_ref: np.ndarray,
        aligned_ref: np.ndarray,
        shifts_px: np.ndarray,
        confidence: np.ndarray,
        *,
        pixel_size: Optional[float] = None,
        m: int = 0,
        n_multipoints: int = 1,
        model: str = "translation",
        reference_mode: str = "first",
        channel: str = "",
        n_channels: int = 0,
        gated: Optional[np.ndarray] = None,
        min_confidence: float = 0.0,
    ) -> None:
        """Feed one multipoint's registration result (reference-channel raw +
        aligned series, per-frame shifts + confidence). ``gated`` (bool per frame)
        flags frames whose transform was held because confidence fell below
        ``min_confidence`` — shaded on the drift plot so degradation is visible."""
        self._applying = True
        self._play_timer.stop()
        self.btn_play.setText("▶")
        self._raw = np.asarray(raw_ref)
        self._aligned = np.asarray(aligned_ref)
        self._shifts = np.asarray(shifts_px, dtype=float)
        self._confidence = np.asarray(confidence, dtype=float)
        self._gated = (np.asarray(gated, dtype=bool) if gated is not None else None)
        self._min_confidence = float(min_confidence)
        self._pixel_size = pixel_size
        self._m = int(m)
        self._n_multipoints = int(max(1, n_multipoints))
        self._model = str(model)
        self._ref_mode = str(reference_mode)
        self._channel = str(channel)
        self._n_channels = int(n_channels)
        self._n_frames = int(self._raw.shape[0]) if self._raw is not None else 1
        self._t = 0

        self.cmb_m.blockSignals(True)
        self.cmb_m.clear()
        self.cmb_m.addItems([f"M{i + 1}" for i in range(self._n_multipoints)])
        if 0 <= self._m < self._n_multipoints:
            self.cmb_m.setCurrentIndex(self._m)
        self.cmb_m.blockSignals(False)
        self.lbl_m.setVisible(self._n_multipoints > 1)
        self.cmb_m.setVisible(self._n_multipoints > 1)

        unit = "µm" if self._pixel_size else "px"
        n_gated = int(self._gated.sum()) if self._gated is not None else 0
        gated_txt = (f" · (!) {n_gated} low-confidence frame(s) held"
                     if n_gated else "")
        self.lbl_meta.setText(
            f"{self._model} · ref={self._ref_mode} · on '{self._channel}' · "
            f"shift in {unit}{gated_txt}")
        self.strip.set_count(self._n_frames)
        self.strip.set_current(self._t, emit=False)
        self._update_frame_label()
        self._applying = False
        self._render()

    def set_compact(self, compact: bool) -> None:
        return

    # ── transport ───────────────────────────────────────────────────────
    def _update_frame_label(self) -> None:
        self.lbl_frame.setText(f"T {int(self._t) + 1}/{self._n_frames}")

    def _on_frame_changed(self, t: int) -> None:
        self._t = int(t)
        self._update_frame_label()
        self._render()

    def _on_m_changed(self, idx: int) -> None:
        if self._applying or idx < 0:
            return
        self.m_change_requested.emit(int(idx))

    def _toggle_play(self) -> None:
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.btn_play.setText("▶")
        else:
            self._play_timer.start(int(1000 / max(1, self.spn_fps.value())))
            self.btn_play.setText("⏸")

    def _advance_frame(self) -> None:
        nxt = (int(self._t) + 1) % max(1, self._n_frames)
        self.strip.set_current(nxt, emit=True)

    def _on_write_clicked(self) -> None:
        self.write_requested.emit(int(self._m))

    def _view_key(self) -> str:
        return _VIEWS[self.cmb_view.currentIndex()][0]

    # ── rendering ───────────────────────────────────────────────────────
    def _render(self, *_):
        if self._applying:
            return
        self.canvas.clear()
        if self._raw is None or self._aligned is None:
            ax = self.canvas.add_subplot(111)
            ax.text(0.5, 0.5, "No registration result.\nRun a Registration node "
                    "first.", ha="center", va="center",
                    color=Settings.FG_SECONDARY, transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas.safe_draw()
            return
        try:
            if self._view_key() == "compare":
                self._render_compare()
            else:
                self._render_shifts()
        except Exception as exc:  # noqa: BLE001 — never crash the GUI
            self.canvas.clear()
            ax = self.canvas.add_subplot(111)
            ax.text(0.5, 0.5, f"Render error:\n{exc}", ha="center", va="center",
                    color=Settings.ACCENT_RED, transform=ax.transAxes, fontsize=8)
            ax.set_axis_off()
        self.canvas.safe_tight_layout()
        self.canvas.safe_draw()

    def _render_compare(self) -> None:
        t = int(self._t)
        ax1 = self.canvas.add_subplot(121)
        ax2 = self.canvas.add_subplot(122)
        raw = frame_to_uint8(np.asarray(self._raw[t]))
        ali = frame_to_uint8(np.asarray(self._aligned[t]))
        ax1.imshow(raw, cmap="gray", aspect="equal")
        ax2.imshow(ali, cmap="gray", aspect="equal")
        ax1.set_title(f"Raw · T{t + 1}", color=Settings.FG_PRIMARY, fontsize=9)
        ax2.set_title(f"Aligned · T{t + 1}", color=Settings.FG_PRIMARY, fontsize=9)
        for ax in (ax1, ax2):
            ax.set_axis_off()

    def _render_shifts(self) -> None:
        ax = self.canvas.add_subplot(111)
        shifts = self._shifts
        conf = self._confidence
        ts = np.arange(shifts.shape[0])
        scale = float(self._pixel_size) if self._pixel_size else 1.0
        unit = "µm" if self._pixel_size else "px"
        dy = shifts[:, 0] * scale   # row
        dx = shifts[:, 1] * scale   # col
        # Shade frames whose transform was HELD (confidence below threshold) so a
        # user sees exactly where/why later timepoints stopped tracking.
        gated = self._gated
        if gated is not None and gated.size == ts.size and gated.any():
            for i in np.where(gated)[0]:
                ax.axvspan(i - 0.5, i + 0.5, color=Settings.ACCENT_RED, alpha=0.12,
                           lw=0, zorder=0)
            ax.axvspan(np.nan, np.nan, color=Settings.ACCENT_RED, alpha=0.25,
                       label="held (low conf.)")
        ax.plot(ts, dx, "-o", ms=3, color=Settings.ACCENT_CYAN, label=f"Δx ({unit})")
        ax.plot(ts, dy, "-o", ms=3, color=Settings.ACCENT_PINK, label=f"Δy ({unit})")
        ax.axvline(int(self._t), color=Settings.FG_SECONDARY, lw=1.0, ls="--")
        ax.set_xlabel("Frame (T)", fontsize=8)
        ax.set_ylabel(f"Shift ({unit})", fontsize=8)
        ax.tick_params(colors=Settings.FG_SECONDARY, labelsize=7)
        ax.set_title(f"Recovered drift · '{self._channel}'",
                     color=Settings.FG_PRIMARY, fontsize=9)
        ax.legend(fontsize=7, facecolor=Settings.BG_TERTIARY,
                  edgecolor=Settings.BORDER_COLOR, labelcolor=Settings.FG_SECONDARY)
        if conf is not None and conf.size == ts.size:
            ax2 = ax.twinx()
            ax2.plot(ts, conf, "-", lw=1.0, color=Settings.ACCENT_YELLOW,
                     alpha=0.7, label="confidence")
            ax2.set_ylabel("Confidence", fontsize=8, color=Settings.ACCENT_YELLOW)
            ax2.set_ylim(-1.05, 1.05)
            ax2.tick_params(colors=Settings.FG_SECONDARY, labelsize=7)
