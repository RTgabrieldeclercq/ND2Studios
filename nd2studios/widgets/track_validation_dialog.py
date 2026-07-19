"""
TrackValidationDialog — step-through review UI for identified objects.

Reviews objects **by unit** in a 1/2/4-up grid of cropped panels.  A unit is
either a *track* (all frames sharing a ``track_id``, when a Track Objects node
linked them) or a *single object* (one label on one frame).  Passing
``include_untracked=True`` makes every object reviewable even when nothing was
tracked — so the Review Objects node works with or without an upstream Track
Objects node.

All visible units share one T slider ("Play All").  Each panel has its own
Accept / Reject buttons.  Once every panel in the current batch has been decided
the next batch loads automatically.

Crop size is 3× the union bounding box of the unit (center ± 1.5× bbox
half-extent), clamped to the full FOV.

Decisions are exposed as row references (``accepted_rows`` / ``rejected_rows``)
plus the derived ``accepted_track_ids`` / ``rejected_track_ids`` (back-compat for
the Results page, which only ever reviews tracks).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from nd2studios.backend.exporters.composite_exporter import (
    CHANNEL_COLORS,
    _composite_frame,
)
from nd2studios.core.settings import Settings
from nd2studios.widgets.common import MplCanvas
from nd2studios.widgets.icon_button import scale_qss, scaled
from nd2studios.widgets.image_viewer import ImageCanvas


# Cell-boundary outline colour for the cropped cell, matching CellTracker's
# Results (Cells) cell-detail panel (R, G, B), drawn at full opacity.
_BOUNDARY = np.array([255, 80, 80], dtype=np.uint8)

# Crop expansion factor: total crop side = CROP_FACTOR × object bbox side.
_CROP_FACTOR = 3.0


class _ReviewUnit:
    """One reviewable thing: a track (many frames) or a single object (one row)."""

    def __init__(self, uid: str, rows: List[Dict[str, Any]],
                 track_id: Optional[int]) -> None:
        self.uid = uid
        self.rows = rows
        self.track_id = track_id
        self.first_frame = min(int(r.get("frame", 0)) for r in rows)

    @property
    def label_text(self) -> str:
        if self.track_id is not None:
            n = len(self.rows)
            return f"Track {self.track_id}  •  {n} frame{'s' if n != 1 else ''}"
        r = self.rows[0]
        return f"Object {r.get('label_id')}  •  frame {int(r.get('frame', 0)) + 1}"


def _build_units(
    measurements: List[Dict[str, Any]], include_untracked: bool,
) -> List[_ReviewUnit]:
    """Group rows into review units (tracks + optionally single objects)."""
    by_track: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    singles: List[Dict[str, Any]] = []
    for r in measurements:
        tid = r.get("track_id")
        if tid is not None:
            by_track[int(tid)].append(r)
        elif include_untracked:
            singles.append(r)
    units: List[_ReviewUnit] = [
        _ReviewUnit(f"t{tid}", rows, tid) for tid, rows in by_track.items()
    ]
    for r in singles:
        uid = (f"o{int(r.get('frame', 0))}_{r.get('label_id')}_"
               f"{r.get('segmentation_channel', '')}")
        units.append(_ReviewUnit(uid, [r], None))
    units.sort(key=lambda u: (u.first_frame, str(u.uid)))
    return units


# ── Per-unit panel ──────────────────────────────────────────────────────────────

class _TrackPanel(QWidget):
    """One slot in the review grid — image canvas + per-unit controls."""

    accept_clicked = Signal(int)   # emits panel_index
    reject_clicked = Signal(int)   # emits panel_index

    # Border colours for decided state.
    _BORDER_ACCEPTED = "#50fa7b"
    _BORDER_REJECTED = "#ff5555"
    _BORDER_NEUTRAL  = Settings.BORDER_COLOR

    def __init__(self, panel_idx: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("trackPanel")
        self._idx = panel_idx
        self._crop_frames: List[np.ndarray] = []
        self._info_texts: List[str] = []
        self._trace_frames: List[int] = []
        self._uid: Optional[str] = None
        self._decided: Optional[str] = None   # "accepted" | "rejected" | None
        self._marker = None                    # current-frame line on the trace

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Unit info label.
        self._lbl = QLabel("")
        self._lbl.setAlignment(Qt.AlignCenter)
        self._lbl.setStyleSheet(scale_qss(
            f"color: {Settings.FG_SECONDARY}; font: 8pt;"
        ))
        layout.addWidget(self._lbl)

        # Image canvas (cropped cell with red boundary outline).
        self._canvas = ImageCanvas(self)
        self._canvas.setMinimumSize(scaled(120), scaled(120))
        layout.addWidget(self._canvas, stretch=2)

        # Per-frame cell-detail text (frame / area / ecc / intensity).
        self._info = QLabel("")
        self._info.setAlignment(Qt.AlignCenter)
        self._info.setWordWrap(True)
        self._info.setStyleSheet(scale_qss(f"color: {Settings.FG_SECONDARY}; font: 8pt;"))
        layout.addWidget(self._info)

        # Per-cell metric trace over the unit's frames (CellTracker-style).
        self._trace = MplCanvas(self, width=3.0, height=1.4, dpi=90)
        self._trace.setMinimumHeight(scaled(96))
        self._trace_ax = self._trace.add_subplot(111)
        layout.addWidget(self._trace, stretch=1)

        # Accept / Reject buttons.
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._btn_accept = QPushButton("Accept")
        self._btn_accept.setObjectName("primaryBtn")
        self._btn_accept.clicked.connect(lambda: self.accept_clicked.emit(self._idx))
        btn_row.addWidget(self._btn_accept)

        self._btn_reject = QPushButton("Reject")
        self._btn_reject.setObjectName("rejectBtn")
        self._btn_reject.clicked.connect(lambda: self.reject_clicked.emit(self._idx))
        btn_row.addWidget(self._btn_reject)
        layout.addLayout(btn_row)

        self._set_border(self._BORDER_NEUTRAL)

    # ── Public API ────────────────────────────────────────────────────────────

    def load(
        self,
        uid: str,
        crop_frames: List[np.ndarray],
        label_text: str,
        info_texts: Optional[List[str]] = None,
        trace: Optional[Tuple[List[int], List[float], str]] = None,
    ) -> None:
        """Attach new crop data + per-frame info + metric trace; reset decision."""
        self._uid = uid
        self._crop_frames = crop_frames
        self._info_texts = info_texts or []
        self._decided = None
        self._lbl.setText(label_text)
        self._btn_accept.setEnabled(True)
        self._btn_reject.setEnabled(True)
        self._set_border(self._BORDER_NEUTRAL)
        self._draw_trace(trace)
        self.show_frame(0)

    def clear(self) -> None:
        """Display an empty (placeholder) state."""
        self._uid = None
        self._crop_frames = []
        self._info_texts = []
        self._decided = None
        self._lbl.setText("—")
        self._info.setText("")
        self._btn_accept.setEnabled(False)
        self._btn_reject.setEnabled(False)
        self._canvas.set_image(
            np.zeros((64, 64, 3), dtype=np.uint8)
        )
        self._draw_trace(None)
        self._set_border(self._BORDER_NEUTRAL)

    def show_frame(self, t: int) -> None:
        if not self._crop_frames:
            return
        t = max(0, min(t, len(self._crop_frames) - 1))
        self._canvas.set_image(self._crop_frames[t])
        if 0 <= t < len(self._info_texts):
            self._info.setText(self._info_texts[t])
        self._update_marker(t)

    def _draw_trace(self, trace: Optional[Tuple[List[int], List[float], str]]) -> None:
        """Render the per-cell metric trace (static per unit); marker moves with T."""
        ax = self._trace_ax
        ax.clear()
        self._marker = None
        self._trace_frames = []
        if trace is not None:
            frames, values, ylabel = trace
            self._trace_frames = list(frames)
            if frames:
                ax.plot(frames, values, color=Settings.ACCENT_CYAN, lw=1.4,
                        marker="o", ms=2.5)
                ax.set_ylabel(ylabel, fontsize=7)
                self._marker = ax.axvline(frames[0], color=Settings.ACCENT_YELLOW,
                                          lw=1.0, ls="--")
        ax.tick_params(labelsize=6)
        ax.set_xlabel("frame", fontsize=7)
        self._trace.fig.tight_layout(pad=0.4)
        self._trace.draw_idle()

    def _update_marker(self, t: int) -> None:
        if self._marker is None:
            return
        self._marker.set_xdata([t, t])
        self._trace.draw_idle()

    def mark_decided(self, decision: str) -> None:
        """Visually lock the panel after a decision has been made."""
        self._decided = decision
        self._btn_accept.setEnabled(False)
        self._btn_reject.setEnabled(False)
        if decision == "accepted":
            self._set_border(self._BORDER_ACCEPTED)
            self._lbl.setStyleSheet(scale_qss(
                f"color: {self._BORDER_ACCEPTED}; font: 8pt bold;"
            ))
        else:
            self._set_border(self._BORDER_REJECTED)
            self._lbl.setStyleSheet(scale_qss(
                f"color: {self._BORDER_REJECTED}; font: 8pt bold;"
            ))

    @property
    def uid(self) -> Optional[str]:
        return self._uid

    @property
    def decided(self) -> Optional[str]:
        return self._decided

    # ── Internal ──────────────────────────────────────────────────────────────

    def _set_border(self, colour: str) -> None:
        self.setStyleSheet(scale_qss(
            f"QWidget#trackPanel {{ border: 2px solid {colour}; border-radius: 4px; }}"
        ))


# ── Main dialog ───────────────────────────────────────────────────────────────

class TrackValidationDialog(QDialog):
    """Step-through review of objects with 1/2/4-up display modes."""

    def __init__(
        self,
        measurements: List[Dict[str, Any]],
        label_masks: Dict[str, np.ndarray],
        channels: Dict[str, np.ndarray],
        channel_display: Dict[str, Dict[str, Any]],
        parent: Optional[QWidget] = None,
        include_untracked: bool = False,
        overlay_style: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Parameters
        ----------
        measurements:
            Full list of measurement rows (already tracked, if applicable).
        label_masks:
            {seg_channel: (T, H, W) int32} — raw label arrays.
        channels:
            {ch_name: (T, H, W)} materialized image arrays.
        channel_display:
            exp.channel_display — provides color/LUT/enabled per channel.
        include_untracked:
            When True, objects without a track_id are each reviewed as their own
            single-object unit (so review works with no Track Objects node). When
            False (default), only tracked objects are queued — the Results-page
            behaviour.
        """
        super().__init__(parent)
        self.setWindowTitle("Review Objects")
        self.setModal(True)

        self._measurements = measurements
        self._label_masks = label_masks
        self._channels = channels
        # Overlay rendering for the cropped cell, matching the segmentation
        # overlay format. Defaults to a red outline (StarDist/CellTracker look);
        # the caller can seed it from the viewer's overlay style.
        st = overlay_style or {}
        self._ov_outline: bool = not (st.get("enabled") and st.get("multicolor"))
        self._ov_color: tuple = tuple(st.get("color") or _BOUNDARY.tolist())
        self._ov_weight: int = int(st.get("weight", 1)) if st.get("enabled") else 1
        self._channel_display = channel_display

        # ── Per-cell metric options (CellTracker-style: intensity / area / ecc) ──
        self._metric_options = self._build_metric_options(measurements)
        self._metric_idx = 0

        # ── Unit queue ──────────────────────────────────────────────────────────
        self._units: List[_ReviewUnit] = _build_units(measurements, include_untracked)
        self._units_by_uid: Dict[str, _ReviewUnit] = {u.uid: u for u in self._units}
        self._unit_queue: List[str] = [u.uid for u in self._units]

        # ── State ─────────────────────────────────────────────────────────────
        self._mode: int = 1           # 1 | 2 | 4
        self._batch_start: int = 0    # index into _unit_queue of batch[0]
        self._panels: List[_TrackPanel] = []
        self._panel_crops: List[List[np.ndarray]] = []  # [panel][t] -> rgb
        self._rejected_uids: Set[str] = set()
        self._accepted_uids: Set[str] = set()

        # Global T / playback state.
        self._n_t: int = 1
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)

        # Image dimensions.
        self._img_h = 1
        self._img_w = 1
        if channels:
            s = next(iter(channels.values()))
            self._img_h, self._img_w = int(s.shape[1]), int(s.shape[2])
            self._n_t = int(s.shape[0])

        # Pre-extract display params (constant for dialog lifetime).
        self._colors, self._enabled, self._lut = self._extract_display_params()

        # ── Size ──────────────────────────────────────────────────────────────
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            sz = screen.size()
            self.resize(int(sz.width() * 0.90), int(sz.height() * 0.90))
        else:
            self.resize(scaled(1400), scaled(900))

        self._build_ui()
        self._apply_style()

        if self._unit_queue:
            self._load_batch(0)
        else:
            self._lbl_progress.setText("No objects to review.")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        # ── Top bar: progress + mode selector ────────────────────────────────
        top_bar = QHBoxLayout()

        self._lbl_progress = QLabel("")
        self._lbl_progress.setStyleSheet(scale_qss(
            f"color: {Settings.FG_PRIMARY}; font: bold 11pt;"
        ))
        top_bar.addWidget(self._lbl_progress, stretch=1)

        # Metric selector for the per-cell trace (CellTracker's "Metric" combo).
        top_bar.addWidget(
            QLabel("Metric:", styleSheet=f"color: {Settings.FG_SECONDARY};")
        )
        self._combo_metric = QComboBox()
        self._combo_metric.addItems([label for (label, _k, _c) in self._metric_options])
        self._combo_metric.setEnabled(len(self._metric_options) > 1)
        self._combo_metric.currentIndexChanged.connect(self._on_metric_changed)
        top_bar.addWidget(self._combo_metric)
        top_bar.addSpacing(12)

        # Overlay choice — how the reviewed cell is drawn on its crop, using the
        # same segmentation overlay format (outline vs filled, color, weight).
        top_bar.addWidget(
            QLabel("Overlay:", styleSheet=f"color: {Settings.FG_SECONDARY};")
        )
        self._combo_overlay = QComboBox()
        self._combo_overlay.addItems(["Outline", "Filled"])
        self._combo_overlay.setCurrentIndex(0 if self._ov_outline else 1)
        self._combo_overlay.currentIndexChanged.connect(self._on_overlay_choice_changed)
        top_bar.addWidget(self._combo_overlay)
        self._btn_ov_color = QPushButton()
        self._btn_ov_color.setFixedSize(scaled(28), scaled(22))
        self._btn_ov_color.setToolTip("Overlay color")
        self._btn_ov_color.clicked.connect(self._on_overlay_color)
        self._refresh_ov_color_btn()
        top_bar.addWidget(self._btn_ov_color)
        self._spin_ov_weight = QSpinBox()
        self._spin_ov_weight.setRange(1, 12)
        self._spin_ov_weight.setValue(self._ov_weight)
        self._spin_ov_weight.setToolTip("Outline thickness (px)")
        self._spin_ov_weight.valueChanged.connect(self._on_overlay_choice_changed)
        top_bar.addWidget(self._spin_ov_weight)
        top_bar.addSpacing(12)

        top_bar.addWidget(
            QLabel("View:", styleSheet=f"color: {Settings.FG_SECONDARY};")
        )
        self._btn_mode: Dict[int, QPushButton] = {}
        for n, label in [(1, "1"), (2, "2"), (4, "4")]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedSize(scaled(32), scaled(26))
            btn.setObjectName("modeBtn")
            btn.clicked.connect(lambda checked, m=n: self._set_mode(m))
            self._btn_mode[n] = btn
            top_bar.addWidget(btn)
        self._btn_mode[1].setChecked(True)

        outer.addLayout(top_bar)

        # ── Canvas grid container ─────────────────────────────────────────────
        self._grid_container = QWidget()
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setSpacing(8)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._grid_container, stretch=1)

        # ── Shared playback row ───────────────────────────────────────────────
        t_row = QHBoxLayout()
        t_row.setSpacing(6)

        self._btn_play_all = QPushButton("▶ Play All")
        self._btn_play_all.setCheckable(True)
        self._btn_play_all.setMinimumWidth(scaled(90))
        self._btn_play_all.toggled.connect(self._on_play_toggled)
        t_row.addWidget(self._btn_play_all)

        t_row.addWidget(QLabel("T:"))

        self._slider_t = QSlider(Qt.Horizontal)
        self._slider_t.setMinimum(0)
        self._slider_t.setMaximum(max(0, self._n_t - 1))
        self._slider_t.setValue(0)
        self._slider_t.valueChanged.connect(self._on_t_changed)
        t_row.addWidget(self._slider_t, stretch=1)

        self._spin_t = QSpinBox()
        self._spin_t.setMinimum(0)
        self._spin_t.setMaximum(max(0, self._n_t - 1))
        self._spin_t.setFixedWidth(scaled(55))
        self._spin_t.valueChanged.connect(self._on_spin_t_changed)
        t_row.addWidget(self._spin_t)

        self._lbl_t_total = QLabel(f"/ {self._n_t}")
        self._lbl_t_total.setStyleSheet(f"color: {Settings.FG_SECONDARY};")
        t_row.addWidget(self._lbl_t_total)

        t_row.addSpacing(12)
        t_row.addWidget(QLabel("FPS:"))
        self._spin_fps = QDoubleSpinBox()
        self._spin_fps.setRange(0.5, 60.0)
        self._spin_fps.setValue(5.0)
        self._spin_fps.setSingleStep(0.5)
        self._spin_fps.setFixedWidth(scaled(65))
        self._spin_fps.valueChanged.connect(self._on_fps_changed)
        t_row.addWidget(self._spin_fps)

        outer.addLayout(t_row)

        # ── Bottom bar: Accept All / Reject All / Close ───────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self._btn_accept_all = QPushButton("Accept All")
        self._btn_accept_all.setObjectName("primaryBtn")
        self._btn_accept_all.setMinimumWidth(scaled(110))
        self._btn_accept_all.setToolTip("Accept all undecided objects in this batch.")
        self._btn_accept_all.clicked.connect(self._on_accept_all)
        btn_row.addWidget(self._btn_accept_all)

        self._btn_reject_all = QPushButton("Reject All")
        self._btn_reject_all.setObjectName("rejectBtn")
        self._btn_reject_all.setMinimumWidth(scaled(110))
        self._btn_reject_all.setToolTip("Reject all undecided objects in this batch.")
        self._btn_reject_all.clicked.connect(self._on_reject_all)
        btn_row.addWidget(self._btn_reject_all)

        btn_row.addStretch(1)

        self._btn_close = QPushButton("Close")
        self._btn_close.setMinimumWidth(scaled(90))
        self._btn_close.clicked.connect(self.reject)
        btn_row.addWidget(self._btn_close)

        outer.addLayout(btn_row)

    def _apply_style(self) -> None:
        self.setStyleSheet(scale_qss(f"""
            QWidget {{
                background: {Settings.BG_PRIMARY};
                color: {Settings.FG_PRIMARY};
                font: 9pt;
            }}
            QSlider::groove:horizontal {{
                height: 4px;
                background: {Settings.BORDER_COLOR};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {Settings.ACCENT_PURPLE};
                width: 12px; height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }}
            QSpinBox, QDoubleSpinBox {{
                background: {Settings.BG_SECONDARY};
                border: 1px solid {Settings.BORDER_COLOR};
                border-radius: 3px;
                padding: 2px 4px;
            }}
            QPushButton {{
                background: {Settings.BG_SECONDARY};
                color: {Settings.FG_PRIMARY};
                border: 1px solid {Settings.BORDER_COLOR};
                border-radius: 4px;
                padding: 4px 10px;
            }}
            QPushButton:hover {{ background: {Settings.BG_HOVER}; }}
            QPushButton#primaryBtn {{
                background: {Settings.ACCENT_PURPLE};
                color: #ffffff;
                border: none;
            }}
            QPushButton#primaryBtn:hover {{ background: {Settings.ACCENT_PINK}; }}
            QPushButton#rejectBtn {{
                background: #8b1a1a;
                color: #f8f8f2;
                border: none;
            }}
            QPushButton#rejectBtn:hover {{ background: #b22222; }}
            QPushButton#modeBtn {{
                padding: 2px 4px;
                font: bold 9pt;
            }}
            QPushButton#modeBtn:checked {{
                background: {Settings.ACCENT_PURPLE};
                color: #ffffff;
                border: none;
            }}
        """))

    # ── Mode control ──────────────────────────────────────────────────────────

    def _set_mode(self, mode: int) -> None:
        """Switch to 1-up, 2-up, or 4-up layout."""
        if mode == self._mode:
            return
        self._stop_playback()
        self._mode = mode

        # Update mode button checked states.
        for n, btn in self._btn_mode.items():
            btn.blockSignals(True)
            btn.setChecked(n == mode)
            btn.blockSignals(False)

        # Realign batch_start to new mode boundary so we don't skip units.
        self._batch_start = (self._batch_start // mode) * mode

        self._load_batch(self._batch_start)

    # ── Batch loading ─────────────────────────────────────────────────────────

    def _load_batch(self, batch_start: int) -> None:
        """Render and display the next up-to-`_mode` units from `batch_start`."""
        self._stop_playback()
        self._batch_start = batch_start

        # Build uids for this batch (may be fewer than _mode at end of queue).
        batch_uids: List[str] = []
        for i in range(self._mode):
            idx = batch_start + i
            if idx < len(self._unit_queue):
                batch_uids.append(self._unit_queue[idx])

        # Update progress label.
        total = len(self._unit_queue)
        if batch_uids:
            end_num = min(batch_start + self._mode, total)
            self._lbl_progress.setText(
                f"Objects {batch_start + 1}–{end_num} of {total}"
            )
        else:
            self._lbl_progress.setText("All objects reviewed.")

        # Rebuild grid panels.
        self._rebuild_panels(len(batch_uids) if batch_uids else 0)

        # Pre-render crops and load into panels.
        self._panel_crops = []
        for i, uid in enumerate(batch_uids):
            unit = self._units_by_uid[uid]
            track_rows = unit.rows

            r0, r1, c0, c1 = self._compute_crop(track_rows)
            frame_to_row: Dict[int, Dict[str, Any]] = {
                int(rr.get("frame", 0)): rr for rr in track_rows
            }
            crops = [
                self._render_one_frame(t, r0, r1, c0, c1, frame_to_row)
                for t in range(self._n_t)
            ]
            self._panel_crops.append(crops)

            info_texts = self._info_texts_for_unit(frame_to_row)
            trace = self._series_for_unit(unit)

            panel = self._panels[i]
            # Preserve already-made decisions (mode switch preserves state).
            panel.load(uid, crops, unit.label_text,
                       info_texts=info_texts, trace=trace)
            if uid in self._accepted_uids:
                panel.mark_decided("accepted")
            elif uid in self._rejected_uids:
                panel.mark_decided("rejected")

        # Reset T slider.
        self._slider_t.blockSignals(True)
        self._spin_t.blockSignals(True)
        self._slider_t.setValue(0)
        self._spin_t.setValue(0)
        self._slider_t.blockSignals(False)
        self._spin_t.blockSignals(False)
        self._show_all_panels(0)

    def _rebuild_panels(self, n_active: int) -> None:
        """Clear the grid and (re)create panels for the current mode."""
        # Detach old panels.
        for panel in self._panels:
            self._grid_layout.removeWidget(panel)
            panel.setParent(None)
            panel.deleteLater()
        self._panels = []

        if n_active == 0:
            return

        n_cols = 2 if self._mode == 4 else self._mode
        n_rows = 2 if self._mode == 4 else 1

        # Reset all stretch factors to 0 first, then set only the active cells.
        for row in range(4):
            self._grid_layout.setRowStretch(row, 0)
        for col in range(4):
            self._grid_layout.setColumnStretch(col, 0)
        for row in range(n_rows):
            self._grid_layout.setRowStretch(row, 1)
        for col in range(n_cols):
            self._grid_layout.setColumnStretch(col, 1)

        for i in range(self._mode):
            r, c = divmod(i, n_cols)
            panel = _TrackPanel(i, self._grid_container)
            panel.accept_clicked.connect(self._on_panel_accepted)
            panel.reject_clicked.connect(self._on_panel_rejected)
            if i >= n_active:
                panel.clear()
            self._grid_layout.addWidget(panel, r, c)
            self._panels.append(panel)

    # ── Per-panel decision handling ───────────────────────────────────────────

    def _on_panel_accepted(self, panel_idx: int) -> None:
        uid = self._panels[panel_idx].uid
        if uid is None:
            return
        self._accepted_uids.add(uid)
        self._rejected_uids.discard(uid)
        self._panels[panel_idx].mark_decided("accepted")
        self._check_batch_complete()

    def _on_panel_rejected(self, panel_idx: int) -> None:
        uid = self._panels[panel_idx].uid
        if uid is None:
            return
        self._rejected_uids.add(uid)
        self._accepted_uids.discard(uid)
        self._panels[panel_idx].mark_decided("rejected")
        self._check_batch_complete()

    def _on_accept_all(self) -> None:
        """Accept every undecided panel in the current batch."""
        any_decided = False
        for panel in self._panels:
            if panel.uid is not None and panel.decided is None:
                self._accepted_uids.add(panel.uid)
                panel.mark_decided("accepted")
                any_decided = True
        if any_decided:
            QTimer.singleShot(400, self._advance_batch)

    def _on_reject_all(self) -> None:
        """Reject every undecided panel in the current batch."""
        any_decided = False
        for panel in self._panels:
            if panel.uid is not None and panel.decided is None:
                self._rejected_uids.add(panel.uid)
                panel.mark_decided("rejected")
                any_decided = True
        if any_decided:
            QTimer.singleShot(400, self._advance_batch)

    def _check_batch_complete(self) -> None:
        """Advance to next batch once every active panel has a decision."""
        for panel in self._panels:
            if panel.uid is not None and panel.decided is None:
                return   # still undecided panels remain
        # Short pause so the user can see the final decision colours, then advance.
        QTimer.singleShot(400, self._advance_batch)

    def _advance_batch(self) -> None:
        next_start = self._batch_start + self._mode
        if next_start >= len(self._unit_queue):
            self.accept()   # all units reviewed
            return
        self._load_batch(next_start)

    # ── Per-cell metric / detail ────────────────────────────────────────────────

    @staticmethod
    def _build_metric_options(
        measurements: List[Dict[str, Any]],
    ) -> List[Tuple[str, str, str]]:
        """(label, kind, column) options for the per-cell trace, derived from the
        columns actually present: one per intensity channel, plus area and (if
        measured) eccentricity. Mirrors CellTracker's Results-page Metric combo."""
        chans: Set[str] = set()
        has_ecc = False
        for r in measurements:
            for k in r.keys():
                if k.startswith("mean_intensity_"):
                    chans.add(k[len("mean_intensity_"):])
            if r.get("eccentricity") is not None:
                has_ecc = True
        opts: List[Tuple[str, str, str]] = [
            (f"Intensity: {ch}", "intensity", f"mean_intensity_{ch}")
            for ch in sorted(chans)
        ]
        opts.append(("Area (px²)", "area", "area_px"))
        if has_ecc:
            opts.append(("Eccentricity", "ecc", "eccentricity"))
        return opts

    def _on_metric_changed(self, idx: int) -> None:
        if 0 <= idx < len(self._metric_options) and idx != self._metric_idx:
            self._metric_idx = idx
            self._load_batch(self._batch_start)   # recompute traces + info text

    # ── Overlay choice (segmentation-format cell rendering) ─────────────────
    def _refresh_ov_color_btn(self) -> None:
        r, g, b = self._ov_color
        self._btn_ov_color.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); border: 1px solid #888;")

    def _on_overlay_color(self) -> None:
        r, g, b = self._ov_color
        col = QColorDialog.getColor(QColor(r, g, b), self, "Overlay color")
        if col.isValid():
            self._ov_color = (col.red(), col.green(), col.blue())
            self._refresh_ov_color_btn()
            self._load_batch(self._batch_start)

    def _on_overlay_choice_changed(self, *_args) -> None:
        self._ov_outline = (self._combo_overlay.currentIndex() == 0)
        self._ov_weight = int(self._spin_ov_weight.value())
        self._load_batch(self._batch_start)   # re-render crops with the new style

    def _series_for_unit(
        self, unit: "_ReviewUnit",
    ) -> Tuple[List[int], List[float], str]:
        """(frames, values, ylabel) for the current metric over the unit's rows."""
        label, _kind, col = self._metric_options[self._metric_idx]
        frames: List[int] = []
        values: List[float] = []
        for r in sorted(unit.rows, key=lambda rr: int(rr.get("frame", 0))):
            v = r.get(col)
            if v is None:
                continue
            frames.append(int(r.get("frame", 0)))
            values.append(float(v))
        return frames, values, label

    def _info_texts_for_unit(
        self, frame_to_row: Dict[int, Dict[str, Any]],
    ) -> List[str]:
        """Per-frame cell-detail text (frame · area · ecc · intensity)."""
        int_col = next(
            (c for (_l, k, c) in self._metric_options if k == "intensity"), None)
        texts: List[str] = []
        for t in range(self._n_t):
            row = frame_to_row.get(t)
            if row is None:
                texts.append(f"frame {t + 1}/{self._n_t} — not in track")
                continue
            parts = [f"frame {t + 1}/{self._n_t}"]
            area = row.get("area_px")
            if area is not None:
                parts.append(f"{int(float(area))} px²")
            ecc = row.get("eccentricity")
            if ecc is not None:
                parts.append(f"ecc {float(ecc):.2f}")
            if int_col is not None and row.get(int_col) is not None:
                ch = int_col[len("mean_intensity_"):]
                parts.append(f"{ch} {float(row[int_col]):.0f}")
            texts.append("  ·  ".join(parts))
        return texts

    # ── Crop computation ──────────────────────────────────────────────────────

    def _compute_crop(self, track_rows: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
        """Return (r0, r1, c0, c1) — CROP_FACTOR × union bbox, clamped."""
        H, W = max(self._img_h, 1), max(self._img_w, 1)
        if not track_rows:
            return 0, H, 0, W

        min_r = min(int(r.get("bbox_min_row", 0)) for r in track_rows)
        max_r = max(int(r.get("bbox_max_row", H)) for r in track_rows)
        min_c = min(int(r.get("bbox_min_col", 0)) for r in track_rows)
        max_c = max(int(r.get("bbox_max_col", W)) for r in track_rows)

        bh = max(max_r - min_r, 1)
        bw = max(max_c - min_c, 1)
        center_r = (min_r + max_r) / 2.0
        center_c = (min_c + max_c) / 2.0
        half = _CROP_FACTOR / 2.0   # 1.5 for 3×

        r0 = max(0, int(center_r - bh * half))
        r1 = min(H, int(center_r + bh * half))
        c0 = max(0, int(center_c - bw * half))
        c1 = min(W, int(center_c + bw * half))

        if r1 - r0 < 4 or c1 - c0 < 4:
            return 0, H, 0, W
        return r0, r1, c0, c1

    # ── Frame rendering ───────────────────────────────────────────────────────

    def _extract_display_params(
        self,
    ) -> Tuple[
        Dict[str, Tuple[int, int, int]],
        Dict[str, bool],
        Dict[str, Tuple[float, float, float]],
    ]:
        colors: Dict[str, Tuple[int, int, int]] = {}
        enabled: Dict[str, bool] = {}
        lut_settings: Dict[str, Tuple[float, float, float]] = {}
        for name in self._channels:
            cd = self._channel_display.get(name, {})
            color_name = cd.get("color", "gray")
            if isinstance(color_name, (list, tuple)) and len(color_name) == 3:
                colors[name] = (int(color_name[0]), int(color_name[1]), int(color_name[2]))
            else:
                colors[name] = CHANNEL_COLORS.get(str(color_name), (255, 255, 255))
            enabled[name] = bool(cd.get("enabled", True))
            if "lut_lo" in cd and "lut_hi" in cd:
                lut_settings[name] = (
                    float(cd["lut_lo"]),
                    float(cd["lut_hi"]),
                    float(cd.get("lut_gamma", 1.0)),
                )
        return colors, enabled, lut_settings

    def _render_one_frame(
        self,
        t: int,
        r0: int, r1: int, c0: int, c1: int,
        frame_to_row: Dict[int, Dict[str, Any]],
    ) -> np.ndarray:
        if not self._channels:
            return np.zeros((max(r1 - r0, 1), max(c1 - c0, 1), 3), dtype=np.uint8)

        frame_dict: Dict[str, np.ndarray] = {}
        for name, arr in self._channels.items():
            t_idx = min(t, arr.shape[0] - 1)
            frame_dict[name] = arr[t_idx, r0:r1, c0:c1]

        try:
            rgb = _composite_frame(frame_dict, self._colors, self._enabled, self._lut)
        except Exception:
            rgb = np.zeros((r1 - r0, c1 - c0, 3), dtype=np.uint8)

        row = frame_to_row.get(t)
        if row is not None:
            seg_ch = str(row.get("segmentation_channel", ""))
            label_id = row.get("label_id")
            mask_arr = self._label_masks.get(seg_ch)
            if mask_arr is not None and label_id is not None:
                t_idx = min(t, mask_arr.shape[0] - 1)
                mask_crop = mask_arr[t_idx, r0:r1, c0:c1]
                cell = (mask_crop == int(label_id))
                if cell.any():
                    # Render the cell with the SAME segmentation overlay format
                    # (outline / filled, color, weight) via the shared helper, per
                    # the dialog's overlay choice.
                    from nd2studios.pages.analysis_page import _overlay_labels
                    single = cell.astype(np.int32)
                    rgb = _overlay_labels(
                        rgb, single,
                        alpha=1.0 if self._ov_outline else 0.5,
                        color=self._ov_color,
                        outline=self._ov_outline,
                        thickness=self._ov_weight)
        return rgb

    # ── Playback ──────────────────────────────────────────────────────────────

    def _show_all_panels(self, t: int) -> None:
        for i, panel in enumerate(self._panels):
            if i < len(self._panel_crops) and self._panel_crops[i]:
                panel.show_frame(t)

    def _stop_playback(self) -> None:
        self._play_timer.stop()
        self._btn_play_all.blockSignals(True)
        self._btn_play_all.setChecked(False)
        self._btn_play_all.setText("▶ Play All")
        self._btn_play_all.blockSignals(False)

    def _on_t_changed(self, value: int) -> None:
        self._spin_t.blockSignals(True)
        self._spin_t.setValue(value)
        self._spin_t.blockSignals(False)
        self._show_all_panels(value)

    def _on_spin_t_changed(self, value: int) -> None:
        self._slider_t.blockSignals(True)
        self._slider_t.setValue(value)
        self._slider_t.blockSignals(False)
        self._show_all_panels(value)

    def _on_play_toggled(self, on: bool) -> None:
        if on:
            self._btn_play_all.setText("⏸ Pause All")
            fps = max(0.5, self._spin_fps.value())
            self._play_timer.start(int(1000 / fps))
        else:
            self._btn_play_all.setText("▶ Play All")
            self._play_timer.stop()

    def _on_play_tick(self) -> None:
        max_t = max(0, self._n_t - 1)
        next_t = (self._slider_t.value() + 1) % (max_t + 1)
        self._slider_t.setValue(next_t)

    def _on_fps_changed(self, fps: float) -> None:
        if self._play_timer.isActive():
            self._play_timer.start(int(1000 / max(0.5, fps)))

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._play_timer.stop()
        super().closeEvent(event)

    def done(self, code: int) -> None:
        self._play_timer.stop()
        super().done(int(code))

    # ── Public results ────────────────────────────────────────────────────────

    def _rows_for(self, uids: Set[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for uid in uids:
            unit = self._units_by_uid.get(uid)
            if unit is not None:
                out.extend(unit.rows)
        return out

    @property
    def rejected_rows(self) -> List[Dict[str, Any]]:
        """The actual measurement-row dicts the user rejected (references)."""
        return self._rows_for(self._rejected_uids)

    @property
    def accepted_rows(self) -> List[Dict[str, Any]]:
        """The actual measurement-row dicts the user accepted (references)."""
        return self._rows_for(self._accepted_uids)

    @property
    def rejected_track_ids(self) -> Set[int]:
        return {u.track_id for uid in self._rejected_uids
                if (u := self._units_by_uid.get(uid)) is not None
                and u.track_id is not None}

    @property
    def accepted_track_ids(self) -> Set[int]:
        return {u.track_id for uid in self._accepted_uids
                if (u := self._units_by_uid.get(uid)) is not None
                and u.track_id is not None}
