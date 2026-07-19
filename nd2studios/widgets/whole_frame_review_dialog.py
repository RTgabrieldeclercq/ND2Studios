"""
WholeFrameReviewDialog — review objects on whole frames.

A single viewer that composites the current ``(M, T)`` frame and draws every
object's **label id in white at its centroid**.  The user steps through **T**
with the slider (and switches **M** with the position dropdown for
multi-position files; Z is collapsed at load).  Decisions:

* **click an object** (its label / mask) to toggle reject (white → red);
* **Accept Frame** / **Reject Frame** to decide every object on the current
  frame at once.

Decided objects are exposed as row references (``accepted_rows`` /
``rejected_rows``), matching :class:`TrackValidationDialog`, so the Run / preview
paths apply both review modes the same way.  Image channels are fetched lazily
per position via a provider callable, so only the viewed M is materialized.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QRect, QTimer
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
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
from nd2studios.widgets.icon_button import scale_qss, scaled_pt, scaled
from nd2studios.widgets.image_viewer import ImageCanvas

# Click hit-test radius (image px) when the click misses every label mask.
_CLICK_RADIUS_PX = 25.0

# Corner inset: fraction of the smaller canvas dimension, and crop padding factor
# (square crop = max bbox span of the selected track × this factor).
_INSET_FRAC = 0.30
_INSET_PAD = 1.6
# Outline colour drawn around the selected cell in the corner inset (cyan).
_INSET_OUTLINE = (80, 220, 255)


class WholeFrameReviewDialog(QDialog):
    """Whole-frame object review with click-reject + per-frame accept/reject."""

    def __init__(
        self,
        rows_by_m: Dict[int, List[Dict[str, Any]]],
        masks_by_m: Dict[int, Dict[str, np.ndarray]],
        channel_provider: Callable[[int], Dict[str, np.ndarray]],
        channel_display: Dict[str, Dict[str, Any]],
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Parameters
        ----------
        rows_by_m:
            {m: [measurement rows]} — the actual row dicts (decisions are
            reported as references back into these).
        masks_by_m:
            {m: {seg_channel: (T, H, W) int32}} — label arrays for hit-testing.
        channel_provider:
            ``m -> {channel: (T, H, W)}``; called once per position and cached.
        channel_display:
            exp.channel_display — color / LUT / enabled per channel.
        """
        super().__init__(parent)
        self.setWindowTitle("Review Objects — Whole frame")
        self.setModal(True)

        self._rows_by_m = rows_by_m
        self._masks_by_m = masks_by_m
        self._provider = channel_provider
        self._channel_display = channel_display

        self._m_list: List[int] = sorted(
            set(rows_by_m.keys()) | set(masks_by_m.keys()))
        if not self._m_list:
            self._m_list = [0]
        self._m: int = self._m_list[0]
        self._t: int = 0

        # Decisions keyed by id(row) -> "accepted" | "rejected".
        self._decision: Dict[int, str] = {}
        self._row_by_id: Dict[int, Dict[str, Any]] = {}
        # Per-(m, t) index: {(m, t): {(channel, label_id): row}} for hit-testing.
        self._rows_at: Dict[Tuple[int, int], Dict[Tuple[str, int], Dict[str, Any]]] = \
            defaultdict(dict)
        # Per-track index: {m: {track_id: {t: row}}} for the corner cell-over-time view.
        self._track_index: Dict[int, Dict[int, Dict[int, Dict[str, Any]]]] = \
            defaultdict(lambda: defaultdict(dict))
        for m, rows in rows_by_m.items():
            for r in rows:
                self._row_by_id[id(r)] = r
                key = (str(r.get("segmentation_channel", "")), r.get("label_id"))
                if key[1] is not None:
                    self._rows_at[(m, int(r.get("frame", 0)))][
                        (key[0], int(key[1]))] = r
                tid = r.get("track_id")
                if tid is not None:
                    self._track_index[m][int(tid)][int(r.get("frame", 0))] = r

        # Corner inset selection: a tracked cell (track id) or a single untracked
        # row (by id()). Only one is set at a time; both None hides the inset.
        self._sel_track_id: Optional[int] = None
        self._sel_row_id: Optional[int] = None
        # Latest full composite RGB (set in _render); the inset crops from it.
        self._last_rgb: Optional[np.ndarray] = None

        self._chan_cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)

        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            sz = screen.size()
            self.resize(int(sz.width() * 0.85), int(sz.height() * 0.88))
        else:
            self.resize(scaled(1200), scaled(850))

        self._build_ui()
        self._apply_style()
        self._refresh_t_range()
        self._render()

    # ── data helpers ────────────────────────────────────────────────────────

    def _channels(self, m: int) -> Dict[str, np.ndarray]:
        if m not in self._chan_cache:
            try:
                self._chan_cache[m] = self._provider(m) or {}
            except Exception:  # noqa: BLE001
                self._chan_cache[m] = {}
        return self._chan_cache[m]

    def _n_t_for(self, m: int) -> int:
        for arr in self._channels(m).values():
            return int(arr.shape[0])
        for arr in self._masks_by_m.get(m, {}).values():
            return int(np.asarray(arr).shape[0])
        return 1

    def _display_params(self, channels: Dict[str, np.ndarray]):
        colors: Dict[str, Tuple[int, int, int]] = {}
        enabled: Dict[str, bool] = {}
        lut: Dict[str, Tuple[float, float, float]] = {}
        for name in channels:
            cd = self._channel_display.get(name, {})
            cn = cd.get("color", "gray")
            if isinstance(cn, (list, tuple)) and len(cn) == 3:
                colors[name] = (int(cn[0]), int(cn[1]), int(cn[2]))
            else:
                colors[name] = CHANNEL_COLORS.get(str(cn), (255, 255, 255))
            enabled[name] = bool(cd.get("enabled", True))
            if "lut_lo" in cd and "lut_hi" in cd:
                lut[name] = (float(cd["lut_lo"]), float(cd["lut_hi"]),
                             float(cd.get("lut_gamma", 1.0)))
        return colors, enabled, lut

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        top = QHBoxLayout()
        self._lbl_progress = QLabel("")
        self._lbl_progress.setStyleSheet(scale_qss(
            f"color: {Settings.FG_PRIMARY}; font: bold 11pt;"))
        top.addWidget(self._lbl_progress, stretch=1)

        if len(self._m_list) > 1:
            top.addWidget(QLabel("Position:",
                                 styleSheet=f"color: {Settings.FG_SECONDARY};"))
            self._combo_m = QComboBox()
            for m in self._m_list:
                self._combo_m.addItem(f"M{m + 1}", m)
            self._combo_m.currentIndexChanged.connect(self._on_m_changed)
            top.addWidget(self._combo_m)
        else:
            self._combo_m = None
        outer.addLayout(top)

        self._canvas = ImageCanvas(self)
        self._canvas.setMinimumSize(scaled(320), scaled(320))
        self._canvas.set_overlay(self._paint_overlay)
        self._canvas.clicked.connect(self._on_canvas_clicked)
        outer.addWidget(self._canvas, stretch=1)

        # Playback row.
        t_row = QHBoxLayout()
        t_row.setSpacing(6)
        self._btn_play = QPushButton("▶ Play")
        self._btn_play.setCheckable(True)
        self._btn_play.setMinimumWidth(scaled(80))
        self._btn_play.toggled.connect(self._on_play_toggled)
        t_row.addWidget(self._btn_play)
        t_row.addWidget(QLabel("T:"))
        self._slider_t = QSlider(Qt.Horizontal)
        self._slider_t.setMinimum(0)
        self._slider_t.valueChanged.connect(self._on_t_changed)
        t_row.addWidget(self._slider_t, stretch=1)
        self._spin_t = QSpinBox()
        self._spin_t.setFixedWidth(scaled(55))
        self._spin_t.valueChanged.connect(self._on_spin_t_changed)
        t_row.addWidget(self._spin_t)
        self._lbl_t_total = QLabel("")
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

        # Decision row.
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        hint = QLabel("Click a cell to inspect it over time  •  Alt-click to toggle reject.")
        hint.setStyleSheet(scale_qss(f"color: {Settings.FG_SECONDARY}; font: 8pt italic;"))
        btn_row.addWidget(hint)
        btn_row.addStretch(1)
        # Track-level actions — visible only while a cell is selected.
        self._btn_accept_track = QPushButton("Accept Track")
        self._btn_accept_track.setObjectName("primaryBtn")
        self._btn_accept_track.setMinimumWidth(scaled(110))
        self._btn_accept_track.setToolTip("Accept every frame of the selected cell/track.")
        self._btn_accept_track.clicked.connect(lambda: self._decide_track("accepted"))
        self._btn_accept_track.setVisible(False)
        btn_row.addWidget(self._btn_accept_track)
        self._btn_reject_track = QPushButton("Reject Track")
        self._btn_reject_track.setObjectName("rejectBtn")
        self._btn_reject_track.setMinimumWidth(scaled(110))
        self._btn_reject_track.setToolTip("Reject every frame of the selected cell/track.")
        self._btn_reject_track.clicked.connect(lambda: self._decide_track("rejected"))
        self._btn_reject_track.setVisible(False)
        btn_row.addWidget(self._btn_reject_track)
        btn_row.addSpacing(20)
        self._btn_accept_frame = QPushButton("Accept Frame")
        self._btn_accept_frame.setObjectName("primaryBtn")
        self._btn_accept_frame.setMinimumWidth(scaled(120))
        self._btn_accept_frame.setToolTip("Accept every object on this frame.")
        self._btn_accept_frame.clicked.connect(lambda: self._decide_frame("accepted"))
        btn_row.addWidget(self._btn_accept_frame)
        self._btn_reject_frame = QPushButton("Reject Frame")
        self._btn_reject_frame.setObjectName("rejectBtn")
        self._btn_reject_frame.setMinimumWidth(scaled(120))
        self._btn_reject_frame.setToolTip("Reject every object on this frame.")
        self._btn_reject_frame.clicked.connect(lambda: self._decide_frame("rejected"))
        btn_row.addWidget(self._btn_reject_frame)
        btn_row.addSpacing(20)
        self._btn_done = QPushButton("Done")
        self._btn_done.setMinimumWidth(scaled(90))
        self._btn_done.clicked.connect(self.accept)
        btn_row.addWidget(self._btn_done)
        outer.addLayout(btn_row)

    def _apply_style(self) -> None:
        self.setStyleSheet(scale_qss(f"""
            QWidget {{
                background: {Settings.BG_PRIMARY};
                color: {Settings.FG_PRIMARY};
                font: 9pt;
            }}
            QSlider::groove:horizontal {{
                height: 4px; background: {Settings.BORDER_COLOR}; border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {Settings.ACCENT_PURPLE};
                width: 12px; height: 12px; margin: -4px 0; border-radius: 6px;
            }}
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background: {Settings.BG_SECONDARY};
                border: 1px solid {Settings.BORDER_COLOR};
                border-radius: 3px; padding: 2px 4px;
            }}
            QPushButton {{
                background: {Settings.BG_SECONDARY}; color: {Settings.FG_PRIMARY};
                border: 1px solid {Settings.BORDER_COLOR};
                border-radius: 4px; padding: 4px 10px;
            }}
            QPushButton:hover {{ background: {Settings.BG_HOVER}; }}
            QPushButton#primaryBtn {{
                background: {Settings.ACCENT_PURPLE}; color: #ffffff; border: none;
            }}
            QPushButton#primaryBtn:hover {{ background: {Settings.ACCENT_PINK}; }}
            QPushButton#rejectBtn {{
                background: #8b1a1a; color: #f8f8f2; border: none;
            }}
            QPushButton#rejectBtn:hover {{ background: #b22222; }}
        """))

    # ── navigation ──────────────────────────────────────────────────────────

    def _refresh_t_range(self) -> None:
        n_t = self._n_t_for(self._m)
        self._t = max(0, min(self._t, n_t - 1))
        for w in (self._slider_t, self._spin_t):
            w.blockSignals(True)
            w.setMaximum(max(0, n_t - 1))
            w.setValue(self._t)
            w.blockSignals(False)
        self._lbl_t_total.setText(f"/ {n_t}")

    def _on_m_changed(self, idx: int) -> None:
        if self._combo_m is None:
            return
        self._stop_playback()
        self._m = int(self._combo_m.itemData(idx))
        self._clear_selection()
        self._refresh_t_range()
        self._render()

    def _on_t_changed(self, value: int) -> None:
        self._t = int(value)
        self._spin_t.blockSignals(True)
        self._spin_t.setValue(value)
        self._spin_t.blockSignals(False)
        self._render()

    def _on_spin_t_changed(self, value: int) -> None:
        self._t = int(value)
        self._slider_t.blockSignals(True)
        self._slider_t.setValue(value)
        self._slider_t.blockSignals(False)
        self._render()

    def _on_play_toggled(self, on: bool) -> None:
        if on:
            self._btn_play.setText("⏸ Pause")
            self._play_timer.start(int(1000 / max(0.5, self._spin_fps.value())))
        else:
            self._btn_play.setText("▶ Play")
            self._play_timer.stop()

    def _on_play_tick(self) -> None:
        n_t = self._n_t_for(self._m)
        self._slider_t.setValue((self._t + 1) % max(1, n_t))

    def _on_fps_changed(self, fps: float) -> None:
        if self._play_timer.isActive():
            self._play_timer.start(int(1000 / max(0.5, fps)))

    def _stop_playback(self) -> None:
        self._play_timer.stop()
        self._btn_play.blockSignals(True)
        self._btn_play.setChecked(False)
        self._btn_play.setText("▶ Play")
        self._btn_play.blockSignals(False)

    # ── rendering ─────────────────────────────────────────────────────────────

    def _render(self) -> None:
        channels = self._channels(self._m)
        frame: Dict[str, np.ndarray] = {}
        h = w = 0
        for name, arr in channels.items():
            t_idx = min(self._t, arr.shape[0] - 1)
            frame[name] = arr[t_idx]
            h, w = arr.shape[1], arr.shape[2]
        if frame:
            colors, enabled, lut = self._display_params(channels)
            try:
                rgb = _composite_frame(frame, colors, enabled, lut)
            except Exception:  # noqa: BLE001
                rgb = np.zeros((max(h, 1), max(w, 1), 3), dtype=np.uint8)
        else:
            rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        self._last_rgb = rgb
        self._canvas.set_image(rgb)
        self._canvas.update()
        n_obj = len(self._rows_at.get((self._m, self._t), {}))
        n_rej = sum(1 for r in self._rows_by_m.get(self._m, [])
                    if int(r.get("frame", 0)) == self._t
                    and self._decision.get(id(r)) == "rejected")
        self._lbl_progress.setText(
            f"M{self._m + 1}  •  T{self._t + 1}  •  {n_obj} object(s)"
            + (f"  •  {n_rej} rejected" if n_rej else ""))

    def _paint_overlay(self, painter, scale: float, ox: int, oy: int,
                       pw: int, ph: int) -> None:
        """Draw each object's label id at its centroid (white / red / green)."""
        rows = self._rows_at.get((self._m, self._t), {})
        has_selection = self._sel_track_id is not None or self._sel_row_id is not None
        if not rows and not has_selection:
            return
        font = QFont("Helvetica Neue")
        font.setPointSizeF(scaled_pt(9))
        font.setBold(True)
        painter.setFont(font)
        sel = self._selected_row_at_current()
        for (_ch, label_id), r in rows.items():
            cy = float(r.get("centroid_y_px") or 0.0)
            cx = float(r.get("centroid_x_px") or 0.0)
            wx = ox + cx * scale
            wy = oy + cy * scale
            decision = self._decision.get(id(r))
            if decision == "rejected":
                col = QColor(255, 85, 85)
            elif decision == "accepted":
                col = QColor(80, 250, 123)
            else:
                col = QColor(255, 255, 255)
            painter.setPen(QPen(col, 1.5))
            painter.drawEllipse(int(wx) - 2, int(wy) - 2, 4, 4)
            painter.drawText(int(wx) + 4, int(wy) - 4, str(int(label_id)))
            # Ring the selected cell so it's clear which one feeds the inset.
            if r is sel:
                painter.setPen(QPen(QColor(*_INSET_OUTLINE), 2.0))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(int(wx) - 9, int(wy) - 9, 18, 18)

        if self._sel_track_id is not None or self._sel_row_id is not None:
            self._paint_corner_inset(painter, pw, ph)

    # ── decisions ───────────────────────────────────────────────────────────

    def _on_canvas_clicked(self, iy: float, ix: float) -> None:
        r = self._hit_test(iy, ix)
        mods = QGuiApplication.keyboardModifiers()
        reject_click = bool(mods & (Qt.KeyboardModifier.AltModifier
                                    | Qt.KeyboardModifier.ControlModifier))
        if r is None:
            # Empty space: a plain click dismisses the inset.
            if not reject_click and (self._sel_track_id is not None
                                     or self._sel_row_id is not None):
                self._clear_selection()
                self._canvas.update()
            return
        if reject_click:
            # Toggle reject: undecided/accepted -> rejected, rejected -> undecided.
            if self._decision.get(id(r)) == "rejected":
                self._decision.pop(id(r), None)
            else:
                self._decision[id(r)] = "rejected"
            self._canvas.update()
            self._render_status_only()
            return
        # Plain click: select the cell for the corner cell-over-time inset.
        tid = r.get("track_id")
        if tid is not None:
            self._sel_track_id = int(tid)
            self._sel_row_id = None
        else:
            self._sel_track_id = None
            self._sel_row_id = id(r)
        self._update_track_buttons()
        self._canvas.update()

    # ── corner cell-over-time inset ───────────────────────────────────────────

    def _clear_selection(self) -> None:
        self._sel_track_id = None
        self._sel_row_id = None
        self._update_track_buttons()

    def _update_track_buttons(self) -> None:
        on = self._sel_track_id is not None or self._sel_row_id is not None
        self._btn_accept_track.setVisible(on)
        self._btn_reject_track.setVisible(on)

    def _selected_rows(self) -> List[Dict[str, Any]]:
        """All rows of the current selection (track frames, or one untracked row)."""
        if self._sel_track_id is not None:
            return list(self._track_index.get(self._m, {})
                        .get(self._sel_track_id, {}).values())
        if self._sel_row_id is not None:
            r = self._row_by_id.get(self._sel_row_id)
            return [r] if r is not None else []
        return []

    def _selected_row_at_current(self) -> Optional[Dict[str, Any]]:
        """The selected cell's row on the current (m, t), or None on a gap frame."""
        if self._sel_track_id is not None:
            return self._track_index.get(self._m, {}).get(
                self._sel_track_id, {}).get(self._t)
        if self._sel_row_id is not None:
            r = self._row_by_id.get(self._sel_row_id)
            if r is not None and int(r.get("frame", 0)) == self._t:
                return r
        return None

    def _crop_px_for_selection(self) -> int:
        """Constant square crop size (image px) for the selection across frames."""
        span = 0
        for r in self._selected_rows():
            bh = int(r.get("bbox_max_row", 0)) - int(r.get("bbox_min_row", 0))
            bw = int(r.get("bbox_max_col", 0)) - int(r.get("bbox_min_col", 0))
            span = max(span, bh, bw)
        return max(24, int(span * _INSET_PAD))

    def _paint_corner_inset(self, painter, pw: int, ph: int) -> None:
        canvas_w, canvas_h = self._canvas.width(), self._canvas.height()
        side = max(80, int(min(canvas_w, canvas_h) * _INSET_FRAC))
        margin = 10
        rect = QRect(canvas_w - side - margin, margin, side, side)

        # Frame + dark backing.
        painter.fillRect(rect, QColor(0, 0, 0, 200))
        painter.setPen(QPen(QColor(*_INSET_OUTLINE), 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)

        font = QFont("Helvetica Neue")
        font.setPointSizeF(scaled_pt(8))
        font.setBold(True)
        painter.setFont(font)

        if self._sel_track_id is not None:
            title = f"Track {self._sel_track_id}"
        else:
            r0 = self._row_by_id.get(self._sel_row_id or 0)
            lbl = r0.get("label_id") if r0 else None
            title = f"Object {int(lbl)}" if lbl is not None else "Object"

        row = self._selected_row_at_current()
        if row is None or self._last_rgb is None:
            painter.setPen(QPen(QColor(200, 200, 200)))
            painter.drawText(rect.adjusted(6, 6, -6, -6),
                             int(Qt.AlignmentFlag.AlignCenter),
                             f"{title}\nno object at T{self._t + 1}")
            return

        crop_rgb = self._crop_for_row(row)
        if crop_rgb is None or crop_rgb.size == 0:
            return
        crop_rgb = np.ascontiguousarray(crop_rgb)
        crop_h, crop_w = crop_rgb.shape[0], crop_rgb.shape[1]
        img = QImage(crop_rgb.data, crop_w, crop_h, crop_w * 3,
                     QImage.Format.Format_RGB888)
        painter.drawImage(rect, img)
        # Re-draw the frame border over the image, plus a title strip.
        painter.setPen(QPen(QColor(*_INSET_OUTLINE), 1.5))
        painter.drawRect(rect)
        painter.fillRect(QRect(rect.left(), rect.top(), rect.width(), 16),
                         QColor(0, 0, 0, 170))
        area = row.get("area_px")
        label = f"{title}  •  T{self._t + 1}"
        if area is not None:
            label += f"  •  {int(float(area))}px²"
        painter.setPen(QPen(QColor(*_INSET_OUTLINE)))
        painter.drawText(rect.left() + 4, rect.top() + 12, label)

    def _crop_for_row(self, row: Dict[str, Any]) -> Optional[np.ndarray]:
        """Square crop of _last_rgb centred on the row, with its label outlined."""
        rgb = self._last_rgb
        if rgb is None:
            return None
        H, W = rgb.shape[0], rgb.shape[1]
        cy = float(row.get("centroid_y_px") or 0.0)
        cx = float(row.get("centroid_x_px") or 0.0)
        half = self._crop_px_for_selection() // 2
        r0 = int(round(cy)) - half
        c0 = int(round(cx)) - half
        side = half * 2
        # Build a fixed-size square canvas so the cell stays centred even at edges.
        out = np.zeros((side, side, 3), dtype=np.uint8)
        sr0, sr1 = max(0, r0), min(H, r0 + side)
        sc0, sc1 = max(0, c0), min(W, c0 + side)
        if sr1 <= sr0 or sc1 <= sc0:
            return out
        dr0, dc0 = sr0 - r0, sc0 - c0
        out[dr0:dr0 + (sr1 - sr0), dc0:dc0 + (sc1 - sc0)] = rgb[sr0:sr1, sc0:sc1]

        # Outline the selected label within the crop (reuse the analysis helper).
        seg_ch = str(row.get("segmentation_channel", ""))
        label_id = row.get("label_id")
        mask_arr = self._masks_by_m.get(self._m, {}).get(seg_ch)
        if mask_arr is not None and label_id is not None:
            a = np.asarray(mask_arr)
            t_idx = min(self._t, a.shape[0] - 1)
            mask_full = a[t_idx]
            mask_out = np.zeros((side, side), dtype=np.int32)
            mask_out[dr0:dr0 + (sr1 - sr0), dc0:dc0 + (sc1 - sc0)] = (
                mask_full[sr0:sr1, sc0:sc1] == int(label_id)).astype(np.int32)
            if mask_out.any():
                from nd2studios.pages.analysis_page import _overlay_labels
                out = _overlay_labels(out, mask_out, alpha=1.0,
                                      color=_INSET_OUTLINE, outline=True, thickness=2)
        return out

    def _decide_track(self, decision: str) -> None:
        rows = self._selected_rows()
        if not rows:
            return
        for r in rows:
            self._decision[id(r)] = decision
        self._canvas.update()
        self._render_status_only()

    def _hit_test(self, iy: float, ix: float) -> Optional[Dict[str, Any]]:
        """Object under the click — label-mask hit first, else nearest centroid."""
        yi, xi = int(round(iy)), int(round(ix))
        masks = self._masks_by_m.get(self._m, {})
        for ch, arr in masks.items():
            a = np.asarray(arr)
            t_idx = min(self._t, a.shape[0] - 1)
            if 0 <= yi < a.shape[1] and 0 <= xi < a.shape[2]:
                val = int(a[t_idx, yi, xi])
                if val > 0:
                    r = self._rows_at.get((self._m, self._t), {}).get((str(ch), val))
                    if r is not None:
                        return r
        # Fallback: nearest centroid within radius.
        best: Optional[Dict[str, Any]] = None
        best_d = _CLICK_RADIUS_PX
        for r in self._rows_at.get((self._m, self._t), {}).values():
            cy = float(r.get("centroid_y_px") or 0.0)
            cx = float(r.get("centroid_x_px") or 0.0)
            d = ((cy - iy) ** 2 + (cx - ix) ** 2) ** 0.5
            if d < best_d:
                best_d, best = d, r
        return best

    def _decide_frame(self, decision: str) -> None:
        for r in self._rows_at.get((self._m, self._t), {}).values():
            self._decision[id(r)] = decision
        self._canvas.update()
        self._render_status_only()

    def _render_status_only(self) -> None:
        n_obj = len(self._rows_at.get((self._m, self._t), {}))
        n_rej = sum(1 for r in self._rows_by_m.get(self._m, [])
                    if int(r.get("frame", 0)) == self._t
                    and self._decision.get(id(r)) == "rejected")
        self._lbl_progress.setText(
            f"M{self._m + 1}  •  T{self._t + 1}  •  {n_obj} object(s)"
            + (f"  •  {n_rej} rejected" if n_rej else ""))

    # ── cleanup / results ─────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._play_timer.stop()
        super().closeEvent(event)

    def done(self, code: int) -> None:
        self._play_timer.stop()
        super().done(int(code))

    @property
    def rejected_rows(self) -> List[Dict[str, Any]]:
        return [r for rid, r in self._row_by_id.items()
                if self._decision.get(rid) == "rejected"]

    @property
    def accepted_rows(self) -> List[Dict[str, Any]]:
        return [r for rid, r in self._row_by_id.items()
                if self._decision.get(rid) == "accepted"]
