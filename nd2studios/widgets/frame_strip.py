"""NIS-Elements-style rectangular frame strip (V1.43).

Replaces the plain ``QSlider`` for an axis (M / T / Z) with a row of
rectangular tiles that fill the full horizontal width of the viewer and wrap
to additional rows when there are more frames than fit at the minimum tile
width. One tile per frame.

Capabilities (per the QoL request):

* **Current frame** highlighted in the theme accent.
* **Selection** — click selects a single frame; **Shift-click** selects the
  inclusive range from the current frame; selected tiles render in a distinct
  highlight. Selection is a ``set[int]``.
* **Keyboard** (when focused): ``←``/``→`` step ±1, ``Shift``+arrow ±5,
  ``Ctrl``+arrow ±10. Documented inline via tooltip.
* **Right-click** on a selection emits :attr:`crop_requested` with the selected
  indices so the host can crop the dataset to them.
* **Per-frame metadata** shown as a tooltip via an injected ``meta_fn``.

Drawing is done in a single :meth:`paintEvent` rather than with N child
widgets, so thousands of T frames stay cheap.

The widget is deliberately decoupled from the data model: the host
(:class:`MultiAxisViewer`) keeps a (hidden) ``QSlider`` as the source of truth
for the current index and syncs it with the strip, so existing playback / value
code keeps working unchanged.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Set

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from nd2studios.core.settings import Settings


class FrameStrip(QWidget):
    """A wrapping strip of selectable rectangular frame tiles."""

    current_changed = Signal(int)          # user changed the current frame
    selection_changed = Signal(object)     # frozenset[int] of selected indices
    crop_requested = Signal(object)        # frozenset[int] — right-click → crop

    def __init__(self, axis_label: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._axis = axis_label
        self._count = 0
        self._current = 0
        self._selection: Set[int] = set()
        self._meta_fn: Optional[Callable[[int], str]] = None
        self._hover_idx = -1

        # DPI-aware sizing. A tile must be at least ~half a mouse cursor wide;
        # a default cursor is ~16 px logical, so ~9 px scaled is the floor.
        scale = self._dpi_scale()
        self._min_tile_w = max(8, int(round(9 * scale)))
        self._tile_h = max(14, int(round(18 * scale)))
        self._gap = max(1, int(round(2 * scale)))
        self._row_gap = max(1, int(round(3 * scale)))

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setToolTip(
            "Frames — click to select, Shift-click for a range.\n"
            "←/→ step 1 · Shift+←/→ step 5 · Ctrl+←/→ step 10.\n"
            "Right-click a selection to crop the data to it."
        )
        self._recompute_height()

    # ── DPI ──
    def _dpi_scale(self) -> float:
        try:
            dpi = float(self.logicalDpiX())
            if dpi > 0:
                return max(1.0, dpi / 96.0)
        except Exception:
            pass
        return 1.0

    # ── Model ──
    def set_count(self, n: int) -> None:
        """Set the number of frames (tiles). Clamps current/selection."""
        n = max(0, int(n))
        if n == self._count:
            return
        self._count = n
        if self._current >= n:
            self._current = max(0, n - 1)
        self._selection = {i for i in self._selection if i < n}
        self._recompute_height()
        self.update()

    def count(self) -> int:
        return self._count

    def set_current(self, i: int, *, emit: bool = False) -> None:
        """Set the current frame. ``emit=False`` (default) updates silently so
        the host can drive this from the backing slider without feedback loops."""
        if self._count == 0:
            return
        i = max(0, min(int(i), self._count - 1))
        if i == self._current:
            return
        self._current = i
        self.update()
        if emit:
            self.current_changed.emit(i)

    def current(self) -> int:
        return self._current

    def selection(self) -> frozenset:
        return frozenset(self._selection)

    def set_selection(self, indices, *, emit: bool = True) -> None:
        self._selection = {int(i) for i in indices if 0 <= int(i) < self._count}
        self.update()
        if emit:
            self.selection_changed.emit(self.selection())

    def clear_selection(self, *, emit: bool = True) -> None:
        if not self._selection:
            return
        self._selection.clear()
        self.update()
        if emit:
            self.selection_changed.emit(self.selection())

    def set_meta_fn(self, fn: Optional[Callable[[int], str]]) -> None:
        """Inject a callable returning per-frame tooltip text (or None)."""
        self._meta_fn = fn

    # ── Geometry ──
    def _tiles_per_row(self) -> int:
        w = max(1, self.width())
        return max(1, (w + self._gap) // (self._min_tile_w + self._gap))

    def _rows(self) -> int:
        if self._count == 0:
            return 1
        return max(1, math.ceil(self._count / self._tiles_per_row()))

    def _tile_rect(self, idx: int) -> QRect:
        per_row = self._tiles_per_row()
        row = idx // per_row
        col = idx % per_row
        w = max(1, self.width())
        # Fill the full width: distribute leftover pixels across the first cols.
        avail = w - self._gap * (per_row - 1)
        base_w = avail // per_row
        extra = avail - base_w * per_row
        x = col * self._gap + col * base_w + min(col, extra)
        tw = base_w + (1 if col < extra else 0)
        y = row * (self._tile_h + self._row_gap)
        return QRect(x, y, tw, self._tile_h)

    def _idx_at(self, pos) -> int:
        per_row = self._tiles_per_row()
        if self._tile_h <= 0:
            return -1
        row = pos.y() // (self._tile_h + self._row_gap)
        for col in range(per_row):
            idx = int(row) * per_row + col
            if idx >= self._count:
                break
            if self._tile_rect(idx).contains(pos):
                return idx
        return -1

    def _recompute_height(self) -> None:
        h = self._rows() * (self._tile_h + self._row_gap)
        self.setFixedHeight(max(self._tile_h + self._row_gap, h))

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._recompute_height()
        super().resizeEvent(event)

    # ── Painting ──
    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        base = QColor(Settings.BG_TERTIARY)
        hover = QColor(Settings.BG_HOVER)
        sel = QColor(Settings.ACCENT_CYAN)
        sel.setAlpha(150)
        cur = QColor(Settings.ACCENT_PURPLE)
        border = QColor(Settings.BORDER_COLOR)

        show_labels = self._min_tile_w >= 22 and self._tiles_per_row() <= 64
        font = QFont("Helvetica Neue")
        font.setPointSize(7)
        p.setFont(font)
        fm = QFontMetrics(font)

        for idx in range(self._count):
            r = self._tile_rect(idx)
            if r.bottom() < event.rect().top() or r.top() > event.rect().bottom():
                continue
            if idx == self._current:
                fill = cur
            elif idx in self._selection:
                fill = sel
            elif idx == self._hover_idx:
                fill = hover
            else:
                fill = base
            p.fillRect(r, fill)
            p.setPen(QPen(border, 1))
            p.drawRect(r.adjusted(0, 0, -1, -1))
            if show_labels:
                label = str(idx + 1)
                if fm.horizontalAdvance(label) <= r.width() - 2:
                    p.setPen(QColor(Settings.FG_PRIMARY)
                             if idx == self._current else QColor(Settings.FG_SECONDARY))
                    p.drawText(r, Qt.AlignmentFlag.AlignCenter, label)
        p.end()

    # ── Mouse ──
    def mousePressEvent(self, event) -> None:  # noqa: N802
        idx = self._idx_at(event.position().toPoint())
        if idx < 0:
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if event.button() == Qt.MouseButton.RightButton:
            # Right-click: if the tile isn't in the selection, select just it.
            if idx not in self._selection:
                self._selection = {idx}
                self.selection_changed.emit(self.selection())
            self.update()
            return
        mods = event.modifiers()
        if mods & Qt.KeyboardModifier.ShiftModifier:
            lo, hi = sorted((self._current, idx))
            self._selection = set(range(lo, hi + 1))
            self.selection_changed.emit(self.selection())
        elif mods & Qt.KeyboardModifier.ControlModifier:
            # Toggle this tile in/out of the selection.
            if idx in self._selection:
                self._selection.discard(idx)
            else:
                self._selection.add(idx)
            self.selection_changed.emit(self.selection())
        else:
            self._selection = {idx}
            self.selection_changed.emit(self.selection())
        self.set_current(idx, emit=True)
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        idx = self._idx_at(event.position().toPoint())
        if idx != self._hover_idx:
            self._hover_idx = idx
            if idx >= 0 and self._meta_fn is not None:
                try:
                    self.setToolTip(self._meta_fn(idx) or "")
                except Exception:
                    pass
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802
        if self._hover_idx != -1:
            self._hover_idx = -1
            self.update()
        super().leaveEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        from PySide6.QtWidgets import QMenu
        if not self._selection:
            return
        menu = QMenu(self)
        n = len(self._selection)
        act_crop = menu.addAction(f"Crop data to {n} selected {self._axis or 'frame'}(s)")
        act_clear = menu.addAction("Clear selection")
        chosen = menu.exec(event.globalPos())
        if chosen == act_crop:
            self.crop_requested.emit(self.selection())
        elif chosen == act_clear:
            self.clear_selection()

    # ── Keyboard ──
    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key not in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            super().keyPressEvent(event)
            return
        mods = event.modifiers()
        step = 1
        if mods & Qt.KeyboardModifier.ControlModifier:
            step = 10
        elif mods & Qt.KeyboardModifier.ShiftModifier:
            step = 5
        delta = -step if key == Qt.Key.Key_Left else step
        self.set_current(self._current + delta, emit=True)
