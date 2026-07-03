"""Reusable collapsible side panel (V1.44).

Wraps an arbitrary content widget in a panel with a header that carries a
collapse toggle, mirroring the right-hand :class:`~nd2studios.widgets.lut_sidebar.LutSidebar`
so the left control panels across tabs collapse/expand with the same button and
animation. Collapsing animates the panel width down to a thin rail that still
shows the toggle, so the user can bring it back.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from nd2studios.core.settings import Settings


class CollapsibleSidebar(QFrame):
    """A panel that animates between an expanded width and a thin rail."""

    COLLAPSED_WIDTH = 28
    ANIMATION_MS = 220

    collapse_changed = Signal(bool)  # True when expanded

    def __init__(
        self,
        content: QWidget,
        *,
        side: str = "left",
        title: str = "Controls",
        expanded_width: int = 320,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("contentArea")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._side = side
        self._expanded_width = int(expanded_width)
        self._expanded = True
        self._content = content
        self._anim_min: Optional[QPropertyAnimation] = None
        self._anim_max: Optional[QPropertyAnimation] = None

        self.setMinimumWidth(self._expanded_width)
        self.setMaximumWidth(self._expanded_width)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header bar with the collapse toggle, placed to mirror LutSidebar:
        # the toggle hugs the edge nearest the viewer (right edge for a left
        # panel, left edge for a right panel).
        top_bar = QFrame()
        top_bar.setFixedHeight(28)
        tl = QHBoxLayout(top_bar)
        tl.setContentsMargins(8, 0, 4, 0)
        tl.setSpacing(4)
        self._title = QLabel(title)
        self._title.setStyleSheet(
            f"color: {Settings.ACCENT_PURPLE}; font: bold 9pt 'Helvetica Neue';")
        self._toggle = QPushButton()
        self._toggle.setFixedSize(20, 20)
        self._toggle.setToolTip("Collapse / expand this panel")
        self._toggle.setStyleSheet(
            "QPushButton { background: transparent; "
            f"color: {Settings.FG_SECONDARY}; border: none; padding: 0; font: 10pt; }}"
            "QPushButton:hover { color: " + Settings.FG_PRIMARY + "; }"
        )
        self._toggle.clicked.connect(self.toggle)
        if side == "left":
            tl.addWidget(self._title)
            tl.addStretch(1)
            tl.addWidget(self._toggle)
        else:
            tl.addWidget(self._toggle)
            tl.addWidget(self._title)
            tl.addStretch(1)
        outer.addWidget(top_bar)

        outer.addWidget(content, stretch=1)
        self._update_glyph()

    # ── collapse/expand ──
    @property
    def is_expanded(self) -> bool:
        return self._expanded

    def toggle(self) -> None:
        self.set_expanded(not self._expanded)

    def set_expanded(self, value: bool) -> None:
        value = bool(value)
        if value == self._expanded:
            return
        self._expanded = value
        start = self.width()
        end = self._expanded_width if value else self.COLLAPSED_WIDTH
        self._content.setVisible(value)
        self._title.setVisible(value)

        self._anim_min = QPropertyAnimation(self, b"minimumWidth")
        self._anim_min.setDuration(self.ANIMATION_MS)
        self._anim_min.setStartValue(start)
        self._anim_min.setEndValue(end)
        self._anim_min.setEasingCurve(QEasingCurve.Type.InOutQuart)
        self._anim_max = QPropertyAnimation(self, b"maximumWidth")
        self._anim_max.setDuration(self.ANIMATION_MS)
        self._anim_max.setStartValue(start)
        self._anim_max.setEndValue(end)
        self._anim_max.setEasingCurve(QEasingCurve.Type.InOutQuart)
        self._anim_min.start()
        self._anim_max.start()
        self._update_glyph()
        self.collapse_changed.emit(value)

    def _update_glyph(self) -> None:
        # Arrow points toward the viewer when expanded (to collapse away),
        # and back toward the panel when collapsed (to expand).
        if self._side == "left":
            self._toggle.setText("◀" if self._expanded else "▶")
        else:
            self._toggle.setText("▶" if self._expanded else "◀")
