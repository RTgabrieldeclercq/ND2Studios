"""Floating parameter editor for a selected node (V1.45).

A frameless ``Qt.Tool`` window that embeds the existing
``widgets/common.py:ParamEditor`` so it renders every ``ParamSpec`` type
(float / int / bool / choice / str / hidden, with ``visible_when``) exactly as
the Recipe and Analysis pages do. Using a Tool window (rather than ``Qt.Popup``)
keeps embedded ``QComboBox`` dropdowns working — a ``Qt.Popup`` parent closes
when the combo's own popup opens. Closes on the close button or Esc.
"""
from __future__ import annotations

from typing import Any, Dict, List

from typing import Optional

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from nd2studios.core.plugin_registry import ParamSpec
from nd2studios.core.settings import Settings
from nd2studios.widgets.common import ParamEditor
from nd2studios.widgets.icon_button import icon_button, scaled, scale_qss


class ParamPopup(QWidget):
    """In-board popup hosting a :class:`ParamEditor` for one node's params."""

    params_changed = Signal(dict)
    edit_requested = Signal()   # the "Edit…" action button was clicked

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setObjectName("nodeParamPopup")
        self.setMinimumWidth(scaled(240))

        frame = QFrame(self)
        frame.setObjectName("nodeParamPopupFrame")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        header = QHBoxLayout()
        self._title = QLabel("Parameters")
        self._title.setStyleSheet(scale_qss(
            f"color: {Settings.FG_PRIMARY}; font: bold 10pt;"
        ))
        header.addWidget(self._title, stretch=1)
        close_btn = icon_button("fa5s.times", "Close", icon_px=12)
        close_btn.setFlat(True)
        close_btn.clicked.connect(self.hide)
        header.addWidget(close_btn)
        layout.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        # Never scroll sideways — the popup is sized to fit its content width, so
        # rows compress to the (clamped) width instead of overflowing the page.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        self.editor = ParamEditor()
        self.editor.params_changed.connect(self.params_changed)
        inner_layout.addWidget(self.editor)
        inner_layout.addStretch(1)
        scroll.setWidget(inner)
        layout.addWidget(scroll)

        self._empty = QLabel("This node has no parameters.")
        self._empty.setStyleSheet(scale_qss(f"color: {Settings.FG_SECONDARY}; font: 9pt;"))
        self._empty.setVisible(False)
        layout.addWidget(self._empty)

        # Optional action button for nodes with a richer editor (if-else
        # condition, Compute Measurements metric picker) — shown via show_for.
        self._edit_btn = QPushButton("Edit…")
        self._edit_btn.setObjectName("pipelineToolBtn")
        self._edit_btn.clicked.connect(self.edit_requested)
        self._edit_btn.setVisible(False)
        layout.addWidget(self._edit_btn)

        self.resize(scaled(280), scaled(320))

    def show_for(
        self,
        title: str,
        specs: List[ParamSpec],
        values: Dict[str, Any],
        global_pos: QPoint,
        edit_label: Optional[str] = None,
    ) -> None:
        """Populate the editor for a node and show the popup near ``global_pos``.

        ``edit_label`` (e.g. "Edit condition…") shows an action button that emits
        :attr:`edit_requested` — used by nodes with a dedicated editor dialog."""
        self._title.setText(title)
        self.editor.set_params(specs)
        if values:
            self.editor.set_values(values)
        has_params = bool(specs)
        self.editor.setVisible(has_params)
        # Only show the "no parameters" note when there's also no editor button.
        self._empty.setVisible(not has_params and not edit_label)
        if edit_label:
            self._edit_btn.setText(edit_label)
            self._edit_btn.setVisible(True)
        else:
            self._edit_btn.setVisible(False)
        self._resize_to_content(has_params, bool(edit_label), global_pos)
        self._move_within_page(global_pos)
        self.show()
        self.raise_()

    # ── sizing / placement ──────────────────────────────────────────────────

    def _resize_to_content(self, has_params: bool, has_edit: bool,
                           anchor: QPoint) -> None:
        """Size the popup to its editor content, capped to the page so it always
        fits (the scroll area absorbs any remaining vertical overflow)."""
        avail = self._page_rect(anchor)
        # The QScrollArea's own sizeHint doesn't track its content, so measure
        # the embedded editor directly.
        ed = self.editor.sizeHint()
        content_w = ed.width() if has_params else scaled(200)
        content_h = ed.height() if has_params else 0
        # Chrome: title row + frame margins + spacing (+ edit button / empty note).
        chrome_h = scaled(54)
        if has_edit:
            chrome_h += scaled(40)
        if not has_params and not has_edit:
            chrome_h += scaled(24)
        w = max(self.minimumWidth(), content_w + scaled(36))
        w = min(w, scaled(460), max(scaled(240), avail.width() - scaled(24)))
        h = content_h + chrome_h
        h = max(scaled(110), min(h, avail.height() - scaled(24)))
        self.resize(int(w), int(h))

    def _move_within_page(self, global_pos: QPoint) -> None:
        """Place the popup at ``global_pos`` but fully inside the GUI page."""
        avail = self._page_rect(global_pos)
        w, h = self.width(), self.height()
        x, y = global_pos.x(), global_pos.y()
        if x + w > avail.right():
            x = avail.right() - w
        if y + h > avail.bottom():
            y = avail.bottom() - h
        x = max(avail.left(), x)
        y = max(avail.top(), y)
        self.move(int(x), int(y))

    def _page_rect(self, anchor: QPoint) -> QRect:
        """The area the popup must stay within: the top-level app window,
        intersected with the screen's available geometry."""
        win = self.parentWidget().window() if self.parentWidget() is not None else None
        screen = (win.screen() if win is not None else None) \
            or QGuiApplication.screenAt(anchor) or QGuiApplication.primaryScreen()
        area = (screen.availableGeometry() if screen is not None
                else QRect(0, 0, 1920, 1080))
        if win is not None:
            wr = win.frameGeometry()
            if wr.isValid() and not wr.isEmpty():
                inter = area.intersected(wr)
                if not inter.isEmpty():
                    area = inter
        return area

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
            return
        super().keyPressEvent(event)
