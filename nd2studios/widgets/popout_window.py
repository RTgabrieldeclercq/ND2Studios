"""Reusable pop-out / maximize window (V1.46.3).

A :class:`PopOutWindow` *borrows* a live widget out of its host layout, shows it
in a standalone maximizable window, and returns the very same widget to its
original slot when the window is closed. Because the widget is reparented rather
than rebuilt, all of its live state — tab selection, slider positions, channel
LUTs, zoom/pan, plot contents — is preserved automatically.

The host supplies an ``on_restore(widget)`` callback that re-inserts the widget
into its original parent layout (e.g. ``QSplitter.insertWidget``). The window
calls it exactly once, whether the close comes from the title-bar button or a
programmatic :meth:`restore`.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QDialog, QVBoxLayout, QWidget


class PopOutWindow(QDialog):
    """Host a borrowed widget full-window and restore it on close."""

    # Emitted after the borrowed widget has been handed back to the host.
    restored = Signal()

    def __init__(
        self,
        widget: QWidget,
        *,
        title: str,
        on_restore: Callable[[QWidget], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        # A full top-level window with min / maximize / close buttons (not a
        # modal sheet) so the user can keep interacting with the main window.
        self.setWindowFlags(Qt.WindowType.Window)
        self.setModal(False)

        self._widget = widget
        self._on_restore = on_restore
        self._restored = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        self.resize(1100, 800)

    def restore(self) -> None:
        """Hand the widget back to the host and close the window."""
        if self._restored:
            return
        self._hand_back()
        self.close()

    def _hand_back(self) -> None:
        if self._restored:
            return
        self._restored = True
        # Detach from this dialog's layout so the host is free to re-parent it.
        self._widget.setParent(None)
        # The host re-inserts the widget. Guard it so a callback error can never
        # leave the widget orphaned (parent=None, not in any layout) — which is
        # what makes the panel "disappear" instead of docking back.
        try:
            self._on_restore(self._widget)
        finally:
            self.restored.emit()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        # Covers the title-bar close button as well as programmatic close().
        self._hand_back()
        super().closeEvent(event)

    def reject(self) -> None:  # noqa: N802
        # Esc (and any programmatic reject) on a QDialog calls reject() → hide(),
        # which does NOT fire closeEvent — so without this the widget would be
        # left inside the hidden dialog and the panel would vanish. Hand it back
        # first, then let the dialog close.
        self._hand_back()
        super().reject()
