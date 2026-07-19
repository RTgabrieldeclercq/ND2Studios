"""Placeholder shown in place of the 3-D canvas when PyVista is not installed.

Keeps the 2D/3D toggle discoverable even before the optional extra is present:
toggling to 3-D shows an install hint instead of crashing.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from nd2studios.core.settings import Settings
from nd2studios.widgets.icon_button import scale_qss, scaled
from nd2studios.widgets.viewer3d.deps import PIP_COMMAND, missing_reason


class Missing3DDeps(QWidget):
    """A friendly 'install to enable 3-D' panel with a copy-command button."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(scaled(24), scaled(24), scaled(24), scaled(24))
        root.setSpacing(scaled(10))
        root.addStretch(1)

        title = QLabel("3-D view unavailable")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(scale_qss(
            f"color: {Settings.FG_PRIMARY}; font: 600 15pt 'Helvetica Neue';"))
        root.addWidget(title)

        reason = missing_reason() or "Optional 3-D rendering support is not installed."
        body = QLabel(
            f"{reason}.\n\nThe 2-D viewer remains fully available. To enable the\n"
            "volumetric 3-D viewer, install the optional extras:")
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body.setWordWrap(True)
        body.setStyleSheet(scale_qss(f"color: {Settings.FG_SECONDARY}; font: 10pt;"))
        root.addWidget(body)

        cmd = QLabel(PIP_COMMAND)
        cmd.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cmd.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        cmd.setStyleSheet(scale_qss(
            f"color: {Settings.ACCENT_CYAN}; font: 10pt 'Courier New';"
            f" background: {Settings.BG_SECONDARY}; padding: 6px; border-radius: 4px;"))
        root.addWidget(cmd, alignment=Qt.AlignmentFlag.AlignCenter)

        copy_btn = QPushButton(" Copy pip command")
        copy_btn.setObjectName("primaryBtn")
        copy_btn.clicked.connect(self._copy)
        root.addWidget(copy_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        root.addStretch(2)

    def _copy(self) -> None:
        cb = QApplication.clipboard()
        if cb is not None:
            cb.setText(PIP_COMMAND)
