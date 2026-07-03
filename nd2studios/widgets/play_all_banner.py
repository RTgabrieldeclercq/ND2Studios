"""Universal "Play All" banner (V1.44).

A single centered play/pause control + FPS setter that spans the full width of
the image-viewer area (not per-viewer). It drives the T-axis playback of every
viewer it targets simultaneously, so multiple loaded files animate together.

The banner is only meaningful with more than one viewer loaded, so the host
calls :meth:`set_target_viewers` whenever the set of viewers changes; the banner
hides itself when fewer than two targets are present.
"""
from __future__ import annotations

from typing import List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDoubleSpinBox, QFrame, QHBoxLayout, QLabel

from nd2studios.core.settings import Settings
from nd2studios.widgets.icon_button import bind_toggle_icon, icon_button


class PlayAllBanner(QFrame):
    """Centered play/pause + FPS banner that plays all target viewers' T axis."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("playAllBanner")
        self.setStyleSheet(
            "QFrame#playAllBanner { background: %s; border-top: 1px solid %s; "
            "border-bottom: 1px solid %s; }"
            % (Settings.BG_SECONDARY, Settings.BORDER_COLOR, Settings.BORDER_COLOR)
        )
        self._targets: List[object] = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        layout.addStretch(1)

        self._btn = icon_button(
            "fa5s.play", "Play / pause the time-lapse on all loaded viewers",
            text=" Play All", checkable=True, object_name="playBtn", icon_px=12)
        bind_toggle_icon(self._btn, "fa5s.play", "fa5s.pause",
                         color_checked=Settings.ACCENT_GREEN)
        self._btn.toggled.connect(self._on_toggled)
        layout.addWidget(self._btn)

        self._fps = QDoubleSpinBox()
        self._fps.setRange(0.1, 60.0)
        self._fps.setValue(10.0)
        self._fps.setSingleStep(0.5)
        self._fps.setSuffix(" fps")
        self._fps.setFixedWidth(84)
        self._fps.setToolTip("Playback speed for Play All")
        layout.addWidget(self._fps)

        layout.addStretch(1)
        self.hide()

    def set_target_viewers(self, viewers: List[object]) -> None:
        """Set the viewers this banner controls. Hidden unless 2+ are present."""
        self._targets = [v for v in viewers if v is not None]
        multi = len(self._targets) >= 2
        self.setVisible(multi)
        if not multi and self._btn.isChecked():
            self._btn.setChecked(False)

    def _on_toggled(self, playing: bool) -> None:
        fps = self._fps.value()
        for v in self._targets:
            if hasattr(v, "set_t_playing"):
                try:
                    v.set_t_playing(playing, fps=fps)
                except Exception:  # noqa: BLE001
                    pass
