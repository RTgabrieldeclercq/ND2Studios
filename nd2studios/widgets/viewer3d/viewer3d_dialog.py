"""``Viewer3DDialog`` — a standalone window hosting a :class:`PyVista3DViewer`.

Used by the "View in 3D" result actions (DVC, PTV) that want a separate window
without disturbing a page's layout. For the in-tab 2D/3D toggle the viewer is
embedded directly instead (and rides the page's existing pop-out mechanism).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtWidgets import QDialog, QVBoxLayout, QWidget

from nd2studios.widgets.viewer3d.pyvista_viewer import PyVista3DViewer


class Viewer3DDialog(QDialog):
    """A resizable window wrapping a single :class:`PyVista3DViewer`."""

    def __init__(self, parent: Optional[QWidget] = None,
                 title: str = "3-D View") -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.resize(900, 700)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.viewer = PyVista3DViewer(self)
        layout.addWidget(self.viewer)

    def set_volume(self, volume: Any, *,
                   channel_display: Optional[Dict[str, Dict[str, Any]]] = None,
                   m: int = 0, t: int = 0, z: int = 0) -> None:
        self.viewer.set_volume(volume, channel_display=channel_display,
                               m=m, t=t, z=z)

    def set_overlay(self, overlay: Any, scalar: Optional[str] = None) -> None:
        self.viewer.set_overlay(overlay, scalar=scalar)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        try:
            self.viewer.on_close()
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(event)
