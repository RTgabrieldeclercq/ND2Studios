"""viewer3d — the V1.65 PyVista-backed 3-D viewer widget subsystem.

Public surface:

- :data:`PYVISTA_AVAILABLE` — True when ``pyvista`` and ``pyvistaqt`` import.
- :class:`PyVista3DViewer` — the reusable embedded 3-D viewer (drop-in alternate
  to ``MultiAxisViewer`` behind a 2D/3D toggle).
- :class:`Viewer3DDialog` — a standalone window wrapping one viewer (result
  "View in 3D" actions).
- :class:`Missing3DDeps` — the install-hint placeholder shown when the optional
  extra is absent.

Importing this package never imports PyVista/VTK: the classes import cleanly
whether or not the optional extra is installed, and the heavy import happens
lazily inside :class:`PyVista3DViewer` only when a canvas is actually created.
"""
from __future__ import annotations

from nd2studios.widgets.viewer3d.deps import PIP_COMMAND, PYVISTA_AVAILABLE
from nd2studios.widgets.viewer3d.placeholder import Missing3DDeps
from nd2studios.widgets.viewer3d.pyvista_viewer import (
    MODE_ISO,
    MODE_MIP,
    MODE_SLICES,
    MODE_VOLUME,
    RENDER_MODES,
    PyVista3DViewer,
)
from nd2studios.widgets.viewer3d.viewer3d_dialog import Viewer3DDialog

__all__ = [
    "PYVISTA_AVAILABLE",
    "PIP_COMMAND",
    "Missing3DDeps",
    "PyVista3DViewer",
    "Viewer3DDialog",
    "MODE_VOLUME",
    "MODE_MIP",
    "MODE_SLICES",
    "MODE_ISO",
    "RENDER_MODES",
]
