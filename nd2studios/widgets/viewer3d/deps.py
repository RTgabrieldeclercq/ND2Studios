"""Optional-dependency gate for the 3-D viewer.

Kept in its own tiny, import-cheap module (no Qt, no PyVista) so both
``__init__`` and ``pyvista_viewer`` can import the flag without a cycle and
without paying the VTK import cost. Mirrors the Cellpose/StarDist pattern
(``importlib.util.find_spec`` → availability flag, friendly message when absent).

V1.66: the viewer renders **off-screen** (VTK → image → QLabel) rather than
embedding a native ``pyvistaqt.QtInteractor`` — a native OpenGL window cannot
composite into ND2Studios' frameless, ``WA_TranslucentBackground`` main window
(it punches a "hole" through to the desktop). Off-screen needs only ``pyvista``
(which pulls ``vtk``); ``pyvistaqt`` is no longer required.
"""
from __future__ import annotations

import importlib.util

PYVISTA_AVAILABLE: bool = importlib.util.find_spec("pyvista") is not None

#: One-line install hint surfaced in the placeholder widget.
PIP_COMMAND: str = "pip install pyvista"


def missing_reason() -> str:
    """Human-readable reason the 3-D viewer is unavailable (empty if available)."""
    if PYVISTA_AVAILABLE:
        return ""
    return "3-D view needs: pyvista"
