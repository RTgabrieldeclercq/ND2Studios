"""
ND2Studios entry point.

Sets up the Qt application, applies the dark theme, opens the main window.
"""
from __future__ import annotations

import os
import sys


def _setup_environment() -> None:
    """Set environment variables that must be in place before Qt imports."""
    # PyDracula's high-DPI workaround.
    os.environ.setdefault("QT_FONT_DPI", "96")

    # V1.36 Phase 4: pin pyqtgraph to PySide6 before any pyqtgraph
    # import in the process — otherwise its auto-detect can pick a
    # stray PyQt6 install and the layouts in :class:`GpuImageCanvas`
    # will refuse pyqtgraph's widgets as foreign.
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

    # SSL: some plugins (e.g. those that fetch reference data online) need
    # certifi's bundle. Set it eagerly; harmless if unused.
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass


def main() -> None:
    _setup_environment()

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    # Enable high-DPI pixmaps for the image viewer.
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QGuiApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QGuiApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    from nd2studios.core.settings import Settings
    from nd2studios.core.theme import STYLESHEET
    from nd2studios.core.main_window import MainWindow

    # V1.36 Phase 4: configure pyqtgraph defaults before any
    # :class:`GpuImageCanvas` is constructed. ``row-major`` makes
    # numpy ``(H, W)`` arrays display with ``(0, 0)`` at the top-left
    # without a transpose. ``useOpenGL=False`` is the safer default on
    # Windows and remote-desktop sessions; can be flipped later once
    # the GL path is benchmarked on the McGhee Lab workstation.
    try:
        import pyqtgraph as pg
        pg.setConfigOptions(
            imageAxisOrder="row-major",
            useOpenGL=False,
            antialias=False,
            background=Settings.BG_SECONDARY,
        )
    except Exception:
        # If pyqtgraph isn't importable, the viewer's defensive
        # fallback will pick the legacy canvas — no crash here.
        pass

    # Force-import plugin/pipeline modules so decorator-based registration
    # runs at startup before any page is constructed.
    import nd2studios.plugins.enhancement.builtin  # noqa: F401
    import nd2studios.backend.analysis.nuclei_segmentation  # noqa: F401
    import nd2studios.backend.analysis.tear_detection  # noqa: F401
    import nd2studios.backend.analysis.histogram_threshold_pipeline  # noqa: F401
    import nd2studios.backend.analysis.spots_pipeline  # noqa: F401
    import nd2studios.backend.analysis.manual_mask  # noqa: F401

    app = QApplication(sys.argv)
    app.setApplicationName(Settings.APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)

    # V1.39 Phase 7: log GPU status once and seed the analysis-GPU
    # dispatch flag from Settings. Both calls are safe on CPU-only
    # boxes — :func:`configure(False)` is a no-op; :func:`log_gpu_status`
    # emits a single info line either way. Performance dialog flips
    # the flag at runtime via the same :func:`configure` API.
    import logging
    logging.basicConfig(level=logging.INFO)
    try:
        from nd2studios.compute.gpu import configure, log_gpu_status
        log_gpu_status()
        configure(bool(getattr(Settings, "USE_GPU_ANALYSIS", False)))
    except Exception:  # noqa: BLE001 — GPU bootstrap must never break startup
        pass

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
