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
    from nd2studios.core.theme import build_stylesheet
    from nd2studios.core.main_window import MainWindow

    # V1.41 — apply the user's saved preferences to ``Settings`` before
    # any subsystem reads them. Best-effort; a corrupt or missing
    # preferences file just means the user gets defaults.
    try:
        from nd2studios.utils.user_config import load_user_preferences
        load_user_preferences()
    except Exception:  # noqa: BLE001
        pass

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
    import nd2studios.plugins.enhancement.registration  # noqa: F401  # registers RegistrationPlugin (drift correction)
    import nd2studios.backend.analysis.nuclei_segmentation  # noqa: F401
    import nd2studios.backend.analysis.stardist_segmentation  # noqa: F401
    import nd2studios.backend.analysis.tear_detection  # noqa: F401
    import nd2studios.backend.analysis.histogram_threshold_pipeline  # noqa: F401
    import nd2studios.backend.analysis.spots_pipeline  # noqa: F401
    import nd2studios.backend.analysis.manual_mask  # noqa: F401
    import nd2studios.backend.dvc.method  # noqa: F401  # registers ALDVCMethod (DVC node)
    import nd2studios.backend.registration.method  # noqa: F401  # registers RigidRegistration (registration node)

    app = QApplication(sys.argv)
    app.setApplicationName(Settings.APP_NAME)
    app.setStyle("Fusion")
    # build_stylesheet() injects rendered arrow images (needs the QApplication).
    app.setStyleSheet(build_stylesheet())

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

    # V1.41 startup diagnostic sequence: emit one info line per
    # subsystem and apply safe defaults so a low-RAM laptop doesn't
    # try to build pyramids or push 4-byte-per-pixel RGBA to a 1 GB
    # iGPU.  Each block is independently try/excepted because the
    # full chain must never block the app from starting.
    try:
        from nd2studios.utils.resources import log_system_resources, detect
        log_system_resources()
        res = detect()
        if res.available_ram_gb < 8 and getattr(Settings, "BUILD_PYRAMIDS", False):
            Settings.BUILD_PYRAMIDS = False
            import logging as _logging
            _logging.getLogger(__name__).info(
                "Low RAM (%.1f GB available) — disabling pyramid build by default.",
                res.available_ram_gb,
            )
    except Exception:  # noqa: BLE001
        pass
    try:
        from pathlib import Path
        from nd2studios.utils.storage import log_default_storage_class
        # Probe the working directory; a per-file probe will refine
        # this when LoadWorker opens an actual ND2 / TIFF.
        log_default_storage_class(Path(Settings.PROJECT_DIR))
    except Exception:  # noqa: BLE001
        pass
    try:
        from nd2studios.compute.gpu import gpu_status
        gpu = gpu_status()
        if (
            gpu.get("available")
            and float(gpu.get("memory_gb", 0.0)) < 2.0
            and getattr(Settings, "USE_GPU_DISPLAY", False)
        ):
            Settings.USE_GPU_DISPLAY = False
            import logging as _logging
            _logging.getLogger(__name__).info(
                "GPU has only %.1f GB VRAM — disabling pyqtgraph GPU canvas.",
                float(gpu.get("memory_gb", 0.0)),
            )
    except Exception:  # noqa: BLE001
        pass

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
