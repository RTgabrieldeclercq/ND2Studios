"""Measure tab-switch latency under realistic Qt conditions.

Spins up a real :class:`nd2studios.core.main_window.MainWindow` under
``QT_QPA_PLATFORM=offscreen`` (so no display is needed), then walks
through every page key in :data:`Settings.PAGES` in a round-robin and
times each ``MainWindow._navigate(key)`` call.

We deliberately do **not** load an ND2 file here — ``PAGE_PREREQS``
gates several pages behind ``status >= imported``, but ND2Studios's
nav guard pops a ``QMessageBox.information`` dialog when the prereq is
unmet, which would block the offscreen run. To work around this we
force-promote the active experiment's status to the highest level for
the duration of the scenario; the prereq check then never trips and
all pages are exercised. This is profiling-only and does not affect
the running app.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

from nd2studios.utils.profiling import measure

from profiling.harness.fixtures import TAB_SWITCH_COUNT


def run() -> List[Dict[str, Any]]:
    """Cycle through every page TAB_SWITCH_COUNT times, timing each navigate."""
    results: List[Dict[str, Any]] = []

    # Run headless. Must be set before any PySide6.QtGui / QtWidgets import.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        return [{
            "name": "tab_switch",
            "error": f"PySide6 unavailable: {exc}",
        }]

    # Force plugin / pipeline registries to populate before constructing
    # any page that depends on them — same order as nd2studios/__main__.py.
    try:
        import nd2studios.plugins.enhancement.builtin  # noqa: F401
        import nd2studios.backend.analysis.tear_detection  # noqa: F401
        import nd2studios.backend.analysis.histogram_threshold_pipeline  # noqa: F401
        import nd2studios.backend.analysis.spots_pipeline  # noqa: F401
        import nd2studios.backend.analysis.manual_mask  # noqa: F401
        try:
            import nd2studios.backend.analysis.nuclei_segmentation  # noqa: F401
        except Exception:
            # Optional Cellpose-backed module — fine to skip if it can't import.
            pass
    except Exception as exc:  # noqa: BLE001
        return [{
            "name": "tab_switch",
            "error": f"registry import failed: {type(exc).__name__}: {exc}",
        }]

    from nd2studios.core.main_window import MainWindow
    from nd2studios.core.settings import Settings

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = None
    try:
        with measure("tab_switch_construct_main_window") as m_init:
            window = MainWindow()
        m_init.extra.update({"pages": [k for k, *_ in Settings.PAGES]})
        results.append(m_init.to_dict())

        # Bypass PAGE_PREREQS by maxing out the active experiment's
        # status. The prereq check is `STATUS_ORDER.index(active.status)
        # >= required` — promote to the last entry to satisfy every page.
        if window.exp_manager.active is not None and Settings.STATUS_ORDER:
            window.exp_manager.active.status = Settings.STATUS_ORDER[-1]

        page_keys = [key for key, *_ in Settings.PAGES]
        app.processEvents()

        for i in range(TAB_SWITCH_COUNT):
            target = page_keys[i % len(page_keys)]
            with measure(f"tab_switch_to_{target}", track_pyalloc=False) as m:
                window._navigate(target)
                # Drain any deferred slot work the page enqueued.
                app.processEvents()
            m.extra["target_page"] = target
            m.extra["iteration"] = i
            results.append(m.to_dict())
    except Exception as exc:  # noqa: BLE001
        results.append({
            "name": "tab_switch_run",
            "error": f"{type(exc).__name__}: {exc}",
        })
    finally:
        if window is not None:
            try:
                window.close()
            except Exception:
                pass

    return results


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
