"""
ND2Studios main window.

Layout (PyDracula-derived):

    ┌────────────────────────────────────────────────┐
    │  custom title bar (drag, min/max/close)       │
    ├──────┬─────────────────────────────────────────┤
    │ side │  top bar (page title + status badge)   │
    │ bar  ├─────────────────────────────────────────┤
    │      │                                         │
    │ ☰    │  page stack (QStackedWidget)            │
    │ 📂   │                                         │
    │ 🧪   │                                         │
    │ 💾   │                                         │
    │      │                                         │
    │ New  │                                         │
    │ Save ├─────────────────────────────────────────┤
    │ Load │  bottom bar (progress + status text)   │
    └──────┴─────────────────────────────────────────┘

Sidebar collapses 60↔240 px with `QPropertyAnimation`. Frameless window
has `CustomGrip` resize handles on all four edges and a drop shadow on
the `#bgApp` background frame.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from PySide6.QtCore import (
    QEasingCurve, QEvent, QPropertyAnimation, QSize, Qt, QTimer, Signal,
)
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtWidgets import (
    QButtonGroup, QFileDialog, QGraphicsDropShadowEffect, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QSizeGrip, QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

from nd2studios.compute import BuildPyramidJob, JobRunner
from nd2studios.core.experiment_manager import (
    ND2StudiosManager, SESSION_EXTENSION,
)
from nd2studios.core.settings import Settings
from nd2studios.pipeline import (
    AnalysisStage,
    EnhancedDataset,
    HAS_ZARR,
    PyramidStage,
    PyramidUnavailable,
    RecipeStage,
    Session,
    workspace_disabled,
)
from nd2studios.widgets.custom_grips import CustomGrip
from nd2studios.widgets.icon_button import icon_button, scaled, tool_button

# Page key → qtawesome icon name for the top tab bar (V1.44).
_PAGE_ICONS = {
    "import": "fa5s.folder-open",
    "recipe": "fa5s.flask",
    "analysis": "fa5s.microscope",
    "results": "fa5s.chart-bar",
    "batch": "fa5s.bolt",
    "export": "fa5s.save",
}


class MainWindow(QMainWindow):
    """Frameless main window with collapsible sidebar and three pages."""

    # Emitted whenever a macro action is recorded so the MacroDialog can
    # append the block to its live-feed list while the user works.
    macro_action_recorded = Signal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{Settings.APP_NAME} v{Settings.APP_VERSION}")
        self.setMinimumSize(Settings.MIN_WIDTH, Settings.MIN_HEIGHT)
        self.resize(1440, 900)

        # Frameless + translucent so the rounded #bgApp shows through cleanly.
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # V1.37 Phase 5 — project-wide job runner for short, cancellable
        # background work (Analysis-page previews + commits). Built
        # before any page so pages can grab a reference in their
        # ``__init__``. Long-running QThread workers (Recipe, Batch,
        # Load, Prefetch, IO) are unaffected and keep their existing
        # lifecycles.
        self.job_runner = JobRunner(parent=self)
        # V1.39 Phase 7 — listen for pyramid-build completion so the
        # status bar and viewer attachments can update. Other job
        # keys (analysis_preview / analysis_commit) keep their own
        # per-page wiring; the slot filters by ``result.key``.
        self.job_runner.job_done.connect(self._on_pyramid_job_done)

        # V1.38 Phase 6 — per-source workspace. Attached by the Import
        # page after a successful load (see ``_attach_session_for``).
        # None until a file is imported; None forever if
        # ``ND2STUDIOS_DISABLE_WORKSPACE=1`` is set.
        self.session: Optional[Session] = None

        # State
        self.exp_manager = ND2StudiosManager(self)
        self.exp_manager.active_changed.connect(self._on_experiment_changed)
        self.exp_manager.status_changed.connect(self._on_status_changed)

        self.pages: Dict[str, QWidget] = {}
        self._current_page_key: Optional[str] = None
        self._drag_pos = None
        self._is_maximized = False

        # Config file state — set on Load or after a successful Save.
        self._cfg_path: Optional[str] = None
        self._cfg_data: Optional[Dict[str, Any]] = None

        # Macro recorder — records meaningful user actions for replay.
        from nd2studios.backend.macro_engine import MacroRecorder
        from nd2studios.core.macro_event_filter import MacroEventFilter
        self.macro_recorder = MacroRecorder()
        self.macro_event_filter = MacroEventFilter(self)
        self._macro_dialog: Optional[object] = None

        # V1.41 — global memory-pressure monitor. Built before pages so
        # any viewer or worker constructed below can grab the singleton
        # and subscribe to band signals.
        from nd2studios.core.memory_monitor import install_global
        self.memory_monitor = install_global(parent=self)
        self._memory_modal_shown: bool = False
        self._heavy_ops_blocked: bool = False
        self.memory_monitor.sampled.connect(self._on_memory_sampled)
        self.memory_monitor.critical.connect(self._on_memory_critical)
        self.memory_monitor.emergency.connect(self._on_memory_emergency)
        self.memory_monitor.recovered.connect(self._on_memory_recovered)

        self._build_ui()
        self._install_grips()
        self.memory_monitor.start()

        # Start with a blank session.
        self.exp_manager.new_experiment("Untitled")

    # ── UI construction ────────────────────────────────────────────
    def _build_ui(self) -> None:
        # Outer container so we can put a drop shadow on the inner #bgApp.
        outer = QWidget(self)
        self.setCentralWidget(outer)
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(10, 10, 10, 10)  # room for drop shadow
        outer_layout.setSpacing(0)

        bg_app = QWidget()
        bg_app.setObjectName("bgApp")
        outer_layout.addWidget(bg_app)
        self._bg_app = bg_app

        # Drop shadow on the rounded card.
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(Settings.SHADOW_BLUR_RADIUS)
        shadow.setXOffset(Settings.SHADOW_OFFSET_X)
        shadow.setYOffset(Settings.SHADOW_OFFSET_Y)
        shadow.setColor(QColor(0, 0, 0, Settings.SHADOW_ALPHA))
        bg_app.setGraphicsEffect(shadow)

        bg_layout = QVBoxLayout(bg_app)
        bg_layout.setContentsMargins(0, 0, 0, 0)
        bg_layout.setSpacing(0)

        # ── Custom title bar ──
        bg_layout.addWidget(self._build_title_bar())

        # ── Top tab bar (V1.44 — migrated from the old left sidebar) ──
        bg_layout.addWidget(self._build_top_tabs())

        # ── Body: content area fills the rest ──
        bg_layout.addWidget(self._build_content_area(), stretch=1)

    def _build_title_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("titleBarWidget")
        bar.setFixedHeight(Settings.TITLE_BAR_HEIGHT)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        app_label = QLabel(Settings.APP_NAME)
        app_label.setObjectName("titleBarApp")
        layout.addWidget(app_label)

        info_label = QLabel(Settings.APP_DESCRIPTION)
        info_label.setObjectName("titleBarInfo")
        layout.addWidget(info_label)
        self._title_info_label = info_label

        layout.addStretch(1)

        # Min / Max / Close — crisp DPI-scaling vector icons (V1.44).
        self._btn_min = icon_button("fa5s.window-minimize", "Minimize",
                                    object_name="titleBarBtn", icon_px=12)
        self._btn_min.clicked.connect(self.showMinimized)

        self._btn_max = icon_button("fa5s.window-maximize", "Maximize / Restore",
                                    object_name="titleBarBtn", icon_px=12)
        self._btn_max.clicked.connect(self._toggle_max_restore)

        self._btn_close = icon_button("fa5s.times", "Close",
                                      object_name="titleBarCloseBtn", icon_px=14)
        self._btn_close.clicked.connect(self.close)

        for b in (self._btn_min, self._btn_max, self._btn_close):
            layout.addWidget(b)

        # The whole title bar acts as a drag handle.
        bar.mousePressEvent = self._title_mouse_press
        bar.mouseMoveEvent = self._title_mouse_move
        bar.mouseDoubleClickEvent = self._title_mouse_double_click
        return bar

    def _build_top_tabs(self) -> QWidget:
        """Horizontal tab strip under the title bar (V1.44).

        Replaces the old animated left sidebar. Page tabs sit on the left with
        crisp DPI-scaling qtawesome icons; the session actions
        (Save/Load/Configure/Performance/Macro) cluster on the right.
        """
        bar = QWidget()
        bar.setObjectName("topTabBar")
        bar.setFixedHeight(scaled(46))
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(scaled(8), 0, scaled(8), 0)
        layout.setSpacing(scaled(4))

        # Page tabs (one per page), exclusive selection.
        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        self._nav_buttons: Dict[str, QPushButton] = {}

        for key, _icon, title, tooltip in Settings.PAGES:
            btn = tool_button(
                _PAGE_ICONS.get(key, ""), tooltip, text=title,
                checkable=True, object_name="topTabBtn",
            )
            btn.clicked.connect(lambda _checked=False, k=key: self._navigate(k))
            self._nav_group.addButton(btn)
            self._nav_buttons[key] = btn
            layout.addWidget(btn)

        layout.addStretch(1)

        # Session actions on the right.
        for icon_name, label, slot, tooltip in [
            ("fa5s.save",      "Save",        self._save_config,   "Save configuration to a .nd2s_cfg file"),
            ("fa5s.folder-open", "Load",      self._load_config,   "Load a .nd2s_cfg configuration file"),
            ("fa5s.cog",       "Configure",   self._configure,     "Open the pipeline configuration wizard"),
            ("fa5s.tachometer-alt", "Performance",
             self._open_performance_settings,
             "GPU acceleration and multi-resolution pyramid settings (V1.39)"),
            ("fa5s.film",      "Macro",       self._open_macro,
             "Record, edit, and replay action macros across files"),
        ]:
            btn = icon_button(icon_name, tooltip, text=label,
                              object_name="sessionTabBtn")
            btn.clicked.connect(slot)
            layout.addWidget(btn)

        return bar

    def _build_content_area(self) -> QWidget:
        content = QWidget()
        content.setObjectName("contentArea")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # V1.44 — the per-page header bar (page title + data-status badge) was
        # removed: the top tab bar already shows the active page, and the
        # status badge took space for little value. The page stack fills the
        # content area directly.

        # Page stack — pages are imported lazily here so that core/ does
        # not have a hard cycle with pages/ at import time.
        from nd2studios.pages.import_page import ImportPage
        from nd2studios.pages.recipe_page import RecipePage
        from nd2studios.pages.export_page import ExportPage
        from nd2studios.pages.analysis_page import AnalysisPage
        from nd2studios.pages.results_page import ResultsPage
        from nd2studios.pages.pipelines_page import PipelinesPage
        from nd2studios.pages.batch_page import BatchPage

        page_classes = {
            "import": ImportPage,
            "recipe": RecipePage,
            "export": ExportPage,
            "analysis": AnalysisPage,
            "results": ResultsPage,
            "pipelines": PipelinesPage,
            "batch": BatchPage,
        }
        self._stack = QStackedWidget()
        for key, _icon, _title, _tooltip in Settings.PAGES:
            cls = page_classes.get(key)
            page = cls(self) if cls is not None else self._stub(key)
            self.pages[key] = page
            self._stack.addWidget(page)

        layout.addWidget(self._stack, stretch=1)

        # Bottom bar: version + progress bar + status text.
        bottom_bar = QWidget()
        bottom_bar.setObjectName("bottomBar")
        bottom_bar.setFixedHeight(Settings.BOTTOM_BAR_HEIGHT)
        bb_layout = QHBoxLayout(bottom_bar)
        bb_layout.setContentsMargins(16, 0, 16, 0)

        version_label = QLabel(f"{Settings.APP_NAME} v{Settings.APP_VERSION}")
        version_label.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 8pt;")
        bb_layout.addWidget(version_label)
        bb_layout.addStretch(1)

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedWidth(220)
        self._progress_bar.setFixedHeight(16)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        bb_layout.addWidget(self._progress_bar)

        # V1.41 — live memory-pressure gauge. Sits between progress bar
        # and status text in the bottom bar.  Colour switches at band
        # boundaries via ``_on_memory_sampled``.
        self._memory_gauge = QLabel("RAM —")
        self._memory_gauge.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 8pt;")
        self._memory_gauge.setToolTip("System memory usage")
        bb_layout.addWidget(self._memory_gauge)

        self._status_text = QLabel("Ready")
        self._status_text.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 8pt;")
        bb_layout.addWidget(self._status_text)

        layout.addWidget(bottom_bar)

        # Pick the first nav button.
        first_key = Settings.PAGES[0][0]
        self._nav_buttons[first_key].setChecked(True)
        self._current_page_key = first_key

        return content

    def _stub(self, key: str) -> QWidget:
        """Fallback page if a real page class fails to import (dev safety net)."""
        w = QWidget()
        layout = QVBoxLayout(w)
        lbl = QLabel(f"{key.title()} — coming soon")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 14pt;")
        layout.addWidget(lbl)
        return w

    # ── Frameless window: title bar interactions ───────────────────
    def _title_mouse_press(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint()
            event.accept()

    def _title_mouse_move(self, event: QMouseEvent) -> None:
        if self._drag_pos is None or event.buttons() != Qt.LeftButton:
            return
        # If maximized, restore first (mimics PyDracula).
        if self._is_maximized:
            self._toggle_max_restore()
            self._drag_pos = event.globalPosition().toPoint()
            return
        delta = event.globalPosition().toPoint() - self._drag_pos
        self.move(self.pos() + delta)
        self._drag_pos = event.globalPosition().toPoint()
        event.accept()

    def _title_mouse_double_click(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            QTimer.singleShot(100, self._toggle_max_restore)

    def _toggle_max_restore(self) -> None:
        from nd2studios.widgets.icon_button import make_icon
        if self._is_maximized:
            self.showNormal()
            self._is_maximized = False
            self._btn_max.setIcon(make_icon("fa5s.window-maximize"))
            for grip in self._grips:
                grip.show()
        else:
            self.showMaximized()
            self._is_maximized = True
            self._btn_max.setIcon(make_icon("fa5s.window-restore"))
            for grip in self._grips:
                grip.hide()

    def _install_grips(self) -> None:
        # Edge grips for resizing the frameless window.
        self._grips = [
            CustomGrip(self, Qt.LeftEdge),
            CustomGrip(self, Qt.RightEdge),
            CustomGrip(self, Qt.TopEdge),
            CustomGrip(self, Qt.BottomEdge),
        ]
        self._size_grip = QSizeGrip(self)
        self._size_grip.setFixedSize(Settings.GRIP_SIZE, Settings.GRIP_SIZE)
        self._reposition_grips()

    def _reposition_grips(self) -> None:
        if not hasattr(self, "_grips"):
            return
        g = Settings.GRIP_SIZE
        w, h = self.width(), self.height()
        # Edges: hug the outer 10 px margin we left on `outer`.
        self._grips[0].setGeometry(0, g, g, h - 2 * g)            # left
        self._grips[1].setGeometry(w - g, g, g, h - 2 * g)        # right
        self._grips[2].setGeometry(0, 0, w, g)                    # top
        self._grips[3].setGeometry(0, h - g, w, g)                # bottom
        self._size_grip.move(w - g, h - g)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        self._reposition_grips()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        """Drain the V1.37 Phase 5 job runner before exit.

        Without this hook, Python's GC may release a ``QThread``
        wrapper while the underlying OS thread is still executing
        a preview job — the source of "QThread: Destroyed while
        thread is still running" warnings.

        Pages may also expose ``on_close`` to flush their own
        debounce timers or wait on legacy ``QThread`` workers
        they still own.
        """
        try:
            self.job_runner.shutdown(wait_ms=3000)
        except Exception:  # noqa: BLE001 — shutdown must never raise
            pass
        for page in self.pages.values():
            on_close = getattr(page, "on_close", None)
            if callable(on_close):
                try:
                    on_close()
                except Exception:  # noqa: BLE001
                    pass
        super().closeEvent(event)

    def _reset_active_viewer_zoom(self) -> None:
        page = self.pages.get(self._current_page_key)
        # ImportPage exposes reset_all_viewers() to fit every panel at once.
        reset_all = getattr(page, "reset_all_viewers", None)
        if callable(reset_all):
            reset_all()
            return
        viewer = getattr(page, "viewer", None)
        if viewer and hasattr(viewer, "canvas"):
            viewer.canvas.reset_zoom()

    # ── Navigation ─────────────────────────────────────────────────
    def _navigate(self, page_key: str) -> None:
        prereq = Settings.PAGE_PREREQS.get(page_key)
        if prereq is not None and self.exp_manager.active is not None:
            required, msg = prereq
            order = Settings.STATUS_ORDER
            if order.index(self.exp_manager.active.status) < order.index(required):
                QMessageBox.information(self, "Step Required", msg)
                # Re-check the previous nav button.
                if self._current_page_key:
                    self._nav_buttons[self._current_page_key].setChecked(True)
                return

        # Save state from outgoing page if it implements `save_to_experiment`.
        if (self._current_page_key
                and self.exp_manager.active is not None):
            old_page = self.pages.get(self._current_page_key)
            if hasattr(old_page, "save_to_experiment"):
                old_page.save_to_experiment(self.exp_manager.active)

            # V1.38 Phase 6 — release-on-leave for committed stages.
            # Frees RAM held by the outgoing page when the workspace has
            # a persistent copy of its output (recipe → re-applies
            # lazily; analysis → reads back from disk on demand).
            self._release_outgoing_stage(self._current_page_key)

        # Switch.
        keys = [p[0] for p in Settings.PAGES]
        idx = keys.index(page_key)
        self._stack.setCurrentIndex(idx)
        self._current_page_key = page_key
        self._nav_buttons[page_key].setChecked(True)

        # Record page navigation for macro playback.
        if self.macro_recorder.recording and not self.macro_recorder.replaying:
            from nd2studios.backend.macro_engine import MacroAction as _MA
            self.record_macro_action(_MA("navigate", f"Navigate: {page_key}", {"page": page_key}))

        # Notify incoming page.
        new_page = self.pages.get(page_key)
        if hasattr(new_page, "on_activated"):
            new_page.on_activated()

    def _on_experiment_changed(self) -> None:
        exp = self.exp_manager.active
        if exp is None:
            return
        # Push state into every page that wants it.
        for page in self.pages.values():
            if hasattr(page, "load_from_experiment"):
                page.load_from_experiment(exp)

    def _on_status_changed(self, status: str) -> None:
        # Data-status badge removed in V1.44; status text still lands in the
        # bottom status bar via the worker callbacks.
        return

    # ── V1.38 Phase 6 — workspace lifecycle ────────────────────────
    def attach_session_for(self, filepath: str) -> Optional[Session]:
        """Create or reuse the per-source workspace for ``filepath``.

        Called from :meth:`ImportPage._on_confirm` after a successful
        load. Honours ``ND2STUDIOS_DISABLE_WORKSPACE`` as a no-op
        escape hatch — returns ``None`` in that mode so callers can
        skip the warm-start prompt.
        """
        if workspace_disabled():
            self.session = None
            return None
        try:
            self.session = Session(filepath)
        except Exception as exc:  # noqa: BLE001 — workspace must never crash import
            self.set_status_text(f"Workspace unavailable ({exc.__class__.__name__})")
            self.session = None
            return None
        exp = self.exp_manager.active
        if exp is not None:
            self.session.set_shape(
                n_t=exp.n_frames,
                n_m=max(1, exp.n_multipoints),
                n_z=max(1, exp.n_zslices),
                n_channels=len(exp.channel_display) or len(exp._raw_channels or {}),
                height=exp.frame_height,
                width=exp.frame_width,
                pixel_size_um=exp.pixel_size_um,
                channel_names=list((exp._raw_channels or {}).keys()),
            )
        return self.session

    def detach_session(self) -> None:
        """Forget the active workspace (e.g. after ``New session``)."""
        self.session = None

    def recipe_stage(self) -> Optional["RecipeStage"]:
        return RecipeStage(self.session) if self.session is not None else None

    def analysis_stage(self, pipeline_name: str) -> Optional["AnalysisStage"]:
        if self.session is None:
            return None
        return AnalysisStage(self.session, pipeline_name)

    # ── V1.39 Phase 7 — pyramid lifecycle ──────────────────────────
    def pyramid_stage(self) -> Optional["PyramidStage"]:
        """Return a :class:`PyramidStage` for the active session, or ``None``.

        Returns ``None`` when no session is attached, when the env
        override disables pyramids, or when ``zarr`` is missing.
        Pyramid reads require zarr's sub-chunk indexing; without it
        we keep the viewer pinned to level 0.
        """
        if self.session is None:
            return None
        if os.environ.get(
            getattr(Settings, "PYRAMID_ENV_DISABLE", "ND2_DISABLE_PYRAMIDS"),
            "",
        ) == "1":
            return None
        if not HAS_ZARR:
            return None
        return PyramidStage(self.session)

    def start_pyramid_build(self, volume) -> bool:
        """Submit a :class:`BuildPyramidJob` for ``volume``.

        Returns True iff a build was actually submitted. Callers (the
        Import page) use the return value to decide whether to show
        the "Building pyramid..." status note. Submitting a build
        while one is in flight coalesces — the in-flight job is
        cancelled and the new one takes over.
        """
        stage = self.pyramid_stage()
        if stage is None or volume is None:
            return False
        # Already committed for this source? Just (re-)attach.
        if stage.is_committed():
            self._attach_pyramid_reader_to_viewers(stage.reader())
            return False
        try:
            self.job_runner.submit(
                BuildPyramidJob("pyramid_build", stage, volume)
            )
        except Exception as exc:  # noqa: BLE001 — never crash on a soft optimization
            self.set_status_text(
                f"Pyramid build skipped ({type(exc).__name__}: {exc})"
            )
            return False
        self.set_status_text("Building multi-resolution pyramid in background…")
        return True

    def _on_pyramid_job_done(self, result) -> None:
        """Handle the ``pyramid_build`` :class:`JobResult`.

        Connected to :attr:`JobRunner.job_done` once at startup; we
        filter by ``result.key`` so other consumers (analysis preview
        / commit) keep their existing slot wiring.
        """
        if getattr(result, "key", None) != "pyramid_build":
            return
        if not getattr(result, "ok", False):
            err = getattr(result, "error", "unknown error")
            self.set_status_text(f"Pyramid build failed: {err}")
            return
        # Reattach the freshly-built reader to whichever viewer is
        # currently bound to the active experiment.
        stage = self.pyramid_stage()
        reader = stage.reader() if stage is not None else None
        if reader is None:
            self.set_status_text("Pyramid built (reader unavailable).")
            return
        n_levels = reader.n_levels
        self._attach_pyramid_reader_to_viewers(reader)
        self.set_status_text(f"Pyramid built: {n_levels} levels.")

    def _attach_pyramid_reader_to_viewers(self, reader) -> None:
        """Push a :class:`PyramidReader` into every page that owns a viewer.

        The multi-axis viewer is the only consumer today, but the
        method scans all pages defensively so a future page that
        exposes ``viewer`` automatically benefits.
        """
        if reader is None:
            return
        for page in self.pages.values():
            viewer = getattr(page, "viewer", None)
            if viewer is None:
                continue
            attach = getattr(viewer, "attach_pyramid", None)
            if callable(attach):
                try:
                    attach(reader)
                except Exception:  # noqa: BLE001 — viewer must never break on attach
                    pass

    # ── V1.39 Phase 7 / V1.41 — Performance settings dialog ────────
    def _open_performance_settings(self) -> None:
        """Show the Performance + Diagnostics settings modal.

        V1.41 expands the original two-checkbox dialog into a tabbed
        view:

        * **GPU** — the original opt-in for GPU analysis dispatch.
        * **Memory** — live RAM gauge + ``LoadStrategy`` override.
        * **Storage** — recorded storage class probe results.
        * **Diagnostics** — Run Benchmark + Save Snapshot buttons.

        State still lives on :class:`Settings` for the session;
        persistence across launches is wired into
        :mod:`nd2studios.utils.user_config`.
        """
        from PySide6.QtWidgets import (
            QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
            QLabel, QPushButton, QTabWidget, QVBoxLayout, QWidget,
        )

        from nd2studios.compute.gpu import configure as gpu_configure
        from nd2studios.compute.gpu import gpu_status

        dlg = QDialog(self)
        dlg.setWindowTitle("Performance & Diagnostics")
        dlg.resize(560, 420)
        root = QVBoxLayout(dlg)
        root.setContentsMargins(16, 16, 16, 12)
        tabs = QTabWidget(dlg)
        root.addWidget(tabs)

        # ── GPU tab ────────────────────────────────────────────────
        gpu_tab = QWidget()
        gpu_layout = QVBoxLayout(gpu_tab)

        status = gpu_status()
        if status["available"]:
            mem = status.get("memory_gb", 0.0)
            free_mem = status.get("free_memory_gb", mem)
            gpu_label = (
                f"Use GPU acceleration ({status['device_name']}, "
                f"{mem:.1f} GB total, {free_mem:.1f} GB free)"
            )
            if not status.get("cucim", False):
                gpu_label += "  — cucim missing; install for full speedup"
        else:
            gpu_label = f"Use GPU acceleration — unavailable: {status['reason']}"
        gpu_cb = QCheckBox(gpu_label)
        gpu_cb.setChecked(bool(getattr(Settings, "USE_GPU_ANALYSIS", False))
                          and status["available"])
        gpu_cb.setEnabled(bool(status["available"]))
        gpu_layout.addWidget(gpu_cb)

        pyr_cb = QCheckBox("Build multi-resolution pyramids on import")
        pyr_cb.setChecked(bool(getattr(Settings, "BUILD_PYRAMIDS", True)))
        pyr_cb.setEnabled(HAS_ZARR)
        if not HAS_ZARR:
            pyr_cb.setToolTip(
                "Pyramids require the optional `zarr` package. "
                "Install with `pip install zarr`."
            )
        gpu_layout.addWidget(pyr_cb)

        rebuild_btn = QPushButton("Rebuild pyramid for current file")
        rebuild_btn.setEnabled(self.session is not None and HAS_ZARR)
        gpu_layout.addWidget(rebuild_btn)
        gpu_layout.addWidget(QLabel(
            f"Workspace: {self.session.session_dir if self.session else '— (no file imported)'}",
        ))
        gpu_layout.addStretch(1)
        tabs.addTab(gpu_tab, "GPU")

        # ── Memory tab ─────────────────────────────────────────────
        mem_tab = QWidget()
        mem_layout = QVBoxLayout(mem_tab)
        from nd2studios.utils.resources import detect
        res = detect()
        mem_layout.addWidget(QLabel(
            f"Total RAM:     {res.total_ram_gb:.1f} GB"
        ))
        mem_layout.addWidget(QLabel(
            f"Available RAM: {res.available_ram_gb:.1f} GB"
        ))
        if self.memory_monitor is not None:
            mem_layout.addWidget(QLabel(
                f"Current use:   {self.memory_monitor.current_percent():.0f}%"
                f" (band: {self.memory_monitor.current_band().name})"
            ))
        mem_layout.addWidget(QLabel(
            f"Cache budget:  {(res.available_ram_bytes * 0.4) / 1024**3:.2f} GB "
            "(40% of available)"
        ))
        mem_layout.addWidget(QLabel("Load strategy override:"))
        strategy_combo = QComboBox()
        strategy_combo.addItem("Auto", "")
        strategy_combo.addItem("Force Eager (full)", "eager_full")
        strategy_combo.addItem("Force Eager (Z-collapsed)", "eager_reduced")
        strategy_combo.addItem("Force Lazy (cached)", "lazy_cached")
        current_forced = getattr(Settings, "FORCED_LOAD_STRATEGY", "") or ""
        idx = max(0, strategy_combo.findData(current_forced))
        strategy_combo.setCurrentIndex(idx)
        mem_layout.addWidget(strategy_combo)
        mem_layout.addStretch(1)
        tabs.addTab(mem_tab, "Memory")

        # ── Storage tab ───────────────────────────────────────────
        st_tab = QWidget()
        st_layout = QVBoxLayout(st_tab)
        from nd2studios.utils.storage import _load_probe_cache
        try:
            probes = _load_probe_cache()
        except Exception:
            probes = {}
        if probes:
            st_layout.addWidget(QLabel("Cached storage-class probes:"))
            for drive, fast in sorted(probes.items()):
                st_layout.addWidget(QLabel(
                    f"  {drive}  {'fast (NVMe/SSD)' if fast else 'slow / unknown'}"
                ))
        else:
            st_layout.addWidget(QLabel(
                "No drives probed yet — class is measured on first file open."
            ))
        st_layout.addStretch(1)
        tabs.addTab(st_tab, "Storage")

        # ── Diagnostics tab ───────────────────────────────────────
        diag_tab = QWidget()
        diag_layout = QVBoxLayout(diag_tab)
        diag_layout.addWidget(QLabel(
            "Save a JSON snapshot of the current detection state for support tickets."
        ))
        snapshot_btn = QPushButton("Save snapshot…")
        diag_layout.addWidget(snapshot_btn)
        diag_layout.addStretch(1)
        tabs.addTab(diag_tab, "Diagnostics")

        # ── OK / Cancel ───────────────────────────────────────────
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
        )
        root.addWidget(buttons)

        def _do_rebuild() -> None:
            exp = self.exp_manager.active
            volume = getattr(exp, "_raw_volume", None) if exp is not None else None
            if volume is None:
                self.set_status_text("No active volume to build a pyramid from.")
                return
            stage = self.pyramid_stage()
            if stage is None:
                return
            self.session.remove_stage(stage.name, delete_artifacts=True)  # type: ignore[union-attr]
            self.start_pyramid_build(volume)
            dlg.accept()

        def _save_snapshot() -> None:
            from datetime import datetime
            import json
            snap = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "system": {
                    "total_ram_gb": res.total_ram_gb,
                    "available_ram_gb": res.available_ram_gb,
                    "cpu_physical": res.cpu_count_physical,
                    "cpu_logical": res.cpu_count_logical,
                },
                "gpu": status,
                "storage_probes": probes,
                "settings": {
                    "USE_GPU_DISPLAY": getattr(Settings, "USE_GPU_DISPLAY", None),
                    "USE_GPU_ANALYSIS": getattr(Settings, "USE_GPU_ANALYSIS", None),
                    "BUILD_PYRAMIDS": getattr(Settings, "BUILD_PYRAMIDS", None),
                    "EAGER_MAX_FRACTION": getattr(Settings, "EAGER_MAX_FRACTION", None),
                },
            }
            target, _ = QFileDialog.getSaveFileName(
                dlg,
                "Save diagnostics snapshot",
                f"nd2studios_diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "JSON (*.json)",
            )
            if not target:
                return
            try:
                with open(target, "w", encoding="utf-8") as f:
                    json.dump(snap, f, indent=2, default=str)
                self.set_status_text(f"Snapshot saved: {target}")
            except OSError as exc:
                QMessageBox.warning(dlg, "Snapshot failed", str(exc))

        rebuild_btn.clicked.connect(_do_rebuild)
        snapshot_btn.clicked.connect(_save_snapshot)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            Settings.USE_GPU_ANALYSIS = bool(gpu_cb.isChecked())
            Settings.BUILD_PYRAMIDS = bool(pyr_cb.isChecked())
            Settings.FORCED_LOAD_STRATEGY = strategy_combo.currentData() or ""
            effective = gpu_configure(Settings.USE_GPU_ANALYSIS)
            if Settings.USE_GPU_ANALYSIS and not effective:
                self.set_status_text(
                    "GPU acceleration requested but unavailable — staying on CPU."
                )
            else:
                self.set_status_text(
                    "Performance settings updated "
                    f"(GPU analysis: {'on' if effective else 'off'}, "
                    f"pyramids: {'on' if Settings.BUILD_PYRAMIDS else 'off'}, "
                    f"strategy: {Settings.FORCED_LOAD_STRATEGY or 'auto'})."
                )
            # Persist to ~/.config/nd2studios/preferences.json so the
            # next launch picks up the user's choices.
            try:
                from nd2studios.utils.user_config import save_user_preferences
                save_user_preferences()
            except Exception:  # noqa: BLE001 — never block dialog close on a disk error
                pass

    # ── Macro recorder ─────────────────────────────────────────────
    def _open_macro(self) -> None:
        from nd2studios.widgets.macro_dialog import MacroDialog
        if self._macro_dialog is None or not self._macro_dialog.isVisible():
            self._macro_dialog = MacroDialog(self, parent=self)
            self._macro_dialog.show()
        else:
            self._macro_dialog.raise_()
            self._macro_dialog.activateWindow()

    def start_recording(self) -> None:
        """Start a new macro recording session (called by MacroDialog)."""
        self.macro_recorder.start()
        self.macro_event_filter.install()

    def stop_recording(self, cancel: bool = False):
        """Stop recording and return the captured actions (called by MacroDialog)."""
        self.macro_event_filter.uninstall()
        if cancel:
            self.macro_recorder.cancel()
            return []
        return self.macro_recorder.finish()

    def record_macro_action(self, action: "MacroAction") -> None:  # type: ignore[name-defined]
        """Record *action* and emit signal.  No-op when replaying."""
        from dataclasses import asdict
        if self.macro_recorder.record(action):
            self.macro_action_recorded.emit(asdict(action))

    def upgrade_last_macro_action(self, action: "MacroAction") -> None:  # type: ignore[name-defined]
        """Replace the last generic event-filter action with a richer semantic one."""
        from dataclasses import asdict
        if self.macro_recorder.record_upgrade(action):
            self.macro_action_recorded.emit(asdict(action))

    def replay_macro_action(self, action: "MacroAction") -> None:  # type: ignore[name-defined]
        """Dispatch one macro action to the appropriate page for replay."""
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtWidgets import QAbstractSpinBox, QCheckBox, QComboBox, QPushButton
        from nd2studios.core.macro_event_filter import find_widget

        self.macro_recorder.start_replay()
        try:
            t = action.action_type

            # ── Navigation ──
            if t == "navigate":
                page_key = action.params.get("page", "")
                if page_key:
                    self._navigate(page_key)

            # ── Generic button click ──
            elif t == "button_click":
                import time as _time
                page_key = action.params.get("page", "")
                widget_id = action.params.get("widget_id", "")
                page = self.pages.get(page_key)
                if page is None:
                    return
                self._navigate(page_key)
                QCoreApplication.processEvents()
                widget = find_widget(page, widget_id)
                if isinstance(widget, QPushButton):
                    # Buttons are often temporarily disabled while an async
                    # operation (analysis, recipe commit) is finishing.
                    # Retry for up to 3 s before giving up.
                    if not widget.isEnabled():
                        deadline = _time.time() + 3.0
                        while not widget.isEnabled() and _time.time() < deadline:
                            QCoreApplication.processEvents()
                    if widget.isEnabled():
                        widget.click()
                        # If the click started a worker (progress bar appears),
                        # wait for it to finish before advancing to the next action.
                        _time.sleep(0.15)
                        QCoreApplication.processEvents()
                        if self._progress_bar.isVisible():
                            deadline = _time.time() + 120.0
                            while self._progress_bar.isVisible() and _time.time() < deadline:
                                QCoreApplication.processEvents()

            # ── Combo box ──
            elif t == "combo_change":
                page_key = action.params.get("page", "")
                widget_id = action.params.get("widget_id", "")
                value = str(action.params.get("value", ""))
                page = self.pages.get(page_key)
                if page is None:
                    return
                widget = find_widget(page, widget_id)
                if isinstance(widget, QComboBox):
                    widget.setCurrentText(value)

            # ── Spinbox ──
            elif t == "spinbox_change":
                page_key = action.params.get("page", "")
                widget_id = action.params.get("widget_id", "")
                value = action.params.get("value", 0)
                page = self.pages.get(page_key)
                if page is None:
                    return
                widget = find_widget(page, widget_id)
                if isinstance(widget, QAbstractSpinBox):
                    widget.setValue(float(value))

            # ── Checkbox ──
            elif t == "checkbox_change":
                page_key = action.params.get("page", "")
                widget_id = action.params.get("widget_id", "")
                value = bool(action.params.get("value", False))
                page = self.pages.get(page_key)
                if page is None:
                    return
                widget = find_widget(page, widget_id)
                if isinstance(widget, QCheckBox):
                    widget.setChecked(value)

            # ── ParamEditor ──
            elif t == "param_change":
                page_key = action.params.get("page", "")
                widget_id = action.params.get("widget_id", "")
                values = action.params.get("values", {})
                page = self.pages.get(page_key)
                if page is None:
                    return
                widget = find_widget(page, widget_id)
                if widget is not None and hasattr(widget, "set_values"):
                    widget.set_values(values)

            # ── Configuration / recipe / template loading ──
            elif t == "load_recipe":
                page = self.pages.get("recipe")
                if page is None:
                    return
                self._navigate("recipe")
                QCoreApplication.processEvents()
                page._replay_load_recipe(action)

            elif t == "load_config":
                self._replay_load_path = action.params.get("path", "")
                try:
                    self._load_config()
                finally:
                    self._replay_load_path = ""

            elif t == "load_template":
                page = self.pages.get("batch")
                if page is None:
                    return
                self._navigate("batch")
                QCoreApplication.processEvents()
                page._load_template_from_path(action.params.get("path", ""))

            # ── Channel display state (enable/disable, color, LUT) ──
            elif t == "channel_state":
                page_key = action.params.get("page", "")
                viewer_attr = action.params.get("viewer_attr", "viewer")
                channels = action.params.get("channels", {})
                page = self.pages.get(page_key)
                if page is None:
                    return
                self._navigate(page_key)
                QCoreApplication.processEvents()
                viewer = getattr(page, viewer_attr, None)
                if viewer is not None and hasattr(viewer, "apply_channel_state"):
                    viewer.apply_channel_state(channels)

            # ── Semantic actions (rich records from page hooks) ──
            elif t in ("recipe_add_step", "recipe_remove_last", "recipe_clear"):
                page = self.pages.get("recipe")
                if page is None:
                    return
                self._navigate("recipe")
                QCoreApplication.processEvents()
                if t == "recipe_add_step":
                    page._replay_add_step(action)
                elif t == "recipe_remove_last":
                    page._on_remove_last()
                elif t == "recipe_clear":
                    page._on_clear()

            elif t == "analysis_run":
                page = self.pages.get("analysis")
                if page is None:
                    return
                self._navigate("analysis")
                QCoreApplication.processEvents()
                page._replay_run(action)

            elif t in ("export_tiff", "export_composite", "export_movie",
                       "export_image_sequence", "export_tracked_objects"):
                page = self.pages.get("export")
                if page is None:
                    return
                self._navigate("export")
                QCoreApplication.processEvents()
                page._replay_export(action)

            elif t in ("export_csv", "export_label_masks", "export_overlay_images"):
                page = self.pages.get("results")
                if page is None:
                    return
                self._navigate("results")
                QCoreApplication.processEvents()
                page._replay_result_export(action)

        finally:
            self.macro_recorder.stop_replay()

        QCoreApplication.processEvents()

    def _release_outgoing_stage(self, page_key: str) -> None:
        """Drop in-memory stage outputs when the workspace owns them.

        Called from :meth:`_navigate` after the outgoing page has
        flushed its state. The release is conditional on a successful
        commit in the workspace manifest — we never throw data away if
        the on-disk copy is missing.
        """
        if self.session is None:
            return
        exp = self.exp_manager.active
        if exp is None:
            return

        if page_key == "recipe" and self.session.is_committed("recipe"):
            self._release_processed_channels(exp)
        elif page_key == "analysis":
            self._release_committed_analysis(exp)

    def _release_processed_channels(self, exp) -> None:
        """Swap ``_processed_channels`` for a lazy ``EnhancedDataset``.

        Skips when ``_raw_channels`` is missing (nothing to reapply
        against) or when the recipe is empty (the processed dict is
        a cheap alias to raw — no win from dropping it).
        """
        if not exp._raw_channels:
            return
        recipe = list(exp.recipe)
        if not recipe and not exp.recipe_normalized:
            return
        if exp._processed_channels is None:
            return
        exp._processed_view = EnhancedDataset(
            raw_channels=exp._raw_channels,
            recipe=recipe,
            normalized=bool(exp.recipe_normalized),
            pixel_size_um=exp.pixel_size_um,
        )
        exp._processed_channels = None
        self.set_status_text("Recipe stage released to workspace.")

    def _release_committed_analysis(self, exp) -> None:
        """Drop label-mask arrays for analysis results that hit disk.

        We keep the per-pipeline ``measurements`` and ``summary`` in
        RAM because they are small and the Results page reads them
        directly. ``label_masks`` is the multi-GB payload — that comes
        back via :meth:`AnalysisStage.rehydrate_m` on demand.
        """
        if not exp.analysis_results:
            return
        released_any = False
        for pipeline_name, per_m in list(exp.analysis_results.items()):
            stage = self.analysis_stage(pipeline_name)
            if stage is None or not stage.is_committed():
                continue
            committed_ms = set(stage.committed_m_indices())
            if isinstance(per_m, dict):
                for m, result in per_m.items():
                    if m in committed_ms and getattr(result, "label_masks", None):
                        result.label_masks = {}
                        result.secondary_label_masks = {}
                        released_any = True
            else:
                # Legacy single-result shape — only release if the
                # workspace has m=0 committed.
                if 0 in committed_ms and getattr(per_m, "label_masks", None):
                    per_m.label_masks = {}
                    per_m.secondary_label_masks = {}
                    released_any = True
        if released_any:
            self.set_status_text("Analysis stage released to workspace.")

    # ── V1.41 memory monitor slots ────────────────────────────────
    def _on_memory_sampled(self, percent: float) -> None:
        """Update the bottom-bar gauge on every sample.

        The label uses GB-of-total-RAM-used rather than just the
        percent so the user knows where the boundary is in absolute
        terms — 80% of 16 GB and 80% of 64 GB feel very different.
        """
        try:
            import psutil
            vm = psutil.virtual_memory()
            used_gb = (vm.total - vm.available) / (1024 ** 3)
            total_gb = vm.total / (1024 ** 3)
            text = f"RAM {used_gb:.1f}/{total_gb:.1f} GB ({percent:.0f}%)"
        except Exception:  # noqa: BLE001
            text = f"RAM {percent:.0f}%"
        # Colour bands match the same thresholds the monitor uses.
        if percent >= getattr(Settings, "MEMORY_PRESSURE_EMERGENCY_PCT", 95.0):
            color = Settings.ACCENT_RED
        elif percent >= getattr(Settings, "MEMORY_PRESSURE_CRITICAL_PCT", 90.0):
            color = Settings.ACCENT_ORANGE
        elif percent >= getattr(Settings, "MEMORY_PRESSURE_WARNING_PCT", 80.0):
            color = Settings.ACCENT_YELLOW
        else:
            color = Settings.FG_SECONDARY
        self._memory_gauge.setStyleSheet(f"color: {color}; font: 8pt;")
        self._memory_gauge.setText(text)

    def _on_memory_critical(self, percent: float) -> None:
        """Drop pre-render caches on every active viewer."""
        self.set_status_text(
            f"Memory critical ({percent:.0f}%) — dropping render caches."
        )
        self._drop_viewer_caches()

    def _on_memory_emergency(self, percent: float) -> None:
        """Block new heavy ops and surface a one-time modal warning."""
        self._heavy_ops_blocked = True
        if self._memory_modal_shown:
            return
        self._memory_modal_shown = True
        QMessageBox.critical(
            self,
            "Memory critical",
            (
                f"System memory is at {percent:.0f}% — close other "
                f"applications before starting new recipes, analyses, or "
                f"exports.  The current session remains usable but new "
                f"heavy operations are blocked until pressure drops."
            ),
        )

    def _on_memory_recovered(self, percent: float) -> None:
        """Re-enable heavy ops once pressure clears."""
        self._heavy_ops_blocked = False
        self._memory_modal_shown = False
        self.set_status_text(f"Memory recovered ({percent:.0f}%).")

    def _drop_viewer_caches(self) -> None:
        """Walk active pages and invalidate any viewer render caches.

        Each :class:`~nd2studios.widgets.multi_axis_viewer.MultiAxisViewer`
        owns its own caches; calling ``_invalidate_render_cache`` on
        each instance is the cheapest way to free the 6+ GB the
        pre-render pipeline can occupy.
        """
        from nd2studios.widgets.multi_axis_viewer import MultiAxisViewer
        for viewer in self.findChildren(MultiAxisViewer):
            try:
                invalidate = getattr(viewer, "_invalidate_render_cache", None)
                if callable(invalidate):
                    invalidate()
            except Exception:  # noqa: BLE001 — best-effort, never crash
                continue

    def heavy_ops_blocked(self) -> bool:
        """Public accessor pages consult before starting a heavy worker."""
        return self._heavy_ops_blocked

    # ── Public hooks the pages call ────────────────────────────────
    def set_progress(self, value: int, text: str = "") -> None:
        self._progress_bar.setVisible(value > 0)
        self._progress_bar.setValue(value)
        if text:
            self._status_text.setText(text)

    def set_status_text(self, text: str) -> None:
        self._status_text.setText(text)

    def get_confirmed_records(self):
        """Return all confirmed ND2StudiosRecord objects from the Import page."""
        page = self.pages.get("import")
        if page is None:
            return []
        fn = getattr(page, "get_confirmed_records", None)
        if callable(fn):
            return fn()
        # Fallback: single active record if it has been imported.
        exp = self.exp_manager.active
        if exp is not None and getattr(exp, "status", None) not in (None, "new"):
            return [exp]
        return []

    # ── Config controls ────────────────────────────────────────────
    def _configure(self) -> None:
        from PySide6.QtWidgets import QDialog
        from nd2studios.widgets.config_wizard import ConfigWizard
        recipe_page = self.pages.get("recipe")
        analysis_page = self.pages.get("analysis")
        results_page = self.pages.get("results")
        dlg = ConfigWizard(
            recipe_page, analysis_page, results_page,
            parent=self, initial_path=self._cfg_path or "",
        )
        if self._cfg_data is not None:
            dlg.apply_config(self._cfg_data)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        # The wizard saved the file in _on_accept; record the baseline.
        self._cfg_path = dlg.save_path
        self._cfg_data = dlg.get_config()

        # Persist into the active experiment so load_from_experiment
        # re-applies everything correctly on future tab visits.
        exp = self.exp_manager.active
        if exp is not None:
            cfg = self._cfg_data
            if "analysis" in cfg:
                exp.analysis_config = dict(cfg["analysis"])
            if "results" in cfg:
                exp.results_config = dict(cfg["results"])
            processing = cfg.get("processing", {})
            recipe_list = processing.get("recipe")
            if recipe_list is not None:
                exp.recipe = [
                    (s["name"], dict(s.get("params", {}))) for s in recipe_list
                ]
            if "normalized" in processing:
                exp.recipe_normalized = bool(processing["normalized"])

        self.set_status_text(f"Config saved: {os.path.basename(self._cfg_path)}")

    def _save_config(self) -> None:
        from PySide6.QtWidgets import QDialog
        from nd2studios.widgets.config_wizard import (
            CONFIG_EXTENSION, SavePreviewDialog, build_config_from_pages,
        )

        if self._cfg_data is None:
            # No baseline — open Configure (it handles file naming and saving).
            self._configure()
            return

        recipe_page = self.pages.get("recipe")
        analysis_page = self.pages.get("analysis")
        results_page = self.pages.get("results")

        # Baseline exists — show a diff of what changed.
        new_cfg = build_config_from_pages(analysis_page, results_page, recipe_page)
        preview = SavePreviewDialog(
            old_config=self._cfg_data,
            new_config=new_cfg,
            default_path=self._cfg_path or "",
            parent=self,
        )
        if preview.exec() != QDialog.DialogCode.Accepted:
            return
        path = preview.save_path
        if not path:
            from nd2studios.core.settings import Settings
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Configuration", Settings.PROJECT_DIR,
                f"ND2Studios Config (*{CONFIG_EXTENSION})",
            )
            if not path:
                return
        if not path.endswith(CONFIG_EXTENSION):
            path += CONFIG_EXTENSION
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(new_cfg, fh, indent=2)
        self._cfg_path = path
        self._cfg_data = new_cfg
        self.set_status_text(f"Config saved: {os.path.basename(path)}")

    def _load_config(self) -> None:
        from nd2studios.widgets.config_wizard import (
            CONFIG_EXTENSION, apply_config_to_pages,
        )
        _rp = getattr(self, "_replay_load_path", "")
        from nd2studios.core.settings import Settings
        path = _rp or QFileDialog.getOpenFileName(
            self, "Load Configuration", Settings.PROJECT_DIR,
            f"ND2Studios Config (*{CONFIG_EXTENSION})",
        )[0]
        if not path:
            return
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        recipe_page = self.pages.get("recipe")
        analysis_page = self.pages.get("analysis")
        results_page = self.pages.get("results")
        apply_config_to_pages(cfg, analysis_page, results_page, recipe_page)

        # Persist config into the active experiment record so that
        # load_from_experiment (called on future tab switches and experiment
        # changes) re-applies the settings correctly.
        exp = self.exp_manager.active
        if exp is not None:
            if "analysis" in cfg:
                exp.analysis_config = dict(cfg["analysis"])
            if "results" in cfg:
                exp.results_config = dict(cfg["results"])
            processing = cfg.get("processing", {})
            recipe_list = processing.get("recipe")
            if recipe_list is not None:
                exp.recipe = [
                    (s["name"], dict(s.get("params", {}))) for s in recipe_list
                ]
            if "normalized" in processing:
                exp.recipe_normalized = bool(processing["normalized"])

        self._cfg_path = path
        self._cfg_data = cfg
        self.set_status_text(f"Config loaded: {os.path.basename(path)}")
        from nd2studios.backend.macro_engine import MacroAction as _MA
        self.upgrade_last_macro_action(_MA(
            "load_config", f"Load Config: {path}", {"path": path}
        ))

    # ── Session controls (internal / legacy) ───────────────────────
    def _new_session(self) -> None:
        self.exp_manager.new_experiment("Untitled")
        self.detach_session()
        self.set_status_text("New session")

    def _save_session(self) -> None:
        if self.exp_manager.active is None:
            return
        # Let pages flush their state first.
        for page in self.pages.values():
            if hasattr(page, "save_to_experiment"):
                page.save_to_experiment(self.exp_manager.active)
        from nd2studios.core.settings import Settings
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", Settings.PROJECT_DIR,
            f"ND2Studios Session (*{SESSION_EXTENSION})",
        )
        if path:
            self.exp_manager.save_session(path)
            self.set_status_text(f"Saved: {os.path.basename(path)}")

    def _load_session(self) -> None:
        from nd2studios.core.settings import Settings
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", Settings.PROJECT_DIR,
            f"ND2Studios Session (*{SESSION_EXTENSION})",
        )
        if path:
            self.exp_manager.load_session(path)
            self.set_status_text(f"Loaded: {os.path.basename(path)}")
