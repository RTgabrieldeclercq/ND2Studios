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
    QEasingCurve, QEvent, QPropertyAnimation, QSize, Qt, QTimer,
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
from nd2studios.widgets.common import StatusIndicator
from nd2studios.widgets.custom_grips import CustomGrip


class MainWindow(QMainWindow):
    """Frameless main window with collapsible sidebar and three pages."""

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
        self._sidebar_animation: Optional[QPropertyAnimation] = None
        self._sidebar_expanded: bool = True
        self._drag_pos = None
        self._is_maximized = False

        # Config file state — set on Load or after a successful Save.
        self._cfg_path: Optional[str] = None
        self._cfg_data: Optional[Dict[str, Any]] = None

        self._build_ui()
        self._install_grips()

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

        # ── Body: sidebar + content area ──
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self._body_splitter = QSplitter(Qt.Horizontal)
        self._body_splitter.setChildrenCollapsible(False)
        self._body_splitter.addWidget(self._build_sidebar())
        self._body_splitter.addWidget(self._build_content_area())
        self._body_splitter.setStretchFactor(0, 0)
        self._body_splitter.setStretchFactor(1, 1)
        body_layout.addWidget(self._body_splitter, stretch=1)

        bg_layout.addWidget(body, stretch=1)

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

        # Min / Max / Close
        self._btn_min = QPushButton("—")
        self._btn_min.setObjectName("titleBarBtn")
        self._btn_min.setToolTip("Minimize")
        self._btn_min.clicked.connect(self.showMinimized)

        self._btn_max = QPushButton("□")
        self._btn_max.setObjectName("titleBarBtn")
        self._btn_max.setToolTip("Maximize / Restore")
        self._btn_max.clicked.connect(self._toggle_max_restore)

        self._btn_close = QPushButton("✕")
        self._btn_close.setObjectName("titleBarCloseBtn")
        self._btn_close.setToolTip("Close")
        self._btn_close.clicked.connect(self.close)

        for b in (self._btn_min, self._btn_max, self._btn_close):
            layout.addWidget(b)

        # The whole title bar acts as a drag handle.
        bar.mousePressEvent = self._title_mouse_press
        bar.mouseMoveEvent = self._title_mouse_move
        bar.mouseDoubleClickEvent = self._title_mouse_double_click
        return bar

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("leftMenuBg")
        sidebar.setMinimumWidth(Settings.SIDEBAR_COLLAPSED_WIDTH)
        self._sidebar = sidebar

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toggle (hamburger) at the top.
        self._toggle_btn = QPushButton("☰")
        self._toggle_btn.setObjectName("toggleBtn")
        self._toggle_btn.setToolTip("Collapse / Expand sidebar")
        self._toggle_btn.clicked.connect(self._toggle_sidebar)
        layout.addWidget(self._toggle_btn)

        # Nav buttons (one per page).
        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        self._nav_buttons: Dict[str, QPushButton] = {}

        for key, icon, title, tooltip in Settings.PAGES:
            btn = QPushButton(f"  {icon}    {title}")
            btn.setObjectName("navBtn")
            btn.setCheckable(True)
            btn.setToolTip(tooltip)
            btn.clicked.connect(lambda _checked, k=key: self._navigate(k))
            self._nav_group.addButton(btn)
            self._nav_buttons[key] = btn
            layout.addWidget(btn)

        layout.addStretch(1)

        # Controls at the bottom.
        for label, slot, tooltip in [
            ("  💾  Save",      self._save_config,   "Save configuration to a .nd2s_cfg file"),
            ("  📂  Load",      self._load_config,   "Load a .nd2s_cfg configuration file"),
            ("  🔧  Configure", self._configure,     "Open the pipeline configuration wizard"),
            ("  ⚙  Performance",
             self._open_performance_settings,
             "GPU acceleration and multi-resolution pyramid settings (V1.39)"),
        ]:
            btn = QPushButton(label)
            btn.setObjectName("sessionBtn")
            btn.setToolTip(tooltip)
            btn.clicked.connect(slot)
            layout.addWidget(btn)

        return sidebar

    def _build_content_area(self) -> QWidget:
        content = QWidget()
        content.setObjectName("contentArea")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Top bar: page title + status indicator.
        top_bar = QWidget()
        top_bar.setObjectName("topBar")
        top_bar.setFixedHeight(48)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(16, 0, 16, 0)

        self._title_label = QLabel(Settings.PAGES[0][2])
        self._title_label.setObjectName("titleLabel")
        top_layout.addWidget(self._title_label)
        top_layout.addStretch(1)

        self._status_indicator = StatusIndicator()
        top_layout.addWidget(self._status_indicator)

        layout.addWidget(top_bar)

        # Page stack — pages are imported lazily here so that core/ does
        # not have a hard cycle with pages/ at import time.
        from nd2studios.pages.import_page import ImportPage
        from nd2studios.pages.recipe_page import RecipePage
        from nd2studios.pages.export_page import ExportPage
        from nd2studios.pages.analysis_page import AnalysisPage
        from nd2studios.pages.results_page import ResultsPage
        from nd2studios.pages.batch_page import BatchPage

        page_classes = {
            "import": ImportPage,
            "recipe": RecipePage,
            "export": ExportPage,
            "analysis": AnalysisPage,
            "results": ResultsPage,
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
        if self._is_maximized:
            self.showNormal()
            self._is_maximized = False
            self._btn_max.setText("□")
            for grip in self._grips:
                grip.show()
        else:
            self.showMaximized()
            self._is_maximized = True
            self._btn_max.setText("❐")
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

    # ── Sidebar collapse/expand ────────────────────────────────────
    def _toggle_sidebar(self) -> None:
        start = self._sidebar.width()
        end = (
            Settings.SIDEBAR_COLLAPSED_WIDTH if self._sidebar_expanded
            else Settings.SIDEBAR_EXPANDED_WIDTH
        )
        self._sidebar_expanded = not self._sidebar_expanded

        anim = QPropertyAnimation(self._sidebar, b"minimumWidth")
        anim.setDuration(Settings.SIDEBAR_ANIMATION_MS)
        anim.setStartValue(start)
        anim.setEndValue(end)
        anim.setEasingCurve(QEasingCurve.InOutQuart)

        anim2 = QPropertyAnimation(self._sidebar, b"maximumWidth")
        anim2.setDuration(Settings.SIDEBAR_ANIMATION_MS)
        anim2.setStartValue(start)
        anim2.setEndValue(end)
        anim2.setEasingCurve(QEasingCurve.InOutQuart)

        anim.finished.connect(self._on_sidebar_anim_done)
        anim.start()
        anim2.start()
        # Keep refs alive until they finish.
        self._sidebar_animation = anim
        self._sidebar_animation_2 = anim2

    def _on_sidebar_anim_done(self) -> None:
        if self._sidebar_expanded:
            # Remove upper bound so the splitter handle can grow the sidebar past 240 px.
            self._sidebar.setMaximumWidth(16777215)
        self._reset_active_viewer_zoom()

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
        self._title_label.setText(Settings.PAGES[idx][2])
        self._nav_buttons[page_key].setChecked(True)

        # Notify incoming page.
        new_page = self.pages.get(page_key)
        if hasattr(new_page, "on_activated"):
            new_page.on_activated()

    def _on_experiment_changed(self) -> None:
        exp = self.exp_manager.active
        if exp is None:
            return
        self._status_indicator.set_status(exp.status)
        # Push state into every page that wants it.
        for page in self.pages.values():
            if hasattr(page, "load_from_experiment"):
                page.load_from_experiment(exp)

    def _on_status_changed(self, status: str) -> None:
        self._status_indicator.set_status(status)

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

    # ── V1.39 Phase 7 — Performance settings dialog ────────────────
    def _open_performance_settings(self) -> None:
        """Show the Performance settings modal.

        Two checkboxes (GPU analysis, build pyramids) plus a Rebuild
        Pyramid button. State lives on :class:`Settings` for the
        lifetime of the process; persisting across launches is a
        V1.40 follow-up.
        """
        # Imports kept local so the main window does not pay for them
        # at startup if the dialog never opens.
        from PySide6.QtWidgets import (
            QCheckBox, QDialog, QDialogButtonBox, QLabel, QVBoxLayout,
        )

        from nd2studios.compute.gpu import configure as gpu_configure
        from nd2studios.compute.gpu import gpu_status

        dlg = QDialog(self)
        dlg.setWindowTitle("Performance")
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(16, 16, 16, 12)

        status = gpu_status()
        if status["available"]:
            mem = status.get("memory_gb", 0.0)
            gpu_label = (
                f"Use GPU acceleration ({status['device_name']}, {mem:.1f} GB)"
            )
            if not status.get("cucim", False):
                gpu_label += "  — cucim missing; install for full speedup"
        else:
            gpu_label = f"Use GPU acceleration — unavailable: {status['reason']}"

        gpu_cb = QCheckBox(gpu_label)
        gpu_cb.setChecked(bool(getattr(Settings, "USE_GPU_ANALYSIS", False))
                          and status["available"])
        gpu_cb.setEnabled(bool(status["available"]))
        layout.addWidget(gpu_cb)

        pyr_cb = QCheckBox("Build multi-resolution pyramids on import")
        pyr_cb.setChecked(bool(getattr(Settings, "BUILD_PYRAMIDS", True)))
        pyr_cb.setEnabled(HAS_ZARR)
        if not HAS_ZARR:
            pyr_cb.setToolTip(
                "Pyramids require the optional `zarr` package. "
                "Install with `pip install zarr`."
            )
        layout.addWidget(pyr_cb)

        # Rebuild button + helper label
        from PySide6.QtWidgets import QPushButton
        rebuild_btn = QPushButton("Rebuild pyramid for current file")
        rebuild_btn.setEnabled(self.session is not None and HAS_ZARR)
        layout.addWidget(rebuild_btn)
        layout.addWidget(QLabel(
            f"Workspace: {self.session.session_dir if self.session else '— (no file imported)'}",
        ))

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
        )
        layout.addWidget(buttons)

        def _do_rebuild() -> None:
            exp = self.exp_manager.active
            volume = getattr(exp, "_raw_volume", None) if exp is not None else None
            if volume is None:
                self.set_status_text("No active volume to build a pyramid from.")
                return
            stage = self.pyramid_stage()
            if stage is None:
                return
            # Wipe the prior record so the build is forced.
            self.session.remove_stage(stage.name, delete_artifacts=True)  # type: ignore[union-attr]
            self.start_pyramid_build(volume)
            dlg.accept()

        rebuild_btn.clicked.connect(_do_rebuild)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            Settings.USE_GPU_ANALYSIS = bool(gpu_cb.isChecked())
            Settings.BUILD_PYRAMIDS = bool(pyr_cb.isChecked())
            effective = gpu_configure(Settings.USE_GPU_ANALYSIS)
            if Settings.USE_GPU_ANALYSIS and not effective:
                self.set_status_text(
                    "GPU acceleration requested but unavailable — staying on CPU."
                )
            else:
                self.set_status_text(
                    "Performance settings updated "
                    f"(GPU analysis: {'on' if effective else 'off'}, "
                    f"pyramids: {'on' if Settings.BUILD_PYRAMIDS else 'off'})."
                )

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
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Configuration", "",
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
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "",
            f"ND2Studios Config (*{CONFIG_EXTENSION})",
        )
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
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "",
            f"ND2Studios Session (*{SESSION_EXTENSION})",
        )
        if path:
            self.exp_manager.save_session(path)
            self.set_status_text(f"Saved: {os.path.basename(path)}")

    def _load_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "",
            f"ND2Studios Session (*{SESSION_EXTENSION})",
        )
        if path:
            self.exp_manager.load_session(path)
            self.set_status_text(f"Loaded: {os.path.basename(path)}")
