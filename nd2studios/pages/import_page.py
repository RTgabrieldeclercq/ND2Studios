"""
Import page (V1.1 → multi-file).

Hosts one or more :class:`~nd2studios.widgets.file_panel.FilePanel` widgets
laid out side-by-side in a horizontal QSplitter.  An "+ Add File" toolbar
button appends a new panel.  Closing a panel removes it (minimum 1 panel
always remains).

The **primary panel** (index 0) is directly wired to
``main_window.exp_manager.active`` — its ``record`` attribute *is* the active
experiment record, so every write (channels, volume, timestamps …) lands in
the shared experiment state without an extra sync step.

Secondary panels own standalone :class:`ND2StudiosRecord` objects and are
independent viewers; Confirm on a secondary panel updates only that panel's
local record.

The outer splitter's ``splitterMoved`` signal triggers
:meth:`FilePanel.fit_viewer` on every panel so images zoom-to-fit
automatically when the user drags a divider.  The same callback fires at the
end of a sidebar collapse/expand animation (handled inside FilePanel itself).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QSplitter, QVBoxLayout, QWidget,
)

from nd2studios.core.experiment_manager import ND2StudiosRecord
from nd2studios.widgets.file_panel import FilePanel


class ImportPage(QWidget):
    """Page 1: multi-file browse + metadata + side-by-side previews."""

    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self._panels: List[FilePanel] = []
        self._build_ui()
        # Start with one (primary) panel.
        self._add_panel()

    # ── UI construction ──────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 4)
        root.setSpacing(4)

        # Toolbar row
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        lbl = QLabel("Import", objectName="sectionHeader")
        toolbar.addWidget(lbl)
        toolbar.addStretch(1)
        self._btn_add = QPushButton("+ Add File")
        self._btn_add.setObjectName("primaryBtn")
        self._btn_add.setToolTip(
            "Open another ND2 / TIFF file in a new side-by-side panel"
        )
        self._btn_add.clicked.connect(self._add_panel)
        toolbar.addWidget(self._btn_add)

        self._btn_play_all = QPushButton("> Play All")
        self._btn_play_all.setObjectName("secondaryBtn")
        self._btn_play_all.setCheckable(True)
        self._btn_play_all.setToolTip("Play / pause T-axis on all open panels")
        self._btn_play_all.toggled.connect(self._on_play_all_toggled)
        toolbar.addWidget(self._btn_play_all)

        fps_lbl = QLabel("FPS:")
        toolbar.addWidget(fps_lbl)
        self._spin_fps = QDoubleSpinBox()
        self._spin_fps.setRange(0.1, 60.0)
        self._spin_fps.setValue(10.0)
        self._spin_fps.setSingleStep(1.0)
        self._spin_fps.setDecimals(1)
        self._spin_fps.setFixedWidth(64)
        self._spin_fps.setToolTip("Playback speed for Play All")
        toolbar.addWidget(self._spin_fps)

        root.addLayout(toolbar)

        # Outer splitter — one child per FilePanel
        self._outer_splitter = QSplitter(Qt.Horizontal)
        self._outer_splitter.setChildrenCollapsible(False)
        self._outer_splitter.splitterMoved.connect(self._on_outer_splitter_moved)
        root.addWidget(self._outer_splitter, stretch=1)

    # ── Panel management ─────────────────────────────────────────────

    def _add_panel(self) -> None:
        is_primary = len(self._panels) == 0
        panel = FilePanel(
            parent=self,
            # All panels report progress/status to the shared status bar.
            on_progress=self._on_progress,
            on_status=self._on_status,
            on_confirm=(
                self._on_confirm_primary if is_primary
                else self._on_confirm_secondary
            ),
            on_stitch=(
                self._on_stitch_primary if is_primary else None
            ),
            show_close_button=True,
        )
        if is_primary and self.main_window is not None:
            exp = self.main_window.exp_manager.active
            if exp is not None:
                panel.record = exp

        panel.panel_close_requested.connect(self._remove_panel)
        self._outer_splitter.addWidget(panel)
        self._panels.append(panel)

        # Equalise widths after adding a panel.
        n = len(self._panels)
        total = max(self._outer_splitter.width(), 800)
        self._outer_splitter.setSizes([total // n] * n)

    def _remove_panel(self, panel: FilePanel) -> None:
        """Remove *panel* — the primary panel (index 0) cannot be removed."""
        if len(self._panels) <= 1:
            return
        idx = self._panels.index(panel)
        self._panels.pop(idx)
        panel.setParent(None)  # type: ignore[arg-type]
        panel.deleteLater()

    # ── Splitter callback ────────────────────────────────────────────

    def _on_outer_splitter_moved(self, _pos: int, _idx: int) -> None:
        for panel in self._panels:
            panel.fit_viewer()

    # ── Play All ─────────────────────────────────────────────────────

    def _on_play_all_toggled(self, playing: bool) -> None:
        fps = self._spin_fps.value()
        self._btn_play_all.setText("|| Stop All" if playing else "> Play All")
        for panel in self._panels:
            viewer = panel.viewer
            if viewer is not None and hasattr(viewer, "set_t_playing"):
                viewer.set_t_playing(playing, fps=fps)

    # ── Confirmed records ────────────────────────────────────────────

    def get_confirmed_records(self) -> List[ND2StudiosRecord]:
        """Return records for all panels whose status is 'imported' or later."""
        confirmed_statuses = {"imported", "preprocessed", "analyzed", "exported"}
        records = []
        for panel in self._panels:
            rec = panel.record
            if rec is not None and getattr(rec, "status", None) in confirmed_statuses:
                records.append(rec)
        return records

    # ── Viewer property for backward compatibility ───────────────────

    @property
    def viewer(self):
        """Return the primary panel's viewer (used by MainWindow for zoom reset)."""
        return self._panels[0].viewer if self._panels else None

    def reset_all_viewers(self) -> None:
        """Fit all panels' canvases to their current viewport."""
        for panel in self._panels:
            panel.fit_viewer()

    # ── Progress / status (primary panel only) ───────────────────────

    def _on_progress(self, p: int) -> None:
        if self.main_window is not None:
            self.main_window.set_progress(p)

    def _on_status(self, msg: str) -> None:
        if self.main_window is not None:
            self.main_window.set_status_text(msg)

    # ── Confirm callbacks ────────────────────────────────────────────

    def _on_confirm_primary(self, panel: FilePanel) -> None:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return
        exp = self.main_window.exp_manager.active
        if not exp._raw_channels and exp._raw_volume is None:
            QMessageBox.information(
                self, "Nothing to import",
                "Browse for an ND2 / TIFF file first.",
            )
            return

        try:
            self.save_to_experiment(exp)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Import error",
                                f"Could not read viewer state: {exc}")
            return

        if not exp._raw_channels:
            QMessageBox.information(
                self, "Nothing to import",
                "Browse for an ND2 / TIFF file first.",
            )
            return

        self.main_window.exp_manager.set_status("imported")
        self.main_window.set_status_text("Imported.")

        try:
            self._attach_workspace_and_maybe_resume(panel, exp)
        except Exception as exc:  # noqa: BLE001
            self.main_window.set_status_text(
                f"Imported (workspace unavailable: {type(exc).__name__})"
            )

    def _on_confirm_secondary(self, panel: FilePanel) -> None:
        """Confirm a secondary panel — marks it ready and shows feedback."""
        rec = panel.record
        if rec._raw_channels is None and rec._raw_volume is None:
            QMessageBox.information(
                self, "Nothing to import",
                "Browse for an ND2 / TIFF file first.",
            )
            return
        rec.status = "imported"
        rec.import_config = {
            "filepath": panel.filepath or "",
            "z_projection": panel.combo_zproj.currentText(),
            "t_stride": int(panel.spin_t_stride.value()),
        }
        name = os.path.basename(panel.filepath or "") or "panel"
        if self.main_window is not None:
            self.main_window.set_status_text(
                f"'{name}' confirmed (secondary panel — available for side-by-side review)."
            )

    # ── Stitch callback (primary panel only) ────────────────────────

    def _on_stitch_primary(self, panel: FilePanel) -> None:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return
        exp = self.main_window.exp_manager.active
        if exp._raw_volume is None:
            QMessageBox.information(
                self, "No volume",
                "Stitching needs a multi-position ND2 file. Browse first.",
            )
            return
        from nd2studios.pages.stitch_dialog import StitchDialog
        stage_xy = exp.nd2_metadata.get("stage_xy_um") or []
        dialog = StitchDialog(
            volume=exp._raw_volume,
            stage_xy_um=list(stage_xy),
            channel_display=exp.channel_display,
            main_window=self.main_window,
            parent=self,
        )
        dialog.exec()

    # ── Page lifecycle ───────────────────────────────────────────────

    def on_activated(self) -> None:
        pass

    def load_from_experiment(self, exp: ND2StudiosRecord) -> None:
        """Restore primary panel's UI from a loaded session record."""
        if not self._panels:
            return
        panel = self._panels[0]
        # Always point the primary panel at the current active record.
        panel.record = exp

        if not exp.import_config:
            return
        cfg = exp.import_config
        if "z_projection" in cfg:
            panel.combo_zproj.setCurrentText(cfg["z_projection"])
        if "t_stride" in cfg:
            panel.spin_t_stride.setValue(int(cfg["t_stride"]))
        if exp.nd2_metadata:
            panel._meta_dict = exp.nd2_metadata
            panel._populate_metadata_table(exp.nd2_metadata)
            panel._populate_info_rows(exp.nd2_metadata)
        if cfg.get("filepath"):
            panel._filepath = cfg["filepath"]
            panel.lbl_filepath.setText(os.path.basename(panel._filepath))
            panel._lbl_name.setText(os.path.basename(panel._filepath))

    def save_to_experiment(self, exp: ND2StudiosRecord) -> None:
        """Flush primary panel's UI state into *exp*."""
        if not self._panels:
            return
        panel = self._panels[0]
        exp.import_config = {
            "filepath": panel._filepath,
            "z_projection": panel.combo_zproj.currentText(),
            "t_stride": int(panel.spin_t_stride.value()),
        }
        viewer_state = panel.viewer.channel_state()
        if viewer_state:
            exp.channel_display = viewer_state
        m, t, z = panel.viewer.coords()
        exp.m_index = int(m)
        exp.z_view_index = int(z)
        exp.z_view_mode = panel.combo_zproj.currentText()
        if exp._raw_volume is not None:
            channels = exp._raw_volume.all_channels_as_lazy(
                m=exp.m_index,
                z_mode=exp.z_view_mode,
                z_index=exp.z_view_index,
            )
            enabled_names = [
                name for name, cfg in (exp.channel_display or {}).items()
                if cfg.get("enabled", True)
            ]
            if enabled_names:
                channels = {
                    n: channels[n] for n in enabled_names if n in channels
                }
            exp._raw_channels = channels

    # ── Workspace / workspace resume (primary panel) ─────────────────

    def _attach_workspace_and_maybe_resume(
        self, panel: FilePanel, exp: ND2StudiosRecord,
    ) -> None:
        if panel.filepath is None or self.main_window is None:
            return
        session = self.main_window.attach_session_for(panel.filepath)
        if session is None:
            return

        self._kick_off_pyramid(exp)

        prior_recipe = session.is_committed("recipe")
        prior_analyses = [
            name for name in session.manifest.stages
            if name.startswith("analysis:")
            and session.is_committed(name)
        ]
        if not (prior_recipe or prior_analyses):
            return

        bits: List[str] = []
        if prior_recipe:
            bits.append("an accepted recipe")
        if prior_analyses:
            bits.append(
                f"{len(prior_analyses)} analysis pipeline"
                + ("s" if len(prior_analyses) != 1 else "")
            )
        joined = " and ".join(bits)
        reply = QMessageBox.question(
            self, "Resume prior workspace?",
            f"A prior workspace exists for this file with {joined}.\n\n"
            "Resume to populate the Recipe / Results pages from saved "
            "artifacts, or Start fresh to archive the prior data.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.No:
            archived = session.archive()
            self.main_window.set_status_text(
                f"Started fresh; prior workspace archived → {archived.name}"
            )
            return

        if prior_recipe:
            self._rehydrate_recipe(exp, session)
        if prior_analyses:
            self._rehydrate_analyses(exp, session, prior_analyses)
        self.main_window.set_status_text("Imported — workspace resumed.")

    def _rehydrate_recipe(self, exp: ND2StudiosRecord, session) -> None:
        from nd2studios.pipeline import RecipeStage
        stage = RecipeStage(session)
        recipe, normalized, _channels = stage.get_recipe()
        if recipe:
            exp.recipe = recipe
            exp.recipe_normalized = normalized
            self.main_window.exp_manager.set_status("preprocessed")

    def _rehydrate_analyses(
        self, exp: ND2StudiosRecord, session, stage_names: List[str],
    ) -> None:
        from nd2studios.pipeline import AnalysisStage
        for stage_name in stage_names:
            pipeline_name = stage_name[len("analysis:"):]
            stage = AnalysisStage(session, pipeline_name)
            per_m: Dict[int, Any] = exp.analysis_results.setdefault(
                pipeline_name, {}
            )
            if not isinstance(per_m, dict):
                per_m = {}
                exp.analysis_results[pipeline_name] = per_m
            for m in stage.committed_m_indices():
                result = stage.rehydrate_m(m)
                if result is not None:
                    per_m[m] = result

    def _kick_off_pyramid(self, exp: ND2StudiosRecord) -> None:
        if self.main_window is None:
            return
        from nd2studios.core.settings import Settings
        if not bool(getattr(Settings, "BUILD_PYRAMIDS", True)):
            return
        volume = getattr(exp, "_raw_volume", None)
        if volume is None:
            return
        try:
            self.main_window.start_pyramid_build(volume)
        except Exception:  # noqa: BLE001 — pyramid must never block import
            pass
