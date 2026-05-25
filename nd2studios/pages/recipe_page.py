"""
Recipe page — build a linear pipeline of enhancement steps with the
trial / accept / reject pattern adapted from CellTracker.

Workflow:
1. User picks a plugin from the combo box.
2. The `ParamEditor` shows that plugin's parameters.
3. "Trial" runs the *full current pipeline + the trial step* on the raw
   channels. The result lands on the experiment as
   `_processed_channels` and is shown in the preview.
4. "Accept" appends the trial step to the recipe and locks the trial
   result as the new committed state.
5. "Reject" discards the trial; the committed state is unchanged.
6. "Remove Last" pops the last step from the recipe and reprocesses.
7. "Save Recipe" / "Load Recipe" persist the recipe (without dataset)
   as `.nd2s_recipe.json` so it can be applied to other ND2 files.

Multi-file layout (V1.1):
- When 1 confirmed file: Raw | Processed side-by-side (horizontal split).
- When >1 confirmed files: each file gets a vertical column (Raw over
  Processed); columns sit side-by-side in the outer horizontal splitter.
- Trial / Accept / Reject operate on ALL confirmed files in parallel.
  Crop only affects the primary (index 0) file.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from nd2studios.core.experiment_manager import ND2StudiosRecord
from nd2studios.core.plugin_registry import PluginBase
from nd2studios.core.settings import Settings
from nd2studios.widgets.common import ParamEditor
from nd2studios.widgets.image_viewer import CHANNEL_COLORS
from nd2studios.widgets.multi_axis_viewer import MultiAxisViewer
from nd2studios.workers.recipe_worker import RecipeWorker
from nd2studios.backend.recipes import (
    save_recipe as save_recipe_json,
    load_recipe as load_recipe_json,
    RECIPE_EXTENSION,
)


class _RecipeColumn:
    """Per-confirmed-file viewer pair on the Recipe page."""

    def __init__(
        self,
        record: ND2StudiosRecord,
        label: str,
        vertical: bool,
    ) -> None:
        self.record = record
        self.worker: Optional[RecipeWorker] = None

        raw_w = QWidget()
        raw_l = QVBoxLayout(raw_w)
        raw_l.setContentsMargins(0, 0, 0, 0)
        raw_hdr = f"{label} — Raw" if vertical else "Raw"
        raw_l.addWidget(QLabel(raw_hdr, objectName="sectionHeader"))
        self.viewer_raw = MultiAxisViewer(raw_w)
        raw_l.addWidget(self.viewer_raw, stretch=1)

        proc_w = QWidget()
        proc_l = QVBoxLayout(proc_w)
        proc_l.setContentsMargins(0, 0, 0, 0)
        proc_hdr = "Processed" if vertical else "Processed (current state + trial)"
        proc_l.addWidget(QLabel(proc_hdr, objectName="sectionHeader"))
        self.viewer_proc = MultiAxisViewer(proc_w)
        proc_l.addWidget(self.viewer_proc, stretch=1)

        orientation = (Qt.Orientation.Vertical if vertical
                       else Qt.Orientation.Horizontal)
        self.widget = QSplitter(orientation)
        self.widget.addWidget(raw_w)
        self.widget.addWidget(proc_w)
        self.widget.setStretchFactor(0, 1)
        self.widget.setStretchFactor(1, 1)


class RecipePage(QWidget):
    """Page 2: build a processing recipe."""

    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        # Committed recipe (list of (plugin_name, params) pairs).
        self._recipe: List[Tuple[str, Dict[str, Any]]] = []
        self._normalized: bool = False
        # Trial state.
        self._trial_step: Optional[Tuple[str, Dict[str, Any]]] = None
        self._worker: Optional[RecipeWorker] = None
        self._extra_workers: List[RecipeWorker] = []
        self._pending_secondary_recipe: Optional[List[Tuple[str, Dict[str, Any]]]] = None
        # Per-file columns (populated by _rebuild_columns).
        self._columns: List[_RecipeColumn] = []
        # Crop state — in-session only, not persisted.
        self._crop_rect: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
        # Tracks the M position currently shown in the raw viewer so that
        # _raw_channels stays in sync when the user moves the M slider.
        self._current_m: int = 0

        self._build_ui()
        self._populate_plugin_list()

    # ── UI ──
    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        # Left column: plugin picker + params + buttons.
        left = QWidget()
        left.setFixedWidth(420)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(8)

        # Plugin picker
        pick_group = QGroupBox("Add a step")
        pl = QVBoxLayout(pick_group)
        self.combo_plugin = QComboBox()
        self.combo_plugin.currentTextChanged.connect(self._on_plugin_changed)
        pl.addWidget(self.combo_plugin)
        self.lbl_plugin_desc = QLabel("")
        self.lbl_plugin_desc.setWordWrap(True)
        self.lbl_plugin_desc.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        pl.addWidget(self.lbl_plugin_desc)
        self.param_editor = ParamEditor()
        pl.addWidget(self.param_editor)
        ll.addWidget(pick_group)

        # Trial / accept / reject row
        trial_row = QHBoxLayout()
        self.btn_trial = QPushButton("Trial")
        self.btn_trial.setObjectName("primaryBtn")
        self.btn_trial.clicked.connect(self._on_trial)
        trial_row.addWidget(self.btn_trial)
        self.btn_accept = QPushButton("Accept")
        self.btn_accept.setObjectName("successBtn")
        self.btn_accept.setEnabled(False)
        self.btn_accept.clicked.connect(self._on_accept)
        trial_row.addWidget(self.btn_accept)
        self.btn_reject = QPushButton("Reject")
        self.btn_reject.setObjectName("dangerBtn")
        self.btn_reject.setEnabled(False)
        self.btn_reject.clicked.connect(self._on_reject)
        trial_row.addWidget(self.btn_reject)
        ll.addLayout(trial_row)

        # Crop group
        crop_group = QGroupBox("Crop")
        cl = QVBoxLayout(crop_group)
        self.btn_crop_mode = QPushButton("Crop Mode")
        self.btn_crop_mode.setCheckable(True)
        self.btn_crop_mode.setToolTip(
            "Enable crop tool on the Raw viewer.\n"
            "Click anywhere to enter the corner and size manually.\n"
            "Click-drag to draw the crop rectangle — values are pre-filled."
        )
        cl.addWidget(self.btn_crop_mode)
        self.lbl_crop_status = QLabel("No crop applied")
        self.lbl_crop_status.setStyleSheet(
            f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        cl.addWidget(self.lbl_crop_status)
        self.btn_reset_crop = QPushButton("Reset Crop")
        self.btn_reset_crop.setEnabled(False)
        cl.addWidget(self.btn_reset_crop)
        ll.addWidget(crop_group)

        # Recipe list
        rec_group = QGroupBox("Recipe")
        rl = QVBoxLayout(rec_group)
        self.cb_normalized = QCheckBox("Frame-mean normalization (applied first)")
        self.cb_normalized.stateChanged.connect(self._on_normalized_changed)
        rl.addWidget(self.cb_normalized)
        self.list_recipe = QListWidget()
        rl.addWidget(self.list_recipe)
        rec_buttons = QHBoxLayout()
        self.btn_remove = QPushButton("Remove Last")
        self.btn_remove.clicked.connect(self._on_remove_last)
        rec_buttons.addWidget(self.btn_remove)
        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.clicked.connect(self._on_clear)
        rec_buttons.addWidget(self.btn_clear)
        rl.addLayout(rec_buttons)
        save_load = QHBoxLayout()
        self.btn_save_recipe = QPushButton("Save Recipe…")
        self.btn_save_recipe.clicked.connect(self._on_save_recipe)
        save_load.addWidget(self.btn_save_recipe)
        self.btn_load_recipe = QPushButton("Load Recipe…")
        self.btn_load_recipe.clicked.connect(self._on_load_recipe)
        save_load.addWidget(self.btn_load_recipe)
        rl.addLayout(save_load)
        ll.addWidget(rec_group)
        ll.addStretch(1)
        outer.addWidget(left)

        # Dynamic viewer area — rebuilt in _rebuild_columns().
        self._viewer_area = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(self._viewer_area, stretch=1)

        # Crop signal wiring is deferred to _rebuild_columns() because the
        # primary viewer_raw is not created until the columns are built.
        self.btn_crop_mode.toggled.connect(self._on_crop_mode_toggled)
        self.btn_reset_crop.clicked.connect(self._reset_crop)

    # ── Column helpers ───────────────────────────────────────────────

    @property
    def viewer_raw(self) -> Optional[MultiAxisViewer]:
        return self._columns[0].viewer_raw if self._columns else None

    @property
    def viewer_proc(self) -> Optional[MultiAxisViewer]:
        return self._columns[0].viewer_proc if self._columns else None

    def _rebuild_columns(self) -> bool:
        """Rebuild the viewer area to match currently confirmed records.

        Returns True if columns were actually rebuilt (callers may use this
        to decide whether to repopulate viewers), False on fast-path.
        """
        # Collect confirmed records; fall back to primary active record.
        records: List[ND2StudiosRecord] = []
        if self.main_window is not None:
            fn = getattr(self.main_window, "get_confirmed_records", None)
            if callable(fn):
                records = fn()
        if not records and self.main_window is not None:
            exp = self.main_window.exp_manager.active
            if exp is not None:
                records = [exp]

        # Fast path: if records match current columns, skip rebuild.
        current_recs = [c.record for c in self._columns]
        if records == current_recs and self._columns:
            return False

        # Disconnect crop and coord signals from old primary viewer.
        if self._columns:
            old_raw = self._columns[0].viewer_raw
            try:
                old_raw.canvas.clicked.disconnect(self._on_canvas_click)
                old_raw.crop_rect_selected.disconnect(self._on_crop_drag)
                old_raw.coords_changed.disconnect(self._on_raw_coords_changed)
            except RuntimeError:
                pass

        # Remove all column widgets from viewer_area.
        while self._viewer_area.count() > 0:
            w = self._viewer_area.widget(0)
            if w is not None:
                w.setParent(None)  # type: ignore[arg-type]

        self._columns.clear()

        vertical = len(records) > 1
        for i, rec in enumerate(records):
            import os as _os
            label = _os.path.basename(
                (getattr(rec, "import_config", {}) or {}).get("filepath", "")
                or (getattr(rec, "nd2_metadata", {}) or {}).get("filepath", "")
            ) or f"File {i + 1}"
            col = _RecipeColumn(rec, label, vertical)
            self._viewer_area.addWidget(col.widget)
            self._columns.append(col)

        if not self._columns:
            # No confirmed records: create a blank placeholder column.
            col = _RecipeColumn(
                ND2StudiosRecord(),
                "Raw",
                vertical=False,
            )
            self._viewer_area.addWidget(col.widget)
            self._columns.append(col)

        # Equal widths.
        n = max(self._viewer_area.count(), 1)
        total = max(self._viewer_area.width(), 400)
        self._viewer_area.setSizes([total // n] * n)

        # Reconnect crop and coord signals to the new primary viewer.
        primary_raw = self._columns[0].viewer_raw
        primary_raw.canvas.clicked.connect(self._on_canvas_click)
        primary_raw.crop_rect_selected.connect(self._on_crop_drag)
        primary_raw.coords_changed.connect(self._on_raw_coords_changed)
        return True

    def _populate_viewers_from_records(self) -> None:
        """Load channel data into each column's viewers from its record."""
        for col in self._columns:
            rec = col.record
            if rec is None:
                continue
            if rec._raw_channels or rec._raw_volume is not None:
                self._populate_column_raw(col, rec)
                if rec._processed_channels:
                    col.viewer_proc.set_channels(
                        rec._processed_channels,
                        channel_display=rec.channel_display,
                    )
                elif rec._processed_view is not None and self._recipe:
                    try:
                        processed = rec._processed_view.materialize_all()
                    except Exception:  # noqa: BLE001
                        pass
                    else:
                        rec._processed_channels = processed
                        col.viewer_proc.set_channels(
                            processed, channel_display=rec.channel_display,
                        )

    def _populate_column_raw(
        self, col: _RecipeColumn, rec: ND2StudiosRecord
    ) -> None:
        if rec._raw_volume is not None and rec.crop_rect is None:
            col.viewer_raw.set_volume(
                rec._raw_volume,
                channel_display=rec.channel_display,
                z_mode=rec.z_view_mode or "max",
                z_index=rec.z_view_index or 0,
                m=rec.m_index, t=0, z=rec.z_view_index or 0,
            )
        else:
            col.viewer_raw.set_channels(
                rec._raw_channels or {},
                channel_display=rec.channel_display,
            )

    def _on_raw_coords_changed(self, m: int, t: int, z: int) -> None:
        """Sync _raw_channels and m_index when the raw viewer's M slider moves.

        This keeps the recipe worker's input in step with what the user is
        looking at, so Trial / Reject always operate on the displayed M.
        """
        if self._current_m == m:
            return
        self._current_m = m
        if self.main_window is None:
            return
        exp = self.main_window.exp_manager.active
        if exp is None or exp._raw_volume is None:
            return
        exp.m_index = m
        channels = exp._raw_volume.all_channels_as_lazy(
            m=m,
            z_mode=exp.z_view_mode or "max",
            z_index=exp.z_view_index or 0,
        )
        if exp.channel_display:
            enabled = [
                n for n, cfg in exp.channel_display.items()
                if cfg.get("enabled", True)
            ]
            if enabled:
                channels = {n: channels[n] for n in enabled if n in channels}
        exp._raw_channels = channels

    # ── Crop ──
    def _on_crop_mode_toggled(self, enabled: bool) -> None:
        vr = self.viewer_raw
        if vr is not None:
            vr.set_crop_mode(enabled)
            if enabled:
                vr.zoom_toolbar.btn_pan.setChecked(False)

    def _on_canvas_click(self, iy: float, ix: float) -> None:
        if not self.btn_crop_mode.isChecked():
            return
        result = self._show_crop_dialog(x=int(ix), y=int(iy), w=0, h=0)
        if result is not None:
            self._apply_crop(*result)

    def _on_crop_drag(self, x: int, y: int, w: int, h: int) -> None:
        result = self._show_crop_dialog(x=x, y=y, w=w, h=h)
        if result is not None:
            self._apply_crop(*result)

    def _show_crop_dialog(
        self, x: int, y: int, w: int, h: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """Open a dialog for the user to confirm/edit the crop rectangle.

        Returns (x, y, w, h) or None if cancelled.
        """
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return None
        exp = self.main_window.exp_manager.active
        source = exp._original_raw_channels or exp._raw_channels
        if not source:
            return None

        # Determine current image dimensions from the *original* (pre-crop) source.
        sample = next(iter(source.values()))
        if hasattr(sample, "shape"):
            arr_shape = sample.shape
        else:
            arr_shape = np.asarray(sample).shape
        img_h = int(arr_shape[-2])
        img_w = int(arr_shape[-1])

        dlg = QDialog(self)
        dlg.setWindowTitle("Crop Image")
        layout = QVBoxLayout(dlg)

        info = QLabel(f"Image: {img_w} × {img_h} px  "
                      f"(X = columns from left, Y = rows from top)")
        info.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        layout.addWidget(info)

        form = QFormLayout()
        sp_x = QSpinBox()
        sp_x.setRange(0, img_w - 1)
        sp_x.setValue(max(0, min(x, img_w - 1)))
        sp_x.setToolTip("Left edge of crop (pixels from image left)")

        sp_y = QSpinBox()
        sp_y.setRange(0, img_h - 1)
        sp_y.setValue(max(0, min(y, img_h - 1)))
        sp_y.setToolTip("Top edge of crop (pixels from image top)")

        sp_w = QSpinBox()
        sp_w.setRange(1, img_w)
        sp_w.setValue(w if w > 0 else max(1, img_w - x))
        sp_w.setToolTip("Width of crop in pixels")

        sp_h = QSpinBox()
        sp_h.setRange(1, img_h)
        sp_h.setValue(h if h > 0 else max(1, img_h - y))
        sp_h.setToolTip("Height of crop in pixels")

        form.addRow("X (left corner):", sp_x)
        form.addRow("Y (top corner):", sp_y)
        form.addRow("Width (px):", sp_w)
        form.addRow("Height (px):", sp_h)
        layout.addLayout(form)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None

        cx, cy, cw, ch = sp_x.value(), sp_y.value(), sp_w.value(), sp_h.value()
        if cx + cw > img_w or cy + ch > img_h:
            QMessageBox.warning(
                self, "Invalid crop",
                f"Crop region ({cx}+{cw}={cx+cw}, {cy}+{ch}={cy+ch}) "
                f"exceeds image bounds ({img_w}, {img_h}).\n"
                "Clamping to image boundary."
            )
            cw = min(cw, img_w - cx)
            ch = min(ch, img_h - cy)
        return (cx, cy, cw, ch)

    def _apply_crop(self, x: int, y: int, w: int, h: int) -> None:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return
        exp = self.main_window.exp_manager.active
        if not exp._raw_channels:
            return

        # Always crop from original so the user can re-crop without nesting.
        if exp._original_raw_channels is None:
            exp._original_raw_channels = dict(exp._raw_channels)
        source = exp._original_raw_channels

        cropped: Dict[str, Any] = {}
        for ch_name, ch_data in source.items():
            crop_fn = getattr(ch_data, "crop", None)
            if callable(crop_fn):
                cropped[ch_name] = crop_fn(y, y + h, x, x + w)
            else:
                arr = np.asarray(ch_data)
                cropped[ch_name] = arr[:, y:y + h, x:x + w]

        self._crop_rect = (x, y, w, h)
        exp.crop_rect = (x, y, w, h)
        exp._raw_channels = cropped
        exp._processed_channels = None

        self.btn_crop_mode.setChecked(False)
        self.btn_reset_crop.setEnabled(True)
        self.lbl_crop_status.setText(f"Crop: x={x}, y={y}, {w}×{h} px")

        primary_col = self._columns[0] if self._columns else None
        if primary_col:
            primary_col.viewer_raw.set_channels(
                cropped, channel_display=exp.channel_display)
            if self._recipe:
                self._run_recipe_on(
                    exp, list(self._recipe),
                    label="Re-running recipe on cropped data…",
                    on_done=lambda processed, col=primary_col, e=exp: self._on_revert_done_col(processed, col, e),
                )
            else:
                primary_col.viewer_proc.set_channels(
                    cropped, channel_display=exp.channel_display)

    def _reset_crop(self) -> None:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return
        exp = self.main_window.exp_manager.active
        if exp._original_raw_channels is None:
            return

        exp._raw_channels = exp._original_raw_channels
        exp._original_raw_channels = None
        exp._processed_channels = None
        exp.crop_rect = None
        self._crop_rect = None
        self.btn_reset_crop.setEnabled(False)
        self.lbl_crop_status.setText("No crop applied")

        primary_col = self._columns[0] if self._columns else None
        if primary_col:
            self._populate_column_raw(primary_col, exp)
            if self._recipe:
                self._run_recipe_on(
                    exp, list(self._recipe),
                    label="Re-running recipe after crop reset…",
                    on_done=lambda processed, col=primary_col, e=exp: self._on_revert_done_col(processed, col, e),
                )
            else:
                primary_col.viewer_proc.set_channels(
                    exp._raw_channels, channel_display=exp.channel_display)

    def _populate_plugin_list(self) -> None:
        # Force-import to ensure registration.
        import nd2studios.plugins.enhancement.builtin  # noqa: F401
        plugins = PluginBase.get_plugins("enhancement")
        names = sorted([p.name for p in plugins])
        self.combo_plugin.clear()
        self.combo_plugin.addItems(names)
        if names:
            self._on_plugin_changed(names[0])

    # ── Page lifecycle ──
    def on_activated(self) -> None:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return
        exp = self.main_window.exp_manager.active
        self._current_m = int(getattr(exp, "m_index", 0))

        # Rebuild columns to match currently confirmed records.
        # Only re-populate viewers when the column set actually changed
        # (avoids redundant set_channels calls on every tab switch).
        rebuilt = self._rebuild_columns()

        # Restore crop UI from the experiment record.
        self.btn_crop_mode.setChecked(False)
        if exp.crop_rect is not None:
            x, y, w, h = exp.crop_rect
            self._crop_rect = exp.crop_rect
            self.btn_reset_crop.setEnabled(True)
            self.lbl_crop_status.setText(f"Crop: x={x}, y={y}, {w}×{h} px")
        else:
            self._crop_rect = None
            self.btn_reset_crop.setEnabled(False)
            self.lbl_crop_status.setText("No crop applied")

        if rebuilt:
            self._populate_viewers_from_records()

    def _refresh_raw_viewer(self, exp: ND2StudiosRecord) -> None:
        """Refresh the primary column's raw viewer (backward compat)."""
        if self._columns:
            self._populate_column_raw(self._columns[0], exp)

    def load_from_experiment(self, exp: ND2StudiosRecord) -> None:
        self._recipe = list(exp.recipe)
        self._normalized = bool(exp.recipe_normalized)
        self.cb_normalized.blockSignals(True)
        self.cb_normalized.setChecked(self._normalized)
        self.cb_normalized.blockSignals(False)
        self._refresh_recipe_list()

    def save_to_experiment(self, exp: ND2StudiosRecord) -> None:
        exp.recipe = list(self._recipe)
        exp.recipe_normalized = bool(self._normalized)

    # ── Plugin combo ──
    def _on_plugin_changed(self, name: str) -> None:
        cls = PluginBase.get_plugin("enhancement", name)
        if cls is None:
            return
        plugin = cls()
        self.lbl_plugin_desc.setText(getattr(plugin, "description", ""))
        self.param_editor.set_params(plugin.get_params())

    def _on_normalized_changed(self, _state: int) -> None:
        self._normalized = self.cb_normalized.isChecked()

    # ── Trial / Accept / Reject ──
    def _on_trial(self) -> None:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return
        exp = self.main_window.exp_manager.active
        if not exp._raw_channels:
            QMessageBox.information(self, "Nothing to process",
                                    "Import a file first.")
            return

        plugin_name = self.combo_plugin.currentText()
        params = self.param_editor.get_values()
        full_recipe = list(self._recipe) + [(plugin_name, params)]
        self._trial_step = (plugin_name, params)
        self._pending_secondary_recipe = full_recipe

        # Cancel any running secondary workers before starting a new trial.
        self._cancel_extra_workers()

        # Primary column (wired to exp_manager.active).
        if self.main_window is not None:
            self.main_window.set_status_text("Trial running…")

        primary_col = self._columns[0] if self._columns else None
        if primary_col:
            self._run_recipe_on(
                exp, full_recipe,
                label="",
                on_done=self._on_trial_done_primary,
                col=primary_col,
            )

    def _on_trial_done_primary(self, processed: Dict[str, Any]) -> None:
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return
        exp = self.main_window.exp_manager.active
        primary_col = self._columns[0] if self._columns else None
        if primary_col:
            primary_col.viewer_proc.set_channels(
                processed, channel_display=exp.channel_display)
        exp._processed_channels = processed
        self.btn_accept.setEnabled(True)
        self.btn_reject.setEnabled(True)

        # Start secondary workers only after primary finishes to avoid I/O
        # contention on large files.
        recipe = getattr(self, "_pending_secondary_recipe", None)
        if recipe:
            for col in self._columns[1:]:
                if col.record._raw_channels:
                    self._run_recipe_on_secondary(col, recipe)

    def _on_accept(self) -> None:
        if self._trial_step is None:
            return
        self._recipe.append(self._trial_step)
        self._trial_step = None
        self.btn_accept.setEnabled(False)
        self.btn_reject.setEnabled(False)
        self._refresh_recipe_list()
        if self.main_window is not None:
            self.main_window.set_status_text(f"Step accepted ({len(self._recipe)} total).")
            self.main_window.exp_manager.set_status("preprocessed")
            self._commit_recipe_stage()
        # Mark secondary records preprocessed too.
        for col in self._columns[1:]:
            if col.record is not None:
                col.record.status = "preprocessed"

    def _commit_recipe_stage(self) -> None:
        """Persist the committed recipe to the workspace."""
        stage = self.main_window.recipe_stage() if self.main_window else None
        if stage is None:
            return
        exp = self.main_window.exp_manager.active
        if exp is None:
            return
        channel_names = list(
            (exp._raw_channels or exp._processed_channels or {}).keys()
        )
        stage.set_recipe(self._recipe, self._normalized, channel_names)
        try:
            stage.commit()
        except OSError as exc:
            self.main_window.set_status_text(
                f"Workspace write failed ({exc.__class__.__name__}); "
                "in-memory recipe unchanged."
            )

    def _on_reject(self) -> None:
        self._trial_step = None
        self.btn_accept.setEnabled(False)
        self.btn_reject.setEnabled(False)
        self._cancel_extra_workers()
        if self.main_window is not None and self.main_window.exp_manager.active is not None:
            exp = self.main_window.exp_manager.active
            self._revert_all_columns(exp)

    def _revert_all_columns(self, primary_exp: ND2StudiosRecord) -> None:
        """Revert all column viewers to the committed recipe state."""
        for i, col in enumerate(self._columns):
            rec = primary_exp if i == 0 else col.record
            if rec is None or not rec._raw_channels:
                continue
            if self._recipe:
                self._run_recipe_on(
                    rec, list(self._recipe),
                    label="Reverting…" if i == 0 else "",
                    on_done=lambda processed, c=col, r=rec: self._on_revert_done_col(processed, c, r),
                    col=col,
                )
            else:
                rec._processed_channels = None
                if rec._raw_channels:
                    col.viewer_proc.set_channels(
                        rec._raw_channels, channel_display=rec.channel_display)

    def _on_revert_done_col(
        self,
        processed: Dict[str, Any],
        col: _RecipeColumn,
        rec: ND2StudiosRecord,
    ) -> None:
        rec._processed_channels = processed
        col.viewer_proc.set_channels(processed, channel_display=rec.channel_display)

    def _on_revert_done(self, processed: Dict[str, Any]) -> None:
        """Backward-compat single-column revert handler."""
        if self.main_window is None or self.main_window.exp_manager.active is None:
            return
        exp = self.main_window.exp_manager.active
        if self._columns:
            self._on_revert_done_col(processed, self._columns[0], exp)

    # ── Remove / Clear ──
    def _on_remove_last(self) -> None:
        if not self._recipe:
            return
        self._recipe.pop()
        self._refresh_recipe_list()
        if self.main_window is not None and self.main_window.exp_manager.active is not None:
            exp = self.main_window.exp_manager.active
            self._revert_all_columns(exp)

    def _on_clear(self) -> None:
        self._recipe.clear()
        self._trial_step = None
        self.btn_accept.setEnabled(False)
        self.btn_reject.setEnabled(False)
        self._refresh_recipe_list()
        if self.main_window is not None and self.main_window.exp_manager.active is not None:
            exp = self.main_window.exp_manager.active
            self._cancel_extra_workers()
            for i, col in enumerate(self._columns):
                rec = exp if i == 0 else col.record
                if rec is None:
                    continue
                rec._processed_channels = None
                if rec._raw_channels:
                    col.viewer_proc.set_channels(
                        rec._raw_channels, channel_display=rec.channel_display)

    # ── Save / Load recipe ──
    def _on_save_recipe(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Recipe", "",
            f"ND2Studios Recipe (*{RECIPE_EXTENSION})",
        )
        if not path:
            return
        if not path.endswith(RECIPE_EXTENSION):
            path += RECIPE_EXTENSION
        try:
            save_recipe_json(path, self._recipe, self._normalized,
                             name=os.path.basename(path),
                             notes="Saved from ND2Studios")
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))
            return
        if self.main_window is not None:
            self.main_window.set_status_text(f"Saved recipe: {os.path.basename(path)}")

    def _on_load_recipe(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Recipe", "",
            f"ND2Studios Recipe (*{RECIPE_EXTENSION});;All files (*)",
        )
        if not path:
            return
        try:
            data = load_recipe_json(path)
        except Exception as e:
            QMessageBox.warning(self, "Load failed", str(e))
            return
        self._recipe = [(s["name"], dict(s.get("params", {}))) for s in data.get("pipeline", [])]
        self._normalized = bool(data.get("normalized", False))
        self.cb_normalized.blockSignals(True)
        self.cb_normalized.setChecked(self._normalized)
        self.cb_normalized.blockSignals(False)
        self._refresh_recipe_list()
        # Re-apply against current data on all columns.
        if self.main_window is not None and self.main_window.exp_manager.active is not None:
            exp = self.main_window.exp_manager.active
            if self._recipe:
                self._revert_all_columns(exp)

    # ── Helpers ──
    def _refresh_recipe_list(self) -> None:
        self.list_recipe.clear()
        for i, (name, params) in enumerate(self._recipe):
            line = f"{i + 1}. {name}"
            if params:
                line += f"   ({', '.join(f'{k}={v}' for k, v in params.items())})"
            self.list_recipe.addItem(QListWidgetItem(line))

    def _run_recipe_on(
        self,
        rec: ND2StudiosRecord,
        recipe: List[Tuple[str, Dict[str, Any]]],
        label: str,
        on_done,
        col: Optional[_RecipeColumn] = None,
    ) -> None:
        """Run recipe on a record's raw channels; result delivered to on_done."""
        if col is not None and col.worker is not None and col.worker.isRunning():
            col.worker.cancel()
            col.worker.wait(500)

        worker = RecipeWorker(
            channels=rec._raw_channels or {},
            recipe=recipe,
            normalized=self._normalized,
        )
        worker.progress.connect(self._on_progress)
        if label:
            worker.status.connect(self._on_status)
        worker.finished.connect(on_done)
        worker.error.connect(self._on_error)
        if label and self.main_window is not None:
            self.main_window.set_status_text(label)

        if col is not None:
            col.worker = worker
        else:
            # Primary worker (backward compat).
            if self._worker is not None and self._worker.isRunning():
                self._worker.cancel()
                self._worker.wait(500)
            self._worker = worker

        worker.start()

    def _run_recipe_on_secondary(
        self,
        col: _RecipeColumn,
        recipe: List[Tuple[str, Dict[str, Any]]],
    ) -> None:
        """Launch a worker for a secondary column; result shown in col.viewer_proc."""
        rec = col.record

        def on_done(processed: Dict[str, Any]) -> None:
            rec._processed_channels = processed
            col.viewer_proc.set_channels(
                processed, channel_display=rec.channel_display)

        if col.worker is not None and col.worker.isRunning():
            col.worker.cancel()
            col.worker.wait(500)

        worker = RecipeWorker(
            channels=rec._raw_channels or {},
            recipe=recipe,
            normalized=self._normalized,
        )
        worker.finished.connect(on_done)
        worker.error.connect(self._on_error)
        col.worker = worker
        self._extra_workers.append(worker)
        worker.start()

    def _cancel_extra_workers(self) -> None:
        for w in self._extra_workers:
            if w.isRunning():
                w.cancel()
        self._extra_workers.clear()

    # ── Legacy _run_recipe shim (used by crop path) ──
    def _run_recipe(
        self,
        exp: ND2StudiosRecord,
        recipe: List[Tuple[str, Dict[str, Any]]],
        label: str,
        on_done,
    ) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(500)
        self._worker = RecipeWorker(
            channels=exp._raw_channels or {},
            recipe=recipe,
            normalized=self._normalized,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.finished.connect(on_done)
        self._worker.error.connect(self._on_error)
        if self.main_window is not None:
            self.main_window.set_status_text(label)
        self._worker.start()

    def _on_progress(self, p: int) -> None:
        if self.main_window is not None:
            self.main_window.set_progress(p)

    def _on_status(self, msg: str) -> None:
        if self.main_window is not None:
            self.main_window.set_status_text(msg)

    def _on_error(self, msg: str) -> None:
        QMessageBox.warning(self, "Recipe failed", msg)
        if self.main_window is not None:
            self.main_window.set_progress(0)
