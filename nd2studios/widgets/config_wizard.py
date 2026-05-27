"""
Configuration wizard and save-preview dialog for ND2Studios.

ConfigWizard — interactive three-tab wizard (Processing, Analysis, Results).
    Opened by the sidebar "Configure" button and by Save when no config file
    has been loaded yet.  On Accept it saves the configuration to a .nd2s_cfg
    file chosen by the user (required before OK is enabled).

SavePreviewDialog — read-only diff of what changed since the config was last
    loaded/saved.  Opened by Save when a baseline config already exists.

Config file format: JSON with extension CONFIG_EXTENSION (.nd2s_cfg).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from nd2studios.core.settings import Settings

CONFIG_EXTENSION = ".nd2s_cfg"

# ── Shared dialog stylesheet ──────────────────────────────────────────────────
# Applied to both dialogs so all child widgets inherit a consistent dark theme
# regardless of the system default (prevents invisible-white-on-white text).
_DLG_STYLE = f"""
    QWidget {{
        background: {Settings.BG_PRIMARY};
        color: {Settings.FG_PRIMARY};
        font: 9pt;
    }}
    QLabel {{
        background: transparent;
        color: {Settings.FG_PRIMARY};
    }}
    QCheckBox {{
        background: transparent;
        color: {Settings.FG_PRIMARY};
    }}
    QCheckBox::indicator {{
        width: 14px; height: 14px;
        border: 1px solid {Settings.BORDER_COLOR};
        border-radius: 2px;
        background: {Settings.BG_SECONDARY};
    }}
    QCheckBox::indicator:checked {{
        background: {Settings.ACCENT_PURPLE};
        border-color: {Settings.ACCENT_PURPLE};
    }}
    QComboBox {{
        background: {Settings.BG_SECONDARY};
        color: {Settings.FG_PRIMARY};
        border: 1px solid {Settings.BORDER_COLOR};
        padding: 3px 8px;
        border-radius: 3px;
    }}
    QComboBox QAbstractItemView {{
        background: {Settings.BG_SECONDARY};
        color: {Settings.FG_PRIMARY};
        selection-background-color: {Settings.BG_TERTIARY};
    }}
    QPushButton {{
        background: {Settings.BG_TERTIARY};
        color: {Settings.FG_PRIMARY};
        border: 1px solid {Settings.BORDER_COLOR};
        border-radius: 3px;
        padding: 4px 14px;
    }}
    QPushButton:hover {{ background: {Settings.BG_HOVER}; }}
    QPushButton:pressed {{ background: {Settings.BORDER_COLOR}; }}
    QPushButton:disabled {{
        color: {Settings.FG_SECONDARY};
        background: {Settings.BG_SECONDARY};
    }}
    QListWidget {{
        background: {Settings.BG_SECONDARY};
        color: {Settings.FG_PRIMARY};
        border: 1px solid {Settings.BORDER_COLOR};
        font: 9pt;
    }}
    QListWidget::item {{
        color: {Settings.FG_PRIMARY};
        padding: 4px 8px;
    }}
    QListWidget::item:selected {{
        background: {Settings.BG_TERTIARY};
        color: {Settings.ACCENT_PURPLE};
    }}
    QScrollArea {{
        background: transparent;
        border: none;
    }}
    QScrollBar:vertical {{
        background: {Settings.BG_SECONDARY};
        width: 8px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical {{
        background: {Settings.BORDER_COLOR};
        border-radius: 4px;
        min-height: 20px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_config_from_pages(
    analysis_page,
    results_page,
    recipe_page=None,
) -> dict:
    """Snapshot current page UI state as a serialisable config dict."""
    config: Dict[str, Any] = {
        "version": "1.0",
        "processing": {},
        "analysis": {},
        "results": {},
    }
    if recipe_page is not None:
        config["processing"] = {
            "normalized": bool(recipe_page._normalized),
            "recipe": [
                {"name": name, "params": dict(params)}
                for name, params in recipe_page._recipe
            ],
        }
    if analysis_page is not None:
        config["analysis"] = {
            "pipeline": analysis_page.combo_pipeline.currentText(),
            "params": analysis_page.param_editor.get_values(),
        }
    if results_page is not None:
        config["results"] = {
            "selected_columns": [
                k for k, cb in results_page._col_checkboxes.items()
                if cb.isChecked()
            ],
        }
    return config


def apply_config_to_pages(
    config: dict,
    analysis_page,
    results_page,
    recipe_page=None,
) -> None:
    """Push a config dict into the page widgets."""
    if recipe_page is not None:
        processing = config.get("processing", {})
        normalized = processing.get("normalized")
        recipe = processing.get("recipe")
        if normalized is not None:
            recipe_page._normalized = bool(normalized)
            recipe_page.cb_normalized.blockSignals(True)
            recipe_page.cb_normalized.setChecked(bool(normalized))
            recipe_page.cb_normalized.blockSignals(False)
        if recipe is not None:
            recipe_page._recipe = [
                (s["name"], dict(s.get("params", {}))) for s in recipe
            ]
            recipe_page._refresh_recipe_list()

    if analysis_page is not None:
        analysis = config.get("analysis", {})
        pipeline_name = analysis.get("pipeline", "")
        params = analysis.get("params") or {}
        if pipeline_name:
            # Update the combo first so the page reflects the right pipeline
            # immediately, even before the user visits the Analysis tab.
            idx = analysis_page.combo_pipeline.findText(pipeline_name)
            if idx >= 0:
                analysis_page.combo_pipeline.blockSignals(True)
                analysis_page.combo_pipeline.setCurrentIndex(idx)
                analysis_page.combo_pipeline.blockSignals(False)
            analysis_page._on_pipeline_changed(
                pipeline_name, restore_values=params or None
            )

    if results_page is not None:
        sel_cols = config.get("results", {}).get("selected_columns")
        if sel_cols is not None:
            sel_set = set(sel_cols)
            for key, cb in results_page._col_checkboxes.items():
                cb.blockSignals(True)
                cb.setChecked(key in sel_set)
                cb.blockSignals(False)
            if results_page._measurements:
                results_page._update_table(results_page._measurements)


def build_config_diff(old: dict, new: dict) -> Dict[str, List[str]]:
    """Return per-section lists of human-readable change descriptions.

    Returns a dict with keys "processing", "analysis", and "results", each
    a list of strings.  An empty list means no changes in that section.
    """
    diff: Dict[str, List[str]] = {
        "processing": [],
        "analysis": [],
        "results": [],
    }

    # ── Processing ──
    old_p = old.get("processing", {})
    new_p = new.get("processing", {})

    old_norm = old_p.get("normalized", False)
    new_norm = new_p.get("normalized", False)
    if old_norm != new_norm:
        diff["processing"].append(
            f"Normalization:  {old_norm}  →  {new_norm}"
        )

    old_recipe = old_p.get("recipe") or []
    new_recipe = new_p.get("recipe") or []
    if len(old_recipe) != len(new_recipe):
        diff["processing"].append(
            f"Recipe steps:  {len(old_recipe)}  →  {len(new_recipe)}"
        )
    else:
        for i, (os_, ns_) in enumerate(zip(old_recipe, new_recipe)):
            if os_.get("name") != ns_.get("name") or os_.get("params") != ns_.get("params"):
                diff["processing"].append(
                    f"  Step {i + 1}:  {os_.get('name')}  →  {ns_.get('name')}"
                )

    # ── Analysis ──
    old_a = old.get("analysis", {})
    new_a = new.get("analysis", {})

    old_pipe = old_a.get("pipeline", "—")
    new_pipe = new_a.get("pipeline", "—")
    if old_pipe != new_pipe:
        diff["analysis"].append(f"Pipeline:  {old_pipe}  →  {new_pipe}")

    old_params = old_a.get("params") or {}
    new_params = new_a.get("params") or {}
    for key in sorted(set(old_params) | set(new_params)):
        ov, nv = old_params.get(key), new_params.get(key)
        if ov != nv:
            diff["analysis"].append(f"  {key}:  {ov}  →  {nv}")

    # ── Results ──
    old_r = old.get("results", {})
    new_r = new.get("results", {})

    old_cols = set(old_r.get("selected_columns") or [])
    new_cols = set(new_r.get("selected_columns") or [])
    for c in sorted(new_cols - old_cols):
        diff["results"].append(f"  + {c}")
    for c in sorted(old_cols - new_cols):
        diff["results"].append(f"  − {c}")

    return diff


# ── Shared nav-sidebar builder ────────────────────────────────────────────────

def _make_nav_panel(title: str, items: List[str]) -> tuple:
    """Return (nav_panel_widget, QListWidget) ready to connect to a QStackedWidget."""
    nav_panel = QWidget()
    nav_panel.setFixedWidth(180)
    nav_panel.setStyleSheet(f"background: {Settings.BG_SECONDARY};")
    layout = QVBoxLayout(nav_panel)
    layout.setContentsMargins(0, 20, 0, 8)
    layout.setSpacing(0)

    title_lbl = QLabel(title)
    title_lbl.setStyleSheet(
        f"color: {Settings.FG_SECONDARY}; font: bold 9pt; "
        f"padding: 0 12px 12px 12px; background: transparent;"
    )
    layout.addWidget(title_lbl)

    nav = QListWidget()
    nav.setStyleSheet(f"""
        QListWidget {{
            background: {Settings.BG_SECONDARY};
            border: none;
            font: 10pt;
            color: {Settings.FG_PRIMARY};
            outline: none;
        }}
        QListWidget::item {{
            padding: 10px 16px;
            border-left: 3px solid transparent;
            color: {Settings.FG_PRIMARY};
        }}
        QListWidget::item:selected {{
            background: {Settings.BG_TERTIARY};
            color: {Settings.ACCENT_PURPLE};
            border-left: 3px solid {Settings.ACCENT_PURPLE};
        }}
    """)
    for item in items:
        nav.addItem(item)
    layout.addWidget(nav)
    layout.addStretch(1)
    return nav_panel, nav


# ── Config wizard ─────────────────────────────────────────────────────────────

class ConfigWizard(QDialog):
    """Interactive three-tab configuration wizard: Processing, Analysis, Results.

    On Accept the wizard saves the configuration to a .nd2s_cfg file the user
    must name (via Browse… or a file dialog on first OK click).  The caller
    reads ``save_path`` and ``get_config()`` after a successful exec().
    """

    def __init__(
        self,
        recipe_page,
        analysis_page,
        results_page,
        parent=None,
        initial_path: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configure Pipelines")
        self.setMinimumSize(800, 600)
        self.setStyleSheet(_DLG_STYLE)

        self._recipe_page = recipe_page
        self._analysis_page = analysis_page
        self._results_page = results_page
        self._cfg_path = initial_path

        # Snapshot recipe-page state for Cancel.
        self._original_recipe: list = []
        self._original_normalized: bool = False
        if recipe_page is not None:
            self._original_recipe = list(recipe_page._recipe)
            self._original_normalized = bool(recipe_page._normalized)

        # Working copy of recipe for the wizard panel.
        self._wizard_recipe: list = list(self._original_recipe)
        self._wizard_normalized: bool = self._original_normalized

        # Snapshot results-page checkbox states for Cancel.
        self._original_col_states: Dict[str, bool] = {}
        if results_page is not None:
            self._original_col_states = {
                k: cb.isChecked()
                for k, cb in results_page._col_checkboxes.items()
            }

        # Separate wizard column-checkboxes (not the live sidebar ones).
        self._wizard_col_checkboxes: Dict[str, QCheckBox] = {}

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        nav_panel, self._nav = _make_nav_panel(
            "Configure", ["Processing", "Analysis", "Results"]
        )
        self._nav.currentRowChanged.connect(
            lambda row: self._stack.setCurrentIndex(row)
        )
        outer.addWidget(nav_panel)

        right = QWidget()
        right.setStyleSheet(f"background: {Settings.BG_PRIMARY};")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(20, 20, 20, 12)
        right_layout.setSpacing(12)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_processing_panel())
        self._stack.addWidget(self._build_analysis_panel())
        self._stack.addWidget(self._build_results_panel())
        right_layout.addWidget(self._stack, stretch=1)

        # Save-path row — must be set before clicking OK.
        path_row = QHBoxLayout()
        path_lbl = QLabel("Config file:")
        path_lbl.setStyleSheet(f"color: {Settings.FG_SECONDARY};")
        path_row.addWidget(path_lbl)
        self._lbl_path = QLabel(os.path.basename(initial_path) or "— not set —")
        self._lbl_path.setStyleSheet(f"color: {Settings.FG_SECONDARY};")
        path_row.addWidget(self._lbl_path, stretch=1)
        btn_browse = QPushButton("Browse…")
        btn_browse.setFixedWidth(80)
        btn_browse.clicked.connect(self._browse_path)
        path_row.addWidget(btn_browse)
        right_layout.addLayout(path_row)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        right_layout.addWidget(btns)

        outer.addWidget(right, stretch=1)
        self._nav.setCurrentRow(0)

    # ── Processing panel ──────────────────────────────────────────────────────

    def _build_processing_panel(self) -> QWidget:
        from nd2studios.backend.recipes import RECIPE_EXTENSION

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        hdr = QLabel("Processing Recipe")
        hdr.setStyleSheet(f"font: bold 12pt; color: {Settings.FG_PRIMARY};")
        layout.addWidget(hdr)

        desc = QLabel(
            "Configure the image processing recipe applied to raw data before analysis."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        layout.addWidget(desc)

        self._cb_normalized = QCheckBox("Frame-mean normalization (applied first)")
        self._cb_normalized.setChecked(self._wizard_normalized)
        layout.addWidget(self._cb_normalized)

        steps_lbl = QLabel("Recipe steps:")
        steps_lbl.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: bold 9pt;")
        layout.addWidget(steps_lbl)

        self._recipe_list_widget = QListWidget()
        self._recipe_list_widget.setFixedHeight(160)
        self._recipe_list_widget.setStyleSheet(f"""
            QListWidget {{
                background: {Settings.BG_SECONDARY};
                color: {Settings.FG_PRIMARY};
                border: 1px solid {Settings.BORDER_COLOR};
                font: 9pt;
            }}
            QListWidget::item {{
                color: {Settings.FG_PRIMARY};
                padding: 3px 8px;
            }}
        """)
        self._refresh_wizard_recipe_list()
        layout.addWidget(self._recipe_list_widget)

        btn_row = QHBoxLayout()
        btn_load = QPushButton("Load Recipe…")
        btn_load.clicked.connect(self._on_load_recipe)
        btn_row.addWidget(btn_load)
        btn_clear = QPushButton("Clear Recipe")
        btn_clear.clicked.connect(self._on_clear_recipe)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self._recipe_ext = RECIPE_EXTENSION
        layout.addStretch(1)
        return page

    def _refresh_wizard_recipe_list(self) -> None:
        self._recipe_list_widget.clear()
        if not self._wizard_recipe:
            self._recipe_list_widget.addItem("(no steps)")
            return
        for i, (name, params) in enumerate(self._wizard_recipe):
            line = f"{i + 1}.  {name}"
            if params:
                line += f"   ({', '.join(f'{k}={v}' for k, v in params.items())})"
            self._recipe_list_widget.addItem(line)

    def _on_load_recipe(self) -> None:
        from nd2studios.backend.recipes import (
            RECIPE_EXTENSION,
            load_recipe as _load,
        )
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Recipe", "",
            f"ND2Studios Recipe (*{RECIPE_EXTENSION});;All files (*)",
        )
        if not path:
            return
        try:
            data = _load(path)
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Load failed", str(exc))
            return
        self._wizard_recipe = [
            (s["name"], dict(s.get("params", {})))
            for s in data.get("pipeline", [])
        ]
        self._wizard_normalized = bool(data.get("normalized", False))
        self._cb_normalized.blockSignals(True)
        self._cb_normalized.setChecked(self._wizard_normalized)
        self._cb_normalized.blockSignals(False)
        self._refresh_wizard_recipe_list()

    def _on_clear_recipe(self) -> None:
        self._wizard_recipe = []
        self._refresh_wizard_recipe_list()

    # ── Analysis panel ────────────────────────────────────────────────────────

    def _build_analysis_panel(self) -> QWidget:
        from nd2studios.core.analysis_registry import AnalysisPipeline
        from nd2studios.widgets.common import ParamEditor

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        hdr = QLabel("Analysis Pipeline")
        hdr.setStyleSheet(f"font: bold 12pt; color: {Settings.FG_PRIMARY};")
        layout.addWidget(hdr)

        desc = QLabel(
            "Select the analysis pipeline and configure its parameters."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        layout.addWidget(desc)

        row = QHBoxLayout()
        row.addWidget(QLabel("Pipeline:"))
        self._combo_pipeline = QComboBox()
        self._combo_pipeline.setMinimumWidth(280)
        for cls in AnalysisPipeline.get_pipelines():
            self._combo_pipeline.addItem(cls.name)
        if self._analysis_page is not None:
            cur = self._analysis_page.combo_pipeline.currentText()
            idx = self._combo_pipeline.findText(cur)
            if idx >= 0:
                self._combo_pipeline.setCurrentIndex(idx)
        self._combo_pipeline.currentTextChanged.connect(self._on_pipeline_changed)
        row.addWidget(self._combo_pipeline)
        row.addStretch(1)
        layout.addLayout(row)

        self._param_editor = ParamEditor()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        il = QVBoxLayout(inner)
        il.setContentsMargins(0, 0, 0, 0)
        il.addWidget(self._param_editor)
        il.addStretch(1)
        scroll.setWidget(inner)
        layout.addWidget(scroll, stretch=1)

        self._reload_params()
        return page

    def _reload_params(self) -> None:
        from nd2studios.core.analysis_registry import AnalysisPipeline
        name = self._combo_pipeline.currentText()
        cls = AnalysisPipeline.get_pipeline(name)
        if cls is None:
            return
        specs = cls().get_params()
        ch_names = (
            getattr(self._analysis_page, "_current_channel_names", []) or []
        )
        for spec in specs:
            if spec.name == "channel_name" and ch_names:
                spec.choices = ch_names
                if spec.default not in ch_names:
                    spec.default = ch_names[0]
            elif spec.name == "counterstain_channel" and ch_names:
                spec.choices = ["None"] + ch_names
        self._param_editor.set_params(specs)
        if (self._analysis_page is not None
                and self._analysis_page.combo_pipeline.currentText() == name):
            self._param_editor.set_values(
                self._analysis_page.param_editor.get_values()
            )

    def _on_pipeline_changed(self, _name: str) -> None:
        self._reload_params()

    # ── Results panel ─────────────────────────────────────────────────────────

    def _build_results_panel(self) -> QWidget:
        from nd2studios.pages.results_page import _COLUMN_GROUPS, _DEFAULT_COLUMNS

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        hdr = QLabel("Result Columns")
        hdr.setStyleSheet(f"font: bold 12pt; color: {Settings.FG_PRIMARY};")
        layout.addWidget(hdr)

        desc = QLabel(
            "Choose which measurement columns to include in the results table."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        layout.addWidget(desc)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        il = QVBoxLayout(inner)
        il.setContentsMargins(0, 2, 8, 4)
        il.setSpacing(0)

        existing = (
            getattr(self._results_page, "_col_checkboxes", {})
            if self._results_page else {}
        )
        intensity_items = [
            (k, cb.text())
            for k, cb in existing.items()
            if k.startswith("mean_intensity_") or k.startswith("std_intensity_")
        ]
        all_groups = list(_COLUMN_GROUPS)
        if intensity_items:
            all_groups.append(("Intensity", intensity_items))

        for group_name, cols in all_groups:
            lbl = QLabel(group_name.upper())
            lbl.setFixedHeight(24)
            lbl.setStyleSheet(
                f"color: {Settings.FG_SECONDARY}; font: bold 8pt; "
                f"letter-spacing: 1px; padding-left: 4px; background: transparent;"
            )
            il.addWidget(lbl)

            for key, label in cols:
                src = existing.get(key)
                cb = QCheckBox(label)
                cb.setFixedHeight(24)
                cb.setChecked(
                    src.isChecked() if src is not None else key in _DEFAULT_COLUMNS
                )
                cb.toggled.connect(
                    lambda checked, k=key: self._on_col_toggled(k, checked)
                )
                cb.setStyleSheet("padding-left: 4px; background: transparent;")
                self._wizard_col_checkboxes[key] = cb
                il.addWidget(cb)

            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setFixedHeight(1)
            sep.setStyleSheet(
                f"background: {Settings.BORDER_COLOR}; margin: 4px 0;"
            )
            il.addWidget(sep)

        il.addStretch(1)
        scroll.setWidget(inner)
        layout.addWidget(scroll, stretch=1)

        btn_row = QHBoxLayout()
        btn_all = QPushButton("Select all")
        btn_all.setFixedWidth(90)
        btn_all.clicked.connect(
            lambda: [cb.setChecked(True) for cb in self._wizard_col_checkboxes.values()]
        )
        btn_none = QPushButton("Clear all")
        btn_none.setFixedWidth(90)
        btn_none.clicked.connect(
            lambda: [cb.setChecked(False) for cb in self._wizard_col_checkboxes.values()]
        )
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        return page

    def _on_col_toggled(self, key: str, checked: bool) -> None:
        if self._results_page is not None:
            cb = self._results_page._col_checkboxes.get(key)
            if cb is not None:
                cb.blockSignals(True)
                cb.setChecked(checked)
                cb.blockSignals(False)

    # ── Path picker ───────────────────────────────────────────────────────────

    def _browse_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration As",
            self._cfg_path or "",
            f"ND2Studios Config (*{CONFIG_EXTENSION})",
        )
        if path:
            if not path.endswith(CONFIG_EXTENSION):
                path += CONFIG_EXTENSION
            self._cfg_path = path
            self._lbl_path.setText(os.path.basename(path))

    # ── Accept / reject ───────────────────────────────────────────────────────

    def _on_accept(self) -> None:
        # Require a file path — prompt if none has been browsed to yet.
        if not self._cfg_path:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Configuration As", "",
                f"ND2Studios Config (*{CONFIG_EXTENSION})",
            )
            if not path:
                return
            if not path.endswith(CONFIG_EXTENSION):
                path += CONFIG_EXTENSION
            self._cfg_path = path
            self._lbl_path.setText(os.path.basename(path))

        # Apply processing to recipe page.
        if self._recipe_page is not None:
            self._recipe_page._recipe = list(self._wizard_recipe)
            self._recipe_page._normalized = self._cb_normalized.isChecked()
            self._recipe_page.cb_normalized.blockSignals(True)
            self._recipe_page.cb_normalized.setChecked(
                self._recipe_page._normalized
            )
            self._recipe_page.cb_normalized.blockSignals(False)
            self._recipe_page._refresh_recipe_list()

        # Apply analysis pipeline.
        if self._analysis_page is not None:
            name = self._combo_pipeline.currentText()
            params = self._param_editor.get_values()
            self._analysis_page._on_pipeline_changed(
                name, restore_values=params or None
            )

        # Apply results columns.
        if self._results_page is not None:
            for key, cb in self._wizard_col_checkboxes.items():
                src = self._results_page._col_checkboxes.get(key)
                if src is not None:
                    src.blockSignals(True)
                    src.setChecked(cb.isChecked())
                    src.blockSignals(False)
            if self._results_page._measurements:
                self._results_page._update_table(
                    self._results_page._measurements
                )

        # Save the config to the chosen file.
        cfg = self.get_config()
        with open(self._cfg_path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)

        self.accept()

    def reject(self) -> None:
        # Restore recipe-page state.
        if self._recipe_page is not None:
            self._recipe_page._recipe = list(self._original_recipe)
            self._recipe_page._normalized = self._original_normalized
            self._recipe_page.cb_normalized.blockSignals(True)
            self._recipe_page.cb_normalized.setChecked(self._original_normalized)
            self._recipe_page.cb_normalized.blockSignals(False)
            self._recipe_page._refresh_recipe_list()

        # Restore results-page checkboxes.
        if self._results_page is not None:
            for key, was in self._original_col_states.items():
                cb = self._results_page._col_checkboxes.get(key)
                if cb is not None:
                    cb.blockSignals(True)
                    cb.setChecked(was)
                    cb.blockSignals(False)
        super().reject()

    # ── Config I/O ────────────────────────────────────────────────────────────

    @property
    def save_path(self) -> str:
        return self._cfg_path or ""

    def get_config(self) -> dict:
        """Return the wizard's current settings as a serialisable dict."""
        return {
            "version": "1.0",
            "processing": {
                "normalized": self._cb_normalized.isChecked(),
                "recipe": [
                    {"name": name, "params": dict(params)}
                    for name, params in self._wizard_recipe
                ],
            },
            "analysis": {
                "pipeline": self._combo_pipeline.currentText(),
                "params": self._param_editor.get_values(),
            },
            "results": {
                "selected_columns": [
                    k for k, cb in self._wizard_col_checkboxes.items()
                    if cb.isChecked()
                ],
            },
        }

    def apply_config(self, config: dict) -> None:
        """Pre-populate the wizard fields from a saved config dict."""
        # Processing
        processing = config.get("processing", {})
        normalized = processing.get("normalized")
        if normalized is not None:
            self._wizard_normalized = bool(normalized)
            self._cb_normalized.blockSignals(True)
            self._cb_normalized.setChecked(bool(normalized))
            self._cb_normalized.blockSignals(False)
        recipe = processing.get("recipe")
        if recipe is not None:
            self._wizard_recipe = [
                (s["name"], dict(s.get("params", {}))) for s in recipe
            ]
            self._refresh_wizard_recipe_list()

        # Analysis
        analysis = config.get("analysis", {})
        pipeline = analysis.get("pipeline", "")
        if pipeline:
            idx = self._combo_pipeline.findText(pipeline)
            if idx >= 0:
                self._combo_pipeline.blockSignals(True)
                self._combo_pipeline.setCurrentIndex(idx)
                self._combo_pipeline.blockSignals(False)
            self._reload_params()
        params = analysis.get("params")
        if params:
            self._param_editor.set_values(params)

        # Results
        sel_cols = config.get("results", {}).get("selected_columns")
        if sel_cols is not None:
            sel_set = set(sel_cols)
            for key, cb in self._wizard_col_checkboxes.items():
                cb.blockSignals(True)
                cb.setChecked(key in sel_set)
                cb.blockSignals(False)


# ── Save preview dialog ───────────────────────────────────────────────────────

class SavePreviewDialog(QDialog):
    """Shows a diff of what changed since the config was last loaded/saved.

    The left nav mirrors the ConfigWizard so the visual language is consistent.
    Each panel shows the current values with added/changed/removed lines
    clearly marked.  The user confirms or cancels; the save path can be
    changed via a Browse button.
    """

    def __init__(
        self,
        old_config: dict,
        new_config: dict,
        default_path: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save Configuration")
        self.setMinimumSize(800, 560)
        self.setStyleSheet(_DLG_STYLE)

        self._save_path = default_path
        diff = build_config_diff(old_config, new_config)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        nav_panel, self._nav = _make_nav_panel(
            "Review Changes", ["Processing", "Analysis", "Results"]
        )
        self._nav.currentRowChanged.connect(
            lambda row: self._stack.setCurrentIndex(row)
        )
        outer.addWidget(nav_panel)

        right = QWidget()
        right.setStyleSheet(f"background: {Settings.BG_PRIMARY};")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(20, 20, 20, 12)
        right_layout.setSpacing(12)

        self._stack = QStackedWidget()
        self._stack.addWidget(
            self._build_diff_panel(
                "Processing Recipe",
                old_config.get("processing", {}),
                new_config.get("processing", {}),
                diff["processing"],
            )
        )
        self._stack.addWidget(
            self._build_diff_panel(
                "Analysis Pipeline",
                old_config.get("analysis", {}),
                new_config.get("analysis", {}),
                diff["analysis"],
            )
        )
        self._stack.addWidget(
            self._build_diff_panel(
                "Result Columns",
                old_config.get("results", {}),
                new_config.get("results", {}),
                diff["results"],
            )
        )
        right_layout.addWidget(self._stack, stretch=1)

        # Save-path row
        path_row = QHBoxLayout()
        path_lbl = QLabel("Save to:")
        path_lbl.setStyleSheet(f"color: {Settings.FG_SECONDARY};")
        path_row.addWidget(path_lbl)
        self._lbl_path = QLabel(os.path.basename(default_path) or "—")
        self._lbl_path.setStyleSheet(f"color: {Settings.FG_SECONDARY};")
        path_row.addWidget(self._lbl_path, stretch=1)
        btn_browse = QPushButton("Browse…")
        btn_browse.setFixedWidth(80)
        btn_browse.clicked.connect(self._browse_path)
        path_row.addWidget(btn_browse)
        right_layout.addLayout(path_row)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        right_layout.addWidget(btns)

        outer.addWidget(right, stretch=1)
        self._nav.setCurrentRow(0)

    def _build_diff_panel(
        self,
        title: str,
        old_section: dict,
        new_section: dict,
        changes: List[str],
    ) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        hdr = QLabel(title)
        hdr.setStyleSheet(f"font: bold 12pt; color: {Settings.FG_PRIMARY};")
        layout.addWidget(hdr)

        if not changes:
            no_change = QLabel("No changes in this section.")
            no_change.setStyleSheet(
                f"color: {Settings.FG_SECONDARY}; font: 10pt;"
            )
            layout.addWidget(no_change)
        else:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            inner = QWidget()
            il = QVBoxLayout(inner)
            il.setContentsMargins(0, 0, 0, 0)
            il.setSpacing(4)

            for line in changes:
                lbl = QLabel(line)
                lbl.setWordWrap(True)
                if line.strip().startswith("+"):
                    lbl.setStyleSheet(
                        f"color: {Settings.ACCENT_GREEN}; font: 9pt; "
                        f"padding: 2px 0; font-family: monospace; "
                        f"background: transparent;"
                    )
                elif line.strip().startswith("−"):
                    lbl.setStyleSheet(
                        f"color: {Settings.ACCENT_RED}; font: 9pt; "
                        f"padding: 2px 0; font-family: monospace; "
                        f"background: transparent;"
                    )
                elif "→" in line:
                    lbl.setStyleSheet(
                        f"color: {Settings.ACCENT_ORANGE}; font: 9pt; "
                        f"padding: 2px 0; font-family: monospace; "
                        f"background: transparent;"
                    )
                else:
                    lbl.setStyleSheet(
                        f"color: {Settings.FG_SECONDARY}; font: 9pt; "
                        f"padding: 2px 0; font-family: monospace; "
                        f"background: transparent;"
                    )
                il.addWidget(lbl)

            il.addStretch(1)
            scroll.setWidget(inner)
            layout.addWidget(scroll, stretch=1)

        layout.addStretch(1)
        return page

    def _browse_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration As",
            self._save_path,
            f"ND2Studios Config (*{CONFIG_EXTENSION})",
        )
        if path:
            if not path.endswith(CONFIG_EXTENSION):
                path += CONFIG_EXTENSION
            self._save_path = path
            self._lbl_path.setText(os.path.basename(path))

    @property
    def save_path(self) -> str:
        return self._save_path
