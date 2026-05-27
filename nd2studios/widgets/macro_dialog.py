"""
Macro dialog (V1.0).

Non-modal QDialog with three panels:

  0  Manage  — browse / create / delete saved macros
  1  Record  — live action feed during recording; pause / cancel / finish
  2  Edit    — drag-reorder / enable / edit / delete action blocks, then replay

Signal contract (set up externally by MainWindow):
  main_window.macro_action_recorded(dict)  →  MacroDialog._on_action_recorded
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from PySide6.QtCore import QMimeData, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QScrollArea,
    QSizePolicy, QStackedWidget, QTextEdit, QVBoxLayout, QWidget,
)

from nd2studios.backend.macro_engine import (
    MACRO_EXTENSION, MacroAction, load_macro, save_macro,
)
from nd2studios.core.settings import Settings

if TYPE_CHECKING:
    pass


# ── Action-row widget (used in Edit panel) ────────────────────────────────────

class _ActionRowWidget(QFrame):
    """One block in the Edit panel: [⠿] [☐] [label ...] [Edit] [✕]"""

    edit_requested = Signal()
    delete_requested = Signal()
    enabled_toggled = Signal(bool)

    def __init__(self, action: MacroAction, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("actionRow")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedHeight(36)
        self._action = action

        h = QHBoxLayout(self)
        h.setContentsMargins(4, 2, 4, 2)
        h.setSpacing(6)

        drag_lbl = QLabel("⠿")
        drag_lbl.setFixedWidth(18)
        drag_lbl.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font-size: 14px;")
        drag_lbl.setToolTip("Drag to reorder")
        h.addWidget(drag_lbl)

        self._chk = QCheckBox()
        self._chk.setChecked(action.enabled)
        self._chk.setToolTip("Enable / disable this step during replay")
        self._chk.toggled.connect(self.enabled_toggled)
        h.addWidget(self._chk)

        self._lbl = QLabel(action.label)
        self._lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._lbl.setStyleSheet(
            f"color: {Settings.FG_PRIMARY};"
            if action.enabled
            else f"color: {Settings.FG_SECONDARY}; text-decoration: line-through;"
        )
        h.addWidget(self._lbl, stretch=1)

        btn_edit = QPushButton("Edit")
        btn_edit.setFixedWidth(52)
        btn_edit.setObjectName("sessionBtn")
        btn_edit.clicked.connect(self.edit_requested)
        h.addWidget(btn_edit)

        btn_del = QPushButton("✕")
        btn_del.setFixedWidth(28)
        btn_del.setObjectName("dangerBtn")
        btn_del.clicked.connect(self.delete_requested)
        h.addWidget(btn_del)

        self._chk.toggled.connect(self._on_enabled_toggled)

    @property
    def action(self) -> MacroAction:
        return self._action

    def refresh_label(self) -> None:
        self._lbl.setText(self._action.label)
        enabled = self._chk.isChecked()
        self._lbl.setStyleSheet(
            f"color: {Settings.FG_PRIMARY};"
            if enabled
            else f"color: {Settings.FG_SECONDARY}; text-decoration: line-through;"
        )

    def _on_enabled_toggled(self, checked: bool) -> None:
        self._action.enabled = checked
        self.refresh_label()


# ── Edit dialog (per-action param editor) ─────────────────────────────────────

class _ActionEditDialog(QDialog):
    """Simple key-value editor for a MacroAction's params dict."""

    def __init__(self, action: MacroAction, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Edit Action")
        self.setMinimumWidth(420)
        self._action = action
        self._editors: Dict[str, QLineEdit] = {}

        layout = QVBoxLayout(self)

        # Action type (read-only info)
        info = QLabel(f"<b>{action.label}</b>")
        info.setWordWrap(True)
        layout.addWidget(info)

        type_lbl = QLabel(f"Type: <code>{action.action_type}</code>")
        type_lbl.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        layout.addWidget(type_lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)

        # Params
        if action.params:
            form = QFormLayout()
            for key, value in action.params.items():
                edit = QLineEdit(json.dumps(value, ensure_ascii=False))
                edit.setToolTip(f"JSON value for '{key}'")
                self._editors[key] = edit
                form.addRow(f"{key}:", edit)
            layout.addLayout(form)
        else:
            layout.addWidget(QLabel("(no editable parameters)"))

        # Name / label
        lbl_label = QLabel("Label (shown in block):")
        layout.addWidget(lbl_label)
        self._edit_label = QLineEdit(action.label)
        layout.addWidget(self._edit_label)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _on_accept(self) -> None:
        new_params: Dict[str, Any] = {}
        for key, edit in self._editors.items():
            raw = edit.text().strip()
            try:
                new_params[key] = json.loads(raw)
            except json.JSONDecodeError:
                new_params[key] = raw  # keep as string if not valid JSON
        self._action.params = new_params
        self._action.label = self._edit_label.text().strip() or self._action.label
        self.accept()


# ── Draggable action list (Edit panel) ────────────────────────────────────────

class _ActionListWidget(QWidget):
    """
    Scrollable, drag-to-reorder list of _ActionRowWidget blocks.
    Reordering is implemented with QListWidget (internal move) where
    each item carries its index in UserRole; widgets are rebuilt after drops.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._list = QListWidget()
        self._list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setSpacing(2)
        self._list.model().rowsMoved.connect(self._on_rows_moved)
        layout.addWidget(self._list)

        self._actions: List[MacroAction] = []
        self._row_widgets: List[_ActionRowWidget] = []

    # ── public API ───────────────────────────────────────────────────

    def set_actions(self, actions: List[MacroAction]) -> None:
        self._actions = [MacroAction(
            action_type=a.action_type,
            label=a.label,
            params=dict(a.params),
            enabled=a.enabled,
            timestamp=a.timestamp,
        ) for a in actions]
        self._rebuild_list()

    def get_actions(self) -> List[MacroAction]:
        return list(self._actions)

    # ── internals ────────────────────────────────────────────────────

    def _rebuild_list(self) -> None:
        self._list.clear()
        self._row_widgets = []
        for i, action in enumerate(self._actions):
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, i)
            item.setSizeHint(QSize(0, 38))
            # Item text is used for the drag visual
            item.setText(action.label)
            item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsDragEnabled
            )
            self._list.addItem(item)
            row = _ActionRowWidget(action)
            row.edit_requested.connect(lambda checked=False, idx=i: self._on_edit(idx))
            row.delete_requested.connect(lambda checked=False, idx=i: self._on_delete(idx))
            self._list.setItemWidget(item, row)
            self._row_widgets.append(row)

    def _on_rows_moved(self, *_: Any) -> None:
        """Sync _actions order after an internal-move drag-drop."""
        new_order: List[MacroAction] = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            old_idx = item.data(Qt.ItemDataRole.UserRole)
            if 0 <= old_idx < len(self._actions):
                new_order.append(self._actions[old_idx])
        self._actions = new_order
        self._rebuild_list()

    def _on_edit(self, idx: int) -> None:
        if not (0 <= idx < len(self._actions)):
            return
        action = self._actions[idx]
        dlg = _ActionEditDialog(action, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            # action is mutated in-place by _ActionEditDialog._on_accept()
            self._rebuild_list()

    def _on_delete(self, idx: int) -> None:
        if not (0 <= idx < len(self._actions)):
            return
        del self._actions[idx]
        self._rebuild_list()


# ── Main dialog ───────────────────────────────────────────────────────────────

class MacroDialog(QDialog):
    """
    Non-modal macro management dialog.

    Panels:
      0 — Manage  (browse, create, open)
      1 — Record  (live action feed + controls)
      2 — Edit    (reorder / edit / run)
    """

    def __init__(self, main_window, parent: Optional[QWidget] = None):
        super().__init__(parent or main_window)
        self._mw = main_window
        self._macro_name: str = ""
        self._macro_path: str = ""
        self._recorded_actions: List[MacroAction] = []

        self.setWindowTitle("Macro Recorder")
        self.setMinimumSize(480, 560)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowStaysOnTopHint
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        self._stack = QStackedWidget()
        root.addWidget(self._stack, stretch=1)

        self._stack.addWidget(self._build_manage_panel())   # 0
        self._stack.addWidget(self._build_record_panel())   # 1
        self._stack.addWidget(self._build_edit_panel())     # 2

        self._stack.setCurrentIndex(0)

        # Connect to main_window signal for live action feed
        if hasattr(main_window, "macro_action_recorded"):
            main_window.macro_action_recorded.connect(self._on_action_recorded)

        # REC indicator blink timer
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(600)
        self._blink_timer.timeout.connect(self._blink_rec)
        self._blink_state: bool = True

    # ── Panel builders ────────────────────────────────────────────────────────

    def _build_manage_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        v.addWidget(QLabel(
            "<b>Macros</b> record your actions (recipe steps, analysis runs, "
            "exports) so you can replay them on any file."
        ))

        # New macro group
        new_group = QGroupBox("New Macro")
        ng = QVBoxLayout(new_group)
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self._edit_new_name = QLineEdit()
        self._edit_new_name.setPlaceholderText("My Macro")
        self._edit_new_name.returnPressed.connect(self._on_new_macro)
        name_row.addWidget(self._edit_new_name, stretch=1)
        ng.addLayout(name_row)

        btn_new = QPushButton("  ● Start Recording")
        btn_new.setObjectName("successBtn")
        btn_new.clicked.connect(self._on_new_macro)
        ng.addWidget(btn_new)
        v.addWidget(new_group)

        # Open existing
        open_group = QGroupBox("Open Existing Macro")
        og = QHBoxLayout(open_group)
        btn_open = QPushButton("Browse…")
        btn_open.clicked.connect(self._on_open_macro)
        og.addWidget(btn_open)
        og.addStretch(1)
        v.addWidget(open_group)

        v.addStretch(1)
        return w

    def _build_record_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        # Header with macro name + REC indicator
        header = QHBoxLayout()
        self._lbl_rec_name = QLabel()
        self._lbl_rec_name.setStyleSheet(f"font-weight: bold; color: {Settings.FG_PRIMARY};")
        header.addWidget(self._lbl_rec_name, stretch=1)
        self._lbl_rec_indicator = QLabel("● REC")
        self._lbl_rec_indicator.setStyleSheet("color: #ff5555; font-weight: bold;")
        header.addWidget(self._lbl_rec_indicator)
        v.addLayout(header)

        self._lbl_rec_status = QLabel("Interact with the application — actions will appear here.")
        self._lbl_rec_status.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        self._lbl_rec_status.setWordWrap(True)
        v.addWidget(self._lbl_rec_status)

        # Live action list (read-only display)
        self._rec_list = QListWidget()
        self._rec_list.setDragEnabled(False)
        self._rec_list.setEnabled(False)
        v.addWidget(self._rec_list, stretch=1)

        # Count label
        self._lbl_rec_count = QLabel("0 actions recorded")
        self._lbl_rec_count.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        v.addWidget(self._lbl_rec_count)

        # Control buttons
        btn_row = QHBoxLayout()
        self._btn_pause = QPushButton("⏸  Pause")
        self._btn_pause.setObjectName("sessionBtn")
        self._btn_pause.clicked.connect(self._on_toggle_pause)
        btn_row.addWidget(self._btn_pause)

        btn_cancel = QPushButton("✕  Cancel")
        btn_cancel.setObjectName("dangerBtn")
        btn_cancel.clicked.connect(self._on_cancel_recording)
        btn_row.addWidget(btn_cancel)

        btn_finish = QPushButton("✔  Finish")
        btn_finish.setObjectName("successBtn")
        btn_finish.clicked.connect(self._on_finish_recording)
        btn_row.addWidget(btn_finish)

        v.addLayout(btn_row)
        return w

    def _build_edit_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        # Header
        name_row = QHBoxLayout()
        self._lbl_edit_name = QLabel()
        self._lbl_edit_name.setStyleSheet(f"font-weight: bold; color: {Settings.FG_PRIMARY};")
        name_row.addWidget(self._lbl_edit_name, stretch=1)
        btn_save_as = QPushButton("Save As…")
        btn_save_as.setObjectName("sessionBtn")
        btn_save_as.clicked.connect(self._on_save_as)
        name_row.addWidget(btn_save_as)
        v.addLayout(name_row)

        self._lbl_edit_path = QLabel()
        self._lbl_edit_path.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 8pt;")
        self._lbl_edit_path.setWordWrap(True)
        v.addWidget(self._lbl_edit_path)

        hint = QLabel("Drag rows to reorder · check/uncheck to enable/disable · Edit to change params")
        hint.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 8pt;")
        hint.setWordWrap(True)
        v.addWidget(hint)

        self._action_list_widget = _ActionListWidget()
        v.addWidget(self._action_list_widget, stretch=1)

        # Run macro button
        run_row = QHBoxLayout()
        self._btn_run_macro = QPushButton("▶  Run Macro on Current File")
        self._btn_run_macro.setObjectName("successBtn")
        self._btn_run_macro.clicked.connect(self._on_run_macro)
        run_row.addWidget(self._btn_run_macro, stretch=1)

        btn_record_again = QPushButton("● Record New")
        btn_record_again.setObjectName("sessionBtn")
        btn_record_again.clicked.connect(self._on_record_again)
        run_row.addWidget(btn_record_again)

        v.addLayout(run_row)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        v.addWidget(btn_close)

        return w

    # ── Panel 0 — Manage actions ──────────────────────────────────────────────

    def _on_new_macro(self) -> None:
        name = self._edit_new_name.text().strip() or "My Macro"
        self._macro_name = name
        self._macro_path = ""
        self._recorded_actions = []
        self._rec_list.clear()
        self._lbl_rec_name.setText(f"Recording: {name}")
        self._lbl_rec_status.setText(
            "Interact with the application — actions will appear here."
        )
        self._lbl_rec_count.setText("0 actions recorded")
        self._btn_pause.setText("⏸  Pause")
        self._btn_pause.setEnabled(True)
        self._blink_state = True
        self._lbl_rec_indicator.setStyleSheet("color: #ff5555; font-weight: bold;")

        # Tell the recorder to start
        if hasattr(self._mw, "macro_recorder"):
            self._mw.macro_recorder.start()

        self._blink_timer.start()
        self._stack.setCurrentIndex(1)

    def _on_open_macro(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Macro", "",
            f"ND2Studios Macro (*{MACRO_EXTENSION});;All files (*)",
        )
        if not path:
            return
        try:
            name, actions = load_macro(path)
        except Exception as exc:
            QMessageBox.warning(self, "Load Failed", str(exc))
            return
        self._macro_name = name
        self._macro_path = path
        self._load_into_edit_panel(name, actions, path)

    # ── Panel 1 — Record actions ──────────────────────────────────────────────

    def _on_action_recorded(self, action_dict: dict) -> None:
        """Slot connected to MainWindow.macro_action_recorded signal."""
        if self._stack.currentIndex() != 1:
            return
        label = action_dict.get("label", action_dict.get("action_type", "Action"))
        item = QListWidgetItem(f"  {label}")
        action_type = action_dict.get("action_type", "")
        icon_map = {
            "recipe_add_step": "🧪",
            "recipe_remove_last": "↩",
            "recipe_clear": "🗑",
            "analysis_run": "🔬",
            "export_tiff": "📄",
            "export_composite": "🖼",
            "export_movie": "🎬",
        }
        icon = icon_map.get(action_type, "•")
        item.setText(f"  {icon}  {label}")
        self._rec_list.addItem(item)
        self._rec_list.scrollToBottom()
        self._recorded_actions.append(MacroAction.from_dict(action_dict))
        n = len(self._recorded_actions)
        self._lbl_rec_count.setText(f"{n} action{'s' if n != 1 else ''} recorded")

    def _on_toggle_pause(self) -> None:
        recorder = getattr(self._mw, "macro_recorder", None)
        if recorder is None:
            return
        if recorder.paused:
            recorder.resume()
            self._btn_pause.setText("⏸  Pause")
            self._lbl_rec_status.setText(
                "Recording — interact with the application."
            )
            self._lbl_rec_indicator.setStyleSheet("color: #ff5555; font-weight: bold;")
            self._blink_timer.start()
        else:
            recorder.pause()
            self._btn_pause.setText("▶  Resume")
            self._lbl_rec_status.setText("Paused — actions will NOT be recorded.")
            self._lbl_rec_indicator.setStyleSheet(
                f"color: {Settings.FG_SECONDARY}; font-weight: bold;"
            )
            self._blink_timer.stop()

    def _on_cancel_recording(self) -> None:
        reply = QMessageBox.question(
            self, "Cancel Recording",
            "Discard all recorded actions and stop recording?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        recorder = getattr(self._mw, "macro_recorder", None)
        if recorder is not None:
            recorder.cancel()
        self._blink_timer.stop()
        self._recorded_actions = []
        self._rec_list.clear()
        self._stack.setCurrentIndex(0)

    def _on_finish_recording(self) -> None:
        recorder = getattr(self._mw, "macro_recorder", None)
        actions = recorder.finish() if recorder is not None else list(self._recorded_actions)
        self._blink_timer.stop()

        if not actions:
            QMessageBox.information(
                self, "No Actions",
                "No actions were recorded. Perform some operations first.",
            )
            self._stack.setCurrentIndex(0)
            return

        # Prompt for save path
        default_name = self._macro_name.replace(" ", "_")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Macro", default_name,
            f"ND2Studios Macro (*{MACRO_EXTENSION})",
        )
        if not path:
            # User cancelled save — still go to edit with unsaved data
            self._macro_path = ""
        else:
            try:
                path = save_macro(self._macro_name, actions, path)
                self._macro_path = path
            except Exception as exc:
                QMessageBox.warning(self, "Save Failed", str(exc))
                self._macro_path = ""

        self._load_into_edit_panel(self._macro_name, actions, self._macro_path)

    def _blink_rec(self) -> None:
        recorder = getattr(self._mw, "macro_recorder", None)
        if recorder is not None and recorder.paused:
            return
        self._blink_state = not self._blink_state
        style = (
            "color: #ff5555; font-weight: bold;"
            if self._blink_state
            else f"color: {Settings.BG_PRIMARY}; font-weight: bold;"
        )
        self._lbl_rec_indicator.setStyleSheet(style)

    # ── Panel 2 — Edit actions ────────────────────────────────────────────────

    def _load_into_edit_panel(
        self,
        name: str,
        actions: List[MacroAction],
        path: str,
    ) -> None:
        self._lbl_edit_name.setText(name)
        self._lbl_edit_path.setText(path or "(unsaved)")
        self._action_list_widget.set_actions(actions)
        self._stack.setCurrentIndex(2)

    def _on_save_as(self) -> None:
        actions = self._action_list_widget.get_actions()
        default = self._macro_name.replace(" ", "_")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Macro As", default,
            f"ND2Studios Macro (*{MACRO_EXTENSION})",
        )
        if not path:
            return
        try:
            final = save_macro(self._macro_name, actions, path)
            self._macro_path = final
            self._lbl_edit_path.setText(final)
            if hasattr(self._mw, "set_status_text"):
                self._mw.set_status_text(f"Macro saved: {os.path.basename(final)}")
        except Exception as exc:
            QMessageBox.warning(self, "Save Failed", str(exc))

    def _on_run_macro(self) -> None:
        actions = self._action_list_widget.get_actions()
        enabled = [a for a in actions if a.enabled]
        if not enabled:
            QMessageBox.information(
                self, "Nothing to Run",
                "No actions are enabled. Check the boxes next to each step.",
            )
            return
        if self._mw is None or self._mw.exp_manager.active is None:
            QMessageBox.warning(
                self, "No File Loaded",
                "Import a file before running a macro.",
            )
            return

        if not hasattr(self._mw, "replay_macro_action"):
            QMessageBox.warning(self, "Not Supported", "Replay not available.")
            return

        for action in enabled:
            self._mw.replay_macro_action(action)

        if hasattr(self._mw, "set_status_text"):
            self._mw.set_status_text(
                f"Macro '{self._macro_name}' applied ({len(enabled)} steps)."
            )

    def _on_record_again(self) -> None:
        """Return to the Manage panel to start a fresh recording."""
        self._stack.setCurrentIndex(0)

    # ── Accessors used by Batch page ──────────────────────────────────────────

    def current_actions(self) -> List[MacroAction]:
        """Return actions currently shown in the Edit panel."""
        return self._action_list_widget.get_actions()
