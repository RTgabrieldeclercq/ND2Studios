"""
Macro event filter (V1.0).

Installed on QApplication when a macro recording starts; removed when it
stops.  Captures every meaningful user interaction across all known pages:

  button_click   — any QPushButton press inside a page
  combo_change   — QComboBox selection change
  spinbox_change — QAbstractSpinBox value change (debounced 600 ms)
  checkbox_change — QCheckBox toggle
  param_change   — ParamEditor.params_changed (full params dict)

Interactions inside QDialogs (file pickers, message boxes, config wizard)
and inside the MacroDialog itself are ignored.

Widget identification uses a stable three-tier scheme:
  "attr:<name>"  — the widget is a named attribute of its page object
  "name:<name>"  — widget.objectName() is set and meaningful
  "text:<text>"  — button text (fallback for unnamed buttons)

The matching helper ``find_widget(page, widget_id)`` is exported so that
``MainWindow.replay_macro_action`` can locate widgets at replay time.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtWidgets import (
    QAbstractSpinBox, QApplication, QCheckBox, QComboBox,
    QDialog, QPushButton, QWidget,
)

from nd2studios.backend.macro_engine import MacroAction

if TYPE_CHECKING:
    pass

# objectNames that belong to UI chrome — never record these.
_SKIP_NAMES = frozenset({
    "titleBarBtn", "titleBarCloseBtn", "sessionBtn", "toggleBtn",
    "navBtn",          # page navigation handled via _navigate() hook
    "dangerBtn",       # generic class name used on many cancel/delete buttons,
                       # but dangerBtn buttons inside pages ARE caught via text
    "qt_spinbox_lineedit",
})


# ---------------------------------------------------------------------------
# Widget identification helpers
# ---------------------------------------------------------------------------

def _widget_id(widget: QWidget, page: QWidget) -> str:
    """Return a stable string identifier for *widget* relative to *page*."""
    for attr_name, val in vars(page).items():
        if val is widget:
            return f"attr:{attr_name}"
    name = widget.objectName()
    if name and name not in _SKIP_NAMES and not name.startswith("qt_"):
        return f"name:{name}"
    if isinstance(widget, QPushButton):
        text = widget.text().strip()
        if text:
            return f"text:{text}"
    return ""  # unidentifiable — skip


def find_widget(page: QWidget, widget_id: str) -> Optional[QWidget]:
    """Locate a widget inside *page* using its stored identifier."""
    if not widget_id:
        return None
    if widget_id.startswith("attr:"):
        return getattr(page, widget_id[5:], None)
    if widget_id.startswith("name:"):
        return page.findChild(QWidget, widget_id[5:])
    if widget_id.startswith("text:"):
        text = widget_id[5:]
        for btn in page.findChildren(QPushButton):
            if btn.text().strip() == text:
                return btn
    return None


# ---------------------------------------------------------------------------
# Event filter
# ---------------------------------------------------------------------------

class MacroEventFilter(QObject):
    """
    Application-level event filter + signal watcher for macro recording.

    Call ``install()`` to start capturing and ``uninstall()`` to stop.
    """

    def __init__(self, main_window) -> None:
        super().__init__(main_window)
        self._mw = main_window
        self._signal_connections: List[Tuple[Any, Any]] = []
        self._spinbox_timers: Dict[int, QTimer] = {}
        self._channel_timers: Dict[int, QTimer] = {}  # viewer id → debounce timer
        self._installed = False

    # ── lifecycle ────────────────────────────────────────────────────────

    def install(self) -> None:
        if self._installed:
            return
        QApplication.instance().installEventFilter(self)
        self._connect_value_widgets()
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        QApplication.instance().removeEventFilter(self)
        self._disconnect_value_widgets()
        for timer in self._spinbox_timers.values():
            timer.stop()
        self._spinbox_timers.clear()
        for timer in self._channel_timers.values():
            timer.stop()
        self._channel_timers.clear()
        self._installed = False

    # ── Qt event filter ──────────────────────────────────────────────────

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if (isinstance(obj, QPushButton)
                and event.type() == QEvent.Type.MouseButtonRelease
                and obj.isEnabled()):
            self._on_button(obj)
        return False  # never consume events

    def _on_button(self, btn: QPushButton) -> None:
        # Ignore UI chrome by objectName.
        if btn.objectName() in _SKIP_NAMES:
            return
        # Only record buttons that live inside a known page.
        page_key = self._page_for_widget(btn)
        if page_key is None:
            return
        # Skip interactions inside any QDialog (file picker, message box, etc.)
        if self._inside_dialog(btn):
            return
        page = self._mw.pages.get(page_key)
        wid = _widget_id(btn, page) if page else f"text:{btn.text().strip()}"
        if not wid:
            return
        self._record(MacroAction(
            "button_click",
            f"Click: {btn.text().strip()}",
            {"page": page_key, "widget_id": wid},
        ))

    # ── Signal connections (combos, spinboxes, checkboxes, param editors) ──

    def _connect_value_widgets(self) -> None:
        for page_key, page in self._mw.pages.items():
            # ParamEditor widgets — record full params dict on change.
            for attr_name, val in vars(page).items():
                if hasattr(val, "params_changed") and hasattr(val, "get_values"):
                    wid = f"attr:{attr_name}"
                    slot = self._make_param_slot(page_key, wid, val)
                    val.params_changed.connect(slot)
                    self._signal_connections.append((val.params_changed, slot))

            # Individual combos, spinboxes, checkboxes (page attributes only).
            for attr_name, val in vars(page).items():
                if isinstance(val, QComboBox) and not isinstance(val, QAbstractSpinBox):
                    wid = f"attr:{attr_name}"
                    slot = self._make_combo_slot(page_key, wid)
                    val.currentTextChanged.connect(slot)
                    self._signal_connections.append((val.currentTextChanged, slot))

                elif isinstance(val, QAbstractSpinBox):
                    wid = f"attr:{attr_name}"
                    slot = self._make_spinbox_slot(page_key, wid, id(val))
                    val.valueChanged.connect(slot)
                    self._signal_connections.append((val.valueChanged, slot))

                elif isinstance(val, QCheckBox):
                    wid = f"attr:{attr_name}"
                    slot = self._make_checkbox_slot(page_key, wid)
                    val.stateChanged.connect(slot)
                    self._signal_connections.append((val.stateChanged, slot))

            # Viewer channel state (chip enable/disable, color, LUT contrast/gamma).
            # channels_changed fires for ALL of these, so one signal is enough.
            self._connect_viewer_signals(page_key, page)

    def _connect_viewer_signals(self, page_key: str, page: QWidget) -> None:
        """Connect channels_changed on every MultiAxisViewer found on *page*."""
        try:
            from nd2studios.widgets.multi_axis_viewer import MultiAxisViewer
        except ImportError:
            return

        seen_ids: set = set()

        # Direct instance variables (e.g. analysis_page.viewer)
        candidates = [(n, v) for n, v in vars(page).items()
                      if isinstance(v, MultiAxisViewer)]
        # Properties / dynamic accessors (e.g. recipe_page.viewer_raw/proc)
        for attr in ("viewer", "viewer_raw", "viewer_proc"):
            v = getattr(page, attr, None)
            if isinstance(v, MultiAxisViewer):
                candidates.append((attr, v))

        for attr_name, viewer in candidates:
            vid = id(viewer)
            if vid in seen_ids:
                continue
            seen_ids.add(vid)
            slot = self._make_channel_state_slot(page_key, attr_name, viewer)
            viewer.channels_changed.connect(slot)
            self._signal_connections.append((viewer.channels_changed, slot))

    def _disconnect_value_widgets(self) -> None:
        for signal, slot in self._signal_connections:
            try:
                signal.disconnect(slot)
            except RuntimeError:
                pass
        self._signal_connections.clear()

    # ── Slot factories ────────────────────────────────────────────────────

    def _make_combo_slot(self, page_key: str, widget_id: str):
        name = widget_id.split(":", 1)[-1]
        def slot(text: str) -> None:
            if not text:
                return
            self._record(MacroAction(
                "combo_change",
                f"Select [{name}]: {text}",
                {"page": page_key, "widget_id": widget_id, "value": text},
            ))
        return slot

    def _make_spinbox_slot(self, page_key: str, widget_id: str, obj_id: int):
        name = widget_id.split(":", 1)[-1]
        def slot(value: float) -> None:
            existing = self._spinbox_timers.pop(obj_id, None)
            if existing is not None:
                existing.stop()
            timer = QTimer()
            timer.setSingleShot(True)
            timer.setInterval(600)

            def fire() -> None:
                self._spinbox_timers.pop(obj_id, None)
                self._record(MacroAction(
                    "spinbox_change",
                    f"Set [{name}] = {value}",
                    {"page": page_key, "widget_id": widget_id, "value": value},
                ))
            timer.timeout.connect(fire)
            self._spinbox_timers[obj_id] = timer
            timer.start()
        return slot

    def _make_checkbox_slot(self, page_key: str, widget_id: str):
        def slot(state: int) -> None:
            checked = bool(state)
            label = f"{'Check' if checked else 'Uncheck'}: {widget_id.split(':',1)[-1]}"
            self._record(MacroAction(
                "checkbox_change",
                label,
                {"page": page_key, "widget_id": widget_id, "value": checked},
            ))
        return slot

    def _make_channel_state_slot(self, page_key: str, viewer_attr: str, viewer):
        viewer_id = id(viewer)
        def slot() -> None:
            existing = self._channel_timers.pop(viewer_id, None)
            if existing is not None:
                existing.stop()
            timer = QTimer()
            timer.setSingleShot(True)
            timer.setInterval(800)

            def fire() -> None:
                self._channel_timers.pop(viewer_id, None)
                try:
                    state = viewer.channel_state()
                except Exception:
                    return
                parts = []
                for ch, info in state.items():
                    en = "✓" if info.get("enabled", True) else "✗"
                    color = info.get("color", "")
                    lo = info.get("lut_lo", 0)
                    hi = info.get("lut_hi", 65535)
                    g = info.get("lut_gamma", 1.0)
                    parts.append(f"{ch} {en} {color} LUT[{lo:.0f}–{hi:.0f} γ{g:.2f}]")
                label = "Channels: " + "  |  ".join(parts)
                self._record(MacroAction(
                    "channel_state",
                    label,
                    {"page": page_key, "viewer_attr": viewer_attr, "channels": state},
                ))
            timer.timeout.connect(fire)
            self._channel_timers[viewer_id] = timer
            timer.start()
        return slot

    def _make_param_slot(self, page_key: str, widget_id: str, param_editor):
        name = widget_id.split(":", 1)[-1]
        def slot() -> None:
            try:
                values = param_editor.get_values()
            except Exception:
                return
            pairs = [f"{k}={v}" for k, v in values.items()]
            summary = ", ".join(pairs) if pairs else "(no params)"
            self._record(MacroAction(
                "param_change",
                f"Params [{name}]: {summary}",
                {"page": page_key, "widget_id": widget_id, "values": values},
            ))
        return slot

    # ── Core recording helper ─────────────────────────────────────────────

    def _record(self, action: MacroAction) -> None:
        recorder = self._mw.macro_recorder
        if recorder.recording and not recorder.paused and not recorder.replaying:
            self._mw.record_macro_action(action)

    # ── Widget location helpers ────────────────────────────────────────────

    def _page_for_widget(self, widget: QWidget) -> Optional[str]:
        """Walk up the widget's parent chain to find which page it belongs to."""
        parent: Optional[QWidget] = widget
        while parent is not None:
            for key, page in self._mw.pages.items():
                if parent is page:
                    return key
            parent = parent.parent()
        return None

    def _inside_dialog(self, widget: QWidget) -> bool:
        """Return True if *widget* is inside any QDialog (file picker, etc.)."""
        parent: Optional[QWidget] = widget.parent()
        while parent is not None:
            if isinstance(parent, QDialog):
                return True
            parent = parent.parent()
        return False
