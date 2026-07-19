"""
Shared widgets: MplCanvas, ParamEditor, StatusIndicator.
Adapted from SerialTrack's widgets/common.py.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox, QLineEdit,
)
from PySide6.QtCore import Signal

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from nd2studios.core.plugin_registry import ParamSpec
from nd2studios.core.settings import Settings
from nd2studios.widgets.icon_button import scale_qss


class MplCanvas(FigureCanvas):
    """Embedded matplotlib figure with Dracula-friendly styling.

    Every canvas ships with a :class:`PlotExporter` bound as ``self.exporter``
    that installs a right-click context menu for axis styling, image export
    (PNG/PDF/SVG/TIFF), and — if the host page registered a frame provider —
    per-frame video export (MP4 / AVI / TIFF). Capture always uses a white
    background regardless of the on-screen Dracula theme.
    """

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi,
                          facecolor=Settings.BG_SECONDARY)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.15)


    def add_subplot(self, *args, **kwargs):
        ax = self.fig.add_subplot(*args, **kwargs)
        ax.set_facecolor(Settings.BG_SECONDARY)
        ax.tick_params(colors=Settings.FG_SECONDARY, labelsize=8)
        ax.xaxis.label.set_color(Settings.FG_SECONDARY)
        ax.yaxis.label.set_color(Settings.FG_SECONDARY)
        ax.title.set_color(Settings.FG_PRIMARY)
        for spine in ax.spines.values():
            spine.set_color(Settings.BORDER_COLOR)
        return ax

    def clear(self):
        self.fig.clear()
        self.draw()

    def safe_tight_layout(self, **kwargs) -> None:
        """``fig.tight_layout()`` guarded against a zero-size figure.

        Rendering while the canvas is hidden or not yet laid out (e.g. a panel in a
        just-shown / collapsed splitter) leaves the figure with a zero dimension,
        which makes matplotlib's ``tight_layout`` raise ``'box_aspect' and
        'fig_aspect' must be positive``. Skip it then (the next render at a real
        size lays out correctly), and never let a layout pass crash the GUI.
        """
        try:
            w, h = self.fig.get_size_inches()
            if float(w) > 0 and float(h) > 0:
                self.fig.tight_layout(**kwargs)
        except Exception:  # noqa: BLE001 — layout must never crash the GUI
            pass

    def safe_draw(self) -> None:
        """``draw()`` guarded against a zero-size figure.

        Drawing while the canvas is hidden or not yet laid out (e.g. a panel that
        is a not-yet-shown tab in a stacked viewer) leaves the figure with a zero
        dimension. Axes with a fixed aspect (``imshow(..., aspect="equal")``) then
        make matplotlib's ``apply_aspect`` raise ``'box_aspect' and 'fig_aspect'
        must be positive`` during the draw. Skip the draw then — the canvas repaints
        at its real size once shown — and never let it crash the GUI.
        """
        try:
            w, h = self.fig.get_size_inches()
            if float(w) > 0 and float(h) > 0:
                self.draw()
        except Exception:  # noqa: BLE001 — a draw must never crash the GUI
            pass


class ParamEditor(QWidget):
    """Auto-generates a parameter form from a list of ParamSpec."""

    params_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QFormLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._specs: List[ParamSpec] = []
        self._widgets: Dict[str, QWidget] = {}
        # Hidden params store their value here — no widget, no form row.
        self._hidden_values: Dict[str, Any] = {}

    def set_params(self, specs: List[ParamSpec]):
        """Replace the form with new parameter specifications."""
        # Clear existing
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._specs = specs
        self._widgets = {}
        self._hidden_values = {}

        for spec in specs:
            if spec.param_type == "hidden":
                # Internal-state slot — populated via set_values(), read via
                # get_values(). Never rendered.
                self._hidden_values[spec.name] = spec.default
                continue
            widget = self._make_widget(spec)
            self._widgets[spec.name] = widget
            label = QLabel(spec.label)
            if spec.tooltip:
                label.setToolTip(spec.tooltip)
                widget.setToolTip(spec.tooltip)
            self._layout.addRow(label, widget)

        self._update_visibility()

    def _make_widget(self, spec: ParamSpec) -> QWidget:
        if spec.param_type == "float":
            w = QDoubleSpinBox()
            w.setDecimals(3)
            if spec.min_val is not None:
                w.setMinimum(spec.min_val)
            if spec.max_val is not None:
                w.setMaximum(spec.max_val)
            if spec.step is not None:
                w.setSingleStep(spec.step)
            w.setValue(spec.default)
            w.valueChanged.connect(lambda: self._emit_changed())
            return w
        elif spec.param_type == "int":
            w = QSpinBox()
            if spec.min_val is not None:
                w.setMinimum(spec.min_val)
            if spec.max_val is not None:
                w.setMaximum(spec.max_val)
            if spec.step is not None:
                w.setSingleStep(spec.step)
            w.setValue(spec.default)
            w.valueChanged.connect(lambda: self._emit_changed())
            return w
        elif spec.param_type == "bool":
            w = QCheckBox()
            w.setChecked(spec.default)
            w.stateChanged.connect(lambda: self._emit_changed())
            return w
        elif spec.param_type == "choice":
            w = QComboBox()
            w.addItems(spec.choices)
            if spec.default in spec.choices:
                w.setCurrentText(spec.default)
            w.currentTextChanged.connect(lambda *_: self._update_visibility())
            w.currentTextChanged.connect(lambda *_: self._emit_changed())
            return w
        else:
            w = QLineEdit(str(spec.default))
            w.textChanged.connect(lambda: self._emit_changed())
            return w

    def _update_visibility(self):
        current_choices = {
            spec.name: self._widgets[spec.name].currentText()
            for spec in self._specs
            if spec.param_type == "choice" and spec.name in self._widgets
        }
        for spec in self._specs:
            if spec.param_type == "hidden" or spec.visible_when is None:
                continue
            w = self._widgets.get(spec.name)
            if w is None:
                continue
            # A visible_when value may be a single choice or a list/tuple/set of
            # choices (the row shows when the current choice is any of them), so
            # one knob can be shared by several methods (e.g. the gap budget used
            # by both the fingerprint and mask-overlap linkers).
            def _match(cur, want):
                return cur in want if isinstance(want, (list, tuple, set)) else cur == want
            visible = all(_match(current_choices.get(k), v)
                          for k, v in spec.visible_when.items())
            self._layout.setRowVisible(w, visible)

    def _emit_changed(self):
        self.params_changed.emit(self.get_values())

    def get_values(self) -> Dict[str, Any]:
        vals: Dict[str, Any] = {}
        for spec in self._specs:
            if spec.param_type == "hidden":
                vals[spec.name] = self._hidden_values.get(spec.name, spec.default)
                continue
            w = self._widgets.get(spec.name)
            if w is None:
                continue
            if spec.param_type == "float":
                vals[spec.name] = w.value()
            elif spec.param_type == "int":
                vals[spec.name] = w.value()
            elif spec.param_type == "bool":
                vals[spec.name] = w.isChecked()
            elif spec.param_type == "choice":
                vals[spec.name] = w.currentText()
            else:
                vals[spec.name] = w.text()
        return vals

    def set_values(self, values: Dict[str, Any]):
        for name, val in values.items():
            spec = next((s for s in self._specs if s.name == name), None)
            if spec is None:
                continue
            if spec.param_type == "hidden":
                self._hidden_values[name] = val
                continue
            w = self._widgets.get(name)
            if w is None:
                continue
            w.blockSignals(True)
            if spec.param_type == "float":
                w.setValue(val)
            elif spec.param_type == "int":
                w.setValue(val)
            elif spec.param_type == "bool":
                w.setChecked(val)
            elif spec.param_type == "choice":
                w.setCurrentText(str(val))
            else:
                w.setText(str(val))
            w.blockSignals(False)
        self._update_visibility()


class StatusIndicator(QLabel):
    """Colored status badge."""

    STATUS_COLORS = {
        "new": Settings.FG_SECONDARY,
        "imported": Settings.ACCENT_ORANGE,
        "preprocessed": Settings.ACCENT_CYAN,
        "ready_to_export": Settings.ACCENT_GREEN,
        "error": Settings.ACCENT_RED,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("statusIndicator")
        self.set_status("new")

    def set_status(self, status: str):
        color = self.STATUS_COLORS.get(status, Settings.FG_SECONDARY)
        self.setText(status.replace("_", " ").title())
        self.setStyleSheet(scale_qss(
            f"background-color: {color}; color: #282a36; "
            f"font: bold 9pt 'Helvetica Neue'; padding: 4px 12px; border-radius: 4px;"
        ))
