"""Loop / iteration settings dialog (V1.49).

Edits a loop edge's config (``loop_edge.params`` — see
:mod:`pipeline_graph.loop`): the iteration mode (parameter sweep / fixed count /
until-condition), the swept parameter axes, the stop condition, the combine rule
for merging per-iteration results, and the multipoint scope.

This is a Qt view over the Qt-free loop config; it never executes — the Run
driver in :mod:`pages.pipelines_page` reads ``result_config()`` and runs the
loop. Opened when a loop wire is created or double-clicked.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea,
    QSpinBox, QVBoxLayout, QWidget,
)

from nd2studios.core.settings import Settings
from nd2studios.pipeline_graph.conditions import Condition, describe_condition
from nd2studios.pipeline_graph.loop import (
    BEST_OBJECT_COUNT, BEST_TRACKING_RATIO, COMBINE_GRID, COMBINE_ZIP,
    DEDUP_CENTROID, DEDUP_IOU, MODE_COUNT, MODE_SWEEP, MODE_UNTIL,
    RULE_BEST, RULE_KEEP_ALL, RULE_LAST, RULE_UNION_DEDUP,
    default_loop_config, expand_axis, iteration_plan,
)
from nd2studios.widgets.icon_button import scaled, scale_qss
from nd2studios.widgets.node_board.condition_builder_dialog import (
    ConditionBuilderDialog,
)

# Node param options passed in by the page:
#   [{"node_id": str, "title": str,
#     "params": [{"name","label","min","max","step","default"}]}]
BodyParamOptions = List[Dict[str, Any]]

_MODE_LABELS = {
    MODE_SWEEP: "Parameter sweep",
    MODE_COUNT: "Fixed count",
    MODE_UNTIL: "Until condition",
}
_RULE_LABELS = {
    RULE_UNION_DEDUP: "Union + dedup by overlap",
    RULE_BEST: "Best iteration",
    RULE_LAST: "Last iteration",
    RULE_KEEP_ALL: "Keep all (tagged)",
}


class _AxisRow(QFrame):
    """One swept parameter axis: node + numeric param, start/stop/step or values."""

    def __init__(self, axis: Dict[str, Any], options: BodyParamOptions,
                 dialog: "LoopSettingsDialog") -> None:
        super().__init__()
        self._options = options
        self._dialog = dialog
        self.setObjectName("loopAxisRow")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(scale_qss(
            f"#loopAxisRow{{background:{Settings.BG_SECONDARY};"
            f"border:1px solid {Settings.BORDER_COLOR};border-radius:6px;}}"
        ))
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(6)

        self._node = QComboBox()
        for opt in options:
            self._node.addItem(opt["title"], opt["node_id"])
        idx = self._node.findData(str(axis.get("node_id", "")))
        if idx >= 0:
            self._node.setCurrentIndex(idx)
        self._node.currentIndexChanged.connect(self._on_node_changed)
        row.addWidget(self._node)

        self._param = QComboBox()
        self._param.setMinimumWidth(scaled(120))
        self._param.currentIndexChanged.connect(self._on_param_changed)
        row.addWidget(self._param)

        row.addWidget(self._lbl("start"))
        self._start = self._spin(axis.get("start", 0.0))
        row.addWidget(self._start)
        row.addWidget(self._lbl("stop"))
        self._stop = self._spin(axis.get("stop", 0.0))
        row.addWidget(self._stop)
        row.addWidget(self._lbl("step"))
        self._step = self._spin(axis.get("step", 1.0))
        row.addWidget(self._step)

        row.addWidget(self._lbl("or values"))
        self._values = QLineEdit()
        self._values.setPlaceholderText("0.3, 0.5, 0.7")
        self._values.setMaximumWidth(scaled(120))
        vals = axis.get("values")
        if isinstance(vals, (list, tuple)) and vals:
            self._values.setText(", ".join(str(v) for v in vals))
        self._values.textChanged.connect(lambda _=None: self._dialog._refresh_count())
        row.addWidget(self._values)

        for w in (self._start, self._stop, self._step):
            w.valueChanged.connect(lambda _=None: self._dialog._refresh_count())

        rm = QPushButton("✕")
        rm.setFixedSize(scaled(22), scaled(22))
        rm.setToolTip("Remove this axis")
        rm.clicked.connect(lambda: self._dialog._remove_axis(self))
        row.addWidget(rm)

        self._populate_params(str(axis.get("param", "")))

    def _lbl(self, text: str) -> QLabel:
        w = QLabel(text)
        w.setStyleSheet(scale_qss(f"color:{Settings.FG_SECONDARY};font:8pt;"))
        return w

    def _spin(self, value: Any) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setDecimals(4)
        w.setRange(-1e9, 1e9)
        try:
            w.setValue(float(value))
        except (TypeError, ValueError):
            w.setValue(0.0)
        w.setMaximumWidth(scaled(80))
        return w

    def _current_params(self) -> List[Dict[str, Any]]:
        nid = self._node.currentData()
        for opt in self._options:
            if opt["node_id"] == nid:
                return opt.get("params", [])
        return []

    def _populate_params(self, select: str = "") -> None:
        self._param.blockSignals(True)
        self._param.clear()
        for p in self._current_params():
            self._param.addItem(p["label"], p["name"])
        if select:
            i = self._param.findData(select)
            if i >= 0:
                self._param.setCurrentIndex(i)
        self._param.blockSignals(False)

    def _on_node_changed(self) -> None:
        self._populate_params()
        self._on_param_changed()

    def _on_param_changed(self) -> None:
        # Seed start/stop/step from the param spec's range when empty-ish.
        name = self._param.currentData()
        spec = next((p for p in self._current_params() if p["name"] == name), None)
        if spec is not None and not self._values.text().strip():
            lo = spec.get("min")
            hi = spec.get("max")
            st = spec.get("step")
            if lo is not None:
                self._start.setValue(float(lo))
            if hi is not None:
                self._stop.setValue(float(hi))
            if st:
                self._step.setValue(float(st))
        self._dialog._refresh_count()

    def to_axis(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "node_id": self._node.currentData(),
            "param": self._param.currentData(),
        }
        text = self._values.text().strip()
        if text:
            vals: List[float] = []
            for tok in text.replace(";", ",").split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    vals.append(float(tok))
                except ValueError:
                    continue
            out["values"] = vals
        else:
            out["start"] = float(self._start.value())
            out["stop"] = float(self._stop.value())
            out["step"] = float(self._step.value())
        return out


class LoopSettingsDialog(QDialog):
    """Edit a loop edge's iteration / stop / combine config."""

    def __init__(self, config: Optional[Dict[str, Any]],
                 body_param_options: BodyParamOptions,
                 region_label: str = "",
                 channels: Optional[List[str]] = None,
                 metrics: Optional[List[str]] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Loop settings")
        self.setModal(True)
        self.setMinimumSize(scaled(760), scaled(560))
        self._cfg = dict(config or default_loop_config())
        self._options = list(body_param_options or [])
        self._channels = list(channels or [])
        self._metrics = list(metrics or [])
        self._axis_rows: List[_AxisRow] = []
        self._stop_condition: Dict[str, Any] = dict(self._cfg.get("stop") or {}) \
            if isinstance(self._cfg.get("stop"), dict) else {}
        self._build_ui(region_label)
        self._load()

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self, region_label: str) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        if region_label:
            hdr = QLabel(f"Loop region: {region_label}")
            hdr.setStyleSheet(scale_qss(f"color:{Settings.FG_SECONDARY};font:9pt;"))
            outer.addWidget(hdr)

        # Iteration group
        it_box = QGroupBox("Iteration")
        it_form = QFormLayout(it_box)
        self._mode = QComboBox()
        for key in (MODE_SWEEP, MODE_COUNT, MODE_UNTIL):
            self._mode.addItem(_MODE_LABELS[key], key)
        self._mode.currentIndexChanged.connect(self._on_mode_changed)
        it_form.addRow("Mode", self._mode)

        self._count = QSpinBox()
        self._count.setRange(1, 100000)
        self._count.valueChanged.connect(lambda _=None: self._refresh_count())
        self._count_row = self._count
        it_form.addRow("Iterations", self._count)

        self._combine_axes = QComboBox()
        self._combine_axes.addItem("Grid (all combinations)", COMBINE_GRID)
        self._combine_axes.addItem("Zip (paired)", COMBINE_ZIP)
        self._combine_axes.currentIndexChanged.connect(
            lambda _=None: self._refresh_count())
        it_form.addRow("Combine axes", self._combine_axes)

        self._max_iter = QSpinBox()
        self._max_iter.setRange(1, 100000)
        self._max_iter.valueChanged.connect(lambda _=None: self._refresh_count())
        it_form.addRow("Max iterations (cap)", self._max_iter)
        outer.addWidget(it_box)

        # Axes group (sweep / until)
        self._axes_box = QGroupBox("Swept parameters")
        axes_v = QVBoxLayout(self._axes_box)
        add_bar = QHBoxLayout()
        self._add_axis_btn = QPushButton(" Add axis")
        self._add_axis_btn.setObjectName("pipelineToolBtn")
        self._add_axis_btn.clicked.connect(lambda: self._add_axis({}))
        add_bar.addWidget(self._add_axis_btn)
        add_bar.addStretch(1)
        self._count_lbl = QLabel("")
        self._count_lbl.setStyleSheet(scale_qss(f"color:{Settings.ACCENT_GOLD};font:bold 9pt;"))
        add_bar.addWidget(self._count_lbl)
        axes_v.addLayout(add_bar)

        self._axes_host = QWidget()
        self._axes_layout = QVBoxLayout(self._axes_host)
        self._axes_layout.setContentsMargins(0, 0, 0, 0)
        self._axes_layout.setSpacing(6)
        self._axes_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self._axes_host)
        scroll.setMinimumHeight(scaled(140))
        axes_v.addWidget(scroll)
        outer.addWidget(self._axes_box)

        # Stop condition (until)
        self._stop_box = QGroupBox("Stop condition")
        stop_h = QHBoxLayout(self._stop_box)
        self._stop_readout = QLabel("")
        self._stop_readout.setWordWrap(True)
        self._stop_readout.setStyleSheet(scale_qss(f"color:{Settings.FG_PRIMARY};font:9pt;"))
        stop_h.addWidget(self._stop_readout, 1)
        self._edit_stop_btn = QPushButton("Edit condition…")
        self._edit_stop_btn.clicked.connect(self._edit_stop)
        stop_h.addWidget(self._edit_stop_btn)
        outer.addWidget(self._stop_box)

        # Combine results
        cb_box = QGroupBox("Combine results")
        cb_form = QFormLayout(cb_box)
        self._rule = QComboBox()
        for key in (RULE_UNION_DEDUP, RULE_BEST, RULE_LAST, RULE_KEEP_ALL):
            self._rule.addItem(_RULE_LABELS[key], key)
        self._rule.currentIndexChanged.connect(self._on_rule_changed)
        cb_form.addRow("Rule", self._rule)

        self._dedup_metric = QComboBox()
        self._dedup_metric.addItem("Mask IoU (centroid fallback)", DEDUP_IOU)
        self._dedup_metric.addItem("Centroid distance", DEDUP_CENTROID)
        cb_form.addRow("Duplicate test", self._dedup_metric)
        self._iou = QDoubleSpinBox()
        self._iou.setRange(0.0, 1.0)
        self._iou.setSingleStep(0.05)
        self._iou.setDecimals(2)
        cb_form.addRow("IoU threshold", self._iou)
        self._cdist = QDoubleSpinBox()
        self._cdist.setRange(0.0, 100000.0)
        self._cdist.setDecimals(1)
        cb_form.addRow("Centroid distance (px)", self._cdist)
        self._best_metric = QComboBox()
        self._best_metric.addItem("Most objects", BEST_OBJECT_COUNT)
        self._best_metric.addItem("Best tracking ratio", BEST_TRACKING_RATIO)
        cb_form.addRow("Best by", self._best_metric)
        self._dedup_metric.currentIndexChanged.connect(self._on_rule_changed)
        # Retain every iteration's result so the viewer's iteration dropdown can
        # step through them (independent of the combine rule; opt-in for RAM).
        self._save_iters = QCheckBox("Save all iterations for viewing")
        self._save_iters.setToolTip(
            "Keep each iteration's masks / tracks so the image viewer's iteration "
            "dropdown can step through them. Holds every iteration in memory.")
        cb_form.addRow("", self._save_iters)
        outer.addWidget(cb_box)

        # Multipoint
        mp_box = QGroupBox("Multipoint scope")
        mp_form = QFormLayout(mp_box)
        self._multipoint = QComboBox()
        self._multipoint.addItem("Current multipoint only", "current")
        self._multipoint.addItem("All multipoints", "all")
        mp_form.addRow("Run loop for", self._multipoint)
        outer.addWidget(mp_box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    # ── load / mode ─────────────────────────────────────────────────────────
    def _load(self) -> None:
        cfg = self._cfg
        mi = self._mode.findData(str(cfg.get("mode", MODE_COUNT)))
        self._mode.setCurrentIndex(mi if mi >= 0 else 1)
        self._count.setValue(int(cfg.get("count", 3) or 3))
        cai = self._combine_axes.findData(str(cfg.get("combine_axes", COMBINE_GRID)))
        self._combine_axes.setCurrentIndex(cai if cai >= 0 else 0)
        self._max_iter.setValue(int(cfg.get("max_iterations", 64) or 64))
        for axis in cfg.get("axes", []) or []:
            self._add_axis(axis)
        combine = cfg.get("combine", {}) or {}
        ri = self._rule.findData(str(combine.get("rule", RULE_LAST)))
        self._rule.setCurrentIndex(ri if ri >= 0 else 2)
        dedup = combine.get("dedup", {}) or {}
        dmi = self._dedup_metric.findData(str(dedup.get("metric", DEDUP_IOU)))
        self._dedup_metric.setCurrentIndex(dmi if dmi >= 0 else 0)
        self._iou.setValue(float(dedup.get("iou_threshold", 0.3)))
        self._cdist.setValue(float(dedup.get("centroid_distance", 10.0)))
        bmi = self._best_metric.findData(str(combine.get("best_metric", BEST_OBJECT_COUNT)))
        self._best_metric.setCurrentIndex(bmi if bmi >= 0 else 0)
        mpi = self._multipoint.findData(str(cfg.get("multipoint", "current")))
        self._multipoint.setCurrentIndex(mpi if mpi >= 0 else 0)
        self._save_iters.setChecked(bool(cfg.get("save_iterations", False)))
        self._on_mode_changed()
        self._on_rule_changed()
        self._refresh_stop_readout()

    def _on_mode_changed(self) -> None:
        mode = self._mode.currentData()
        self._count.setEnabled(mode == MODE_COUNT)
        self._axes_box.setVisible(mode in (MODE_SWEEP, MODE_UNTIL))
        self._combine_axes.setEnabled(mode in (MODE_SWEEP, MODE_UNTIL))
        self._stop_box.setVisible(mode == MODE_UNTIL)
        self._refresh_count()

    def _on_rule_changed(self) -> None:
        rule = self._rule.currentData()
        is_dedup = rule == RULE_UNION_DEDUP
        metric = self._dedup_metric.currentData()
        self._dedup_metric.setEnabled(is_dedup)
        self._iou.setEnabled(is_dedup and metric == DEDUP_IOU)
        self._cdist.setEnabled(is_dedup)
        self._best_metric.setEnabled(rule == RULE_BEST)

    # ── axes ──────────────────────────────────────────────────────────────
    def _add_axis(self, axis: Dict[str, Any]) -> None:
        if not self._options:
            return
        row = _AxisRow(axis, self._options, self)
        self._axis_rows.append(row)
        self._axes_layout.insertWidget(self._axes_layout.count() - 1, row)
        self._refresh_count()

    def _remove_axis(self, row: _AxisRow) -> None:
        if row in self._axis_rows:
            self._axis_rows.remove(row)
            self._axes_layout.removeWidget(row)
            row.deleteLater()
            self._refresh_count()

    def _refresh_count(self) -> None:
        try:
            plan = iteration_plan(self._peek_config())
            n = len(plan)
        except Exception:  # noqa: BLE001
            n = 0
        self._count_lbl.setText(f"{n} iteration(s)")

    # ── stop condition ────────────────────────────────────────────────────
    def _edit_stop(self) -> None:
        dlg = ConditionBuilderDialog(
            self._stop_condition or None,
            channels=self._channels, metrics=self._metrics, parent=self)
        dlg.setWindowTitle("Edit loop stop condition")
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._stop_condition = dlg.result_dict()
            self._refresh_stop_readout()

    def _refresh_stop_readout(self) -> None:
        if self._stop_condition:
            cond = Condition.from_dict(self._stop_condition)
            self._stop_readout.setText("Stop when:  " + describe_condition(cond))
        else:
            self._stop_readout.setText("Stop when the sweep / cap is exhausted "
                                       "(no condition set).")

    # ── result ─────────────────────────────────────────────────────────────
    def _peek_config(self) -> Dict[str, Any]:
        mode = self._mode.currentData()
        return {
            "mode": mode,
            "count": int(self._count.value()),
            "max_iterations": int(self._max_iter.value()),
            "combine_axes": self._combine_axes.currentData(),
            "axes": [r.to_axis() for r in self._axis_rows],
        }

    def result_config(self) -> Dict[str, Any]:
        cfg = default_loop_config()
        cfg.update(self._peek_config())
        cfg["version"] = 1
        cfg["stop"] = self._stop_condition or None
        cfg["combine"] = {
            "rule": self._rule.currentData(),
            "dedup": {
                "metric": self._dedup_metric.currentData(),
                "iou_threshold": float(self._iou.value()),
                "centroid_distance": float(self._cdist.value()),
            },
            "best_metric": self._best_metric.currentData(),
        }
        cfg["multipoint"] = self._multipoint.currentData()
        cfg["save_iterations"] = bool(self._save_iters.isChecked())
        return cfg
