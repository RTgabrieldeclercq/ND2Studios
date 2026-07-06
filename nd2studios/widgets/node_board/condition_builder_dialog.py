"""If-else condition builder dialog (V1.45, Phase 2).

Edits an if-else node's nested condition (``pipeline_graph/conditions.py``): a
list of **blocks** combined with ALL (and) / ANY (or), each block optionally
negated (NOT). Blocks are added from a family-grouped menu (results-number /
object-population / timelapse), and each renders its own small param editors.
A live readout mirrors ``describe_condition``.

This is a Qt view over the Qt-free condition model; it never evaluates — the Run
executor does that via ``evaluate_condition``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFrame, QHBoxLayout,
    QLabel, QMenu, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from nd2studios.core.settings import Settings
from nd2studios.pipeline_graph.conditions import (
    FAMILY_ORDER, BLOCK_KINDS, Condition, ConditionBlock, LENS_OBJECT,
    block_label, block_param_schema, describe_condition, families,
    is_object_lens_only, make_block, metric_choices,
)
from nd2studios.widgets.icon_button import scaled


class _BlockRow(QFrame):
    """One condition block: NOT toggle, title, its param editors, and remove."""

    def __init__(self, block: ConditionBlock, channels: List[str],
                 dialog: "ConditionBuilderDialog") -> None:
        super().__init__()
        self.block = block
        self._channels = channels
        self._dialog = dialog
        self.setObjectName("conditionBlockRow")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            f"#conditionBlockRow{{background:{Settings.BG_SECONDARY};"
            f"border:1px solid {Settings.BORDER_COLOR};border-radius:6px;}}"
        )
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(8)

        self._not_btn = QPushButton("NOT")
        self._not_btn.setCheckable(True)
        self._not_btn.setChecked(bool(block.negate))
        self._not_btn.setFixedWidth(scaled(46))
        self._not_btn.setToolTip("Negate this block")
        self._not_btn.setStyleSheet(
            "QPushButton{border:1px solid %s;border-radius:4px;padding:2px;"
            "color:%s;font:bold 8pt;}"
            "QPushButton:checked{background:%s;color:%s;border-color:%s;}"
            % (Settings.BORDER_COLOR, Settings.FG_SECONDARY,
               Settings.ACCENT_RED, Settings.BG_PRIMARY, Settings.ACCENT_RED)
        )
        self._not_btn.toggled.connect(self._on_negate)
        row.addWidget(self._not_btn)

        title = QLabel(block_label(block.kind))
        title.setStyleSheet(f"color:{Settings.ACCENT_PURPLE};font:bold 9pt;")
        title.setMinimumWidth(scaled(150))
        row.addWidget(title)

        for pspec in block_param_schema(block.kind):
            lab = QLabel(pspec["label"])
            lab.setStyleSheet(f"color:{Settings.FG_SECONDARY};font:8.5pt;")
            row.addWidget(lab)
            row.addWidget(self._make_editor(pspec))
        row.addStretch(1)

        # Always-visible text close button (an icon-font glyph can fail to
        # render on some installs, leaving no visible way to remove a block).
        rm = QPushButton("✕")
        rm.setObjectName("conditionRemoveBtn")
        rm.setToolTip("Remove this condition block")
        rm.setFixedSize(scaled(24), scaled(24))
        rm.setStyleSheet(
            "QPushButton{border:1px solid %s;border-radius:4px;"
            "color:%s;font:bold 11pt;background:transparent;}"
            "QPushButton:hover{background:%s;color:%s;border-color:%s;}"
            % (Settings.BORDER_COLOR, Settings.FG_SECONDARY,
               Settings.ACCENT_RED, Settings.BG_PRIMARY, Settings.ACCENT_RED)
        )
        rm.clicked.connect(lambda: self._dialog.remove_row(self))
        row.addWidget(rm)

    def _set(self, name: str, value: Any) -> None:
        self.block.params[name] = value
        self._dialog.refresh_readout()

    def _on_negate(self, on: bool) -> None:
        self.block.negate = bool(on)
        self._dialog.refresh_readout()

    def _make_editor(self, pspec: Dict[str, Any]) -> QWidget:
        name = pspec["name"]
        ptype = pspec["type"]
        value = self.block.params.get(name, pspec.get("default"))
        if ptype in ("choice", "channel"):
            w = QComboBox()
            if ptype == "channel":
                opts = self._channels or [""]
            elif name == "metric":
                # Union the static metric list with the live columns discovered
                # upstream (per-channel intensities, Cell-Tracker results, …).
                opts = metric_choices(self._dialog._metrics)
            else:
                opts = [str(c) for c in pspec["choices"]]
            w.addItems(opts)
            if str(value) in opts:
                w.setCurrentText(str(value))
            elif opts:
                self.block.params[name] = opts[0]
            w.setMinimumWidth(scaled(96))
            w.currentTextChanged.connect(lambda v, n=name: self._set(n, v))
            return w
        if ptype == "int":
            w = QSpinBox()
            w.setRange(-1_000_000, 1_000_000)
            w.setValue(int(value or 0))
            w.valueChanged.connect(lambda v, n=name: self._set(n, int(v)))
            return w
        w = QDoubleSpinBox()
        w.setDecimals(3)
        w.setRange(-1e9, 1e9)
        try:
            w.setValue(float(value or 0.0))
        except (TypeError, ValueError):
            w.setValue(0.0)
        w.valueChanged.connect(lambda v, n=name: self._set(n, float(v)))
        return w


class ConditionBuilderDialog(QDialog):
    """Build an if-else condition. ``result_condition()`` is the edited tree."""

    def __init__(self, condition: Optional[Dict[str, Any]],
                 channels: Optional[List[str]] = None,
                 metrics: Optional[List[str]] = None,
                 lens: Optional[str] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit branch condition")
        self.setObjectName("conditionBuilder")
        self.setModal(True)
        self.setMinimumSize(scaled(720), scaled(440))
        self._channels = list(channels or [])
        # Live metric columns discovered upstream (unioned into the metric
        # dropdown on top of the static catalog).
        self._metrics = list(metrics or [])
        # The if-else lens (whole-frame vs object). Object-lens-only blocks (e.g.
        # per-track persistence) are hidden from the add menu for a whole-frame
        # if-else, where they cannot work.
        self._object_lens = str(lens or "") == LENS_OBJECT
        self._cond = Condition.from_dict(condition)
        self._rows: List[_BlockRow] = []
        self._build_ui()
        for blk in self._cond.blocks:
            self._add_row(blk)
        self.refresh_readout()

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(QLabel("Branch to TRUE when"))
        self._combine = QComboBox()
        self._combine.addItems(["ALL of these are true", "ANY of these are true"])
        self._combine.setCurrentIndex(0 if self._cond.combine.upper() == "ALL" else 1)
        self._combine.currentIndexChanged.connect(self.refresh_readout)
        top.addWidget(self._combine)
        top.addStretch(1)

        self._add_btn = QPushButton(" Add block")
        self._add_btn.setObjectName("pipelineToolBtn")
        self._add_menu = QMenu(self._add_btn)
        self._build_add_menu()
        self._add_btn.setMenu(self._add_menu)
        top.addWidget(self._add_btn)
        outer.addLayout(top)

        # Scrollable block list.
        self._blocks_host = QWidget()
        self._blocks_layout = QVBoxLayout(self._blocks_host)
        self._blocks_layout.setContentsMargins(0, 0, 0, 0)
        self._blocks_layout.setSpacing(6)
        self._blocks_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self._blocks_host)
        outer.addWidget(scroll, stretch=1)

        self._readout = QLabel("")
        self._readout.setWordWrap(True)
        self._readout.setStyleSheet(
            f"color:{Settings.FG_PRIMARY};background:{Settings.BG_PRIMARY};"
            f"border:1px solid {Settings.BORDER_COLOR};border-radius:6px;"
            "padding:6px 8px;font:9pt;")
        outer.addWidget(self._readout)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _build_add_menu(self) -> None:
        fam = families()
        for family in FAMILY_ORDER:
            kinds = fam.get(family) or []
            if not kinds:
                continue
            visible = [k for k in kinds
                       if self._object_lens or not is_object_lens_only(k)]
            if not visible:
                continue
            sub = self._add_menu.addMenu(family)
            for kind in visible:
                act = sub.addAction(BLOCK_KINDS[kind]["label"])
                act.triggered.connect(lambda _c=False, k=kind: self._add_kind(k))

    # ── block rows ──────────────────────────────────────────────────────────
    def _add_kind(self, kind: str) -> None:
        self._add_row(make_block(kind))
        self.refresh_readout()

    def _add_row(self, block: ConditionBlock) -> None:
        row = _BlockRow(block, self._channels, self)
        self._rows.append(row)
        # Insert before the trailing stretch.
        self._blocks_layout.insertWidget(self._blocks_layout.count() - 1, row)

    def remove_row(self, row: _BlockRow) -> None:
        if row in self._rows:
            self._rows.remove(row)
            self._blocks_layout.removeWidget(row)
            row.deleteLater()
            self.refresh_readout()

    # ── readout / result ──────────────────────────────────────────────────
    def refresh_readout(self) -> None:
        cond = self.result_condition()
        self._readout.setText("Branch TRUE when:  " + describe_condition(cond))

    def result_condition(self) -> Condition:
        combine = "ALL" if self._combine.currentIndex() == 0 else "ANY"
        return Condition(combine=combine, blocks=[r.block for r in self._rows])

    def result_dict(self) -> Dict[str, Any]:
        return self.result_condition().to_dict()
