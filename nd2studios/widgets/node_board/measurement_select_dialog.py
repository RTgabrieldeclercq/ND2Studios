"""Measurement-selection dialog for the Compute Measurements node (V1.45, Phase 2).

A roomy checklist of every quantity ``results_engine.compute_measurements`` can
produce, grouped (Size / Change / Position / Shape / Bounding box / Intensity).
The user ticks the metrics they want; only those are computed (the analysis is
**not** re-run — measurements come from the already-computed label masks), so
asking for a couple of metrics is much cheaper than the full set.

Identity columns (segmentation channel, frame, label id) are always included and
not shown. Tracking columns are added by the object tracker, not here.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from nd2studios.core.settings import Settings
from nd2studios.widgets.icon_button import scaled, scale_qss

# (group, [(metric_key, label)]). ``mean_intensity`` / ``std_intensity`` are
# group tokens that apply to every channel.
METRIC_GROUPS: List[tuple] = [
    ("Size", [
        ("area_px", "Area (px²)"), ("area_um2", "Area (µm²)"),
        ("volume_um3", "Volume (µm³)"),
    ]),
    ("Change over time (Δ)", [
        ("delta_area_px", "Δ Area (px²)"), ("delta_area_um2", "Δ Area (µm²)"),
        ("delta_volume_um3", "Δ Volume (µm³)"),
    ]),
    ("Position", [
        ("centroid_y_px", "Centroid Y (px)"), ("centroid_x_px", "Centroid X (px)"),
        ("centroid_y_um", "Centroid Y (µm)"), ("centroid_x_um", "Centroid X (µm)"),
        ("centroid_y_stage_um", "Centroid Y stage (µm)"),
        ("centroid_x_stage_um", "Centroid X stage (µm)"),
    ]),
    ("Shape", [
        ("perimeter", "Perimeter"), ("circularity", "Circularity"),
        ("eccentricity", "Eccentricity"), ("solidity", "Solidity"),
    ]),
    ("Bounding box", [
        ("bbox_min_row", "BBox min row"), ("bbox_min_col", "BBox min col"),
        ("bbox_max_row", "BBox max row"), ("bbox_max_col", "BBox max col"),
    ]),
    ("Intensity (per channel)", [
        ("mean_intensity", "Mean intensity"), ("std_intensity", "Std intensity"),
    ]),
]

# Sensible starting selection (matches the Results page defaults).
DEFAULT_METRICS: Set[str] = {
    "area_um2", "delta_area_um2", "centroid_y_um", "centroid_x_um",
    "circularity", "eccentricity", "solidity", "mean_intensity",
}


def all_metric_keys() -> Set[str]:
    return {k for _g, items in METRIC_GROUPS for k, _l in items}


class MeasurementSelectDialog(QDialog):
    """Pick which per-object metrics to compute. ``selected()`` after Accepted."""

    def __init__(self, selected: Optional[Set[str]] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select measurements")
        self.setObjectName("measurementSelect")
        self.setModal(True)
        self.setMinimumSize(scaled(560), scaled(560))
        self._boxes: Dict[str, QCheckBox] = {}
        start = set(selected) if selected is not None else set(DEFAULT_METRICS)
        self._build_ui(start)

    def _build_ui(self, start: Set[str]) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        intro = QLabel(
            "Choose the quantities to measure for each object. Only the ticked "
            "metrics are computed — fewer metrics run faster. The analysis is not "
            "re-run; measurements come from the existing segmentation.")
        intro.setWordWrap(True)
        intro.setStyleSheet(scale_qss(f"color:{Settings.FG_SECONDARY}; font:9pt;"))
        outer.addWidget(intro)

        # Select-all / none.
        bar = QHBoxLayout()
        bar.addStretch(1)
        btn_all = QPushButton("Select all")
        btn_none = QPushButton("Select none")
        btn_all.clicked.connect(lambda: self._set_all(True))
        btn_none.clicked.connect(lambda: self._set_all(False))
        for b in (btn_all, btn_none):
            b.setObjectName("pipelineToolBtn")
            bar.addWidget(b)
        outer.addLayout(bar)

        host = QWidget()
        col = QVBoxLayout(host)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(10)
        for group, items in METRIC_GROUPS:
            frame = QFrame()
            frame.setObjectName("metricGroup")
            frame.setStyleSheet(scale_qss(
                f"#metricGroup{{border:1px solid {Settings.BORDER_COLOR};"
                f"border-radius:6px;background:{Settings.BG_SECONDARY};}}"))
            gl = QVBoxLayout(frame)
            gl.setContentsMargins(10, 8, 10, 8)
            gl.setSpacing(4)
            head = QLabel(group)
            head.setStyleSheet(scale_qss(f"color:{Settings.ACCENT_GREEN}; font:bold 9.5pt;"))
            gl.addWidget(head)
            grid = QGridLayout()
            grid.setHorizontalSpacing(16)
            grid.setVerticalSpacing(2)
            for i, (key, label) in enumerate(items):
                cb = QCheckBox(label)
                cb.setChecked(key in start)
                self._boxes[key] = cb
                grid.addWidget(cb, i // 2, i % 2)
            gl.addLayout(grid)
            col.addWidget(frame)
        col.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(host)
        outer.addWidget(scroll, stretch=1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _set_all(self, on: bool) -> None:
        for cb in self._boxes.values():
            cb.setChecked(on)

    def selected(self) -> Set[str]:
        return {k for k, cb in self._boxes.items() if cb.isChecked()}
