"""
DVC page — Digital Volume / Image Correlation.

Pick a DVC method, a channel, an M position, and a reference + deformed
timepoint; Run correlates the deformed volume against the reference and
reports the displacement field.

Data source (deliberately *not* the recipe/enhancement path, which
Z-collapses): the page pulls full ``(Z, H, W)`` volumes straight from the
record's ``LazyND2Volume`` (``_raw_volume``) via ``get_frame(z_mode="none")``.
When the data has no Z (``_raw_volume`` absent or single-slice), it falls
back to the Z-collapsed ``(H, W)`` frame — i.e. **2D DIC**.

Phase 0: the result panel is a text summary. The dense displacement /
strain field viewer (quiver + heatmap slices) lands in Phase 5.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from nd2studios.core.dvc_registry import DVCMethod, DVCResult
from nd2studios.core.experiment_manager import ND2StudiosRecord
from nd2studios.widgets.common import ParamEditor
from nd2studios.workers.dvc_worker import DVCWorker


class DVCPage(QWidget):
    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self._worker: Optional[DVCWorker] = None
        self._result: Optional[DVCResult] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel("Digital Volume Correlation")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        root.addWidget(title)

        body = QHBoxLayout()
        body.setSpacing(16)
        root.addLayout(body, stretch=1)

        # ── Left: configuration ──
        cfg = QVBoxLayout()
        cfg.setSpacing(12)
        body.addLayout(cfg, stretch=0)

        method_box = QGroupBox("Method")
        method_form = QFormLayout(method_box)
        self.method_combo = QComboBox()
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        method_form.addRow("Method", self.method_combo)
        self.method_desc = QLabel("")
        self.method_desc.setWordWrap(True)
        self.method_desc.setStyleSheet("color: #b0b0b0;")
        method_form.addRow(self.method_desc)
        cfg.addWidget(method_box)

        data_box = QGroupBox("Reference → Deformed")
        data_form = QFormLayout(data_box)
        self.channel_combo = QComboBox()
        data_form.addRow("Channel", self.channel_combo)
        self.m_spin = QSpinBox()
        self.m_spin.setMinimum(0)
        data_form.addRow("M position", self.m_spin)
        self.tref_spin = QSpinBox()
        self.tref_spin.setMinimum(0)
        data_form.addRow("Reference T", self.tref_spin)
        self.tdef_spin = QSpinBox()
        self.tdef_spin.setMinimum(0)
        data_form.addRow("Deformed T", self.tdef_spin)
        self.shape_label = QLabel("—")
        self.shape_label.setStyleSheet("color: #b0b0b0;")
        data_form.addRow("Volume", self.shape_label)
        cfg.addWidget(data_box)

        param_box = QGroupBox("Parameters")
        param_layout = QVBoxLayout(param_box)
        self.param_editor = ParamEditor()
        param_layout.addWidget(self.param_editor)
        cfg.addWidget(param_box)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Run DVC")
        self.btn_run.clicked.connect(self._on_run)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_cancel)
        cfg.addLayout(btn_row)
        cfg.addStretch(1)

        # ── Right: progress + results ──
        right = QVBoxLayout()
        right.setSpacing(8)
        body.addLayout(right, stretch=1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        right.addWidget(self.progress_bar)
        self.status_label = QLabel("Load a file (Import), then run DVC.")
        self.status_label.setStyleSheet("color: #b0b0b0;")
        right.addWidget(self.status_label)

        results_box = QGroupBox("Result")
        results_layout = QVBoxLayout(results_box)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        right.addWidget(results_box, stretch=1)

        self._populate_methods()

    # ── lifecycle ──
    def on_activated(self) -> None:
        """Called by MainWindow when this page is shown."""
        self._refresh_from_record()

    def _active_exp(self) -> Optional[ND2StudiosRecord]:
        if self.main_window is None:
            return None
        return self.main_window.exp_manager.active

    # ── population ──
    def _populate_methods(self) -> None:
        self.method_combo.blockSignals(True)
        self.method_combo.clear()
        for cls in DVCMethod.get_methods():
            self.method_combo.addItem(cls.name)
        self.method_combo.blockSignals(False)
        self._on_method_changed(self.method_combo.currentText())

    def _on_method_changed(self, name: str) -> None:
        cls = DVCMethod.get_method(name)
        if cls is None:
            self.param_editor.set_params([])
            self.method_desc.setText("")
            return
        method = cls()
        self.method_desc.setText(method.description)
        self.param_editor.set_params(method.get_params())

    def _refresh_from_record(self) -> None:
        exp = self._active_exp()
        if exp is None:
            self.status_label.setText("No active experiment.")
            self.shape_label.setText("—")
            return

        vol = getattr(exp, "_raw_volume", None)
        channels = getattr(exp, "_raw_channels", None) or {}

        # Channel names.
        if vol is not None and getattr(vol, "channel_names", None):
            names = list(vol.channel_names)
        else:
            names = list(channels.keys())

        prev = self.channel_combo.currentText()
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItems(names)
        if prev in names:
            self.channel_combo.setCurrentText(prev)
        self.channel_combo.blockSignals(False)

        # M / T ranges.
        if vol is not None:
            n_m = int(getattr(vol, "n_multipoints", 1) or 1)
            n_t = int(getattr(vol, "n_timepoints", 1) or 1)
        elif channels:
            first = next(iter(channels.values()))
            n_m = 1
            n_t = int(np.asarray(first).shape[0]) if np.asarray(first).ndim >= 1 else 1
        else:
            n_m, n_t = 1, 1

        self.m_spin.setMaximum(max(0, n_m - 1))
        self.tref_spin.setMaximum(max(0, n_t - 1))
        self.tdef_spin.setMaximum(max(0, n_t - 1))
        if self.tdef_spin.value() == self.tref_spin.value() and n_t > 1:
            self.tdef_spin.setValue(min(self.tref_spin.value() + 1, n_t - 1))

        if not names:
            self.status_label.setText("Import a file first (Page 1).")
        else:
            n_z = int(getattr(vol, "n_zslices", 1) or 1) if vol is not None else 1
            mode = f"3D DVC ({n_z} Z-slices)" if n_z > 1 else "2D DIC (single plane)"
            self.status_label.setText(f"Ready — {mode}.")
        self._update_shape_label()

    def _update_shape_label(self) -> None:
        exp = self._active_exp()
        if exp is None or not self.channel_combo.count():
            self.shape_label.setText("—")
            return
        try:
            ref = self._extract_volume(
                exp, self.channel_combo.currentText(),
                self.m_spin.value(), self.tref_spin.value(),
            )
            self.shape_label.setText(f"{'×'.join(str(s) for s in ref.shape)} ({ref.ndim}D)")
        except Exception as exc:  # noqa: BLE001
            self.shape_label.setText(f"(unavailable: {type(exc).__name__})")

    # ── volume extraction (record → numpy; keeps the engine Qt-free) ──
    def _extract_volume(
        self, exp: ND2StudiosRecord, channel: str, m: int, t: int
    ) -> np.ndarray:
        """Return ``(Z, H, W)`` (3D) or ``(H, W)`` (2D) for one (channel, m, t)."""
        vol = getattr(exp, "_raw_volume", None)
        if vol is not None and getattr(vol, "channel_names", None):
            names = list(vol.channel_names)
            c_idx = names.index(channel) if channel in names else 0
            n_z = int(getattr(vol, "n_zslices", 1) or 1)
            if n_z > 1:
                planes = [
                    np.asarray(vol.get_frame(c=c_idx, m=m, t=t, z=zi, z_mode="none"))
                    for zi in range(n_z)
                ]
                return np.stack(planes, axis=0).astype(np.float32)
            frame = vol.get_frame(c=c_idx, m=m, t=t, z=0, z_mode="none")
            return np.asarray(frame, dtype=np.float32)

        channels = getattr(exp, "_raw_channels", None) or {}
        arr = channels.get(channel) if channel in channels else next(iter(channels.values()))
        arr = np.asarray(arr)
        return np.asarray(arr[t] if arr.ndim >= 3 else arr, dtype=np.float32)

    def _voxel_size(self, exp: ND2StudiosRecord, dim: int) -> Tuple[float, ...]:
        pixel = float(getattr(exp, "pixel_size_um", 1.0) or 1.0)
        if dim == 3:
            vol = getattr(exp, "_raw_volume", None)
            z_step = float(getattr(vol, "z_step_um", 1.0) or 1.0) if vol else 1.0
            return (z_step, pixel, pixel)
        return (pixel, pixel)

    # ── run / cancel ──
    def _on_run(self) -> None:
        exp = self._active_exp()
        if exp is None or not self.channel_combo.count():
            QMessageBox.information(self, "No data", "Import a file first (Page 1).")
            return
        cls = DVCMethod.get_method(self.method_combo.currentText())
        if cls is None:
            return

        channel = self.channel_combo.currentText()
        m = self.m_spin.value()
        t_ref = self.tref_spin.value()
        t_def = self.tdef_spin.value()
        if t_ref == t_def:
            QMessageBox.information(
                self, "Same timepoint",
                "Reference and deformed timepoints are identical — "
                "the displacement field will be ~zero.",
            )

        try:
            ref = self._extract_volume(exp, channel, m, t_ref)
            mov = self._extract_volume(exp, channel, m, t_def)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Extraction failed", str(exc))
            return

        params = self.param_editor.get_values()
        voxel = self._voxel_size(exp, ref.ndim)

        self._worker = DVCWorker(cls(), ref, mov, voxel, params, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setValue(0)
        self.results_text.clear()
        self.status_label.setText("Running…")
        self._worker.start()

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.status_label.setText("Cancelling…")

    # ── worker signal handlers ──
    def _on_progress(self, value: int) -> None:
        self.progress_bar.setValue(int(value))
        if self.main_window is not None:
            self.main_window.set_progress(int(value))

    def _on_status(self, text: str) -> None:
        self.status_label.setText(text)
        if self.main_window is not None:
            self.main_window.set_status_text(text)

    def _on_finished(self, result: DVCResult) -> None:
        self._result = result
        self._worker = None
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_label.setText("Done.")
        self.results_text.setPlainText(self._summarize(result))

    def _on_error(self, message: str) -> None:
        self._worker = None
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.status_label.setText("Error.")
        self.results_text.setPlainText(message)

    @staticmethod
    def _summarize(r: DVCResult) -> str:
        mag = r.magnitude
        mag_um = r.magnitude_um()
        lines = [
            f"Method:           {r.method}",
            f"Dimensionality:   {r.dim}D",
            f"Grid:             {'×'.join(str(s) for s in r.grid_coords.shape[:-1])}"
            f"  ({int(np.prod(r.grid_coords.shape[:-1]))} subsets)",
            f"Voxel size (um):  {tuple(round(v, 4) for v in r.voxel_size_um)}",
            f"Converged:        {r.converged} (iters={r.iterations})",
            "",
            f"Displacement |u| (voxels):  mean={np.nanmean(mag):.4f}  "
            f"max={np.nanmax(mag):.4f}",
            f"Displacement |u| (um):      mean={np.nanmean(mag_um):.4f}  "
            f"max={np.nanmax(mag_um):.4f}",
        ]
        if r.diagnostics:
            lines.append("")
            lines.append("Diagnostics:")
            for k, v in r.diagnostics.items():
                lines.append(f"  {k}: {v}")
        if r.notes:
            lines.append("")
            lines.append(r.notes)
        return "\n".join(lines)
