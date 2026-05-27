"""
ReconstructDialog (V1.28 + V1.31 block UI) — multi-file import with
explicit chain axis and drag-reorderable assembly blocks.

The user picks N ND2 or TIFF files, sees each file's
``(T, M, Z, C, H, W)`` in a table, and chooses which axis to chain
them along (``T`` / ``M`` / ``Z`` / ``C``). Below the table, three
:class:`AxisBlocksWidget` rows show T, M, Z as colored blocks (one
per source slice). The chain-axis row is drag-reorderable so the
user can decide which file's slice ends up at which output index;
the other two rows are display-only context.

On accept the dialog exposes :meth:`filepaths`, :meth:`chain_axis`,
and :meth:`chain_mapping` so the caller (Import page) can spin up a
``LoadWorker`` with the right multi-file parameters.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QListView, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QStyle, QStyledItemDelegate, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)


CHAIN_AXES = ("T", "M", "Z", "C")

PATTERNS = (
    "Sequential (ascending)",   # file 0 → file 1 → … → file N (default)
    "Sequential (descending)",  # file N → … → file 1 → file 0
    "Interleaved",              # frame 0 from each file, frame 1 from each, …
    "Custom",                   # manual drag on chain-axis blocks
)


class _DraggableFileTable(QTableWidget):
    """QTableWidget with drag-and-drop row reordering.

    Rows can be dragged to new positions within the table.  A
    ``rows_reordered`` signal is emitted after each successful drop so
    the dialog can sync its internal ``_entries`` list.
    """

    rows_reordered = Signal()

    def __init__(self, rows: int, cols: int, parent=None) -> None:
        super().__init__(rows, cols, parent)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.MoveAction)

    def dropEvent(self, event) -> None:  # noqa: D401
        if event.source() is not self:
            super().dropEvent(event)
            return
        event.accept()

        drop_pos = event.position().toPoint()
        target_row = self.rowAt(drop_pos.y())
        if target_row < 0:
            target_row = self.rowCount()

        selected_rows = sorted({idx.row() for idx in self.selectedIndexes()})
        if not selected_rows:
            return

        # Pull selected rows out of the table (reversed so indices stay valid).
        saved: List[List[QTableWidgetItem | None]] = []
        for r in reversed(selected_rows):
            row_items = [self.takeItem(r, c) for c in range(self.columnCount())]
            saved.insert(0, row_items)
            self.removeRow(r)
            if r < target_row:
                target_row -= 1

        target_row = max(0, min(target_row, self.rowCount()))

        # Re-insert at target position.
        for i, row_items in enumerate(saved):
            insert_at = target_row + i
            self.insertRow(insert_at)
            for c, item in enumerate(row_items):
                if item is not None:
                    self.setItem(insert_at, c, item)

        self.clearSelection()
        for i in range(len(saved)):
            self.selectRow(target_row + i)

        self.rows_reordered.emit()


# Filename-pattern heuristics: regex → suggested axis. First pattern that
# matches at least two filenames wins. Order matters — more specific
# patterns first.
_AXIS_PATTERNS: List[Tuple[str, str]] = [
    (r"count\d+", "T"),
    (r"time\d+", "T"),
    (r"t\d+(?![a-z])", "T"),       # 't' not followed by another letter
    (r"zstack\d+", "Z"),
    (r"z\d+(?![a-z])", "Z"),
    (r"pos\d+", "M"),
    (r"m\d+(?![a-z])", "M"),
    (r"p\d+(?![a-z])", "M"),
    (r"channel\w+", "C"),
    (r"ch\d+", "C"),
    (r"c\d+(?![a-z])", "C"),
]


def _suggest_chain_axis(filenames: List[str]) -> Optional[str]:
    """Pick a chain axis from filename patterns; None if no clear winner."""
    if len(filenames) < 2:
        return None
    bases = [os.path.basename(p).lower() for p in filenames]
    for pattern, axis in _AXIS_PATTERNS:
        hits = sum(1 for b in bases if re.search(pattern, b))
        if hits >= 2:
            return axis
    return None


def _format_pattern_for_display(filenames: List[str]) -> str:
    """Return a short description of the matched pattern (for the UI)."""
    bases = [os.path.basename(p).lower() for p in filenames]
    for pattern, axis in _AXIS_PATTERNS:
        hits = sum(1 for b in bases if re.search(pattern, b))
        if hits >= 2:
            return f'"{pattern}" → {axis}'
    return "no shared pattern"


def file_color_palette(n_files: int) -> Dict[int, QColor]:
    """Distinct color per source file, evenly distributed in HSV hue.

    Saturation and value are tuned to read well on the Dracula dark
    theme — bright enough that block text in black is legible, dim
    enough that the brightest channel preview can still overlay.
    """
    if n_files <= 0:
        return {}
    out: Dict[int, QColor] = {}
    for i in range(n_files):
        h = int(360 * i / n_files)
        out[i] = QColor.fromHsv(h, 180, 220)
    return out


class _BlockDelegate(QStyledItemDelegate):
    """Paint each item with its per-row background brush.

    The app-wide Dracula stylesheet rules
    ``QListWidget::item { background-color: … }`` take precedence over
    ``QListWidgetItem.setBackground()`` brushes, which means our
    per-file colors are otherwise invisible. Painting directly in a
    delegate bypasses the QSS and gives us full control over the
    block visuals (fill, border, selection outline, text).
    """

    def paint(self, painter: QPainter, option, index) -> None:  # noqa: D401
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, False)

        rect: QRect = option.rect

        # Fill with the per-item background brush.
        bg = index.data(Qt.BackgroundRole)
        if isinstance(bg, QBrush):
            painter.fillRect(rect, bg)
        elif isinstance(bg, QColor):
            painter.fillRect(rect, bg)
        else:
            painter.fillRect(rect, QColor("#3a3a3a"))

        # Selection / hover overlay so the user has drag feedback even
        # though we own the painting.
        state = option.state
        if state & QStyle.State_Selected:
            painter.fillRect(rect, QColor(255, 255, 255, 60))
            border = QPen(QColor("#bd93f9"), 2)
        elif state & QStyle.State_MouseOver:
            painter.fillRect(rect, QColor(255, 255, 255, 20))
            border = QPen(QColor("#888"), 1)
        else:
            border = QPen(QColor("#222"), 1)
        painter.setPen(border)
        painter.drawRect(rect.adjusted(0, 0, -1, -1))

        # Centered text in the configured foreground color.
        fg = index.data(Qt.ForegroundRole)
        if isinstance(fg, QBrush):
            text_color = fg.color()
        elif isinstance(fg, QColor):
            text_color = fg
        else:
            text_color = QColor("white")
        painter.setPen(text_color)
        painter.drawText(rect, Qt.AlignCenter,
                          str(index.data(Qt.DisplayRole) or ""))

        painter.restore()


class AxisBlocksWidget(QListWidget):
    """A horizontal strip of colored blocks representing one axis.

    Each block carries a ``(file_idx, local_idx)`` payload in
    ``Qt.UserRole`` and a tooltip naming the source file + frame.

    When constructed with ``draggable=True`` the strip enables Qt's
    built-in ``InternalMove`` drag-and-drop so the user can reorder
    blocks; otherwise the strip is read-only and only used for visual
    context.

    Painting is delegated to :class:`_BlockDelegate` so per-item colors
    are not overridden by the application stylesheet.
    """

    BLOCK_SIZE = QSize(78, 38)

    reordered = Signal()

    def __init__(self, axis_label: str, draggable: bool,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.axis_label = axis_label
        self._draggable = draggable

        self.setFlow(QListView.LeftToRight)
        self.setWrapping(True)
        self.setResizeMode(QListView.Adjust)
        self.setSpacing(3)
        self.setUniformItemSizes(True)
        self.setMaximumHeight(160)  # enough for ~3 wrapped rows of blocks
        self.setMovement(QListView.Snap)
        self.setItemDelegate(_BlockDelegate(self))
        # Local style override: clear the QSS item rules so the
        # delegate sees raw option.rects and selection highlights stay
        # cleanly under our control.
        self.setStyleSheet(
            "QListWidget::item { background: transparent; padding: 0px; "
            "border: none; } "
            "QListWidget::item:selected, QListWidget::item:hover { "
            "background: transparent; }"
        )

        if draggable:
            self.setSelectionMode(QAbstractItemView.ExtendedSelection)
            self.setDragDropMode(QAbstractItemView.InternalMove)
            self.setDragEnabled(True)
            self.setAcceptDrops(True)
            self.setDropIndicatorShown(True)
            # Fires after the user drops; we propagate so the dialog
            # can refresh its preview.
            self.model().rowsMoved.connect(
                lambda *_: self.reordered.emit()
            )
        else:
            self.setSelectionMode(QAbstractItemView.NoSelection)
            self.setDragDropMode(QAbstractItemView.NoDragDrop)
            self.setFocusPolicy(Qt.NoFocus)

    def set_blocks(self,
                    blocks: List[Dict[str, Any]],
                    file_colors: Dict[int, QColor],
                    color_by_file: bool) -> None:
        """Rebuild the strip from a list of block descriptors.

        Each block descriptor must contain:
            file_idx, local_idx, filename, frame_label

        ``color_by_file`` colors each block by its source file (used on
        the chain-axis row, where blocks come from different files).
        On non-chain rows every slice exists in every file, so blocks
        get a neutral gray.
        """
        self.clear()
        for i, b in enumerate(blocks):
            item = QListWidgetItem(f"{self.axis_label}{i}")
            if color_by_file:
                color = file_colors.get(int(b["file_idx"]),
                                         QColor("#888888"))
            else:
                color = QColor("#3a3a3a")
            item.setBackground(QBrush(color))
            item.setForeground(QBrush(
                QColor("black") if color_by_file else QColor("#cccccc")
            ))
            item.setToolTip(
                f"{b['filename']}\n"
                f"Source: {b['frame_label']}\n"
                f"Output {self.axis_label}={i}"
            )
            item.setData(Qt.UserRole, (int(b["file_idx"]),
                                         int(b["local_idx"])))
            item.setTextAlignment(Qt.AlignCenter)
            item.setSizeHint(self.BLOCK_SIZE)
            self.addItem(item)

    def mapping(self) -> List[Tuple[int, int]]:
        """Read back the current per-block mapping in display order."""
        out: List[Tuple[int, int]] = []
        for r in range(self.count()):
            data = self.item(r).data(Qt.UserRole)
            if isinstance(data, tuple) and len(data) == 2:
                out.append((int(data[0]), int(data[1])))
            else:
                out.append((0, r))
        return out


def _probe_file(filepath: str) -> Dict[str, Any]:
    """Cheap dim/dtype probe for one file. Returns metadata or {} on fail."""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".nd2":
            from nd2studios.backend.nd2_loader import read_nd2_metadata
            meta = read_nd2_metadata(filepath)
            return {
                "filepath": filepath,
                "n_timepoints": int(meta.n_timepoints),
                "n_multipoints": int(meta.n_multipoints),
                "n_zslices": int(meta.n_zslices),
                "n_channels": int(meta.n_channels),
                "height": int(meta.height),
                "width": int(meta.width),
                "dtype": str(meta.dtype),
                "channel_names": list(meta.channel_names),
                "format": "ND2",
            }
        if ext in (".tif", ".tiff"):
            from nd2studios.backend.tiff_loader import read_tiff_meta_fast
            m = read_tiff_meta_fast(filepath)
            m["format"] = "TIFF"
            return m
    except Exception as exc:  # noqa: BLE001 — surface in the table
        return {"filepath": filepath, "error": str(exc),
                "format": ext.lstrip(".").upper() or "?"}
    return {}


class ReconstructDialog(QDialog):
    """Modal dialog for V1.28 multi-file reconstruction."""

    COL_INDEX, COL_NAME, COL_T, COL_M, COL_Z, COL_C, COL_HXW, COL_DTYPE = range(8)
    HEADERS = ["#", "File", "T", "M", "Z", "C", "H×W", "dtype"]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Reconstruct from multiple files")
        self.setMinimumSize(900, 720)

        self._entries: List[Dict[str, Any]] = []   # one dict per file row
        self._file_type: Optional[str] = None      # "ND2" or "TIFF" once locked
        # AxisBlocksWidget instances keyed by axis label; populated in
        # _build_ui and refilled by _rebuild_blocks() whenever the file
        # set or chain axis changes.
        self._block_rows: Dict[str, AxisBlocksWidget] = {}

        self._build_ui()
        self._refresh_preview()

    # ── UI ──
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        # Toolbar: Add / Remove / Move ↑ ↓ + file-type lock indicator.
        toolbar = QHBoxLayout()
        self.btn_add = QPushButton("Add files…")
        self.btn_add.clicked.connect(self._on_add)
        toolbar.addWidget(self.btn_add)
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.clicked.connect(self._on_remove)
        toolbar.addWidget(self.btn_remove)
        self.btn_up = QPushButton("Move ↑")
        self.btn_up.clicked.connect(lambda: self._move(-1))
        toolbar.addWidget(self.btn_up)
        self.btn_down = QPushButton("Move ↓")
        self.btn_down.clicked.connect(lambda: self._move(+1))
        toolbar.addWidget(self.btn_down)
        toolbar.addStretch(1)
        self.lbl_filetype = QLabel("Format: —")
        toolbar.addWidget(self.lbl_filetype)
        outer.addLayout(toolbar)

        # File table (drag-reorderable rows).
        self.table = _DraggableFileTable(0, len(self.HEADERS), self)
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setToolTip("Drag rows to reorder files")
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_NAME, QHeaderView.Stretch)
        self.table.rows_reordered.connect(self._on_table_rows_moved)
        outer.addWidget(self.table, stretch=1)

        # Axis controls + preview.
        ctrl_group = QGroupBox("Reconstruction")
        form = QFormLayout(ctrl_group)
        self.combo_axis = QComboBox()
        self.combo_axis.addItems(list(CHAIN_AXES))
        self.combo_axis.setCurrentText("Z")
        self.combo_axis.currentTextChanged.connect(self._on_axis_changed)
        form.addRow("Chain along axis", self.combo_axis)
        self.combo_pattern = QComboBox()
        self.combo_pattern.addItems(list(PATTERNS))
        self.combo_pattern.setCurrentText("Sequential (ascending)")
        self.combo_pattern.setToolTip(
            "How to order frames from the chain axis across files.\n"
            "Sequential: all frames from each file in file-list order.\n"
            "Interleaved: frame 0 from all files, then frame 1, etc.\n"
            "Custom: drag blocks below to set a manual order."
        )
        self.combo_pattern.currentTextChanged.connect(self._on_pattern_changed)
        form.addRow("Frame order pattern", self.combo_pattern)
        self.lbl_suggested = QLabel("Add files to see suggestion.")
        form.addRow("Auto-detected", self.lbl_suggested)
        self.lbl_combined = QLabel("—")
        self.lbl_combined.setWordWrap(True)
        form.addRow("Combined", self.lbl_combined)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        form.addRow("Status", self.lbl_status)
        outer.addWidget(ctrl_group)

        # Per-axis block rows. Each axis gets its own AxisBlocksWidget;
        # drag is enabled on the chain-axis row only when pattern = Custom.
        blocks_group = QGroupBox(
            "Assembly blocks  —  hover blocks for source info · "
            "set pattern to \"Custom\" to drag-reorder slices"
        )
        bl = QVBoxLayout(blocks_group)
        bl.setSpacing(4)
        for axis in ("T", "M", "Z"):
            row_wrap = QHBoxLayout()
            row_wrap.setContentsMargins(0, 0, 0, 0)
            label = QLabel(f"<b>{axis}</b>")
            label.setMinimumWidth(24)
            label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            row_wrap.addWidget(label)
            row = AxisBlocksWidget(axis_label=axis, draggable=False)
            row.reordered.connect(self._on_blocks_reordered)
            row_wrap.addWidget(row, stretch=1)
            bl.addLayout(row_wrap)
            self._block_rows[axis] = row
        outer.addWidget(blocks_group)

        # OK / Cancel.
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Cancel | QDialogButtonBox.Ok, parent=self,
        )
        self.buttons.button(QDialogButtonBox.Ok).setText("Load")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        outer.addWidget(self.buttons)

    # ── Actions ──
    def _on_add(self) -> None:
        # Filter the dialog by the locked file type, if any.
        if self._file_type == "ND2":
            fltr = "ND2 files (*.nd2)"
        elif self._file_type == "TIFF":
            fltr = "TIFF files (*.tif *.tiff)"
        else:
            fltr = "Microscopy files (*.nd2 *.tif *.tiff)"

        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add files to reconstruct", "", fltr,
        )
        if not paths:
            return

        # Lock file type from the first batch added.
        new_type = self._classify_type(paths)
        if new_type is None:
            QMessageBox.warning(
                self, "Mixed formats",
                "Pick all ND2 or all TIFF files — not both.",
            )
            return
        if self._file_type is None:
            self._file_type = new_type
            self.lbl_filetype.setText(f"Format: {new_type}")
        elif self._file_type != new_type:
            QMessageBox.warning(
                self, "Format mismatch",
                f"Already adding {self._file_type} files; "
                f"can't mix {new_type} into the same reconstruction.",
            )
            return

        # Probe each file and append to the entry list.
        for p in paths:
            if any(e["filepath"] == p for e in self._entries):
                continue
            self._entries.append(_probe_file(p))

        # Sort newly added rows by basename for predictability; the user
        # can reorder afterwards.
        self._entries.sort(key=lambda e: os.path.basename(e.get("filepath", "")))

        self._rebuild_table()
        self._suggest_axis_from_filenames()
        self._rebuild_blocks()
        self._refresh_preview()

    def _on_remove(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            if 0 <= r < len(self._entries):
                self._entries.pop(r)
        if not self._entries:
            self._file_type = None
            self.lbl_filetype.setText("Format: —")
        self._rebuild_table()
        self._rebuild_blocks()
        self._refresh_preview()

    def _move(self, delta: int) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows:
            return
        if delta < 0:
            iterable = rows
        else:
            iterable = list(reversed(rows))
        for r in iterable:
            nr = r + delta
            if 0 <= nr < len(self._entries):
                self._entries[r], self._entries[nr] = self._entries[nr], self._entries[r]
        self._rebuild_table()
        # Re-select moved rows so the user can chain Move-↑/↓ clicks.
        self.table.clearSelection()
        for r in rows:
            nr = r + delta
            if 0 <= nr < len(self._entries):
                self.table.selectRow(nr)
        self._rebuild_blocks()
        self._refresh_preview()

    # ── Helpers ──
    @staticmethod
    def _classify_type(paths: List[str]) -> Optional[str]:
        exts = {os.path.splitext(p)[1].lower() for p in paths}
        if exts == {".nd2"}:
            return "ND2"
        if exts.issubset({".tif", ".tiff"}):
            return "TIFF"
        return None

    def _rebuild_table(self) -> None:
        self.table.setRowCount(len(self._entries))
        for r, e in enumerate(self._entries):
            name = os.path.basename(e.get("filepath", ""))
            if "error" in e:
                vals = [str(r + 1), name + " (error)", "?", "?", "?", "?", "?", e["error"]]
            else:
                vals = [
                    str(r + 1),
                    name,
                    str(e.get("n_timepoints", "?")),
                    str(e.get("n_multipoints", "?")),
                    str(e.get("n_zslices", "?")),
                    str(e.get("n_channels", "?")),
                    f'{e.get("height", "?")}×{e.get("width", "?")}',
                    str(e.get("dtype", "?")),
                ]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if c == self.COL_INDEX:
                    item.setTextAlignment(Qt.AlignCenter)
                    # Store filepath so we can recover order after a drag.
                    item.setData(Qt.UserRole, e.get("filepath", ""))
                self.table.setItem(r, c, item)

    def _suggest_axis_from_filenames(self) -> None:
        paths = [e["filepath"] for e in self._entries if "filepath" in e]
        suggested = _suggest_chain_axis(paths)
        display = _format_pattern_for_display(paths) if paths else "—"
        self.lbl_suggested.setText(display)
        if suggested is not None:
            # Only preselect; never override an explicit user choice.
            self.combo_axis.blockSignals(True)
            self.combo_axis.setCurrentText(suggested)
            self.combo_axis.blockSignals(False)

    # ── Block rows ──────────────────────────────────────────────────
    def _on_axis_changed(self, _axis: str) -> None:
        """User picked a different chain axis from the combo box."""
        self._rebuild_blocks()
        self._refresh_preview()

    def _on_blocks_reordered(self) -> None:
        """A drop happened on the chain-axis row — ensure Custom is active."""
        if self.combo_pattern.currentText() != "Custom":
            self.combo_pattern.blockSignals(True)
            self.combo_pattern.setCurrentText("Custom")
            self.combo_pattern.blockSignals(False)
        self._refresh_preview()

    def _on_table_rows_moved(self) -> None:
        """File rows were drag-reordered in the table — sync _entries."""
        filepath_order: List[str] = []
        for r in range(self.table.rowCount()):
            item = self.table.item(r, self.COL_INDEX)
            if item is not None:
                fp = item.data(Qt.UserRole)
                if fp:
                    filepath_order.append(fp)

        entry_map = {e.get("filepath", ""): e for e in self._entries}
        self._entries = [entry_map[fp] for fp in filepath_order if fp in entry_map]

        # Renumber the # column to reflect new order.
        for r in range(self.table.rowCount()):
            num_item = self.table.item(r, self.COL_INDEX)
            if num_item is not None:
                num_item.setText(str(r + 1))

        self._rebuild_blocks()
        self._refresh_preview()

    def _on_pattern_changed(self, _pattern: str) -> None:
        """User picked a different frame-order pattern."""
        self._rebuild_blocks()
        self._refresh_preview()

    def _compute_pattern_mapping(
        self, pattern: str, entries: List[Dict[str, Any]]
    ) -> List[Tuple[int, int]]:
        """Return a chain mapping for the given pattern and current file list."""
        chain = self.combo_axis.currentText()
        attr_map = {
            "T": "n_timepoints", "M": "n_multipoints",
            "Z": "n_zslices",    "C": "n_channels",
        }
        attr = attr_map[chain]
        sizes = [int(e.get(attr, 1)) for e in entries]

        if pattern == "Sequential (ascending)":
            return [
                (fi, li)
                for fi, sz in enumerate(sizes)
                for li in range(sz)
            ]
        if pattern == "Sequential (descending)":
            return [
                (fi, li)
                for fi in range(len(sizes) - 1, -1, -1)
                for li in range(sizes[fi])
            ]
        if pattern == "Interleaved":
            max_sz = max(sizes) if sizes else 0
            return [
                (fi, li)
                for li in range(max_sz)
                for fi, sz in enumerate(sizes)
                if li < sz
            ]
        # Custom — no auto-mapping; caller preserves current block order.
        return []

    def _rebuild_blocks(self) -> None:
        """Refill the T / M / Z block rows from the current file list."""
        entries = [e for e in self._entries
                    if "error" not in e and "filepath" in e]
        chain = self.combo_axis.currentText()
        pattern = self.combo_pattern.currentText()
        is_custom = (pattern == "Custom")
        attr = {"T": "n_timepoints", "M": "n_multipoints",
                "Z": "n_zslices", "C": "n_channels"}

        file_colors = file_color_palette(len(entries))

        for axis, row in self._block_rows.items():
            is_chain = (axis == chain)
            # Drag is only available on the chain-axis row AND only when
            # the user has chosen the Custom pattern.
            drag_ok = is_chain and is_custom
            row.setSelectionMode(
                QAbstractItemView.ExtendedSelection if drag_ok
                else QAbstractItemView.NoSelection
            )
            row.setDragDropMode(
                QAbstractItemView.InternalMove if drag_ok
                else QAbstractItemView.NoDragDrop
            )
            row.setDragEnabled(drag_ok)
            row.setAcceptDrops(drag_ok)
            row.setDropIndicatorShown(drag_ok)
            row.setFocusPolicy(Qt.StrongFocus if drag_ok else Qt.NoFocus)
            row._draggable = drag_ok

            blocks: List[Dict[str, Any]] = []
            if not entries:
                row.set_blocks([], {}, color_by_file=is_chain)
                continue

            if is_chain:
                # Build the natural (ascending) block lookup keyed by
                # (file_idx, local_idx) so any pattern can reorder them.
                natural: List[Dict[str, Any]] = []
                block_lookup: Dict[Tuple[int, int], Dict[str, Any]] = {}
                for fi, e in enumerate(entries):
                    n_local = int(e.get(attr[axis], 1))
                    fname = os.path.basename(e.get("filepath", ""))
                    for li in range(n_local):
                        b = {
                            "file_idx": fi,
                            "local_idx": li,
                            "filename": fname,
                            "frame_label": f"{axis}={li}",
                        }
                        natural.append(b)
                        block_lookup[(fi, li)] = b

                if is_custom:
                    # Preserve the current visual order if it still covers
                    # all the same (file_idx, local_idx) pairs; otherwise
                    # fall back to ascending so new/removed files are shown.
                    current_mapping = row.mapping()
                    current_keys = set(current_mapping)
                    expected_keys = set(block_lookup.keys())
                    if current_keys == expected_keys:
                        blocks = [block_lookup[k] for k in current_mapping]
                    else:
                        blocks = natural
                else:
                    mapping = self._compute_pattern_mapping(pattern, entries)
                    blocks = [block_lookup[k] for k in mapping if k in block_lookup]
            else:
                # Display-only: one block per slice on the (shared) axis.
                e0 = entries[0]
                n_local = int(e0.get(attr[axis], 1))
                fname = "all files (shared axis)"
                for li in range(n_local):
                    blocks.append({
                        "file_idx": 0,
                        "local_idx": li,
                        "filename": fname,
                        "frame_label": f"{axis}={li}",
                    })

            row.set_blocks(blocks, file_colors, color_by_file=is_chain)

    def _refresh_preview(self) -> None:
        entries = [e for e in self._entries if "error" not in e and "filepath" in e]
        n = len(entries)
        axis = self.combo_axis.currentText()
        ok_btn = self.buttons.button(QDialogButtonBox.Ok)

        if n < 2:
            self.lbl_combined.setText("—")
            self.lbl_status.setText("Add at least two files.")
            ok_btn.setEnabled(False)
            ok_btn.setToolTip("Reconstruction needs at least 2 files.")
            return

        mismatch = self._mismatch_summary(entries, axis)
        attr_map = {
            "T": "n_timepoints", "M": "n_multipoints",
            "Z": "n_zslices", "C": "n_channels",
        }
        attr = attr_map[axis]
        chain_total = sum(int(e.get(attr, 0)) for e in entries)
        f0 = entries[0]
        # Show the combined shape using the locked non-chain axes.
        combined_T = chain_total if axis == "T" else int(f0.get("n_timepoints", 1))
        combined_M = chain_total if axis == "M" else int(f0.get("n_multipoints", 1))
        combined_Z = chain_total if axis == "Z" else int(f0.get("n_zslices", 1))
        combined_C = chain_total if axis == "C" else int(f0.get("n_channels", 1))
        h = int(f0.get("height", 0))
        w = int(f0.get("width", 0))
        dtype = str(f0.get("dtype", "?"))
        self.lbl_combined.setText(
            f"T={combined_T}  M={combined_M}  Z={combined_Z}  "
            f"C={combined_C}  {h}×{w}  {dtype}"
        )

        if mismatch:
            self.lbl_status.setText(
                f"Non-chain axes disagree across files: {mismatch}"
            )
            ok_btn.setEnabled(False)
            ok_btn.setToolTip(
                "All axes except the chosen chain axis must match across files."
            )
            return
        if chain_total <= 0:
            self.lbl_status.setText(
                f"Chosen chain axis ({axis}) has size 0 in the selected files."
            )
            ok_btn.setEnabled(False)
            ok_btn.setToolTip("Pick a different chain axis.")
            return

        self.lbl_status.setText("Ready to reconstruct.")
        ok_btn.setEnabled(True)
        ok_btn.setToolTip("")

    @staticmethod
    def _mismatch_summary(entries: List[Dict[str, Any]], axis: str) -> str:
        attr_map = {
            "T": "n_timepoints", "M": "n_multipoints",
            "Z": "n_zslices", "C": "n_channels",
        }
        skip = attr_map[axis]
        labels = [
            ("n_timepoints", "T"), ("n_multipoints", "M"),
            ("n_zslices", "Z"), ("n_channels", "C"),
            ("height", "Y"), ("width", "X"),
        ]
        diffs: List[str] = []
        for attr, label in labels:
            if attr == skip:
                continue
            vals = Counter(int(e.get(attr, 0)) for e in entries)
            if len(vals) > 1:
                diffs.append(f"{label}={dict(vals)}")
        return ", ".join(diffs)

    # ── Public accessors used by ImportPage on accept ──
    def filepaths(self) -> List[str]:
        return [e["filepath"] for e in self._entries
                if "filepath" in e and "error" not in e]

    def chain_axis(self) -> str:
        return self.combo_axis.currentText()

    def chain_mapping(self) -> List[Tuple[int, int]]:
        """Per-block ``(file_idx, local_idx)`` mapping in display order.

        Read from the chain-axis :class:`AxisBlocksWidget`, which the
        user may have reordered via drag-and-drop. Returns an empty
        list if the chain row hasn't been populated yet — the caller
        should fall back to the natural default in that case.
        """
        chain = self.combo_axis.currentText()
        row = self._block_rows.get(chain)
        if row is None or row.count() == 0:
            return []
        return row.mapping()
