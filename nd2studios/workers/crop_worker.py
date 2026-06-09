"""CropWorker — build a frame-subset :class:`MaterializedDataset` off-thread.

V1.43. When the user selects frames on a tile strip and chooses "Crop data to
selected frames", the selected indices on one axis (T / M / Z) are used to build
a *new* in-RAM dataset via :meth:`MaterializedDataset.subset`. A T selection
keeps all M and Z (the hierarchy rule); an M or Z selection constrains only that
axis. The source dataset is never mutated, so the original file/data is
untouched and a "revert to full" simply restores the original reference.

The copy can be multi-GB, so it runs on a background thread with frame-accurate
progress (the GUI stays responsive, per the V1.41 smoothness contract).
"""
from __future__ import annotations

from typing import Optional, Sequence

from nd2studios.backend.materialized_dataset import MaterializedDataset
from nd2studios.workers.base_worker import BaseWorker


class CropWorker(BaseWorker):
    """Subset a dataset to selected indices on one axis, off the GUI thread."""

    def __init__(
        self,
        volume: MaterializedDataset,
        axis: str,
        indices: Sequence[int],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._volume = volume
        self._axis = axis
        self._indices = sorted({int(i) for i in indices})

    def run_task(self) -> MaterializedDataset:
        self.set_status(
            f"Cropping to {len(self._indices)} selected {self._axis.upper()} "
            "frame(s)…"
        )
        kwargs = {self._axis: self._indices}
        # subset reports 0–100 per channel; pass it straight through.
        cropped = self._volume.subset(progress_cb=self.set_progress, **kwargs)
        self.set_status("Crop complete.")
        return cropped
