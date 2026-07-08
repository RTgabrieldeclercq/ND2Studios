"""
Background worker for the V1.54 stitch pipeline.

Wraps :func:`nd2studios.backend.stitch.run_stitch` so the GUI shows progress and
status while the regime-aware pipeline (registration / coordinate placement,
blending, illumination, pyramidal OME-TIFF, QC) runs off the GUI thread.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from nd2studios.backend.stitch import StitchConfig, StitchResult, run_stitch
from nd2studios.workers.base_worker import BaseWorker


@dataclass
class StitchRequest:
    volume: object
    stage_xy_um: List[Tuple[float, float]]
    m_indices: List[int]
    channel_indices: List[int]
    config: StitchConfig = field(default_factory=StitchConfig)
    filepath: str = ""
    meta: Optional[Dict] = None


class StitchWorker(BaseWorker):
    """Run a stitch job in the background; returns the output path."""

    def __init__(self, request: StitchRequest, parent=None):
        super().__init__(parent)
        self.request = request
        self.result: Optional[StitchResult] = None

    def run_task(self) -> str:
        req = self.request
        self.set_status("Stitching tiles…")
        self.result = run_stitch(
            volume=req.volume,
            stage_xy_um=req.stage_xy_um,
            m_indices=req.m_indices,
            channel_indices=req.channel_indices,
            config=req.config,
            out_path=req.filepath,
            meta=req.meta,
            progress_cb=self.set_progress,
        )
        return self.result.out_path
