"""QThread worker for DVC/DIC methods.

Thin Qt wrapper: it iterates nothing itself and owns no parallelism. The
real core-saturation happens *inside* the engine (process-pool fan-out
over subsets, Phases 2-5). This worker simply runs one reference/deformed
pair off the GUI thread and relays progress / status / result / error.

Emits ``finished(DVCResult)`` on success, ``error(str)`` on failure.
Cancellation via ``cancel()`` — the engine checks ``cancelled_cb``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from nd2studios.core.dvc_registry import DVCMethod, DVCResult
from nd2studios.workers.base_worker import BaseWorker


class DVCWorker(BaseWorker):
    def __init__(
        self,
        method: DVCMethod,
        ref_vol: np.ndarray,
        def_vol: np.ndarray,
        voxel_size_um: Tuple[float, ...],
        params: Dict[str, Any],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.method = method
        self.ref_vol = ref_vol
        self.def_vol = def_vol
        self.voxel_size_um = voxel_size_um
        self.params = params

    def run_task(self) -> DVCResult:
        self.set_status("Correlating reference vs deformed…")
        return self.method.run(
            self.ref_vol,
            self.def_vol,
            self.voxel_size_um,
            self.params,
            progress_cb=self.set_progress,
            cancelled_cb=lambda: self.cancelled,
        )
