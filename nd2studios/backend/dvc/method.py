"""
Registered DVC methods.

Phase 0 ships a single method, :class:`ALDVCMethod`, whose ``run`` calls
the (currently stubbed) :func:`nd2studios.backend.dvc.engine.run_aldvc`.
Force-imported in ``__main__.py`` so its ``@DVCMethod.register`` decorator
fires at startup and the method appears in the DVC page.

Pure backend module — no PySide6.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from nd2studios.backend.dvc.engine import run_aldvc
from nd2studios.core.dvc_registry import DVCMethod, DVCResult
from nd2studios.core.plugin_registry import ParamSpec


@DVCMethod.register
class ALDVCMethod(DVCMethod):
    name = "ALDVC"
    description = (
        "Augmented Lagrangian Digital Volume Correlation — hybrid local "
        "(IC-GN subset) + global (compatibility) DVC. 3D volumes or 2D "
        "images (DIC). [Phase 0: global-shift stub.]"
    )

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec(
                name="subset_size", label="Subset size (voxels)",
                param_type="int", default=16, min_val=4, max_val=128, step=2,
                tooltip="Edge length of each correlation subset window.",
            ),
            ParamSpec(
                name="subset_spacing", label="Subset spacing (voxels)",
                param_type="int", default=10, min_val=1, max_val=128, step=1,
                tooltip="Spacing between subset centers (the measurement grid).",
            ),
            ParamSpec(
                name="correlation", label="Correlation",
                param_type="choice", default="zncc",
                choices=["zncc", "phase"],
                tooltip="ZNCC (robust to brightness changes) or phase correlation.",
            ),
            ParamSpec(
                name="strain_type", label="Strain measure",
                param_type="choice", default="infinitesimal",
                choices=["infinitesimal", "green-lagrange"],
                tooltip="Strain tensor derived from the displacement gradient.",
            ),
            ParamSpec(
                name="admm_iterations", label="ADMM iterations",
                param_type="int", default=4, min_val=1, max_val=12, step=1,
                tooltip="Augmented-Lagrangian outer iterations.",
            ),
            ParamSpec(
                name="mu", label="ADMM penalty (mu)",
                param_type="float", default=1e-3, min_val=1e-6, max_val=1.0,
                step=1e-3,
                tooltip="Augmented-Lagrangian coupling penalty.",
            ),
            ParamSpec(
                name="use_gpu", label="Use GPU (if available)",
                param_type="bool", default=False,
                tooltip="Route the FFT seed + resampling through CuPy when present.",
            ),
        ]

    def run(
        self,
        ref_vol: np.ndarray,
        def_vol: np.ndarray,
        voxel_size_um: Tuple[float, ...],
        params: Dict[str, Any],
        progress_cb: Optional[Callable[[int], None]] = None,
        cancelled_cb: Optional[Callable[[], bool]] = None,
    ) -> DVCResult:
        return run_aldvc(
            ref_vol, def_vol, voxel_size_um, params,
            progress_cb=progress_cb, cancelled_cb=cancelled_cb,
        )
