"""
Registered DVC methods.

Ships :class:`ALDVCMethod`, whose ``run`` calls the full ALDVC pipeline in
:func:`nd2studios.backend.dvc.engine.run_aldvc`. Force-imported in
``__main__.py`` so its ``@DVCMethod.register`` decorator fires at startup and the
method is discoverable by the DVC node (its ``get_params`` also supplies the
node's ParamSpec list, via ``registry_adapter.param_specs_for``).

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
        "images (DIC)."
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
                name="search_radius", label="Seed search radius (voxels)",
                param_type="int", default=0, min_val=0, max_val=128, step=1,
                tooltip="Max seed displacement per axis for the FFT integer "
                        "search. 0 = auto (= subset size).",
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
                choices=["infinitesimal", "green-lagrange", "almansi", "hencky"],
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
                name="seed_levels", label="Seed pyramid levels",
                param_type="int", default=3, min_val=1, max_val=5, step=1,
                tooltip="Coarse-to-fine multigrid levels for the FFT integer seed. "
                        "Higher brackets larger displacement robustly (1 = "
                        "single-scale).",
            ),
            ParamSpec(
                name="newFFTSearch", label="Re-seed every frame",
                param_type="bool", default=False,
                tooltip="Re-run the FFT seed on every frame. Off (default) = "
                        "warm-start each frame from the previous frame's field "
                        "(ALDVC's cross-frame U0) — faster + more robust for a "
                        "smooth series.",
            ),
            ParamSpec(
                name="n_workers", label="Parallel workers (0=auto)",
                param_type="int", default=0, min_val=0, max_val=64, step=1,
                tooltip="Process-pool workers for the IC-GN sweep (the dominant "
                        "cost). 0 = auto (cores−1), 1 = serial. Ignored on tiny grids.",
            ),
            ParamSpec(
                name="use_gpu", label="Use GPU for seed (if available)",
                param_type="bool", default=False,
                tooltip="Route the FFT integer-search seed through CuPy when "
                        "present (falls back to CPU otherwise). The IC-GN sweep "
                        "runs on CPU cores (see workers).",
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
        **kwargs: Any,
    ) -> DVCResult:
        # ``kwargs`` forwards the series-level extras (``u0_seed`` / ``use_fft_seed``
        # for cross-frame warm-start) that the DVC worker passes per frame.
        return run_aldvc(
            ref_vol, def_vol, voxel_size_um, params,
            progress_cb=progress_cb, cancelled_cb=cancelled_cb, **kwargs,
        )
