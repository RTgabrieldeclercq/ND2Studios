"""Registered pyALDIC (2D AL-DIC) method.

Ships :class:`PyALDICMethod`, registered under the shared
:class:`~nd2studios.core.dvc_registry.DVCMethod` registry (``name="pyALDIC"``) so
the 2D DIC node reuses the DVC data model + viewer. Its ``get_params`` is the
single source of truth for the DIC node's engine ParamSpec list (consumed by
``registry_adapter.param_specs_for``); its ``run`` wraps the optional ``al-dic``
package via :func:`nd2studios.backend.dic.engine.run_pyaldic_pair`.

Force-imported in ``__main__.py`` so the ``@DVCMethod.register`` decorator fires
at startup. Pure backend module — no PySide6 (``al_dic`` is imported lazily by the
engine, so importing this module never requires the extra).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from nd2studios.core.dvc_registry import DVCMethod, DVCResult
from nd2studios.core.plugin_registry import ParamSpec


@DVCMethod.register
class PyALDICMethod(DVCMethod):
    name = "pyALDIC"
    description = (
        "2D Augmented-Lagrangian Digital Image Correlation (AL-DIC) via the "
        "open-source 'al-dic' package (pyALDIC, Yang & Bhattacharya 2019) — hybrid "
        "local IC-GN + global ADMM over an adaptive quadtree mesh. Optional extra: "
        "pip install al-dic."
    )

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec(
                name="winsize", label="Subset size (px)",
                param_type="int", default=40, min_val=4, max_val=256, step=2,
                tooltip="Edge length of each correlation subset (even integer). "
                        "Larger = more robust, lower spatial resolution.",
            ),
            ParamSpec(
                name="winstepsize", label="Subset spacing / step (px)",
                param_type="int", default=16, min_val=2, max_val=128, step=2,
                tooltip="Spacing between subset centers = the base mesh pitch. "
                        "Snapped to a power of two (pyALDIC requirement).",
            ),
            ParamSpec(
                name="winsize_min", label="Min subset (adaptive, px)",
                param_type="int", default=8, min_val=2, max_val=128, step=2,
                tooltip="Smallest element the adaptive quadtree may refine to "
                        "(power of two, ≤ step). Used by the Mesh Refinement node.",
            ),
            ParamSpec(
                name="init_guess_mode", label="Initial guess",
                param_type="choice", default="auto",
                choices=["auto", "fft", "previous", "seed_propagation"],
                tooltip="How the first-frame seed displacement is found. 'auto' "
                        "picks FFT; 'previous' warm-starts from the last frame; "
                        "'seed_propagation' grows from reliable seed points.",
            ),
            ParamSpec(
                name="admm_max_iter", label="ADMM iterations",
                param_type="int", default=3, min_val=1, max_val=20, step=1,
                tooltip="Augmented-Lagrangian outer iterations coupling the local "
                        "IC-GN and global compatibility steps.",
            ),
            ParamSpec(
                name="icgn_max_iter", label="IC-GN max iterations",
                param_type="int", default=100, min_val=5, max_val=500, step=5,
                tooltip="Max inverse-compositional Gauss-Newton iterations per "
                        "subset (inner local solve).",
            ),
            ParamSpec(
                name="mu", label="ADMM penalty (mu)",
                param_type="float", default=1e-3, min_val=1e-6, max_val=1.0,
                step=1e-3,
                tooltip="Augmented-Lagrangian coupling penalty.",
            ),
            ParamSpec(
                name="tol", label="Convergence tolerance",
                param_type="float", default=1e-2, min_val=1e-6, max_val=0.5,
                step=1e-3,
                tooltip="ADMM stopping tolerance (0–1).",
            ),
            ParamSpec(
                name="disp_smoothness", label="Displacement smoothness",
                param_type="float", default=5e-4, min_val=0.0, max_val=1.0,
                step=1e-4,
                tooltip="Global regularization weight on the displacement field.",
            ),
            ParamSpec(
                name="strain_smoothness", label="Strain smoothness",
                param_type="float", default=1e-5, min_val=0.0, max_val=1.0,
                step=1e-5,
                tooltip="Smoothing applied when deriving strain from displacement.",
            ),
            ParamSpec(
                name="compute_strain", label="Compute strain",
                param_type="bool", default=True,
                tooltip="Also compute the strain tensor (exx/eyy/exy, von Mises). "
                        "The viewer can derive strain from displacement either way.",
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
        """Correlate a single 2D reference/deformed image pair via pyALDIC.

        ``kwargs`` may carry ``roi_mask`` (a boolean ``(H, W)`` ROI) and
        ``refinement`` (an adaptive-mesh spec) forwarded from the DIC job."""
        from nd2studios.backend.dic.engine import run_pyaldic_pair
        return run_pyaldic_pair(
            np.asarray(ref_vol), np.asarray(def_vol), voxel_size_um, params,
            progress_cb=progress_cb, cancelled_cb=cancelled_cb,
            roi_mask=kwargs.get("roi_mask"),
            refinement=kwargs.get("refinement"),
        )
