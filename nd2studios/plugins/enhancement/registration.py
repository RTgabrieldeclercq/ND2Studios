"""Image-registration enhancement plugin (V1.56).

A Processing-stage node that stabilizes a ``(T, H, W)`` channel series onto a
reference frame — temporal drift correction (stage / thermal drift), with an
optional ECC rigid/affine refinement for stage rotation / re-mount.

Drops straight into the recipe pipeline and the Pipelines palette with no
node-graph or page changes: ``registry_adapter.enhancement_specs()`` enumerates
registered enhancement plugins with no hardcoded names, and
``workers/recipe_worker.py`` already invokes ``execute``. It sees one channel's
``(T, H, W)`` at a time (no cross-channel), so it is scoped to per-channel
stabilization; cross-channel "register once on a reference channel, apply to
all" is the future ``RegistrationMethod`` registry's job (review Part 3.4).

Backend-pure: the heavy lifting lives in
:mod:`nd2studios.backend.registration.estimate` (no PySide6 here either).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from nd2studios.core.plugin_registry import EnhancementPlugin, ParamSpec


@EnhancementPlugin.register
class RegistrationPlugin(EnhancementPlugin):
    name = "Registration (Drift Correction)"
    description = (
        "Stabilize a (T,H,W) series onto a reference frame by sub-pixel phase "
        "correlation (translation), with optional ECC rigid/affine refinement."
    )

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec(
                name="model", label="Transform model", param_type="choice",
                default="translation",
                choices=["translation", "euclidean", "affine"],
                tooltip="Degrees of freedom. 'translation' (shift only) is the "
                        "workhorse for stage/thermal drift; 'euclidean' adds "
                        "rotation; 'affine' adds scale/shear. Use the simplest "
                        "model the physics allows.",
            ),
            ParamSpec(
                name="reference", label="Reference frame", param_type="choice",
                default="previous", choices=["first", "previous", "mean"],
                tooltip="Register each frame to the first frame, the previous "
                        "frame (cumulative), or the series mean.",
            ),
            ParamSpec(
                name="upsample", label="Sub-pixel factor", param_type="int",
                default=20, min_val=1, max_val=100, step=1,
                tooltip="Registration accuracy = 1/upsample pixels.",
            ),
            ParamSpec(
                name="highpass_sigma", label="High-pass sigma (px)",
                param_type="float", default=2.0, min_val=0.0, max_val=20.0,
                step=0.5,
                tooltip="Band-pass whitening before correlation (0 = off).",
            ),
            ParamSpec(
                name="interp_order", label="Interpolation order", param_type="int",
                default=1, min_val=0, max_val=3, step=1,
                tooltip="0 nearest, 1 bilinear, 3 cubic.",
            ),
            ParamSpec(
                name="min_confidence", label="Min confidence", param_type="float",
                default=0.0, min_val=0.0, max_val=1.0, step=0.05,
                tooltip="Frames whose registration confidence (correlation) is "
                        "below this are left unshifted — guards spurious shifts on "
                        "near-blank / low-texture frames. 0 = never gate.",
            ),
        ]

    def execute(self, volume: np.ndarray, params: Dict[str, Any],
                progress_cb: Optional[Callable[[int], None]] = None) -> np.ndarray:
        from nd2studios.backend.registration.estimate import stabilize

        vol = np.asarray(volume)
        if vol.ndim != 3 or vol.shape[0] < 2:
            # Nothing to stabilize (single frame or non-series) — pass through.
            if progress_cb:
                progress_cb(100)
            return vol

        aligned, _shifts, _conf = stabilize(
            vol,
            model=str(params.get("model", "translation")),
            reference=str(params.get("reference", "previous")),
            upsample=int(params.get("upsample", 20)),
            highpass_sigma=float(params.get("highpass_sigma", 2.0)),
            interp_order=int(params.get("interp_order", 1)),
            min_confidence=float(params.get("min_confidence", 0.0)),
            progress_cb=progress_cb,
        )
        return aligned
