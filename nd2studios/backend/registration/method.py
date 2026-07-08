"""Registered registration methods (V1.56).

Ships :class:`RigidRegistration`, whose ``run`` estimates per-frame transforms on
a reference-channel ``(T,H,W)`` series (via :mod:`nd2studios.backend.registration.
estimate`) and returns them plus the aligned reference series. Force-imported in
``__main__.py`` so its ``@RegistrationMethod.register`` decorator fires at startup
and the method is discoverable by the registration node (its ``get_params`` also
supplies the node's ParamSpec list, via ``registry_adapter.param_specs_for``).

Pure backend module — no PySide6.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from nd2studios.backend.registration import estimate
from nd2studios.core.plugin_registry import ParamSpec
from nd2studios.core.registration_registry import RegistrationMethod, RegistrationResult


@RegistrationMethod.register
class RigidRegistration(RegistrationMethod):
    name = "Rigid / Translation"
    description = (
        "Phase-correlation translation (optionally + ECC euclidean/affine refine) "
        "of a (T,H,W) series onto a reference frame. Register once on the wired "
        "channel, apply to all channels."
    )

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec(
                name="model", label="Transform model", param_type="choice",
                default="translation",
                choices=["translation", "euclidean", "affine", "feature"],
                tooltip="Degrees of freedom / method. 'translation' (shift) is the "
                        "workhorse for stage/thermal drift; 'euclidean' adds "
                        "rotation; 'affine' adds scale/shear (ECC, seeded from the "
                        "phase-correlation shift). 'feature' uses ORB keypoints + "
                        "RANSAC — robust to LARGE motion / rotation / scale / "
                        "partial overlap (re-mount, multi-round), needs texture.",
            ),
            ParamSpec(
                name="reference", label="Reference frame", param_type="choice",
                default="template",
                choices=["template", "first", "previous", "mean"],
                tooltip="What each frame is registered to. 'template' (recommended) "
                        "is a robust two-pass anchor (rough-stabilize, then average) "
                        "— best for long or photobleaching series where later frames "
                        "otherwise drift off. 'first' = frame 0; 'previous' = "
                        "cumulative (can accumulate error); 'mean' = series average.",
            ),
            ParamSpec(
                name="feature_transform", label="Feature transform",
                param_type="choice", default="affine",
                choices=["euclidean", "similarity", "affine"],
                visible_when={"model": "feature"},
                tooltip="Model RANSAC fits to the matched keypoints: euclidean "
                        "(rigid), similarity (+scale), or affine (+shear).",
            ),
            ParamSpec(
                name="min_inliers", label="Min feature inliers", param_type="int",
                default=8, min_val=3, max_val=1000, step=1,
                visible_when={"model": "feature"},
                tooltip="Minimum RANSAC inlier matches to accept a frame's "
                        "transform; below this the frame falls back (holds the last "
                        "good transform).",
            ),
            ParamSpec(
                name="upsample", label="Sub-pixel factor", param_type="int",
                default=20, min_val=1, max_val=100, step=1,
                tooltip="Registration accuracy = 1/upsample pixels.",
            ),
            ParamSpec(
                name="highpass_sigma", label="High-pass sigma (px)",
                param_type="float", default=2.0, min_val=0.0, max_val=20.0, step=0.5,
                tooltip="Band-pass whitening before correlation (0 = off).",
            ),
            ParamSpec(
                name="normalize", label="Per-frame normalization",
                param_type="choice", default="none", choices=["none", "zscore"],
                tooltip="'zscore' normalizes each frame's intensity before "
                        "correlation — counters photobleaching / intensity decay so "
                        "later, dimmer frames still register.",
            ),
            ParamSpec(
                name="interp_order", label="Interpolation order", param_type="int",
                default=1, min_val=0, max_val=3, step=1,
                tooltip="0 nearest, 1 bilinear, 3 cubic.",
            ),
            ParamSpec(
                name="min_confidence", label="Min confidence", param_type="float",
                default=0.2, min_val=0.0, max_val=1.0, step=0.05,
                tooltip="Frames whose registration confidence (correlation / inlier "
                        "fraction) is below this HOLD the last good transform instead "
                        "of applying a spurious one — the key guard against later "
                        "timepoints drifting off when SNR drops. 0 = never gate.",
            ),
        ]

    def run(
        self,
        reference: np.ndarray,
        moving: np.ndarray,
        pixel_size_um: Tuple[float, ...],
        params: Dict[str, Any],
        progress_cb: Optional[Callable[[int], None]] = None,
        cancelled_cb: Optional[Callable[[], bool]] = None,
    ) -> RegistrationResult:
        series = np.asarray(reference)
        model = str(params.get("model", "translation"))
        ref_mode = str(params.get("reference", "template"))
        upsample = int(params.get("upsample", 20))
        highpass = float(params.get("highpass_sigma", 2.0))
        order = int(params.get("interp_order", 1))
        min_conf = float(params.get("min_confidence", 0.2))
        normalize = str(params.get("normalize", "none"))
        roi = params.get("roi")  # serializable ROI spec (rect / shapes) or None
        feature_transform = str(params.get("feature_transform", "affine"))
        min_inliers = int(params.get("min_inliers", 8))

        tf = estimate.estimate_series(
            series, model=model, reference=ref_mode, upsample=upsample,
            highpass_sigma=highpass, min_confidence=min_conf, normalize=normalize,
            roi=roi, feature_transform=feature_transform, min_inliers=min_inliers,
            progress_cb=progress_cb, cancelled_cb=cancelled_cb,
        )
        aligned = estimate.apply_series(series, tf, interp_order=order)
        return RegistrationResult(
            model=model,
            shifts_px=tf["shifts"],
            transforms=tf["warps"],
            aligned=aligned,
            confidence=tf["confidence"],
            pixel_size_um=tuple(pixel_size_um) if pixel_size_um else (),
            reference_mode=ref_mode,
            method=self.name,
            diagnostics={"transforms": tf, "interp_order": order,
                         "gated": tf.get("gated")},
        )
