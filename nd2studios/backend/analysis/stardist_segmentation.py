"""StarDist segmentation pipeline (ported from CellTracker).

StarDist is an optional dependency. The pipeline class is always importable;
calling run() without StarDist installed raises ImportError with install
instructions. The page catches this and shows a friendly dialog.

Requires (if running):
    pip install stardist tensorflow csbdeep

Mirrors the structure of ``nuclei_segmentation.py`` (the Cellpose node): per-frame
2D instance segmentation driven by ``plane_runner``, producing integer label masks
+ per-object measurements. The backend call is the vendored
:mod:`nd2studios.backend.celltracker.segmentation`.
"""
from __future__ import annotations

import importlib.util
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from nd2studios.backend.analysis.plane_runner import (
    make_frame_cb, make_label_writers, run_planes_to_labels,
)
from nd2studios.backend.analysis.source_utils import (
    filter_and_relabel, read_plane, source_shape,
)
from nd2studios.core.analysis_registry import AnalysisPipeline, AnalysisResult
from nd2studios.core.plugin_registry import ParamSpec

STARDIST_AVAILABLE: bool = importlib.util.find_spec("stardist") is not None


@AnalysisPipeline.register
class StarDistSegmentationPipeline(AnalysisPipeline):
    """Per-frame 2D instance segmentation via StarDist (star-convex nuclei).

    Segments each time-point independently with a pretrained StarDist2D model
    (fluorescence / H&E / DSB2018). Produces integer label masks and per-object
    measurements (area, centroid, mean channel intensity). Boundaries render as
    outlines by default (``outline``) — the cell-boundary-drawing CellTracker is
    built around.

    Optional dependency: ``pip install stardist tensorflow csbdeep``
    """

    name = "StarDist Segmentation"
    description = (
        "Per-frame 2D instance segmentation of star-convex nuclei using StarDist. "
        "Outputs integer label masks (drawn as cell-boundary outlines) and "
        "per-object measurements. Requires: pip install stardist tensorflow csbdeep"
    )

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec(
                "channel_name", "Channel", "choice", "",
                choices=[],
                tooltip="Channel to segment (e.g. DAPI). Populated from loaded data.",
            ),
            ParamSpec(
                "model_name", "Model", "choice", "2D_versatile_fluo",
                choices=["2D_versatile_fluo", "2D_versatile_he", "2D_paper_dsb2018"],
                tooltip="Pretrained StarDist2D model: 'fluo' for fluorescence, "
                        "'he' for H&E brightfield, 'dsb2018' for the DSB nuclei set. "
                        "Downloaded on first use (needs internet).",
            ),
            ParamSpec(
                "prob_thresh", "Probability threshold", "float", 0.5,
                min_val=0.01, max_val=0.99, step=0.05,
                tooltip="Object-probability cutoff. Lower → more (incl. faint) "
                        "detections; higher → only confident nuclei.",
            ),
            ParamSpec(
                "nms_thresh", "Overlap (NMS) threshold", "float", 0.3,
                min_val=0.01, max_val=0.99, step=0.05,
                tooltip="Non-maximum-suppression overlap. Lower splits touching "
                        "nuclei; higher allows more overlap.",
            ),
            ParamSpec(
                "scale", "Scale", "float", 1.0,
                min_val=0.25, max_val=4.0, step=0.25,
                tooltip="Rescale before inference. <1 for nuclei larger than the "
                        "model's training size, >1 for smaller. 1.0 = no rescale.",
            ),
            ParamSpec(
                "min_area", "Min area (px²)", "int", 50,
                min_val=1, max_val=100_000, step=10,
                tooltip="Discard objects smaller than this area in pixels².",
            ),
            ParamSpec(
                "max_area", "Max area (px²)", "int", 50_000,
                min_val=100, max_val=10_000_000, step=100,
                tooltip="Discard objects larger than this area in pixels².",
            ),
            ParamSpec(
                "outline", "Draw boundaries as outlines", "bool", True,
                tooltip="Render the detected cells as boundary outlines instead of "
                        "filled regions in the overlay / export.",
            ),
        ]

    def run(
        self,
        channels: Dict[str, np.ndarray],
        metadata: Dict[str, Any],
        params: Dict[str, Any],
        progress_cb: Optional[Callable[[int], None]] = None,
        cancelled_cb: Optional[Callable[[], bool]] = None,
    ) -> AnalysisResult:
        if not STARDIST_AVAILABLE:
            raise ImportError(
                "StarDist is not installed.\n\n"
                "Install with:\n    pip install stardist tensorflow csbdeep\n\n"
                "Then restart ND2Studios."
            )

        from skimage.measure import regionprops  # already a project dependency

        from nd2studios.backend.celltracker.segmentation import (
            get_stardist_model, segment_frame,
        )

        # Honour the runtime GPU flag (StarDist/TF). When disabled we clear
        # CUDA_VISIBLE_DEVICES before the first TF import (see segmentation.py).
        try:
            from nd2studios.compute.gpu import is_gpu_enabled
            use_gpu = bool(is_gpu_enabled())
        except Exception:  # noqa: BLE001
            use_gpu = False
        disable_gpu = not use_gpu

        # ── Resolve channel ──────────────────────────────────────────────────
        channel_name = params.get("channel_name", "")
        if channel_name not in channels:
            channel_name = next(iter(channels))

        source = channels[channel_name]
        T, H, W = source_shape(source)
        pixel_size_um: float = float(metadata.get("pixel_size_um", 1.0))

        # ── StarDist parameters ───────────────────────────────────────────────
        model_name: str = params.get("model_name", "2D_versatile_fluo")
        prob_thresh: float = float(params.get("prob_thresh", 0.5))
        nms_thresh: float = float(params.get("nms_thresh", 0.3))
        raw_scale = float(params.get("scale", 1.0))
        scale: Optional[float] = raw_scale if raw_scale and raw_scale != 1.0 else None
        min_area: int = int(params.get("min_area", 50))
        max_area: int = int(params.get("max_area", 50_000))
        outline: bool = bool(params.get("outline", True))

        if progress_cb:
            progress_cb(0)

        # Load once up front (singleton) so the model download / TF init failure
        # surfaces before the per-frame loop.
        model = get_stardist_model(model_name, disable_gpu=disable_gpu)

        def _per_frame(t: int):
            frame = read_plane(source, t).astype(np.float32)
            mask, _ = segment_frame(
                frame, model=model,
                prob_thresh=prob_thresh, nms_thresh=nms_thresh, scale=scale,
                model_name=model_name, disable_gpu=disable_gpu,
            )
            mask = np.asarray(mask, dtype=np.int32)

            # Area filter + contiguous relabel in one O(pixels) pass.
            mask = filter_and_relabel(mask, min_area, max_area)

            rows: List[Dict[str, Any]] = []
            for region in regionprops(mask, intensity_image=frame):
                cy, cx = region.centroid
                rows.append({
                    "frame": t,
                    "label_id": int(region.label),
                    "area_px": float(region.area),
                    "area_um2": float(region.area) * pixel_size_um ** 2,
                    "centroid_y": float(cy),
                    "centroid_x": float(cx),
                    "mean_intensity": float(region.mean_intensity),
                })
            return mask, rows, None

        primary_writer, _ = make_label_writers(params, channel_name, (T, H, W))
        # StarDist / TensorFlow are not thread-safe → sequential (n_workers=1);
        # streaming still bounds RAM to one frame + the mask sink.
        out = run_planes_to_labels(
            n_frames=T, height=H, width=W, per_frame_fn=_per_frame,
            primary_writer=primary_writer, n_workers=1,
            progress_cb=progress_cb, cancelled_cb=cancelled_cb,
            frame_cb=make_frame_cb(params),
        )
        measurements = out.measurements

        areas = [m["area_px"] for m in measurements]
        summary: Dict[str, Any] = {
            "total_objects": len(measurements),
            "n_frames_with_objects": len({m["frame"] for m in measurements}),
            "mean_area_px": float(np.mean(areas)) if areas else 0.0,
            "std_area_px": float(np.std(areas)) if areas else 0.0,
            "mean_area_um2": float(np.mean(areas)) * pixel_size_um ** 2 if areas else 0.0,
            "std_area_um2": float(np.std(areas)) * pixel_size_um ** 2 if areas else 0.0,
        }

        # Match CellTracker's boundary rendering exactly: a single red
        # (255, 50, 50) outline at full brightness (see CellTracker
        # widgets/cell_overlay.draw_boundaries_on_painter). When outlines are
        # off, fall back to the multi-color filled palette.
        overlay_color = (255, 50, 50) if outline else None
        overlay_alpha = 1.0 if outline else 0.45

        return AnalysisResult(
            label_masks={channel_name: out.primary},
            measurements=measurements,
            summary=summary,
            overlay_color=overlay_color,
            overlay_alpha=overlay_alpha,
            overlay_outline=outline,
        )
