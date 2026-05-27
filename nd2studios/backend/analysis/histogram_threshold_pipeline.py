"""Histogram Threshold Segmenter — AnalysisPipeline adapter.

Wraps the histothresh backend into the ND2Studios analysis registry so it
appears in the General Analysis tab alongside TearDetectionPipeline and
NucleiSegmentationPipeline.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from nd2studios.compute.parallel.thread_map import thread_map_planes
from nd2studios.core.analysis_registry import AnalysisPipeline, AnalysisResult
from nd2studios.core.plugin_registry import ParamSpec

from .histothresh.config import ThresholdConfig
from .histothresh.identifier import HistogramThresholdSegmenter


@AnalysisPipeline.register
class HistogramThresholdPipeline(AnalysisPipeline):
    """Identify image regions by intensity threshold.

    Supports four threshold methods (hysteresis, single, percentile, relative)
    and four directions (below, above, between, outside). Spatial cleanup via
    morphological opening/closing and small-object/hole removal. Stain-agnostic;
    works as a drop-in exclusion-mask producer alongside tear detection and
    nuclei segmentation.
    """

    name = "Histogram Threshold Segmenter"
    description = (
        "Identify regions by intensity threshold. "
        "Supports hysteresis, single, percentile, and relative methods across "
        "four directions (below, above, between, outside). "
        "Includes morphological spatial cleanup."
    )

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec(
                "channel_name", "Channel", "choice", "",
                choices=[],
                tooltip="Channel to threshold.",
            ),
            ParamSpec(
                "method", "Method", "choice", "hysteresis",
                choices=["hysteresis", "single", "percentile", "relative"],
                tooltip=(
                    "hysteresis: two-level threshold (strict core + permissive fringe). "
                    "single: hard cut. "
                    "percentile: resolved from the image histogram. "
                    "relative: fraction of within-frame median."
                ),
            ),
            ParamSpec(
                "direction", "Direction", "choice", "below",
                choices=["below", "above", "between", "outside"],
                tooltip=(
                    "below: pixels ≤ threshold (dark regions). "
                    "above: pixels ≥ threshold (bright regions). "
                    "between: band-pass. "
                    "outside: band-reject."
                ),
            ),
            ParamSpec(
                "bit_depth", "Bit depth", "int", 12,
                min_val=8, max_val=16, step=2,
                tooltip="Declared acquisition bit depth. Used to set the LUT range.",
            ),
            ParamSpec(
                "strict", "Strict threshold", "int", 30,
                min_val=0, max_val=65535, step=10,
                tooltip="(hysteresis) Core threshold — pixels definitely inside the mask.",
            ),
            ParamSpec(
                "permissive", "Permissive threshold", "int", 80,
                min_val=0, max_val=65535, step=10,
                tooltip="(hysteresis) Fringe threshold — pixels included if connected to the core.",
            ),
            ParamSpec(
                "low", "Low threshold", "int", 50,
                min_val=0, max_val=65535, step=10,
                tooltip="(single / between / outside) Hard lower bound.",
            ),
            ParamSpec(
                "high", "High threshold", "int", 500,
                min_val=0, max_val=65535, step=10,
                tooltip="(single / above / between / outside) Hard upper bound.",
            ),
            ParamSpec(
                "percentile_low", "Percentile low", "float", 5.0,
                min_val=0.0, max_val=100.0, step=0.5,
                tooltip="(percentile) Resolve the low bound from this image percentile.",
            ),
            ParamSpec(
                "percentile_high", "Percentile high", "float", 95.0,
                min_val=0.0, max_val=100.0, step=0.5,
                tooltip="(percentile) Resolve the high bound from this image percentile.",
            ),
            ParamSpec(
                "fraction_low", "Fraction low", "float", 0.30,
                min_val=0.0, max_val=2.0, step=0.05,
                tooltip="(relative) low = fraction × within-frame median.",
            ),
            ParamSpec(
                "fraction_high", "Fraction high", "float", 1.50,
                min_val=0.0, max_val=5.0, step=0.05,
                tooltip="(relative) high = fraction × within-frame median.",
            ),
            ParamSpec(
                "min_area", "Min area (px)", "int", 100,
                min_val=0, max_val=1_000_000, step=10,
                tooltip="Discard objects smaller than this area in pixels.",
            ),
            ParamSpec(
                "max_area", "Max area (px)", "int", 0,
                min_val=0, max_val=100_000_000, step=1000,
                tooltip=(
                    "Discard objects larger than this area in pixels. "
                    "Set to 0 to disable. Use this to exclude background blobs "
                    "that would otherwise dominate the mean area statistic."
                ),
            ),
            ParamSpec(
                "opening_radius", "Opening radius (px)", "int", 1,
                min_val=0, max_val=10, step=1,
                tooltip="Morphological opening removes isolated noise before closing.",
            ),
            ParamSpec(
                "closing_radius", "Closing radius (px)", "int", 2,
                min_val=0, max_val=10, step=1,
                tooltip="Morphological closing bridges small gaps inside objects.",
            ),
            ParamSpec(
                "min_hole_size", "Min hole size (px)", "int", 50,
                min_val=0, max_val=100_000, step=10,
                tooltip="Fill interior holes smaller than this area in pixels.",
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

        ch = params.get("channel_name", "")
        if not ch or ch not in channels:
            available = list(channels.keys())
            ch = available[0] if available else ""
        if not ch:
            raise ValueError("No channel available to analyse.")

        stack: np.ndarray = np.asarray(channels[ch])  # (T, H, W)
        if stack.ndim == 2:
            stack = stack[np.newaxis]  # treat as single frame

        pixel_size_um: float = float(metadata.get("pixel_size_um", 0.0))
        voxel_size = (pixel_size_um, pixel_size_um) if pixel_size_um > 0 else None

        cfg = self._build_config(params)
        seg = HistogramThresholdSegmenter(cfg)

        def _run_frame(frame: np.ndarray):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return seg.run(frame, voxel_size=voxel_size)

        _progress = (
            (lambda done, total: progress_cb(int(done / total * 100)))
            if progress_cb is not None else None
        )

        label_stack = np.zeros(stack.shape, dtype=np.int32)
        measurements: list[dict[str, Any]] = []

        raw = thread_map_planes(
            stack, _run_frame,
            progress_cb=_progress,
            cancelled_cb=cancelled_cb,
        )
        for t in sorted(raw):
            result = raw[t]
            label_stack[t] = result.labels
            for row in result.regions:
                measurements.append({
                    "frame": t,
                    "label_id": row["label_id"],
                    "area_px": row["area_px"],
                    "area_um2": row.get("area_um2", 0.0),
                    "centroid_y": row["centroid_y"],
                    "centroid_x": row["centroid_x"],
                    "mean_intensity": row["mean_intensity"],
                })

        summary = _build_summary(measurements)
        return AnalysisResult(
            label_masks={ch: label_stack},
            measurements=measurements,
            summary=summary,
        )

    @staticmethod
    def _build_config(params: Dict[str, Any]) -> ThresholdConfig:
        method: str = params.get("method", "hysteresis")
        direction: str = params.get("direction", "below")
        bit_depth: int = int(params.get("bit_depth", 12))

        # Collect all raw values; validation inside ThresholdConfig.__post_init__
        # will catch invalid combinations only for strict methods.
        # We pass None for fields not relevant to the chosen method so the
        # dataclass validator doesn't trip on unrelated fields.
        kwargs: dict[str, Any] = dict(
            method=method,
            direction=direction,
            bit_depth=bit_depth,
            bit_depth_strict=False,  # TIFF inputs may overflow declared depth
            min_area=int(params.get("min_area", 100)),
            max_area=int(params.get("max_area", 0)),
            opening_radius=int(params.get("opening_radius", 1)),
            closing_radius=int(params.get("closing_radius", 2)),
            min_hole_size=int(params.get("min_hole_size", 50)),
        )

        if method == "hysteresis":
            kwargs["strict"] = int(params.get("strict", 30))
            kwargs["permissive"] = int(params.get("permissive", 80))
        elif method == "single":
            if direction in ("below", "between", "outside"):
                kwargs["low"] = int(params.get("low", 50))
            if direction in ("above", "between", "outside"):
                kwargs["high"] = int(params.get("high", 500))
        elif method == "percentile":
            if direction in ("below", "between", "outside"):
                kwargs["percentile_low"] = float(params.get("percentile_low", 5.0))
            if direction in ("above", "between", "outside"):
                kwargs["percentile_high"] = float(params.get("percentile_high", 95.0))
        elif method == "relative":
            if direction in ("below", "between", "outside"):
                kwargs["fraction_low"] = float(params.get("fraction_low", 0.30))
            if direction in ("above", "between", "outside"):
                kwargs["fraction_high"] = float(params.get("fraction_high", 1.50))

        return ThresholdConfig(**kwargs)


def _build_summary(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    if not measurements:
        return {
            "total_objects": 0,
            "mean_area_px": 0.0,
            "std_area_px": 0.0,
            "mean_area_um2": 0.0,
            "std_area_um2": 0.0,
            "n_frames_with_objects": 0,
        }
    areas_px = np.array([m["area_px"] for m in measurements], dtype=float)
    areas_um2 = np.array([m["area_um2"] for m in measurements], dtype=float)
    frames_with = len({m["frame"] for m in measurements})
    return {
        "total_objects": len(measurements),
        "mean_area_px": float(np.mean(areas_px)),
        "std_area_px": float(np.std(areas_px)),
        "mean_area_um2": float(np.mean(areas_um2)),
        "std_area_um2": float(np.std(areas_um2)),
        "n_frames_with_objects": frames_with,
    }
