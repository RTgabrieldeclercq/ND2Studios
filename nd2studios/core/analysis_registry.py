"""
Analysis Pipeline Registry for ND2Studios.

Analysis pipelines are distinct from enhancement plugins: they operate on
whole multi-frame datasets, produce structured results (label masks +
measurements), and are run once per session rather than chained in a recipe.

Usage:
    @AnalysisPipeline.register
    class MyPipeline(AnalysisPipeline):
        name = "My Pipeline"
        description = "Does something."
        def get_params(self): ...
        def run(self, channels, metadata, params, progress_cb, cancelled_cb): ...
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np

from nd2studios.core.plugin_registry import ParamSpec  # reuse, do not duplicate


@dataclass
class AnalysisResult:
    """Structured output from an AnalysisPipeline.run() call."""
    label_masks: Dict[str, Any] = field(default_factory=dict)
    # {channel_name: (T, H, W) int32 label store — 0 = background}.
    # V1.46: a value may be an in-RAM ndarray (eager) OR a disk-backed lazy
    # reader (zarr Array / memmapped .npy from a streamed run). Both support
    # ``[t]`` per-frame indexing and ``.shape``, so all consumers must index
    # per frame rather than treating it as a fully-resident ndarray.

    measurements: List[Dict[str, Any]] = field(default_factory=list)
    # One dict per detected object:
    # {frame, label_id, area_px, area_um2, centroid_y, centroid_x, mean_intensity}

    summary: Dict[str, Any] = field(default_factory=dict)
    # {total_objects, mean_area_px, std_area_px, mean_area_um2, std_area_um2,
    #  n_frames_with_objects}

    overlay_color: Optional[tuple] = None
    # When set, all non-zero labels are rendered with this single (R, G, B) color
    # instead of the multi-color palette.  Used by pipelines that output binary
    # masks where per-label coloring is meaningless.

    overlay_alpha: float = 0.45
    # Blend weight for the label overlay (0.0 = transparent, 1.0 = opaque).

    overlay_outline: bool = False
    # When True, label masks are rendered as boundary OUTLINES (via
    # skimage.segmentation.find_boundaries) instead of filled regions — used by
    # cell-boundary-drawing pipelines (e.g. StarDist Segmentation). Honored by the
    # analysis-page / pipeline-preview overlays and the results_engine exporters.

    secondary_label_masks: Dict[str, np.ndarray] = field(default_factory=dict)
    # Optional secondary overlays keyed by a display name.  Rendered on top of
    # the primary overlay.  Intended for inverse/background masks.

    secondary_overlay_color: Optional[tuple] = None
    # Single (R, G, B) color applied to all secondary masks.

    secondary_overlay_alpha: float = 0.3
    # Blend weight for secondary overlays.

    volumetric_voxel_counts: Optional[Dict[str, Dict[Tuple[int, int], int]]] = None
    # Optional per-channel per-(frame, label_id) voxel count. When supplied,
    # the Results tab uses these counts to compute true 3D volume (voxels x
    # pixel_size^2 x z_step). When None, volume falls back to the uniform-Z
    # formula area_um2 x n_zslices x z_step_um. Manual Mask sets this whenever
    # the user has drawn per-Z shapes so volumes reflect the real per-slice
    # mask rather than the uniform-Z assumption.


class AnalysisPipeline(ABC):
    """Base class for all General Analysis pipelines.

    Subclasses declare class attributes `name` and `description`, then
    decorate with @AnalysisPipeline.register so the Analysis page can
    discover them.
    """
    name: str = "Unnamed"
    description: str = ""

    # V1.46: when True, the adaptive streaming gate keeps this pipeline on the
    # in-RAM path (it needs whole-stack / multi-frame context, e.g. temporal
    # ops). All current pipelines are strictly per-frame, so this is False.
    needs_full_stack: bool = False

    _registry: Dict[str, Type["AnalysisPipeline"]] = {}

    @classmethod
    def register(cls, pipeline_cls: Type["AnalysisPipeline"]) -> Type["AnalysisPipeline"]:
        """Decorator: register a pipeline subclass under its name."""
        AnalysisPipeline._registry[pipeline_cls.name] = pipeline_cls
        return pipeline_cls

    @classmethod
    def get_pipelines(cls) -> List[Type["AnalysisPipeline"]]:
        return list(AnalysisPipeline._registry.values())

    @classmethod
    def get_pipeline(cls, name: str) -> Optional[Type["AnalysisPipeline"]]:
        return AnalysisPipeline._registry.get(name)

    @abstractmethod
    def get_params(self) -> List[ParamSpec]:
        """Return a fresh list of ParamSpec objects describing this pipeline's
        tunable parameters. Called on each activation so the page can inject
        live channel names into 'choice' params before rendering."""
        ...

    @abstractmethod
    def run(
        self,
        channels: Dict[str, np.ndarray],
        metadata: Dict[str, Any],
        params: Dict[str, Any],
        progress_cb: Optional[Callable[[int], None]] = None,
        cancelled_cb: Optional[Callable[[], bool]] = None,
    ) -> AnalysisResult:
        """Execute the analysis pipeline.

        Args:
            channels:     {channel_name: (T, H, W) float/int array}
            metadata:     nd2_metadata dict from ND2StudiosRecord
            params:       {param_name: value} from ParamEditor
            progress_cb:  optional callable that accepts an int 0-100
            cancelled_cb: optional zero-arg callable returning True if the
                          worker was cancelled (check between frames)
        """
        ...
