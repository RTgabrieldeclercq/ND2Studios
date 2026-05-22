"""
V1.38 Phase 6 — per-source workspace + stage commits.

Owns the on-disk artifacts that survive across page transitions:
the committed recipe (parameter-only) and the analysis label masks
(zarr if available, NPZ fallback). See
:mod:`nd2studios.pipeline.session` for the workspace layout.

The package is Qt-free. Stage commits are called from the GUI thread
after a worker has already produced the heavy data; the writes are
synchronous and small (manifest JSON + a label-stack write per M).
"""
from __future__ import annotations

from nd2studios.pipeline.session import (
    DISABLE_WORKSPACE_ENV,
    Session,
    SessionManifest,
    StageRecord,
    default_workspace_root,
    hash_source_file,
    workspace_disabled,
)
from nd2studios.pipeline.stage import PipelineStage
from nd2studios.pipeline.stages.analysis_stage import AnalysisStage
from nd2studios.pipeline.stages.pyramid_stage import (
    PyramidReader,
    PyramidStage,
    PyramidUnavailable,
    default_pyramid_levels,
)
from nd2studios.pipeline.stages.recipe_stage import EnhancedDataset, RecipeStage
from nd2studios.pipeline.storage import (
    HAS_ZARR,
    read_label_stack,
    write_label_stack,
)

__all__ = [
    "AnalysisStage",
    "DISABLE_WORKSPACE_ENV",
    "EnhancedDataset",
    "HAS_ZARR",
    "PipelineStage",
    "PyramidReader",
    "PyramidStage",
    "PyramidUnavailable",
    "RecipeStage",
    "Session",
    "SessionManifest",
    "StageRecord",
    "default_pyramid_levels",
    "default_workspace_root",
    "hash_source_file",
    "read_label_stack",
    "workspace_disabled",
    "write_label_stack",
]
