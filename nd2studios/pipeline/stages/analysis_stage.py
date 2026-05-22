"""
Analysis stage (V1.38 Phase 6) — per-M label-mask persistence.

Each ``AnalysisStage`` instance is scoped to one analysis pipeline name
(e.g. ``"Histogram Threshold"``) so the manifest can carry independent
records per pipeline. Per-M commits stream label masks to disk as the
Phase 5 multi-M state machine drains its run queue, so peak RAM stays
bounded to roughly one M's worth of labels.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from nd2studios.core.analysis_registry import AnalysisResult
from nd2studios.pipeline.session import StageRecord
from nd2studios.pipeline.stage import PipelineStage
from nd2studios.pipeline.storage import (
    open_label_stack,
    read_label_stack,
    write_label_stack,
)


_STAGE_PREFIX = "analysis:"
_SUMMARY_FILENAME = "summary.json"
_MAX_M_DIGITS = 3  # m_000 .. m_999 is plenty for realistic ND2 files


def stage_name_for(pipeline_name: str) -> str:
    """Manifest key for a given analysis pipeline."""
    return f"{_STAGE_PREFIX}{pipeline_name}"


def _safe_for_path(name: str) -> str:
    """Sanitize a free-form pipeline / channel name for use in a path."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "unnamed"


class AnalysisStage(PipelineStage):
    """Persistent representation of one analysis pipeline's outputs.

    Lifecycle::

        stage = AnalysisStage(session, pipeline_name)
        # Phase 5 commit job finishes for M=k →
        stage.commit_m(k, analysis_result)
        # ...repeat for every M...
        stage.finalize()              # writes summary.json + manifest record

    Reading back::

        rehydrated = stage.rehydrate_m(k, channel_names)
        # AnalysisResult with .label_masks populated.
    """

    def __init__(self, session, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.name = stage_name_for(pipeline_name)
        super().__init__(session)

    # ── commit per M ─────────────────────────────────────────────────

    def commit_m(self, m: int, result: AnalysisResult) -> Dict[str, str]:
        """Write label masks for one M position.

        Returns a dict mapping channel name to the artifact's
        session-relative path for inclusion in the manifest record.
        """
        m_dir = self._m_dir(m)
        m_dir.mkdir(parents=True, exist_ok=True)

        artifacts: Dict[str, str] = {}
        for channel, labels in result.label_masks.items():
            base = m_dir / f"labels_{_safe_for_path(channel)}"
            written_name = write_label_stack(base, labels)
            rel = self._relative(m_dir / written_name)
            artifacts[f"labels_{channel}_m_{m:0{_MAX_M_DIGITS}d}"] = rel

        # Secondary masks (used by tear detection's inverse overlay etc.)
        for channel, labels in (result.secondary_label_masks or {}).items():
            base = m_dir / f"secondary_{_safe_for_path(channel)}"
            written_name = write_label_stack(base, labels)
            rel = self._relative(m_dir / written_name)
            artifacts[f"secondary_{channel}_m_{m:0{_MAX_M_DIGITS}d}"] = rel

        self._merge_artifacts(artifacts)
        self._persist_m_summary(m, result)
        return artifacts

    def commit(self) -> StageRecord:
        """Finalize the stage record after one or more :meth:`commit_m` calls.

        Idempotent — calling commit() with no per-M writes simply
        stamps an empty record, which is harmless.
        """
        existing = self.session.get_stage(self.name)
        artifacts: Dict[str, str] = dict(existing.artifacts) if existing else {}
        parameters: Dict[str, Any] = dict(existing.parameters) if existing else {}
        parameters["pipeline_name"] = self.pipeline_name

        record = StageRecord(
            name=self.name,
            parameters=parameters,
            artifacts=artifacts,
        )
        self.session.record_stage(record)
        return record

    # ── rehydrate ────────────────────────────────────────────────────

    def committed_m_indices(self) -> List[int]:
        """Return M positions that have a label-mask artifact on disk.

        Inspects the manifest record, not the filesystem, so the list
        always reflects what was committed (artifacts removed by hand
        on disk will surface as missing files at read time).
        """
        rec = self.session.get_stage(self.name)
        if rec is None:
            return []
        ms: set[int] = set()
        for key in rec.artifacts:
            m = _extract_m_index(key)
            if m is not None:
                ms.add(m)
        return sorted(ms)

    def rehydrate_m(self, m: int) -> Optional[AnalysisResult]:
        """Read back an :class:`AnalysisResult` for a single M position.

        Returns None if no artifacts are recorded for that M.
        """
        rec = self.session.get_stage(self.name)
        if rec is None:
            return None
        suffix = f"_m_{m:0{_MAX_M_DIGITS}d}"

        label_masks: Dict[str, np.ndarray] = {}
        secondary: Dict[str, np.ndarray] = {}
        for key, rel in rec.artifacts.items():
            if not key.endswith(suffix):
                continue
            path = self.session.session_dir / rel
            if not path.exists():
                continue
            try:
                arr = read_label_stack(path)
            except Exception:  # noqa: BLE001 — bad file shouldn't kill rehydrate
                continue
            channel = _extract_channel(key, suffix)
            if channel is None:
                continue
            if key.startswith("labels_"):
                label_masks[channel] = arr
            elif key.startswith("secondary_"):
                secondary[channel] = arr

        if not label_masks and not secondary:
            return None

        summary, measurements = self._read_m_summary(m)
        result = AnalysisResult(
            label_masks=label_masks,
            secondary_label_masks=secondary,
            measurements=measurements,
            summary=summary,
        )
        return result

    def open_label_lazy(self, m: int, channel: str):
        """Open the label stack for one channel without copying into RAM.

        Returns a zarr ``Array`` when zarr-backed (supports ``[t]``
        indexing), or a numpy array when NPZ-backed.
        """
        rec = self.session.get_stage(self.name)
        if rec is None:
            return None
        key = f"labels_{channel}_m_{m:0{_MAX_M_DIGITS}d}"
        rel = rec.artifacts.get(key)
        if rel is None:
            return None
        path = self.session.session_dir / rel
        if not path.exists():
            return None
        return open_label_stack(path)

    # ── helpers ──────────────────────────────────────────────────────

    def _m_dir(self, m: int) -> Path:
        return self.session.session_dir / "analysis" / _safe_for_path(
            self.pipeline_name
        ) / f"m_{m:0{_MAX_M_DIGITS}d}"

    def _summary_path(self, m: int) -> Path:
        return self._m_dir(m) / _SUMMARY_FILENAME

    def _relative(self, path: Path) -> str:
        return str(path.relative_to(self.session.session_dir)).replace("\\", "/")

    def _merge_artifacts(self, new: Dict[str, str]) -> None:
        """Append per-M artifacts into the manifest record incrementally.

        We update the manifest after every M so a crash mid-run leaves a
        consistent record of what made it to disk — the next run can
        skip those M positions instead of redoing them.
        """
        existing = self.session.get_stage(self.name)
        artifacts: Dict[str, str] = dict(existing.artifacts) if existing else {}
        artifacts.update(new)
        record = StageRecord(
            name=self.name,
            parameters={"pipeline_name": self.pipeline_name},
            artifacts=artifacts,
        )
        self.session.record_stage(record)

    def _persist_m_summary(self, m: int, result: AnalysisResult) -> None:
        summary_path = self._summary_path(m)
        payload = {
            "pipeline_name": self.pipeline_name,
            "m": m,
            "summary": _to_jsonable(result.summary),
            "measurements": _to_jsonable(result.measurements),
            "overlay_color": list(result.overlay_color)
            if result.overlay_color is not None
            else None,
            "overlay_alpha": float(result.overlay_alpha),
            "secondary_overlay_color": list(result.secondary_overlay_color)
            if result.secondary_overlay_color is not None
            else None,
            "secondary_overlay_alpha": float(result.secondary_overlay_alpha),
            "channel_names": list(result.label_masks.keys()),
            "voxel_counts_path": None,
        }
        if result.volumetric_voxel_counts is not None:
            voxel_path = self._m_dir(m) / "voxel_counts.json"
            voxel_path.write_text(
                json.dumps(_voxel_counts_to_jsonable(result.volumetric_voxel_counts))
            )
            payload["voxel_counts_path"] = _SUMMARY_FILENAME.replace(
                _SUMMARY_FILENAME, "voxel_counts.json"
            )
        summary_path.write_text(json.dumps(payload, indent=2))

    def _read_m_summary(
        self, m: int
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        path = self._summary_path(m)
        if not path.exists():
            return ({}, [])
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return ({}, [])
        return (
            dict(data.get("summary", {})),
            list(data.get("measurements", [])),
        )


# ── module helpers ───────────────────────────────────────────────────


_M_INDEX_RE = re.compile(r"_m_(\d+)$")


def _extract_m_index(key: str) -> Optional[int]:
    m = _M_INDEX_RE.search(key)
    return int(m.group(1)) if m else None


def _extract_channel(key: str, suffix: str) -> Optional[str]:
    """Pull the channel name out of an artifact key.

    Keys look like ``labels_<channel>_m_<NNN>`` or
    ``secondary_<channel>_m_<NNN>``. The channel itself may contain
    underscores, so we strip the known prefix and the known M suffix.
    """
    if not key.endswith(suffix):
        return None
    body = key[: -len(suffix)]
    for prefix in ("labels_", "secondary_"):
        if body.startswith(prefix):
            return body[len(prefix):]
    return None


def _to_jsonable(value):
    """Recursively coerce numpy scalars/arrays into JSON-safe Python types."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _voxel_counts_to_jsonable(
    counts: Dict[str, Dict[tuple, int]],
) -> Dict[str, Dict[str, int]]:
    """Stringify ``(frame, label_id)`` tuple keys so JSON can hold them."""
    out: Dict[str, Dict[str, int]] = {}
    for channel, per_pair in counts.items():
        out[channel] = {
            f"{int(t)},{int(label_id)}": int(n)
            for (t, label_id), n in per_pair.items()
        }
    return out
