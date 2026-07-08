"""
Recipe stage (V1.38 Phase 6) — parameter-only commit.

The recipe is just an ordered ``List[(plugin_name, params)]`` plus a
``normalized`` flag. We persist it to ``recipe/recipe.json`` inside the
session workspace and expose an :class:`EnhancedDataset` proxy that
lazily applies the recipe to the raw channels on demand. This lets the
Export / Analysis / Results pages keep working after the Recipe page
releases ``exp._processed_channels`` from RAM.

We deliberately do **not** materialize a ``processed.zarr`` by default.
The generic Phase 6 doc treats materialization as opt-in for users who
"export many derivatives", and benchmarks on ND2Studios's enhancement
plugins (skimage / OpenCV at a few ms/frame) don't justify the disk and
write-time cost. A later phase can revisit if profiling changes its mind.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from nd2studios.core.plugin_registry import PluginBase
from nd2studios.pipeline.session import StageRecord
from nd2studios.pipeline.stage import PipelineStage
from nd2studios.utils.resources import recommended_worker_count


_RECIPE_ARTIFACT = "recipe"
_RECIPE_RELATIVE = "recipe/recipe.json"


class RecipeStage(PipelineStage):
    """Persistent representation of the user's accepted recipe.

    The Recipe page calls :meth:`set_recipe` with the new
    ``(recipe, normalized, channel_names)`` after every Accept, then
    :meth:`commit` to flush. ``commit()`` is idempotent; calling it twice
    in a row simply rewrites the same file.
    """

    name = "recipe"

    def __init__(self, session):
        super().__init__(session)
        self._recipe: List[Tuple[str, Dict[str, Any]]] = []
        self._normalized: bool = False
        self._channel_names: List[str] = []
        self._hydrate_from_disk()

    # ── public API ────────────────────────────────────────────────────

    def set_recipe(
        self,
        recipe: List[Tuple[str, Dict[str, Any]]],
        normalized: bool,
        channel_names: List[str],
    ) -> None:
        """Update the in-memory recipe to be persisted on next commit."""
        self._recipe = [(n, dict(p)) for (n, p) in recipe]
        self._normalized = bool(normalized)
        self._channel_names = list(channel_names)

    def get_recipe(self) -> Tuple[List[Tuple[str, Dict[str, Any]]], bool, List[str]]:
        """Return ``(recipe, normalized, channel_names)`` as last committed/set."""
        return (
            [(n, dict(p)) for (n, p) in self._recipe],
            self._normalized,
            list(self._channel_names),
        )

    def commit(self) -> StageRecord:
        path = self.session.stage_path(_RECIPE_RELATIVE)
        payload = {
            "recipe": [{"name": n, "params": p} for (n, p) in self._recipe],
            "normalized": self._normalized,
            "channel_names": list(self._channel_names),
        }
        path.write_text(json.dumps(payload, indent=2))
        record = StageRecord(
            name=self.name,
            parameters={
                "n_steps": len(self._recipe),
                "normalized": self._normalized,
            },
            artifacts={_RECIPE_ARTIFACT: _RECIPE_RELATIVE},
        )
        self.session.record_stage(record)
        return record

    # ── private ──────────────────────────────────────────────────────

    def _hydrate_from_disk(self) -> None:
        """Populate in-memory state from a prior commit, if any."""
        rec = self.session.get_stage(self.name)
        if rec is None:
            return
        rel = rec.artifacts.get(_RECIPE_ARTIFACT, _RECIPE_RELATIVE)
        path = self.session.session_dir / rel
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        self._recipe = [
            (item["name"], dict(item.get("params", {})))
            for item in data.get("recipe", [])
        ]
        self._normalized = bool(data.get("normalized", False))
        self._channel_names = list(data.get("channel_names", []))


class EnhancedDataset:
    """Lazy view: raw channels with the recipe re-applied per-channel.

    Used after the Recipe page releases ``exp._processed_channels`` to
    free RAM. Calling :meth:`materialize_channel` runs the same plugin
    chain that :class:`~nd2studios.workers.recipe_worker.RecipeWorker`
    would, on a single channel at a time, so peak RAM is one channel
    rather than ``n_channels``.

    The wrapper is duck-typed to look enough like a
    ``Dict[channel_name, ndarray]`` for the Export / Analysis / Results
    pages: ``__getitem__`` and ``__contains__`` are implemented and
    materialize on access.
    """

    def __init__(
        self,
        raw_channels: Dict[str, Any],
        recipe: List[Tuple[str, Dict[str, Any]]],
        normalized: bool,
        pixel_size_um: float = 1.0,
        recipe_by_channel: Optional[Dict[str, List[Tuple[str, Dict[str, Any]]]]] = None,
    ):
        self._raw = raw_channels
        self._recipe = [(n, dict(p)) for (n, p) in recipe]
        self._normalized = bool(normalized)
        # V1.48: per-channel recipes (channel-wire pipeline). When set, each
        # channel uses its own recipe; a channel absent / empty ⇒ raw (unwired
        # channels stay raw). Overrides the single ``recipe``.
        self._recipe_by_channel: Optional[Dict[str, List[Tuple[str, Dict[str, Any]]]]] = (
            {k: [(n, dict(p)) for (n, p) in v] for k, v in recipe_by_channel.items()}
            if recipe_by_channel is not None else None
        )
        self._materialized: Dict[str, np.ndarray] = {}
        # Addendum Phase 6: ``materialize_all`` now fans out across
        # channels via :class:`ThreadPoolExecutor`. The recipe plugins
        # (skimage / opencv / scipy) release the GIL so the speed-up
        # is real, but ``materialize_channel`` reads-then-writes
        # ``self._materialized`` — a non-atomic operation that needs
        # a lock so two threads on the same channel don't both
        # recompute and race the cache write.
        self._materialized_lock = threading.Lock()
        self.pixel_size_um = float(pixel_size_um)

    # ── dict-like surface ────────────────────────────────────────────

    def __contains__(self, name: str) -> bool:
        return name in self._raw

    def __iter__(self):
        return iter(self._raw)

    def keys(self):
        return self._raw.keys()

    def __len__(self) -> int:
        return len(self._raw)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.materialize_channel(name)

    # ── materialization ──────────────────────────────────────────────

    def _recipe_for(self, name: str) -> List[Tuple[str, Dict[str, Any]]]:
        """The recipe for channel ``name`` — its per-channel recipe when set
        (absent ⇒ raw), else the single recipe (legacy)."""
        if self._recipe_by_channel is not None:
            return self._recipe_by_channel.get(name) or []
        return self._recipe

    def materialize_channel(
        self,
        name: str,
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> np.ndarray:
        """Apply the recipe to one channel, caching the result.

        The cache is keyed by channel name so repeated reads from the
        same page (Export looping channels, Results computing
        measurements) skip the work.
        """
        with self._materialized_lock:
            cached = self._materialized.get(name)
        if cached is not None:
            return cached

        raw = self._raw.get(name)
        if raw is None:
            raise KeyError(name)

        # Mirror RecipeWorker: materialize the lazy proxy first,
        # then run the recipe.
        if hasattr(raw, "materialize") and callable(raw.materialize):
            current = raw.materialize()
        else:
            current = np.asarray(raw)

        # V1.48: per-channel recipe. An unwired channel (no recipe) stays fully
        # raw — no normalize, no steps.
        recipe = self._recipe_for(name)

        if recipe and self._normalized:
            try:
                from nd2studios.backend.normalization import normalize_timeseries

                current = normalize_timeseries(current)
            except Exception:
                # Best-effort, like RecipeWorker.
                pass

        n_steps = max(1, len(recipe))
        for idx, (plugin_name, params) in enumerate(recipe):
            plugin_cls = PluginBase.get_plugin("enhancement", plugin_name)
            if plugin_cls is None:
                continue
            plugin = plugin_cls()
            current = plugin.execute(current, params, progress_cb=None)
            if progress_cb is not None:
                progress_cb(int((idx + 1) / n_steps * 100))

        with self._materialized_lock:
            # Another concurrent caller may have populated the cache
            # while we were computing — return the cached value so we
            # don't waste the duplicate work on the next read either.
            existing = self._materialized.get(name)
            if existing is not None:
                return existing
            self._materialized[name] = current
        return current

    def materialize_all(
        self,
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, np.ndarray]:
        """Run the recipe over every channel.

        Restores the same shape :attr:`ND2StudiosRecord._processed_channels`
        held before the release hook fired.

        Addendum Phase 6: channels are independent — different keys,
        different output buffers — so we fan out across them with a
        :class:`ThreadPoolExecutor`. The recipe plugins (skimage /
        opencv / scipy) release the GIL during their hot loops, so a
        4-channel volume on a 4-core machine returns ~4× faster than
        the V1.38 sequential walk. Single-channel recipes pay zero
        extra cost (the pool degenerates to one task).
        """
        names = list(self._raw.keys())
        if not names:
            return {}

        out: Dict[str, np.ndarray] = {}
        # One worker per channel, capped at the host's recommended
        # CPU pool size. Going beyond N=channel-count buys nothing
        # because each task is already a whole channel.
        n_workers = max(1, min(recommended_worker_count(), len(names)))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(self.materialize_channel, name): name
                       for name in names}
            done = 0
            for fut in as_completed(futures):
                name = futures[fut]
                # Surface plugin errors on the calling thread so the
                # release-and-rematerialize path matches the V1.38
                # sequential behavior.
                out[name] = fut.result()
                done += 1
                if progress_cb is not None:
                    progress_cb(int(done / len(names) * 100))
        return out

    def clear_cache(self) -> None:
        """Forget any per-channel results — next read will recompute."""
        with self._materialized_lock:
            self._materialized.clear()
