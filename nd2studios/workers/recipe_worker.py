"""
RecipeWorker — apply an ordered list of enhancement plugins to a
multi-channel timeseries in a background thread.

Returns:
    {channel_name: (T, H, W) np.ndarray}
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from nd2studios.core.plugin_registry import PluginBase
from nd2studios.utils.progress import FrameProgress
from nd2studios.workers.base_worker import BaseWorker


class RecipeWorker(BaseWorker):
    """Apply a recipe (ordered (plugin_name, params) pairs) to each channel."""

    def __init__(
        self,
        channels: Dict[str, Any],
        recipe: List[Tuple[str, Dict[str, Any]]],
        normalized: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.channels = channels
        self.recipe = recipe
        self.normalized = normalized

    def run_task(self) -> Dict[str, np.ndarray]:
        from nd2studios.backend.normalization import normalize_timeseries

        results: Dict[str, np.ndarray] = {}
        n_channels = len(self.channels)
        n_steps = len(self.recipe)

        # Frame-accurate progress (V1.41). Each channel costs, in
        # frame-equivalents: T frames to read + T per normalization pass
        # + T per recipe step (a step processes the whole (T,H,W) stack at
        # once, so we credit it a full T-block when it finishes). Reading
        # advances one frame at a time so the bar tracks slow disk I/O.
        first = next(iter(self.channels.values()), None)
        n_t = 1
        shape = getattr(first, "shape", None)
        if shape is not None and len(shape) >= 1:
            try:
                n_t = max(1, int(shape[0]))
            except Exception:
                n_t = 1
        passes_per_ch = 1 + (1 if self.normalized else 0) + n_steps
        total_frames = max(1, n_channels * n_t * passes_per_ch)
        fp = FrameProgress(total_frames, self.set_progress)

        for ch_idx, (ch_name, data) in enumerate(self.channels.items()):
            if self.cancelled:
                return results

            # Lazy proxies must be materialized for plugins that mutate.
            # Stream frame-by-frame so the GUI shows progress and the
            # Cancel button can interrupt long reads from disk. Falls
            # back to one-shot ``data.materialize()`` / ``np.asarray()``
            # when ``data`` isn't indexable per-frame.
            current = self._materialize_with_progress(ch_name, data, fp, n_t)
            if current is None:  # cancelled mid-load
                return results

            if self.normalized:
                self.set_status(f"Channel {ch_name}: frame-mean normalization")
                try:
                    current = normalize_timeseries(current)
                except Exception:
                    # Normalization is best-effort; don't fail the whole recipe.
                    pass
                fp.advance(n_t)

            for step_idx, (plugin_name, params) in enumerate(self.recipe):
                if self.cancelled:
                    return results
                self.set_status(f"Channel {ch_name}: {plugin_name}")
                plugin_cls = PluginBase.get_plugin("enhancement", plugin_name)
                if plugin_cls is None:
                    # Skip unknown plugins; never crash the worker.
                    fp.advance(n_t)
                    continue
                plugin = plugin_cls()
                current = plugin.execute(current, params, progress_cb=None)
                fp.advance(n_t)

            results[ch_name] = current

        fp.finish()
        return results

    def _materialize_with_progress(
        self, ch_name: str, data, fp: FrameProgress, n_t: int,
    ) -> "np.ndarray | None":
        """Read a lazy proxy frame-by-frame so the GUI shows progress.

        Reports per-frame status updates so users can distinguish a
        slow disk read from a hang. Advances ``fp`` one frame at a time as
        it reads. Returns ``None`` if the worker was cancelled mid-load.
        """
        # Per-frame path: data has a usable ``shape[0]`` and supports
        # ``data[t]`` indexing. Covers LazyND2Channel,
        # MultiFileLazyChannel, LazyTIFFChannel, and plain ndarrays.
        shape = getattr(data, "shape", None)
        if (shape is not None and len(shape) >= 1
                and hasattr(data, "__getitem__")):
            try:
                n_read = int(shape[0])
            except Exception:
                n_read = 0
            if n_read > 0:
                size_mb = 0
                try:
                    if len(shape) >= 3:
                        size_mb = (n_read * int(shape[-2]) * int(shape[-1])
                                   * np.dtype(getattr(data, "dtype",
                                                       np.uint16)).itemsize
                                   ) // (1024 * 1024)
                except Exception:
                    size_mb = 0
                hint = f" (~{size_mb} MiB)" if size_mb else ""
                self.set_status(
                    f"Channel {ch_name}: reading {n_read} frames{hint}…"
                )
                frames = []
                for t in range(n_read):
                    if self.cancelled:
                        return None
                    try:
                        f = np.asarray(data[t])
                    except Exception:
                        # If indexing fails fall back to one-shot below.
                        frames = None
                        break
                    if f.ndim > 2:
                        f = f.squeeze()
                    frames.append(f)
                    fp.advance()
                    if t % 4 == 0 or t == n_read - 1:
                        self.set_status(
                            f"Channel {ch_name}: read {t + 1}/{n_read}"
                        )
                if frames:
                    try:
                        return np.stack(frames, axis=0)
                    except Exception:
                        # If frames have inconsistent shape (shouldn't
                        # happen with our lazy proxies), fall through.
                        pass

        # One-shot fallback for proxies that can't be frame-indexed. We
        # couldn't advance per frame, so credit the whole read block now to
        # keep the accountant aligned with ``passes_per_ch``.
        self.set_status(f"Channel {ch_name}: loading frames…")
        fp.advance(n_t)
        if hasattr(data, "materialize") and callable(getattr(data, "materialize")):
            return data.materialize()
        return np.asarray(data).copy()
