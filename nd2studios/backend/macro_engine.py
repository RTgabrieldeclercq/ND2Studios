"""
Macro engine (V1.0).

Pure-Python (no Qt) model for recording, storing, and replaying sequences of
user actions.  Pages and the main window call ``MacroRecorder.record()``; the
Macro dialog reads back the resulting list to display, reorder, and replay.

File format: .nd2s_macro.json
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MACRO_EXTENSION = ".nd2s_macro.json"
_FORMAT_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------
# recipe_add_step   params: plugin, plugin_params, normalized
# recipe_remove_last params: (none)
# recipe_clear      params: (none)
# analysis_run      params: pipeline, pipeline_params
# export_tiff       params: bit_depth
# export_composite  params: bit_depth
# export_movie      params: fps, format, scale_bar, timestamp, channel_labels


@dataclass
class MacroAction:
    action_type: str
    label: str
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MacroAction":
        return MacroAction(
            action_type=d["action_type"],
            label=d["label"],
            params=d.get("params", {}),
            enabled=d.get("enabled", True),
            timestamp=d.get("timestamp", 0.0),
        )


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class MacroRecorder:
    """Stateful recorder.  Call start() → record() × N → finish() or cancel()."""

    def __init__(self) -> None:
        self._recording: bool = False
        self._paused: bool = False
        self._replaying: bool = False
        self._actions: List[MacroAction] = []

    # -- public state ---

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def actions(self) -> List[MacroAction]:
        return list(self._actions)

    # -- lifecycle ---

    def start(self) -> None:
        self._recording = True
        self._paused = False
        self._actions = []

    def pause(self) -> None:
        if self._recording:
            self._paused = True

    def resume(self) -> None:
        if self._recording:
            self._paused = False

    def cancel(self) -> None:
        self._recording = False
        self._paused = False
        self._actions = []

    def finish(self) -> List[MacroAction]:
        self._recording = False
        self._paused = False
        result = list(self._actions)
        self._actions = []
        return result

    # -- replay flag (suppresses all recording during replay) ---

    @property
    def replaying(self) -> bool:
        return self._replaying

    def start_replay(self) -> None:
        self._replaying = True

    def stop_replay(self) -> None:
        self._replaying = False

    # -- recording ---

    def record(self, action: MacroAction) -> bool:
        """Append action if currently recording and not paused.  Returns True if added."""
        if self._replaying:
            return False
        if self._recording and not self._paused:
            self._actions.append(action)
            return True
        return False

    def record_upgrade(self, action: MacroAction) -> bool:
        """Record a rich action, replacing the immediately preceding generic action if any.

        When the event filter records a generic ``button_click`` and then a
        semantic hook fires within 300 ms for the same button press, calling
        this method swaps the placeholder for the richer record so the live
        list and saved macro both show the meaningful label.
        """
        if self._replaying:
            return False
        if not (self._recording and not self._paused):
            return False
        _GENERIC = {"button_click", "combo_change", "spinbox_change", "checkbox_change"}
        if (self._actions
                and self._actions[-1].action_type in _GENERIC
                and (action.timestamp - self._actions[-1].timestamp) < 0.5):
            self._actions[-1] = action
        else:
            self._actions.append(action)
        return True


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_macro(name: str, actions: List[MacroAction], path: str) -> str:
    """Serialize *actions* to *path*.  Appends MACRO_EXTENSION if absent.
    Returns the final path used."""
    p = Path(path)
    if not str(p).endswith(MACRO_EXTENSION):
        p = Path(str(p) + MACRO_EXTENSION)
    payload = {
        "version": _FORMAT_VERSION,
        "name": name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "actions": [a.to_dict() for a in actions],
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(p)


def load_macro(path: str) -> Tuple[str, List[MacroAction]]:
    """Load a .nd2s_macro.json file.  Returns (name, actions)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    name: str = data.get("name", Path(path).stem)
    actions = [MacroAction.from_dict(d) for d in data.get("actions", [])]
    return name, actions


def list_macros(directory: str) -> List[Tuple[str, str]]:
    """Return [(name, path), …] for all macro files in *directory*."""
    result: List[Tuple[str, str]] = []
    for p in sorted(Path(directory).glob(f"*{MACRO_EXTENSION}")):
        try:
            name, _ = load_macro(str(p))
            result.append((name, str(p)))
        except Exception:
            result.append((p.stem, str(p)))
    return result
