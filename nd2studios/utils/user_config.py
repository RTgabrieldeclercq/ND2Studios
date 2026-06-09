"""Per-user persistent preferences (V1.41).

Stores the small set of ``Settings`` fields that the Performance
dialog can flip — GPU analysis opt-in, pyramid build flag, forced
load strategy, eager-budget fraction — to a JSON file under

* Windows: ``%APPDATA%\\nd2studios\\preferences.json``
* Linux / macOS: ``~/.config/nd2studios/preferences.json``

so the user's choices survive across launches.  ``Settings`` itself
stays as the source of truth at runtime; this module just round-trips
the values to disk.

The on-disk schema is intentionally a flat dict so a user can hand-edit
it.  Unknown keys are ignored on load; missing keys keep the
hard-coded :class:`Settings` defaults.  Schema migrations would add a
``_version`` field — we don't have one today because the schema is
small enough that backward-compatibility is achieved by additive
extension.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

log = logging.getLogger(__name__)


_TRACKED_FIELDS = (
    "USE_GPU_DISPLAY",
    "USE_GPU_ANALYSIS",
    "BUILD_PYRAMIDS",
    "EAGER_MAX_FRACTION",
    "MEMORY_RESERVE_OVERHEAD",
    "MEMORY_PRESSURE_WARNING_PCT",
    "MEMORY_PRESSURE_CRITICAL_PCT",
    "MEMORY_PRESSURE_EMERGENCY_PCT",
    "FORCED_LOAD_STRATEGY",
)


def _config_path() -> Path:
    """Return the on-disk preferences file location."""
    if os.name == "nt":
        base = os.environ.get("APPDATA", str(Path.home()))
        return Path(base) / "nd2studios" / "preferences.json"
    return Path.home() / ".config" / "nd2studios" / "preferences.json"


def _read_json() -> Dict[str, Any]:
    p = _config_path()
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("preferences read failed (%s): %s", p, exc)
    return {}


def _write_json(data: Dict[str, Any]) -> None:
    p = _config_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        log.warning("preferences write failed (%s): %s", p, exc)


def load_user_preferences() -> Dict[str, Any]:
    """Read preferences from disk and apply them to :class:`Settings`.

    Returns the loaded dict so callers can inspect what was applied;
    the side effect on ``Settings`` is the primary use.  Called once
    from :func:`nd2studios.__main__.main` after Settings is imported
    but before any UI is constructed.
    """
    from nd2studios.core.settings import Settings  # local to avoid cycle

    data = _read_json()
    if not data:
        return {}
    applied: Dict[str, Any] = {}
    for key in _TRACKED_FIELDS:
        if key in data:
            try:
                setattr(Settings, key, data[key])
                applied[key] = data[key]
            except Exception as exc:  # noqa: BLE001
                log.debug("could not apply preference %s=%r: %s", key, data[key], exc)
    if applied:
        log.info("Loaded %d user preferences from %s", len(applied), _config_path())
    return applied


def save_user_preferences() -> Path:
    """Snapshot the tracked :class:`Settings` fields to disk.

    Returns the path written so the GUI can show it in a status
    message.  Best-effort: a write failure is logged but does not
    raise — the user's runtime state is unaffected and they can try
    again from the Performance dialog.
    """
    from nd2studios.core.settings import Settings  # local to avoid cycle

    snapshot: Dict[str, Any] = {}
    for key in _TRACKED_FIELDS:
        if hasattr(Settings, key):
            snapshot[key] = getattr(Settings, key)
    _write_json(snapshot)
    log.info("Saved %d user preferences to %s", len(snapshot), _config_path())
    return _config_path()
