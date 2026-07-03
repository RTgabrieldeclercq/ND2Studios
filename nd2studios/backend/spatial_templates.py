"""Spatial-map template library — save/load named Spatial Maps configurations.

A *template* is a JSON-serializable dict of the Spatial Maps tab's sidebar state
(field, grid, scale, display, scale-bar … — whatever ``SpatialMapsPanel.get_config``
emits). Templates persist to a global, app-wide library under
``~/.nd2studios/spatial_map_templates`` (one JSON file per template), so a
configuration built once is reusable across pipelines and sessions and can be
referenced by **name** from a Spatial Maps node.

Portability: a pipeline saved on one machine references templates by name; on a
machine whose library lacks them, :func:`import_file` merges an exported JSON
(single template or a bundle of many) into the local library so the names
resolve again. Pure stdlib — no Qt, no numpy — so the backend-purity rule holds.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

TEMPLATE_VERSION = 1
TEMPLATE_KIND = "nd2studios.spatialmap"
BUNDLE_KIND = "nd2studios.spatialmap.bundle"
TEMPLATE_EXTENSION = ".nd2s_spatialmap.json"


def templates_dir() -> Path:
    """The on-disk template library directory (created on demand)."""
    d = Path.home() / ".nd2studios" / "spatial_map_templates"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return d


def _slug(name: str) -> str:
    """Filesystem-safe stem for a display name (collisions resolved by the
    stored ``name`` field, not the filename)."""
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", str(name).strip()) or "template"
    return s[:120]


def _read_one(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Template file is not a JSON object.")
    return data


def list_templates() -> List[str]:
    """Sorted display names of every template in the library."""
    names: List[str] = []
    d = templates_dir()
    if not d.exists():
        return names
    for path in d.glob("*" + TEMPLATE_EXTENSION):
        try:
            data = _read_one(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        name = str(data.get("name") or path.name[: -len(TEMPLATE_EXTENSION)])
        names.append(name)
    return sorted(names, key=str.lower)


def _find_path(name: str) -> Path | None:
    """The library file whose stored ``name`` matches ``name`` (else None)."""
    d = templates_dir()
    if not d.exists():
        return None
    for path in d.glob("*" + TEMPLATE_EXTENSION):
        try:
            data = _read_one(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if str(data.get("name") or "") == str(name):
            return path
    return None


def load_template(name: str) -> Dict[str, Any]:
    """Return the ``config`` dict for the named template.

    Raises :class:`KeyError` when no template with that name exists.
    """
    path = _find_path(name)
    if path is None:
        raise KeyError(f"No spatial-map template named {name!r} in the library.")
    data = _read_one(path)
    cfg = data.get("config")
    return dict(cfg) if isinstance(cfg, dict) else {}


def save_template(name: str, config: Dict[str, Any]) -> Path:
    """Write (or overwrite) a named template; returns the file path."""
    name = str(name).strip()
    if not name:
        raise ValueError("Template name cannot be empty.")
    path = _find_path(name) or (templates_dir() / (_slug(name) + TEMPLATE_EXTENSION))
    payload = {
        "version": TEMPLATE_VERSION,
        "kind": TEMPLATE_KIND,
        "name": name,
        "config": dict(config or {}),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def delete_template(name: str) -> bool:
    """Remove a named template; returns True if one was deleted."""
    path = _find_path(name)
    if path is None:
        return False
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def read_template_file(path: str) -> Dict[str, Dict[str, Any]]:
    """Parse an external JSON file into ``{name: config}`` without importing it.

    Accepts both a single-template file (``kind == TEMPLATE_KIND``) and a bundle
    (``kind == BUNDLE_KIND``). Raises :class:`ValueError` for unrecognised files.
    """
    data = _read_one(Path(path))
    kind = data.get("kind")
    out: Dict[str, Dict[str, Any]] = {}
    if kind == BUNDLE_KIND:
        templates = data.get("templates")
        if not isinstance(templates, dict):
            raise ValueError("Bundle file has no 'templates' object.")
        for name, cfg in templates.items():
            if isinstance(cfg, dict):
                out[str(name)] = dict(cfg)
    elif kind == TEMPLATE_KIND:
        name = str(data.get("name") or Path(path).stem)
        cfg = data.get("config")
        out[name] = dict(cfg) if isinstance(cfg, dict) else {}
    else:
        raise ValueError("File is not an ND2Studios spatial-map template.")
    if not out:
        raise ValueError("No templates found in file.")
    return out


def import_file(path: str) -> List[str]:
    """Merge an external template file into the local library.

    Returns the list of imported display names. Existing names are overwritten.
    """
    parsed = read_template_file(path)
    for name, cfg in parsed.items():
        save_template(name, cfg)
    return sorted(parsed.keys(), key=str.lower)


def export_file(names: List[str], path: str) -> List[str]:
    """Write the named templates to ``path`` as a portable bundle.

    Unknown names are skipped; returns the names actually exported.
    """
    bundle: Dict[str, Dict[str, Any]] = {}
    for name in names:
        try:
            bundle[name] = load_template(name)
        except KeyError:
            continue
    payload = {
        "version": TEMPLATE_VERSION,
        "kind": BUNDLE_KIND,
        "templates": bundle,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return sorted(bundle.keys(), key=str.lower)
