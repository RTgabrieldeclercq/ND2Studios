"""
Per-source workspace and manifest (V1.38 Phase 6).

A :class:`Session` owns one workspace directory per source file. The
directory key is the first 16 hex chars of
``sha256(size || mtime || first_MiB || last_MiB)``: cheap to compute on
multi-GB files and stable across renames or moves to a different folder
on the same disk.

Layout::

    <workspace_root>/sessions/<source_hash>/
    ├── manifest.json
    ├── recipe/recipe.json
    └── analysis/<pipeline>/m_NN/labels_<channel>.{zarr,npz}

The workspace is *automatic* and lives next to the OS-user cache by
default. It is distinct from the user-saved ``.nd2s`` bundle handled by
:class:`~nd2studios.core.experiment_manager.ND2StudiosManager`.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# Setting this env var to "1" turns the workspace into a no-op (rollback
# escape hatch — see V1.38 plan). Touchpoints in main_window.py,
# import_page.py, recipe_page.py, and analysis_page.py honour it.
DISABLE_WORKSPACE_ENV = "ND2STUDIOS_DISABLE_WORKSPACE"
WORKSPACE_ENV = "ND2STUDIOS_WORKSPACE"

_MANIFEST_NAME = "manifest.json"
_HASH_CHUNK = 1024 * 1024  # 1 MiB


def default_workspace_root() -> Path:
    """Return the workspace root directory, honouring the env var override.

    Defaults to ``~/.nd2studios/workspace``. Override with
    ``ND2STUDIOS_WORKSPACE`` (useful for the profiling harness and tests).
    """
    override = os.environ.get(WORKSPACE_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".nd2studios" / "workspace"


def workspace_disabled() -> bool:
    """Return True when the rollback env var asks Phase 6 to no-op."""
    return os.environ.get(DISABLE_WORKSPACE_ENV, "") == "1"


def hash_source_file(path: str | Path) -> str:
    """Fingerprint a source file: size + mtime + first/last MiB content.

    The first and last MiBs are read regardless of file size; for files
    smaller than 2 MiB we just read the whole content, which is still
    cheap. The hash is truncated to 16 hex chars (64 bits) — plenty of
    entropy for keying a single user's workspace.
    """
    p = Path(path)
    st = p.stat()
    h = hashlib.sha256()
    h.update(str(st.st_size).encode())
    h.update(str(int(st.st_mtime)).encode())
    with open(p, "rb") as f:
        if st.st_size <= 2 * _HASH_CHUNK:
            h.update(f.read())
        else:
            h.update(f.read(_HASH_CHUNK))
            f.seek(-_HASH_CHUNK, os.SEEK_END)
            h.update(f.read(_HASH_CHUNK))
    return h.hexdigest()[:16]


@dataclass
class StageRecord:
    """One row in :class:`SessionManifest.stages`.

    ``name`` is the stage key — ``"recipe"`` or
    ``"analysis:<pipeline_name>"``. ``artifacts`` maps a logical name
    (``"recipe"``, ``"labels_DAPI_m_00"``, ...) to a session-relative
    path so consumers do not have to know the on-disk layout.
    """

    name: str
    committed_at: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StageRecord":
        return cls(
            name=d.get("name", ""),
            committed_at=d.get("committed_at"),
            parameters=dict(d.get("parameters", {})),
            artifacts=dict(d.get("artifacts", {})),
        )


@dataclass
class SessionManifest:
    """JSON-only manifest for one source file's workspace.

    Snapshots the source-file shape so consumers can be sanity-checked
    on rehydrate (e.g. refuse to read prior label masks if the user
    re-opened a file whose ``n_t`` changed).
    """

    session_id: str
    source_path: str
    source_hash: str
    created_at: str
    n_t: int = 0
    n_m: int = 1
    n_z: int = 1
    n_channels: int = 0
    height: int = 0
    width: int = 0
    pixel_size_um: float = 1.0
    channel_names: List[str] = field(default_factory=list)
    stages: Dict[str, StageRecord] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "source_path": self.source_path,
            "source_hash": self.source_hash,
            "created_at": self.created_at,
            "n_t": self.n_t,
            "n_m": self.n_m,
            "n_z": self.n_z,
            "n_channels": self.n_channels,
            "height": self.height,
            "width": self.width,
            "pixel_size_um": self.pixel_size_um,
            "channel_names": list(self.channel_names),
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionManifest":
        stages = {
            k: StageRecord.from_dict(v) for k, v in d.get("stages", {}).items()
        }
        return cls(
            session_id=d.get("session_id", ""),
            source_path=d.get("source_path", ""),
            source_hash=d.get("source_hash", ""),
            created_at=d.get("created_at", ""),
            n_t=int(d.get("n_t", 0)),
            n_m=int(d.get("n_m", 1)),
            n_z=int(d.get("n_z", 1)),
            n_channels=int(d.get("n_channels", 0)),
            height=int(d.get("height", 0)),
            width=int(d.get("width", 0)),
            pixel_size_um=float(d.get("pixel_size_um", 1.0)),
            channel_names=list(d.get("channel_names", [])),
            stages=stages,
        )


class Session:
    """Owns one on-disk workspace directory keyed by source-file hash.

    Construction is cheap: hash the source, ensure the directory exists,
    load any prior manifest. Heavy I/O happens only inside stage
    ``commit()`` / ``rehydrate()`` calls.
    """

    def __init__(
        self,
        source_path: str | Path,
        workspace_root: Optional[str | Path] = None,
    ):
        self.source_path = Path(source_path)
        root = (
            Path(workspace_root).expanduser()
            if workspace_root is not None
            else default_workspace_root()
        )
        self.sessions_root = root / "sessions"
        self.sessions_root.mkdir(parents=True, exist_ok=True)

        self.source_hash = hash_source_file(self.source_path)
        self.session_dir = self.sessions_root / self.source_hash
        self.session_dir.mkdir(exist_ok=True)
        self._manifest_path = self.session_dir / _MANIFEST_NAME

        if self._manifest_path.exists():
            self.manifest = self._load_manifest()
            # The source path might have moved since the last commit —
            # update the recorded path so consumers can find it again.
            self.manifest.source_path = str(self.source_path)
        else:
            self.manifest = SessionManifest(
                session_id=self.source_hash,
                source_path=str(self.source_path),
                source_hash=self.source_hash,
                created_at=_utc_now(),
            )
            self.save_manifest()

    # ── manifest I/O ──────────────────────────────────────────────────

    def _load_manifest(self) -> SessionManifest:
        try:
            data = json.loads(self._manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            # Treat unreadable manifest as a missing one: start fresh.
            return SessionManifest(
                session_id=self.source_hash,
                source_path=str(self.source_path),
                source_hash=self.source_hash,
                created_at=_utc_now(),
            )
        return SessionManifest.from_dict(data)

    def save_manifest(self) -> None:
        self._manifest_path.write_text(
            json.dumps(self.manifest.to_dict(), indent=2)
        )

    def set_shape(
        self,
        *,
        n_t: int,
        n_m: int,
        n_z: int,
        n_channels: int,
        height: int,
        width: int,
        pixel_size_um: float,
        channel_names: List[str],
    ) -> None:
        """Stamp the source file's shape onto the manifest.

        Called from the Import page after a successful load so the
        manifest tells future readers what to expect. A subsequent
        attach for the same file overwrites these (mostly identical)
        values — cheap.
        """
        self.manifest.n_t = int(n_t)
        self.manifest.n_m = int(n_m)
        self.manifest.n_z = int(n_z)
        self.manifest.n_channels = int(n_channels)
        self.manifest.height = int(height)
        self.manifest.width = int(width)
        self.manifest.pixel_size_um = float(pixel_size_um)
        self.manifest.channel_names = list(channel_names)
        self.save_manifest()

    # ── stage-record helpers ─────────────────────────────────────────

    def stage_path(self, *parts: str) -> Path:
        """Return ``session_dir / parts...`` and ensure its parent exists."""
        p = self.session_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def get_stage(self, name: str) -> Optional[StageRecord]:
        return self.manifest.stages.get(name)

    def is_committed(self, name: str) -> bool:
        rec = self.manifest.stages.get(name)
        return rec is not None and rec.committed_at is not None

    def record_stage(self, record: StageRecord) -> None:
        """Insert/replace a stage record, stamp committed_at, persist."""
        record.committed_at = _utc_now()
        self.manifest.stages[record.name] = record
        self.save_manifest()

    def remove_stage(self, name: str, *, delete_artifacts: bool = True) -> None:
        """Drop a stage record and (optionally) its artifact files."""
        rec = self.manifest.stages.pop(name, None)
        if rec is not None and delete_artifacts:
            for rel in rec.artifacts.values():
                target = self.session_dir / rel
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target, ignore_errors=True)
                    else:
                        try:
                            target.unlink()
                        except OSError:
                            pass
        self.save_manifest()

    # ── lifecycle ────────────────────────────────────────────────────

    def archive(self, suffix: Optional[str] = None) -> Path:
        """Rename the session dir aside so a fresh one can be created.

        Used by the Import page's "Start fresh" path. Returns the new
        path; callers can surface it in a status message if they want.
        """
        stamp = suffix or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        archived = self.sessions_root / f"{self.source_hash}.archived-{stamp}"
        i = 0
        while archived.exists():
            i += 1
            archived = (
                self.sessions_root / f"{self.source_hash}.archived-{stamp}-{i}"
            )
        # Close any cached handles before renaming on Windows.
        shutil.move(str(self.session_dir), str(archived))
        self.session_dir.mkdir(exist_ok=True)
        self.manifest = SessionManifest(
            session_id=self.source_hash,
            source_path=str(self.source_path),
            source_hash=self.source_hash,
            created_at=_utc_now(),
        )
        self.save_manifest()
        return archived


def _utc_now() -> str:
    """ISO-8601 UTC timestamp with 'Z' suffix."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
