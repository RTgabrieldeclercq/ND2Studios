#!/usr/bin/env python3
"""
ND2Studios launcher.

Runs the GUI with the project root on PYTHONPATH so the `nd2studios`
package imports work regardless of where this script is invoked from.

It also makes sure the app runs on a Python interpreter that has the full
dependency stack. Some nodes — notably StarDist Segmentation, which needs
TensorFlow — only work on a Python version TensorFlow ships wheels for
(e.g. TF has no wheels for Python 3.14). If the interpreter used to launch
this script is missing those optional deps but another installed interpreter
has them, run.py re-launches itself under that better interpreter, so
`py run.py` (which may default to a TF-less 3.14) still lands on a working
Python.

Override the auto-pick by setting ND2STUDIOS_PYTHON to a python.exe path.

Usage:
    python run.py
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent

# Modules that must import for the GUI to launch at all.
_REQUIRED = ("PySide6", "numpy", "nd2", "cv2")
# Optional-but-preferred stack. An interpreter that has these can run every
# node (StarDist needs tensorflow + stardist + csbdeep). We prefer an
# interpreter that satisfies more of these.
_PREFERRED = ("tensorflow", "stardist", "csbdeep")

# Guard against a re-exec loop: the child is launched with this set.
_REEXEC_FLAG = "ND2STUDIOS_REEXEC"


def _has(module: str) -> bool:
    """True if `module` is importable in *this* interpreter (no import cost)."""
    import importlib.util
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:  # noqa: BLE001 — a broken/partial install counts as absent
        return False


def _score(has_required: bool, n_preferred: int) -> Tuple[int, int]:
    """Rank key: (can-launch, #preferred-present). Higher is better."""
    return (1 if has_required else 0, n_preferred)


def _current_score() -> Tuple[int, int]:
    return _score(all(_has(m) for m in _REQUIRED),
                  sum(_has(m) for m in _PREFERRED))


def _probe(exe: str) -> Optional[Tuple[int, int]]:
    """Query another interpreter for its (can-launch, #preferred) score.

    Runs a tiny `find_spec` probe in `exe` (find_spec doesn't import the
    modules, so this is fast). Returns None if the interpreter can't be run.
    """
    code = (
        "import importlib.util as u\n"
        "def h(m):\n"
        " try: return u.find_spec(m) is not None\n"
        " except Exception: return False\n"
        f"req={_REQUIRED!r}\npref={_PREFERRED!r}\n"
        "print(int(all(h(m) for m in req)), sum(h(m) for m in pref))\n"
    )
    try:
        out = subprocess.run(
            [exe, "-c", code], capture_output=True, text=True, timeout=30
        )
    except Exception:  # noqa: BLE001
        return None
    if out.returncode != 0:
        return None
    try:
        r, n = out.stdout.split()
        return _score(bool(int(r)), int(n))
    except Exception:  # noqa: BLE001
        return None


def _candidate_interpreters() -> List[str]:
    """Other python.exe paths worth probing, most-explicit first.

    ND2STUDIOS_PYTHON override → the `py -0p` registered list → PATH.
    De-duplicated, excluding the current interpreter.
    """
    seen = set()
    out: List[str] = []

    def add(path: Optional[str]) -> None:
        if not path:
            return
        p = os.path.normcase(os.path.abspath(path))
        if p in seen or p == os.path.normcase(os.path.abspath(sys.executable)):
            return
        if os.path.exists(path):
            seen.add(p)
            out.append(path)

    add(os.environ.get("ND2STUDIOS_PYTHON"))

    # Windows `py` launcher: `py -0p` lists every registered interpreter's path.
    # Paths can contain spaces, so grab everything after the version/`*` columns.
    launcher = shutil.which("py")
    if launcher:
        try:
            listing = subprocess.run(
                [launcher, "-0p"], capture_output=True, text=True, timeout=15
            )
            for line in listing.stdout.splitlines():
                m = re.match(r"\s*-V:\S+\s+(?:\*\s+)?(.+\S)\s*$", line)
                if m:
                    add(m.group(1))
        except Exception:  # noqa: BLE001
            pass

    for name in ("python", "python3"):
        add(shutil.which(name))
    return out


def _maybe_reexec() -> None:
    """If a better interpreter than the current one exists, re-launch there.

    "Better" = can still launch the GUI *and* has more of the preferred stack
    (so a run started on a TF-less Python 3.14 hops to a Python that has
    TensorFlow/StarDist). No-ops when the current interpreter is already best
    or a re-exec already happened.
    """
    if os.environ.get(_REEXEC_FLAG):
        return  # already re-exec'd once — don't loop

    cur = _current_score()
    if cur[0] and cur[1] == len(_PREFERRED):
        return  # current interpreter has everything — fast path, no probing

    best_exe: Optional[str] = None
    best = cur
    for exe in _candidate_interpreters():
        sc = _probe(exe)
        if sc is not None and sc > best:  # strictly better (tuple compare)
            best, best_exe = sc, exe

    if best_exe is None:
        return  # nothing better available — fall through and run here

    env = dict(os.environ)
    env[_REEXEC_FLAG] = "1"
    # Ensure the child can import the package even if PYTHONPATH was unset.
    existing = env.get("PYTHONPATH", "")
    parts = [str(PROJECT_ROOT)] + ([existing] if existing else [])
    env["PYTHONPATH"] = os.pathsep.join(parts)

    print(
        f"[ND2Studios] Re-launching under {best_exe} "
        f"(current interpreter is missing part of the dependency stack).",
        flush=True,
    )
    proc = subprocess.run([best_exe, str(Path(__file__).resolve()), *sys.argv[1:]],
                          env=env)
    sys.exit(proc.returncode)


def _launch() -> None:
    # Make sure the project root (the directory containing this file) is on
    # sys.path. Without this, `python path/to/run.py` from another working
    # directory fails to import `nd2studios`.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from nd2studios.__main__ import main
    main()


if __name__ == "__main__":
    _maybe_reexec()
    _launch()
