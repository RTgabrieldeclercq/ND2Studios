"""Run every profiling scenario and emit one JSON snapshot.

Each subsequent optimization phase should re-run this aggregator and
write a new ``profiling/baselines/<label>.json`` file so we can diff
the numbers and confirm regressions / improvements.

Usage
-----
::

    # default label = phase_00_baseline
    python -m profiling.harness.run_all

    # custom label per phase
    python -m profiling.harness.run_all phase_02_lazy_loading
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import psutil


BASELINES_DIR: Path = Path(__file__).resolve().parent.parent / "baselines"


def _system_info() -> Dict[str, Any]:
    """Snapshot of the machine the baseline was captured on."""
    vm = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_full": sys.version,
        "executable": sys.executable,
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(vm.total / (1024 ** 3), 2),
        "available_ram_gb": round(vm.available / (1024 ** 3), 2),
    }


def _run_scenario(name: str):
    """Import a scenario module lazily and call its ``run()`` entry point.

    Lazy import insulates the aggregator from a single scenario's
    optional-dep failure (e.g. PySide6 missing kills tab-switch only).
    """
    try:
        mod = __import__(f"profiling.harness.scenario_{name}",
                         fromlist=["run"])
        return mod.run()
    except Exception as exc:  # noqa: BLE001
        return [{
            "name": f"{name}_scenario",
            "error": f"{type(exc).__name__}: {exc}",
        }]


def main(label: str = "phase_00_baseline") -> Path:
    """Run all scenarios; write a single JSON snapshot. Return its path."""
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    snapshot: Dict[str, Any] = {
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": _system_info(),
        "scenarios": {
            "load":         _run_scenario("load"),
            "scrub":        _run_scenario("scrub"),
            "tab_switch":   _run_scenario("tab_switch"),
            "analysis":     _run_scenario("analysis"),
        },
    }

    out = BASELINES_DIR / f"{label}.json"
    out.write_text(json.dumps(snapshot, indent=2, default=str))
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "phase_00_baseline"
    main(label)
