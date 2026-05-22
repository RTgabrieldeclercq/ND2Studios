# Profiling harness — ND2Studios

This directory captures **baseline performance numbers** for ND2Studios
and the matching tooling to re-capture them after each optimization
phase. Plan: [`CodeLog/ClaudesPlan/V1.33_phase1_profiling_baseline.md`](../CodeLog/ClaudesPlan/V1.33_phase1_profiling_baseline.md).

## Layout

```
profiling/
├── README.md                         # you are here
├── harness/
│   ├── fixtures.py                   # test file paths + scrub counts
│   ├── scenario_load.py              # cold open + first frame
│   ├── scenario_scrub.py             # T / Z / M scrubbing latency
│   ├── scenario_tab_switch.py        # offscreen page transitions
│   ├── scenario_analysis.py          # registered AnalysisPipelines (synthetic input)
│   └── run_all.py                    # aggregator → baselines/<label>.json
├── baselines/                        # JSON snapshots, one per phase
└── reports/                          # optional py-spy flamegraphs (SVG)
```

The helper module that records measurements lives in
[`nd2studios/utils/profiling.py`](../nd2studios/utils/profiling.py) so
worker tests and other phases can reuse the same dataclass.

## Quick start

```powershell
# Capture a baseline (no lab data required — analysis runs on synthetic input)
python -m profiling.harness.run_all phase_00_baseline
```

The aggregator writes `profiling/baselines/phase_00_baseline.json`.

Run an individual scenario for ad-hoc inspection:

```powershell
python -m profiling.harness.scenario_tab_switch
python -m profiling.harness.scenario_analysis
```

## Pointing the harness at real lab files

The `load` and `scrub` scenarios skip cleanly when no test files are
present. To exercise them, drop files into a folder and point the
environment variable at it:

```powershell
$env:PROFILING_TEST_DATA_DIR = "D:\profiling_data"
python -m profiling.harness.run_all phase_00_baseline
```

Default filenames the harness looks for inside that folder are listed
in [`harness/fixtures.py`](harness/fixtures.py): `small.nd2`,
`large.nd2`, `small.tif`, `large.tif`. Edit
`fixtures.py` to point at different names without touching the
scenarios.

## What each scenario measures

| Scenario       | Measures                                                                                   |
| -------------- | ------------------------------------------------------------------------------------------ |
| `load`         | `LazyND2Volume(...)` / `LazyMultiFileTIFFVolume(...)` construction + first `get_frame()`   |
| `scrub`        | `get_frame()` per call while walking T / Z / M; reports mean / p50 / p99 FPS               |
| `tab_switch`   | Offscreen `MainWindow._navigate(key)` for each page in `Settings.PAGES`                    |
| `analysis`     | One `AnalysisPipeline.run(...)` per entry in the registry, on a synthetic (T, 256, 256) channel |

## Reading the JSON

Each scenario entry follows the
[`Measurement.to_dict()`](../nd2studios/utils/profiling.py) shape:

```json
{
  "name": "scrub_t_large_nd2",
  "wall_ms": 1284.7,
  "cpu_ms":  1240.1,
  "rss_before_mb": 412.0,
  "rss_after_mb":  468.3,
  "rss_peak_mb":   468.3,
  "py_alloc_peak_mb": 21.4,
  "extra": {
    "axis": "t",
    "filepath": ".../large.nd2",
    "mean_fps": 78.2,
    "p50_fps":  82.0,
    "p99_fps":  41.5,
    "mean_ms":  12.78,
    "p50_ms":   12.20,
    "p99_ms":   24.10,
    "frames":   100
  }
}
```

Failed measurements record an `error` field with the exception type and
message but never raise — the rest of the run continues.

## Optional flamegraphs (py-spy)

`run_all.py` does **not** require py-spy; it's a stand-alone
sampling profiler you point at any scenario. Install once:

```powershell
pip install py-spy
```

Then capture a flamegraph of the slowest scenario:

```powershell
py-spy record `
    -o profiling\reports\scrub_baseline.svg `
    --duration 30 `
    -- python -m profiling.harness.scenario_scrub
```

The `--native` flag also shows time in C extensions (helpful when the
hot path is inside `nd2`, `tifffile`, or scikit-image):

```powershell
py-spy record --native `
    -o profiling\reports\analysis_baseline.svg `
    --duration 30 `
    -- python -m profiling.harness.scenario_analysis
```

## Comparing two snapshots

A bare diff over `wall_ms` per scenario name is usually enough — both
snapshots use stable scenario names so jq / Python one-liners suffice
until we need a richer comparator.

```powershell
# example: compare scrub_t_large_nd2 across two phases
python -c "import json, sys; a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); a_scrub={x['name']:x['wall_ms'] for x in a['scenarios']['scrub'] if 'wall_ms' in x}; b_scrub={x['name']:x['wall_ms'] for x in b['scenarios']['scrub'] if 'wall_ms' in x}; print('\n'.join(f'{k}: {a_scrub.get(k,0):.1f} → {b_scrub.get(k,0):.1f} ms' for k in sorted(set(a_scrub)|set(b_scrub))))" `
    profiling\baselines\phase_00_baseline.json `
    profiling\baselines\phase_02_lazy_loading.json
```

## Common pitfalls

- **Cold OS cache** — the first scrub after boot will look artificially
  slow because the file pages aren't cached yet. Run twice and prefer
  the second number for warm-cache measurements.
- **`cProfile` overhead** — adds 2–5× to pure-Python paths. Stick with
  py-spy / `time.perf_counter` (what this harness uses) for whole-app
  profiling.
- **Display required for tab switching?** No — `scenario_tab_switch`
  forces `QT_QPA_PLATFORM=offscreen` before importing Qt.
- **Nuclei Segmentation absent** — Cellpose is optional. The pipeline
  is registered but `analysis_nuclei_segmentation` will record an
  `error` when `cellpose` isn't installed. Harness keeps running.
