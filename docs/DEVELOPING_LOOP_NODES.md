# Wiring a node into the iterative-loop system

*How to make a new Pipelines node work as the entry of a **loop connector** (the
amber back-edge that re-runs a region while sweeping parameters / iterating to a
target). Audience: developers adding a node to the Analysis graph.*

See also: `nd2studios/pipeline_graph/loop.py` (the pure loop core) and the loop
driver in `nd2studios/pages/pipelines_page.py` (search `# ── loop / iteration
driver`).

---

## 1. What a loop does

A loop edge (`Edge.kind == "loop"`) exits a node's bottom and returns to the top
of the same node or an upstream node. Its `Edge.params` hold the loop config
(`loop.default_loop_config()`): the **iteration plan** (parameter sweep grid /
`zip`, fixed count, or "until" a stop condition), the **combine rule**
(`union_dedup` / `best` / `last` / `keep_all`), a **multipoint scope**, and a
**save-all-iterations** toggle.

At Run time the driver:

1. detects the loop entry node,
2. for each iteration: applies that iteration's parameter overrides to the body
   nodes, **re-runs the entry node**, and **records** the result,
3. checks the stop condition (for "until"),
4. **combines** all iterations and publishes the combined result downstream, then
   restores the body's original params.

The pure logic (region, plan expansion, dedup, combine, tracking ratio) lives in
`loop.py` and is Qt-free / unit-tested. The page only orchestrates the async
re-runs and the display.

---

## 2. Two kinds of loopable node

| | **Row / mask nodes** | **Custom-output nodes** |
|---|---|---|
| Examples | Analysis (StarDist…), Track Objects, Cell-Tracker Metrics | DVC, Registration |
| Per-iteration product | measurement **rows** + label **masks** | a field series / transforms in the node's own tab |
| Combine rules | all four (`union_dedup` needs masks) | effectively **last** (others fall back) |
| Per-iteration plots | yes (Cells/frame, Area, Tracks/frame, Track length) | no |
| Viewer iteration dropdown ("Save all iterations") | yes | no |

The driver decides which behaviors apply from the node's **kind** (see the
registry below). Pick the row/mask path if your node's output is rows/masks;
otherwise the custom-output path.

---

## 3. The extension points

All in `PipelinesPage` (`pages/pipelines_page.py`). Adding a node touches a small
registry plus its own run/finish handlers.

### 3a. Registry (class-level tables)

```python
_LOOP_KIND_BY_OP = {          # op_key → kind   (analysis:* is matched separately)
    SPECIAL_TRACK_OP_KEY:     "track",
    SPECIAL_DVC_OP_KEY:       "dvc",
    SPECIAL_REGISTER_OP_KEY:  "register",
    SPECIAL_CT_METRICS_OP_KEY:"ct_metrics",
}
_LOOP_RUN_BY_KIND = {         # kind → the run handler to call for each iteration
    "analysis": "_run_analysis_node", "track": "_run_track_objects",
    "dvc": "_run_dvc", "register": "_run_register", "ct_metrics": "_run_ct_metrics",
}
_LOOP_ROW_KINDS = {"analysis", "track", "ct_metrics"}   # get plots + selector
```

`_loop_entry_kind(node)` returns the kind (or `""` = not loopable → the node runs
once with a status note).

### 3b. Driver helpers you reuse

- `_loop_maybe_begin(node)` — call once at the **top** of your run handler when
  `self._loop_ctx is None`. It builds `_loop_ctx`, applies iteration-0's param
  overrides, and records the entry kind. No-op if the node isn't a loop entry.
- `_loop_step(node, publish)` — call from your **finish** handler. It records the
  iteration, then either re-runs the next iteration (returns `True`; you must
  `return`) or combines synchronously and calls your `publish` callable (returns
  `True`). Returns `False` when no loop is active — then publish yourself. Used by
  every node **except analysis** (analysis keeps its bespoke per-M + off-thread
  union-dedup path in `_advance_run_m`).
- `_loop_record_iteration()` — snapshots `self._run_all_rows` (deep-copied) +
  `self._run_results_by_m` (label masks). Your handler must set `_run_all_rows`
  before the loop step so the snapshot is meaningful. Called for you by
  `_loop_step`.
- `_loop_begin_combine(force_sync=…)`, `_loop_apply_combined_sync`,
  `_loop_finish_cleanup`, `_populate_iteration_selector` — invoked by the driver;
  you don't call these directly.

### The contract

The loop reads two page attributes to record/combine an iteration:

- `self._run_all_rows: List[dict]` — the measurement rows this iteration produced.
- `self._run_results_by_m: Dict[int, AnalysisResult]` — per-multipoint label
  masks (only needed for `union_dedup` / the viewer selector).

Set whichever your node produces before calling `_loop_step`. A custom-output node
(no rows/masks) can leave them as the upstream values — combine will just carry
those through while your node's own tab shows the final result.

---

## 4. Checklist — wiring a new node

1. **Register the kind.** Add `op_key → kind` to `_LOOP_KIND_BY_OP` and
   `kind → "_run_yournode"` to `_LOOP_RUN_BY_KIND`. Add the kind to
   `_LOOP_ROW_KINDS` **iff** it produces rows/masks.
2. **Begin the loop.** At the top of `_run_yournode(node)`:
   ```python
   if getattr(self, "_loop_ctx", None) is None:
       self._loop_maybe_begin(node)
   ```
   Put this **before** you read `node.params`, so the swept overrides apply.
3. **Step + publish.** Factor your node's "publish" tail (update table / plots /
   overlay / tab, then `self._run_finish_node(node.id)`) into a
   `_finalize_yournode_publish(node, …)` method. In the finish handler:
   ```python
   self._run_all_rows = rows              # if you produce rows
   if not self._loop_step(node, lambda: self._finalize_yournode_publish(node, …)):
       self._finalize_yournode_publish(node, …)
   ```
   (For a **synchronous** node, the "finish handler" is just the back half of the
   run handler; for an **async** node it's the `_finish_*` job-done handler.)
4. **Numeric params.** Only `float` / `int` `ParamSpec`s appear in the Loop
   Settings sweep-axis picker (`_loop_body_param_options`). Give your sweepable
   knobs those types with sensible `min_val` / `max_val` / `step`.
5. **Done.** The Loop toolbar toggle, wire drawing, the Loop Settings dialog,
   combine rules, stop conditions, per-iteration plots and the viewer selector all
   work automatically for a row/mask kind. A custom-output kind gets parameter
   sweeping + "last".

That's it — no changes to `loop.py`, the node board, or the dialog are needed for
a standard node.

---

## 5. Worked examples (already in the tree)

- **Track Objects** (`_run_track_objects` / `_finish_track_objects`, async): a
  row kind. `_run_track_objects` calls `_loop_maybe_begin`; `_finish_track_objects`
  sets `_run_all_rows = rows` then `_loop_step(node, publish)` where publish is
  `_finalize_track_publish`. Sweeps tracking params; per-iteration Tracks/frame +
  Track-length plots; best-by-tracking-ratio combine.
- **DVC** (`_run_dvc` / `_finish_dvc`, async): a custom-output kind. Same two
  hooks; `_finalize_dvc_publish` opens the DVC tab for the **final** field series.
- **Cell-Tracker Metrics** (`_run_ct_metrics`, synchronous): a row kind; the run
  and finish live in one method.

---

## 6. Notes, limits & gotchas

- **Analysis is special.** It runs per-multipoint and can combine the heavy
  `union_dedup` **off the GUI thread** (`_LoopCombineJob`). Don't route analysis
  through `_loop_step`; it uses `_advance_run_m`. Model new *segmentation* nodes on
  it only if they need per-M streaming + mask dedup.
- **`union_dedup` needs masks.** For a node with no masks (tracking / DVC /
  registration) it falls back to `last`. Dedup compares by bbox-gated mask IoU with
  a centroid fallback (`loop.deduplicate_objects`), keyed per `(m, channel, frame)`.
- **Params are restored** after the loop (`_loop_restore_params` in
  `_loop_finish_cleanup`), and on a cancelled / finished Run — so a swept graph is
  never left mutated.
- **Save all iterations** (`save_iterations`) retains each iteration's rows/masks
  for the viewer's **Iteration** dropdown — row kinds only (`_LOOP_ROW_KINDS`); it
  holds every iteration in RAM, so it's opt-in.
- **Multipoint scope** (`multipoint: "current" | "all"`) is honored by
  `_run_analysis_node`; a custom node should read
  `self._loop_ctx["multipoint"]` if it iterates M itself.
- **Not loopable:** structural source/sink (Input/Output), if-else, Checkpoint,
  Pause, Review, Dismiss, Export, Send-to-Results, spatial-map openers — they have
  no meaningful "sweep + combine". `_loop_entry_kind` returns `""` for these and
  the loop runs the node once with a status note.

---

## 7. Testing a new loop node (offscreen, no data)

```python
import os; os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtWidgets import QApplication; QApplication([])
from nd2studios.pages.pipelines_page import PipelinesPage

class N:  # node stub
    def __init__(self, op): self.op_key = op; self.title = op
class S:  # minimal harness carrying the registry
    _LOOP_KIND_BY_OP = PipelinesPage._LOOP_KIND_BY_OP
    _loop_entry_kind = PipelinesPage._loop_entry_kind

assert S()._loop_entry_kind(N("special:your_op")) == "your_kind"
```

For the pure combine/dedup/plan logic, exercise `nd2studios.pipeline_graph.loop`
directly (see the existing loop self-tests). For plots, bind
`_update_track_plots_loop` / `_update_analysis_plots_loop` to a stub with a
`QTabWidget` `_data_tabs` and assert the legend labels.
