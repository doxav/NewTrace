# EXP24 — code delta from EXP23

Package: `~/code/Trace/opto/features/recursive_opt/coevolution/` (not yet committed; see snapshots below).
Full diff: [`code_delta_exp23_to_exp24.patch`](code_delta_exp23_to_exp24.patch) (7 files).

| Snapshot | Archive | SHA-256 |
|---|---|---|
| Version that ran EXP23 (reconstructed by reversing the EXP24 edits; its 21 tests pass on it) | `source_snapshots/coevolution_exp23.tar.gz` | `ae2393cadaf891b263dca9e75c3c812d4c08efa4ce596ed32df7b8f23205393a` |
| Version before the EXP24 clean runs | `source_snapshots/coevolution_exp24_pre_rerun.tar.gz` | `b158fe8c73a66795285dca0d36ce2294bfefc673a1f72b7c28c397094a0b0031` |

Each run writes the SHA-256 of every executed `coevolution/*.py` file in `run_manifest.json`.

## Why the change

EXP23's best PRISM score (30.8766) came from a program that crashes on 47 of 50 cases: PRISM averages KVPR over
*solved* cases, so crashing on hard cases raises the score. The search followed that signal because the engine used the
benchmark's raw metric as its only guide, saw only aggregate numbers, and lost about a quarter of its budget to broken
candidates. Trace separates the guide from the task and traces intermediate values; EXP24 adds those ideas to the
co-evolution engine, generically.

## Added

| File | What it provides |
|---|---|
| `guides.py` | `GuidedEvaluator`: the score the search follows (`guided_score`) is separate from the raw metric; hard constraints (`penalize` or `reject`); raw metrics kept for reporting. `format_case_diagnostics`: per-case feedback (failures grouped by cause, worst cases with ratio to a per-case bound). |
| `projections.py` | Projections applied to every candidate before evaluation: `CompileCheck` (syntax and required names, rejected before evaluation) and `FallbackWrapper` (per call, falls back to a feasible baseline when the candidate raises or returns an invalid result, logging the cause). Registry: `register_projection` / `make_projection` so specs declare projections by versioned ref. |

## Changed

| File | Change |
|---|---|
| `operator.py` | `PopulationOperator(projections=...)`: projects each parsed candidate; a `ProjectionError` is a failed attempt with its message fed back; evaluation runs on the projected source; the child keeps its editable source and stores the deployable one in `metadata['deployable']`; projection notes appear in the artifacts. |
| `engine.py` | `CoevolutionEngine(projections=...)`: passed to the operator and applied to the initial program; the report returns `best_source` (deployable) and `best_editable_source`. |
| `control_plane.py` | `engine.config.projections: [{ref, config}]` built through the registry; `coevolution_spec(score_key=...)` sets the objective and selection key (default unchanged). |
| `__init__.py` | Exports the new names. |

With no projections and the default score key, behavior is unchanged: the SkyDiscover EvoX equivalence matrix
(`EXP23/equivalence/run_matrix.sh`, 5 scenarios) passes on the EXP24 version.

## Tests

`tests/unit_tests/test_recursive_coevolution.py`: 21 EXP23 tests unchanged, plus 7 new ones (guide scores and
constraints, case diagnostics, fallback wrapper, compile check, registry, operator projection path, deployable report).
Result: 27 passed, 1 skipped (`TraceProposer` test skipped only where `litellm` is incompatible with the installed
`openai`; it passes in the EXP22 venv). `EXP24/tests/test_exp24.py`: 5 offline tests (white-box evaluator reproduces the
stock metric on three programs, valid score penalizes both exploits, GPU-range check, exact optimum, all three arms end to
end with a mock LLM).

## Experiment-side (EXP24 only)

- `prism/whitebox_worker.py`, `prism/whitebox.py`: per-case PRISM evaluation (stock generator, checks and 10 s timeout),
  stock metric reproduced exactly, `valid_score`, per-case feedback, and the check/fallback sources for `FallbackWrapper`.
  The check also rejects GPU ids outside `range(gpu_num)`, which the stock evaluator does not verify.
- `scripts/run_prism.py`: arms `fixed` / `llm_rewrite` / `trace`, seeds, `strict_budget` (exactly 100 calls; EvoX can
  overshoot by retrying in its last iteration), transport retries up to one hour per call (an outage no longer consumes
  budget; it cost 3 attempts in the pilot), run manifest with code hashes.
- `scripts/prism_exact.py`, `scripts/prism_validity.py`, `scripts/analyze.py`.

Known repository effect: `tests/unit_tests/test_recursive_control_plane_v2.py::test_35_source_provenance` pins a hash of
the whole `recursive_opt/` tree, so any new file there fails it by design until the readiness lock is refreshed.
