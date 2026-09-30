STATUS: STOPPED_PROVIDER — fixed controls complete; recursive replacements partial

The requested four workers ran TRACE-RECURSIVE and TRACE-FIXED on PRISM and Signal Processing, targeting 100 actual solution HTTP attempts each. The frozen model was `z-ai/glm-5.3-flash`, routed through OpenRouter to Novita with low reasoning effort. Both fixed controls completed. Recursive runs restarted once after a local context-growth repair; both replacements subsequently stopped on Novita HTTP 429 (`upstream_provider_shared_pool`). No workers remain running.

| Task | Arm | Solution calls / outcomes | Best score | Policies deployed | Valid / invalid | Status |
|---|---|---:|---:|---:|---:|---|
| PRISM | TRACE-RECURSIVE | 77 / 77 | 24.851144 | 3 | 52 / 25 | Partial: upstream 429 |
| PRISM | TRACE-FIXED | 100 / 100 | 26.364386 | 0 | 70 / 30 | Complete |
| Signal | TRACE-RECURSIVE | 93 / 93 | 0.543789 | 8 | 90 / 3 | Partial: upstream 429 |
| Signal | TRACE-FIXED | 100 / 100 | 0.606292 | 0 | 98 / 2 | Complete |

Higher combined scores are better. PRISM best-solution success rates are 0.76 (recursive) and 0.92 (fixed); interpret pressure performance alongside this metric. Both Signal best solutions have success rate 1. All saved best solutions reproduce their non-timing metrics on reevaluation.

The fixed arms have higher observed endpoint scores in this replacement comparison, with unequal budgets. This does not establish that meta-optimization has no advantage. Neither task has a completed matched 100-attempt pair, and one stochastic trajectory per arm cannot establish statistical reliability even after completion. Recursive search exercised three policies on PRISM and eight on Signal. Proposals retain the active policy as parent and use the inherited stagnation trigger. Fresh starts occurred only when restarting entire runs after the context repair.

The unchanged fixed controls are reused under amendment v9. Provider and local resource load varied as workers finished and replacements started, so wall-time comparisons require caution. Equal solution-attempt budgets also do not imply equal total model compute. Earlier sequential runs are excluded.

Preserved original recursive attempts are diagnostics, excluded from the replacement comparison:

| Task | Solution calls / outcomes | Best score | Policies deployed | Stop reason |
|---|---:|---:|---:|---|
| PRISM | 98 / 98 | 26.439704 | 5 | HTTP 400: duplicated observations exceeded provider context |
| Signal | 52 / 52 | 0.555813 | 3 | HTTP 400: duplicated observations exceeded provider context |

The context repair encodes repeated complete JSON values by lossless references and supplies feedback once in the Trace prompt. All 11 observations from the replacement runs decode exactly to their full saved observations. No source, population member, metric, or metadata version is discarded. Replacement runs had no context-limit failures; this does not guarantee unbounded future context capacity.

Provider-error accounting was repaired before launch: every consumed solution HTTP request has a recorded outcome, including failed requests. The original PRISM recursive run also exercised the process timeout repair in a live evaluation: the evaluator was killed and reaped after 360 seconds, and the run continued. Current stops are upstream provider refusals.

Across all six execution runs, 520 solution attempts and 559 total completion requests were recorded. Provider-reported cost is $1.26224769; four refused requests have unknown cost. These totals include abandoned context-overflow attempts and exclude two credential/provider preflight probes.

Evidence:

- [Detailed results and native metrics](results/artifacts/parallel_trace_20260927T194336Z/v9_comparison/results.md), [machine-readable trajectories](results/artifacts/parallel_trace_20260927T194336Z/v9_comparison/results.json), and [worker manifest](results/artifacts/parallel_trace_20260927T194336Z/v9_comparison/manifest.json).
- [PRISM curves](results/artifacts/parallel_trace_20260927T194336Z/v9_comparison/prism_strict_curves.png) and [Signal curves](results/artifacts/parallel_trace_20260927T194336Z/v9_comparison/signal_processing_strict_curves.png); partial curves are not extrapolated to 100.
- [Final validation](results/artifacts/parallel_trace_20260927T194336Z/v9_comparison/final_validation.json), [feedback round trips](results/artifacts/parallel_trace_20260927T194336Z/v9_comparison/feedback_roundtrip.json), and [all-attempt cost accounting](results/artifacts/parallel_trace_20260927T194336Z/v9_comparison/all_attempts_cost.json).
- [Previous report](results/artifacts/parallel_trace_20260927T194336Z/previous_RESULTS.md) and [previous stop evidence](results/artifacts/parallel_trace_20260927T194336Z/previous_STOP.json).

Verification commands (from repository root):

```sh
experiments/recursive_opt/EXP22/.venv/bin/python -I -m unittest discover -s experiments/recursive_opt/EXP22/tests -v
ruff check experiments/recursive_opt/EXP22/src experiments/recursive_opt/EXP22/scripts experiments/recursive_opt/EXP22/tests
python3 -m compileall -q experiments/recursive_opt/EXP22/src experiments/recursive_opt/EXP22/scripts experiments/recursive_opt/EXP22/tests
python3 experiments/recursive_opt/EXP22/scripts/analyze_parallel.py --manifest experiments/recursive_opt/EXP22/artifacts/parallel_trace_20260927T194336Z/v9_comparison/manifest.json
python3 experiments/recursive_opt/EXP22/scripts/plot_results.py --parallel-manifest experiments/recursive_opt/EXP22/artifacts/parallel_trace_20260927T194336Z/v9_comparison/manifest.json
git diff --check -- experiments/recursive_opt/EXP22
```

All 50 runtime/report tests pass. Lint, compilation, runtime source-lock verification, credential scan, four concurrent golden evaluator checks, and saved-solution reevaluations pass. Both plots were visually reviewed. The scoped EXP22 diff check passes. Repository-wide `git diff --check` reports pre-existing trailing whitespace in unrelated `opto/features/recursive_opt/spec.py`; that file was left untouched.

A [reviewable recovery proposal](results/artifacts/parallel_trace_20260927T194336Z/rate_limit_proposal/proposal.md) and [patch](results/artifacts/parallel_trace_20260927T194336Z/rate_limit_proposal/review.patch) are prepared but unactivated. Its 11 offline tests pass. It permits at most two precisely identified Novita solution 429 refusals per run, pauses 30 seconds after each failed batch, retains each failed call within the 100-attempt budget, and preserves all other guards. A third refusal still stops. Approval of this stop-rule amendment is pending; no additional paid run has been launched. If approved, fresh recursive runs would preserve these partial trajectories and reuse the unchanged fixed controls.
