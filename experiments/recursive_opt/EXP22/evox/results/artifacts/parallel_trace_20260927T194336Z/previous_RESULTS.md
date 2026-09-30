STATUS: STOPPED_PROVIDER

FACT: The local evaluator timeout leak was repaired and validated before resumption. The resumed strict run stopped on a separate provider failure: Novita returned HTTP 429 (upstream_provider_shared_pool) on solution HTTP attempt 91. Its 91 solution HTTP attempts produced 88 recorded curve outcomes; the final failed batch is retained without inventing missing curve points.

FACT: The stock EvoX fallback treats an error result with no prompt as a database failure. It restored its fallback database and retried before recording the three consumed attempts; the EXP22 HTTP-versus-curve accounting guard then stopped further calls. The raw result and 88-point curve are unchanged. This is separate from the repaired evaluator timeout leak.

MEASURED RESULT: All 44 evaluator stages in the strict run completed and their workers were reaped; no evaluator timeout occurred. The remaining seven strict arms and the advanced phase were not run. This incomplete run cannot establish a Trace-versus-SkyDiscover contrast.

FACT: Current gates: {'S0': True, 'S1': True, 'S2': True, 'S3': True, 'S4': True, 'S5': True}. The process-stage-v1 amendment isolates stock evaluation stages for both frameworks. Earlier thread-evaluator results and the timeout diagnosis are archived in `artifacts/thread_evaluator_low/` and excluded from this comparison.

FACT: The fixed route is `z-ai/glm-5.3-flash` through OpenRouter, provider `novita`, reasoning effort `low`, session `benchmark-PRIMS-SIGNAL-run-001`, temperature 0.7, maximum 32,000 tokens and timeout 600 seconds. Trace uses CP-A: exact transport with a nonportable, nonpromotable control-plane override. CP-B was excluded because its empty-response fallback changes the token ceiling.

FACT: Current smoke and pilot outcomes are listed in `artifacts/diagnostic_summary.json`; pilots are excluded from strict quality comparisons.

MEASURED RESULT: Strict runs below start from the stock initial solution and consume up to 100 solution HTTP attempts. Partial runs have no 100-attempt AUC and are excluded from contrasts.

| Task | Arm | Solution calls / outcomes | Initial | Final best | Gain | Relative gain | AUC / 100 | First / best iteration | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| prism | SD-EVOX | 91 / 88 | 21.891622 | 25.965907 | 4.0742847 | 0.1861116 | unmeasured | 1 / 10 | diagnostic stop |

| Task / arm | Valid / invalid | Policy switches | Valid proposals / total | Improving windows | Solution / meta / guide calls | Input / output / cached tokens | Cost USD | Wall seconds |
|---|---|---:|---|---:|---|---|---:|---:|
| prism / SD-EVOX | 38 / 53 | 3 | 3 / 3 | 0.25 | 91 / 5 / 9 | 574098 / 133726 / 6400 | 0.11415727 | 2193.72 |

MEASURED RESULT: PRISM native components. The stock `max_kvpr` field is inverse mean maximum pressure over successful cases; lower implied pressure is better, but success rate must also be considered.

| Arm | Combined score | Inverse pressure | Implied mean maximum pressure | Success rate |
|---|---:|---:|---:|---:|
| SD-EVOX | 25.965907 | 25.005907 | 0.039990551 | 0.96 |

MEASURED RESULT: Signal Processing native components.

| Arm | combined_score | composite_score | correlation | noise_reduction | slope_changes | lag_error | success_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| No strict Signal run executed | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |

MEASURED RESULT: Completed strict contrasts only. Positive score differences favor the first arm.

| Task | Contrast | Final score difference | Relative-gain difference | AUC difference |
|---|---|---:|---:|---:|
| Unmeasured | No completed strict pair | unmeasured | unmeasured | unmeasured |

![prism trajectories and compute](artifacts/prism_strict_curves.png)

INFERENCE: The strict matrix is incomplete. The unrun within-framework contrasts cannot establish whether Trace or EvoX improves over its fixed policy, or whether Trace matches EvoX at equal budget.

INFERENCE: Advanced phase is not authorized by the current evidence gate. It has not run; no additional-freedom benefit has been measured.

LIMITATION: This is a controlled single-run benchmark, not a statistical replication. No p-values are computed. Equal solution attempts do not equal total compute. The shared session and sequential run order can affect cache, latency and cost. Costs are provider-reported; calls with missing cost are not treated as free. Process-series evaluator invocation and stage counts are recorded separately in the detailed metrics.

MEASURED RESULT: Current evaluator series, including diagnostics and pilots: 202 completion requests, 1326929 reported tokens, $0.20054529 reported cost; 1 calls have unknown cost. Historical evaluator/provider/reasoning series are excluded; unchanged transport smoke evidence is reused and its cost remains in the historical series.

FACT: Detailed curves, role accounting, semantic retries, policy validation and per-window gains are in `artifacts/diagnostic_summary.json`. Runtime source hashes are frozen in `artifacts/strict_source_hashes.json`. Each immutable run directory retains source, requests, results and logs.

Validation commands: `experiments/recursive_opt/EXP22/.venv/bin/python -I -m unittest discover -s experiments/recursive_opt/EXP22/tests -v`; `ruff check experiments/recursive_opt/EXP22/src experiments/recursive_opt/EXP22/scripts experiments/recursive_opt/EXP22/tests`; `python3 experiments/recursive_opt/EXP22/scripts/analyze.py`; `python3 experiments/recursive_opt/EXP22/scripts/plot_results.py`. Existing targeted Trace tests: 162 passed, six unrelated integration cases deselected. Credential scan required before publication. Generated stock YAML/log whitespace is preserved as execution evidence.

Diagnostic stop: Provider/transport validation failed: HTTP 429, provider Novita, limit source upstream_provider_shared_pool. Last recorded iteration: 88; last valid score: 25.96590682442097. See `artifacts/STOP.json`.

Recommended next experiment: Wait for the frozen provider route to be available. Before another strict run, address the stock fallback path that can skip counting failed HTTP batches; preserve this stopped run and preregister any runtime amendment. Do not disable the accounting or serving-identity guards.

Reproduce the timeout diagnosis without paid calls: `experiments/recursive_opt/EXP22/.venv/bin/python -I experiments/recursive_opt/EXP22/scripts/timeout_diagnostic.py`.

FACT: Final validation passed: EXP22 tests, lint, bytecode compilation, unchanged source hashes and frozen runtime, credential scan, saved-source re-evaluation, and visual plot review. No owned benchmark process remains active. Exact commands and counts: `artifacts/final_validation.json`.
