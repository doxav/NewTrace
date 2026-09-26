STATUS: PARTIAL

FACT: Current gates: {'S0': True, 'S1': True, 'S2': True, 'S3': True, 'S4': True, 'S5': True}. The process-stage-v1 amendment isolates stock evaluation stages for both frameworks. Earlier thread-evaluator results and the timeout diagnosis are archived in `artifacts/thread_evaluator_low/` and excluded from this comparison.

FACT: The fixed route is `z-ai/glm-5.3-flash` through OpenRouter, provider `novita`, reasoning effort `low`, session `benchmark-PRIMS-SIGNAL-run-001`, temperature 0.7, maximum 32,000 tokens and timeout 600 seconds. Trace uses CP-A: exact transport with a nonportable, nonpromotable control-plane override. CP-B was excluded because its empty-response fallback changes the token ceiling.

FACT: Current smoke and pilot outcomes are listed in `artifacts/diagnostic_summary.json`; pilots are excluded from strict quality comparisons.

MEASURED RESULT: Strict runs below start from the stock initial solution and consume up to 100 solution HTTP attempts. Partial runs have no 100-attempt AUC and are excluded from contrasts.

| Task | Arm | Attempts | Initial | Final best | Gain | Relative gain | AUC / 100 | First / best iteration | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|

| Task / arm | Valid / invalid | Policy switches | Valid proposals / total | Improving windows | Solution / meta / guide calls | Input / output / cached tokens | Cost USD | Wall seconds |
|---|---|---:|---|---:|---|---|---:|---:|

MEASURED RESULT: PRISM native components. The stock `max_kvpr` field is inverse mean maximum pressure over successful cases; lower implied pressure is better, but success rate must also be considered.

| Arm | Combined score | Inverse pressure | Implied mean maximum pressure | Success rate |
|---|---:|---:|---:|---:|

MEASURED RESULT: Signal Processing native components.

| Arm | combined_score | composite_score | correlation | noise_reduction | slope_changes | lag_error | success_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| No strict Signal run executed | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |

MEASURED RESULT: Completed strict contrasts only. Positive score differences favor the first arm.

| Task | Contrast | Final score difference | Relative-gain difference | AUC difference |
|---|---|---:|---:|---:|
| Unmeasured | No completed strict pair | unmeasured | unmeasured | unmeasured |

INFERENCE: The strict matrix is incomplete. The unrun within-framework contrasts cannot establish whether Trace or EvoX improves over its fixed policy, or whether Trace matches EvoX at equal budget.

INFERENCE: Advanced phase is not authorized by the current evidence gate. It has not run; no additional-freedom benefit has been measured.

LIMITATION: This is a controlled single-run benchmark, not a statistical replication. No p-values are computed. Equal solution attempts do not equal total compute. The shared session and sequential run order can affect cache, latency and cost. Costs are provider-reported; calls with missing cost are not treated as free. Process-series evaluator invocation and stage counts are recorded separately in the detailed metrics.

MEASURED RESULT: Current evaluator series, including diagnostics and pilots: 97 completion requests, 619105 reported tokens, $0.08638801 reported cost; 0 calls have unknown cost. Historical evaluator/provider/reasoning series are excluded; unchanged transport smoke evidence is reused and its cost remains in the historical series.

FACT: Detailed curves, role accounting, semantic retries, policy validation and per-window gains are in `artifacts/diagnostic_summary.json`. Runtime source hashes are frozen in `artifacts/strict_source_hashes.json`. Each immutable run directory retains source, requests, results and logs.

Validation commands: `experiments/recursive_opt/EXP22/.venv/bin/python -I -m unittest discover -s experiments/recursive_opt/EXP22/tests -v`; `ruff check experiments/recursive_opt/EXP22/src experiments/recursive_opt/EXP22/scripts experiments/recursive_opt/EXP22/tests`; `python3 experiments/recursive_opt/EXP22/scripts/analyze.py`; `python3 experiments/recursive_opt/EXP22/scripts/plot_results.py`. Existing targeted Trace tests: 162 passed, six unrelated integration cases deselected. Credential scan required before publication. Generated stock YAML/log whitespace is preserved as execution evidence.
