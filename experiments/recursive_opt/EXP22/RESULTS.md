STATUS: STOPPED_PROVIDER

The requested provider replacement is implemented: every new EXP22 request uses
`provider.only=["novita"]`. The actual serving provider was Novita on all 14
HTTP completion responses. Model `z-ai/glm-5.3-flash`, session
`benchmark-PRIMS-SIGNAL-run-001`, temperature 0.7, generation max_tokens=32000
and timeout=600 seconds remain unchanged.

Completed strict runs: **none**. Pilots: **none**. Generated solution candidates:
**zero**. Two PRISM/SD-EVOX generation smokes completed diagnostically and failed.
This is a generation failure under the frozen settings, not an optimizer-quality
comparison or a Novita routing failure.

**FACT — provenance and amendment.** EXPERIMENT.md amendments v5/v5.1 record the
user-requested provider change, CP-A fallback and bounded diagnostic retry before
execution. Trace remains at recursive_opt `846580defe935195c1f5f39d6336079ef6ff1e10`
in owned detached worktrees; SkyDiscover remains at
`3f7a611fe83980970dd14f1d65c49e40de61b7df`. Source checkouts and evaluators were
not changed. All six stock benchmark hashes still match. Previous DeepInfra
observations/configs are retained under `artifacts/deepinfra_pre_novita/` and in
their original run directories; they are excluded from Novita totals.

**MEASURED RESULT — gates.**

| Gate | Result |
|---|---|
| S0 sources/environment | Passed; unchanged source/environment evidence remains applicable. |
| S1 evaluators | Passed; earlier ten repetitions per task/path and invalid-candidate cascade parity remain applicable. |
| S2 transport | Direct, SkyDiscover and CP-A each passed a single Novita call. CP-B diagnostic failed its single-call condition; excluded from live runs. |
| S3 compile | Eight Novita specifications compile; canonical CP-A execution also passes a mocked kernel test. |
| S4 generation | Both PRISM/SD-EVOX attempts returned empty content after reaching 32000 tokens. Other task/framework pairs not run. |
| S5 pilots | Not run; blocked by S4. |

**MEASURED RESULT — generation failure.**

| PRISM S4 start (UTC) | Completion tokens | Reasoning tokens | Finish reason | Valid candidates | Wall time (s) |
|---|---:|---:|---|---:|---:|
| 2026-09-26 11:04:57 | 32000 | 31997 | length | 0 | 337.94 |
| 2026-09-26 11:11:48 | 32000 | 32000 | length | 0 | 343.17 |

All eight S4 HTTP responses were successful and correctly routed. Per attempt:
one meta availability probe, one guide availability probe, one guide-label request,
and one solution request. Guide-label generation succeeded. Stock SkyDiscover
reported `LLM returned None response` for each solution request. The first finish
reason was independently retrieved from OpenRouter generation metadata; the
second was retained directly by the observer alongside content_lengths=[0].
No request timed out and no HTTP 429 occurred in the Novita series.

The tiny one-token startup probes intentionally check connectivity, not content.
Their empty length-limited responses are not solution-generation failures.
`invalid_attempts=1` denotes an empty generation attempt, not evaluated invalid
source. The best_solution.py/best_policy.py retained in each directory are stock
baselines, not generated improvements.

**MEASURED RESULT — Trace control plane.** The successful-response CP-A/CP-B mocked
fixture still agrees on final bodies, measured feedback, real OptoPrimeV2 parsing,
stock validation, deployed policy hash and preserved solution population. Live
CP-B nevertheless retried an empty 32-token smoke with 64 and 128 tokens. Source
inspection confirms this recovery can also raise 32000 to 32768. CP-B is therefore
excluded under the protocol's predefined fallback rule. CP-A keeps the exact
limit, uses an explicit llm_factory and is labeled non-portable/non-promotable.
A new unit test verifies that empty CP-A responses do not increase its token limit.
The canonical engine/factory path is tested without paid generation; live Trace
solution/meta optimization remains unmeasured.

**MEASURED RESULT — usage.** Novita totals across six transport requests (including
CP-B's three requests) and eight S4 requests: **86713 tokens**, **$0.028471665**
reported cost, no missing cost records. These are diagnostic costs, not arm-level
optimization efficiency. The separate historical DeepInfra series reported
$0.000019875 for four completions, with unknown cost for six rejected requests.
The shared session produced cache hits on the second Novita attempt; cost/latency
comparisons are consequently not independent replications.

**MEASURED RESULT — stock initial programs only.** These are golden evaluator
checks, not optimization outcomes. Each path repeated deterministically.

| Metric | PRISM | Signal |
|---|---:|---:|
| combined_score | 21.891622105209393 | 0.49904861783269006 |
| max_kvpr (stock inverse-pressure score) | 20.891622105209393 | — |
| composite_score | — | 0.4518100150842093 |
| correlation | — | 0.8420816095203836 |
| noise_reduction | — | 0.3347443615622788 |
| slope_changes | — | 223.4 |
| lag_error | — | 0.3090769071827435 |
| success_rate | 1.0 | 1.0 |

The PRISM implied mean maximum placement pressure is approximately
`1 / 20.891622105209393`. Timing is excluded from parity and quality claims.
Signal stage1 reports runs_successfully=1, composite_score=0.7, output_length=91.
The adapter delegates the entire cascade to stock SkyDiscover, including its
threshold, stage2 merge and error handling; no full-eval/cascade mismatch is used.


**INFERENCE.** Changing provider resolved the earlier HTTP overload rejection,
but did not establish usable solution generation at the frozen token limit. Two
empty capped generations trigger the preregistered diagnostic stop. There is no
measured evidence for or against either meta-optimizer's search quality.

**LIMITATION.** No candidate trajectory, AUC, quality effect, p-value or comparison
plot can be estimated from zero generated programs. The first Novita run's copied
source_manifest retains the prior status/counters; its source SHAs remain valid,
and its local routing config plus HTTP records establish actual Novita execution.
The root manifest was refreshed before the second attempt. Strict intermediate
checks, full live accounting/analysis and advanced-phase implementation still need
validation before any full run. Historical unrelated Trace test exclusions remain
as documented in the archived report; no provenance lock was rewritten.

| Decision | Observed answer |
|---|---|
| Trace recursive versus Trace fixed | Not measured. |
| EvoX versus SD fixed | Not measured. |
| Trace versus EvoX at equal generation budget | Not measured. |
| Extra total compute by arm | Not measured; diagnostic usage is reported above. |
| Advanced phase warranted / run / beneficial | No / no / not tested. |

**Verification.** All 23 EXP22 unit tests and 162 targeted Trace tests pass.
Ruff, compileall, diff whitespace checks and the final recursive secret scan pass.
No new production dependency was added. Exact commands (repository cwd unless
specified):

```bash
experiments/recursive_opt/EXP22/.venv/bin/python -I -m unittest discover -s experiments/recursive_opt/EXP22/tests -v
ruff check experiments/recursive_opt/EXP22/src experiments/recursive_opt/EXP22/scripts experiments/recursive_opt/EXP22/tests
experiments/recursive_opt/EXP22/.venv/bin/python -I -m compileall -q experiments/recursive_opt/EXP22/src experiments/recursive_opt/EXP22/scripts experiments/recursive_opt/EXP22/tests
git diff --check -- experiments/recursive_opt/EXP22
python3 experiments/recursive_opt/EXP22/scripts/analyze.py
```

From `experiments/recursive_opt/EXP22/worktrees/trace_cp_b`:

```bash
../../.venv/bin/python -I -c "import sys; sys.path.insert(0, '.'); import pytest; raise SystemExit(pytest.main(['tests/unit_tests/test_recursive_control_plane_v2.py','tests/unit_tests/test_recursive_transport.py','tests/unit_tests/test_recursive_budget_experiments.py','tests/unit_tests/test_recursive_final_hardening.py','tests/unit_tests/test_objectives.py','-q','--tb=short','-k','not real_gepa and not public_optimize_anything and not notebook_executes and not source_provenance and not readiness_uses_source_digests']))"
```

Live commands (credential loaded in environment without logging it) were:

```bash
experiments/recursive_opt/EXP22/.venv/bin/python -I experiments/recursive_opt/EXP22/scripts/transport_smoke.py
experiments/recursive_opt/EXP22/.venv/bin/python -I experiments/recursive_opt/EXP22/scripts/framework_smoke.py
experiments/recursive_opt/EXP22/.venv/bin/python -I experiments/recursive_opt/EXP22/scripts/run_stage.py --stage one --task prism --arm SD-EVOX
```

The last command was executed twice in separate immutable directories. The first
framework-smoke invocation exited 2 on CP-B's three-call behavior; CP-A was then
selected using the successful existing direct/Sky/CP-A evidence. That failure
remains visible in the transport artifact. No rerun was hidden.

Evidence: [stop](artifacts/STOP.json), [analysis](artifacts/diagnostic_summary.json),
[transport](artifacts/openrouter_transport_validation.json),
[first generation metadata](artifacts/novita_first_s4_generation_metadata.json),
[CP equivalence](artifacts/control_plane_equivalence.json), [gates](artifacts/gates.json),
[DeepInfra history](artifacts/deepinfra_pre_novita/RESULTS.md), [manifest](manifest.json).
