STATUS: STOPPED_PROVIDER

Completed strict runs: **none**. Completed pilots: **none**. Successful generated
solution candidates: **zero**. Two PRISM/SD-EVOX S4 attempts were rejected by
DeepInfra. This is an availability failure, not an optimizer-quality result.

**FACT — continuation and provenance.** The initial branch mismatch report was
committed and pushed as `3004bf10ba` on `codex/exp22-evox-comparison`. On the user's
instruction to continue, execution moved to an EXP22-owned detached worktree at
exactly `recursive_opt`: `846580defe935195c1f5f39d6336079ef6ff1e10`.
The original dirty Trace checkout remains unchanged. CP-B is a separate worktree
at the same commit with only the recorded routing patch. SkyDiscover remains at
`3f7a611fe83980970dd14f1d65c49e40de61b7df`; all six benchmark files match the supplied
hashes and HEAD. No evolved solution or historical experiment output was used.
The original [S0 stop report](artifacts/initial_stopped_precheck_report.md) is preserved.

**MEASURED RESULT — gates.**

| Gate | Observed result |
|---|---|
| S0 sources/environment | Exact branch-ref isolation; both frameworks, NumPy and SciPy import; pip check passes. |
| S1 evaluators | Ten sequential repetitions per task/path; golden metrics and stock/Trace cascade parity pass within 1e-12; broken candidates rank identically. |
| S2 transport | Direct, SkyDiscover, CP-A and CP-B live responses confirm the exact GLM model and DeepInfra provider. Serialized requests include the exact PRIMS session ID. |
| S3 compile | Eight plans compile: two tasks × two Trace arms × two CP variants; registered refs, roles and 100-generation budgets are persisted. |
| S4 candidate | Two PRISM/SD-EVOX attempts fail on HTTP 429; other task/framework candidates not run. |
| S5 pilot | Not run. |

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

**MEASURED RESULT — routing/control checks.** CP-A and CP-B were tested in separate
processes against their respective source trees. Their identical fixed HTTP
response passes through the real OptoPrimeV2 parser, stock policy validator and
stock database migration. Initial policy/population, history, feedback, complete
HTTP bodies, validation, retained population and deployed policy hashes agree.
The parser's native Black formatting is reflected in the recorded deployed hash.
CP-A remains an explicitly non-portable behavioral override; CP-B uses the
normalized `openrouter_routing` extension. The patch also routes the automatic
Trace startup probe. Full-run portability and search behavior remain unmeasured.

**MEASURED RESULT — provider failure.**

| S4 attempt (UTC) | HTTP responses | Valid generated candidates |
|---|---|---:|
| 2026-09-26 10:26:37 | 3 × 429 engine_overloaded | 0 |
| 2026-09-26 10:28:00 | 3 × 429 engine_overloaded | 0 |

Each attempt rejected guide/meta startup probes and the solution request.
Provider metadata identifies `DeepInfra`, `upstream_provider_shared_pool` and
`engine_overloaded`. The delayed retry kept the same model/provider/session and
was preregistered before execution. No provider fallback was used. Stock EvoX's
availability fallback was observed, but the EXP22 gate correctly rejected that
run rather than crediting it as a successful EvoX candidate smoke.

Four earlier successful tiny transport calls consumed **125 returned tokens**
and reported **$0.000019875**. The six rejected requests returned no usage/cost;
their billing is unknown, not asserted to be zero. One tiny CP-A response spent
its 32-token smoke allowance without final text; it established routing only.
No evidence suggests a problem with the 32000-token optimization limit or 600-second
timeout: the actual generation requests were refused immediately.

**INFERENCE.** Provider availability blocked the candidate-generation gate after
transport compatibility passed. This says nothing about the relative quality or
value of Trace meta-optimization. No candidate-level trajectory, AUC, effect size,
p-value or quality comparison can be estimated from these failures.

**LIMITATION.** The hybrid engine is implemented and exercised by the mocked policy
fixture, but its live canonical run, all S5 properties, intermediate strict-run
checks, final compute accounting, complete strict-run analysis/plots and advanced
phase remain unvalidated. The strict command is gated off. Do not turn the baseline
score retained in a failed smoke's final_result into a completed optimization run.
`invalid_attempts=1` in each smoke denotes a rejected generation request, not
invalid generated source. Source hash locks are EXP22-local; historical locks were
not rewritten. Shared-session caching remains a possible confound for future
latency/cost comparisons. A future single-run-per-arm comparison would not be a
statistical replication.

Final decisions:

| Question | Answer |
|---|---|
| Does Trace meta-optimization improve over Trace fixed? | Not measured. |
| Does EvoX improve over SD-FIXED? | Not measured. |
| How close are frameworks at equal generation budget? | Not measured. |
| What is each arm's extra total compute? | Not measured; diagnostic usage is reported separately above. |
| Was the advanced phase warranted or run? | No: strict eligibility conditions were not reached. |
| Did additional freedom improve results? | Not tested. |

Evidence: [stop record](artifacts/STOP.json), [diagnostic summary](artifacts/diagnostic_summary.json),
[evaluator parity](artifacts/evaluator_parity.json), [live transport](artifacts/openrouter_transport_validation.json),
[CP equivalence](artifacts/control_plane_equivalence.json), [routing patch](artifacts/trace_cp_b.patch),
[manifest](manifest.json), [environment](artifacts/environment.json), [gates](artifacts/gates.json).
Run directories contain sanitized request evidence, histories, usage, final result,
baseline best sources, configs and stdout/stderr logs. Provider account identifiers
were redacted before publication; scientific fields and failures were preserved.
The recursive secret scan includes environment/worktree files. Its revised pattern
uses token boundaries and actual key lengths to avoid matching package CSS class
names or substrings of wheel hashes; no directory is excluded.

Verification, from `/home/xav/code/Trace-experiment0`:

```bash
experiments/recursive_opt/EXP22/.venv/bin/python -I -m unittest discover -s experiments/recursive_opt/EXP22/tests -v
ruff check experiments/recursive_opt/EXP22/src experiments/recursive_opt/EXP22/scripts experiments/recursive_opt/EXP22/tests
experiments/recursive_opt/EXP22/.venv/bin/python -I experiments/recursive_opt/EXP22/scripts/evaluator_preflight.py
experiments/recursive_opt/EXP22/.venv/bin/python -m pip check
python3 experiments/recursive_opt/EXP22/scripts/analyze.py
```

The EXP22 suite has 18 tests. Lint, evaluator parity, dependency consistency and
analysis pass. See [unit output](artifacts/exp22_unit_tests.txt) and
[Trace runtime output](artifacts/trace_cp_b_runtime_tests.txt).
The initial broader Trace baseline returned 162 passes and six failures: three
optional GEPA dependency failures, one optional notebook dependency failure, and
two pre-existing historical provenance-lock failures. The CP-B targeted runtime
rerun passed all 162 applicable tests with those six deselected, explicitly:

```bash
cd experiments/recursive_opt/EXP22/worktrees/trace_cp_b
../../.venv/bin/python -I -c "import sys; sys.path.insert(0, '.'); import pytest; raise SystemExit(pytest.main(['tests/unit_tests/test_recursive_control_plane_v2.py','tests/unit_tests/test_recursive_transport.py','tests/unit_tests/test_recursive_budget_experiments.py','tests/unit_tests/test_recursive_final_hardening.py','tests/unit_tests/test_objectives.py','-q','--tb=short','-k','not real_gepa and not public_optimize_anything and not notebook_executes and not source_provenance and not readiness_uses_source_digests']))"
../../.venv/bin/python -I -c "import sys; sys.path.insert(0, '.'); import pytest; raise SystemExit(pytest.main(['tests/unit_tests/test_recursive_opt.py','-q','--tb=short','-k','code or artifact or optimizer']))"
```

Code-artifact test output is [recorded separately](artifacts/trace_code_artifact_tests.txt).
No full historical CI pass is claimed. No unrelated source changes were staged.

Resume only after the exact GLM/DeepInfra route becomes available. Rerun S4 in a
new directory, complete all task/framework candidates and S5, then finish the
remaining validation before enabling strict runs. Do not change routing to make
an availability failure disappear.
