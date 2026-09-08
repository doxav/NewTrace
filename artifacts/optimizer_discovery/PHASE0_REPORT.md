# Phase 0 report

Phase 0's instrument and Patrick interface are complete. **Phase 1 optimizer-search
execution is NO-GO under the frozen open-ended generation prompt.** The portable
interface is ready for Patrick and for Phase-1 protocol development. These are
separate conclusions: the real provider can deliver and execute a trivial optimizer,
but the preregistered optimizer-generation calibration produced no usable code.
There is **no genuine external provider/model blocker** and no performance claim.

Base: `846580defe935195c1f5f39d6336079ef6ff1e10` (fetch/ff-only pull found it current).
Branch: `codex/phase0-patrick-interface`; rollback: `codex/phase0-rollback-846580de`.
Final runtime checkpoint: `40df4a11` (full SHA in validation_summary.json).
The final documentation/evidence commit is identified by `git rev-parse HEAD`.
No merge, push or PR was made. The user's untracked probe_aa_results.json remains untouched.

| criterion | status | evidence | command/artifact | remaining risk |
|---|---|---|---|---|
| A existing offline tests | PASS | requested baseline suites now pass | baseline_tests.txt; final_tests.txt | 2 existing optional telemetry skips |
| B new tests | PASS | 40 new cases, including real subprocess and Trace trainer execution | test_recursive_optimizer_program.py; test_recursive_menu_evidence.py; test_recursive_phase0_calibration.py; added legacy-return test | tests do not prove arbitrary hostile-code safety |
| C menu evidence | PASS, scoped | every canonical run reports evidence; legacy public results expose it; actual evaluations retained before queue eviction | metadata.menu_evidence / menu_observations; menu tests | scalar-only legacy evaluators cannot establish behavioral headroom; scope is common measured inputs |
| D executable documented contract | PASS | portable generated optimizer.py and host API | OPTIMIZER_PROGRAM_V0.md; live/interface_smoke/optimizer.py | fixture is not the Phase-1 benchmark |
| E typed valid/invalid | PASS | absent invalid metrics; typed proposal failures; no exported invalid ranking penalties | invalid aggregation and trajectory tests; canonical invalid-program test | internal legacy/GEPA/Trace ranking adapters still use numeric penalties; exports exclude them from valid scientific scores |
| F timeout and clean environment | PASS | 2-second process-group timeout, temporary directory, allowlisted environment, isolated stdlib worker | subprocess timeout/environment/worker tests | subprocess is not a security sandbox; filesystem/network access is not blocked |
| G exact deterministic budget | PASS | midpoint program runs 8 objective calls for each of seeds 0,1,2; local proposal replay uses no objective budget | live/interface_smoke/result.json; final_artifact_replay.json | determinism checks detect observed disagreements, not universal determinism |
| H canonical integration | PASS | existing trainable component module plus versioned evaluator; real Trace source update improves deterministic fixture | test_optimizer_artifact_is_trainable_through_real_trace | CodeArtifactLevel uses in-process self methods; deliberately reused canonical component path instead |
| I real exact-model validation | PASS, interface only | OpenRouter DeepSeek generated a complete midpoint proposer; validated/executed on 3 seeds | request gen-1788857418-EIBlbPCalJInJykWsqNC; live/interface_smoke/ | original three-request generation calibration failed; this separate preregistered engineering smoke does not replace it |
| J historical evidence isolated | PASS | original EXP-12/13 records unchanged; missing ephemeral EXP-13 prompt confirmed | BASELINE.md; assessment §23 | historical exact request configuration/reproducibility unavailable |
| K final repository checks | PASS, offline scope | 551 passed, 2 optional skips; new-file Ruff and Black checks pass; zero new legacy correctness-lint diagnostics; diff clean | final_tests.txt; lint_verification.json | hosted CI has not verified this tree; one unchanged unused import in an existing test |
| L Phase-1 readiness assessed | NO-GO for search execution | 0/3 open-ended generation responses contained parseable code; only interface-specific pilot succeeded | validation_summary.json; ENGINEERING_SMOKE_SPEC.md | preregister Phase-1 task family, search design and generation calibration before benchmarking |

## Measured live outcomes

All four requests used OpenRouter `deepseek/deepseek-v4-flash-0731`, temperature 0.6,
top_p 1.0, max_tokens 3000, seed 17, concurrency 1, request timeout 120 seconds,
cache disabled. The first three had an identical frozen prompt; the fourth had
its own separately committed interface-only engineering preregistration. Settings,
seeds, fixture, budget and acceptance thresholds were not changed after results.

| request | provider outcome | program outcome | token usage (prompt / completion / total) |
|---|---|---|---|
| generation_1 | success, length | no parseable code; retained invalid response | 331 / 3000 / 3331 |
| generation_2 | success, length | no parseable code; retained invalid response | 331 / 3000 / 3331 |
| clean_smoke | success, length | no parseable code; retained invalid response | 331 / 3000 / 3331 |
| interface_smoke | success, stop | valid midpoint proposer; 3/3 fixture executions valid | 53 / 110 / 163 |

Total reported tokens: **10,156**. No price/cost estimate is inferred. There were
no transport failures or engineering retries in these live requests. Unit tests
prove that persistent transient failure retains four attempts with delays 2/4/8s.
The successful artifact SHA-256 is
`835a53670003f8e4b6ec9580fe5ef487fc0fd5cfae62186f2f88a861ec8cc76d`.
It proposes [0,0] and scores **2.125** on the public shifted sphere. This is valid
and intentionally trivial, not evidence of discovery, transfer or improvement.

The seed parameter was accepted in real requests. The identical-prompt pair did
not produce programs, so optimizer-code reproducibility cannot be established
from it. Phase 1 must treat outer LLM randomness as uncontrolled and replicate it.
Local optimizer execution is separately checked for repeatability in fresh processes.

Each live directory contains its frozen request, every attempt, safe provider
metadata/usage and response content, plus typed results. The successful directory
also contains optimizer.py and canonical per-seed spec/result exports. The final
runtime re-evaluated that same artifact offline: eight evaluations, value 2.125.

## Changes and scientific scope

Production changes are confined to recursive_opt: new optimizer_program.py; narrow
menu evidence in measurement.py; evaluation observation, invalid aggregation and
legacy return integration in spec.py; an explicit invalid flag in levels.py; and
corrected measurement guidance in tracebench.py. Four test files cover the new
behavior. Runtime inventory/readiness digests were refreshed while retaining the
truthful pre-CI status. See CHANGED_FILES.txt for the exact file inventory.

**MC-d:** automatic for canonical results and legacy run_spec public rows; no opt-in
diagnostic is required. **MC-b:** resolved for evaluators supplying actual behavior
signatures, including optimizer trajectories. Legacy scalar-only evaluators remain
explicitly limited: metric-vector equivalence is not behavioral identity. Disjoint
input panels and observed stochasticity report unknown, never certified headroom.
Source bytes identify candidates but never establish behavioral equivalence.

A regression exposed that a trainer's active memory can be empty even after real
evaluations. The final legacy adapter therefore observes each evaluation before
queue eviction, preserves original outputs/parameters, and excludes final evaluation.
Tests also preserve legacy fallback-on-final-error behavior and returned module types.
The failed queue-based checkpoint was removed; no failing code checkpoint remains on
this branch. No existing assertion was weakened, removed, skipped or marked xfail.

EXP-12 and EXP-13 remain **historical/unresolved**. Their probe README reports the
same OpenRouter DeepSeek model, but EXP-12's result lacks a complete request manifest,
and EXP-13's required ephemeral found-prompt file is absent. Exact historical
reproduction/comparability cannot be established. No new results were appended to
those experiments. H1, H2, H3 and H5 are unchanged.

## Verification commands and results

Baseline command (Python 3.13.13):
```
python -m pytest -q tests/unit_tests/test_recursive_control_plane_v2.py tests/unit_tests/test_recursive_spec.py tests/unit_tests/test_recursive_opt.py tests/unit_tests/test_recursive_final_hardening.py tests/unit_tests/test_recursive_measurement.py tests/unit_tests/test_recursive_surface_guard.py tests/unit_tests/test_recursive_transport.py tests/unit_tests/test_recursive_search_policy.py tests/unit_tests/test_recursive_budget_experiments.py
```
Before production edits: **345 passed, 2 failed, 2 skipped**. Both failures were stale
source-readiness hashes. An earlier plain `pytest` invocation failed collection
because its entrypoint did not include the checkout; `python -m pytest` resolved it.

For the broader suite, missing LangGraph was installed only in a temporary test
venv; no production dependency manifest changed:
```
python -m venv --system-site-packages /tmp/phase0-venv
/tmp/phase0-venv/bin/python -m pip install langgraph
/tmp/phase0-venv/bin/python -m pip install pytest-socket==0.7.0
/tmp/phase0-venv/bin/python -m pip install gepa==0.1.4
```
Installed LangGraph: 1.2.11; GEPA 0.1.4 was already installed. The first broad run
stopped at collection before that dependency was available. A subsequent broad run
was interrupted at 278 passed / 3 skipped while the existing missing-backend test
retried an external provider. Network-disabled and explicit missing-backend probes
also stalled in transport retries; this module exercises an external backend path. The final matrix explicitly omits
that backend-dependent module (two tests); it does not alter its tests or assertions.

Final command:
```
mapfile -t phase0_tests < <(rg --files tests/unit_tests | rg '/test_recursive.*\.py$' | rg -v '/test_recursive_opt_review_regression.py$' | sort)
/tmp/phase0-venv/bin/python -m pytest -q -rs --disable-socket --allow-hosts=127.0.0.1,localhost "${phase0_tests[@]}" tests/unit_tests/test_objectives.py tests/unit_tests/test_evaluators_vector.py tests/unit_tests/test_trainers_multiobjective.py
```
**551 passed, 2 skipped, 0 failed** in 20.55 seconds. The existing optional skips are
`test_tracebench_adapter_collects_trace_type_feedback` and
`test_multitrace_session_uses_available_trace_io_backends` (telemetry imports absent).

New files:
```
python -m ruff check opto/features/recursive_opt/optimizer_program.py tests/unit_tests/test_recursive_optimizer_program.py tests/unit_tests/test_recursive_menu_evidence.py artifacts/optimizer_discovery/phase0.py tests/unit_tests/test_recursive_phase0_calibration.py
python -m black --target-version py310 --check opto/features/recursive_opt/optimizer_program.py tests/unit_tests/test_recursive_optimizer_program.py tests/unit_tests/test_recursive_menu_evidence.py artifacts/optimizer_discovery/phase0.py tests/unit_tests/test_recursive_phase0_calibration.py
```
Both pass. For legacy files, `python -m ruff check --isolated --select E4,E9,F`
was compared against `git show 846580de:<path>` using `--stdin-filename`; zero new
diagnostics (lint_verification.json). Full ambient style lint also reports existing
style debt; unrelated files were not reformatted. `git diff --check` passes.

Real requests (already executed; directories are protected against overwrite):
```
python -m artifacts.optimizer_discovery.phase0 calibrate --requests generation_1 generation_2
python -m artifacts.optimizer_discovery.phase0 calibrate --requests clean_smoke
python -m artifacts.optimizer_discovery.phase0 calibrate --requests interface_smoke
```
The first spec was committed at d6da5ca7; the separate engineering spec at 22071080,
both before their respective requests. All failed and successful observations remain.

## Begin Phase 1 development

Use this exact command to check Patrick's starting artifact against the shared evaluator:
```
python -m artifacts.optimizer_discovery.phase0 evaluate --program artifacts/optimizer_discovery/live/interface_smoke/optimizer.py --seed 0 --budget 8
```
This is the interface handoff, not a Phase-1 search benchmark. A full Phase-1 runner,
task family, splits, seed replication and generation calibration must be preregistered
separately. No human credential/provider action is needed to unblock Phase 0.
