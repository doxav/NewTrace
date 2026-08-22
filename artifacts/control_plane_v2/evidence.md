# Control plane v2 implementation evidence

Original implementation baseline: `6fc278a398709fe79a0fc9be22bae99bffd8cba6`

Semantic-closure baseline: `21a0ad3d2f4f835ce2ffb1eef18c36a622265418`

Semantic-closure implementation: `c92f0af4af3e72a12b0228dbed215f86f8c9475b`

Completion-audit correction: `05dabf68e77ef2b9c59a8fc20c68bf4f8d2c1eaf`

Prompt 17.7 starting HEAD: `14b832c82341bbc55e9c662ebaebcba4e3e8e95b`

## GEPA 0.1.4 public evaluator contract hotfix — 2026-08-22

- Published GEPA 0.1.4 accepts a public evaluator result of `score | (score, side_info)`. Its actual wheel-local `EvaluatorWrapper` converts the public pair to the internal `(score, None, side_info)` triple consumed by `OptimizeAnythingAdapter`; a public 3-tuple reproduces `ValueError: too many values to unpack (expected 2)`.
- `_run_gepa_engine` now returns only `float(score), info`. The candidate is owned by GEPA, canonical metrics remain at `side_info["metrics"]`, and no `side_info["scores"]` falsely exposes minimize metrics as higher-is-better Pareto axes.
- The corrected tests cover the public callback, actual wrapper, internal adapter, and actual public `optimize_anything()` entry point with a one-evaluation budget, deterministic local reflection callable, removed keys, and blocked sockets.
- Weighted accuracy/maximize plus `forward_token_ratio`/minimize preserves direction; invalid candidates retain the explicit floor and `valid=false`.
- Network-blocked results: focused seam `4 passed`; control plane `45 passed`; final hardening `21 passed`; recursive spec `47 passed`; objective matrix `89 passed`; all recursive units `224 passed, 2 skipped`; complete units `487 passed, 3 skipped`; clean-kernel notebook `1 passed`. No GEPA test skipped. Ruff and both diff checks passed.
- Runtime footprint remains 8,850 lines (zero-line hotfix delta). Authoritative digests are `runtime_tree_sha256=5b460d771ca0b0f9bd914b2c8330860e6f5771a8447d40e50db0d554986e0642` and `registry_sha256=f1a94b02c94607c2d22e6f10bce25b10d9b642814915ec01c72765280be26fa4`.
- Pre-push readiness remains false pending an observed green `recursive-opt v2 offline (required)` run.

## Prompt 17.7 final-hardening gate — 2026-08-22

- A counting stochastic workflow proves fixed, Trace, and injected GEPA execute exactly one workflow forward per evaluator invocation. The output evaluator receives the exact traced output, its evaluation attachment retains the real parameter dependency, and standard optimizer feedback reaches that parameter.
- Canonical Trace disables legacy environment overrides. Changing all nine listed optimizer/trainer/model environment variables leaves the normalized fingerprint, resolved trainer/optimizer/iteration/candidate/kwargs values, selected artifact, and result unchanged.
- Per-profile and per-fallback `request_params` are normalized, fingerprinted, manifested, and sent to fake providers. Recursive identity, endpoint, credential, and secret overrides fail validation.
- Resolved module/evaluator/dataset/codec/engine provenance is persisted. After the GEPA 0.1.4 public-contract hotfix, authoritative digests are `runtime_tree_sha256=5b460d771ca0b0f9bd914b2c8330860e6f5771a8447d40e50db0d554986e0642` and `registry_sha256=f1a94b02c94607c2d22e6f10bce25b10d9b642814915ec01c72765280be26fa4` for the golden readiness spec. Replacing evaluator code behind the same ref invalidates resume.
- Candidate accounting distinguishes reserved, proposed, and evaluated counts. Runtime is 8,850 lines, +100 from Prompt 17.7's 8,750-line start and at its +100 bound.
- Network-blocked results: focused `21 passed`; mandated regression `305 passed, 2 skipped`; complete units `485 passed, 3 skipped`; clean-kernel notebook `1 passed`; isolated-worktree readiness `26 passed`. Ruff, `git diff --check`, and workflow YAML validation pass. No provider call occurred.
- The required workflow installs `.[gepa]` and runs the hardening suite, but this unpushed workflow cannot be invoked or queried here. The CI and Prompt-18 readiness gates are therefore false pending an observed green `recursive-opt v2 offline (required)` run.

## Completion-audit gate — 2026-08-22

- A scripted optimizer through the real `recursive_opt.optimize.optimize` path initially changed both the declared target and a protected component. The failing causal test exposed that candidate application did not honor `ParameterNode.trainable`; `_EvaluatedModule.parameters()` now restricts the trainer to declared targets, and the same test proves the target changes while the protected component does not.
- Invalid high-metric candidates are floored before both Trace ranking and GEPA projection. Aggregation, feedback channels, intent fingerprinting, exact module snapshot/restore, budget counters, and migration representatives now have direct assertions.
- Registered dataset resolution and optimizer construction are executed inside the unit seed scope and are tested for same-seed equality and different-seed divergence across Python and NumPy RNGs.
- Holdout materialization occurs only after fitting and candidate selection. Test-override identity is persisted in the resolved manifest.
- Level and final resume payloads carry canonical SHA-256 result digests. A second process replaces the evaluator with a raising implementation to prove a valid resume makes zero evaluator calls; partial and tampered artifacts are rejected and repaired.
- The pre-commit network-blocked mandated regression passed: `283 passed, 3 skipped, 1 warning in 11.72s`. The complete unit suite passed: `463 passed, 4 skipped, 1 warning in 33.27s`. The control-plane file alone passed `42 passed, 1 skipped in 5.90s`; the legacy spec file passed `47 passed in 1.92s`; changed-file/graph Ruff and `git diff --check` passed.
- Exact-SHA verification at `05dabf68e77ef2b9c59a8fc20c68bf4f8d2c1eaf` passed: mandated regression `284 passed, 2 skipped, 1 warning in 11.08s`; complete unit suite `464 passed, 3 skipped, 1 warning in 30.12s`; changed-file/graph Ruff and `git diff --check` passed again. The SHA gate ran rather than skipping.
- No live provider or paid call was executed.

## Corrective semantic-closure gate — 2026-08-22

The phase log below is retained as historical implementation evidence. This section supersedes its execution-shape, GEPA holdout, migration-category, footprint, and final-test claims where they differ.

- Canonical normalization now always produces top-level fully specified `levels`; flat v2 and legacy inputs migrate into that shape.
- `run_spec` has one route: migrate → normalize → compile immutable execution units/level plans → ordered multilevel execution. The independent legacy orchestration loop was removed; migrated legacy levels execute inside the canonical runner.
- Trace invokes the existing real `optimize` path over actual trainable `ParameterNode`s. Config, initial artifact, inputs, targets, evaluator/dataset refs, objective selection, role clients, knowledge, bindings, budgets, seeds, persistence, and resume are all causal.
- Fit/proposal contexts contain no holdout data. GEPA 0.1.4 receives only train/validation and holdout is evaluated after best-candidate extraction.
- The installed GEPA package contract was exercised without a provider: exact version/imports, config construction, keyword-only evaluator triple, batch evaluation, result construction, and best-candidate extraction.
- Migration categories are now `execution_replayable=0`, `normalized_only=10`, `missing_dependency=23`, `historical_only=46`, `invalid=0`, and `local_nonportable=6`. Precise config/family-policy/prior missing dependencies are in `migration_report.json.representatives`.
- The graph package is unchanged at the semantic-closure baseline; the imported files match the tracked minimal contract.
- Current footprint is 8,730 recursive-opt runtime lines and 2,613 `spec.py` lines, respectively 73 and 75 below the `21a0ad3` baseline. Public package exports remain 114.
- After synchronizing footprint evidence, the pre-commit network-blocked mandated regression passed: `277 passed, 3 skipped, 1 warning in 11.14s`. The complete unit suite passed: `457 passed, 4 skipped, 1 warning in 33.28s`. The authoritative post-commit reruns are recorded in `proof.md`.
- No live provider or paid call was executed.
- Exact-SHA post-commit verification at `c92f0af4af3e72a12b0228dbed215f86f8c9475b`: mandated regression `278 passed, 2 skipped, 1 warning in 11.17s`; complete unit suite `458 passed, 3 skipped, 1 warning in 32.03s`; changed-file/graph Ruff `All checks passed`.

## Phase 0 — baseline and footprint

- Targeted baseline before implementation: `234 passed, 2 skipped in 8.33s` before graph import; `236 passed in 7.04s` after the user supplied compatible graph/Trace IO source.
- Complete suite: initial collection failure documented; after imports, 16 failures reproduced independently at pristine SHA and a live-network retry was interrupted.
- Environment and exact outputs: `baseline.md`.
- Starting footprint: `code_footprint_before.json`.

## Phases 1–2 — ADR, versioning, migration, normalization

Red test:

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_recursive_control_plane_v2.py
```

Result before implementation: `6 failed` because `normalize_spec` and `migrate_legacy_spec` did not exist.

Targeted green test:

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_recursive_control_plane_v2.py tests/unit_tests/test_recursive_spec.py
```

Result: `53 passed in 1.50s`.

Regression:

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_recursive_spec.py tests/unit_tests/test_recursive_opt.py tests/unit_tests/test_recursive_budget_experiments.py tests/unit_tests/test_recursive_field_activation.py tests/unit_tests/test_recursive_numeric_optimizers.py tests/unit_tests/test_recursive_opt_three_way.py tests/unit_tests/test_recursive_opt_traces.py tests/unit_tests/test_objectives.py tests/unit_tests/test_evaluators_vector.py tests/unit_tests/test_trainers_multiobjective.py tests/unit_tests/test_recursive_control_plane_v2.py
```

Result: `242 passed in 6.75s`.

Lint:

```bash
ruff check opto/features/recursive_opt/spec.py opto/features/recursive_opt/__init__.py tests/unit_tests/test_recursive_control_plane_v2.py
```

Result: `All checks passed!`.

Footprint checkpoint: recursive-opt runtime grew from 6,824 to 7,250 physical lines (`+426`: `spec.py` +414 net, `__init__.py` +12). This is temporary implementation debt: the final target remains neutral/negative after replacing legacy orchestration and notebook helpers. No claim of final footprint compliance is made.

Covered invariants: exact schema/kind, legacy migration, materialized canonical blocks, strict unknown-key handling, namespaced extensions, JSON round-trip, stable SHA-256 fingerprint, deep immutability, callable rejection, secret-value rejection, strict versioned refs, exact OpenRouter default model, resolved LiteLLM model, environment credential reference, and four materialized LLM roles.

## Phase 3 — module registry, snapshots, and ExecutionPlan

Red result: the expanded control-plane test file reported `3 failed, 6 passed`; `build_module` and `compile_plan` did not exist.

Targeted result:

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_recursive_control_plane_v2.py tests/unit_tests/test_recursive_spec.py
```

Result: `56 passed in 1.53s`.

Regression result: mandated recursive suite plus the v2alpha tests, `245 passed in 6.71s`.

Lint result: `All checks passed!` for the changed runtime, exports, and v2alpha tests.

Footprint checkpoint: 7,573 recursive-opt runtime lines, `+749` from the 6,824-line starting point. The phase adds one lightweight registry-entry type with two concrete built-in consumers, one immutable plan type consumed by compilation/explanation, generic multi-component `trace.Module` construction, JSON snapshot/restore validation, strict module-ref resolution, and deterministic internal expansion. It remains temporary debt pending notebook/legacy deletion.

## Phase 4 — EvaluationResult and objective compilation

Red result: `3 failed, 9 passed`; the canonical result adapter and spec objective compiler did not exist.

Targeted objective result: `101 passed in 1.99s` across v2alpha, scalar/vector evaluator, objective, and multi-objective trainer tests.

Regression result: mandated recursive suite plus v2alpha tests, `248 passed in 6.61s`.

Lint result after export cleanup: `All checks passed!`.

Footprint checkpoint: 7,650 recursive-opt lines (`+826` from start) plus 153 net lines in the existing shared `opto/trainer/objectives.py`. This phase deliberately reuses `ObjectiveConfig`, `select_best`, weighted scalarization, and Pareto ranking rather than duplicating them under recursive-opt. Added behavior is limited to canonical result normalization, explicit validity, role-structured usage, hard-constraint filtering, and spec-to-`ObjectiveConfig` compilation.

Covered invariants: legacy float/dict/tuple adaptation, valid negative score, invalidity independent of score, natural-language feedback retention, no implicit dict averaging, scalar/weighted/Pareto compilation, minimize directions, hard constraints before selection, and compile-time objective capability rejection.

## Phase 5 — LLM roles and runtime usage

Red result: `2 failed, 12 passed`; normalized roles were unresolved profile names and no runtime usage adapter existed.

Targeted result: `24 passed in 3.91s` across v2alpha and budget/experiment tests.

Regression result: mandated recursive suite plus v2alpha tests, `250 passed in 6.68s`.

Lint result: `All checks passed!`.

Footprint checkpoint: 7,786 recursive-opt lines (`+962` from start); `runmode.py` accounts for 67 net lines of the delta. Role configs are now fully resolved into the fingerprint, level-local overrides are explicit, exact resolved models are preflighted once, and one thin runtime wrapper accumulates provider-reported usage. Guides do not duplicate token estimation.

## Phase 6 — knowledge policy on MemoryLite

Red result: test collection failed because `KnowledgeCard` did not exist.

Targeted result: `131 passed in 1.71s` across v2alpha, recursive-opt, and recursive-spec tests. Broader memory/search regression: `139 passed in 5.24s`.

Mandated regression result: `251 passed in 6.67s`. Lint: `All checks passed!`.

Footprint checkpoint: 7,969 recursive-opt lines (`+1,145` from start); `memory.py` adds 138 net lines. No second store was created. The existing artifact ledger now carries artifact type, status, scope, and supersession metadata; retrieval adds promoted-status and scope filters; explicit status transitions support rollback; and runner-side `retrieve_knowledge` applies the spec policy before module construction.

Covered invariants: knowledge-card schema, promoted-only default reuse, scoped retrieval, persistence, explicit negative-transfer rollback, and no retrieval hidden inside a `trace.Module`.

## Phase 7 — causal bindings and holdout capabilities

Red result: `3 failed, 15 passed`; decorative dependencies were accepted and binding/dataset capability APIs did not exist.

Targeted result: `134 passed in 1.45s` across v2alpha, recursive-spec, and recursive-opt tests.

Mandated regression result: `254 passed in 6.70s`. Lint: `All checks passed!`.

Footprint checkpoint: 8,183 recursive-opt lines (`+1,359` from start). The delta implements a small typed codec registry (not a model hierarchy), causal application with artifact lineage, strict rejection of unbound dependency edges, and a phase-capability dataset gate shared by future engines.

Covered invariants: decorative dependency rejection, explicit ordering-only exception, versioned codec refs, counterfactual output-to-input propagation, codec type checking, per-injection lineage, and holdout denial during fit/proposal/induction/candidate selection with access restricted to final evaluation/promotion/report.

## Phase 8 — engine registry, Trace contract, canonical RunResult

Red result: `2 failed, 18 passed`; `run_spec` had no v2 resource/execution path.

Targeted result: `146 passed in 5.08s` across v2alpha, recursive-spec, recursive-opt, and budget experiments.

Mandated regression result: `256 passed in 6.55s`. Lint: `All checks passed!`.

Footprint checkpoint: 8,468 recursive-opt lines (`+1,644` from start). The phase adds one lightweight engine-entry type with two concrete consumers (`fixed`, `trace`), a canonical `RunResult`, shared fixed/Trace execution, capability checks during plan compilation, JSON export, rich evaluator retention, runner-side binding/knowledge injection, and canonical error results.

Contract evidence: the same normalized spec executes fixed and Trace arms with only `engine.name` changed; Trace accepts an arbitrary registered `trace.Module`, its fit hook changes a heterogeneous component artifact, vector metrics and natural feedback survive, role usage and rich trace survive, and fault-injected holdout access during fit produces a failed canonical result.

## Phase 9 — optional GEPA OptimizeAnything adapter

Primary API evidence: official GEPA `optimize_anything` documentation and the official `gepa-ai/gepa` immutable release list were checked on 2026-08-21. The documented API accepts `seed_candidate`, scalar-or-`(score, info)` evaluator, `dataset`, `valset`, `test_set`, `objective`, and `config`. Latest release observed: `0.1.4`.

Red result: `3 failed, 20 passed`; the GEPA engine was unregistered and the optional dependency was unpinned.

Targeted result: `216 passed in 1.55s` across v2alpha, recursive spec/opt, objectives, and vector evaluators.

Mandated regression result: `259 passed in 6.90s`. Lint: `All checks passed!`.

Footprint checkpoint: 8,650 recursive-opt lines (`+1,826` from start). `gepa==0.1.4` is an optional extra only. The adapter imports GEPA lazily with the original `ImportError` as cause, verifies the installed version, converts canonical component artifacts to seed candidates, rebuilds the same registered `trace.Module` for evaluator calls, projects weighted/scalar objectives explicitly, retains validity/metrics/feedback/trace/usage/artifacts/error in GEPA info, passes train/validation/test separately, and converts `best_candidate` into canonical artifact/`RunResult`. Pareto is rejected at compile time because the engine does not declare that objective capability.

## Phase 10 — minimal graph integration

Red result: `2 failed, 23 deselected`; the experimental import exposed no `GraphExecutor` and no serializable adapter snapshot.

Targeted result: the dependency-free fake executor and optional LangGraph smoke contracts passed (`2 passed, 23 deselected`). Existing LangGraph ABC behavior plus multi-trace fallback also passed (`9 passed`).

Regression command: the mandated recursive suite plus v2alpha and ABC graph tests. Result: `266 passed, 2 skipped in 6.27s`. Lint: `All checks passed!` for graph, spec, traces, the compatibility probe, and v2alpha tests.

The user-supplied experimental source was reduced from 5,023 lines (1,019 graph + 4,004 Trace IO) to 384 graph lines. No `opto.trace.io` source was retained. The accepted graph surface is only `GraphExecutor`, `GraphAdapter`, `GraphModule`, and the optional `LangGraphAdapter`; it has explicit versioned input/output codecs, capabilities, JSON config/state/artifact, and snapshot/restore. Optional dependency loading catches only `ImportError` and preserves its cause. The module registry provides `recursive_opt.module.graph@1`, resolving executors only from explicit runtime resources.

The redundant `graph_to_module` path was removed from `recursive_opt/traces.py`. The historical ABC live optimizer import is now lazy and raises with its original import cause; supported graph execution goes through the registry. Recursive-opt runtime is 8,689 lines (`+1,865` from start, only `+39` during this phase), still temporary debt pending notebook migration.

## Phase 11 — spec-only notebook

Red result: notebook-focused v2alpha tests reported `2 failed, 1 passed`; the old notebook contained function definitions and no `UC4_SPEC`/`UC14_SPEC` control-plane specs.

Green result: AST audit, UC4 positive control, UC14 negative-transfer control, and a clean Python kernel execution passed (`4 passed`). The final notebook has 16 code lines, zero orchestration helpers, zero direct optimizer/level/memory calls, zero loops, and no environment mutation. It only loads two complete fingerprinted specs, calls `normalize_spec`, `explain_spec`, `run_spec`, and displays results.

The registered `recursive_opt.evaluator.reasoning@1` makes the controls deterministic without embedding a callable. UC4 returns valid accuracy 1.0; UC14 returns `constraint_failed` after its deliberately unsupported transfer policy scores 0.0. Both golden specs say `historical_replay=false`.

## Phase 12 — historical migration

Every git-tracked `examples/**/artifacts.jsonl` and `*spec.json` was classified. Result: 85 files total — 46 `replayable`, 16 `migrated_replayable`, and 23 `missing_dependency`; the other required categories are present with zero counts. Source SHA-256 values are rechecked by the migration test. The 16 normalized copies live separately under `artifacts/control_plane_v2/migrated_specs/`; no historical file was changed and no evaluator/module was invented.

The workspace contained 4,291 additional untracked candidate outputs. They are user-owned data outside the git baseline and were not modified or silently included in repository evidence. UC4 and UC14 live historical replays are explicitly `missing_dependency`; the exact pinned historical evaluator/provider execution is absent for UC4 and no complete tracked UC14 execution spec exists.

Targeted notebook/migration result: `4 passed in 6.34s`.

## Phase 13 — budget, resume, CI, and final regressions

Red result: `2 failed`; `RunResult.budget` had no observed accounting and repeated `runtime.resume=true` runs invoked the evaluator twice.

Green result: role wrappers count calls once; results account optimizer/evaluation role calls, candidates, evaluation runs, tokens, and wall time against limits; an explicit mutable result store reuses only a matching fingerprinted `RunResult`. Focused result: `3 passed`.

Cross-engine contract: one spec with fixed, Trace, and injected GEPA arms passed; module/evaluator/objective are identical and only `engine.name` changes.

Mandatory offline CI command (external sockets disabled except localhost for the notebook kernel):

```bash
RECURSIVE_OPT_LIVE=0 PYTHONHASHSEED=0 PYTHONPATH=. pytest -q \
  --disable-socket --allow-hosts=127.0.0.1,localhost \
  tests/unit_tests/test_recursive_control_plane_v2.py \
  tests/unit_tests/test_recursive_spec.py \
  tests/unit_tests/test_objectives.py \
  tests/unit_tests/test_evaluators_vector.py \
  tests/unit_tests/test_trainers_multiobjective.py
```

Latest final rerun: `168 passed, 1 warning in 4.79s`. Complete unit suite under the same network block: `453 passed, 3 skipped in 29.87s`.

The complete repository suite was also attempted with network blocked and no keys. It stopped at the requested `--maxfail=30`: `30 failed, 22 passed, 132 skipped in 7.28s`. Failures are outside the recursive-opt unit suite and are the same families documented at baseline (Flows mock type mismatch, BBH/optimizer tests requiring LLM configuration, and OPRO v2); no such failure is suppressed or claimed green.

Final mandated recursive regression plus v2alpha and ABC graph tests: `273 passed, 2 skipped in 8.81s`. Ruff: `All checks passed!`. Workflow YAML parsed successfully.

Final footprint: recursive runtime 8,803 lines; notebook 16 code lines; combined 8,819 versus 8,716 baseline (`+103`). The notebook removed 1,876 code lines and all 75 helpers. The positive exception is itemized by file and invariant in the ADR and recorded in `code_footprint_after.json`.
