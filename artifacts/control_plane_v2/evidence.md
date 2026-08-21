# Control plane v2 implementation evidence

Baseline SHA: `6fc278a398709fe79a0fc9be22bae99bffd8cba6`

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
