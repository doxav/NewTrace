# recursive-opt v2alpha proof report

- Branch: `recursive_opt`
- Baseline/current HEAD: `6fc278a398709fe79a0fc9be22bae99bffd8cba6`
- Environment: conda `humanllm`, Python `3.12.13`
- Date: 2026-08-21
- Live paid/provider calls executed: **none**

## Invariant matrix

| invariant | test | command | result | artifact |
|---|---|---|---|---|
| Exact version/kind, migration, defaults | normalization/migration tests | offline CI command below | pass | `control_plane_v2alpha.md` |
| Strict unknown keys; namespaced extensions | normalization rejection test | offline CI | pass | golden specs |
| JSON round-trip, immutability, stable SHA-256 | normalization/fingerprint tests | offline CI | pass | golden specs |
| No callable, secret value, or arbitrary import ref | strict normalization tests | offline CI | pass | `evidence.md` |
| Generic `trace.Module` build/snapshot/restore | module registry contract | offline CI | pass | v2alpha tests |
| Multi-component modules | component artifact contract | offline CI | pass | v2alpha tests |
| Deterministic seeds/arms/matrix `ExecutionPlan` | plan expansion test | offline CI | pass | `explain_spec` output |
| Scalar, weighted vector, Pareto capability checks | objective compiler tests | offline CI | pass | `opto/trainer/objectives.py` |
| Negative score valid; invalidity explicit | legacy evaluation adapter test | offline CI | pass | canonical `EvaluationResult` |
| Hard constraints before selection; verbal feedback retained | objective selection tests | offline CI | pass | canonical `EvaluationResult` |
| LLM profiles/roles/overrides and exact model preflight | role/profile test | offline CI | pass | normalized spec |
| Runtime-owned usage by forward/optimizer/feedback/judge | role usage and budget tests | offline CI | pass | `RunResult.usage/budget` |
| Promoted-only knowledge, scope, rollback | MemoryLite knowledge test | offline CI | pass | `KnowledgeCard` ledger |
| Typed causal binding and counterfactual lineage | binding tests | offline CI | pass | `RunResult.lineage` |
| Holdout denied during fit/proposal/induction/selection | capability + fault injection | offline CI | pass | canonical error result |
| Fixed and Trace engine contract | same-spec engine test | offline CI | pass | `RunResult` |
| GEPA OptimizeAnything projection and result conversion | GEPA fake contract | offline CI | pass | pinned `gepa==0.1.4` extra |
| Identical spec across fixed/Trace/GEPA | three-arm contract test | targeted test | `1 passed` | v2alpha tests |
| Fake graph executor and optional LangGraph | graph contracts + ABC probe | mandated regression | pass | minimal graph package |
| Graph JSON config/state/artifact, codecs, capabilities | fake graph snapshot test | offline CI | pass | graph artifact v1 |
| Budget accounting | role/candidate/evaluation/wall accounting test | offline CI | pass | `RunResult.budget` |
| Resume idempotence | fingerprinted result-store test | offline CI | pass | `runtime.resume` |
| Notebook strict AST audit | notebook AST test | offline CI | pass | spec-only notebook |
| Notebook clean-kernel offline | nbclient execution test | offline CI | pass | spec-only notebook |
| UC4 positive / UC14 negative controls | golden control test | offline CI | pass | `golden_specs/` |
| Historical file classification and immutable sources | migration report test | offline CI | pass | `migration_report.json` |
| Required offline CI has no external socket | pytest-socket invocation | exact local CI command | `168 passed` | workflow |
| Footprint measured before/after | footprint JSON comparison | measurement command | `+103` exception | footprint reports + ADR |

## Exact verification commands and results

Baseline and provenance commands/results, including `git status`, branch, SHA, log, Python, `pip freeze`, targeted baseline, complete-suite attempt, durations, and pristine-SHA reproductions are preserved verbatim in `baseline.md`.

Final mandated recursive regression plus v2alpha and graph probes:

```bash
PYTHONPATH=. pytest -q \
  tests/unit_tests/test_recursive_spec.py \
  tests/unit_tests/test_recursive_opt.py \
  tests/unit_tests/test_recursive_budget_experiments.py \
  tests/unit_tests/test_recursive_field_activation.py \
  tests/unit_tests/test_recursive_numeric_optimizers.py \
  tests/unit_tests/test_recursive_opt_three_way.py \
  tests/unit_tests/test_recursive_opt_traces.py \
  tests/unit_tests/test_objectives.py \
  tests/unit_tests/test_evaluators_vector.py \
  tests/unit_tests/test_trainers_multiobjective.py \
  tests/unit_tests/test_recursive_control_plane_v2.py \
  tests/unit_tests/test_recursive_opt_abc_probe.py
```

Result: **273 passed, 2 skipped in 8.81s**.

Required network-blocked CI contract:

```bash
RECURSIVE_OPT_LIVE=0 PYTHONHASHSEED=0 PYTHONPATH=. pytest -q \
  --disable-socket --allow-hosts=127.0.0.1,localhost \
  tests/unit_tests/test_recursive_control_plane_v2.py \
  tests/unit_tests/test_recursive_spec.py \
  tests/unit_tests/test_objectives.py \
  tests/unit_tests/test_evaluators_vector.py \
  tests/unit_tests/test_trainers_multiobjective.py
```

Latest final rerun: **168 passed, 1 warning in 4.79s**.

Complete unit suite, also with keys removed and external sockets blocked:

```bash
env -u OPENAI_API_KEY -u OPENROUTER_API_KEY -u OPENAI_ADMIN_KEY \
  RECURSIVE_OPT_LIVE=0 PYTHONPATH=. pytest -q \
  --disable-socket --allow-hosts=127.0.0.1,localhost tests/unit_tests
```

Result: **453 passed, 3 skipped in 29.87s**.

Complete repository suite attempt used the same guards plus `--maxfail=30 tests`. Result: **30 failed, 22 passed, 132 skipped in 7.28s**. Failure families are outside recursive-opt and match baseline evidence: Flows mock type mismatch, BBH/optimizer tests requiring configured LLM behavior, and OPRO v2. They were not suppressed or repaired.

Lint:

```bash
ruff check opto/features/graph opto/features/recursive_opt/__init__.py \
  opto/features/recursive_opt/memory.py opto/features/recursive_opt/runmode.py \
  opto/features/recursive_opt/spec.py opto/features/recursive_opt/traces.py \
  opto/trainer/objectives.py examples/recursive_opt_abc_probe.py \
  tests/unit_tests/test_recursive_control_plane_v2.py
```

Result: **All checks passed**. All workflow YAML files also parsed successfully.

## Files and API surface

Tracked runtime/client snapshot relative to HEAD: **12 files changed, 2,612 insertions, 2,420 deletions**. Files are:

- `opto/features/recursive_opt/{__init__,memory,runmode,spec,traces}.py`
- `opto/trainer/objectives.py`
- `opto/features/graph/{__init__,adapter,module}.py`
- `examples/recursive_opt_use_cases.ipynb`
- `examples/recursive_opt_abc_probe.py`
- `pyproject.toml`

New test/CI/evidence files total 2,789 physical lines before this proof file: the v2alpha test module, required workflow, ADR, baseline/evidence/footprint/migration reports, two notebook goldens, and 16 migrated normalized specs.

Public APIs added under recursive-opt: `SCHEMA_VERSION`, `SPEC_KIND`, `CANONICAL_SPEC_BLOCKS`, `ExecutionPlan`, `RunResult`, `EvaluationResult`, `DatasetAccess`, `ModuleRegistryEntry`, `EngineRegistryEntry`, `KnowledgeCard`, normalization/migration/explanation, registry/build/snapshot/restore/plan execution, objective/binding/role/preflight/knowledge helpers, and canonical evaluation normalization. Graph adds `GraphExecutor`, `GraphAdapter`, `GraphModule`, optional `LangGraphAdapter`, and three artifact/codec constants.

Removed paths: the notebook's 75 orchestration helpers; its direct level/optimizer/memory/private calls; manual seed/arm/budget loops; the redundant `graph_to_module` helper; and 4,639 lines of the 5,023-line experimental graph/Trace-IO import. No baseline exported recursive-opt API was removed.

## Footprint

| measure | before | after | delta |
|---|---:|---:|---:|
| recursive-opt runtime physical lines | 6,824 | 8,803 | +1,979 |
| notebook code lines | 1,892 | 16 | -1,876 |
| combined budget scope | 8,716 | 8,819 | **+103** |
| notebook helpers | 75 / 764 lines | 0 / 0 lines | -75 / -764 |
| public recursive-opt functions/classes | 132 | 154 | +22 |
| experimental source retained | 5,023 imported | 384 graph | -4,639 |

The +103 default-budget miss is explicit. `control_plane_v2alpha.md` itemizes every physical-line delta by file and required invariant; tests/docs/graph/shared objectives are not used to hide it.

## Historical migration and live limits

`migration_report.json` classifies all 85 tracked historical files: 46 replayable artifact ledgers, 16 migrated-replayable specs, 23 missing-dependency metadata specs, and zero in the remaining required categories. Original SHA-256 values are verified; migrated files are separate. The 4,291 untracked user-output candidates were left untouched.

No live OpenRouter or paid LLM inference was executed. GEPA was exercised through an injected deterministic `optimize_anything` contract; the manual CI job installs and verifies exactly `gepa==0.1.4` but was not run locally. Historical UC4 and UC14 live runs are explicitly non-executable as faithful replays because the exact historical dependencies are not pinned; the notebook controls are not substituted historical claims.
