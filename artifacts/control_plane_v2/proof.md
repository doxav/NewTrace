# recursive-opt v2alpha semantic-closure proof

- Branch: `recursive_opt`
- Baseline SHA: `21a0ad3d2f4f835ce2ffb1eef18c36a622265418`
- Semantic-closure implementation SHA: `c92f0af4af3e72a12b0228dbed215f86f8c9475b`
- Completion-audit implementation SHA: `PENDING_COMPLETION_AUDIT_COMMIT`
- Environment: conda `humanllm`, Python 3.12
- Date: 2026-08-22 (Europe/Paris)
- Live provider or paid calls: **none**
- CI workflow: `.github/workflows/recursive-opt-v2.yml`, job `recursive-opt v2 offline (required)`; manual no-paid-call job `GEPA 0.1.4 contract (manual)`

The completion-audit implementation commit will have parent `fc00beac05fc1a73b0c017b7615b56b4162f12f1`. SHA-bearing evidence is a post-commit worktree update because a commit cannot contain its own hash.

## Invariant matrix

| invariant | causal test | command | result | evidence |
|---|---|---|---|---|
| canonical one/two-level schema and compatibility migration | 01–04 | mandated regression | pass | golden/migrated specs |
| actual recursive execution and upstream counterfactual | 05–06 | mandated regression | pass | result lineage |
| real Trace optimize path and engine config | 07–08 | mandated regression | pass | budget/metadata |
| module artifact, inputs, targets, config validation | 09–10, 13b | mandated regression | pass | module artifact |
| portable evaluator and dataset registries | 11–13 | mandated regression | pass | normalized refs/fingerprint |
| exact role clients, preflight, fallbacks, usage | 14–16 | mandated regression | pass | selected models/usage |
| weighted/Pareto objectives, constraints, rollback | 17–20 | mandated regression | pass | canonical evaluation/artifact |
| structural holdout isolation | 21 | mandated regression | pass | phase-context fault injection |
| GEPA holdout externalization and exact 0.1.4 public API | 22/22b | mandated regression | pass | no-provider contract |
| in-run budgets and all policies | 23–24 | mandated regression | pass | budget report |
| scoped deterministic seeds | 25 | mandated regression | pass | deterministic metrics |
| atomic outputs and cross-process resume | 26–27 | mandated regression | pass | persisted run tree |
| knowledge store and every-card binding/lineage | 28–29 | mandated regression | pass | artifact ids in lineage |
| semantic migration classification | 30 | mandated regression | pass | migration reports |
| fixed/Trace/GEPA same-spec result shape | 31 | mandated regression | pass | canonical `RunResult` |
| spec-only notebook and clean offline kernel | 32–33 | mandated regression | pass | smoke notebook |
| footprint limits | 34 | mandated regression | pass | footprint JSON |
| exact corrective SHA | 35 | post-commit mandated regression | pass | readiness JSON |

The 28 baseline diagnoses and their corrective dispositions are mapped individually in `readiness_audit.md`.

## Verification commands

Mandated recursive regression (network disabled; localhost allowed only for the notebook kernel):

```bash
env -u OPENAI_API_KEY -u OPENROUTER_API_KEY -u ANTHROPIC_API_KEY \
  -u GOOGLE_API_KEY -u TAVILY_API_KEY \
  RECURSIVE_OPT_LIVE=0 PYTHONHASHSEED=0 PYTHONPATH=. \
  python -m pytest -q --disable-socket \
  --allow-hosts=127.0.0.1,localhost \
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

Pre-commit authoritative completion-audit result: **283 passed, 3 skipped, 1 warning in 11.72s**.

Complete unit suite:

```bash
env -u OPENAI_API_KEY -u OPENROUTER_API_KEY -u ANTHROPIC_API_KEY \
  -u GOOGLE_API_KEY -u TAVILY_API_KEY \
  RECURSIVE_OPT_LIVE=0 PYTHONHASHSEED=0 PYTHONPATH=. \
  python -m pytest -q --disable-socket \
  --allow-hosts=127.0.0.1,localhost tests/unit_tests
```

Pre-commit authoritative completion-audit result: **463 passed, 4 skipped, 1 warning in 33.27s**.

Pre-commit skips: two tests require optional graph/telemetry backends; one is the deliberately post-commit final-SHA gate; the complete-unit-only fourth skip requires the Graphviz `dot` executable. GEPA 0.1.4 and LangGraph contract tests ran rather than skipping. The single warning is LangGraph's pending default change for serializer `allowed_objects`.

Lint for every changed Python file and the retained graph contract:

```bash
ruff check opto/features/graph \
  opto/features/recursive_opt/spec.py \
  opto/features/recursive_opt/__init__.py \
  tests/unit_tests/test_recursive_control_plane_v2.py
```

Result: **All checks passed**. A broader unchanged recursive-opt directory scan reports nine baseline findings in untouched files (`capabilities.py`, `experiments.py`, `inspect_utils.py`, `levels.py`, and `tracebench.py`); they are not suppressed or mixed into this corrective patch.

Post-commit exact-SHA verification: **pending completion-audit commit**.

## Migration and footprint

All 85 tracked historical files are classified: `execution_replayable=0`, `normalized_only=10`, `missing_dependency=23`, `historical_only=46`, `invalid=0`, `local_nonportable=6`. The representative config, family-policy, and prior specs normalize but cannot be replayed faithfully because exact dataset/evaluator/provider dependencies are absent; precise dependencies and paths are in `migration_report.json.representatives`. The deterministic UC4/UC14 notebook fixtures explicitly remain non-historical controls.

| measure | baseline | corrective worktree | delta |
|---|---:|---:|---:|
| recursive-opt runtime lines | 8,803 | 8,750 | -53 |
| `spec.py` lines | 2,688 | 2,633 | -55 |
| notebook code lines | 16 | 16 | 0 |
| public package exports | 114 | 114 | 0 |

The graph package is byte-clean against the baseline SHA and was not expanded.

## Changed scope and limitations

The intended corrective commit contains only the canonical runtime/export files, the control-plane test matrix, smoke notebook/goldens, migrated normalized specs, and control-plane evidence. Exact staged paths and post-commit `git status --short` are recorded after staging/commit.

The original semantic-closure implementation changed 29 files. The completion-audit correction is deliberately surgical and changes the runtime, its causal test matrix, and these readiness records. Exact completion-audit commit statistics and final staged evidence are recorded after the commit. The SHA-bearing evidence is staged separately and intentionally not committed because advancing HEAD would invalidate the exact-SHA gate.

```text
pending post-commit SHA evidence
```

Unrelated, pre-existing user-owned experiment/config/output paths remain untracked and are excluded from the corrective commit. The exact tracked status is recorded after SHA-bound verification.

Limitations: no live OpenRouter run, no paid GEPA optimization, and no historical efficacy claim. The real installed GEPA API is checked only where it can run without a provider. Historical config/family-policy/prior behavior remains non-replayable without the precisely listed dependencies. This gate establishes readiness for Prompt 18; it does not begin that experiment.
