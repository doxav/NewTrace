# recursive-opt v2alpha semantic-closure proof

- Branch: `recursive_opt`
- Baseline SHA: `21a0ad3d2f4f835ce2ffb1eef18c36a622265418`
- Corrective implementation SHA: `c92f0af4af3e72a12b0228dbed215f86f8c9475b`
- Environment: conda `humanllm`, Python 3.12
- Date: 2026-08-22 (Europe/Paris)
- Live provider or paid calls: **none**
- CI workflow: `.github/workflows/recursive-opt-v2.yml`, job `recursive-opt v2 offline (required)`; manual no-paid-call job `GEPA 0.1.4 contract (manual)`

The implementation commit has parent `21a0ad3d2f4f835ce2ffb1eef18c36a622265418`. SHA-bearing evidence is a post-commit worktree update because a commit cannot contain its own hash.

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

Pre-commit authoritative result after evidence synchronization: **277 passed, 3 skipped, 1 warning in 11.14s**. A repeat with `-rs` reported the same 277/3 result in 11.31s.

Complete unit suite:

```bash
env -u OPENAI_API_KEY -u OPENROUTER_API_KEY -u ANTHROPIC_API_KEY \
  -u GOOGLE_API_KEY -u TAVILY_API_KEY \
  RECURSIVE_OPT_LIVE=0 PYTHONHASHSEED=0 PYTHONPATH=. \
  python -m pytest -q --disable-socket \
  --allow-hosts=127.0.0.1,localhost tests/unit_tests
```

Pre-commit authoritative result after evidence synchronization: **457 passed, 4 skipped, 1 warning in 33.28s**.

Pre-commit skips: two tests require optional graph/telemetry backends; one is the deliberately post-commit final-SHA gate; the complete-unit-only fourth skip requires the Graphviz `dot` executable. GEPA 0.1.4 and LangGraph contract tests ran rather than skipping. The single warning is LangGraph's pending default change for serializer `allowed_objects`.

Lint for every changed Python file and the retained graph contract:

```bash
ruff check opto/features/graph \
  opto/features/recursive_opt/spec.py \
  opto/features/recursive_opt/__init__.py \
  tests/unit_tests/test_recursive_control_plane_v2.py
```

Result: **All checks passed**. A broader unchanged recursive-opt directory scan reports nine baseline findings in untouched files (`capabilities.py`, `experiments.py`, `inspect_utils.py`, `levels.py`, and `tracebench.py`); they are not suppressed or mixed into this corrective patch.

Post-commit exact-SHA result with `RECURSIVE_OPT_FINAL_SHA=c92f0af4af3e72a12b0228dbed215f86f8c9475b`: **278 passed, 2 skipped, 1 warning in 11.17s** for the mandated regression, and **458 passed, 3 skipped, 1 warning in 32.03s** for the complete unit suite. The SHA gate ran and passed. The two common skips require optional graph/telemetry backends; the complete-suite-only third skip requires Graphviz `dot`. Changed-file/graph Ruff again reported **All checks passed**.

## Migration and footprint

All 85 tracked historical files are classified: `execution_replayable=0`, `normalized_only=10`, `missing_dependency=23`, `historical_only=46`, `invalid=0`, `local_nonportable=6`. The representative config, family-policy, and prior specs normalize but cannot be replayed faithfully because exact dataset/evaluator/provider dependencies are absent; precise dependencies and paths are in `migration_report.json.representatives`. The deterministic UC4/UC14 notebook fixtures explicitly remain non-historical controls.

| measure | baseline | corrective worktree | delta |
|---|---:|---:|---:|
| recursive-opt runtime lines | 8,803 | 8,730 | -73 |
| `spec.py` lines | 2,688 | 2,613 | -75 |
| notebook code lines | 16 | 16 | 0 |
| public package exports | 114 | 114 | 0 |

The graph package is byte-clean against the baseline SHA and was not expanded.

## Changed scope and limitations

The intended corrective commit contains only the canonical runtime/export files, the control-plane test matrix, smoke notebook/goldens, migrated normalized specs, and control-plane evidence. Exact staged paths and post-commit `git status --short` are recorded after staging/commit.

The implementation commit changed 29 files: 3,141 insertions and 3,000 deletions, and initially left the index empty. The SHA-bearing evidence is staged separately and intentionally not committed because advancing HEAD would invalidate the exact-SHA gate. Final `git status --short --untracked-files=no` is exactly:

```text
M  artifacts/control_plane_v2/code_footprint_after.json
M  artifacts/control_plane_v2/control_plane_v2alpha.md
M  artifacts/control_plane_v2/evidence.md
A  artifacts/control_plane_v2/prompt18_readiness.json
M  artifacts/control_plane_v2/proof.md
M  artifacts/control_plane_v2/readiness_audit.md
```

The full short status has 108 entries because 102 unrelated, pre-existing user-owned experiment/config/output paths remain untracked; none was staged or modified. There is no post-commit diff under `opto/`, `tests/`, or `examples/recursive_opt_use_cases.ipynb`, so the tested implementation/client tree is exactly the committed SHA.

Limitations: no live OpenRouter run, no paid GEPA optimization, and no historical efficacy claim. The real installed GEPA API is checked only where it can run without a provider. Historical config/family-policy/prior behavior remains non-replayable without the precisely listed dependencies. This gate establishes readiness for Prompt 18; it does not begin that experiment.
