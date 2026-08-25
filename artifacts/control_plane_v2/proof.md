# recursive-opt v2alpha semantic-closure proof

## Live transport resilience proof

Provider-free causal tests reproduce both observed messages as pre-fix
non-retryable failures, then prove exact identical-request recovery, bounded
three-attempt exhaustion, immediate application-error failure, causal-chain
classification, explicit provider timeout propagation, environment invariance,
one logical guarded usage attribution, deterministic Trace concurrency, and
hard termination of an uncooperative child. The full local results are 225
focused passes, 330 mandated passes with two accepted optional skips, 510 unit
passes with three accepted optional skips, 40 Experiment-0 passes, and one
clean-kernel notebook pass.

Runtime footprint remains 8,850 lines without relaxing the historical gate.
The pre-CI authoritative digests are runtime
`420b5351063a56b0ad274a6c39b6aaa4dc95b9094434600e89ea79f3eccc8872`
and registry
`f1a94b02c94607c2d22e6f10bce25b10d9b642814915ec01c72765280be26fa4`.
Readiness is false until the required Actions job passes the expanded workflow.

## Optimizer empty-text retry proof

Six provider-free causal tests prove one shared optimizer text contract across
Trace and installed GEPA 0.1.4: empty/text succeeds after exactly two identical
metered requests; empty/empty fails explicitly after two; normal text makes one
call; real Trace and GEPA both propose and evaluate candidates after the retry;
and direct `OptoPrimeV2` cannot reach `"TERMINATE" in None`.

The full provider-free results are 212 focused passes, 317 mandated recursive
passes with two accepted optional skips, 497 complete-unit passes with three
accepted optional skips, 23 Experiment-0 passes including its full offline
contract, and one clean-kernel notebook pass. Runtime footprint remains exactly
8,850 recursive-opt lines through whitespace-only compaction; no footprint gate
was relaxed. Runtime digest is
`ba4836d9f43cffcd0271086932745b270d75478b5287a7d8100be4928b623cbc`
and registry digest remains
`f1a94b02c94607c2d22e6f10bce25b10d9b642814915ec01c72765280be26fa4`.
Required Actions run `32669603929`, job `97268256178`, passed for implementation
commit `d63746afbb88d6193cbfedf2932b256d9f33b6e4`; Prompt-18 readiness is true for
the matching source digests.

## GEPA reflection-protocol hotfix proof

The provider-free installed-`gepa==0.1.4` test reaches a real reflective
proposal through `run_spec`, GEPA's public `optimize_anything()`, the private
text/chat adapter, `_GuardedRoleClient`, and a strict chat-only provider fake.
It proves one optimizer call and one usage attribution, a changed selected
artifact, candidate evaluation, and holdout externalization with sockets
blocked. Required Actions run `32650995105` / job `97222458541` passed on
`c1bf6a296d159b92d09ac29ee90556d2c1997a5d`.

Relocked digests are runtime
`37072c1364a02c277a677bf43ad8132a32a9f233488c80cd2b6bf1a7e344f33e`
and registry
`f1a94b02c94607c2d22e6f10bce25b10d9b642814915ec01c72765280be26fa4`.
The evidence-state run `32651342206` / job `97223324883` was also green. The
live GEPA arm subsequently made one guarded reflection call and attributed
1,584 optimizer tokens once without holdout leakage. Its changed proposal was
evaluated but rejected, leaving the Experiment-0 micro gate false and preventing
pilot execution.

- Branch: `recursive_opt`
- Baseline SHA: `21a0ad3d2f4f835ce2ffb1eef18c36a622265418`
- Semantic-closure implementation SHA: `c92f0af4af3e72a12b0228dbed215f86f8c9475b`
- Completion-audit implementation SHA: `05dabf68e77ef2b9c59a8fc20c68bf4f8d2c1eaf`
- Prompt 17.7 starting SHA: `14b832c82341bbc55e9c662ebaebcba4e3e8e95b`
- Environment: conda `humanllm`, Python 3.12
- Date: 2026-08-22 (Europe/Paris)
- Live provider or paid calls: **none**
- CI workflow: `.github/workflows/recursive-opt-v2.yml`, job `recursive-opt v2 offline (required)`, now including the pinned GEPA extra and hardening suite

After the GEPA 0.1.4 public-contract hotfix, authoritative source digests are `runtime_tree_sha256=5b460d771ca0b0f9bd914b2c8330860e6f5771a8447d40e50db0d554986e0642` and golden-spec `registry_sha256=f1a94b02c94607c2d22e6f10bce25b10d9b642814915ec01c72765280be26fa4`. They supersede the historical exact-SHA readiness mechanism below. Required GitHub Actions run `32583433295` for hotfix commit `52a7b0bd86b21975e2de09cec0a957b04e835312` completed successfully; required job `97056076300` passed in 1m07s. Prompt-18 readiness is true.

## GEPA 0.1.4 hotfix verification

| invariant | causal test | command | result | evidence |
|---|---|---|---|---|
| public evaluator is a scalar or pair | 22, 22b | focused GEPA seam | pass | wrapper output is `(1.0, None, {"valid": true})` |
| public entry point supplies the wrapper | 22c | actual `optimize_anything()` with one seed evaluation and blocked sockets | pass | one validation call; no reflection/provider call |
| weighted minimize direction remains scalar | 22d | focused GEPA seam | pass | ratio 0.5 scores above 1.5 |
| rich ASI without reserved Pareto axes | 22, 22d | focused GEPA seam | pass | raw `metrics` retained; `scores` absent |
| invalidity dominates raw metrics | 20b, 22d | control-plane suite | pass | explicit invalid floor and `valid=false` |

Provider-free/network-blocked results in conda `humanllm`:

- focused GEPA seam: **4 passed, 41 deselected in 1.59s**;
- control-plane v2: **45 passed in 10.20s**;
- final hardening: **21 passed in 2.61s**;
- recursive spec: **47 passed in 2.89s**;
- objective/vector/multi-objective: **89 passed in 2.92s**;
- all recursive unit files: **224 passed, 2 skipped, 1 warning in 16.78s**;
- complete unit suite: **487 passed, 3 skipped, 1 warning in 39.26s**;
- clean-kernel notebook: **1 passed, 44 deselected in 4.47s**;
- changed-file Ruff and both diff checks: **passed**.

The two common skips require optional graph/telemetry backends; the third complete-suite skip requires Graphviz `dot`. No GEPA test skipped. No live provider or paid call occurred. The pre-push checkpoint kept readiness false; run `32583433295` subsequently supplied the required `completed / success` evidence.

The completion-audit implementation commit has parent `fc00beac05fc1a73b0c017b7615b56b4162f12f1`. SHA-bearing evidence is a post-commit worktree update because a commit cannot contain its own hash.

## Invariant matrix

| invariant | causal test | command | result | evidence |
|---|---|---|---|---|
| canonical one/two-level schema and compatibility migration | 01–04 | mandated regression | pass | golden/migrated specs |
| actual recursive execution and upstream counterfactual | 05–06 | mandated regression | pass | result lineage |
| real Trace optimize path and engine config | 07–08 | mandated regression | pass | budget/metadata |
| module artifact, inputs, targets, config validation | 09–10, 13b | mandated regression | pass | module artifact |
| portable evaluator and dataset registries | 11–13 | mandated regression | pass | normalized refs/fingerprint |
| exact role clients, preflight, fallbacks, usage | 14–16 | mandated regression | pass | selected models/usage |
| weighted/Pareto objectives, constraints, rollback | 17–20b | mandated regression | pass | canonical evaluation/artifact |
| structural holdout isolation | 21 | mandated regression | pass | phase-context fault injection |
| GEPA holdout externalization and exact 0.1.4 public API | 22/22b/22c | mandated regression | pass | no-provider public wrapper and entry-point contract |
| in-run budgets and all policies | 23–24 | mandated regression | pass | budget report |
| scoped deterministic seeds | 25 | mandated regression | pass | deterministic metrics |
| atomic outputs and cross-process resume | 26–27 | mandated regression | pass | persisted run tree |
| knowledge store and every-card binding/lineage | 28–29 | mandated regression | pass | artifact ids in lineage |
| semantic migration classification | 30 | mandated regression | pass | migration reports |
| fixed/Trace/GEPA same-spec result shape | 31 | mandated regression | pass | canonical `RunResult` |
| spec-only notebook and clean offline kernel | 32–33 | mandated regression | pass | smoke notebook |
| footprint limits | 34 | mandated regression | pass | footprint JSON |
| source/registry provenance and stale-resume rejection | 35 + hardening suite | network-blocked regression | pass | readiness JSON and resolved manifest |

## Prompt 17.7 verification

All commands removed provider keys and used `--disable-socket --allow-hosts=127.0.0.1,localhost`.

- Focused hardening: **21 passed in 2.12s**.
- Mandated recursive/Trace/GEPA regression: **305 passed, 2 skipped, 1 warning in 16.12s**.
- Complete unit suite: **485 passed, 3 skipped, 1 warning in 37.27s**.
- Clean-kernel notebook: **1 passed in 4.11s**.
- Isolated worktree focused readiness, including real Trace, installed GEPA, notebook, footprint, and source gates: **26 passed in 5.32s**.
- Ruff on `spec.py`, `optimize.py`, and both changed test files: **All checks passed**.
- `git diff --check`: **passed**.
- Workflow YAML parse and required-job structure check: **passed**.

The required job installs `python -m pip install -e '.[gepa]'` and runs the control-plane, recursive-spec, objective/vector, multi-objective, and hardening suites. No GitHub Actions result is claimed for the unpushed worktree.

The 28 baseline diagnoses and their corrective dispositions are mapped individually in `readiness_audit.md`.

## Verification commands

Mandated recursive regression (network disabled; localhost allowed only for the notebook kernel):

```bash
env -u OPENAI_API_KEY -u OPENROUTER_API_KEY -u ANTHROPIC_API_KEY \
  -u GOOGLE_API_KEY -u TAVILY_API_KEY \
  RECURSIVE_OPT_LIVE=0 \
  PYTHONHASHSEED=0 PYTHONPATH=. \
  /home/xav/miniconda3/envs/humanllm/bin/python -m pytest -q --disable-socket \
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
  tests/unit_tests/test_recursive_final_hardening.py \
  tests/unit_tests/test_recursive_opt_abc_probe.py
```

Pre-commit authoritative completion-audit result, with only `RECURSIVE_OPT_FINAL_SHA` omitted: **283 passed, 3 skipped, 1 warning in 11.72s**.

Complete unit suite:

```bash
env -u OPENAI_API_KEY -u OPENROUTER_API_KEY -u ANTHROPIC_API_KEY \
  -u GOOGLE_API_KEY -u TAVILY_API_KEY \
  RECURSIVE_OPT_LIVE=0 \
  PYTHONHASHSEED=0 PYTHONPATH=. \
  /home/xav/miniconda3/envs/humanllm/bin/python -m pytest -q --disable-socket \
  --allow-hosts=127.0.0.1,localhost tests/unit_tests
```

Pre-commit authoritative completion-audit result, with only `RECURSIVE_OPT_FINAL_SHA` omitted: **463 passed, 4 skipped, 1 warning in 33.27s**.

Pre-commit skips: two tests require optional graph/telemetry backends; one is the deliberately post-commit final-SHA gate; the complete-unit-only fourth skip requires the Graphviz `dot` executable. GEPA 0.1.4 and LangGraph contract tests ran rather than skipping. The single warning is LangGraph's pending default change for serializer `allowed_objects`.

Lint for every changed Python file and the retained graph contract:

```bash
/home/xav/miniconda3/envs/humanllm/bin/ruff check opto/features/graph \
  opto/features/recursive_opt/spec.py \
  opto/features/recursive_opt/__init__.py \
  tests/unit_tests/test_recursive_control_plane_v2.py
```

Result: **All checks passed**. A broader unchanged recursive-opt directory scan reports nine baseline findings in untouched files (`capabilities.py`, `experiments.py`, `inspect_utils.py`, `levels.py`, and `tracebench.py`); they are not suppressed or mixed into this corrective patch.

Post-commit exact-SHA verification with `RECURSIVE_OPT_FINAL_SHA=05dabf68e77ef2b9c59a8fc20c68bf4f8d2c1eaf`: **284 passed, 2 skipped, 1 warning in 11.08s** for the mandated regression and **464 passed, 3 skipped, 1 warning in 30.12s** for the complete unit suite. The SHA gate ran and passed. The two common skips require optional graph/telemetry backends; the complete-suite-only third skip requires Graphviz `dot`. Changed-file/graph Ruff and `git diff --check` again passed.

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

The complete corrective series contains only the canonical runtime/export files, the control-plane test matrix, smoke notebook/goldens, migrated normalized specs, and control-plane evidence. Exact staged paths and post-commit `git status --short` are recorded below.

The original semantic-closure implementation changed 29 files. The completion-audit commit changes 8 files with 433 insertions and 63 deletions: one runtime file, its causal test matrix, and six readiness records. The SHA-bearing evidence is staged separately and intentionally not committed because advancing HEAD would invalidate the exact-SHA gate.

```text
M  artifacts/control_plane_v2/code_footprint_after.json
M  artifacts/control_plane_v2/control_plane_v2alpha.md
M  artifacts/control_plane_v2/evidence.md
M  artifacts/control_plane_v2/prompt18_readiness.json
M  artifacts/control_plane_v2/proof.md
M  artifacts/control_plane_v2/readiness_audit.md
```

The full short status has 108 entries: the six staged SHA-evidence files above and 102 unrelated, pre-existing user-owned experiment/config/output entries. None of the untracked files is staged or modified. There is no post-commit diff under `opto/`, `tests/`, or `examples/recursive_opt_use_cases.ipynb`, so the tested implementation/client tree is exactly the committed SHA.

Limitations: no live OpenRouter run, no paid GEPA optimization, and no historical efficacy claim. The real installed GEPA API is checked only where it can run without a provider. Historical config/family-policy/prior behavior remains non-replayable without the precisely listed dependencies. Local Prompt 17.7 gates are green, but Prompt-18 readiness remains false until the required GitHub Actions job is observed green; Prompt 18 was not begun.
