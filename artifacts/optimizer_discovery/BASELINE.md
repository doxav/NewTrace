# Phase 0 baseline

Starting and updated base HEAD: 846580defe935195c1f5f39d6336079ef6ff1e10.
`git fetch --all --prune`, `git checkout recursive_opt`,
`git pull --ff-only origin recursive_opt`: successful, already current.
Task branch: codex/phase0-patrick-interface; rollback branch:
codex/phase0-rollback-846580de. One untracked user file,
artifacts/probe_2026/probe_aa_results.json, inspected as a 2063-byte JSON object
and left untouched. No source changes existed.

Python 3.13.13, /home/xav/miniconda3/bin/python. Initial `pytest -q` invocation
failed collection (9 ModuleNotFoundError: opto); use `python -m pytest` to include
the checkout. No dependency installation required.

Exact suite command:
```
python -m pytest -q tests/unit_tests/test_recursive_control_plane_v2.py tests/unit_tests/test_recursive_spec.py tests/unit_tests/test_recursive_opt.py tests/unit_tests/test_recursive_final_hardening.py tests/unit_tests/test_recursive_measurement.py tests/unit_tests/test_recursive_surface_guard.py tests/unit_tests/test_recursive_transport.py tests/unit_tests/test_recursive_search_policy.py tests/unit_tests/test_recursive_budget_experiments.py
```
Result before production edits: **345 passed, 2 failed, 2 skipped**, 16.01s.
Both failures are pre-existing stale source digest checks:
`test_35_source_provenance` and
`test_readiness_uses_source_digests_without_sha_environment`.
Stored runtime hash 70b4d82fedbd71944825d8845dbebe24ccc0d294b5ae9851df60c6b1a2368f18;
actual dae2eecb2cf95d941de3145db1f43fcbabcec0285bed0904493d88841146c887.
Readiness already truthfully says required CI does not cover the current tree.
Refresh provenance after verified changes, without claiming new CI coverage.

| Classification | Issue | Treatment |
|---|---|---|
| BLOCKS PHASE 1 | MC-b/d missing actual candidate equivalence evidence | automatic evidence, scoped equivalence and explicit unknowns |
| BLOCKS PHASE 1 | signature-bound artifacts / no portable evaluator | fixed propose contract and isolated validator |
| BLOCKS PHASE 1 | invalid numerical penalties contaminating evidence | typed invalid and absent metrics in new evaluation path |
| BLOCKS PHASE 1 | stale source readiness | refresh measured digests, retain pre-CI status |
| DOES NOT BLOCK PHASE 1 | Phase-1 task family and preregistration not yet designed | required before performance experiments; Phase 0 proves interface only |
| DOES NOT BLOCK PHASE 1 | hosted CI has not run current tree | offline proof here; no push requested |
| HISTORICAL / LEGACY | EXP-12/13 prose noise and endpoint degradation | preserve; separate calibration |
| HISTORICAL / LEGACY | single-example knobs, saturated tasks, broken backlog specs | no optimizer-discovery benchmark claims on those tasks |
| HISTORICAL / LEGACY | H1/H2/H3/H5 evidence | unchanged |

Historical model: probe README identifies OpenRouter deepseek/deepseek-v4-flash-0731.
EXP-12 uses probe_g QASPER paired smoke; EXP-13's script imports a found prompt
from an ephemeral /tmp/claude-1000 path, inherits provider defaults and uses
max_examples=2, inner_steps=0, evaluation timeout 90s, run SIGALRM 480s, n=6
interleaved conditions. Complete original request settings/seed provenance are
absent. Exact reproducibility/comparability is not established despite model-name
agreement. Do not append fresh calibration to either historical experiment.
