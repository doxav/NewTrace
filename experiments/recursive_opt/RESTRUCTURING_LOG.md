# Recursive-opt restructuring log

Started 29 September 2026; final cleanup completed 30 September 2026. **Current state: legacy artifact/second-worktree compatibility directories removed; canonical runtime and historical evidence retained.** This log records storage and navigation changes, not new experiment results.

## Objective and completion conditions

- Inventory both worktrees, including ignored results, caches, frozen environments, legacy outputs and memory stores.
- Establish a faithful experiment-ID crosswalk; keep UC identifiers distinct from numbered experiments and distinguish both EXP22 studies.
- Establish consistent experiment documentation and result entry points under this repository.
- Consolidate unique experiment material from `Trace-experiment0` without losing dirty files, source versions, run evidence or active execution paths.
- Retire redundant navigation/planning documents with a relocation record and working historical references.
- Verify preservation, links, import/path dependencies and relevant existing tests after each migration batch.

## 2026-09-29 — batch 0: inventory and protection

No scientific files moved or removed in this batch.

- Recorded HEADs, branches, tracked worktree changes, index entries, remote references and registered worktrees in [`_reorganization/20260929/`](_reorganization/20260929).
- Confirmed that `Trace` and `Trace-experiment0` share `/home/xav/code/Trace/.git`; they are two worktrees of the same repository, not unrelated repositories.
- Queried GitHub read-only: `codex/exp22-evox-comparison` exists remotely at `37b0217960ac71406b44a58bc1db013c5ba569e0`. This protects that commit, not current uncommitted data. No remote branch was returned for the current EXP17/18 branch name.
- Initial metadata inventory: 618,913 files in the Trace scopes and 30,156 in Trace-experiment0; 216 Markdown documents outside runtime checkouts/caches. Expanded inventory adds legacy output and memory locations discovered afterward.
- Identified a live EXP23 worker and its EXP22 evaluator dependency. Moving/removing their runtime paths is deferred until safe migration can preserve those paths; no run was stopped or restarted.
- Found path-sensitive Python modules, frozen manifests and replay inputs. A blanket directory rename or deletion of old reports would be unsafe.

## Migration policy

The canonical research entry point will be `Trace/experiments/recursive_opt/`. Each identified experiment will expose the same roles: overview, protocol, results, machine-readable manifest, supporting documents, and raw-evidence access. Shared infrastructure and historical probes retain explicit identities instead of being mislabeled as experiment results.

Original evaluated code, immutable requests/responses, manifests and source archives remain identifiable by their original path and digest. Compatibility paths are permitted where executable imports, frozen manifests or historical links require them. They are migration aids, not competing current reports. New navigation must expose every retained evidence location and its purpose.

Unique EXP22/23 material will be consolidated into Trace. Runtime environments, nested registered Git worktrees and live output need separate treatment from research documents/results. They must not be accidentally copied into publishable experiment evidence, deleted as duplicates, or mutated while a worker depends on them.

## Original batch plan (completed below)

1. Finish expanded inventory and experiment crosswalk, including unresolved early IDs.
2. Establish canonical experiment directories and physically migrate appropriate documents/results with recorded aliases.
3. Consolidate the second worktree's unique research payload and resolve runtime dependencies.
4. Archive redundant indexes/planning documents; update incoming links and active navigation.
5. Run preservation/link/path/import checks and adapted tests; audit every objective above before marking complete.

This file will be extended with exact old/new paths, hashes and verification outcomes as batches execute.

## 2026-09-29 — batch 1: common entries and historical navigation cleanup

- Expanded map completed: 620,001 Trace files and 30,730 Trace-experiment0 files; 230 research Markdown documents. The metadata stream excludes its own output directory.
- Recovered EXP01–14 identities from Git `846580defe:artifacts/RESEARCH_LOG.md` and the September 24 corrected audit. No UC identifiers were silently relabelled.
- Created 24 consistent experiment/variant entries covering EXP01–23, including both EXP22 namespaces. Overview, protocol, result and raw-evidence roles are explicit. Main scientific reports/raw data remain at their existing locations pending the next batch.
- Archived `/home/xav/code/Trace/artifacts/RESULTS_INDEX.md` → `/home/xav/code/Trace/experiments/recursive_opt/_history/navigation/RESULTS_INDEX.md`; old path is a short redirect. Byte-exact original preserved in gzip.
- Archived `/home/xav/code/Trace/artifacts/RESEARCH_LOG.md` → `/home/xav/code/Trace/experiments/recursive_opt/_history/navigation/RESEARCH_LOG.md`; old path is a short redirect. Byte-exact original preserved in gzip.
- Archived `/home/xav/code/Trace/artifacts/optimizer_discovery/exp18/DOCUMENT_UPDATE_MAP.md` → `/home/xav/code/Trace/experiments/recursive_opt/_history/navigation/EXP18_DOCUMENT_UPDATE_MAP.md`; old path is a short redirect. Byte-exact original preserved in gzip.
- Full path/hash ledger: [`navigation_moves.json`](_reorganization/20260929/navigation_moves.json). Historical Markdown links were rebased; originals remain recoverable.
- Verification so far: inventory unit tests (3) and Ruff pass. Link/preservation checks follow before any further physical migration.

## 2026-09-29 — batch 2: principal reports and closed result stores

- `EXP15`: `/home/xav/code/Trace/artifacts/optimizer_discovery/EXP15_REPORT.md` → `/home/xav/code/Trace/experiments/recursive_opt/EXP15/RESULTS.md`. report relocated; historical location redirects to canonical report.
- `EXP16`: `/home/xav/code/Trace/artifacts/optimizer_discovery/investigation16/REPORT.md` → `/home/xav/code/Trace/experiments/recursive_opt/EXP16/RESULTS.md`. report relocated; historical location redirects to canonical report.
- `EXP18`: `/home/xav/code/Trace/artifacts/optimizer_discovery/exp18/REPORT.md` → `/home/xav/code/Trace/experiments/recursive_opt/EXP18/RESULTS.md`. report relocated; historical location redirects to canonical report.
- `EXP19`: `/home/xav/code/Trace/artifacts/o1_learning/EXP19.md` → `/home/xav/code/Trace/experiments/recursive_opt/EXP19/RESULTS.md`. report relocated; symlink preserves existing progress-writer path.
- `EXP20`: `/home/xav/code/Trace/artifacts/o1_learning/EXP20.md` → `/home/xav/code/Trace/experiments/recursive_opt/EXP20/RESULTS.md`. report relocated; historical location redirects to canonical report.
- `EXP21`: `/home/xav/code/Trace/artifacts/o1_learning/EXP21.md` → `/home/xav/code/Trace/experiments/recursive_opt/EXP21/RESULTS.md`. report relocated; historical location redirects to canonical report.
- `EXP22/qa`: `/home/xav/code/Trace/artifacts/o1_learning/EXP22.md` → `/home/xav/code/Trace/experiments/recursive_opt/EXP22/qa/RESULTS.md`. canonical reading view; original report retained byte-exact because runtime hashes it as protocol.
- Report before/after hashes and byte-exact originals: [`report_moves.json`](_reorganization/20260929/report_moves.json). Relative Markdown links rebased; result claims unchanged.
- EXP22-QA Markdown is an actual hashed protocol dependency (`open_optimizer.py`); its original bytes remain untouched. EXP19 retains a symlink because `analysis.update_progress` writes the historical filename.
- Forty-seven closed-evidence moves started after a `/proc` open-descriptor scan found no users of those stores. Each move hashes all files before/after and leaves a relative compatibility link. No live EXP23/EXP22 runtime path is in this batch.
- Durable per-move receipts: [`data_moves.jsonl`](_reorganization/20260929/data_moves.jsonl); per-file hashes: `data_move_*.json.gz`. Completion is recorded only after the post-move hash comparison succeeds.

## 2026-09-29 — batch 3: cross-worktree consolidation

- EXP22-EvoX payload and runtime are physically under `EXP22/evox/`; raw artifacts/runs are under its `results/`. The original scientific `manifest.json` is retained; navigation metadata is `catalog_entry.json`.
- EXP23 payload is physically under `EXP23/`. Its historical README is preserved at `EXP23/docs/ANALYSIS.md`; the canonical README remains the common entry.
- Used Linux atomic directory/file exchange with a compatibility symlink: the old name always resolves, and open descriptors retain the same inode. Unit tests verified continued writes through both an open handle and the old path. No process was signalled, restarted or paused.
- `EXP22/` has documented sibling-API aliases into `evox/` because EXP23 resolves a sibling named EXP22. No frozen experiment source was edited to change that contract.
- All 48 captured Python source hashes and both nested frozen Git HEADs are unchanged. Per-path device/inode receipts: [`cross_worktree_moves.jsonl`](_reorganization/20260929/cross_worktree_moves.jsonl); proof: [`cross_worktree_verification.json`](_reorganization/20260929/cross_worktree_verification.json).

## 2026-09-29 — batch 4: ownership, worktree variants and navigation

- Completed the 47 closed-store moves: **590,522 files** passed before/after SHA-256 comparison. Every original location resolves through its compatibility route.
- Mapped **650,731 original entries** to owners and current locations; none of their original paths is missing. Assigned all **230 original research Markdown files** to an experiment or explicit shared/history collection.
- Canonicalized the 24 experiment/variant entries and assessment references. Added supporting-document indexes; exposed original protocols instead of inventing missing preregistrations.
- Compared **2,265** remaining non-runtime research files across worktrees: **2,255 identical**, **10 distinct/unique versions** copied byte-for-byte into `_history/worktree_versions/Trace-experiment0/`. No main-checkout implementation was overwritten.
- Preserved historical fragment anchors on eight redirect documents. Preserved the original EXP23 analysis bytes separately and rebased its reading-copy links.
- Git refuses symlinked `.gitignore` files even when filesystem resolution works. Replaced the two compatibility ignore links with byte-identical ordinary files; original scientific ignore rules are unchanged. Receipt: `gitignore_compatibility.json`.
- Recorded the user’s EXP23 update in a separate frozen checkpoint: Trace completed 100 solution attempts at 30.876583; the comparison arm was still ongoing. At the common 63-attempt prefix, Trace scored 30.876583 versus 26.172033. The scientific verifier recomputes both prefix scores. This is positive local evidence, not a replicated method comparison.
- Updated `STORAGE_MAP.md` to explain why shared code, hashed protocols, runtime aliases and historical supplements intentionally retain stable paths. Consolidation does not imply deleting the old registered worktree.

## 2026-09-29 — batch 5: verification and completion audit

Commands run from `/home/xav/code/Trace` unless a different directory is shown:

| Command | Result |
|---|---|
| `python -m unittest discover -s experiments/recursive_opt/_reorganization/20260929 -p 'test_*.py' -q` | 8 inventory/migration tests passed, including atomic live-writer continuity and rollback. |
| `experiments/recursive_opt/EXP22/evox/.venv/bin/python -I -m unittest discover -s experiments/recursive_opt/EXP22/evox/tests -v` | 50 offline EXP22 tests passed. Log: `exp22_offline_tests.log`. |
| From EXP23: `../EXP22/.venv/bin/python -I -m unittest discover -s tests -v` | 13 offline EXP23 tests passed. Log: `exp23_offline_tests.log`. |
| `experiments/recursive_opt/EXP22/evox/.venv/bin/python -m pytest -q tests/unit_tests/test_recursive_coevolution.py` | 21 native coevolution tests passed. Log: `native_coevolution_tests.log`. |
| `python artifacts/recursive_opt_synthesis_20260929/verify.py` | PASS: saved arithmetic, five equivalence fixtures, two distinct EXP23 checkpoints, 55 assessment references and linked-source hashes. |
| `python -m ruff check experiments/recursive_opt/_reorganization/20260929/*.py artifacts/recursive_opt_synthesis_20260929/verify.py` | PASS. |
| `python -m black --check --target-version py311 experiments/recursive_opt/_reorganization/20260929/*.py artifacts/recursive_opt_synthesis_20260929/verify.py` | PASS. |

The original and post-migration metadata inventories are separate files. Checks also confirmed unchanged Git indexes, unchanged frozen-source hashes and nested Git HEADs, intact original report backups, the byte-identical EXP22-QA protocol, and continued presence of the live EXP23 worker. No paid call or experiment restart was initiated by this restructuring.

Final machine evidence: [`final_verification.json`](_reorganization/20260929/final_verification.json). Every requested storage/navigation condition is handled; retained compatibility dependencies and local-only uncommitted storage are explicitly documented. Scientific incompleteness of historical or live experiments is unchanged by storage completion.

Final navigation check: **131 current entry documents, 824 links, zero missing targets**. Scoped `git diff --check` passed for the assessment, retired navigation/report paths and `experiments/recursive_opt`. Both `git status --short --untracked-files=no` calls completed without warnings. Total adapted unit tests: **92 passed**, plus five saved equivalence fixtures checked by the scientific verifier.

## 2026-09-30 — final cleanup requested by the user

The user explicitly requested removal of the old compatibility folders, consolidation of remaining artifacts/root clutter, role READMEs and a review of untracked files. The first migration's compatibility-path policy is superseded by this cleanup.

### Physical moves and removals

All 27 moves passed before/after byte checks. [Exact plan](_reorganization/20260930/moves_plan.json), [completed moves](_reorganization/20260930/moves.jsonl), and per-file digests in `move_*.json`.

| Previous location | Current location |
|---|---|
| `Trace/artifacts/control_plane_v2` | `_shared/control_plane_v2` |
| `Trace/artifacts/optimizer_discovery` | `_shared/optimizer_discovery` — shared benchmark, replay code and historical study support; principal EXP15–18 results remain in their numbered entries |
| `Trace/artifacts/o1_learning` | `_shared/o1_learning` |
| `Trace/artifacts/probe_2026` | `_history/probe_2026` |
| `Trace/artifacts/recursive_opt_assessment.md` | `ASSESSMENT.md` |
| September 24 audit and earlier assessment/plans | `_history/audit_20260924` and `_history/reviews` |
| September 29 synthesis | `_analysis/synthesis_20260929` |
| Root `mem_A_multi_param`, `mem_A_online_bin_packing_local`, `memOLD`, `XP_recurse_2`, `notebook_outputs`, `OUTPUTS_mem.zip`, benchmark logs | `_history/legacy_runs` |
| Five `/home/xav/code/Trace-experiment0*.zip` exports | `_history/worktree_exports` |
| Opaque UUID workbook output directory | `EXP16/presentation` |

- Removed the second checkout's EXP22/23 directories: 24 verified alias/compatibility entries, without traversing or deleting their canonical targets.
- Removed 2,017 redundant second-checkout research/package-marker files after checking the canonical version or preserving a distinct version. Six distinct versions from this final pass are in `_history/worktree_versions/Trace-experiment0_20260930/`. Before-edit source/document backups preserve two formerly identical files whose canonical reading code was subsequently updated.
- Removed the second checkout's duplicate root `OUTPUTS_mem.zip` after a byte comparison. Removed empty `trace_memory` directories; scientific memory stores were archived, not discarded.
- Both `artifacts/` roots now contain only a role/relocation README. Retired root navigation stubs were archived before removal. Neither location contains research compatibility symlinks.

### Dependencies and code updates

- Repaired both nested Git worktree registrations to `EXP22/evox/worktrees/`. Updated 31 environment entry-point/editable-install references to canonical locations, then verified executable/import startup after deleting the old EXP22 path.
- Updated 124 Python source/test files mechanically for relocated imports and storage paths, with original source bytes retained in `source_before.zip`. No optimizer or evaluator algorithm was changed.
- Fixed the new-freeze path translation in EXP17: historical predecessor receipts retain old filenames; new freezes resolve canonical sources and record their actual hashes. A regression test verifies that the predecessor receipt stays byte-identical.
- Updated two notebooks' code-cell paths; stored outputs are unchanged. Rebased 105 Markdown reading copies and preserved originals in `documents_before.zip`. EXP22-QA's hashed protocol is byte-identical.
- Canonical EXP22's internal sibling-API aliases remain in place for active EXP24 and EXP23. No dependency points back into the removed second-checkout experiment directories. Active EXP24 workers were not signalled or restarted; the finished runs observed during verification reported success.

### Git-status decisions

[Human-readable review](_reorganization/20260930/GIT_STATUS_REVIEW.md) and [per-entry decisions](_reorganization/20260930/untracked_decisions.json) cover 57 untracked entries at the review checkpoint. Source, tests, notebooks, canonical study evidence and active EXP24 work are retained. Regenerable tool caches and explicitly local backup ZIPs are ignored. The 548 MB EXP18 export and five worktree export ZIPs remain on disk with hashes; this cleanup does not publish them.

Both Git indexes are unchanged. No staging, commit, push, branch reset or framework-worktree deletion occurred. Existing unrelated user changes remain.

### Validation and known baseline failures

The original environment choice lacked dependencies needed by the QA tests. No dependency was installed; the existing `humanllm` environment supplies them. Its before-move baseline was **332 passed, 2 failed**. Both failures were stale recorded runtime-provenance digests and already existed before these moves.

The first post-move run exposed the notebook's old golden-spec path and EXP17's predecessor-source paths. Those issues were fixed and the final expanded suite ran after the fixes: **359 passed, the same 2 pre-existing failures**. The notebook executes in a clean offline kernel; EXP17's generation/resume checks and the new preservation regression test pass.

Exact expanded regression command (from `/home/xav/code/Trace`):

```sh
/home/xav/miniconda3/envs/humanllm/bin/python -m pytest -q tests/unit_tests/test_recursive_optimizer_benchmark.py tests/unit_tests/test_recursive_exp15.py tests/unit_tests/test_recursive_phase0_calibration.py tests/unit_tests/test_recursive_control_plane_v2.py tests/unit_tests/test_recursive_final_hardening.py tests/unit_tests/test_recursive_transport.py tests/unit_tests/test_investigation16_benchmark.py tests/unit_tests/test_investigation16_feedback_experiment.py tests/unit_tests/test_investigation16_generation.py tests/unit_tests/test_investigation16_history.py tests/unit_tests/test_investigation16_history_trace_review.py tests/unit_tests/test_investigation16_production_driver.py tests/unit_tests/test_investigation16_search_experiment.py tests/unit_tests/test_investigation16_trace_schedule.py tests/unit_tests/test_exp17_driver.py tests/unit_tests/test_exp17_study.py tests/unit_tests/test_o1_hotpot.py tests/unit_tests/test_o1_learning_study.py tests/unit_tests/test_o1_qa_axes.py tests/unit_tests/test_o1_qa_campaign.py tests/unit_tests/test_o1_qa_open_optimizer.py tests/unit_tests/test_o1_trace_curriculum.py experiments/recursive_opt/_shared/optimizer_discovery/exp17/test_run_pipeline.py experiments/recursive_opt/_shared/optimizer_discovery/exp18/test_study.py
```

Additional checks:

```sh
experiments/recursive_opt/EXP22/evox/.venv/bin/python -I -m unittest discover -s experiments/recursive_opt/EXP22/evox/tests -q
# From experiments/recursive_opt/EXP23:
../EXP22/.venv/bin/python -I -m unittest discover -s tests -q
# From /home/xav/code/Trace:
python -m unittest discover -s experiments/recursive_opt/_reorganization/20260930 -p 'test_*.py' -q
python -m ruff check experiments/recursive_opt/_reorganization/20260930/*.py
python -m black --check --target-version py311 experiments/recursive_opt/_reorganization/20260930/*.py
python experiments/recursive_opt/_analysis/synthesis_20260929/verify.py
```

Results: EXP22 **50 passed**; EXP23 **13 passed**; cleanup helper **4 passed**; lint/format, updated-source compilation and scoped `git diff --check` passed. The scientific verifier passed arithmetic, five saved equivalence fixtures and assessment references. Error-focused Ruff checks passed for all 124 path-updated Python files; existing unrelated formatting was not rewritten.

The unchanged baseline failures are `test_recursive_control_plane_v2.py::test_35_source_provenance` and `test_recursive_final_hardening.py::test_readiness_uses_source_digests_without_sha_environment`. Historical digests were not silently refreshed or checks disabled.

[Final preservation and test evidence](_reorganization/20260930/verification.json). The old storage roots are clean, required original bytes are retained, and current navigation points to canonical files. Strict replay of historical runs still uses their original frozen sources; historical receipt paths are not rewritten as if those runs used the new layout.

Final current-navigation check: **142 documents, 1,323 actual local links, zero missing targets**. One historical confidence-interval line was excluded because it is prose, not a used link. All 50 retargeted internal aliases resolve. The final per-file audit found 15,281 unchanged moved files; other differences are recorded code/link changes, regenerated verification output or bytecode caches.
