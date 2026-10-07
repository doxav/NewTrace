# Storage and responsibility map

[Experiment catalogue](README.md) · [Assessment](ASSESSMENT.md) · [Restructuring log](RESTRUCTURING_LOG.md) · [Git status review](_reorganization/20260930/GIT_STATUS_REVIEW.md)

## Current locations

| Material | Canonical location | Why it lives here |
|---|---|---|
| Numbered studies | `EXP01/`–`EXP24/`; `EXP22/qa/` and `EXP22/evox/` are distinct | Common overview, protocol, results and evidence entry points; EXP24 is active. |
| Control-plane engineering evidence | [_shared/control_plane_v2](_shared/control_plane_v2/README.md) | Shared ADRs, golden specs and readiness/provenance fixtures. It is not one experiment's performance result. |
| Numerical optimizer-discovery implementation and supplements | [_shared/optimizer_discovery](_shared/optimizer_discovery/README.md) | EXP15–18 share evaluator, generation, analysis and replay modules. Their historically colocated support directories remain together inside the canonical tree; principal reports/results are in each EXP entry. Imports now use `experiments.recursive_opt._shared.optimizer_discovery`. |
| O1 learning implementation and protocols | [_shared/o1_learning](_shared/o1_learning/README.md) and [o1_qa](o1_qa/) | Shared EXP19–22 study code and frozen protocol material. Principal run stores/reports remain in numbered study entries. |
| Early probes | [_history/probe_2026](_history/probe_2026/README.md) | Original EXP01–14 and auxiliary instrument/noise probes, with their original identities. |
| Current cross-study synthesis | [ASSESSMENT.md](ASSESSMENT.md) and [_analysis](_analysis/README.md) | Reconciled conclusions and verification; separate from historical drafts. |
| Retired plans, assessments and indexes | [_history/reviews](_history/reviews/) and [_history/navigation](_history/navigation/) | Historical context, with original-byte backups; not competing current reports. |
| Root memory stores and old notebook outputs | [_history/legacy_runs](_history/legacy_runs/) | Preserved observations and priors formerly scattered at the Trace root. |
| Root ZIP exports | [_history/worktree_exports](_history/worktree_exports/) | Five original Trace-experiment0 snapshots moved from `/home/xav/code`; local-only backups. |
| Distinct worktree versions | [_history/worktree_versions](_history/worktree_versions/README.md) | Both versions preserved where the two checkouts differed. |
| EXP16 workbook deliverables | [EXP16/presentation](EXP16/presentation/) | Previously under an opaque UUID directory in `outputs/`. |
| Experiment-0 implementation | [multiobjective_reasoning](multiobjective_reasoning/) | Code of **EXP00-E** ([EXP00](EXP00/README.md)), kept at this path because it is imported by `o1_qa/task.py` and by the `_history` probe/audit scripts, and recorded in frozen run plans. Its `outputs/recursive_opt/experiment_0` run tree remains an explicit project output location, indexed from the shared entry. |
| Pre-numbered notebook campaigns | `examples/recursive_opt_demo.ipynb`, `examples/recursive_opt_phases*.ipynb`, `examples/recursive_opt_use_cases.ipynb` (outputs at commit `5a148ddba9`), `examples/notebook_outputs/recursive_opt_use_cases/` | Reported as EXP00-A–D in [EXP00](EXP00/README.md); sources stay in `examples/`. |
| Earlier example output | [_history/use_cases](_history/use_cases/README.md) | UC identifiers remain distinct from numbered experiments; outputs retained beside their example runners. |

## Removed locations and runtime dependencies

`Trace/artifacts/` and `Trace-experiment0/artifacts/` now contain only role/relocation READMEs. No old research folders or compatibility symlinks remain there. The second checkout's EXP22/23 directories, duplicate experiment-0 package/output and identical root memory ZIP were removed after verification. Its unrelated framework source and user changes remain.

The EXP22 environment and frozen worktrees remain physically under `EXP22/evox/`. Environment entry-point paths and editable imports now use that location. Git worktree registrations were repaired to the canonical paths. The aliases inside canonical `EXP22/` remain because live EXP24 and EXP23 use that sibling API; none points into Trace-experiment0.

Internal result aliases inside the canonical shared support tree preserve scripts' colocated-file contracts. Old absolute paths in immutable manifests and freeze receipts remain historical provenance, not live filesystem dependencies. New EXP17 freezes translate predecessor source locations and record current hashes without rewriting the predecessor receipt. Old saved runs still require their original frozen sources for strict provenance authentication.

EXP22-QA's original `EXP22.md` remains byte-identical because the runtime hashes it as protocol. Other moved Markdown reading copies have rebased links; originals are preserved in the cleanup document backup. Runtime code was changed only for import/storage paths and the predecessor-path translation.

## Git and preservation

Both checkouts are worktrees of the same Git repository. No files were staged, committed or pushed by this cleanup. Current changes remain local. The [Git status review](_reorganization/20260930/GIT_STATUS_REVIEW.md) records all untracked-entry decisions.

The original 548 MB EXP18 ZIP and five worktree export ZIPs are retained locally and excluded from Git; corresponding ordinary study evidence remains visible. Regenerable `.ruff_cache` directories are ignored. No scientific evidence was deleted as a cache.

## Audit trail

- [September 30 move plan](_reorganization/20260930/moves_plan.json) and [completed movement receipts](_reorganization/20260930/moves.jsonl)
- Per-file original hashes: `_reorganization/20260930/move_*.json`
- [Removed compatibility entries](_reorganization/20260930/removed_legacy_entries.json)
- [Verified duplicate removal and distinct-version preservation](_reorganization/20260930/removed_duplicate_research.json)
- [Import/source changes](_reorganization/20260930/source_path_updates.json) and [document link changes](_reorganization/20260930/document_path_updates.json)
- [Untracked-file decisions](_reorganization/20260930/untracked_decisions.json)
- [Current experiment catalogue](_reorganization/20260930/catalog.json)
- [September 29 inventory and first migration](_reorganization/20260929/): historical checkpoint, superseded where September 30 removes compatibility paths.

