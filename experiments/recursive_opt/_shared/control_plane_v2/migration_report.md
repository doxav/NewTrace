# Historical recursive-opt migration report

- Baseline SHA: `21a0ad3d2f4f835ce2ffb1eef18c36a622265418`
- Scope: every git-tracked `examples/**/artifacts.jsonl` and `*spec.json` file
- Source files modified: **0**
- Migrated copies: `artifacts/control_plane_v2/migrated_specs/`
- Machine-readable per-file evidence: `migration_report.json`

## Summary

| classification | files |
|---|---:|
| `execution_replayable` | 0 |
| `normalized_only` | 10 |
| `historical_only` | 46 |
| `missing_dependency` | 23 |
| `local_nonportable` | 6 |
| `invalid` | 0 |

Total classified: **85**. The workspace also contained **4291** untracked candidate files; those are user-owned outputs outside the git baseline, so they were neither modified nor silently folded into repository migration evidence.

## Decisions

All 46 artifact ledgers are `historical_only`: they are readable data records, not complete executable specs. No original paid/live execution was rerun.

All 16 legacy `spec.json` files passed `normalize_spec`; their fingerprinted normalized copies live in the separate migration directory. Ten are `normalized_only`; six preserve process-local callable strings and are `local_nonportable`. Originals remain byte-identical, with SHA-256 recorded per entry.

The representative config spec requires a pinned `hf:drop` revision and its historical TraceBench evaluator/provider responses. The representative family-policy spec requires pinned `internal:multiobjective_gsm8k` and `hf:qasper` evaluator/provider responses. The representative prior spec additionally requires the faithful upstream family-policy execution. These precise dependencies are absent, so all three are `normalized_only`, not execution replayable; their exact source paths are recorded in `migration_report.json.representatives`.

The 23 `component_spec.json`/`graph_spec.json` files are metadata rather than complete execution specs. They lack a versioned module and evaluator, so classifying them as `missing_dependency` follows the rule not to invent either dependency.

UC4 and UC14 historical live runs are explicitly **non-executable as faithful replays** in this evidence set. UC4 retains legacy specs and artifact data, but the exact historical evaluator/provider runtime and responses are not pinned. No complete git-tracked UC14 execution spec with a versioned evaluator/module exists. Their status is therefore `missing_dependency`; nothing was invented.

The notebook's golden UC4-positive and UC14-negative specs are marked `historical_replay=false` and are deterministic offline control-plane contracts only.
