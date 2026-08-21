# Historical recursive-opt migration report

- Baseline SHA: `6fc278a398709fe79a0fc9be22bae99bffd8cba6`
- Scope: every git-tracked `examples/**/artifacts.jsonl` and `*spec.json` file
- Source files modified: **0**
- Migrated copies: `artifacts/control_plane_v2/migrated_specs/`
- Machine-readable per-file evidence: `migration_report.json`

## Summary

| classification | files |
|---|---:|
| `replayable` | 46 |
| `migrated_replayable` | 16 |
| `historical_only` | 0 |
| `invalid` | 0 |
| `missing_dependency` | 23 |
| `local_nonportable` | 0 |

Total classified: **85**. The workspace also contained **4291** untracked candidate files; those are user-owned outputs outside the git baseline, so they were neither modified nor silently folded into repository migration evidence.

## Decisions

All 46 artifact ledgers are data-replayable: every JSONL record has `artifact_id`, `content`, and `score`. This does not claim that the original paid/live execution was rerun.

All 16 legacy `spec.json` files passed `normalize_spec`; their fingerprinted normalized copies live in the separate migration directory. Originals remain byte-identical, with SHA-256 recorded per entry.

The 23 `component_spec.json`/`graph_spec.json` files are metadata rather than complete execution specs. They lack a versioned module and evaluator, so classifying them as `missing_dependency` follows the rule not to invent either dependency.

UC4 and UC14 historical live runs are explicitly **non-executable as faithful replays** in this evidence set. UC4 retains legacy specs and artifact data, but the exact historical evaluator/provider runtime and responses are not pinned. No complete git-tracked UC14 execution spec with a versioned evaluator/module exists. Their status is therefore `missing_dependency`; nothing was invented.

The notebook's golden UC4-positive and UC14-negative specs are marked `historical_replay=false` and are deterministic offline control-plane contracts only.
