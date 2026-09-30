# EXP22 — two distinct studies

These records share a number, not a protocol or result pool.

- [QA: native nested optimizer learning](qa/README.md)
- [EvoX: PRISM/Signal policy hybrid](evox/README.md)

[All experiments](../README.md)

## Compatibility paths

The top-level `src`, `scripts`, `tests`, `configs`, `artifacts`, `runs`, `worktrees`, `.venv`, `manifest.json`, `EXPERIMENT.md` and `RESULTS.md` names route to the EvoX study. EXP23 resolves these as a sibling EXP22 runtime. The scientific studies remain separated as [QA](qa/README.md) and [EvoX hybrid](evox/README.md). `.gitignore` is an ordinary compatibility file because Git refuses symlinked ignore files.
