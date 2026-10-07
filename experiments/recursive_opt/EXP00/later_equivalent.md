# EXP00 — later equivalents of each campaign element

Most EXP00 questions were re-asked later with a corrected instrument, a pre-registration, the control plane or a
different implementation. **None was re-run as-is.** The control-plane v2 migration (21–22 August 2026) classified the
85 saved use-case specs and found 0 replayable
([migration report](../_shared/control_plane_v2/migration_report.md)).

| EXP00 element | Later equivalent |
|---|---|
| The measurement used by A–D | Re-measured by [EXP01](../EXP01/README.md) (signal vs noise), [EXP05](../EXP05/README.md) (concurrency noise) and [EXP10](../EXP10/README.md) (which knobs reach the score) |
| Component code rewriting (A-B, UC1) | [EXP04](../EXP04/README.md) and [EXP06](../EXP06/README.md) (packing code), [EXP15–18](../EXP15/README.md) (optimizer-program discovery), [EXP23–24](../EXP24/README.md) (coevolution on PRISM), [EXP28](../EXP28/README.md) (`VariationSearch`) |
| Capability with accuracy plus cost (A-C, UC3) | Experiment 0 (EXP00-E), pre-registered with GEPA and a no-validation-gate control |
| Family policy / prior transfer (A-D, UC4) | [EXP02](../EXP02/README.md) (UC4 re-scored on the same tasks: −0.006), [EXP07](../EXP07/README.md)–[EXP08](../EXP08/README.md) (routing priors), [EXP22-QA](../EXP22/qa/README.md); UC4 and UC14 also survive as control-plane golden-spec contracts |
| Declarative spec (A-E), UC12 primitives | [Control plane v2](../_shared/control_plane_v2/README.md), used by EXP19–21 and the coevolution engine in EXP23–28 |
| Trainer choice and standard-vs-recursive (Phase 1, three-way benchmark) | [EXP22](../EXP22/evox/README.md), [EXP24](../EXP24/README.md) and [EXP28](../EXP28/README.md), now with fixed-policy controls |
| Trace type (Phase 2, UC6) | [EXP16](../EXP16/README.md) (rich vs compact feedback), [EXP19](../EXP19/README.md) (Trace capture) |
| Priors and skills (Phase 3, Phase 6, UC2) | [EXP18](../EXP18/README.md) (archive memory), [EXP19](../EXP19/README.md) (nested preparation), [EXP20](../EXP20/README.md) (curriculum) |
| Threads (Phase 5) | [EXP05](../EXP05/README.md): running evaluations in parallel changes their noise |
| QASPER prompt and config (UC2/6/11) | [EXP03](../EXP03/README.md), [EXP12](../EXP12/README.md) and [EXP13](../EXP13/README.md) |
| Routing and code transfer (UC7, UC14) | [EXP07](../EXP07/README.md)–[EXP09](../EXP09/README.md) (code transfer: 0 of 22 executable) |
| UC8/UC10 guarded policies, UC13 numeric search | Kept as library code (`opto/features/recursive_opt/decisions.py`, `numeric_optimizers.py`), with no later experiment |
| Optimizer-side tools and agentic policies (Phase 4, UC5, UC9); Terminal-Bench 2 (Phase 7) | **Never re-tested** |

The detailed version, with how each later study is the same or different, is in
[PROTOCOL.md](PROTOCOL.md#later-re-designs-and-equivalents).
