# EXP00 — pre-numbered campaigns (June–August 2026): notebooks and Experiment 0

These experiments ran before the numbered EXP series and were never reported here. EXP00 groups them as
sub-studies A–E. The later numbered experiments audited and replaced them: EXP01–EXP14 re-measured them with a
corrected instrument.

| Sub-study | Dates | Source | Question | Valid reading now |
|---|---|---|---|---|
| **A** demo | 7–11 Jun 2026 | [`examples/recursive_opt_demo.ipynb`](../../../examples/recursive_opt_demo.ipynb) | Can each recursion type (setup O1, component code O2a, capability, family policy O2 / prior O3, declarative spec) run and improve its target? | Mechanics only. A code rewrite works on a toy validator (0.80 → 1.00); setup, family and prior surfaces are flat |
| **B** phases | 10–13 Jun 2026 | [`examples/recursive_opt_phases.ipynb`](../../../examples/recursive_opt_phases.ipynb) | Phase-by-phase ADOPT/REJECT of O1 choices: trainer, trace type, warm priors, tools, threads, skills | Nothing adoptable. Warm priors reverse on confirmation; skill and thread records are inconsistent |
| **C** phases V2 | 12 Jun 2026 (re-executed 30 Sep) | [`examples/recursive_opt_phases_V2.ipynb`](../../../examples/recursive_opt_phases_V2.ipynb) | Level-by-level evidence table O0–O3 | Its "positive" claims are not supported by its own outputs |
| **D** use cases | 15 Jun – 5 Jul 2026 | `examples/recursive_opt_use_cases.ipynb` at commit `5a148ddba9` (outputs; later versions are stripped) and [`examples/notebook_outputs/recursive_opt_use_cases/`](../../../examples/notebook_outputs/recursive_opt_use_cases/SUPERSEDED.md) | UC1–UC14 plus a three-way equal-budget benchmark: does recursion beat one-level Trace? | The flagship UC4 +0.163 was an arithmetic identity (corrected −0.006, EXP02); other deltas are below resolution |
| **E** Experiment 0 | 23–31 Aug 2026 | [`multiobjective_reasoning/`](../multiobjective_reasoning/) | Pre-registered GSM8K two-stage program: fixed vs Trace vs GEPA vs Trace without validation gate (40 runs) | Neither engine met its success criterion. Trace −13% tokens and GEPA −26% at no significant accuracy change; stopped for missing trajectory provenance |

| Role | Entry point |
|---|---|
| Protocols (reconstructed, one per sub-study) | [PROTOCOL.md](PROTOCOL.md) |
| Results and validity per sub-study | [RESULTS.md](RESULTS.md) |
| Navigation metadata | [manifest.json](manifest.json) |
| August 2026 audit of A–D | [`_history/reviews/recursive_opt_assessment.history_20260929.md`](../_history/reviews/recursive_opt_assessment.history_20260929.md) (§5, §7) |
| Author's June analysis of D (superseded) | [`examples/recursive_opt_use_cases_CURRENT_LIMITS.MD`](../../../examples/recursive_opt_use_cases_CURRENT_LIMITS.MD) |
| Experiment 0 decisions and run data | [`_shared/experiment_0/`](../_shared/experiment_0/README.md), `outputs/recursive_opt/experiment_0/` |

**Where the code stays.** `multiobjective_reasoning/` (Experiment 0) is left at its path. It is imported by
`o1_qa/task.py` and by the `_history/probe_2026` and `_history/audit_20260924` scripts, and its module path is
recorded in the frozen run plans and control-plane locks under `outputs/recursive_opt/experiment_0/`. Moving it
would break imports and invalidate that provenance. EXP00 documents it in place. The notebooks also stay in
`examples/`.

[All experiments](../README.md) · [Assessment](../ASSESSMENT.md)
