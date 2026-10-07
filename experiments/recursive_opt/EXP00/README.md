# EXP00 — pre-numbered campaigns (June–August 2026): notebooks and Experiment 0

These experiments ran before the numbered EXP series and were not reported here until now. EXP00 groups them as
sub-studies A–E. The later numbered experiments audited and replaced them: EXP01–EXP14 re-measured them with a
corrected instrument.

| Sub-study | Dates | Source | Question | Valid reading now |
|---|---|---|---|---|
| **A** demo | 7–11 Jun 2026 | [`notebooks/recursive_opt_demo.ipynb`](notebooks/recursive_opt_demo.ipynb) | Can each recursion type (setup O1, component code O2a, capability, family policy O2 / prior O3, declarative spec) run and improve its target? | Mechanics only. A code rewrite works on a toy validator (0.80 → 1.00); setup, family and prior surfaces are flat |
| **B** phases | 10–13 Jun 2026 | [`notebooks/recursive_opt_phases.ipynb`](notebooks/recursive_opt_phases.ipynb) | Phase-by-phase ADOPT/REJECT of O1 choices: trainer, trace type, warm priors, tools, threads, skills | Nothing adoptable. Warm priors reverse on confirmation; skill and thread records are inconsistent |
| **C** phases V2 | 12 Jun 2026 (re-executed 30 Sep) | [`notebooks/recursive_opt_phases_V2.ipynb`](notebooks/recursive_opt_phases_V2.ipynb) | Level-by-level evidence table O0–O3 | Its "positive" claims are not supported by its own outputs |
| **D** use cases | 15 Jun – 5 Jul 2026 | [`notebooks/recursive_opt_use_cases_executed_5a148ddba9.ipynb`](notebooks/recursive_opt_use_cases_executed_5a148ddba9.ipynb) (archived executed version) and [`examples/notebook_outputs/recursive_opt_use_cases/`](../../../examples/notebook_outputs/recursive_opt_use_cases/SUPERSEDED.md) | UC1–UC14 plus a three-way equal-budget benchmark: does recursion beat one-level Trace? | The flagship UC4 +0.163 was an arithmetic identity (corrected −0.006, EXP02); other deltas are below resolution |
| **E** Experiment 0 | 23–31 Aug 2026 | [`multiobjective_reasoning/`](../multiobjective_reasoning/) | Pre-registered GSM8K two-stage program: fixed vs Trace vs GEPA vs Trace without validation gate (40 runs) | Neither engine met its success criterion. Trace −13% tokens and GEPA −26% at no significant accuracy change; stopped for missing trajectory provenance |

| Role | Entry point |
|---|---|
| Protocols (reconstructed, one per sub-study) | [PROTOCOL.md](PROTOCOL.md) |
| Results and validity per sub-study | [RESULTS.md](RESULTS.md) |
| Navigation metadata | [manifest.json](manifest.json) |
| August 2026 audit of A–D | [`_history/reviews/recursive_opt_assessment.history_20260929.md`](../_history/reviews/recursive_opt_assessment.history_20260929.md) (§5, §7) |
| Author's June analysis of D (superseded) | [`examples/recursive_opt_use_cases_CURRENT_LIMITS.MD`](../../../examples/recursive_opt_use_cases_CURRENT_LIMITS.MD) |
| Experiment 0 decisions and run data | [`_shared/experiment_0/`](../_shared/experiment_0/README.md), `outputs/recursive_opt/experiment_0/` |

**Later re-designs.** Each campaign was later re-asked differently. The full mapping is in
[PROTOCOL.md](PROTOCOL.md#later-re-designs-and-equivalents):
- A–D's instrument was re-measured by EXP01–EXP14; UC4 specifically by EXP02.
- Component-code rewriting continued in EXP15–EXP18, EXP23–EXP24 and EXP28.
- The capability multi-objective question became Experiment 0 (E).
- Declarative specs became control plane v2. The current `examples/recursive_opt_use_cases.ipynb` is its smoke
  notebook for the UC4/UC14 golden specs.
- Standard-vs-recursive comparisons became EXP22, EXP24 and EXP28 with fixed-policy controls.
- Optimizer-side tools and agentic policies were never retested.

**Notebooks.** `recursive_opt_demo`, `recursive_opt_phases` and `recursive_opt_phases_V2` were moved here from
`examples/` with `git mv`, so their history is kept. `recursive_opt_use_cases_executed_5a148ddba9.ipynb` is a copy of
the last executed version of the use-case notebook; the file at `examples/` is now the control-plane smoke notebook.
The archived notebooks are records with their saved outputs. Their cells assume the repository root as working
directory and `examples/` on `sys.path`, so re-running them needs that layout. The helper scripts
(`examples/recursive_opt_example_*.py`, `recursive_opt_three_way.py`, `recursive_opt_abc_probe.py`,
`recursive_opt_review_regression.py`, `recursive_opt_use_cases_CURRENT_LIMITS.MD`) stay in `examples/` because unit
tests, later verification scripts and historical documents reference them there.

**Where the code stays.** `multiobjective_reasoning/` (Experiment 0) is left at its path. It is imported by
`o1_qa/task.py` and by the `_history/probe_2026` and `_history/audit_20260924` scripts, and its module path is
recorded in the frozen run plans and control-plane locks under `outputs/recursive_opt/experiment_0/`. Moving it
would break imports and invalidate that provenance. EXP00 documents it in place. The notebooks also stay in
`examples/`.

[All experiments](../README.md) · [Assessment](../ASSESSMENT.md)
