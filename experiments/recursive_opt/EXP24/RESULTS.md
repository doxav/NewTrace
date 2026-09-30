# EXP24 — results and dated clean-run checkpoint

## Contents

- [Metric correction and intervention](#metric-correction-and-intervention)
- [Exploratory pilot](#exploratory-pilot)
- [Clean-run checkpoint](#clean-run-checkpoint)
- [Interpretation and remaining checks](#interpretation-and-remaining-checks)
- [Evidence links](#evidence-links)

## Metric correction and intervention

PRISM's stock score is `1 / mean(KVPR over solved cases) + success_rate`. Crashing on difficult cases can raise it. EXP23's stock-selected Trace program scores 30.876583 but solves only **3/50** cases; EXP22's record EvoX program solves **7/50**. Counting failed cases at the initial placement's KVPR produces scores **21.084775** and **21.423051**, below the initial program's **21.891622**. [Independent selected-source re-scoring](../_analysis/assessment_20260930/prism_rescoring.json).

The computed optimum on all 50 fixed cases is **26.2559717495**. The solver uses exhaustive bin-packing feasibility with floating-point tolerances and numerical bisection; “exact” denotes the combinatorial search, not symbolic arithmetic. Scores above this all-case ceiling cannot be valid improvements under the same constraints. [Solver](scripts/prism_exact.py), [saved per-case values](results/analysis/prism_exact_optimum.json).

EXP24 changes three components together: a valid all-case guide, per-case feedback, and a fallback projection that replaces invalid placement attempts with the initial placement for the affected case. This combined intervention does not isolate which component helps. [Code delta](docs/CODE_DELTA.md), [white-box evaluator](prism/whitebox.py).

## Exploratory pilot

The pilot used one run per arm before the clean protocol. Original events and receipts remain in [pilot_20260930T064616](results/pilot_20260930T064616/).

| Run | Best fully solved archive score / best valid guide | Calls to ceiling | Wasted attempts |
|---|---:|---:|---:|
| EXP23 EvoX-style, stock metric | 26.203 | Not reached in reported archive | 23 |
| EXP23 Trace, stock metric | 26.233 reported archive maximum; selected exploit re-scores to 21.0848 | Not reached in reported archive | 24 |
| EXP24 pilot, EvoX-style | 26.255972 | 2 | 23 |
| EXP24 pilot, Trace | 26.255972 | 10 | 12 |

The pilot EvoX-style arm reached the ceiling before a learned policy deployment. Therefore a fixed-policy control is necessary to attribute value to meta-optimization. The pilot also had outage/budget-accounting limitations; it is not pooled with the clean comparison. “Wasted” follows `scripts/analyze.py`'s attempt/error convention, not a complete monetary-cost measure. The EXP23 rows use retrospective archive validity, whereas EXP24 optimizes the corrected guide directly.

## Clean-run checkpoint

Design: **fixed policy / EvoX-style `llm_rewrite` / Trace × seeds 42, 43, 44**, target 100 solution calls each, identical lower-level intervention. Primary endpoint is solution calls until first reaching the computed ceiling, using the protocol's numerical tolerance. [Preregistered protocol](PROTOCOL.md).

**Checkpoint: 30 September 2026, 12:12 UTC.** Four runs have terminal success summaries; five are incomplete at this snapshot. All nine already have observed first-hit endpoints. Values were recomputed from `events.jsonl`; hashes and recorded call accounting are preserved in the [audit snapshot](../_analysis/assessment_20260930/verification.json).

| Arm / seed | Completed solution coordinate | First ceiling hit | Best valid score | Terminal success |
|---|---:|---:|---:|---|
| fixed / 42 | 100 | 8 | 26.255972 | Yes |
| fixed / 43 | 100 | 12 | 26.255972 | Yes |
| fixed / 44 | 100 | 32 | 26.255972 | Yes |
| llm_rewrite / 42 | 89 | 12 | 26.255972 | Not yet recorded |
| llm_rewrite / 43 | 73 | 5 | 26.255972 | Not yet recorded |
| llm_rewrite / 44 | 54 | 33 | 26.255972 | Not yet recorded |
| trace / 42 | 100 | 45 | 26.255972 | Yes |
| trace / 43 | 85 | 22 | 26.255972 | Not yet recorded |
| trace / 44 | 77 | 8 | 26.255972 | Not yet recorded |

First-hit medians: **fixed 12, llm_rewrite 12, Trace 22** solution calls. These are descriptive observations from three runs per arm. They show no observed speed advantage for Trace and do not establish statistical equivalence or inferiority. Subsequent completion cannot change these historical first hits, but total costs and final failure/deployment counts are still incomplete. [Raw campaign](results/clean_20260930T115256/).

## Interpretation and remaining checks

- Fixed policy already reaches the ceiling in all three runs. Do not attribute ceiling attainment alone to policy learning or additional recursion.
- Every arm receives the same 50 cases as both feedback and score; this is an in-benchmark speed/robustness test, with no holdout. Policy seeds do not seed LLM sampling.
- Equal solution calls are not equal total compute: account for meta proposals, guide calls, retries, tokens, known costs and unknown billing. Do not rank final campaign cost from this partial snapshot.
- The saved white-box feedback still says “honest ceiling 29.40”. The correct computed value is 26.2559717495; guide arithmetic and endpoint analysis use the latter. Preserve the ongoing treatment and version any future prompt correction prospectively.
- After terminal completion, reconcile all nine summaries with event/call logs and report every run. A new experiment must separate the valid metric, feedback and projection if it seeks causal attribution among those changes.

Five existing offline tests passed during this review in 20.529 seconds: stock-metric reproduction, exploit/fallback behavior, invalid GPU IDs, optimum aggregation with three case spot checks, and mock execution of all three arms. They establish these engineering properties, not a recursive efficacy result.

## Evidence links

- [Reconciled assessment and ranked priorities](../ASSESSMENT.md)
- [EXP23 corrected result](../EXP23/RESULTS.md)
- [EXP22-EvoX historical hybrid](../EXP22/evox/RESULTS.md)
- [Protocol](PROTOCOL.md), [analysis implementation](scripts/analyze.py), [offline tests](tests/test_exp24.py)
- [Pilot evidence](results/pilot_20260930T064616/) and [clean campaign](results/clean_20260930T115256/)
- [Optimum data](results/analysis/prism_exact_optimum.json) and [rewrite archive validity](results/analysis/exp23_llm_rewrite_validity.json)
- [Audit snapshot](../_analysis/assessment_20260930/verification.json) and [selected-source re-scoring](../_analysis/assessment_20260930/prism_rescoring.json)
