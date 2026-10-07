# Recursive optimization experiments

## Contents

- [Assessment and ranked research priorities](ASSESSMENT.md)
- [Experiment catalogue](#experiment-catalogue)
- [Shared infrastructure and history](#shared-infrastructure-and-history)

Start with the [scientific assessment](ASSESSMENT.md) for the reconciled lessons and limitations. For EXP22–EXP27 (Trace versus EvoX), see the [2026-10-06 retrospective](_analysis/retrospective_20261006/RETROSPECTIVE.md): impact on recursive_opt itself is flat; the gains are in measurement, the O0 operator and evaluator, and engineering. This catalogue is the common navigation and storage home for the numbered studies. EXP22 has two separate studies; UC numbers and control-plane prompt numbers are different namespaces.

Current verdict (8 October 2026): local task-learning and engineering gains are supported; a reproducible advantage of deeper recursion is not established. EXP23's raw PRISM lead was a scoring exploit; in EXP24 (complete, 9/9) a fixed policy matched both meta arms. EXP25's EvoX lead came from look-ahead SciPy smoothers and vanishes under causality; EXP26 (truncated by a key limit) showed injected stock labels do not close the gap; EXP27 finds the gap is look-ahead only and not a capability limit: Trace's meta-optimizer explores less (refine-locked policies, elite context on DIVERGE). EXP28 adds an explicit mutation intent at the Trainer level (VariationSearch): with it, Trace matches EvoX's discovery speed on Signal and reaches PRISM's all-case optimum in 3–4 calls, ahead of stock EvoX, but no exploration setting improves the legitimate (causal) Signal score. The pre-numbered notebook campaigns and Experiment 0 (EXP00, June–August) fit the same pattern: code rewriting works on clear-feedback surfaces, setup and prior search showed no resolvable gain, and their flagship UC4 result was a comparison artifact.

## Experiment catalogue

| Experiment | Subject | Status |
|---|---|---|
| [EXP00](EXP00/README.md) | Pre-numbered campaigns (Jun–Aug 2026): the `recursive_opt_demo`, `_phases`, `_phases_V2` and `_use_cases` notebooks (now in `EXP00/notebooks/`; UC1–UC14, three-way benchmark) and Experiment 0 (`multiobjective_reasoning`); later re-designs mapped in its PROTOCOL | Historical, documented retroactively. A–D: mechanics only; setup/family/prior surfaces flat or saturated; UC4 +0.163 was an arithmetic identity (corrected −0.006, EXP02); other deltas below resolution. E: Trace −13% and GEPA −26% tokens at no significant accuracy change; no success criterion met. |
| [EXP01](EXP01/README.md) | Prompt signal versus evaluation noise | Historical; use corrected interpretation in the result entry, not the old registry verdict. |
| [EXP02](EXP02/README.md) | Corrected same-holdout UC4 | Historical; use corrected interpretation in the result entry, not the old registry verdict. |
| [EXP03](EXP03/README.md) | Certified prompt optimization | Historical; use corrected interpretation in the result entry, not the old registry verdict. |
| [EXP04](EXP04/README.md) | Packing optimization and retraction | Historical; use corrected interpretation in the result entry, not the old registry verdict. |
| [EXP05](EXP05/README.md) | Concurrency-dependent evaluation noise | Historical; use corrected interpretation in the result entry, not the old registry verdict. |
| [EXP06](EXP06/README.md) | Menu validity and active code surfaces | Historical; use corrected interpretation in the result entry, not the old registry verdict. |
| [EXP07](EXP07/README.md) | Shared routing-menu optimum | Historical; use corrected interpretation in the result entry, not the old registry verdict. |
| [EXP08](EXP08/README.md) | Routing prior amortization | Historical; use corrected interpretation in the result entry, not the old registry verdict. |
| [EXP09](EXP09/README.md) | Generated routing code and transfer | Historical; use corrected interpretation in the result entry, not the old registry verdict. |
| [EXP10](EXP10/README.md) | Configuration-knob liveness | Historical; use corrected interpretation in the result entry, not the old registry verdict. |
| [EXP11](EXP11/README.md) | Knobs on multi-example BBEH | Historical; use corrected interpretation in the result entry, not the old registry verdict. |
| [EXP12](EXP12/README.md) | Paired QASPER smoke | Historical; use corrected interpretation in the result entry, not the old registry verdict. |
| [EXP13](EXP13/README.md) | QASPER found-prompt versus noise | Historical; use corrected interpretation in the result entry, not the old registry verdict. |
| [EXP14](EXP14/README.md) | Historical spec-backlog audit | Historical; use corrected interpretation in the result entry, not the old registry verdict. |
| [EXP15](EXP15/README.md) | Controlled numerical optimizer-program discovery | Complete; feedback versus independent generation inconclusive. |
| [EXP16](EXP16/README.md) | Numerical optimizer-discovery diagnostics | Complete exploratory study; separate P1 from fixed-parent diagnostics. |
| [EXP17](EXP17/README.md) | Confirmatory parent-selection study | Suspended at 545/736 responses; no completed efficacy audit. |
| [EXP18](EXP18/README.md) | Archive memory and Pareto parent selection | Complete; mechanistic contrasts inconclusive. |
| [EXP19](EXP19/README.md) | Trace capture, curriculum and parent-selection learning | Complete; distinguish S3 recursive preparation from S4 selection intervention. |
| [EXP20](EXP20/README.md) | HotpotQA task learning | Complete; standard task learning positive, curriculum advantage unresolved. |
| [EXP21](EXP21/README.md) | Development axes and planned recursive learning | Incomplete development; recursive stages and confirmation not executed. |
| [EXP22/qa](EXP22/qa/README.md) | Nested QA optimizer learning | Incomplete pilot; no established O1 winner. |
| [EXP22/evox](EXP22/evox/README.md) | PRISM/Signal EvoX versus Trace policy hybrid | Historical partial PRISM/Signal comparisons; frozen source and evidence preserved under this directory. |
| [EXP23](EXP23/README.md) | Simulator and upgraded native coevolution | Both native runs complete. Trace 30.8766 exploits the score (3/50 solved); reported best fully solved archive candidates 26.233 vs 26.203, one run each. No established advantage or equivalence. |
| [EXP24](EXP24/README.md) | Valid guide, white-box feedback and repair projection for native coevolution | Complete: 9/9 clean runs reach the all-case optimum (26.256); no meta-level gain over a fixed policy; 30+ stock scores are refusal exploits. |
| [EXP25](EXP25/README.md) | Signal Processing: best Trace configurations versus stock EvoX | Complete: EvoX leads on the stock/valid score through look-ahead SciPy smoothers; all arms equal when restricted to causal filters. |
| [EXP26](EXP26/README.md) | Label fidelity: does stock EvoX's label contract close EXP25's SciPy gap? | Complete but truncated (key limit at iteration 43–70). Injected stock labels do not close the gap at equal budget; R3 mechanism reading withdrawn (see EXP27). |
| [EXP27](EXP27/README.md) | Root causes of Trace's gap to stock EvoX (five whys, H1–H9, Parts A–E) | Complete: gap is look-ahead only and not a capability limit (same climb once found). The cue is minor. SciPy comes only from DIVERGE calls: stock 6/8, Trace 8/189. Trace's meta-optimizer (OptoPrime without the stock design brief) writes refine-locked, elite-context policies. Next: stock brief at O1, then DIVERGE-context replay at O0. |
| [EXP28](EXP28/README.md) | EvoX-style exploration for Trace: VariationSearch trainer vs coevolution EvoX brief / diverge guard; inspiration ablation; PRISM | Complete. VariationSearch default (DIVERGE after a stall) is the best Trace configuration: Signal median 0.758, look-ahead 3/3 by call 22 (EvoX 0.711), DIVERGE yields SciPy 32% vs 3%; PRISM all-case optimum at calls 3/3/4 (EXP24 12–22; stock EvoX 1/3, the rest exploit refusals). Explicit 'combine' sentence limits DIVERGE; EvoX-like context does not. No causal Signal gain. |
| [EXP29](EXP29/README.md) | Where meta / recursive optimization can pay (planning) | Prior analysis only. Meta gains so far were blocked by measurement, headroom and too few meta-level evaluations per run. Priorities: certified-headroom tasks; isolate VariationSearch; replay-based optimization of instructions; cross-episode mode policy; executable helper library. |

## Shared infrastructure and history

- [Storage roles, compatibility paths and worktree protection](STORAGE_MAP.md)
- [Restructuring log](RESTRUCTURING_LOG.md)
- [Control-plane engineering evidence](_shared/control_plane_v2/README.md)
- [Numerical optimizer benchmark and Phase 0](_shared/optimizer_discovery/README.md)
- [Early probes and auxiliary diagnostics](_shared/early_probes/README.md)
- [Experiment 0](_shared/experiment_0/README.md): reported as **[EXP00-E](EXP00/README.md)**. Its code is the [`multiobjective_reasoning`](multiobjective_reasoning/) package, kept at its path because `o1_qa/task.py`, the `_history` probe/audit scripts and the frozen run plans import or record it. It ran 23–31 August 2026, between the use-case notebooks (EXP00-D) and the EXP01–EXP14 audit probes. Also see the [shared QA implementation](o1_qa).
- [Earlier UC campaigns](_history/use_cases/README.md), [historical reviews](_history/research_reviews/README.md), and [retired navigation](_history/navigation)
- [Distinct versions preserved from the second worktree](_history/worktree_versions/README.md)

Each study exposes README.md, PROTOCOL.md, RESULTS.md, docs/README.md, results/README.md and a navigation manifest. EXP22-EvoX keeps its original scientific manifest.json; its navigation metadata is catalog_entry.json. Shared code and protocols are now inside this tree; old artifacts/worktree compatibility paths have been removed. See the storage map for internal shared dependencies.
