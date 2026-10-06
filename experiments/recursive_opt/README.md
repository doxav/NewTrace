# Recursive optimization experiments

## Contents

- [Assessment and ranked research priorities](ASSESSMENT.md)
- [Experiment catalogue](#experiment-catalogue)
- [Shared infrastructure and history](#shared-infrastructure-and-history)

Start with the [scientific assessment](ASSESSMENT.md) for the reconciled lessons and limitations. This catalogue is the common navigation and storage home for the numbered studies. EXP22 has two separate studies; UC numbers and control-plane prompt numbers are different namespaces.

Current verdict (6 October 2026): local task-learning and engineering gains are supported; a reproducible advantage of deeper recursion is not established. EXP23's raw PRISM lead was a scoring exploit; in EXP24 (complete, 9/9) a fixed policy matched both meta arms. EXP25's EvoX lead came from look-ahead SciPy smoothers and vanishes under causality; EXP26 (truncated by a key limit) showed injected stock labels do not close the gap; EXP27's log analysis finds the gap is look-ahead only, not label selection or content.

## Experiment catalogue

| Experiment | Subject | Status |
|---|---|---|
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
| [EXP27](EXP27/README.md) | Root causes of Trace's gap to stock EvoX (five whys, H1–H9) | Part A (logs): gap is look-ahead only; label rate, labels, LLM settings, selection refuted. Part B prompt ablation pending key top-up. |

## Shared infrastructure and history

- [Storage roles, compatibility paths and worktree protection](STORAGE_MAP.md)
- [Restructuring log](RESTRUCTURING_LOG.md)
- [Control-plane engineering evidence](_shared/control_plane_v2/README.md)
- [Numerical optimizer benchmark and Phase 0](_shared/optimizer_discovery/README.md)
- [Early probes and auxiliary diagnostics](_shared/early_probes/README.md)
- [Experiment 0](_shared/experiment_0/README.md) and [shared QA implementation](o1_qa)
- [Earlier UC campaigns](_history/use_cases/README.md), [historical reviews](_history/research_reviews/README.md), and [retired navigation](_history/navigation)
- [Distinct versions preserved from the second worktree](_history/worktree_versions/README.md)

Each study exposes README.md, PROTOCOL.md, RESULTS.md, docs/README.md, results/README.md and a navigation manifest. EXP22-EvoX keeps its original scientific manifest.json; its navigation metadata is catalog_entry.json. Shared code and protocols are now inside this tree; old artifacts/worktree compatibility paths have been removed. See the storage map for internal shared dependencies.
