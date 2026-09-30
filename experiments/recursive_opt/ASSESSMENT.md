# Recursive optimization: evidence and research priorities

## Contents

1. [Verdict and scope](#verdict)
2. [What was actually optimized](#architecture)
3. [Experiment register: EXP01–24](#experiment-register)
4. [Results that change the conclusion](#findings)
5. [Ranked lessons: gains, failures and limits](#lessons)
6. [Short-term priorities](#short-term)
7. [High-potential research](#future-work)
8. [Verification and evidence links](#audit)

<a id="verdict"></a>
## 1. Verdict and scope

Reviewed **30 September 2026** against the numbered `RESULTS.md` reports and saved evidence. EXP24 is a dated checkpoint, not a completed campaign: the snapshot below was captured at **12:12 UTC**. [Catalogue](README.md), [verification snapshot][verification].

**There is useful progress, but no established reproducible advantage from deeper recursive optimization over strong, equally budgeted alternatives.** The clearest empirical gains are task optimization in EXP20, a parent-selection intervention in EXP19, and numerical optimizer generation relative to the initial program in EXP15. They do not isolate an advantage of additional recursive depth. The control plane, provenance, working search surfaces and native coevolution engine are separate engineering advances.

**The previous assessment's positive EXP23 headline is withdrawn.** Trace's stock PRISM score of 30.876583 rewards solving only 3/50 cases. Both native arms finished 100 solution attempts. Their reported best *fully solved* candidates score about 26.233 and 26.203; this small difference in one run each establishes neither superiority nor statistical equivalence. EXP24's fixed-policy control already reaches the corrected benchmark ceiling in all three runs. [EXP23 correction][exp23], [EXP24][exp24].

This document replaces the repeated error catalogue, 62 overlapping lessons, stale intermediate-run verdicts and obsolete reproduction paths. Historical reports remain available as evidence of what was observed and claimed at the time; later corrections take precedence for interpretation. Use saved results and executed code before narrative summaries. A confidence interval containing zero means an unresolved contrast; a stopped run is not a negative result; passing software tests is not evidence of scientific efficacy.

<a id="architecture"></a>
## 2. What was actually optimized

| Series | Lower task / artifact | Upper intervention actually exercised | Attribution limit |
|---|---|---|---|
| EXP01–14 and earlier UC | Prompts, code, fixed candidate menus, configuration | Several different interventions and instrument audits | UC numbers, experiment numbers and control-plane prompt numbers are different namespaces. No single uniform recursive treatment. |
| EXP15–18 | A generated `propose(history, bounds, seed)` program; 32 numerical objective calls per trajectory | Independent code generation, selected-parent mutation, feedback, memory or Pareto selection | Meta-optimization of a numerical optimizer, not a comparison of increasing Trace recursion depth. |
| EXP19 | Six synthetic four-bit truth tables | Small nested preparation; separately, an internal parent-selection intervention | S4 is neither curriculum nor an O2 efficacy result. |
| EXP20–21 | HotpotQA ranker code, reader instruction, document count and bridge setting | Fixed training configurations; development ablations | EXP20 does not learn O1; EXP21 did not execute its planned O1/O2 stages. |
| [EXP22-QA](EXP22/qa/RESULTS.md) | Native task learning with four documents fixed | Learned update instruction and evidence-selector source | Incomplete pilot; no completed upper-level comparison. |
| [EXP22-EvoX](EXP22/evox/RESULTS.md) | SkyDiscover solution search and population | Trace rewrites `EvolvedProgramDatabase` policy source | Historical hybrid; not two nested native Trace trainers. |
| EXP23–24 native coevolution | Shared population of solution programs | EvoX-style `llm_rewrite` or persistent Trace proposer changes policy source; engine manages triggers, scoring and deployment | Compare these proposers within the native engine. This is not a replicated comparison against the entire stock EvoX system. EXP24 adds a fixed-policy arm. |

EXP19–21 retain the `recursive-opt/v2alpha` declarative schema, with study-specific modules, evaluators and configuration extensions. Sharing a dict format does not mean identical runtime behavior or a self-contained portable experiment. The [control-plane migration](_shared/control_plane_v2/migration_report.md) classified 85 files but certified **zero faithful execution replays**. Trace capture, prompt delivery and demonstrated benefit must be checked separately. [EXP19 configuration and execution](EXP19/RESULTS.md), [shared control-plane evidence](_shared/control_plane_v2/README.md).

<a id="experiment-register"></a>
## 3. Experiment register: EXP01–24

Response counts refer to completed optimizer responses unless specified. They are not total model calls or independent scientific replications. Each experiment link opens its report and raw-evidence links. Accuracy differences are percentage points (**pp**); numerical regret AUC is **lower better**.

<a id="early-lessons"></a>
### Early probes and instrument qualification

| Study | Tested surface and volume | Result and defensible interpretation |
|---|---|---|
| [EXP01](EXP01/RESULTS.md) | 3 prompts × 3 repeats × 2 tasks; zero inner learning steps | Signal/noise ratios 0.96 and 0.74. This small instrument did not cleanly resolve prompt effects. |
| [EXP02](EXP02/RESULTS.md) | Corrected UC4 on the same holdout; 3 seeds planned, only 1 usable pair | Paired difference about −0.006. Earlier +0.163 compared different tasks and is invalid. |
| [EXP03](EXP03/RESULTS.md) | GSM8K prompt search; 5 attempted seeds, 4 usable pairs | Raw paired mean **+0.1768125**, not the old +0.217. Heterogeneity, noise and quality/cost attribution prevent an efficacy claim. |
| [EXP04](EXP04/RESULTS.md) | Packing code search; 3 attempted seeds, 2 usable pairs | Apparent +4.8 withdrawn: purported optimized artifact empty and concurrency noise not calibrated. |
| [EXP05](EXP05/RESULTS.md) | 8 tasks × concurrency 1/8 × 6 repeats | Packing SD 0 serial versus 4.40555 at eight workers. Calibration must match execution conditions. |
| [EXP06](EXP06/RESULTS.md) | Type-correct code menus: 8 packing and 6 admissible-set heuristics | Four distinct scores on each surface; ranges 2,908.2 and 390. Demonstrates responsiveness, not recursive efficacy. |
| [EXP07](EXP07/RESULTS.md) | 9 routing heuristics × 4 tasks; example-limit checks | `nearest` best 4/4; two tasks nearly duplicates. Fixed-menu structure, not a universal condition for transfer. |
| [EXP08](EXP08/RESULTS.md) | Routing order learned from source tasks; exact finite-menu expectations, 360 repeated scores | Menu-optimal quality at target budget 1 versus 9; 18 upstream evaluations, break-even 2.25 deployments. Fixed `nearest` matches without meta-training cost. |
| [EXP09](EXP09/RESULTS.md) | 36 independent code responses; 11 valid source programs transferred to two siblings each | 0/22 sibling transfers executable because of signatures. Historical VRPTW code improves distance 21.81% on its fixture; no recursive attribution. |
| [EXP10](EXP10/RESULTS.md) | Configuration knobs on 13 bundles; inventory of 47 tasks | Single external example in 13/13 inspected bundles and 43/47 tasks; zero inner steps. No useful test of an active curriculum. |
| [EXP11](EXP11/RESULTS.md) | 15-example BBEH; control n=10, five repeats per knob value | Almost saturated score; possible tiny batch-size signal (inter/intra range ratio 2.11). Not evidence all effects lie within noise, nor of useful transfer. |
| [EXP12](EXP12/RESULTS.md) | QASPER paired smoke, 2 seeds / 4 runs | Mean −0.18918, paired SD 0.25365. Too small and poorly controlled to generalize. |
| [EXP13](EXP13/RESULTS.md) | Found versus empty prompt, 6+6 interleaved attempts | **All 12 scores null**, 480 seconds recorded each. Finished without usable measurements; not an ongoing efficacy study. |
| [EXP14](EXP14/RESULTS.md) | Inventory of 90 specifications / 18 variants | Engineering audit; eight variants marked unrunnable, 70 menu levels share five candidates. Not 90 scientific trials. |

<a id="numerical-lessons"></a>
<a id="23--optimizer-discovery-phase-0-instrument-and-interface"></a>
<a id="24--optimizer-generation-readiness-calibration"></a>
<a id="261-what-the-staged-diagnostics-isolate"></a>
### Numerical optimizer discovery and later learning studies

[Phase 0](_shared/optimizer_discovery/PHASE0_REPORT.md) established an executable optimizer interface and generation viability: 0/3 usable outputs at the original 3,000-token setting, then 10/10 valid fixture outputs at the chosen 8,000/low-reasoning setting. This is calibration; settings and service conditions are not an isolated causal treatment.

| Study | Volume / actual comparison | Result and status |
|---|---|---|
| [EXP15][exp15] | 5 outer seeds × 2 arms × 8 responses = 80; initial program, independent A1, feedback A2 | Complete. AUC A0/A1/A2 = 0.139579 / 0.121748 / 0.116680. A2 beats A0 locally; A2 versus A1 unresolved. |
| [EXP16][exp16] | P1: 6 seeds × I/C/R/W × 8 = 192 responses; G1 24, F1 24 and pilot 2 separately | Exploratory. I/C/R/W AUC = 0.077260 / 0.047588 / 0.122517 / 0.112582. Rich feedback R worse than I/C locally; C−I favorable post-hoc. |
| [EXP17](EXP17/RESULTS.md) | Confirm C−I: 46 paired seeds, 736 planned responses | Suspended at **545/736**. No generation freeze, final selection or audit result in the pause record; no confirmatory verdict. |
| [EXP18][exp18] | 6 seeds × L/M/P/PM × 16 = 384 responses; 864 audit trajectories including controls | Complete. Memory, Pareto and interaction contrasts unresolved; no independent-generation N16 arm. |
| [EXP19][exp19] | S3: 72 responses; S4: 3 new seeds × 2 arms × 4 = 24; 255 across the broader campaign | Complete local diagnostics. S3 standard/O1/recursive TEST 47.92/50.00/49.31%; selected recursive configuration unchanged. S4 100% versus 72.22%, a bundled parent-selection intervention. |
| [EXP20][exp20] | 6 seeds × standard/curriculum × 6 = 72; TRAIN/VAL/TEST 60/24/48 | Complete. Primary TEST curve 43.11% standard versus 28.47% unchanged; curriculum 39.73%. Standard learning positive; curriculum increment unresolved. |
| [EXP21](EXP21/RESULTS.md) | 21 configurations × 2 seeds × 6 planned = 252 | **219/252** responses, **33/42** complete curves. Development incomplete; O1/O2 and confirmation not executed. |
| [EXP22-QA](EXP22/qa/RESULTS.md) | Four-document headroom pilot; 4 upper proposals and partial lower chains; 33 optimizer + 730 reader responses including diagnostics | Provider-blocked pilot; nine lower responses remaining at its saved checkpoint. Initial/code/instruction/combined 8/10/10/12 correct of 24. No O1 winner. |
| [EXP22-EvoX](EXP22/evox/RESULTS.md) | v9: PRISM recursive/fixed 77/100 attempts; Signal 93/100 | Fixed arms complete, recursive arms provider-stopped. PRISM stock scores 24.851/26.364; Signal 0.5438/0.6063. Unequal budgets and PRISM's flawed score prohibit a valid method ranking. |
| [EXP23][exp23] | Simulator studies; 120 live policy-model calls evaluated in simulation; later two native PRISM runs × 100 solution attempts | Native runs complete; claimed Trace advantage invalidated by the metric exploit. Simulation and integration evidence remain separately useful. |
| [EXP24][exp24] | Pilot: one run/arm; clean: fixed / llm_rewrite / trace × seeds 42/43/44, 100 solution calls planned each | At 12:12 UTC, 4/9 terminal runs, but all 9 already reached the ceiling. First-hit medians fixed/rewrite/Trace **12/12/22** calls. Descriptive checkpoint, no demonstrated meta-level gain. |

<a id="findings"></a>
## 4. Results that change the conclusion

### EXP15–18: optimizer search is useful locally; feedback's added value is unresolved or adverse

EXP15's feedback arm improves on the original program: A2−A0 = **−0.022899**, reported paired 95% interval **[−0.042456, −0.003341]**. The relevant incremental comparison is A2−independent A1 = **−0.005068 [−0.037728, +0.024551]**. Five outer seeds and a fixed narrow numerical panel do not establish superiority over independent generation. [Saved EXP15 results][exp15-json].

EXP16's corrected rich-feedback treatment was actually delivered and still performed worse: R−I = **+0.045257 [+0.003176, +0.085126]**. Selected-parent code-only C was better than I by 0.029672 post-hoc; C already receives information through TRAIN parent selection. EXP17 has not confirmed it. An informed center-first control B2 achieved AUC **0.031633**, compared with initial A0 **0.179329** on EXP16's separate audit; it does not dominate final regret. Better initialization and better anytime performance are not proof of a feedback or depth effect. [EXP16 analysis][exp16-json], [P1/B2 tables](_shared/optimizer_discovery/investigation16/production/presentation/data.json).

EXP18's memory effect is **−0.004114 [−0.020651, +0.014910]**, Pareto effect **+0.008512 [−0.015019, +0.035682]**. All seven prespecified mechanistic intervals cross zero. Archive exposure and alternative parent selection occurred, but benefit was not established. Six outer seeds remain six replications despite 864 audit trajectories; 48/384 search proposals were ineligible even though selected deployments had zero fallback. [EXP18 analysis][exp18-json], [mechanism exposure](_shared/optimizer_discovery/exp18/PROGRAM_REVIEW.md).

<a id="learning-lessons"></a>
### EXP19–22-QA: lower-level learning gains survive, with attribution limits

EXP19 S4 achieves **100% versus 72.22%** final TEST accuracy after four responses per arm: **+27.78 pp [22.92, 31.25]**, three seeds. It separates internal TRAIN-based parent selection from subsequent external VALIDATION selection. The old path compared scores accumulated over different panels; the intervention changes multiple selection properties. It is a useful local repair, not proof that validation is harmful or that recursion caused the gain. The six synthetic concepts stay fixed; a direct table-memory control can solve this small learning problem without generative search. [S4 data][s4-results], [mechanism review](_shared/o1_learning/s4_mechanism_review_20260923.json).

EXP20's standard treatment increases mean TEST accuracy over prefixes 0–6 from **28.47% to 43.11%**, **+14.63 pp [11.41, 17.81]**; final accuracy is **47.92%**. This is a real local task-learning result. However, **8/12** final learned configurations use ten documents, including **5/6** standard runs, without a same-reader fixed-ten-document control. Code, instruction and information budget changed together. Curriculum−standard is **−3.37 pp [−8.53, +2.38]**, despite 33 actual curriculum transitions. This refutes neither curriculum generally nor additional depth. [EXP20 data][exp20-json], [saved raw audit](EXP20/results/run_store/full/confirmation/raw_recomputation.json).

EXP21's many two-seed development comparisons are hypothesis generators, not a winner-selection result. EXP22-QA fixes the document count and shows both code and instruction headroom, but upper-level learning remains unfinished. More tokens also do not guarantee parsing: one of two 32,000-token diagnostics still failed the native parser. [EXP21 machine status](EXP21/results/run_store/continuation_summary.json), [EXP22-QA machine status](EXP22/qa/results/run_store/execution_summary.json).

<a id="coevolution-lessons"></a>
### EXP23–24: the metric correction reverses the headline

The stock PRISM score uses `1 / mean(KVPR on solved cases) + success_rate`. Exceptions on difficult cases can remove them from the average. Trace's saved best program scores **30.876583 with success_rate=0.06**: 47/50 cases fail. Counting all cases, with the initial placement's KVPR for failed cases, gives **21.084775**, below the initial program's **21.89**. The EXP22 EvoX record has the same failure mode (7/50 solved, corrected score about 21.42). The offline tests reproduce this mechanism; it is not just an interpretation of the leaderboard. [Saved Trace report](EXP23/results/prism100_v2/20260929T215903/trace/report.json), [EXP24 evaluator](EXP24/prism/whitebox.py), [tests](EXP24/tests/test_exp24.py).

Do not confuse **re-scoring the selected exploit** (21.084775) with **retrospectively choosing the best fully solved candidate in its archive** (about 26.233, as reported by EXP24). The latter is not what the raw-score selection deployed. The native `llm_rewrite` run's saved best fully solved candidate is about 26.203. Neither the raw-score lead nor retrospective re-selection establishes a Trace advantage. [EXP23 results][exp23], [selected-source re-scoring and audit limit](_analysis/assessment_20260930/prism_rescoring.json), [rewrite archive re-scoring](EXP24/results/analysis/exp23_llm_rewrite_validity.json).

EXP24 computes an optimum of **26.2559717495** for these **50 fixed cases**, using exhaustive packing feasibility and numerical bisection. “Exact” refers to the combinatorial feasibility search, with floating-point tolerances; it is not a symbolic proof or an unseen-task result. The saved per-case values reproduce the aggregate, and three cases were recomputed by the offline tests. This turns the primary question into calls-to-ceiling, not further final-score improvement. [Computation](EXP24/scripts/prism_exact.py), [saved values](EXP24/results/analysis/prism_exact_optimum.json), [protocol](EXP24/PROTOCOL.md).

EXP24 changes **three things together**: valid guide, case-level feedback and per-case fallback projection. Its pilot reached the ceiling at calls 2 (`llm_rewrite`) and 10 (Trace); the first was before any learned policy deployment. The clean study correctly adds fixed policy with the same lower-level treatment. At the saved checkpoint:

| Arm | First ceiling hit, seeds 42 / 43 / 44 | Median | Completed solution coordinates | Terminal 100-call runs |
|---|---|---:|---|---:|
| Fixed policy | 8 / 12 / 32 | 12 | 100 / 100 / 100 | 3/3 |
| EvoX-style `llm_rewrite` | 12 / 5 / 33 | 12 | 89 / 73 / 54 | 0/3 |
| Trace proposer | 45 / 22 / 8 | 22 | 100 / 85 / 77 | 1/3 |

All nine first-hit endpoints are observed; subsequent completion cannot change their historical first hit. Final costs, waste and deployment counts remain incomplete in five runs. **No observed speed advantage for Trace; no statistical equivalence or inferiority claim at n=3.** Policy RNG seeds do not seed the LLM's sampling. Both feedback and scoring reuse the same 50 cases, so no generalization claim is available. Equal solution calls also omit additional meta/guide calls and transport retries. [Dated snapshot][verification], [EXP24 evidence](EXP24/results/README.md).

A remaining reproducibility defect is explicit: the EXP24 white-box feedback still says “honest ceiling 29.40”, whereas the saved optimum and analysis use 26.2559717495. This inaccurate prompt text is shared across arms; it does not change the arithmetic of the guide. Correct it only in a prospectively versioned follow-up, preserving the current running treatment. The lower-level fixes also need ablation before crediting the improvement specifically to white-box feedback or fallback.

EXP23's simulator results remain evidence about its simulator: stage-dependent policy credit, noisy short windows and across-episode selection. Some simulated PRISM scores exceed the subsequently established live ceiling, so these worlds are not a physically bounded forecast of corrected PRISM performance. The 120 live calls were **policy proposals with simulated task evaluation**, not 120 live task-search comparisons. [Score validation](EXP23/results/score_validation.json), [meta comparison](EXP23/results/meta_comparison.json), [live-policy summary](EXP23/results/live_schedule/live/summary.json).

<a id="lessons"></a>
## 5. Ranked lessons: gains, failures and limits

Ratings express **strength of evidence for the stated lesson**, not a probability of general success: **5/5** directly checked failure mechanism or convergent controls; **4/5** completed local comparison with important scope limits; **3/5** exploratory, unfinished or simulator-based. Priority is a separate judgment about what would most improve the next experiment.

| Rank | Lesson | Evidence rating | What to retain / avoid |
|---:|---|---:|---|
| 1 | Validate what a score rewards before optimizing it. | **5/5** | EXP23 exploit reproduced; EXP24 ceiling and fixed control. Retain per-case validity, all-case accounting and known bounds. Never call a higher proxy score a better algorithm without checking outcomes. |
| 2 | Compare against informed simple controls. | **5/5** | EXP08 fixed `nearest`, EXP16 B2, EXP24 fixed policy. Keep these controls and independent generation; initial-seed wins alone cannot identify meta-learning value. |
| 3 | Make parent and child scores comparable. | **4/5** | EXP19 S4 large local gain; EXP16 selection-bank diagnostics. Keep external validation; isolate common-panel scoring from TRAIN-versus-VAL selection in a new test. |
| 4 | Task learning works in some settings. | **4/5** | EXP20 +14.63 pp primary TEST curve; EXP15 better numerical code than seed. Preserve these positive results, but separate task improvement from feedback, information-budget and recursion-depth effects. |
| 5 | More feedback text is not an established improvement. | **4/5** | EXP16 R−I unfavorable, with a separate fixed-parent F1 warning. Do not enlarge traces blindly; test compact relevant evidence at matched parent, surface and cost. |
| 6 | Verify that a treatment actually reaches execution and the final prompt. | **5/5** | EXP10 inactive knobs, EXP15 lossy/misdirected feedback, EXP22 hybrid naming. Retain end-to-end liveness, resolved-source and prompt checks; a schema field or label is insufficient. |
| 7 | Memory, Pareto and curriculum remain hypotheses, not validated defaults. | **4/5** for the unresolved verdict | EXP18 active mechanisms but wide intervals; EXP20 active curriculum without resolved advantage. Keep opt-in implementations and negative evidence; do not infer equivalence or stack them as known gains. |
| 8 | Count the right replication unit and full cost. | **5/5** | Hundreds of trajectories can come from six search seeds; retries and meta/reader calls exceed solution budgets. Keep failures and distinguish physical calls, allocations, tokens, cost and wall time. |
| 9 | Policy credit and scheduling can constrain online learning. | **3/5** | EXP23 stage-bias simulations and few triggered proposals. Pair or otherwise calibrate credit on fresh live episodes; simulator success is insufficient. |
| 10 | A shared control plane is useful infrastructure, not reproduced science. | **4/5** | Declarative runs, typed outcomes and provenance are reusable; migration certified zero historical replays. Maintain source/version-specific tests and avoid turning test counts into performance evidence. |

<a id="short-term"></a>
## 6. Short-term priorities

Scores below rank **expected information value and practicality**, not expected performance gain.

| Priority | Opportunity | Rating | Smallest informative next step / decision rule |
|---:|---|---:|---|
| 1 | Close EXP24 honestly and retain a strong baseline. | **5/5** | Preserve the running protocol; reconcile all nine terminal summaries, solution/meta/guide/retry counts and missing billing. Report every first hit and total cost. Keep fixed policy as default comparator unless a meta method adds measurable value. No more final-score chasing on saturated PRISM. |
| 2 | Isolate parent-selection improvements. | **5/5** | New-task common-panel versus legacy scoring ablation, with validation policy controlled separately. Require held-out improvement; preserve independent external selection. Treat EXP17 as suspended, not as confirmation. |
| 3 | Separate information access from learning in QA. | **4/5** | Same Qwen reader: fixed four versus ten documents; then fixed instruction versus learned code/instruction at a fixed document budget. Use new questions before claiming the EXP20 gain is algorithmic. |
| 4 | Isolate the EXP24 lower-level changes. | **4/5** | On unsaturated new instances, valid metric for every arm, then add case feedback and fallback separately. Report fallback usage and candidate validity alongside score. Do not reuse metric exploitation as an acceptable control objective. |
| 5 | Test compact feedback against the strongest simple search. | **4/5** | Independent proposals, selected-parent/no-explicit-score, compact scalar/case feedback; same surface and total resource accounting. Predeclare a worthwhile effect and replication plan. Raw full traces are an ablation, not the assumed winner. |

Keep the evidence, portable optimizer interface, control-plane contracts, validity checks, raw receipts and fixed baselines. Keep curriculum, archive memory, Pareto and extra depth as experimental options. Retain engineering fixes only with their relevant regression tests; this documentation audit does not certify a production merge or a features-only commit. A numerical or task-level win must survive the stronger comparison before becoming a recommended default.

<a id="future-work"></a>
## 7. High-potential research

Potential ratings are research judgments; **none of these gains is established by this series**.

| Rank | Direction | Potential / current evidence | Decisive experiment |
|---:|---|---|---|
| 1 | Learn reusable search policies across episodes, then adapt online. | **5/5 potential; exploratory simulator support** | Meta-TRAIN/VAL/TEST episodes on new task families; compare fixed informed policy, independent generation, parent mutation and learned policy at matched quality/cost. Include amortized meta-training cost. |
| 2 | Learn better credit assignment and retention. | **5/5 potential; concrete local failure mechanisms** | Separate solution quality from policy contribution and search stage. Test calibrated common-state or paired credit, with population interference measured, against stock scoring on live unsaturated tasks. |
| 3 | Select useful failure memory and evidence. | **4/5 potential; no demonstrated incremental gain yet** | Source/hash/status-aligned memories of successes and failures; compact versus raw traces, equal context/cost, reserved families. Show the information is consumed and improves held-out candidate yield. |
| 4 | Adaptive allocation, stopping and trigger rules. | **4/5 potential; mostly simulation and engineering evidence** | Vary horizon and policy-update schedule independently; compare against spending the same resources on lower-level search. Require a better anytime quality/cost curve, not merely more policy changes. |
| 5 | Additional recursive depth: learn how to learn policies. | **High conceptual potential, lowest near-term priority; efficacy unestablished** | Only after a useful O1 signal: O2 versus wider/longer O1 under equal total descendant budgets and fresh meta-TEST episodes. Nested execution alone is not success. |

For reusable policies, amortization requires a positive deployment saving at **matched quality**: `break-even deployments = meta-training cost / per-deployment saving`, in the same cost unit. If quality improves at fixed cost, report that tradeoff directly. EXP08's 2.25 is specific to its finite menu and uninformed comparator, not a program-wide estimate.

<a id="audit"></a>
## 8. Verification and evidence links

This review read the canonical reports through EXP24, checked their linked evidence and recomputed selected decisive quantities: EXP03/04 pairs; EXP13 null outcomes; EXP15/16/18 arm means and contrast arithmetic; EXP19 S4; EXP20 prefix/final means; saved EXP17/21/22-QA status; EXP23 terminal reports; EXP24 per-case optimum aggregation and first-hit coordinates. Input hashes and the dated live-campaign snapshot are in [verification.json][verification]. Reported confidence intervals remain the original studies' estimates; they were not all rebootstrapped. The full historical trajectories and every old claim were not regenerated.

Five existing EXP24 offline tests passed in **20.529 s**, including exploit re-scoring, fallback/validity checks, three optimum spot recomputations and three mock runner paths. The initial program and the two selected exploits were also re-scored with the white-box evaluator. A separate full Trace archive re-scoring attempt **timed out at 180 seconds**; the 26.233 archive maximum remains attributed to EXP24's corrected report, not independently regenerated here. These checks make no provider calls; they do not constitute a full regression of the current Trace working tree. No runtime or frozen result file was edited.

Reproduce from the repository root:

```sh
python experiments/recursive_opt/_analysis/assessment_20260930/verify.py
python -m ruff check experiments/recursive_opt/_analysis/assessment_20260930/verify.py
python -m black --target-version py313 --check experiments/recursive_opt/_analysis/assessment_20260930/verify.py
(cd experiments/recursive_opt/EXP24 && ../EXP22/evox/.venv/bin/python -I -m unittest discover -s tests -p test_exp24.py -v)
timeout 180 experiments/recursive_opt/EXP22/evox/.venv/bin/python -I experiments/recursive_opt/EXP24/scripts/prism_validity.py experiments/recursive_opt/EXP23/results/prism100_v2/20260929T215903/trace
```

The verifier prints a fresh checkpoint without changing raw evidence. Live-run counts can advance after this document's timestamp. The EXP24 evaluator depends on the local SkyDiscover checkout, so this is local reproducibility, not a claim that these files alone form a portable environment.

All experiment and evidence references below use canonical locations inside `experiments/recursive_opt`; production-source and test links in individual reports legitimately point to repository code. Old absolute `artifacts/` and second-worktree experiment paths are not current navigation. [Storage map](STORAGE_MAP.md), [earlier assessment](_history/reviews/recursive_opt_assessment.history_20260929.md), [earlier UC evidence index](_history/use_cases/README.md).

[verification]: _analysis/assessment_20260930/verification.json
[exp15]: EXP15/RESULTS.md
[exp15-json]: EXP15/results/exp15_results.json
[exp16]: EXP16/RESULTS.md
[exp16-json]: EXP16/results/production_run/analysis_results.json.gz
[exp18]: EXP18/RESULTS.md
[exp18-json]: EXP18/results/run/analysis_results.json.gz
[exp19]: EXP19/RESULTS.md
[s4-results]: EXP19/results/s4_results.json
[exp20]: EXP20/RESULTS.md
[exp20-json]: EXP20/results/run_store/full/confirmation/results.json
[exp23]: EXP23/RESULTS.md
[exp24]: EXP24/RESULTS.md
