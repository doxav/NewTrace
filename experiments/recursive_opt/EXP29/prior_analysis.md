# EXP29 — prior analysis (v2): what recursive_opt should discover, on which surfaces, tasks and budgets

Date: 2026-10-08. Supersedes v1 of the same day (commit `632b5e0426`). No new runs.

Inputs:
- the four EXP00 notebooks in `EXP00/notebooks/`, re-read cell by cell, including unexecuted cells;
- `EXP00/later_equivalent.md` and `EXP00/PROTOCOL.md`;
- EXP01–EXP28 reports and `ASSESSMENT.md`;
- the control plane v2 code (`opto/features/recursive_opt/spec.py`) and its registries.

## 0. What v1 got wrong

| v1 position | Correction |
|---|---|
| P1 compared `VariationSearch`, `PrioritySearch` and the **coevolution engine** | Coevolution is one engine of recursive_opt, specialised in EvoX-style online selection-policy evolution (§1). It is the narrowest meta surface in the package. EXP29 should be about the general control-plane recursion. |
| P4/P5 proposed that *we* build a helper/lessons library and tools | The purpose of recursive_opt is that an upper level **discovers** such mechanisms from a declared experiment (control-plane spec + task family). We should only provide the surface, the evaluator and the holdout. |
| Mostly read EXP00 through its RESULTS summary | Re-reading the notebooks changes the picture (§2):<br>• the memory, batch-design and credit-horizon knobs were **never active**, so they were never tested;<br>• trace type was only tested on single-call prompt tasks, where it cannot matter;<br>• the under-iteration A/B (use-case notebook, cell 71) was written but **never executed**. |
| Treated EXP21 and EXP22-QA as minor | They are the only studies that are both (a) built on the control plane with a real lower-level learner and (b) on a task with **demonstrated headroom**. Their O1/O2 stages were implemented but stopped by credit, not by a negative result. |

---

## 1. recursive_opt vs coevolution vs VariationSearch

| Term | What it is | Level |
|---|---|---|
| **recursive_opt** | The package plus the control plane v2 (`run_spec` / `compile_plan` / `execute_plan`). | All levels |
| **Episodic recursion (O1, O2, …)** | An upper level whose candidate is scored by running the lower level. EXP15–18, EXP19 S3, EXP21 `o1_qa.meta O1/O2`, EXP22-QA. | O1 over O0 |
| **Coevolution** (`recursive_opt/coevolution`, engine `coevolution`) | The **online** special case: during **one** O0 run, O1 rewrites a population-selection policy (EvoX reproduction). EXP22-EvoX to EXP28. | O1 inside one O0 run |
| **VariationSearch** (`opto/trainer/algorithms`) | An ordinary O0 trainer (EXP28). | A **target** of recursion, not recursion |

**Episodic recursion in detail.**
- A spec declares a level graph. Each level is a `trace.Module` with:
  - a module ref and a surface;
  - an engine (`fixed`, `trace`, `gepa_optimize_anything`, `coevolution`);
  - an objective, with evaluator, metrics, selection and `trace_config`;
  - datasets with holdout gating, LLM roles, knowledge, budget, and experiment arms/seeds.
- Recursion means an upper level's parameters define the lower level, and its evaluator runs the lower level.
- O2 learns the O1 output per task family. O3 learns transferable priors.

**Coevolution's scope.** It reaches only the parent/context/label selection policy, and only within one run. That gives about 10 noisy meta evaluations per run (v1 §5, still valid). v1 analysed the series through it because EXP22–28 used it.

### What the control plane can target today

| Target | Control-plane field | Status |
|---|---|---|
| Trainer choice and its kwargs (e.g. `VariationSearch` schedule, patience, `inspiration_mode`) | `engine.config.trainer`, `trainer_kwargs` | Exists. Categorical/numeric only |
| Optimizer choice, kwargs, objective text | `engine.config.optimizer`, `optimizer_kwargs`, `objective.intent` | Exists |
| Tracing per level | `objective.trace_config` = `{mode: internal\|otel\|sysmon\|hybrid, detail, credit_horizon, max_nodes, max_chars, semantic_names}` (EXP19) | Exists, tested |
| Feedback channels | `objective.feedback_channels` | Exists |
| Curriculum / batch | `trainer_kwargs.curriculum`, `batch_size` (EXP19–21) | Exists |
| Knowledge / memory | `knowledge` = store, retrieval, statuses, scope, top_k, injection codec | Exists. **`promotion_rule` and `rollback_rule` are rejected** unless placed under `extensions` |
| Arbitrary code components (an optimizer program, an evidence selector, a summariser) | `recursive_opt.module.reasoning_workflow@1`, `components: {name: source}` | Exists (EXP15 `optimizer_spec`, EXP22-QA `selector_source`) |
| A portable O1 level whose evaluator runs a child spec | `recursive_opt.module.recursive_level@1` | **Limited to `config`, `family_policy`, `prior`** |
| Code hooks consumed by a trainer or optimizer (a schedule function, a parent-score function, a memory policy) | — | **Missing:** no registry ref passes a validated source into `trainer_kwargs` |

**The structural gap.** Every real O1 study wrote its own nested evaluator:
- `o1_qa/meta.py` (EXP21);
- `_shared/optimizer_discovery/exp15.py` (EXP15);
- `EXP23/src/control_plane.py`.

The migration report found 0 of 85 specs replayable. So recursive_opt today mainly **declares** O0. Recursion is still bespoke code per study. This, not exploration, is the first engineering blocker for "discover via the control plane".

---

## 2. EXP00 re-read: what the original programme set out to discover, and why it could not

The A-list in `LevelConfig` (A.1–A.7) and the B/C/D surfaces are exactly the targets the user names. Each row gives the reason found in the notebooks and the closest later study.

| Original target (notebook) | What actually happened | Why no discovery was possible | Later equivalent and its state |
|---|---|---|---|
| A.1 starting artifact, initial knowledge (demo A, Phase 3, UC2) | ties or reversal (warm +0.013 then −0.026) | saturated or flat tasks; resolution ≈0.24 | EXP20/21: real gain from O0 learning (27 → 54 %), not from O1 |
| A.2 batch size / design (demo A, UC2) | inactive at `inner_steps=0`; **`batch_design` inactive in both modes** | the knob was not wired to the trainer | EXP19–21 curriculum: wired; `batch4` −7.1, `curriculum2` −4.8 pp (2 seeds) |
| A.3 trace type and horizon (Phase 2, UC6) | internal/otel/hybrid tie | O0 was **one prompt call**: OTEL/SysMon see nothing the Trace graph lacks. `credit_horizon` inactive | EXP19 made capture real; EXP21: sysmon +0.6, otel −0.9 pp (2 seeds), still on a prompt+ranker task |
| A.4 memory policy (demo A) | **inactive in both modes** | not wired | EXP18 archive memory (numeric optimisers): CI crosses 0; EXP21 `optimizer_memory3` incomplete |
| A.5 optimizer + tools (Phase 4, UC5, UC9, T3.6) | tools −0.014; **0 tool calls executed** | tool "policies" were text, never executed | never re-tested |
| A.6 guide | not tested | — | EXP24: evaluator feedback was the largest effect in the series |
| A.7 trainer, threads (Phase 1, 5) | ties on a saturating validator; inconsistent records | headroom: the toy validator reaches 1.0 | EXP24, EXP28, now with fixed-policy controls |
| B code of a component (demo B, UC1) | 0.80 → 1.00 | 12-item toy, hard items named in the feedback | EXP15–18 (optimizer programs): better than seed, mechanisms unresolved |
| C capability under cost (demo C, UC3) | accuracy 1.0 everywhere | saturated | EXP00-E: no criterion met |
| D family policy / prior O2–O3 (UC4) | +0.163 | arithmetic identity of two task sets (corrected −0.006) | EXP02, EXP08 (fixed `nearest` = learned) |
| UC7 sub-optimizer routing | initial = final = 1.0 | the initial route was already optimal | EXP07–09 |
| UC8 campaign policy, UC10 promotion policy | standard wins / not LCB-safe | policies scored on hand-labelled cases, not on campaign outcomes | `decisions.py`, never re-tested |
| UC11 code-emitted prompt | standard wins (−0.118) | QASPER noise | EXP03/12/13 |
| Three-way benchmark, Stage 2 | recursive arm under-iterated at equal total budget | user's hypothesis, **never tested** (cell 71 has no output) | none |
| Phase 7 Terminal-Bench 2 | template only | — | none |

**What the re-read changes.**
1. The memory, batch-design and credit-horizon targets were **untested, not refuted**.
2. Trace type was tested only where it is irrelevant by construction. It can only matter when O0 is a multi-step program with calls the Trace graph does not capture (library code, LangGraph nodes, tools).
3. Agentic optimizer tools were never executed.
4. "Too few iterations for the recursive arm" was formulated but never measured.

So most of the original design space is **open**, not closed.

---

## 3. The user's hypotheses, challenged

| Hypothesis | Verdict | Evidence and nuance |
|---|---|---|
| Too few iterations / steps | **True at O1, ambiguous at O0** | O0: PRISM is solved in 3 calls and Signal keeps improving past 50–100 calls. EXP20/21 stopped QA runs at 6 optimizer responses, so whether QA would keep improving is unmeasured. O1:<br>• EXP21 planned only **3 O1 proposals and 2 O2 proposals** (`o1_qa/axes.py`, `meta`) and ran none;<br>• coevolution gets about 10 per run;<br>• EXP00 cell 71 was never run.<br>The child-run length must come from the standard arm's iterations-to-peak, measured per task. |
| Tasks too basic | **True for most of the series, false for the QA task** | Saturated: GSM8K 0.99, DROP 1.0, demo C, UC1, PRISM. Exploitable: Signal, PRISM.<br>Not basic: the EXP20/21 HotpotQA task (unchanged 27 % → standard 54 %). EXP22-QA showed hand-written selector + instruction changes worth +17 pp (8 → 12/24), with an oracle-document ceiling of 58 %. |
| Surfaces and ranges badly designed | **True, in three distinct ways** | (a) **inactive** knobs: batch design, memory, horizon in EXP00;<br>(b) **irrelevant** pairing of surface and task: trace type on single-call prompts;<br>(c) **coarse menus** whose default is already near the best: EXP08, EXP10, EXP21 axes. EXP21 shows axes with large *negative* effects (prompt-only −19.6, minimal goal −10.7 pp), so the surface has variance. A menu can still teach O1 to avoid bad settings, but not to exceed the default. |
| (implicit) Exploration was the main limit | **Only for EXP23–27** | v1 §1–2 still hold: EXP28 fixed the operator; no legitimate-score gain followed. |

**A fourth constraint the user did not name:** there is no portable child-spec evaluator (§1). Each attempt re-implemented recursion, so each was small, bespoke and not replayable.

---

## 4. Meta targets: what optimizing a trainer, optimizer, trace or memory could deeply change

A target is worth an O1 level only if **all three** checks pass:

| Check | Meaning |
|---|---|
| (i) Certificate | A **hand-written or known variant** of the target already changes O0 performance on the task family. |
| (ii) Encoding | The target can be stated as a validated code or text artifact, not only a menu pick. |
| (iii) Budget | The O1 evaluation fits the budget at the required number of O1 steps. |

Per-target lists:

**T1. Optimizer evidence and update rule** (`OptoPrimeV2` instruction and objective sections, evidence selection, problem rendering)
- Surface: text + small code.
- Certificate: **yes**. EXP22-QA: selector code +8 pp, instruction +8 pp, both +17 pp (n = 24, single pilot). EXP21 goal axis: −10.7 pp. EXP28: wording takes productive diverge from 7 % to 38–44 %.
- What it could change: what the LLM sees and is asked to do. Strongest lever measured.
- Control-plane status: `reasoning_workflow@1` components plus optimizer kwargs (EXP22-QA wiring).

**T2. Trainer search policy** (`VariationSearch`/`PrioritySearch` hooks)
- Hooks:
  - when to diverge, refine or combine;
  - the instruction texts;
  - parent score (TRAIN vs common panel vs VAL);
  - which inspirations are shown.
- Surface: small code (`mode(state)`, `parent_score(candidate)`) + text.
- Certificate: **partial**. EXP19 S4: parent selection 72 → 100 % (toy). EXP28: diverge 3 % → 32 % productive calls, but no legitimate-score gain on Signal.
- What it could change: sample efficiency of every O0 run. A policy is cheap to transfer.
- Control-plane status: kwargs exist; a **code-hook ref is missing**.

**T3. A library method** (feedback summariser, batch sampler / curriculum rule, candidate deduplication)
- Surface: code.
- Certificate: toy only (UC1 batch design, trace summariser 0.82 → 0.96, hand-written).
- What it could change: input quality to T1 and T2.
- Control-plane status: `reasoning_workflow@1`; needs a hook ref.

**T4. Tracing per family** (`trace_config`)
- Surface: mixed categorical + numeric.
- Certificate: **none yet, and never tested on a fitting task**.
- What it could change: credit assignment on multi-step programs (agents, LangGraph, library-heavy code).
- Control-plane status: exists (EXP19).

**T5. Memory mechanism** (optimizer memory, archive of attempts, knowledge retrieval and injection)
- Surface: code (selection / retrieval rule) + numeric.
- Certificate: weak. EXP18 contrasts cross 0; EXP00 warm prior reversed; never tested as an active knob in EXP00.
- What it could change: reuse within and across runs.
- Control-plane status: knowledge block exists; promotion and rollback rules need an `extensions` namespace.

**T6. Capitalisation across runs** (O2: which T1–T5 artifact to promote for a family; when to stop or switch)
- Surface: policy over artifacts.
- Certificate: none valid (UC4 invalid; EXP08: the fixed default equals the learned one).
- What it could change: amortises O1 cost over many tasks. This is where recursion pays, if anywhere.
- Control-plane status: knowledge plus `recursive_level@1` `family_policy` / `prior`.

**T7. Online selection policy** (coevolution)
- Surface: code.
- Certificate: none on legitimate scores (EXP22–28).
- What it could change: within-run adaptation, starved at about 10 evaluations.
- Control-plane status: engine `coevolution`.

**Ranking by the three checks:** T1 > T2 > T4 (after a certificate) > T5 > T3 > T6 (needs one O1 success first) > T7.

Categorical menus of existing components (the EXP00 A-list as enums) are kept only as a **baseline** O1 surface, not as the target.

---

## 5. Tasks

A task family qualifies for EXP29 if:
- it has ≥ 6 related instances with TRAIN / VAL / HOLDOUT;
- the score is legitimate and audited against known exploits;
- a hand-written variant of the target changes the score (the certificate);
- O0 evaluation is cheap enough for ≥ 20 O1 evaluations × 3 seeds.

| Family | Status | Best for |
|---|---|---|
| **HotpotQA-style document QA**, fixed 4 documents (EXP20–22-QA) | certified for T1 (+17 pp hand-written, n = 24 pilot); O0 headroom 27 → 54 %; reader cost dominated by Qwen calls. The fixed document count removes EXP20's confound | T1, T2, T5 |
| **Numeric black-box optimiser programs** (EXP15–18: 6 families × dimensions, 24 TRAIN / 12 VAL instances, local evaluation) | O0 headroom vs seed shown (EXP15); headroom vs a strong reference (CMA-ES, SciPy) **not measured**. Cheapest evaluation in the series | T2, T3, T5 |
| **Multi-step agent / graph programs** (e.g. LangGraph PAL on BBEH, `examples/OpenTrace_LangGraph_…_curriculum_clean.ipynb`, untracked; Trace-Bench graph tasks) | never used for meta; needed for T4. Certificate to obtain | T4, T3 |
| **Signal with causality enforced in the evaluator** | legitimate headroom unknown (all configurations 0.50–0.55 causal). Needs a causal hand-written reference first | T2 only after a certificate |
| PRISM all-case | saturated (optimum in 3–4 calls) | regression check only |
| GSM8K, DROP, toy validators, `multi_param` | saturated or flat | excluded |
| `llm4ad` families, Terminal-Bench 2 | `llm4ad` gave −1e6 sentinels in EXP00; TB2 never onboarded | later |

---

## 6. Budget and the shape of recursion

O1 evaluation cost:

> cost = (instances per evaluation) × (child seeds) × (child-run length) × (cost per O0 call)

**Plain episodic O1 is affordable at roughly 20 O1 evaluations if child runs are short.** Example on QA:
- each O1 evaluation = 2 child seeds × the 6-response child run EXP21 used (`CALLS = 6`, `DEV_SEEDS`), so 12 optimizer calls plus reader calls;
- × 20 evaluations ≈ 240 optimizer calls per O1 seed, against the 3 O1 proposals EXP21 had planned (36 calls);
- that is about 7× EXP21's O1 plan, within the order of EXP21's whole development budget (235 DeepSeek optimizer responses, $6.8 in total).

**Replay as a cheap proxy for text and code targets (T1, T2).**
- Score a candidate instruction or selector on **recorded** prompts from earlier runs.
- EXP27 replayed about 1,000 calls for $1.46, roughly 50× cheaper per O1 sample than full child runs.
- Use: rank many candidates by replay, then confirm the top few with full child runs.
- Proxy validity must be checked: rank correlation between replay and full runs on ≥ 5 candidates.

**Is recursion achievable and useful? Revised verdict.**

| Form | Verdict |
|---|---|
| **O1 episodic** over a task family, with a portable child-spec evaluator, replay pre-screening and holdout instances | **Achievable now.** It is the main EXP29 line. |
| **O2** (per-family choice among O1 artifacts; promotion rule) | Useful only after one O1 artifact beats its default on holdout. Before that, O2 has nothing to select. |
| **Online coevolution** | Kept as an existing engine. Not a development target. |

**"Discover, don't build" is compatible with this budget**, provided the discovered object is a **small artifact** (a hook of 10–60 lines or a paragraph of instruction). A whole trainer rewritten from scratch would need far more than 20 evaluations to be distinguished from noise.

---

## 7. Keep / re-design / drop: the `later_equivalent.md` items

| EXP00 element (later equivalent) | Decision for EXP29 |
|---|---|
| Trainer choice / standard vs recursive (EXP22, EXP24, EXP28) | **Re-design as T2:** discover `VariationSearch` / `PrioritySearch` hooks, not pick a trainer name |
| Component code rewriting (EXP15–18, EXP28) | **Keep as T3** on the numeric family; add the CMA-ES/SciPy reference |
| Trace type (EXP16, EXP19, EXP21) | **Re-design as T4** on a multi-step family, after a certificate |
| Priors and skills, memory (EXP18–20) | **Re-design as T5**, with the memory / retrieval rule as code; promotion via `extensions` |
| Family policy / prior transfer (EXP02, EXP07–08, EXP22-QA) | **Defer to O2 (T6)** until an O1 success |
| Declarative spec (control plane v2) | **Extend:** portable child-spec evaluator + code-hook refs (§8) |
| QASPER prompt / config (EXP03, 12, 13) | drop (noise) |
| Threads (EXP05) | drop as a target; fix concurrency in the instrument |
| Routing / code transfer (EXP07–09) | drop until T6 |
| UC8 / UC10 policies (`decisions.py`) | fold into T6: score them on **campaign outcomes**, not hand labels |
| Optimizer tools / agentic, Terminal-Bench 2 (never re-tested) | later; needs executed tool calls first |
| Under-iteration A/B (cell 71, never run) | **Run in P0** as the child-run-length calibration |

---

## 8. Programme, in order

### P0 — make recursion declarable and certify targets

Engineering:
- A portable `recursive_opt.evaluator.child_spec@1`. It:
  - takes a child-spec template and a **binding path** (where the O1 artifact goes, e.g. `levels[0].engine.config.trainer_kwargs.<hook>` or `levels[0].objective.trace_config`);
  - takes TRAIN instances and child seeds;
  - returns the mean child validation score.
  - The child holdout stays closed.
- A versioned **code-hook ref**: validated source in, callable injected by the runner. Generic for trainer, optimizer and memory hooks; no callable in the spec.
- `extensions.recursive_opt.knowledge_rules` for promotion and rollback.
- Port `o1_qa.meta` (EXP21) and the EXP15 nested evaluator onto it, as proof of generality. This also gives the first replayable recursive specs.

Measurement:
- Child-run length from the standard arm's iterations-to-peak on QA and numeric.
- Certificates: hand-written variants per target × family. QA/T1 is already done (EXP22-QA). Still needed:
  - T2 on numeric and QA;
  - T4 on a multi-step family;
  - T5 on QA.

Kill rule: a target without a certificate does not enter P1–P3.

### P1 — O1 discovers the optimizer evidence and update rule (T1) on QA

Completes EXP21/EXP22-QA through the control plane.

Arms, at equal budget:

| Arm | Content |
|---|---|
| (a) | default `OptoPrimeV2` |
| (b) | hand-written certificate |
| (c) | O1 over EXP21's categorical axes (baseline surface) |
| (d) | O1 Trace engine over selector code + update instruction (EXP22-QA's PC arm) |
| (e) | (d) with replay pre-screening |

Success: (d) or (e) beats (a) on **holdout** questions by more than the paired seed noise, and is ≥ (b).

### P2 — O1 discovers the trainer search policy (T2)

- Surface: the `VariationSearch` / `PrioritySearch` hook sources: mode schedule, instruction texts, parent score.
- Families: numeric (cheap) and QA.
- Arms: plain `PrioritySearch`, default `VariationSearch`, O1-discovered, plus coevolution as an online reference (numeric only).
- Success: holdout instances in **both** families improve, i.e. the discovered policy transfers across families.

### P3 — tracing per family (T4), only after its certificate passes

- Family: multi-step graph/agent programs.
- O1 surface: the `trace_config` fields plus a summariser hook.
- Question: does the best tracing differ by family, and does O1 find it?

### P4 — memory and capitalisation (T5 → T6, O2)

- Discover the retrieval/injection rule and the promotion rule across a **sequence** of tasks in a family.
- Score on later held-out tasks: cold vs warm-default vs discovered.
- O2 starts only if P1 or P2 produced a holdout-positive artifact.

**Not in EXP29:**
- coevolution development;
- PRISM as a discriminator;
- categorical-menu O1 as a target;
- optimizer tools (until tool calls actually execute);
- Terminal-Bench 2.

---

## 9. Decisions and pre-commitments

1. **Recursion is declared, not scripted.** Every O1/O2 arm runs from a control-plane spec through the shared child-spec evaluator. Bespoke nested runners are not accepted as evidence.
2. **Certificate before discovery.** No O1 run on a target/family pair without a hand-written variant that moves O0.
3. **Holdout or nothing.** O1 selection sees TRAIN/VAL only. Claims use holdout instances; O2 claims use held-out family members.
4. **Equal budget, counted in O0 calls**, including the O1 evaluations, against a standard arm given the same total.
5. **Fixed-default and hand-written arms are mandatory.** A discovered artifact must beat the default and not lose to the hand-written one.
6. **Exploit audit** on every new task before use (lesson of Signal and PRISM).
7. **Child-run length from iterations-to-peak**, not from a constant (EXP00 cell 71).
8. **Small artifacts:** discovered objects are hooks or texts with a validated contract. Whole-class rewrites only after a hook-level success.

## 10. Conclusion

- The main limit of recursive_opt was not exploration. It was the combination of:
  - targets that were inactive, irrelevant or menu-shaped;
  - tasks that were saturated or exploitable;
  - O1 stages that were bespoke and, on the one good task (QA), never executed.
- The control plane already declares most of the right targets: trainer kwargs, `trace_config`, knowledge, code components.
- It lacks two generic pieces: a **child-spec evaluator** and **code-hook refs**.
- With those, EXP29 should first let O1 discover:
  1. the optimizer evidence/update rule on QA, where headroom is certified (+17 pp hand-written);
  2. then the trainer search policy across QA and numeric families;
  3. then tracing on multi-step programs and the memory/capitalisation rules, each after its own certificate.
