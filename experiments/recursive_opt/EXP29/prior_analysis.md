# EXP29 — prior analysis (v3): what recursive_opt should discover, on which surfaces, tasks and budgets

Date: 2026-10-08.
- v3 refines v2 (`ecb415041c`), which superseded v1 (`632b5e0426`).
- v3 adds the task order (§5), the tracing clarification (§5.3) and the design of the two missing pieces ([design_recursion_pieces.md](design_recursion_pieces.md)).
- v3 corrects one v2 claim (§0).
- New measurement: one local, LLM-free probe of Signal's causal headroom (§5.1).

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
| Treated EXP21 and EXP22-QA as minor | They are the only studies that are both (a) built on the control plane with a real lower-level learner and (b) on a task with **demonstrated O0 headroom**. Their O1/O2 stages were implemented but stopped by credit, not by a negative result. |
| **(v2 → v3)** "EXP22-QA certifies T1: hand-written selector + instruction +17 pp" | **Wrong attribution.** The +17 pp pilot (8 → 12 of 24) changed the **O0 program**: the ranker code and the *reader* instruction. It certifies that the QA task has O0 headroom reachable by admissible edits.<br>The O1 targets (the optimizer's `update_instruction` and `selector_source`) have a mechanism argument (the optimizer must see the bridge-document failures to fix the ranker), but **no hand-written O1 certificate yet**.<br>The numeric task also has demonstrated O0 headroom (seed → hand-written B2 → LLM programs, §5), so QA is not "the only one". |

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
| Tasks too basic | **True for most of the series, false for the QA task** | Saturated: GSM8K 0.99, DROP 1.0, demo C, UC1, PRISM. Exploitable: Signal, PRISM.<br>Not basic:<br>• the EXP20/21 HotpotQA task: unchanged 27 % → standard 54 %. In EXP22-QA, hand-written O0 ranker + reader-instruction edits gave +17 pp (8 → 12 of 24), with a ceiling of 58 % when the supporting documents are given;<br>• the EXP15–18 numeric optimizer-program family: regret AUC seed 0.144 → hand-written 0.041 → LLM-written 0.034 (EXP18). |
| Surfaces and ranges badly designed | **True, in three distinct ways** | (a) **inactive** knobs: batch design, memory, horizon in EXP00;<br>(b) **irrelevant** pairing of surface and task: trace type on single-call prompts;<br>(c) **coarse menus** whose default is already near the best: EXP08, EXP10, EXP21 axes. EXP21 shows axes with large *negative* effects (prompt-only −19.6, minimal goal −10.7 pp), so the surface has variance. A menu can still teach O1 to avoid bad settings, but not to exceed the default. |
| (implicit) Exploration was the main limit | **Only for EXP23–27** | v1 §1–2 still hold: EXP28 fixed the operator; no legitimate-score gain followed. |

**A fourth constraint the user did not name:** there is no portable way to run a child spec as an upper level's evaluation (§1). Each attempt re-implemented recursion, so each was small, bespoke and not replayable.

---

## 4. Meta targets: what optimizing a trainer, optimizer, trace or memory could deeply change

A target is worth an O1 level only if **all three** checks pass:

| Check | Meaning |
|---|---|
| (i) Certificate | A **hand-written or known variant** of the target already changes O0 performance on the task family. |
| (ii) Encoding | The target can be stated as a validated code or text artifact, not only a menu pick. |
| (iii) Budget | The O1 evaluation fits the budget at the required number of O1 steps. |

Per-target lists:

**T1. Optimizer evidence and update rule** (`OptoPrimeV2` objective/instruction, `problem_instance` rendering, evidence selection)
- Surface: text + small code.
- Certificate: **indirect only**.
  - EXP21 goal axis: a minimal optimizer goal costs −10.7 pp.
  - EXP28: instruction wording takes productive diverge from 7 % to 38–44 %.
  - EXP22-QA's +17 pp is an **O0** certificate, not a T1 one (§0).
- What it could change: what the LLM sees and is asked to do.
- Control-plane status: `optimizer_kwargs.objective` exists (text). Rendering and selection need the `OptoPrimeV2` hook (design §2).

**T2. Trainer search policy** (`VariationSearch`/`PrioritySearch` hooks)
- Hooks:
  - when to diverge, refine or combine;
  - the instruction texts;
  - exploration and exploitation priority (parent score);
  - which inspirations are shown.
- Surface: small code + text.
- Certificate: **partial**. EXP19 S4: parent selection 72 → 100 % (toy). EXP28: diverge 3 % → 32 % productive calls, with no legitimate-score gain on Signal.
- What it could change: sample efficiency of every O0 run. A policy is cheap to transfer.
- Control-plane status: kwargs exist; code hooks missing (design §2).

**T3. A library method** (feedback summariser, batch sampler / curriculum rule, candidate filtering)
- Surface: code.
- Certificate: toy only (UC1 batch design, trace summariser 0.82 → 0.96, hand-written).
- What it could change: input quality to T1 and T2.
- Control-plane status: via hooks (`filter_candidates`, `problem_instance`).

**T4. Tracing per family** (`trace_config`)
- Surface: mixed categorical + numeric, plus a summariser hook.
- Certificate: **none yet, and never tested where it can matter** (§5.3).
- What it could change: which execution facts reach the optimizer.
- Control-plane status: exists (EXP19), but it captures the evaluator *process*, not code run in a worker (§5.3).

**T5. Memory mechanism** (optimizer memory, archive of attempts, knowledge retrieval and injection)
- Surface: code (selection / retrieval rule) + numeric.
- Certificate: weak. EXP18 contrasts cross 0; EXP00 warm prior reversed; never tested as an active knob in EXP00.
- What it could change: reuse within and across runs.
- Control-plane status: knowledge block exists; promotion and rollback rules need `extensions` (design §2).

**T6. Capitalisation across runs** (O2: which T1–T5 artifact to promote for a family; when to stop or switch)
- Surface: policy over artifacts.
- Certificate: none valid (UC4 invalid; EXP08: the fixed default equals the learned one).
- What it could change: amortises O1 cost over many tasks. This is where recursion pays, if anywhere.
- Control-plane status: knowledge, then `child_spec@1` applied twice (O2).

**T7. Online selection policy** (coevolution)
- Surface: code.
- Certificate: none on legitimate scores (EXP22–28).
- What it could change: within-run adaptation, starved at about 10 evaluations.
- Control-plane status: engine `coevolution`.

**Ranking by the three checks:** T2 ≥ T1 > T5 > T3 > T4 (after a certificate) > T6 (needs one O1 success first) > T7.
- T2 moves ahead of T1 because its certificates are direct, and the numeric task (§5) can test it cheaply.
- T1 needs its own hand-written certificate, now part of P0.

Categorical menus of existing components (the EXP00 A-list as enums) are kept only as a **baseline** O1 surface, not as the target.

---

## 5. Tasks

A task family qualifies for EXP29 if:
- it has ≥ 6 related instances with TRAIN / VAL / HOLDOUT;
- the score is legitimate and audited against known exploits;
- a hand-written variant changes the score (O0 headroom), and then a hand-written variant of the **O1 target** does too (the certificate);
- a child run is cheap enough for ≥ 20 O1 evaluations × 3 seeds.

### 5.1 Signal Processing: keep as a smoke test, not as the validation task

New probe ([script](scripts/signal_causal_headroom.py), [results](results/signal_causal_headroom.json); local, no LLM, 6 s). It evaluates 34 hand-written **causal** filters with the EXP25 white-box evaluator: EMA, Butterworth `lfilter`, alpha-beta tracker, endpoint polynomial + EMA, each over a small parameter grid. All are causal by the probe (causal fraction 1.0).

| Program | Score |
|---|---:|
| initial program (trailing weighted average) | 0.499 |
| 34 classical causal filters | **0.467 – 0.539** |
| best LLM-found causal program in the whole series (EXP26) | 0.565 |
| run-to-run spread of the best causal score, every EXP28 configuration | 0.499 – 0.554 |
| look-ahead cheat (`filtfilt`, EvoX EXP25) | 0.723 |

Consequences:
- **Narrow range.** The legitimate range above the start is about 0.04–0.07. The seed-to-seed spread of one configuration covers almost all of it, so an O1 effect cannot be resolved at 3 seeds.
- **No held-out instances.** The task is one fixed set of 5 signals.
- **Exploit-prone.** The score rewards a non-causal shortcut, so a causality guard is mandatory.

Keep Signal for engineering smoke tests (children run in seconds), for EvoX comparability and as an exploit regression check. It is not the first validation task.

### 5.2 Order: numeric first, then document QA

| | Numeric optimizer programs (EXP15–18) | Document QA, 4 documents (EXP20–22-QA) |
|---|---|---|
| O0 artifact | `propose(history, bounds, seed)` program, standard library only | ranker code + reader instruction |
| Evaluation | local, deterministic; the program never sees the objective | Qwen reader, ≈100 reader calls per O0 step |
| Cost per child run (measured) | EXP18: 16 proposals × $0.0024 ≈ **$0.04**; no LLM in evaluation | EXP20: $2.85 / 72 runs ≈ **$0.04**. Up to 828 reader calls per O0 chain; long latencies (responses up to 470 s) |
| O0 headroom | AUC 0.144 → 0.041 (hand-written B2) → 0.034 (LLM). Target hits 86 → 93 → 135 / 144 | 27 % → 54 % (EXP21); hand-written +17 pp at fixed 4 documents (EXP22-QA) |
| Remaining headroom | **AUC yes; final regret nearly saturated** (best arm 0.0026 vs target 0.01). Add harder families: multimodal Rastrigin/Ackley, 8-D, noisy | large (54 % vs 58 % with the supporting documents given; exact match on hard bridge questions) |
| Family / holdout | built in: 3 functions × 2 dimensions, 24 TRAIN / 12 VAL / fresh TEST instances, local seeds | episodes of new questions (EXP22-QA META-TRAIN / VALIDATION / TEST design) |
| Exploit risk | low: the host evaluates; the program sees only history and bounds | low at fixed 4 documents; answer leakage audited in EXP22-QA |
| Already on the control plane | `optimizer_program.py`, `optimizer_spec()` | EXP21 `o1_qa` modules |
| Fits | T2, T3, T5, T1 (update instruction) | T1, T2, T5; real LLM-in-the-loop transfer |

**Recommendation: numeric first, then QA if the numeric O1 shows a holdout gain.**
- The numeric family is the "complex optimization code task, easy at first stage but not saturated" the request asks for, once the harder families are added.
- It is the fastest way to debug `child_spec@1` and the hooks on real children. The cost per child is about the same as QA, but numeric children take minutes, not hours.
- QA is the realistic confirmation: a different modality, LLM-evaluated, with large headroom. Running it second avoids paying its wall time while the infrastructure is still being debugged.
- If numeric is null, run QA anyway on T1 before concluding. A null on numeric could come from the task ceiling, not the method.

**A multi-step agent task is not needed** for P1–P2. It is LLM-hungry: every O0 evaluation is several model calls per example. The tracing target T4 does not require one either (§5.3).

### 5.3 Clarification: when can the choice of tracing matter?

| Source | Captures | Present when |
|---|---|---|
| Trace graph (`internal`) | operations on Trace nodes: `@bundle` calls and trainable parameters | always; this is what `OptoPrimeV2` reads |
| OTEL | spans from instrumented code: bundles, LangGraph nodes, LLM / tool clients | only for instrumented libraries |
| SysMon (`sys.monitoring`, Python ≥ 3.12) | call/return events of **any** Python function in the observed thread | arbitrary unbundled code in-process |

Tracing can only help if O0's execution has internal structure that (a) drives the score, (b) is invisible in the Trace graph and (c) is not already in the evaluator's feedback.

**Why EXP00/EXP21 ties were expected:**
- In single-call prompt tasks the program is one LLM call, so all three sources see the same thing.
- On code tasks, today's capture misses the candidate entirely. `capture_evaluation` wraps the evaluator in the parent process (`spec.py`, `_evaluate_example`). The Signal white box and `optimizer_program` run the candidate in a **subprocess**, and SysMon sees only the observed thread (EXP19 limit).
- The evaluators already return rich per-case diagnostics (per-signal metrics, the optimization history). These compete with any trace projection.

**So T4 needs code-level capture inside the worker, not a multi-step agent.**
1. Run SysMon (or OTEL) inside the worker process.
2. Return a bounded call/return profile with the result.
3. Feed it through `trace_config`.

The numeric and Signal tasks then become valid T4 testbeds at no extra LLM cost. Multi-step agents (LangGraph, tools) matter only for the OTEL-span question, which is later.

**T4 certificate, cheap (replay):**
- Take recorded optimizer prompts.
- Add or remove the worker trace projection at an equal character budget.
- Measure the improvement rate of the proposals.
- Run O1 over `trace_config` only if this differs.

| Other families | Status |
|---|---|
| PRISM all-case | saturated (optimum in 3–4 calls): regression check only |
| GSM8K, DROP, toy validators, `multi_param` | saturated or flat: excluded |
| Multi-step agents (LangGraph PAL on BBEH, Trace-Bench graph tasks) | later, for OTEL-span tracing and agent-level T3 |
| `llm4ad` families, Terminal-Bench 2 | later (`llm4ad` gave −1e6 sentinels in EXP00; TB2 never onboarded) |

---

## 6. Budget and the shape of recursion

O1 evaluation cost:

> cost = (instances per evaluation) × (child seeds) × (child-run length) × (cost per O0 call)

**Plain episodic O1 is affordable at roughly 20 O1 evaluations if child runs are short.**

Numeric:
- 20 O1 evaluations × 3 training episodes × one 16-proposal child ≈ 960 proposals, about $2.3 per O1 seed at the EXP18 price;
- local evaluation only.

QA:
- each O1 evaluation = 2 child seeds × the 6-response child run EXP21 used (`CALLS = 6`, `DEV_SEEDS`), so 12 optimizer calls plus reader calls;
- × 20 evaluations ≈ 240 optimizer calls per O1 seed, against the 3 O1 proposals EXP21 had planned (36 calls);
- EXP22-QA's full plan (24 O1 proposals and confirmation) was 654 optimizer responses plus up to 86,940 reader responses. Its authors estimated about $30 of credit. Wall time, not money, is the constraint.

**Replay as a cheap proxy for text and code targets (T1, T2).**
- Score a candidate instruction or selector on **recorded** prompts from earlier runs.
- EXP27 replayed about 1,000 calls for $1.46, roughly 50× cheaper per O1 sample than full child runs.
- Use: rank many candidates by replay, then confirm the top few with full child runs.
- Proxy validity must be checked: rank correlation between replay and full runs on ≥ 5 candidates.

**Is recursion achievable and useful? Revised verdict.**

| Form | Verdict |
|---|---|
| **O1 episodic** over a task family, with `child_spec@1`, replay pre-screening and holdout instances | **Achievable now.** It is the main EXP29 line. |
| **O2** (per-family choice among O1 artifacts; promotion rule) | Useful only after one O1 artifact beats its default on holdout. Before that, O2 has nothing to select. |
| **Online coevolution** | Kept as an existing engine. Not a development target. |

**"Discover, don't build" is compatible with this budget**, provided the discovered object is a **small artifact** (a hook of 10–60 lines or a paragraph of instruction). A whole trainer rewritten from scratch would need far more than 20 evaluations to be distinguished from noise.

---

## 7. Keep / re-design / drop: the `later_equivalent.md` items

| EXP00 element (later equivalent) | Decision for EXP29 |
|---|---|
| Trainer choice / standard vs recursive (EXP22, EXP24, EXP28) | **Re-design as T2:** discover `VariationSearch` / `PrioritySearch` hooks, not pick a trainer name |
| Component code rewriting (EXP15–18, EXP28) | **Keep:** the numeric family becomes the first O0 task family; add harder functions and a strong hand-written reference |
| Trace type (EXP16, EXP19, EXP21) | **Re-design as T4:** capture inside the worker on code tasks (§5.3), after a replay certificate |
| Priors and skills, memory (EXP18–20) | **Re-design as T5**, with the memory / retrieval rule as code; promotion via `extensions` |
| Family policy / prior transfer (EXP02, EXP07–08, EXP22-QA) | **Defer to O2 (T6)** until an O1 success |
| Declarative spec (control plane v2) | **Extend:** `child_spec@1` module + declared code hooks ([design](design_recursion_pieces.md)) |
| QASPER prompt / config (EXP03, 12, 13) | drop (noise) |
| Threads (EXP05) | drop as a target; fix concurrency in the instrument |
| Routing / code transfer (EXP07–09) | drop until T6 |
| UC8 / UC10 policies (`decisions.py`) | fold into T6: score them on **campaign outcomes**, not hand labels |
| Optimizer tools / agentic, Terminal-Bench 2 (never re-tested) | later; needs executed tool calls first |
| Under-iteration A/B (cell 71, never run) | **Run in P0** as the child-run-length calibration |

---

## 8. Programme, in order

### P0 — make recursion declarable, calibrate and certify (mostly local, little LLM)

Engineering (see [design_recursion_pieces.md](design_recursion_pieces.md); decisions D1 and D2 first):
- `recursive_opt.module.child_spec@1`: slots of a child-spec template are the O1 parameters; `forward` runs the child.
- Declared hook points (`HOOKS`) + a generic materializer under `engine.config.hooks`.
- `VariationSearch(variation_instructions=...)` as a text surface.
- Proof of generality: port the EXP15/18 numeric study and EXP21 `o1_qa.meta` onto `child_spec@1`. These become the first replayable recursive specs.

Measurement:
- Numeric family hardening:
  - add multimodal (Rastrigin, Ackley), 8-D and noisy instances;
  - check the gaps seed < hand-written < reference with **no LLM**, as done for Signal in §5.1.
- Child-run length from the standard arm's iterations-to-peak (EXP00 cell 71).
- Certificates, each by a hand-written variant or a replay:
  - T2 on numeric: e.g. stagnation-diverge vs free vs a hand-written parent score;
  - T1 on numeric and QA: hand-written update instruction / evidence rendering;
  - T4 by trace replay (§5.3);
  - T5 on numeric: an archive of past programs shown vs not.

Kill rule: a target without a certificate does not enter P1–P3.

### P1 — O1 discovers the trainer search policy (T2) and update instruction (T1) on the numeric family

| Arm | Content |
|---|---|
| (a) | plain `PrioritySearch` |
| (b) | default `VariationSearch` |
| (c) | hand-written certificate variant |
| (d) | O1 (`child_spec@1` + Trace engine) over the hook / text slots |
| (e) | (d) with replay pre-screening |
| (f) | coevolution as an online reference |

Success: (d) or (e) beats (a) and (b) on **holdout episodes** (new instances, and one held-out function family) by more than the paired seed noise, and is ≥ (c), at equal total budget including O1.

### P2 — the same on document QA, if P1 is positive (or for T1 regardless)

- Completes EXP21 / EXP22-QA through `child_spec@1`, with EXP22-QA's META-TRAIN / VALIDATION / TEST episodes.
- Arms: (a), (b), (c), (d), plus EXP21's categorical axes as the menu baseline.
- Transfer test: apply the P1 numeric-discovered policy unchanged to QA.

### P3 — tracing per family (T4), only after its replay certificate passes

- Worker-side SysMon capture on numeric/Signal programs.
- O1 surface: `trace_config` fields plus a summariser hook.

### P4 — memory and capitalisation (T5 → T6, O2)

- Discover the retrieval/injection rule and the promotion rule across a **sequence** of episodes.
- Score on later held-out episodes: cold vs warm-default vs discovered.
- O2 (`child_spec@1` over an O1 spec) only if P1 or P2 produced a holdout-positive artifact.

**Not in EXP29:**
- coevolution development;
- Signal and PRISM as discriminators (smoke and regression only);
- categorical-menu O1 as a target;
- optimizer tools (until tool calls actually execute);
- Terminal-Bench 2.

---

## 9. Decisions and pre-commitments

1. **Recursion is declared, not scripted.** Every O1/O2 arm runs from a control-plane spec through the shared `child_spec@1` module. Bespoke nested runners are not accepted as evidence.
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
  - tasks that were saturated, exploitable or single-instance;
  - O1 stages that were bespoke and never executed on the tasks with real headroom.
- The control plane already declares most of the right targets: trainer kwargs, `trace_config`, knowledge, code components.
- It lacks two generic pieces: a **`child_spec@1` module** and **declared code hooks**. Both are small; two core-library decisions (D1, D2) come first.
- EXP29 should then let O1 discover:
  1. the trainer search policy and the update instruction on the numeric optimizer-program family (cheap, wide range, built-in holdout);
  2. then the same on document QA, the realistic confirmation;
  3. then tracing (worker-side capture) and the memory/capitalisation rules, each after its own certificate.
- Signal stays a smoke and exploit-regression task: its legitimate range (classical causal filters 0.47–0.54, best LLM 0.565) is narrower than its seed spread.
