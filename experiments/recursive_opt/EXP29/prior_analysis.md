# EXP29 — prior analysis: what to re-test, and where meta / recursive optimization can pay

Date: 2026-10-08. Inputs: EXP00 (with `EXP00/later_equivalent.md`) to EXP28, `ASSESSMENT.md`, and the EXP22–EXP28
retrospective (`_analysis/retrospective_20261006/`). No new runs. The goal is to challenge the working
assumptions before spending budget, then rank what EXP29 should test.

---

## 1. What EXP27–EXP28 actually established, and what they did not

The request starts from the premise that EXP27–EXP28 *solved* Trace's exploration limitation, and that this
limitation is why Trace could not beat EvoX. The evidence supports a narrower statement.

**Established:**

| Claim | Evidence |
|---|---|
| Trace's coevolution engine explored less than EvoX: its policy optimizer locked into REFINE and its DIVERGE calls were unproductive | EXP27: SciPy filters introduced by DIVERGE calls 6/8 for EvoX vs 8/189 for Trace; Trace's evolved policies used REFINE about 72% of the time |
| An explicit, per-step mutation intent in a trainer makes DIVERGE productive | EXP28 Part B, after the leak fix: SciPy introduced by 32% of DIVERGE calls vs 3% of free calls |
| The wording of the instruction is high-leverage | Explicit "combine" sentence: 7% (5/69). Same inspirations shown as plain context: 38–44% |
| `VariationSearch`'s default (DIVERGE after a stall) is the best Trace configuration measured | Signal median 0.758 (EvoX 0.711); PRISM all-case optimum at calls 3, 3 and 4 (EXP24: 12–22; stock EvoX 1 of 3 runs) |

**Not established:**

| Assumption | Why it is not established |
|---|---|
| "Exploration was why Trace could not beat EvoX" | On the legitimate metrics there was nothing to beat. EvoX's Signal lead was the look-ahead loophole (causal medians 0.537 vs 0.532), and its PRISM records were refusal exploits. Better exploration made Trace find the **loophole** faster; it did not raise the causal Signal score (0.499–0.554 for every configuration) |
| "`VariationSearch`'s schedule is the cause of the gain" | No plain-`PrioritySearch` arm was run. The trainer also differs from the coevolution engine in edit format (full-program rewrite vs SEARCH/REPLACE diff) and in having no elite-context prompt. The schedule's own effect is not isolated |
| "PRISM shows a real speed-up from exploration" | PRISM is solved by every configuration within about 10 calls, and EXP24's fixed policy also reached the optimum in 3/3 runs. The task has no headroom left to discriminate |
| "The exploration deficit also limited the earlier recursive / meta experiments" | Only EXP23–EXP27 show it. EXP00–EXP22 failed for other, better-documented reasons (§2) |

**Working conclusion.** EXP28 gives a better operator (DIVERGE that works, and evidence that instruction text matters).
It does not give evidence that meta-optimization works. Every legitimate-score comparison to date (causal Signal,
all-case PRISM, GSM8K) is still a tie.

---

## 2. Why the earlier meta / recursive experiments failed: the binding constraint per study

Each row names the constraint that, by the evidence, prevented a meta-level effect from showing. Most studies had
several; this is the first one that bites.

| Study | Meta surface | Binding constraint | Evidence |
|---|---|---|---|
| EXP00-A/B/C (notebooks) | categorical setup knobs; O2/O3 menus | **inactive or flat surface** | every setup scored the same (−2091.8); causal-effect contract: most knobs did not reach the score at `inner_steps=0` |
| EXP00-D (UC1–UC14) | config, code, policies | **instrument resolution** (≈0.24 at n = 5) and an **invalid comparison** | UC4 +0.163 was an arithmetic identity, corrected −0.006 (EXP02) |
| EXP00-E (Experiment 0) | two prompt instructions | **no headroom** | baseline accuracy 0.99 on GSM8K |
| EXP01–EXP14 (probes) | prompts, menus, knobs | **instrument and menus** | signal/noise 0.96 and 0.74 (EXP01); fixed menus with few distinct scores (EXP06, 07, 14); inactive knobs (EXP10) |
| EXP08 | routing order learned across tasks | **fixed menu**: the informed simple default matched it | fixed `nearest` matches the learned prior at no meta cost |
| EXP15–EXP18 | generated optimizer code; feedback, memory, Pareto | **effect below replication** (5–6 seeds) | feedback vs independent generation unresolved (EXP15); rich feedback worse (EXP16); memory and Pareto CIs cross 0 (EXP18) |
| EXP19 S3 / EXP20 / EXP21 | nested preparation, curriculum, development axes | **no resolved increment** and **unexecuted stages** | curriculum − standard −3.4 pp [−8.5, +2.4]; EXP21's O1/O2 stages never ran |
| EXP22-EvoX / EXP23 / EXP24 | selection-policy code (meta level) | **budget split** plus **saturated task** | fixed policy ≥ recursive (EXP22); fixed as fast as both meta arms with fewer wasted attempts (EXP24) |
| EXP25–EXP27 | coevolution policy code | **loophole** plus **exploration deficit** | look-ahead lead; REFINE-locked policies |
| EXP28 | mutation intent (trainer); EvoX brief / diverge guard (meta) | **loophole** and **no headroom on PRISM** | no causal gain; PRISM solved by every configuration |

Three constraints recur across the whole programme:
1. **Measurement:** noise, invalid comparisons, exploitable metrics.
2. **Headroom:** saturated tasks or flat surfaces.
3. **Too few meta-level evaluations per run:** §5.

Surface design (categorical knobs) explains the early studies. It does not explain EXP15–EXP18 or EXP22–EXP28, whose
meta surfaces were code and which still showed no meta gain.

---

## 3. The hypotheses in the request, challenged

| # | Hypothesis | Verdict | Reasoning |
|---|---|---|---|
| H1 | Too few iterations / steps | **Supported, but at the meta level, not the base level** | 100 base calls were enough to reach PRISM's optimum in 3 calls and Signal's loophole in about 10. The meta level is what starves: a policy is evaluated on about 10-call windows, giving 6–8 noisy evaluations per run (EXP27: 6–8 deployments; EXP23: few triggered proposals). More base iterations alone do not fix this |
| H2 | Tasks too basic | **Supported** | PRISM saturated; GSM8K at 0.99; Signal's legitimate ceiling unknown and nearly reached by the starting program (0.499 → ~0.55); toy validators in EXP00. No task in the series had certified, legitimate headroom *and* an un-exploitable metric |
| H3 | Surfaces badly designed (categorical values instead of deep code / strategy strings) | **Partly supported** | True for EXP00-A–D, EXP06–EXP11 and EXP13. But EXP15–EXP18 (optimizer code) and EXP22–EXP28 (policy code) had deep surfaces and still showed no meta gain. Deep surfaces are necessary, not sufficient |
| H4 | The exploration deficit limited meta-optimization | **Supported only for EXP23–EXP27** | There, the meta level itself was the cause: OptoPrime without a design brief wrote REFINE-locked policies. Elsewhere the binding constraint was measurement or headroom (§2) |
| H5 | `VariationSearch` fixes exploration | **Supported for the operator, not yet for search outcomes** | DIVERGE productivity 32% vs 3%. Legitimate scores unchanged; schedule effect not isolated |
| H6 | Recursive (O2+) optimization is achievable / useful | **Not within a single bounded run; plausible across episodes** | §5–§6 |

Two further assumptions are worth stating because they have quietly shaped the programme:

- **"Meta-optimization should improve the same run it is in."** Every meta level so far (EXP22–EXP28) adapted
  *inside* one 100-call run, competing with the base level for budget. The one positive-looking amortization result
  (EXP08, break-even 2.25 deployments) came from learning *across* tasks.
- **"A higher benchmark score is progress."** Five headlines were withdrawn for this (EXP22, EXP23, EXP25, EXP26 R3,
  EXP27 "root cause = cue"). Any EXP29 claim must be on an enforced, legitimate metric.

---

## 4. The goal, restated, and which axes of "optimizing a trainer / optimizer" have shown leverage

**Goal of the programme:** make Trace's optimization *process* better (faster, more reliable, more transferable) by
optimizing the process itself (O1), and ideally by learning how to do that (O2+), at a cost that pays back.

Axes of the optimization process, ranked by the effect sizes actually measured:

| Axis (what a meta level could change) | Largest measured effect | Where | Status |
|---|---|---|---|
| **Evaluator feedback and validity** (white-box per-case feedback, valid guide, repair projection) | PRISM optimum reached 1/7 → 9/9 runs | EXP24 | strong, at O0 design time; never meta-optimized |
| **Mutation intent and instruction wording** (DIVERGE/REFINE timing, inspiration framing) | SciPy introduced 3% → 32% of calls; framing 7% vs 38–44% | EXP28 | strong per call; outcome effect not isolated |
| **Parent selection and comparability** (which candidate to expand, scored on a common panel) | +27.8 pp TEST [22.9, 31.3] | EXP19 S4 | strong, on a small task |
| **Information budget given to the solver** (documents, context) | +14.6 pp, but confounded with 10 documents | EXP20 | real, attribution open |
| **Context / memory content** (lessons, archive, failures) | none resolved | EXP18, EXP00-B | unresolved |
| **Feedback richness** (more trace text) | negative: rich feedback worse | EXP16 | adverse |
| **Configuration knobs** (batch size, trainer name, trace type) | none | EXP00, EXP10, EXP11 | flat |
| **Selection-policy code optimized at run time** (EvoX-style meta level) | none vs fixed policy | EXP22, EXP24, EXP28 | no gain |

The strong effects are all **operator and evaluator design** (what the optimizer is asked to do and what evidence it
sees), not **knob selection** and not **run-time policy evolution**. That is the most important input for EXP29: a
meta level should search the high-leverage axes (instructions, operator logic, feedback/evidence design, parent
selection), not menus.

---

## 5. Budget arithmetic: why within-run meta-optimization is starved

For a run of `T` base calls, with the meta level re-evaluating its artifact on windows of `w` calls:
- Meta evaluations per run ≈ `T / w`. EvoX defaults give `w ≈ 0.1 T`, so about 10.
- Each meta evaluation is a best-score improvement over `w` calls, and that is mostly noise. Improvements are rare
  and lumpy: one discovery event per run on Signal, the optimum within 3–10 calls on PRISM.
- Credit is confounded with search stage: early windows improve easily and late ones rarely (EXP23 simulator: stage
  bias).

With T = 100 and w = 10, the meta optimizer gets about 10 noisy, stage-biased samples. That is fewer than any base-level
optimizer would be given to learn a code artifact. Raising T to 1,000 helps only if the task keeps headroom that long;
PRISM and Signal do not.

**Consequences:**
1. Within-run meta-optimization can at best help by *structural* priors (good defaults such as EvoX's brief or
   `VariationSearch`'s schedule). It cannot learn much from its own run. EXP22, EXP24 and EXP28 agree: fixed or
   brief-guided policies matched or beat learned ones.
2. Learning a process requires **many cheap episodes**: many short inner runs across a family of tasks, or replay.
3. **Replay is the cheapest meta signal available.**
   - EXP27 Part C measured instruction effects from 1,000 replayed calls (20 recorded prompts × 10 samples × 5 cells)
     for $1.46. That is about 50× more meta-level samples per dollar than full runs.
   - Per-call yields ("did this call introduce a new method family?", "did it beat its parent?") are dense signals
     that the base score is not.

---

## 6. Is recursive optimization achievable and useful?

**Within one bounded run: no, by the arithmetic above.** An O2 level that learns how O1 learns would get one or two
evaluations per run, so it cannot be identified. The repeated null results (EXP22, EXP24, EXP28 meta arms) are what
this predicts.

**Across episodes: plausibly yes, on specific axes, if amortized.** Recursion is useful when the artifact a meta level
produces is reused many times. Each artifact below is cheap to evaluate per call, reusable across tasks, and on an axis
that has shown leverage (§4):

| Artifact (meta output) | Evaluated by | Why it may pay |
|---|---|---|
| Mutation-instruction texts (DIVERGE/REFINE/context framing) | replayed per-call yields on recorded prompts, then full runs on held-out tasks | text was worth 5× per-call productivity in EXP28 |
| Mode policy code (when to diverge, what context, which inspirations) | short inner runs across a task family | EvoX's advantage was its policy; `VariationSearch` is a hand-written one |
| Lessons / helper library (functions, idioms, "what worked" notes) carried to new tasks | held-out tasks, with vs without the library | capitalization that EXP00-B Phase 6 attempted only as text, on a flat surface |
| Feedback / evidence formatter (what the solver sees per candidate) | held-out tasks, same budget | evaluator feedback was the largest effect measured (EXP24) |

O2 (learning how to learn these) is only worth attempting once an O1 artifact shows a resolved held-out gain. That
has not happened yet. Until then, O2 is the lowest priority.

**Amortization test.** Report break-even deployments = meta-training cost / per-deployment saving at matched quality,
as EXP08 did (2.25 on its finite menu).

---

## 7. EXP00 / later-equivalent items: which to re-test, re-design or drop

From [`EXP00/later_equivalent.md`](../EXP00/later_equivalent.md), judged against §2–§6:

| Item | Decision | Why / how |
|---|---|---|
| Capability with accuracy + cost (Experiment 0) | **Drop as is** | GSM8K saturated; nothing to learn. Revisit only on a task with headroom |
| Family policy / prior transfer (UC4, EXP02, EXP08) | **Re-design (priority)** | As cross-episode meta-learning of a `VariationSearch` mode policy or instruction texts, trained on a task family and tested on held-out tasks (P3) |
| Component code rewriting (EXP04, EXP06, EXP15–18, EXP23–24, EXP28) | **Keep as the base level** | The surface where optimization reliably works; use it as O0 for every EXP29 arm |
| Trainer choice and standard-vs-recursive (Phase 1, EXP22, EXP24, EXP28) | **Re-test first (P1)** | Close EXP28's gap: plain `PrioritySearch` vs `VariationSearch` vs coevolution (diff vs rewrite) on tasks with certified headroom |
| Priors and skills (Phase 3, Phase 6, EXP18–20) | **Re-design (P4)** | As an executable helper / lessons library carried across tasks, not a prompt prefix on a flat surface |
| Optimizer-side tools and agentic policies (Phase 4, UC5, UC9) | **Re-design later (P5)** | Never actually executed (0 tool calls). Test only real, executed tools (subset evaluation, retrieval of past candidates) |
| Trace type / feedback richness (Phase 2, UC6, EXP16, EXP19) | **Re-design (P2b)** | As a learned *compact* evidence formatter; rich feedback was adverse |
| QASPER prompt / config (UC2/6/11, EXP03, 12, 13) | **Drop** | Noisy, slow, small effects; no path to a resolved meta effect |
| Routing and code transfer (UC7, UC14, EXP07–09) | **Fold into P3 / P4** | Transfer is the question P3/P4 ask with a better surface |
| Guarded policies, numeric search (UC8, UC10, UC13) | **Drop** | Toy evaluators; numeric search over categorical knobs had no live signal |
| Threads (Phase 5, EXP05) | **Drop as science** | Engineering only: calibrate noise under the execution conditions used |
| Terminal-Bench 2 (Phase 7) | **Park** | Interesting task family with headroom, but needs an adapter first; candidate for the P0 task pool |

---

## 8. Prioritized EXP29 programme

All arms:
- Run on tasks with **certified legitimate headroom** and an **enforced metric** (P0).
- Report equal total calls, including meta calls.
- Use at least 5 seeds for any claim; report n = 3 as descriptive only.
- Pre-register the success and kill thresholds below.

### P0 — Task pool with certified headroom (prerequisite; no LLM cost beyond probes)

- **Candidates:**
  - Signal with causality **enforced** in the score (F1 from EXP27);
  - PRISM with harder generated cases and the all-case score;
  - two or three LLM4AD code tasks, e.g. online bin packing and admissible set, already wired in Trace-Bench;
  - optionally a Terminal-Bench-like sandboxed task.
- **Certification for each task:**
  1. The initial program's score.
  2. A reference upper level: a known bound, an exact optimum, or the best of strong references such as EvoX plus
     long runs.
  3. Headroom of at least 3× the run-to-run SD at n = 5.
  4. The metric checked against the known loopholes: look-ahead, refusals, truncation, format.
- **Kill:** drop any task where a plain fixed policy reaches 90% of the headroom within 30 calls.

### P1 — Isolate what `VariationSearch` changes (closes EXP28's gap; about 4 arms × 3 tasks × 5 seeds)

- **Arms:**
  1. plain `PrioritySearch`;
  2. `VariationSearch` default;
  3. coevolution with diffs;
  4. coevolution with `operator_mode='rewrite'`.
- **Endpoint:** best legitimate score by call; calls to 90% of certified headroom.
- **H1:** `VariationSearch` beats plain `PrioritySearch` on at least 2 of 3 tasks.
  - Success: CI of the paired difference above 0.
  - Kill: CI within ±0.25 SD of the task. If it dies, the EXP28 gain was the trainer format, not the schedule.

### P2 — Replay-based optimization of the mutation instructions (cheap meta signal; O1 on text)

- **Data:** recorded prompts from P1 runs (and EXP27/EXP28 logs) at matched decision points.
- **Surface:** the DIVERGE, REFINE and context-framing texts of `VariationSearch`, as trainable strings.
- **Optimizer:** an LLM proposes variants. Each variant is scored by per-call yield on 20+ held-out recorded prompts
  × 10 samples: new-method-family rate and beats-parent rate, measured on the legitimate metric.
- **Then:** deploy the best texts in full runs on held-out tasks against the default texts.
- **Success:** a per-call yield gain that survives in full runs (paired CI above 0).
- **Kill:** replay gains that do not transfer to full runs, which would mean replay is a misleading proxy.
- **Variant P2b:** the same procedure applied to a compact per-candidate evidence formatter (what the solver sees)
  instead of the instructions.

### P3 — Cross-episode meta-learning of the mode policy (the recursive question, done where it can pay)

- **Surface:** `VariationSearch`'s mode-policy function as **code**: when to diverge or refine, what context and
  inspirations to show, when to restart.
- **Meta training:** many short inner runs (e.g. 30 calls) on a training family of tasks. The meta optimizer gets the
  per-call yields and per-run outcomes of each episode.
- **Test:** full runs on held-out tasks against the hand-written default and the EvoX-brief variant.
- **Success:** a held-out gain at matched cost, with break-even below 10 deployments.
- **Kill:** no held-out gain after 3 meta rounds.
- **O2** (learning the meta optimizer's own instruction) is only allowed if P3 succeeds.

### P4 — Capitalization: an executable lessons / helper library carried across tasks

- **Surface:** a library the solver can import or read. It holds helper functions (e.g. a local-search routine, a
  validated filter family, scoring utilities) and short lessons, both extracted by an LLM from successful candidates of
  previous tasks.
- **Test:** held-out tasks with vs without the library, the same `VariationSearch` and the same budget.
- **Must separate:** information access from learning. Control arm: the library contents shuffled or taken from an
  unrelated family (the EXP20 lesson).

### P5 — Real executed tools for the optimizer (later)

- **Tools:** evaluate a candidate on a small subset before committing, and retrieve past candidates by similarity.
  Count actual tool calls; policy text that names a tool does not count.
- **Comparison:** with vs without tools at equal total calls, *including* tool-triggered evaluations.

### Order and budget

P0, then P1 (one batch), then P2 (cheapest meta signal), then P3 or P4 depending on P2, then P5. At about $0.15–0.20 per
100-call run, P1 is about 60 runs (≈ $10–12). P2's replay is about 1,000–2,000 calls (≈ $1.5–3) plus a deployment batch.
P3 and P4 are about 50–100 short runs each.

---

## 9. Pre-commitments (lessons from five withdrawn headlines)

1. **Score legitimacy first:** every task passes a loophole audit before any comparison.
2. **Same tasks, same scorer, same budget for every arm;** the three-way harness check for identical
   `scored_task_ids` stays on.
3. **Fixed-policy and plain-trainer controls in every comparison.**
4. **Confirm on a larger evaluation and on held-out tasks before adopting anything** (EXP00-B warm priors reversed).
5. **Count meta calls and tool calls in the budget;** report break-even for any meta artifact.
6. **Record per-call decisions** (mode, context, instruction, parent, child scores) so replay and credit analysis are
   possible. Experiment 0 stopped for lacking this.
7. **Check the treatment actually reaches the prompt,** and check later prompts for leaks (the EXP28 leak).

---

## 10. One-paragraph conclusion

Trace's measurable bottleneck was never raw optimizing power. It was what the optimizer was *asked* to do (mutation
intent and wording) and what *evidence* it saw (feedback and context). The meta-optimization attempts failed mainly
because:
- they searched low-leverage knobs or adapted within a single run, which gives the meta level about 10 noisy
  evaluations;
- the tasks had no legitimate headroom, or had exploitable metrics.

EXP28 shows that operator-level choices move per-call behaviour a lot (3% → 32%; 7% vs 38–44%). That makes the
instruction texts, the mode policy, the evidence formatter and a reusable helper library the most promising meta
surfaces. They should be learned **across episodes** (replay first, then short runs on a task family), tested on
held-out tasks with certified headroom, and judged against plain-trainer and fixed-policy controls. Recursion beyond
that (O2) is only justified after one of these O1 artifacts shows a resolved, amortized held-out gain.
