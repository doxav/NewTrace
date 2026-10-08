# Retrospective: did the EvoX-comparison campaign (EXP22–EXP27) improve Trace recursive_opt?

Date 2026-10-06. Sources: each experiment's RESULTS.md and the new learning curves of
[`EXP27/results/learning_curves.json`](../../EXP27/results/learning_curves.json) (`EXP27/scripts/learning_curves.py`,
CPU only: every signal candidate of 32 runs scored as valid and as causal). One-page rendered summary:
[`key_findings.ipynb`](key_findings.ipynb) ([HTML](key_findings.html)), built by `build_key_findings.py`. Same LLM throughout
(`z-ai/glm-5.3-flash` via Novita, temperature 0.7), 100 solution calls per run, 3 seeds per arm unless stated.

## Short answer

- **Measured on the tasks as stated, the impact is flat.** No change made in EXP22–EXP27 has made Trace find better
  legitimate solutions or find them faster, and neither has stock EvoX. Every raw lead on either side was a scoring
  exploit.
- **The real gains are below the meta level.** They come from fixing the evaluator and the per-iteration operator,
  plus making the runs reliable. They help a fixed policy just as much.
- **The meta level (Trace optimizing its own search policy) has never beaten a fixed policy.** It was compared in
  EXP22 and EXP24.
- **The exploration deficit is the one Trace-specific weakness found, but it is not the only finding.** On these
  tasks it has so far cost Trace only the cheats, not a legitimate gain.

## 1. What each attempt changed and what it bought

| Exp | Change made to Trace recursive_opt | Level | Raw headline | Corrected reading |
|---|---|---|---|---|
| EXP22-EvoX | Trace as policy optimizer in a SkyDiscover hybrid; context-growth repair | Meta + engineering | Recursive below fixed on both tasks (PRISM 24.85 vs 26.36; signal 0.544 vs 0.606) | Partial runs, unequal budgets. No meta gain; fixed was ahead. PRISM 26.36 is above the all-case optimum, so it is an exploit (success 0.92) |
| EXP23 | Native coevolution engine (control plane v2); Trace proposer vs EvoX-style `llm_rewrite` | Engine + meta | PRISM: Trace 30.88 vs 26.20 | **Withdrawn**: 30.88 solves 3/50 cases and re-scores to 21.08, below the initial 21.89. Best fully solved archive candidates are 26.233 vs 26.203, one run each |
| EXP24 | Valid all-case guide, per-case white-box feedback, repair projection (O0) | Evaluator + operator | 9/9 runs reach the PRISM optimum 26.256; before this, 1 of 7 did | **Real gain, but not recursive.** A fixed policy is as fast (median calls to optimum: fixed 12, `llm_rewrite` 12, Trace 22) and wastes the fewest attempts. The meta level adds failed attempts, not speed |
| EXP25 | Best Trace configurations (EXP23, EXP24) vs stock EvoX on signal | Comparison | EvoX 0.713 vs Trace 0.586 / 0.555 | **Lead is look-ahead only**: zero-phase SciPy filters read future samples. Causal-only medians are equal: 0.537 / 0.532 / 0.532 |
| EXP26 | Label fidelity: package-aware label generator; stock labels injected | O0 labels | native 0.652, native_pkg 0.554, stocklabels 0.576 vs stock 0.709 | Neither port closed the gap; the package-aware port was the worst arm. Causal-only all 0.525–0.558. Truncated by the key limit |
| EXP27 | Diagnosis: cue ablation (Parts B–C), full runs with the cue hidden (D), strategy logs (E) | Diagnosis | Cue hidden: median 0.638 vs 0.586 shown | Hiding the cue lets Trace cheat a bit more; causal-only is unchanged (0.539). Not a capability limit: the climb once the cheat is found matches stock's (+0.050 vs +0.045). Trace's meta level writes refine-locked, elite-context policies |

## 2. Learning curves on the signal task (median per arm)

Best score among candidates so far; the initial program scores 0.499. "Any" uses the benchmark's own valid score;
"causal" counts only real-time (no look-ahead) candidates, which is the task as stated. EXP26 runs stop producing
candidates at iterations 43–70, so @42 is the last point where every arm has the same budget.

| Arm (runs) | Any @10 | Any @25 | Any @42 | Any @100 | **Causal @10** | **Causal @25** | **Causal @42** | **Causal @100** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stock EvoX (6) | 0.525 | 0.646 | 0.685 | 0.711 | 0.516 | 0.537 | 0.543 | 0.543 |
| EXP25 trace_exp23 (3) | 0.531 | 0.542 | 0.549 | 0.555 | 0.530 | 0.532 | 0.532 | 0.532 |
| EXP25 trace_exp24 (3) | 0.527 | 0.541 | 0.555 | 0.586 | 0.527 | 0.532 | 0.532 | 0.532 |
| EXP26 native (3) | 0.527 | 0.624 | 0.652 | 0.652 | 0.525 | 0.525 | 0.525 | 0.525 |
| EXP26 native_pkg (3) | 0.545 | 0.545 | 0.552 | 0.554 | 0.530 | 0.530 | 0.530 | 0.530 |
| EXP26 native_stocklabels (3) | 0.531 | 0.551 | 0.565 | 0.576 | 0.531 | 0.551 | 0.557 | 0.558 |
| EXP27 cue hidden (8) | 0.547 | 0.555 | 0.576 | 0.637 | 0.536 | 0.539 | 0.539 | 0.539 |
| EXP27 cue shown (3) | 0.532 | 0.537 | 0.539 | 0.548 | 0.521 | 0.525 | 0.525 | 0.525 |

Reading:

- **"Any" is the cheat curve.** EvoX's lead opens between iterations 10 and 25, when its first DIVERGE calls bring
  in zero-phase SciPy. Any Trace arm that rises above about 0.6 does so by look-ahead too.
- **The causal curves are flat and tied.** Every arm gains only +0.02–0.06 over the initial program, and almost all
  of it by iteration 10–25. The causal medians span 0.525–0.558, with stock EvoX in the middle (0.543). That spread
  is within run-to-run noise: individual runs range 0.499–0.565.
- **Nothing has trended up across experiments.** In order, EXP23 0.532, EXP24 0.532, the EXP26 ports 0.525–0.558
  and EXP27 0.525–0.539.
- **Trace does not learn faster.** At @10 Trace arms are level with or slightly above stock (0.52–0.54 against
  0.516), and they stop improving at about the same point.
- **Caveat:** the causal ceiling of this task is unknown, and nobody has exceeded 0.565. With this model the task
  may simply have little legitimate headroom, which would make it a weak discriminator of search strategies.

## 3. PRISM over the same period

| Stage | Best legitimate (all 50 cases solved) | Calls to the optimum 26.256 |
|---|---|---|
| EXP22 (hybrid) | below the optimum (the 26.36 and 24.85 programs leave cases unsolved) | not reached |
| EXP23 (native, stock metric) | Trace 26.233, `llm_rewrite` 26.203 (archive maxima) | not reached (1 of 7 EXP22/EXP23 runs reached it) |
| EXP24 (valid guide + feedback + projection) | **26.256 in 9/9 runs** | fixed 12, `llm_rewrite` 12, Trace 22 (medians) |
| Stock EvoX (EXP23-era runs) | 26.256, also 25.97 and 25.72 | — |

PRISM went from "rarely solved" to "always solved" because of the **evaluator and operator** changes in EXP24.
Meta-optimization did not contribute: a fixed policy did it as fast. Since EXP24, PRISM has had no headroom left for
a meta-level test. The 30+ scores on both sides come from refusing hard cases, which the stock metric rewards.

## 4. Are there real improvements?

| Question | Answer | Evidence |
|---|---|---|
| Better legitimate solutions than before EXP22? | **PRISM yes (reliability), signal no** | PRISM 1/7 → 9/9 at the optimum (EXP24, at O0). Signal causal flat at 0.53–0.56 |
| Better than EvoX? | **No, and EvoX is not better than Trace either** | Equal on causal signal and on PRISM all-case. EvoX's raw leads are exploits |
| Learns faster? | **No** | Signal causal@10/25 tied. PRISM median calls to optimum: Trace 22 vs fixed 12 (n = 3, not significant) |
| Finds more solutions (diversity)? | **No, fewer distinct technique families** | Trace's DIVERGE yield is 4.2% vs stock's 75%. It rarely leaves the hand-written NumPy family |
| Does the meta level (recursion) add value? | **Not shown anywhere** | EXP22: fixed ≥ recursive. EXP24: fixed = meta, with fewer wasted attempts. Signal never had a fixed-policy arm |

**So yes: the impact on recursive_opt itself is nearly flat.** What did improve is real but sits elsewhere:

1. **Measurement.** Exploit detection: PRISM refusals, signal look-ahead, the computed PRISM optimum, the causality
   probe and equal-budget accounting. Five raw headlines have been withdrawn or corrected (EXP22, EXP23, EXP25,
   EXP26 R3, EXP27 "root cause = cue"). This is the most valuable output of the campaign.
2. **O0 operator and evaluator design.** The valid guide, white-box feedback and repair projection made PRISM
   reliably solved (EXP24).
3. **Engineering.** Native coevolution engine, context-growth repair, provider-error accounting, wall-clock
   deadlines, and fidelity checks against stock EvoX.

## 5. Is the exploration deficit the only real finding?

It is the only Trace-specific weakness established, but it is not the only finding, and its cost so far is narrow.

- **What is established (EXP27 Part E).**
  - Trace's meta level (`TraceProposer` = OptoPrimeV2 with a generic system prompt and a 9-line contract) writes
    policies that attach REFINE about 72% of the time, always show 4 top-scoring programs as context, and never
    return to uninstructed iterations.
  - Stock EvoX's meta prompt is a 9.5k-character brief that prescribes "no label by default, labels only when
    stagnating, avoid fixed rules, value diversity". Its policies stay balanced.
  - New technique families arrive only through DIVERGE: 75% of stock's DIVERGE calls bring SciPy, against 4.2% of
    Trace's.
- **What is not established.**
  - **That this exploration deficit costs Trace any legitimate score.** On the causal signal task and on PRISM,
    Trace ties. The deficit has only been shown to cost the cheat.
  - **The mechanism inside a DIVERGE call.** Elite-context anchoring is the leading suspect; it is untested.
- **The other findings matter as much.**
  1. **Both benchmarks, as scored, reward cheating.** A comparison on them measures exploit discovery, not
     optimization.
  2. **A fixed policy is a hard baseline.** Recursive policy search has not beaten it on any task tried, and the
     tasks tried have little legitimate headroom.
  3. **Most of Trace's movement came from lower levels.** Evaluator feedback, operator prompts and projection moved
     results; policy recursion did not.

## 6. What would make the next experiment informative

1. **Pick a task with legitimate headroom and an enforced metric.** Score the signal task with causality enforced,
   PRISM all-case, or a new task whose verified ceiling is clearly above what a fixed policy reaches. Without
   headroom, no meta-level effect can show.
2. **Always include a fixed-policy arm, and use at least 6 seeds.** Three runs per arm have not separated any pair
   in this campaign.
3. **Fix exploration at the meta level first, then test whether recursion helps.** Give `TraceProposer` the stock
   design brief (or run `llm_rewrite` with the full brief). Then compare fixed, stock-brief Trace and current Trace
   on the enforced metric, with calls-to-threshold and best@k as endpoints.
4. **Test DIVERGE context anchoring offline (about $0.40)** before changing the O0 contract.

**Addendum 2026-10-08 ([EXP29](../../EXP29/prior_analysis.md)).** The flat EXP22–28 impact is also a task effect. Signal's legitimate range (hand-written causal filters 0.467–0.539, best LLM 0.565) is inside its seed spread, and PRISM is saturated. The meta studies that could resolve an effect (numeric optimizer programs, 4-document QA) never ran their O1 stage.
