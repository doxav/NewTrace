# EXP28 — results

> **Correction (2026-10-07, see PROTOCOL Part B).** The two trainer arms below ran with an instruction-leak bug. Once
> a candidate created on an exploration step was expanded, its later "free" calls still carried the DIVERGE/COMBINE
> instruction, sometimes stacked. Leaks occurred in 5 of 6 runs, from calls 4–37 on (52–65 of 67–81 free calls). The
> trainer rows therefore measure a contaminated treatment, and their "free vs diverge" yields are not interpretable.
> The first SciPy candidates at call 9 (stagnation seeds 42 and 43) came before any leak. Part B re-runs the trainer
> arms with the fix. The recursive_opt rows are unaffected.

**Complete, 12/12.** Campaign `results/runs_20261007T125224`: 12 runs concurrent, 12:52–14:37 UTC on 2026-10-07,
100 solution calls each. There were 0 failed calls and 0 deadlines; cost $2.18. Analysis: `scripts/analyze.py` writes
[`analysis.json`](results/runs_20261007T125224/analysis.json). Figure (a) and the table are in
[`key_findings.ipynb`](../_analysis/retrospective_20261006/key_findings.ipynb), EXP28 section.

| Arm (3 seeds) | Best, median (top) | Look-ahead ≥ 0.615 (call) | Best causal, median (top) | Mode / label mix | SciPy introduced, per mode |
|---|---|---|---|---|---|
| Trainer: VariationSearch stagnation | **0.748** (0.785) | 3/3 (9, 10, 55) | 0.544 (0.547) | diverge 19%, free 81% | free 11/109, diverge 2/24 |
| Trainer: VariationSearch combine | 0.591 (0.752) | 1/3 (1) | 0.554 (**0.578**) | combine 33%, free 67% | free 1/135, combine 0/65 |
| recursive_opt: EvoX brief | 0.663 (0.752) | 3/3 (20, 34, 77) | 0.536 (0.550) | none 84%, refine 9%, diverge 7% | none 0/178, refine 0/20, diverge 1/15 |
| recursive_opt: diverge guard | 0.606 (0.660) | 1/3 (11) | 0.519 (0.528) | none 63%, refine 19%, diverge 18% | none 0/161, refine 1/2, diverge 0/37 |
| *EvoX (6, reference)* | 0.711 (0.723) | 5/6 (5–29) | 0.543 (0.561) | — | diverge 6/8 |
| *Trace, cue hidden (8, EXP27)* | 0.637 (0.734) | 6/8 (3–92) | 0.539 (0.554) | refine ~72% | diverge 8/189 (all Trace) |
| *EXP22 Trace, fixed policy (1)* | 0.606 | 0/1 | 0.528 | — | — |
| *EXP22 Trace, recursive (2)* | 0.550 (0.556) | 0/2 | 0.544 (0.554) | — | — |

The EXP22 runs used a hybrid with SkyDiscover's operator and stock evaluator (no cue). They were re-scored with the
same white-box evaluator as all other rows.

## Pre-registered reading (n = 3, descriptive)

- **Closes the exploration gap** (at least 2/3 runs reach look-ahead within 50 calls):
  - *stagnation*: **yes** (calls 9 and 10);
  - *brief*: **yes** (calls 20 and 34);
  - *combine*: no; *guard*: no.
- **Improves legitimate search** (median best causal more than 0.02 above 0.539 / 0.543): **no arm.** *combine*
  comes closest (median 0.554). Its seed 42 reached **0.578**, the best real-time score of any run so far (previous
  maximum 0.565), but that is one run.

## What the logs show about the mechanisms

1. **Trainer, stagnation: the highest raw scores, but not because of the schedule.**
   - The median of 0.748 is above EvoX's 0.711, and the top run of 0.785 is the highest benchmark score recorded.
     All of it is look-ahead (`causal_fraction` 0), mostly hand-written centred windows.
   - The first SciPy candidates arrived on ordinary ("free") calls at call 9: 11 of 109 free calls, against 2 of 24
     DIVERGE calls. The trainer's own format already explores. That format is a full-program OptoPrimeV2 rewrite
     with the task in the instruction, no elite context programs and no SEARCH/REPLACE diff.
   - No plain-`PrioritySearch` baseline was run, so the DIVERGE schedule's own effect is not identified. The
     per-call yields give no sign that DIVERGE calls beat free calls.
2. **Trainer, combine: the same trainer, much less look-ahead.**
   - Free calls brought SciPy in 1 of 135 (seed 44, at call 1) and combine calls in 0 of 65.
   - Seeds differ widely: seed variance dominates at n = 3.
   - Showing non-elite inspirations did not trigger new families, but this arm holds the best causal run.
3. **recursive_opt, EvoX brief: the policy changed as intended.**
   - With EvoX's rules, Trace's meta level wrote EvoX-like policies: 84% of iterations unlabelled, 9% refine and 7%
     diverge. Without the brief (EXP27) it was about 72% refine.
   - Look-ahead in 3/3 runs, but weaker forms (median 0.663 vs EvoX 0.711).
   - DIVERGE calls are still rarely productive: 1 of 15. Fixing the policy does not fix what a DIVERGE call yields.
4. **recursive_opt, diverge guard: the context-anchoring hypothesis is refuted.**
   - 37 forced DIVERGE calls with no context programs introduced SciPy 0 times. EXP27's leading suspect (four elite
     context programs anchor the LLM) is not the cause.
   - The guard also lowered results: median 0.606, best causal 0.519.

## Updated diagnosis and next step

**Ruled out as the cause of the low productivity of Trace's DIVERGE calls** (EvoX 6/8; Trace 1/15 even with the
brief, 0/37 with no context):
- the cue;
- the label text (EXP26);
- the policy's label mix (brief);
- the context programs (guard);
- the task-text and feedback framing (EXP27 Part C, H8).

**What remains:**
1. **Parent maturity and timing.** EvoX diverges early (iterations 15–31) from small parents. Trace's DIVERGE calls
   come later, on larger tuned programs (median iteration 49, 10.3k vs 7.5k characters). Part C found no lineage
   effect on eligible parents, but did not vary timing.
2. **Small-sample variance in EvoX's yield.** 6/8 has a 95% CI of 0.41–0.93.

**Separately, the trainer explores readily.** It rewrites the whole program each call, whereas both EvoX and Trace's
coevolution engine edit by SEARCH/REPLACE diffs. So the edit format may explain the trainer's look-ahead discovery,
but not the EvoX–Trace gap.

**Next tests (no new code):**
- Rerun the cue-hidden coevolution configuration with `operator_mode='rewrite'` (3 seeds, about $0.6), to test
  whether full rewrites make the engine explore like the trainer.
- Add a plain-`PrioritySearch` arm (3 seeds) to isolate the VariationSearch schedule.

**Scope reminder:** on this benchmark, faster exploration mostly means finding the look-ahead loophole faster. None
of the four approaches improved the real-time (causal) score beyond run-to-run noise.

## Part B — inspiration ablation with the leak fixed (`results/partB_20261007T160822`)

**Complete, 18/18.** All runs ran concurrently, 16:08–17:38 UTC, 100 solution calls each, about $2.5. There were 0
instruction leaks (free calls carrying an instruction). Failed calls: 0–2 per run, all retried. Analysis:
`scripts/analyze.py` writes [`analysis.json`](results/partB_20261007T160822/analysis.json). The figure and table are
at the end of [`key_findings.ipynb`](../_analysis/retrospective_20261006/key_findings.ipynb).

| Configuration (3 seeds) | Best, median | Look-ahead ≥ 0.615 (calls) | Best causal, median | SciPy introduced on exploration calls | on free calls |
|---|---|---|---|---|---|
| **default**: stagnation, no inspirations | **0.758** | 3/3 (2, 6, 22) | 0.523 | diverge **14/44 (32%)** | 6/189 (3%) |
| stag + alternate combine | 0.760 | 3/3 (2, 3, 26) | 0.502 | diverge 3/8, combine 1/6 | 8/64 |
| stag + alternate context | 0.757 | 3/3 (3, 3, 6) | 0.499 | diverge 4/11, context 4/9 | 9/89 |
| stag + always context | 0.725 | 3/3 (4, 6, 7) | 0.499 | context 8/21 (38%) | 18/91 |
| periodic diverge | 0.751 | 3/3 (1, 1, 7) | 0.499 | diverge 16/48 (33%) | 1/97 |
| periodic combine (Part A arm, fixed) | 0.718 | 3/3 (2, 8, 85) | 0.537 | combine **5/69 (7%)** | 1/140 |
| *EvoX (6, reference)* | 0.711 | 5/6 (5–29) | 0.543 | diverge 6/8 | — |

**Readings** (n = 3; differences in the best score of a few hundredths are within run-to-run noise):

1. **Regression check passed.** The default re-run has a median of 0.758, against Part A's 0.748, and 3/3 runs reach
   look-ahead within 50 calls. The default remains the winner, together with the two "alternate" variants.
2. **With the leak fixed, DIVERGE works as designed.**
   - In the default, DIVERGE calls introduce a SciPy filter in 32% of cases (14/44), against 3% for free calls.
   - Periodic DIVERGE does the same (33%).
   - This is the EvoX-like productivity that Trace's coevolution engine never reached (Trace 4–7%, EvoX 75% on 8
     calls).
3. **The explicit "combine" sentence is the limiter.**
   - Combine calls with that sentence yield 7% (5/69 periodic) and 1/6 (alternate).
   - The same inspirations shown as plain context (EvoX-style) yield 38–44% (8/21, 4/9), as good as plain DIVERGE.
   - The inspirations themselves do not hurt. The instruction to synthesize from them does.
4. **The schedule matters little here.** Stagnation and periodic DIVERGE give similar medians (0.758 / 0.751) and
   yields (32% / 33%).
5. **No legitimate gain.** Every configuration reaches look-ahead (the loophole) in 3/3 runs, mostly within the first
   10 calls. The median best causal scores are 0.499–0.537, not above the references (0.539 / 0.543). Once look-ahead
   wins, causal candidates stop being explored.

**Recommendation for VariationSearch:**
- Keep the default: stagnation, `inspiration_mode='never'`.
- If inspirations are wanted, use `inspiration_style='context'`, which is EvoX-like. `alternate` or `always` work;
  avoid the explicit `'combine'` style.
- On this benchmark, the extra exploration mostly buys faster loophole discovery. Testing legitimate gains needs a
  causality-enforcing score (F1 in EXP27).

## Part C — the same configurations on PRISM, plus stock EvoX (`results/prism_20261007T200146`)

All 21 runs launched concurrently at 20:01 UTC on 2026-10-07. EvoX completed 100 calls per run. The 18 trainer runs
were **stopped at 21:10 UTC by request**, once almost all had reached the all-case optimum, after 14–52 calls each.
The optimum cannot be exceeded, so later calls could not change the primary endpoint. Cost $0.89. Analysis:
`scripts/analyze_prism.py` writes [`analysis.json`](results/prism_20261007T200146/analysis.json); EvoX candidates are
re-scored with EXP24's white-box evaluator.

| Configuration (3 seeds) | Reached the all-case optimum 26.256 | Calls to optimum | Median best all-case | Median best stock |
|---|---|---|---|---|
| **VariationSearch default** (stagnation, no inspirations) | **3/3** | **3, 3, 4** | 26.256 | 26.256 |
| periodic diverge | 3/3 | 2, 3, 7 | 26.256 | 26.256 |
| stag + always context | 3/3 | 3, 5, 9 | 26.256 | 26.256 |
| periodic combine | 3/3 | 6, 7, 7 | 26.256 | 26.256 |
| stag + alternate combine | 2/3 (third at 26.232 when stopped at call 18) | 6, 7 | 26.256 | 26.282 |
| stag + alternate context | 2/3 (third at 26.250 when stopped at call 17) | 4, 14 | 26.256 | 26.355 |
| **stock EvoX** (100 calls) | **1/3** | 37 | **24.149** | 25.877 |
| *EXP24 reference, same evaluator and feedback (coevolution engine)* | 3/3 each | fixed 12, llm_rewrite 12, Trace 22 (medians) | 26.256 | — |

**Readings** (n = 3, descriptive):

1. **VariationSearch is the fastest configuration measured on PRISM.**
   - The default reaches the all-case optimum at calls 3, 3 and 4. EXP24 needed a median of 12 (fixed policy and
     `llm_rewrite`) or 22 (Trace's meta level) with the same evaluator and feedback.
   - All six configurations reach it in 3/3 runs, or 2/3 where the third run was stopped near the optimum.
   - Caveat: the trainer also differs from EXP24's coevolution engine (full-program rewrites, a single lineage), so
     the speed-up is not attributable to the variation schedule alone.
2. **Inspirations neither help nor hurt much here.**
   - PRISM is solved within about 10 calls, and the "combine" variants are slightly slower (6–7 calls).
   - The Signal finding (the explicit combine sentence lowers productivity) is consistent with this, but PRISM has
     too little headroom to separate the configurations.
3. **Stock EvoX is behind, and exploits.**
   - Only 1 of 3 runs reached the optimum (call 37). The other two top out at 23.69 and 24.15 on all 50 cases.
   - Their best stock scores (24.64 and 25.88) come from skipping 22–50% of the cases, the refusal loophole. In
     seed 42 too, the stock record 26.39 is a refusal program (78% solved), while its fully solved best is 26.256.
4. **The trainer's stock scores slightly above the optimum are incidental.**
   - Values such as 26.56 and 26.60 occur in a few trainer runs, whose best all-case score is still 26.256.
   - A case that exceeds the per-case time limit is skipped even under the fallback projection. The trainer optimized
     the all-case score, which counts such cases, so it did not target this.

**Overall (Signal and PRISM):** `VariationSearch`'s default is the best Trace configuration on both tasks.
- On **Signal** it matches or exceeds EvoX's discovery speed, which there means finding the look-ahead loophole.
- On **PRISM** it reaches the legitimate optimum in 3–4 calls, where EvoX reaches it in 1 of 3 runs.

On both tasks the real-time (causal) Signal score and the all-case PRISM optimum are unchanged by exploration: Signal
stays near 0.54 and PRISM's ceiling is reached by everyone who solves it.
