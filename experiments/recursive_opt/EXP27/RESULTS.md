# EXP27 — results

**Complete.**

- Part A: logs only, 21 runs.
- Part B: 72 calls, $0.10, underpowered.
- Part C: powered 2×2 plus an H8 re-test, 1,000 calls, $1.46.

## Part A — what the logs show (`results/mechanisms.json`)

| Run group | Best @42 | First SciPy (iteration, label, score vs best before) | Later parents using SciPy | SciPy candidates that look ahead | Best causal |
|---|---|---|---:|---:|---|
| stock (EXP25 ×3, EXP26 ×3) | 0.584–0.705 | 6/6 runs, it. 16–31, always above best (e.g. 0.675 vs 0.534) | 0.42–0.96 | 92–100% | 0.534–0.561 |
| Trace winners (native s43, s44; stocklabels s42) | 0.565–0.703 | it. 13–47, above best (0.658 vs 0.531; 0.624 vs 0.525; 0.607 vs 0.565) | 0.46–0.92 | 93–100% | 0.532–0.565 |
| Trace, SciPy found but below best (exp23 s42, pkg s42, s43) | 0.544–0.552 | 0.399 vs 0.549; 0.547 vs 0.550; 0.521 vs 0.540 (causal) | 0.00–0.02 | — | 0.530–0.532 |
| Trace, no SciPy (9 runs) | 0.524–0.576 | none | — | — | 0.499–0.558 |

Findings:

1. **The gap is entirely look-ahead (H5).** Stock's winning candidates are zero-phase or centred SciPy filters
   (`filtfilt`, `savgol_filter`, `medfilt`). Among strictly causal candidates the engines are equal: stock 0.534–0.561,
   Trace 0.499–0.565.
2. **Adoption is not the problem (H4 refuted).** In both engines, a SciPy candidate that beats the best so far takes
   over the parents. One that doesn't is dropped. Context sizes are similar (about 3.8 programs).
3. **The DIVERGE rate is not the problem (H1 refuted).** Matched by exact label text, stock selects DIVERGE in 15–24%
   of iterations, or 8–23% of successful ones. Trace's rate is mostly 0–18%. My EXP26 analysis matched labels by a
   40-character prefix that both stock labels share; that error produced the "40–53%" figure and the R3 reading,
   both now withdrawn. Within Trace, the two high-DIVERGE runs (83%, 43%) did win. Stock gets the same discovery at
   ordinary rates, so the rate is a contributor at most.
4. **Labels (H2) and LLM settings (H3) are refuted.** Injected stock labels still miss the gap at equal budget, and
   the model, provider, temperature, token limit and reasoning setting are identical.
5. **The difference is discovery.** A winning look-ahead SciPy candidate appears in 6/6 stock runs against 3/15 Trace
   runs (one-sided Fisher p ≈ 0.0015), so chance alone (H9) is unlikely.
6. **One structural asymmetry remains, in the experiment setup.** Trace arms are scored by EXP25's whitebox evaluator,
   whose metrics include `causal_fraction`, `valid_score` and `fallback_signals`. The operator prints all metrics,
   as SkyDiscover does, so every Trace parent shows `causal_fraction: 1.0000`. Stock's own evaluator computes none
   of these. Part B tests whether that cue (H6), or the other prompt differences (H8), suppress look-ahead.
7. **Program size (H7) is weak evidence only.** Trace's EVOLVE blocks are 9–13k characters against stock's 6–9.5k.
   Not tested causally.

## Part B — prompt ablation (`results/ablation/completions.jsonl`)

There were 6 completions per prompt and variant, across 4 DIVERGE prompts with non-SciPy parents. "Applied" means
the completion's diffs applied to the parent.

| Variant | Applied | Look-ahead among applied | SciPy among applied | Beats parent | Best valid |
|---|---:|---:|---:|---:|---:|
| T0 recorded | 22/24 | 59% | 0/22 | 36% | 0.574 |
| T1 no `causal_fraction` | 18/24 | 67% | 3/18 | 44% | **0.632** |
| T2 stock-like | 18/24 | 50% | 1/18 (causal) | 44% | 0.565 |

Decision rules:

- **H6: inconclusive.** T1 − T0 = +0.08: above the 0.05 refutation bound, below the 0.20 support bound. The sign
  is positive in 3/4 prompts.
- **H8: refuted.** T2 − T1 = −0.17.

Exploratory, not pre-registered: SciPy appeared only once the cue was removed (3/18 against 0/22, one-sided Fisher
p = 0.08). Those three completions were look-ahead zero-phase filters scoring 0.574–0.632, against parents about
0.51. All three came from one prompt (`native_stocklabels_s44`), so this is a hint, not evidence.

What Part B changes:

1. **The cue does not stop Trace from looking ahead in general.** Even as recorded, 59% of completions look ahead,
   with centred windows written by hand. What Trace lacks is the SciPy zero-phase filter specifically.
2. **Part B was underpowered for the quantity that matters.** I should have computed this before the run. Stock's
   first SciPy candidate arrives at iteration 16–31, about 3–6% per call. Trace's runs give about 0.7% per call
   (6 discoveries in roughly 900 calls). Telling those rates apart needs about 200 completions per variant, not 24.

## Part C — powered factorial (`results/factorial/`)

Design: 20 prompts × 5 cells × 10 samples, as pre-registered. The first launch stalled because hung requests held
workers: OpenRouter keep-alive bytes defeat the client's read timeout. It was resumed with a 300 s wall-clock
deadline per call, after which none was hit. The 82 completions written before the stall were kept; the design is
deterministic.

| Cell | SciPy (primary) | SciPy look-ahead | Beats parent | Applied | Best valid |
|---|---:|---:|---:|---:|---:|
| cue on, Trace parent (as Trace runs) | 12.5% | 9.0% | 34% | 82% | 0.693 |
| cue on, stock parent | 11.0% | 7.5% | 36% | 78% | 0.674 |
| cue off, Trace parent | 21.5% | 17.0% | 46% | 82% | 0.700 |
| cue off, stock parent | 19.0% | 14.0% | 45% | 85% | 0.711 |
| stock-like, Trace parent | 16.5% | 12.0% | 37% | 84% | — |

Each effect is the mean over prompts of the per-prompt difference, with a 95% bootstrap CI over prompts.

| Effect on SciPy share | Estimate | 95% CI | Prompts +/− | Verdict |
|---|---:|---|---|---|
| H6: cue off − cue on | **+0.085** | [+0.053, +0.123] | 16 / 0 | **Supported** |
| H10: stock parent − Trace parent | −0.020 | [−0.100, +0.052] | 7 / 9 | Inconclusive by rule; no sign of an effect |
| Interaction | −0.010 | [−0.075, +0.060] | 8 / 8 | none detected |
| H8: stock-like − cue off (Trace parent) | −0.050 | [−0.100, +0.005] | 5 / 12 | Inconclusive by rule; if anything harmful |

Removing the `causal_fraction` line multiplies SciPy discovery by about 1.7 (12% to 20%). It also raises the share
of completions that beat their parent by 10 points (CI [+0.055, +0.153]). The effect is the same on Trace and stock
parents. Swapping in a score-matched stock parent changes nothing, so Trace's larger, hand-tuned programs are not
the cause (H7/H10).

**Consistency check: does the cue account for the run-level gap?** Per call in the runs, stock discovers SciPy at
about 3–6% and Trace at about 0.7%. Under DIVERGE, Trace prompts give 12.5% with the cue and 20% without. A rough
model is: DIVERGE share × the rate under DIVERGE, plus a smaller contribution from other labels. With Trace's DIVERGE
share (about 13%) and the cue, that gives about 1.6%+. With stock's (15–24%) and no cue, about 3–5%+. That is the
right order of magnitude for both engines, but it is an estimate, not a test.

## Status of the root causes

| | Verdict |
|---|---|
| Gap = look-ahead loophole (H5) | **Established**: no causal gap in 21 runs |
| Causal cue in Trace's prompts (H6) | **Established** (Part C): it suppresses SciPy discovery about 1.7× and raises beats-parent by 10 points |
| DIVERGE rate, label text, LLM settings, parent/context selection (H1–H4) | **Refuted** |
| Chance (H9) | **Refuted** as sole cause (p ≈ 0.0015) |
| Parent lineage and size (H7, H10) | **No effect detected**. Estimate −0.02, CI ±0.08 |
| Stock task text and feedback framing (H8) | **Not a cause**. If anything it lowers discovery (−0.05) |

**Root cause, in one line:** EXP25's evaluator exposes the audit metric `causal_fraction` to Trace's LLM, and to
stock's never. Reading "causal_fraction: 1.0000", the LLM avoids the zero-phase SciPy filters that exploit the
task's look-ahead loophole. Stock, uncued, finds and exploits them.

## Implication so far

On the signal task, Trace does not trail EvoX at finding legitimate (causal) filters. It trails at finding a scoring
loophole that the task text ("real-time … minimal phase delay") forbids. Both fixes are now justified:

- **F1, causal scoring for both engines,** makes the loophole worthless and compares the engines on the stated task.
- **F2, symmetric prompts,** keeps audit metrics out of LLM-visible metrics for both engines. Part C shows this
  matters: the cue alone moves discovery about 1.7×.

Run them together. F2 alone would mainly let Trace exploit the loophole as stock does.
