# EXP27 — results

**Complete.** Part A: logs only, 21 runs. Part B: 72 calls, all succeeded, cost $0.10.

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

## Status of the root causes

| | Verdict |
|---|---|
| Gap = look-ahead loophole (H5) | **Established**: no causal gap in 21 runs |
| DIVERGE rate, label text, LLM settings, parent/context selection (H1–H4) | **Refuted** |
| Chance (H9) | **Refuted** as sole cause (p ≈ 0.0015) |
| Stock task text and feedback framing (H8) | **Refuted** (Part B) |
| Causal cue in Trace prompts (H6) | **Inconclusive.** A weak positive hint, from one prompt |
| Remaining: per-call SciPy discovery from Trace's own parents (lineage, size: H7) | Untested. Next test below |

Next test, if the loophole itself is worth chasing: a powered replay of about 200 completions per arm (about $0.30),
crossing two factors. The cue is present or absent. The parent is a Trace program or a stock program of equal score.
This separates the prompt-cue effect from the parent-lineage effect on per-call SciPy discovery.

## Implication so far

On the signal task, Trace does not trail EvoX at finding legitimate (causal) filters. It trails at finding a scoring
loophole that the task text ("real-time … minimal phase delay") forbids. Fix F1 (causal scoring for both engines) is
justified by Part A alone and is the recommended next campaign. Fix F2 (symmetric prompts) is cheap and removes a
confound; Part B shows it is at most a modest lever, so it should go with F1 rather than be relied on alone.
