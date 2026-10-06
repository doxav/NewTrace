# EXP27 — results

**Part A complete (logs only, 21 runs). Part B pending:** it needs the OpenRouter key's total limit raised. EXP26
ended when that key ran out.

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

## Part B — prompt ablation

_Pending._ Running it is 72 calls: `scripts/prompt_ablation.py --out results/ablation`. The harness was checked
offline with `--mock` and `tests/test_exp27.py` (4 passed): the variants change only their factor, and the
evaluator flags a `filtfilt` program as look-ahead.

## Implication so far

On the signal task, Trace does not trail EvoX at finding legitimate (causal) filters. It trails at finding a scoring
loophole that the task text ("real-time … minimal phase delay") forbids. Fix F1 (causal scoring for both engines) is
justified by Part A alone. Fix F2 (symmetric prompts) waits on Part B.
