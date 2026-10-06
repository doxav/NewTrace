# EXP27 protocol — why does Trace trail stock EvoX on the signal task?

Evidence base: EXP25 (9 runs, complete) and EXP26 (12 runs, truncated by an OpenRouter key limit at iteration 43–70;
compared at an equal 42-iteration budget where it matters). Part A uses logs only; Part B is a small LLM probe.

## Five whys

1. **Why does Trace's best valid score trail?** (EXP25/26 medians about 0.58 against 0.71.) Stock's best candidates
   all import SciPy (`filtfilt`, `savgol_filter`, `medfilt`). Most Trace runs never contain such a candidate.
2. **Why do those candidates win?** They look ahead: 0–8% of stock SciPy candidates are causal. Among strictly causal
   candidates the best scores are the same for both engines (stock 0.537–0.561, Trace 0.499–0.565).
3. **Why does stock keep them and Trace not?** Adoption works the same way in both: once a SciPy candidate beats the
   best so far, 42–96% of later parents use SciPy (stock 6/6 runs, Trace 3/3). The difference is the *first* SciPy
   candidate. Stock's beat the best so far in 6/6 runs. Trace produced one in 6/15 runs, and it beat the best in 3.
4. **Why are Trace's first SciPy candidates rarer and weaker?** These are ruled out by the logs: label rate (H1),
   label text (H2), LLM settings (H3) and parent/context selection (H4). What remains is the prompt Trace is shown
   (H6, H8) and program bloat (H7).
5. **Why would the prompts differ?** EXP25's whitebox evaluator returns its audit metrics (`causal_fraction`,
   `valid_score`, `fallback_signals`) with the score. The coevolution operator, faithfully to SkyDiscover, prints every
   numeric metric. Stock runs SkyDiscover's own evaluator, which computes none of them. So the harness, not the
   library, shows the engines different prompts. Every Trace parent reads `causal_fraction: 1.0000`.

## Hypotheses and tests

| ID | Hypothesis | Test | Status before Part B |
|---|---|---|---|
| H1 | Trace selects DIVERGE too rarely | Part A: label shares, stock labels matched by exact text | **Refuted.** Stock 15–24% of iterations (8–23% of successful ones), Trace 0–18% except two runs |
| H2 | Trace's label text lacks SciPy | EXP26 `native_stocklabels` | **Refuted** at equal budget (0.565 against 0.686) |
| H3 | Different LLM settings | Config read | **Refuted.** Same model, provider, temperature 0.7, 32k tokens, low reasoning |
| H4 | Parent/context selection fails to propagate SciPy | Part A: later-parent SciPy share; context sizes | **Refuted.** Same adoption once found; context about 3.8 programs in both |
| H5 | The gap is look-ahead only | Part A: causal re-evaluation | **Supported.** No causal gap |
| H6 | The `causal_fraction` cue steers Trace away from look-ahead | **Part B**: T0 vs T1 | open |
| H7 | Larger programs hinder rewrites | Part A: source size | weak: Trace 9–13k chars, stock 6–9.5k; not tested causally |
| H8 | Other prompt differences (audit metrics, evaluator feedback, shorter task text) | **Part B**: T1 vs T2 | open |
| H10 | Trace's own evolved parents (lineage, size) suppress SciPy discovery | **Part C**: stock vs Trace parent | open |
| H9 | Chance (3 seeds) | Part A: discovery count | **Refuted** as sole cause: winning SciPy discovery in 6/6 stock runs against 3/15 Trace runs |

## Part B — prompt ablation (`scripts/prompt_ablation.py`)

- **Prompts:** the first DIVERGE-labelled solution prompt with a non-SciPy parent, recorded in each of
  `native_stocklabels_s42/s43/s44` and `native_pkg_s43`. These are the runs that missed SciPy.
- **Variants:**
  - T0: as recorded.
  - T1: `causal_fraction` lines removed.
  - T2: T1, plus the other audit metrics and evaluator-feedback sections removed, and stock's task text.
- **Sampling:** 6 completions per prompt and variant, with the same model and settings as the campaigns; 72 calls.
  Jobs are shuffled across variants.
- **Scoring:** diffs are applied to the shown parent and scored by the EXP25 whitebox evaluator.

Endpoint: look-ahead share (`causal_fraction < 1`) among applied completions, pooled per variant.

- **H6 supported** if T1 − T0 ≥ 0.20 and the sign holds in ≥ 3 of 4 prompts. **Refuted** if T1 − T0 ≤ 0.05.
  Inconclusive otherwise.
- **H8:** the same rule on T2 − T1.
- Secondary measures (reported, not decisive): SciPy share, share of completions beating the parent's valid score,
  best valid score.

Limits: one task, one model, four prompts. A positive result shows prompt sensitivity, not a full-run effect.

## Candidate fixes and their triple check

| Fix | For | Check 1 (evidence) | Check 2 (mechanism) | Check 3 (before use) |
|---|---|---|---|---|
| F1 Score both engines under causality (hard constraint `causal_fraction >= 1`, reject) | H5 | No causal gap in 21 runs | `GuidedEvaluator(hard_constraints=..., violation='reject')` exists; stock needs the same filter on its evaluator | 1-seed smoke per engine, then a paired campaign with P2 primary |
| F2 Symmetric prompts: keep audit metrics out of LLM-visible metrics (audit them separately) | H6/H8 | Part B | Harness-only change: evaluator returns the score metrics, audit goes to the log | Unit test that prompts lack audit keys; paired runs |
| F3 DIVERGE quota | H1 | Refuted | — | Not pursued |
| F4 Inject stock labels / package-aware labels | H2 | Refuted (EXP26) | — | Keep `label_packages` off by default |
| F5 Bloat control | H7 | Weak | — | Only if a later test shows an effect |

## Part C — powered factorial (`scripts/factorial.py`), pre-registered 2026-10-06 before any call

Part B could not tell per-call SciPy discovery rates apart: with 4 prompts × 6 samples its power was under 0.5.
Part C was sized beforehand by simulation (`scripts/power.py`, prompt-clustered outcomes, prompt-bootstrap decision
rule). With 20 prompts × 10 samples per cell, power is 0.86–0.97 for each of these effects: base rate 3% with odds
ratio 3.6, 5% with odds ratio 2.5, or 10% with odds ratio 2.0. Prompt heterogeneity was simulated at sd 0.7 and 1.2.

- **Prompts:** 20 distinct recorded EXP26 native parents, up to 4 per run. Each is non-SciPy, causal (1.0) and valid
  on all 5 signals. Each prompt carries its run's DIVERGE label, fixed across cells.
- **Factor cue:** the `causal_fraction` lines present or removed. They are removed from the parent and from the
  context programs.
- **Factor parent:** the recorded Trace parent, or the stock EvoX program whose valid score is closest. The stock
  program must be non-SciPy, causal, all-valid and unused; the largest score gap is 0.007. Both parents are
  re-rendered by the same code from their own whitebox metrics and feedback.
- **5th cell (powered H8 re-test):** the stock-like prompt (T2 of Part B) on the Trace parent, compared with cue-off
  on the Trace parent.
- **Sampling and cost:** 10 samples per cell, 1,000 calls, same model and settings, jobs shuffled, 24 parallel workers.

**Primary endpoint:** SciPy share over all completions; a failed diff counts as no. Effects are the mean over prompts
of the per-prompt difference, with a 95% bootstrap CI over prompts.

- **Supported** if the estimate is ≥ 0.05 and the CI lower bound is > 0.
- **Refuted** if the CI lies inside [−0.05, 0.05].
- **Inconclusive** otherwise.
- Applied to:
  - H6: cue off − cue on.
  - H10 (parent lineage, new): stock parent − Trace parent.
  - H8: stock-like − cue off, on Trace parents.

The interaction is reported but not decided. Secondary measures: look-ahead SciPy share, beats-parent rate, applied
rate, best valid score.
