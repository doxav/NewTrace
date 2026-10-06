# EXP27 — results

**Complete.**

- Part A: logs only, 21 runs.
- Part B: 72 calls, $0.10, underpowered.
- Part C: powered 2×2 plus an H8 re-test, 1,000 calls, $1.46.
- Part D: 11 full Trace runs, cue hidden ×8 / shown ×3, $2.43, no call hit its deadline.
- Part E: exploration strategies compared, logs and transcripts only.

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


The estimate above does not survive Part E: in real runs, Trace's DIVERGE calls yield SciPy at 4.2%, not 12.5%.

## Part D — full runs with the cue hidden (`results/partD_20261006T195738/`, `analysis.json`)

Eleven 100-iteration Trace runs with EXP26 `native`'s exact plan (same plan fingerprint). Eight have `causal_fraction`
removed from every LLM-visible metric (seeds 42–49) and three keep it (seeds 45–47). The audit still records it.
No call hit the 600 s deadline. Prior cued runs: EXP25 trace_exp24 ×3 and EXP26 native ×3.

| Group | Runs | Best (median) | Best ≥ 0.65 | Winning look-ahead SciPy (E) | Climb after E | Best causal (median) |
|---|---:|---:|---:|---:|---:|---:|
| Stock EvoX | 6 | 0.711 | 5 | 6 | +0.045 | 0.546 |
| Trace, cue hidden | 8 | 0.638 | 3 (s44, s46 SciPy; s49 hand-written NumPy) | 2 | +0.050 | 0.539 |
| Trace, cue shown (3 new + 6 prior) | 9 | 0.586 | 3 | 3 | +0.056 (s45) | 0.53 |

- **Pre-registered reading: INCONCLUSIVE.** DISCOVERY needs E ≥ 5/8, and E is 2/8. CAPABILITY LIMIT needs a climb
  below half of stock's, and the climb is +0.050 against +0.045. Fisher test, cue off against cued: p = 0.82.
- **The capability-limit explanation is refuted by the data.** Once Trace finds the strong cheat, it climbs it as far
  as stock and ends level (0.73 against 0.71). Causal-only scores are equal across engines.
- **Trace cheats too, but weakly.** Ten of the 11 Part D runs end on a non-causal best program
  (`causal_fraction` < 1). So do 12 of the 15 archived Trace runs, where the best is above the best causal by
  more than 0.01.
  Most use centred hand-written windows worth +0.02–0.10. The zero-phase SciPy filters are worth about +0.15.
- **The cue matters at most a little.** Hiding it lifts the median best from 0.586 to 0.638 and makes the cheats
  larger, but it does not change how often the strong cheat is found. Part C's per-call effect is real but is not
  the run-level root cause. **The "root cause = cue" line of the previous version is withdrawn.**

## Part E — exploration strategies compared (log-only, exploratory, not pre-registered)

Scripts: `scripts/strategy_timeline.py` and `scripts/instruction_yield.py`. Outputs:
`results/strategy_timeline.json` and `results/instruction_yield.json`. The EXP25 stock label texts were recovered
from SkyDiscover's `search/iteration_1/labels.yaml`; Part A could not match them.

### What prompt the LLM actually receives

Each engine has two LLM levels. The per-iteration **solution prompt** (O0) is written by the engine from what the
evolved **selection policy** returns: a parent, context programs and an optional DIVERGE/REFINE instruction. The
**policy** is itself rewritten by an LLM (O1) when progress stalls. Both engines use the same stall trigger and the
same window score: `(best − start) × (1 + log(1 + start)) / √horizon`.

| | Stock EvoX | Trace (`proposer='trace'`, all Trace arms) |
|---|---|---|
| **O1 system prompt** | `evox_search_sys_prompt.txt`, 9.5k characters, a design brief. It says "parent labels MUST be empty string by default" (said 4 times) and "use labels only when appropriate … when progress is stagnating". It also says to choose labels from the search state, "not a fixed rule", to avoid deterministic selections, to treat diversity as a selection signal, and that a labelled call may return empty context | OptoPrimeV2's generic system prompt ("You're tasked to solve a coding/algorithm problem…"). The policy is a trainable `str` node whose constraint is the 9-line `POLICY_CONTRACT`. There is no design guidance; the empty label appears only as a code comment (`label_name in labels or ''`). `META_SYSTEM`, the compact port of the stock brief, is used only by `proposer='llm_rewrite'` |
| **O1 feedback** | problem summary, search window, population statistics | the same content, plus OptoPrime memory of earlier rounds |
| **O0 solution prompt** | SkyDiscover `diff_user_message.txt`: metrics, previous attempts, context, current program, task | the same layout, plus diagnostic metrics, `## Evaluator Feedback` and a `## projection` note, with 4 context programs on 98% of DIVERGE calls. The prompt grows from 51k characters (iteration 5) to 95k (iteration 60) |
| **DIVERGE text** | generated by stock's label prompt; in all 6 runs it names `butter`, `filtfilt` and `savgol` | generated by Trace's native label prompt; Kalman-first, sometimes `savgol`, never `filtfilt` |

Seen in the transcripts (`cue_off_s45`, first O1 call): the statistics summary says "every traced child has parent
label '[-]' (10 of 10)". OptoPrime answers "use a meaningful label so the selection signal is recorded" and writes
`label = "refine" if "refine" in self.labels else ""`. All 7 later rewrites keep unconditional refine. Their
reasoning cites the accumulated evidence ("refine chains from top-3 parents with elite contexts drive all gains;
diverge/random produced nothing"). Refine is the only thing tried, so it is the only thing credited, and the loop
reinforces itself. All 30 stock policies (EXP25 ×20, EXP26 ×10) keep an unlabelled path. The 10 EXP26 policies that were read
use `""` as the default and add REFINE/DIVERGE under stagnation conditions, sometimes with empty context.

### How the strategies evolve over time (mean share of producing iterations)

| Iterations | Stock: none / refine / diverge | Stock: parent = best | Trace: none / refine / diverge | Trace: parent = best |
|---|---|---:|---|---:|
| 1–25 | 85% / 10% / 5% | 18% | 68% / 23% / 9% | 20% |
| 26–50 | 33% / 33% / 34% | 34% | 9% / 70% / 20% | 22% |
| 51–75 | 27% / 35% / 38% | 14% | 9% / 72% / 19% | 19% |
| 76–100 | 22% / 38% / 41% (3 runs) | 25% | 6% / 75% / 19% | 22% |

Both start the same way, with a random policy and mostly no instruction. After the first rewrites, stock settles on
a balanced mix. Trace locks into refine and attaches DIVERGE about half as often. Parent elitism is similar in both
engines. Trace's elitism is in the context instead: top-k scorers in every call.

### Where SciPy look-ahead comes from

The table counts children of non-SciPy parents up to each run's first SciPy candidate:

| Instruction | Stock | Trace |
|---|---:|---:|
| none | 0 / 96 | 0 / 503 |
| refine | 0 / 6 | 2 / 962 |
| **diverge** | **6 / 8 (75%)** | **8 / 189 (4.2%)** |

- **Only DIVERGE produces the strong cheat.** No instruction and REFINE give about 0% in both engines. The
  "uninstructed iterations" hypothesis is refuted.
- **Stock found it on its first one or two DIVERGE calls in every run** (iterations 15–31).
- **Trace made 24× more DIVERGE calls before discovery, and each was about 18× less productive.** The gap is in what
  a DIVERGE call contains, not in how often DIVERGE is chosen.
- **The DIVERGE text is not the cause.** Trace with stock's filtfilt-naming labels (EXP26 `stocklabels`) yields
  1 / 33, Trace's Kalman labels 5 / 101, and runs without a label file 2 / 55.
- **The context is the main untested difference.** Stock DIVERGE calls with 0 context programs gave 3 / 3, with
  3–4 gave 3 / 5. Trace DIVERGE calls carry 4 top-scoring context programs (186 / 189 calls) plus 3 previous
  attempts. The parent is also later and larger: median iteration 49 against 23, and 10.3k characters against 7.5k.
  Part C held context and history fixed at Trace's, so it never tested them.
- **Part C's 12.5% under-DIVERGE rate is above the real 4.2%.** Part C sampled earlier, eligible parents. That
  difference is itself unexplained.

## Status of the root causes

| | Verdict |
|---|---|
| Gap = look-ahead loophole (H5) | **Established**: no causal gap across 26 Trace and 6 stock runs |
| Capability to exploit once found | **Refuted as a cause** (Part D): same climb (+0.050 vs +0.045), same end score |
| Causal cue (H6) | **Minor**: lowers per-call SciPy 1.7× (Part C) but not run-level discovery (Part D, p = 0.82) |
| DIVERGE rate (H1) | **Partly reinstated**: stock attaches DIVERGE about 2× as often after iteration 25 (correct label matching, EXP25 included). But Trace still made more DIVERGE calls before discovery, so rate alone does not explain the gap |
| DIVERGE label text (H2) | **Refuted**: stock text in Trace yields 1 / 33 |
| Uninstructed iterations | **Refuted**: 0 / 599 SciPy across both engines |
| Productivity of a DIVERGE call | **Main gap**: 75% vs 4.2%. Leading suspect: 4 elite contexts plus history anchoring the LLM to the hand-written NumPy family. Second suspect: later, larger parents. Untested |
| Policy locks into REFINE | **Established as a behaviour** of Trace's O1 (OptoPrime without the stock design brief). It cuts DIVERGE calls and never drops context, so it feeds the gap. Causal weight not yet measured |
| LLM settings (H3), chance (H9) | **Refuted** |

**In one line:** Trace does not lack optimizing power. It explores less radically. Its meta-optimizer writes
exploit-only selection policies (always refine, always the top-k programs as context), because nothing in its meta
prompt asks for exploration and its own evidence credits only what it tries. Its rare DIVERGE calls are buried
under elite context, while stock's evolved policies diverge from sparse context.

## What to explore with Trace, and at which level

1. **Meta / recursive-optimizer level (O1): first and cheapest.** Give `TraceProposer` the stock design brief.
   - Put the label and diversity rules into the policy node's description and constraint, or into the instruction:
     default `''`, labels only under stagnation, no fixed rule, avoid deterministic choices, diversity as a signal,
     empty context allowed for labelled calls.
   - Alternatively, run `proposer='llm_rewrite'` with a `META_SYSTEM` extended to the full brief.
   - Also stop the statistics summary from framing "labels unused" as a defect.
   - This is the change that restores stock-like policies over time.
2. **Solution-operator level (O0 contract): context on DIVERGE.** Let a labelled DIVERGE call return empty (or
   diverse, non-elite) context and skip the previous-attempts history. Test it first offline: replay real Trace
   DIVERGE prompts with 0 / 2 / 4 contexts, about 10 samples × 20 prompts per cell, about $0.40. This settles the
   leading suspect before any engine change.
3. **Trainer / scheduling level: last.** The window score rewards short-term gain in both engines, so it cannot
   explain the difference. Change it, for example with an exploration or diversity bonus, only if steps 1–2 fail.
4. **Measurement.** Score with causality enforced (F1) for both engines, so that more exploration is rewarded for
   legitimate gains, not for finding the loophole faster.
