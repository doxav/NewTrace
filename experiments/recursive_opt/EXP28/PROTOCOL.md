# EXP28 — protocol: giving Trace EvoX-style exploration (periodic DIVERGE + combine)

Pre-registered 2026-10-07, before any paid call.

## Motivation (EXP27)

- SciPy filters (the strong look-ahead family) come almost only from DIVERGE calls: EvoX 6/8, Trace 8/189.
- Trace's meta level (OptoPrimeV2 with a generic instruction) writes policies that attach REFINE about 72% of the
  time and always show 4 elite programs as context. EvoX's 9.5k-character brief says "no label by default, labels
  under stagnation, avoid fixed rules, value diversity, empty context allowed".
- Trace's trainers choose *where* to search (PrioritySearch, UCB, Pareto, POLCA, GEPA) but not *how* to mutate:
  OptoPrime's instruction asks neither for a local refinement nor for a fundamentally different approach.

## Alternatives considered

Columns: **E** = expected gain on exploration, **Cost** = extra LLM calls per step, **Types** = works for
code / text / numeric / categorical parameters.

### Trainer level (a new Trainer derived from an existing one)

| # | Alternative | E | Cost | Types | Pros | Cons / limits |
|---|---|---|---|---|---|---|
| **T1** | **VariationSearch, stagnation schedule** (PrioritySearch + per-step mode: DIVERGE after *p* stalled steps, REFINE for 2 steps after a gain, free otherwise) | high | 0 | all (instruction worded per type) | EvoX's exact mutation-intent axis; no extra calls; any optimizer with an `objective` | Fixed rule; *p* to tune; one parent at a time |
| T1′ | VariationSearch, bandit over modes (UCB on per-mode improvement yield) | high | 0 | all | Learns the mix per task | Few rewards in 100 steps; cold start favours early winners (REFINE), the trap seen in EXP27 |
| **T2** | **VariationSearch, periodic DIVERGE + COMBINE** (every 3rd step, 2 random non-elite memory candidates shown as inspirations) | high | 0 | all (values rendered as text) | EvoX "inspirations" + periodic divergence; semantic recombination, unlike GEPA's per-parameter merge | Longer prompts on combine steps; random inspirations may be weak early |
| T2′ | Combine with diverse inspirations (embedding / Pareto distance, POLCA-style) | high | embedding calls | all | Inspirations chosen for diversity | Needs an embedding API (POLCA's Gemini default); POLCA's `set_context` hook is missing on this branch |
| T3 | POLCA + reject-and-diverge (near-duplicate proposals retried with DIVERGE) | medium | retries | all | Novelty enforced after generation | Embedding distance ≠ algorithmic novelty; extra calls |
| T4 | GEPA-style merge | low here | 0 | multi-parameter | Exists (examples/trainers) | Uniform per-parameter merge with one code parameter just picks parent A or B |
| T5 | Portfolio: 2 proposals per step (REFINE + DIVERGE) | medium | ×2 | all | No schedule to tune | Halves the steps at equal budget |
| T6 | Island restarts from the initial program with DIVERGE | medium | 0 | all | Escapes deep lineages | Discards accumulated tuning |
| T7 | `score_function='ucb'` / StreamingPrioritySearch `exploration_ratio` | low | 0 | all | Already implemented | Changes *where* to search only; same edit intent |

### recursive_opt level (control plane `coevolution` engine)

| # | Alternative | E | Cost | Types | Pros | Cons / limits |
|---|---|---|---|---|---|---|
| **R1** | **EvoX policy brief for Trace's meta-optimizer** (`CoevolutionConfig.meta_brief` = condensed EvoX rules as TraceProposer's `#Instruction`) | high | 0 | policy is code; solutions any | Fixes the cause seen in transcripts (no guidance → REFINE-lock); keeps recursion; the brief is generic | The LLM may still ignore it; effect only after the first rewrite (~iteration 12) |
| R1′ | Full 9.5k EvoX system prompt verbatim, or `proposer='llm_rewrite'` | high | 0 | same | Maximal fidelity | Not Trace-as-meta (llm_rewrite); verbatim text names stock classes |
| **R2** | **Engine diverge guard** (`CoevolutionConfig.diverge_guard`: after 5 iterations without improvement the next selection becomes DIVERGE with no context, whatever the policy says) | high | 0 | labels are text, any solution type | Guarantees periodic sparse-context DIVERGE; policy-independent, so it also protects against bad learned policies | Overrides the learned policy, which partly bypasses recursion; fixed patience |
| R2′ | DIVERGE quota (≥ 1/3 per window) with 2 diverse contexts + combine instruction | high | 0 | same | Matches EvoX's observed mix (34–41% DIVERGE) | Rigid; elite contexts may anchor again |
| R3 | Per-label yield statistics in the O1 feedback | medium | 0 | same | Evidence-driven | REFINE is the only label tried early, so the evidence reinforces it (the EXP27 loop) |
| R4 | Neutral statistics framing (no "labels unused" pattern) | low | 0 | same | Tiny change | Removes one trigger only |
| R5 | Warm-start with an EvoX-evolved policy | medium | 0 | same | Strong prior | Copies EvoX's output, not its mechanism |

## Selection (two per level, each a different mechanism)

- **Trainer:** T1 (`vs_stagnation`) and T2 (`vs_combine`). Both are one new class, `opto.trainer.algorithms.VariationSearch`,
  with different schedules. They are the only options that add EvoX's missing mutation-intent axis at no extra call cost,
  and T2 adds recombination.
- **recursive_opt:** R1 (`brief`) fixes the meta-level cause, and R2 (`guard`) adds an engine-level guarantee. Both are new
  optional fields that leave existing plan fingerprints unchanged (tested).
- **Not run:** the primed variants (T1′, T2′, R1′, R2′) are design alternatives. T1′ and R3 risk the REFINE trap
  documented in EXP27; T2′ needs an embedding service; R1′ is not Trace-as-meta.

## Runs

- **Arms:** `vs_stagnation`, `vs_combine`, `brief`, `guard`. Seeds 42, 43 and 44 for each, giving 12 runs, all
  concurrent.
- **Budget:** 100 solution calls per run (trainer: `--steps 101`, since the first step only initializes).
- **Settings shared with EXP25–27:** `z-ai/glm-5.3-flash` via Novita, temperature 0.7, 32,000 max tokens, reasoning
  low. Same white-box evaluator, projections and feedback. `causal_fraction` hidden (EXP27 Part D condition, as EvoX).
  Deadline 600 s per call, 4 attempts.
- **References, not re-run:** stock EvoX (6 runs), Trace with the cue hidden (EXP27 Part D, 8 runs, the same
  coevolution configuration as `brief` and `guard`), and Trace with the cue shown (18 runs).
- **Caveat on the trainer arms:** they differ from the coevolution runs in more than exploration. They use full-program
  OptoPrimeV2 rewrites (not SEARCH/REPLACE diffs), one lineage via PrioritySearch, and no meta level. Compare them with
  each other and only descriptively with the references. No plain-PrioritySearch Signal baseline exists, and the
  request was not to re-run baselines.

## Endpoints and reading (n = 3 per arm; descriptive)

- **Primary:** the best benchmark (valid) score so far, by solution call. It is plotted as figure (a) of
  `key_findings.ipynb`: median and top run per arm, beside EvoX and both Trace references.
- **Secondary:**
  - look-ahead discovery (a non-causal candidate at or above 0.615, the notebook's threshold) and its call index;
  - best causal score;
  - share of DIVERGE/combine calls that introduce a SciPy filter;
  - the realized mode or label mix.
- **Reading:** an arm "closes the exploration gap" if at least 2/3 runs reach look-ahead at or above 0.615 within 50
  calls (EvoX: 5/6 runs by call 29). An arm "improves legitimate search" only if its median best causal score is above
  the references' (0.539 cue hidden, 0.543 EvoX) by more than 0.02.
- **Caveat on what exploration finds:** on this benchmark, more exploration mainly finds the look-ahead loophole faster.
  Look-ahead discovery measures exploration, not legitimate progress, and the causal endpoint guards that distinction.

## Part B — inspiration ablation and leak fix (pre-registered 2026-10-07, before any paid call)

**Bug found while preparing Part B.** In the Part A trainer arms, the instruction leaked. A candidate created on an
exploration step kept a deep copy of the optimizer with that step's instruction attached. Whenever the candidate was
expanded later, its "free" calls still carried the instruction, sometimes several stacked. Leaked free calls per run:

| | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| stagnation | 60/81 | 52/81 | 0/81 |
| combine | 57/67 | 63/67 | 65/67 |

The fix records each optimizer's base instruction once, rebuilds the instruction from it at every step, and resets
the optimizers of new candidates. A regression test fails on the old code and passes now. The Part A trainer results
therefore describe a contaminated treatment; Part B re-runs them.

**New options:** `inspiration_mode` (`never` (default) / `always` / `alternate`) and `inspiration_style`
(`combine` (default) / `context`, the EvoX-like plain context). The defaults are unchanged: stagnation schedule,
patience 5, plain DIVERGE.

**Arms** (3 seeds: 42, 43, 44; 100 solution calls each; 18 runs, all concurrent):

| Arm | Schedule | Inspirations | Purpose |
|---|---|---|---|
| `vs_default` | stagnation | never | the winner, re-run with the fix (regression check) |
| `vs_stag_alt_combine` | stagnation | alternate, combine | does combine help when alternated? |
| `vs_stag_alt_context` | stagnation | alternate, context | EvoX-like inspirations, alternated |
| `vs_stag_always_context` | stagnation | always, context | EvoX-like inspirations on every exploration step |
| `vs_periodic_diverge` | periodic (3) | never | separates the schedule from combine (Part A confound) |
| `vs_combine` | periodic (3) | always, combine | Part A's combine arm, re-run with the fix |

**Endpoints:** as in Part A, plus the realized instruction per call, checked so that free calls carry no instruction.

**Reading (descriptive, n = 3):**
- *Inspirations:* compare each stagnation inspiration arm with `vs_default`, and `vs_combine` with
  `vs_periodic_diverge`.
- *Schedule:* compare `vs_default` with `vs_periodic_diverge`.
- *Regression check:* "performance maintained" means `vs_default`'s median best benchmark score is within 0.05 of
  Part A's 0.748, and at least 2/3 runs reach look-ahead within 50 calls. Part A was contaminated, so a difference
  will be attributed to the fix, not to noise alone.
