# Pre-registration — W2 (amortisation) on the LLM4AD routing family

Written **before** running probes U, V, W. Probes S and T were run first and are reported as the
precondition; their results are fixed and are not re-analysed after seeing U/V/W.

## 0. Which win condition

**W2 (amortisation) only.** The surfaces are deterministic (`calls_llm=False`; replicate range
0.0 over 3 sequential re-scores of the same artifact on all 4 tasks). On a zero-noise surface W1
(variance / overfitting the target's noise) is structurally impossible, so "no quality difference
at equal budget" is predetermined and would carry zero information. It will not be reported as a
finding. The claim under test is:

> total evaluation compute to reach quality `Q` across `K` target tasks is lower for
> `c_meta + K·c_rec` than for `K·c_std`, with break-even `K* = c_meta / (c_std − c_rec)`.

## 1. Family and precondition (probes S, T — already run)

Probe S dumped the trainable entry point of every `llm4ad:optimization/*` task (excluding
`co_bench`). Result: `online_bin_packing` (`priority(item, bins)`) and `admissible_set`
(`priority(el, n, w)`) — the two tasks of Iteration 3 — have **different signatures, so the code
artifact cannot transfer between them at all**. Option (a) of the mission brief is therefore the
viable one, and the family is:

| task | entry | positional args |
|---|---|---|
| `tsp_construct` | `select_next_node` | `(current, destination, unvisited, D)` |
| `cvrp_construct` | `select_next_node` | `(current, depot, unvisited, rest_cap, demands, D)` |
| `ovrp_construct` | `select_next_node` | `(current, depot, unvisited, rest_cap, demands, D)` |
| `vrptw_construct` | `select_next_node` | `(current, depot, unvisited, rest_cap, time, demands, D, tw)` |

`unvisited` is always positional index 1 and `D` is always the last square 2-D array, so a single
`*args`-tolerant artifact text is byte-identically valid on all four. That is what makes a *code
artifact* transfer experiment possible; without it only `LevelConfig` knobs could transfer.

Probe T (menu of 9 ranking-distinct construction heuristics, executability controls
`raises`/`syntax_err`/`wrong_name` → `−1e6`):

- effective menu size (distinct valid scores): tsp 7, cvrp 9, ovrp 9, vrptw 9 — all > 1.
- score range: 29.03 / 25.70 / 26.25 / 15.08.
- replicate range on the best candidate, 3 sequential re-scores: **0.0** on all four.
- Spearman ρ of the candidate ranking between task pairs: 0.51 – 1.00, all six positive;
  **argmax is `nearest` on 4/4 tasks.** → a shared optimum exists.
- `cvrp` vs `ovrp` have ρ = 1.00: they are near-duplicates. Effective family size is 3, not 4.

`max_examples` is inert on these bundles (each has exactly 1 "example", which is the whole LLM4AD
evaluator; that evaluator internally averages 16 instances of size 50). All scoring uses the
benchmark's own instance average.

## 2. Probe U — Tier 1, exact W2 sweep (no LLM)

Score table `S[t][c]` for all 4 tasks × 9 menu candidates, measured once (deterministic).

- **standard(b)**: draw `b` of the 9 candidates uniformly at random without replacement, evaluate
  each on the target, keep the best. Reported as the **exact expectation over all `C(9,b)`
  subsets**, plus the full distribution — not a Monte-Carlo estimate, so there is no seed noise.
- **recursive(b)**: leave-one-out. Prior = the 9 candidates ordered by mean *per-source min-max
  normalised* score over the source tasks `T \ {target}`. Evaluate the first `b` in that order on
  the target, keep the best. Deterministic given the fold.
- **quality** is per-target min-max normalised to [0,1] so the four scales are comparable; raw
  scores are reported alongside.
- **`c_meta`** = evaluations spent on non-target tasks = `|M| × |sources|` (27 for the 4-task
  family, 18 for the de-duplicated 3-task family). A budgeted-meta variant with `m ∈ {3,5,9}`
  candidates per source is reported so `K*` is given as a function of `c_meta`.
- **Headline:** for `Q ∈ {1.00, 0.99, 0.95}`, `b_std(Q)` = smallest `b` with `E[q_norm] ≥ Q`,
  `b_rec(Q)` likewise; `K* = c_meta / (b_std − b_rec)`. The deliverable is the **curve** of
  `E[q_norm]` vs `b` for both arms, and the **horizontal** gap at fixed `Q`.
- **De-duplicated secondary analysis** on `{tsp, cvrp, vrptw}` (drop `ovrp`) is the headline
  robustness check, because a LOO fold whose source set contains the target's ρ = 1.00 twin
  inflates transfer.

## 3. Probe V — Tier 2, the production harness

`run_spec` with the routing family, `starting_artifact` target, the probe-T code menu, arms
`standard` (cold prior on the target level) vs `recursive` (`o2_policy` → `o3_prior`, priors
reused), both **scored on the same level `o3_prior` and the same task set**. Budgets
`num_candidates ∈ {1, 2, 4, 8}`, LOO over targets, ≥ 3 seeds. Records `artifacts_differ`,
`scored_task_ids`, and an in-run replicate of the identical initial artifact (n ≥ 10).

## 4. Probe W — Tier 2b, the cold-LLM control (the decisive confounder)

`c_std` in probe U assumes standard searches the menu **uninformed**. A real LLM optimizer given
the TSP task description would very plausibly propose nearest-neighbour on its first candidate, in
which case `c_std ≈ c_rec`, the saving vanishes and `K* → ∞`. Probe W measures this directly:
give the cold optimizer the target task and count how many candidates it takes to reach the
family optimum. This is pre-registered as the dominant threat to the Tier-1 result, not an
afterthought.

## 5. Kill conditions (pre-registered)

| id | condition | consequence |
|---|---|---|
| K1 | mean cross-task ρ ≤ 0, or family argmax disagrees across members | transfer impossible, `K* = ∞`, stop | 
| K2 | `b_rec(Q) ≥ b_std(Q)` at `Q = 1.00` | recursion does not amortise → **report null** |
| K3 | LOO prior's top-1 ≠ target's argmax on a majority of folds | transfer unreliable → no win claimed |
| K4 | probe V `artifacts_differ == False` | delta not attributable to the arms → **retract** |
| K5 | cold LLM optimizer reaches the family optimum within 1 candidate on a majority of targets | `c_std ≈ c_rec` on the production path, `K* → ∞`; the Tier-1 win **does not transfer to a real optimizer** and must be reported as such |

K1 has already been evaluated and passed (§1). K2–K5 are evaluated after the fact and reported
whichever way they fall.

## 6. Measurement rules held to

- both arms scored on the same level and the same task set;
- effective menu size verified > 1 per task, checked for the ranking-equivalence collapse mode as
  well as the type-incompatibility one;
- noise floor measured at the concurrency actually used (all probes run **sequentially**, one
  process; if any parallelism is introduced the floor is re-measured);
- in-run replicate control on the identical initial artifact reported beside every effect;
- `artifacts_differ` required for any harness-level claim;
- no win claimed inside the noise floor.
