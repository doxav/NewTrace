# Results — W2 (amortisation) on the LLM4AD routing family

Pre-registration: `PREREG_W2_routing.md` (written before probes U, U2, V, W, X were run).
Raw data: `probe_s_signatures.json`, `probe_t_routing_menu.json`, `probe_u_w2_sweep.json`,
`probe_u2_default_baseline.json`, `probe_v0_replicate_c1.json`, `probe_w_cold_llm.json`,
`probe_x_llm_w2.json`, `probe_v_results.json`.

**This experiment tests W2 and only W2.** All four surfaces are deterministic; W1 is structurally
impossible on them, so "no quality difference at equal budget" is predetermined there and is not
reported as a finding.

---

## 1. Precondition — is there a family that can transfer at all? (probes S, T)

### 1.1 The two Iteration-3 tasks cannot transfer to each other, by construction

`probe_s_signatures.py` dumps the trainable entry point of every `llm4ad:optimization/*` task
(co_bench excluded). `online_bin_packing` trains `priority(item, bins)`; `admissible_set` trains
`priority(el, n, w)`. **Different arity — the code artifact cannot execute on the other task at
all.** Any transfer experiment on that pair is testing nothing. Mission option (a) is therefore
the only viable one.

### 1.2 The routing family

Four tasks train the *same entry point* `select_next_node`, invoked positionally:

| task | positional args after `current_node` |
|---|---|
| `tsp_construct` | `destination, unvisited, D` |
| `cvrp_construct` | `depot, unvisited, rest_cap, demands, D` |
| `ovrp_construct` | `depot, unvisited, rest_cap, demands, D` |
| `vrptw_construct` | `depot, unvisited, rest_cap, time, demands, D, time_windows` |

`unvisited` is always positional index 1 and `D` is always the last square 2-D array, so a
`*args`-tolerant artifact is byte-identically valid on all four. Note this is a property of a
*hand-written* artifact, not of arbitrary code — §4 measures what happens without it.

### 1.3 Menu validity and shared optimum (probe T)

Nine ranking-distinct construction heuristics (no monotone transforms of one another — the
collapse mode that made the Iteration-3 menu effectively size 1), plus executability controls.

| task | distinct valid scores | range | replicate range (3×) | controls |
|---|---:|---:|---:|---|
| `tsp_construct` | 7 | 29.03 | 0.0 | `raises`/`syntax_err`/`wrong_name` → −1e6 |
| `cvrp_construct` | 9 | 25.70 | 0.0 | same |
| `ovrp_construct` | 9 | 26.25 | 0.0 | same |
| `vrptw_construct` | 9 | 15.08 | 0.0 | same |

Spearman ρ of the candidate ranking between task pairs: **0.51 – 1.00, all six positive**, and
**`nearest` is the argmax on 4/4 tasks**. A shared optimum exists → K1 does not fire.

`cvrp` vs `ovrp` have ρ = 1.00 — near-duplicates. **Effective family size is 3, not 4**, and the
de-duplicated `{tsp, cvrp, vrptw}` analysis is the headline everywhere below.

`max_examples` is inert on these bundles (each holds exactly one "example", which is the whole
LLM4AD evaluator; that evaluator internally averages 16 instances of problem size 50).

---

## 2. Noise floor and replicate controls

| measurement | n | range |
|---|---:|---:|
| probe T, best candidate re-scored, sequential | 3 per task | **0.0** |
| probe U, *every* (task × candidate) re-scored, sequential | 10 × 36 = 360 | **0.0** |
| probe V0, production harness, identical initial artifact, budget 1 (0 optimizer LLM calls), sequential | 10 per task | **0.0** (sd 0.0, 1 distinct artifact) |
| probe X, `nearest` re-scored at concurrency 6 | 10 per task | see §4 |

The evaluator is bit-deterministic on this family. Every effect below is therefore outside the
noise floor by construction; the risk in this experiment is *bias*, not noise.

---

## 3. Tier 1 — the W2 sweep with an uninformed searcher (probes U, U2)

`standard(b)` = `b` of the 9 candidates drawn uniformly without replacement, best kept, reported
as the **exact expectation over all C(9,b) subsets** (no Monte-Carlo error, no seed noise).
`recursive(b)` = leave-one-out; the menu ordered by mean per-source min-max-normalised score,
first `b` evaluated. Quality is per-target min-max normalised.

### 3.1 Menu-only (probe U), family `{tsp, cvrp, vrptw}`

| b | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| standard E[q] | 0.569 | 0.772 | 0.863 | 0.913 | 0.943 | 0.963 | 0.978 | 0.990 | 1.000 |
| recursive q | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| standard P(optimum) | 0.148 | 0.287 | 0.417 | 0.537 | 0.648 | 0.750 | 0.843 | 0.926 | 1.000 |

### 3.2 With the bundle default as the common starting point (probe U2)

Probe U ignores that every real run starts from the task's own template heuristic. That baseline
is **not neutral**: `tsp_construct`'s evaluator hands the heuristic a *pre-sorted*
`unvisited_near_nodes` array, so the template's `unvisited_nodes[0]` already *is*
nearest-neighbour and the default is already the menu optimum (default q_norm: tsp 1.000,
cvrp 0.945, vrptw 0.602).

| b | 0 | 1 | 2 | 3 | 4 | 6 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| standard E[q] | 0.849 | 0.887 | 0.915 | 0.936 | 0.952 | 0.975 | 0.992 | 1.000 |
| recursive q | 0.849 | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### 3.3 Break-even (headline: horizontal gap, not vertical)

`c_meta` is measured as evaluations spent on non-target tasks: `menu_size × |sources|`.

| analysis | Q | b_std | b_rec | saved/task | c_meta | **K\*** |
|---|---|---:|---:|---:|---:|---:|
| U2 (default baseline), full meta | 1.00 | 9 | 1 | 8 | 18 | **2.25** |
| U2 | 0.99 | 8 | 1 | 7 | 18 | **2.57** |
| U2 | 0.95 | 4 | 1 | 3 | 18 | **6.00** |
| U (menu only), meta_m = 3 | 1.00 | 9 | 1 | 8 | 6 | **0.75** |
| U, meta_m = 5 | 1.00 | 9 | 1 | 8 | 10 | **1.25** |
| U, meta_m = 9 | 1.00 | 9 | 1 | 8 | 18 | **2.25** |

The prior's top-1 equals the target's argmax on **3/3 folds** (4/4 including `ovrp`) → K3 does
not fire. K2 does not fire (`b_rec` < `b_std` at every Q).

**Read literally: against an uninformed searcher, one meta pass over 2 sibling tasks pays for
itself after 3–6 target tasks.** Whether "uninformed" describes any optimizer anyone would
actually run is §4, and it is the whole ball game.
