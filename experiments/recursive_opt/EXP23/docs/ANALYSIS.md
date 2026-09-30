# EXP23: repairing the meta signal, traced selection, and which experiment to run next

Simulator-first follow-up to EXP22. Everything below the "live" line runs offline at
zero cost. Numbers are from `results/*.json`; commands are at the end.

## TL;DR (critical)

1. **The fixed score is correct but not sufficient.** The paired score (challenger interleaved with
   the incumbent on the live population) removes EvoX's stage bias. For two identical
   policies, EvoX's rule scores the later one −2.6 SD in PRISM at stage 0, while the paired score stays ≈ 0 at every stage.
   A single 20-call comparison still orders two clearly different policies correctly only
   62–75% of the time (EvoX: 60–69%). **One 100-call run cannot learn a policy with either score.**
2. **Online meta-optimization inside a 100-call run captures ≤ 30% of what the lever is worth.**
   In the PRISM simulator the best fixed parent policy is worth +1.5 to +4.6 SD over stock.
   Every online arm (EvoX, EXP22-Trace, paired EvoX, paired Trace) gets +0.2 to +1.3 SD.
   Deploying *anything* greedier than uniform explains most of that gain, not the score.
3. **Amortized meta-training with the real Trace trainer works.** PrioritySearch + OptoPrimeV2
   scores each candidate on whole 100-call episodes, paired against stock on the same seeds.
   It reaches **+1.2 to +3.8 SD in PRISM (70–100%+ of the oracle)** and correctly stays at stock in
   Signal, where no policy beats uniform (Signal's ceiling is at most +0.1 SD). This is the design to take live.
4. **Offline, "Trace vs EvoX" is a tie by construction.** The mock proposer ignores feedback.
   Arms with the same mechanics (`trace_paired` vs `evox_paired`) are equal within CI. `trace_exp22`
   edges out `evox_log` in all three PRISM worlds (+0.2 to +0.3 SD; the CIs overlap) because *continue from current* escapes the stock
   policy, while EvoX's biased "best parent" keeps returning to it. Whether traced, compact feedback lets a real LLM propose
   better policies is untested and is exactly what A3 measures (under $1).

## What the arms compare

| Level | Surface | Range | Method | Score / feedback |
|---|---|---|---|---|
| **Task** (all alternatives) | Solution program (live: EVOLVE-BLOCK code). Sim: a replay of 549 real EXP22 child−parent deltas, binned by parent score, with invalid rates and parent wear-out | 100 solution calls per run (A7: 300/1000) | EvoX solution loop: policy picks the parent; the LLM rewrites it; evaluate; insert. Population is never reset | Benchmark `combined_score`; best-so-far |
| Meta: EvoX (`evox_log`) | Full `EvolvedProgramDatabase` source (live); 4 knobs (sim) | Unbounded code / bounded knobs | Stagnation trigger (10), archive, parent = argmax window score, LLM rewrite, always deployed | LogWindowScorer (**stage-biased**) |
| Meta: EXP22 Trace (`trace_exp22`) | Same | Same | Fresh OptoPrimeV2 per trigger (memory 0), parent = current policy, identity graph | LogWindowScorer + raw JSON dump (sim 13–16k chars; live 286k) |
| Meta: paired EvoX (`evox_paired`) | Same | Same | Same trigger; challenger interleaved with the incumbent for 20 calls; promoted only if score > 0; parent = incumbent | Paired new-best rate (**stage-free**) |
| Meta: traced Trace (`trace_paired`) | Same | Same | One **persistent** OptoPrimeV2 (memory 5); **the solution calls execute inside `run_selection_window`**, so the graph holds the real decisions | Paired score + per-rank/per-reuse outcome table (~8k chars, bounded) |
| Meta: amortized trainer (A2) | Knobs or `select_parent` code | Across runs (train/validation/holdout seed splits) | **PrioritySearch** (UCB, archive, best-candidate exploration) + OptoPrimeV2 via the standard `trace` engine | Whole-episode paired gain vs stock (common random numbers) |

## Alternatives (pick by number)

Cost basis for live runs: EXP22 measured $1.26 for 559 requests, about **$0.24 per 100-call run** (glm-5.3-flash via Novita). Rate limits, not money, were the binding constraint.

| # | Question | Meta: surface / range / method | Task: surface / range / method | Budget, cost | Expected result (sim evidence) | Pros | Cons (critical) |
|---|---|---|---|---|---|---|---|
| **A0** | Does the sim reproduce EXP22? | knobs / in-run / `evox_log`, `trace_exp22` + fixed references | Sim replay / 100 / EvoX loop | 200 seeds × 6 worlds, $0 | **Done.** Online arms reach +0.4 to +1.3 SD in PRISM vs an oracle of +1.5 to +4.6; Signal ≈ 0 or slightly harmful (−0.1 to −0.26) | Free; paired seeds; cross-checks every EXP22 claim | The sim's children are independent given the parent; context programs and LLM memory aren't modelled; Signal levels are optimistic |
| **A1** | Does the fixed score rescue in-run meta-opt? | knobs or code / in-run / `evox_paired`, `trace_paired`, stagnation or periodic trigger | Same | $0 | **Done: no.** +0.17 to +0.59 SD in PRISM, *below* `evox_log` (too conservative with ~3 decisions per run). It does less harm in Signal (−0.05 vs −0.21 in W2; the CIs overlap). With ~30 decisions (h1000, patience 10) it matches or beats the biased arms | Removes the bias exactly; safe gate | Halves the calls per candidate; 3 decisions per run is too few to learn from |
| **A2** | Can a *real trainer* learn the policy across runs? | knobs or code / cross-run / **PrioritySearch + OptoPrimeV2** (standard `trace` engine) | Same | ≤288 episodes (24 iterations) or ≤96 (8 iterations), $0 | **Done: yes.** PRISM +3.8/+1.7/+1.7 SD at 24 iterations and +2.6/+1.6/+1.2 at 8; code surface +2.8 / +1.8; Signal 0 (correct) | Uses Trace's actual strengths: archive, UCB for noisy scores, validation, holdout | Meta-training runs cost solution calls; the learned policy is specific to one task and model |
| **A3** | Does Trace's traced, compact feedback help a *real LLM* propose better policies than blind proposals? | code / A1 and A2 settings / OptoPrimeV2 with a **live LLM** | Sim (solutions stay free) | ~90–400 optimizer calls, **< $1** | Unknown; this is the only offline-cheap test of Trace's differentiator | Cheapest real test of "Trace vs EvoX"; the solution level is noise-controlled | The LLM may just "discover greedy" from its prior; include a no-feedback LLM control |
| **A4** | EXP22-v2: stock kernel with the paired score, live | Full DB source or `select_parent` / in-run / SD-FIXED, SD-EVOX, SD-EVOX-PAIRED, TRACE-PAIRED | Live PRISM + Signal / 100 / stock | 8 seeds × 4 × 2 = 6,400 calls, **~$15** + rate limits | Small (sim ≤ +0.6 SD); **8 seeds only detect effects ≥ ~1 SD** | Closest to the paper setting | Probably another inconclusive run; I recommend **not** running it at 100 calls |
| **A5** | Does the A2 result transfer to live? | code / cross-run / PrioritySearch over live 100-call runs, then frozen | Live / 100 / stock EvoX loop with the learned policy | ≤96 training + 16 holdout runs, **~$27 per task** | PRISM: a positive effect ≥ 1 SD if the sim is right; Signal ≈ 0 | Directly tests the working design; holdout seeds | Most expensive; the policy may overfit training seeds; live noise is higher than simulated |
| **A6** | Is Trace competitive at the *task* level? (no meta) | none | Solution code / 100 / **Trace PrioritySearch + OptoPrimeV2** vs EvoX solution loop | 8 × 2 × 2 = 3,200 calls, **~$8** | Unknown | If Trace loses here, no meta layer can win; if it wins, "Trace vs EvoX" is settled more cheaply | Different mechanism, so it doesn't isolate meta-learning |
| **A7** | Does online meta pay off with more iterations? | knobs / in-run, horizon 300 or 1000 | Sim / 300–1000 | $0 | **Done.** EvoX-style patience (10% of horizon) keeps proposals at 2–7 whatever the horizon. With fixed patience 10 at h1000 (30–88 proposals), online arms reach +14 to +18 SD (PRISM) but stay far below the fixed top-5 policy (+36) | Explains the "too few iterations" failure directly | Horizons beyond the logged scores are extrapolation; trust the direction, not the magnitude |

**Recommendation:** A3 first (< $1; it decides whether Trace adds anything beyond the trainer design), then A5 on PRISM with A6 as the baseline (~$35 total). Skip A4 at 100 calls. Only revisit it at ≥ 1,000 calls with fixed patience, where the paired score clearly wins (Signal +20.6 vs +6.6/+9.9 SD).

## Evidence

**Score validation** (`scripts/validate_scores.py`: 1,000-seed ground truth, 200 trials per pair, clear pairs = true gap > 0.5 SD, equal cost of 20 calls):

| Design | PRISM W0/W1/W2 | Signal W0/W1/W2 | Stage bias (same policy, stage 0 → 60) |
|---|---|---|---|
| EvoX (incumbent first, LogWindowScorer) | 0.61 / 0.64 / 0.69 | 0.60 / 0.66 / 0.61 | −2.63 → −0.04 (PRISM W0) |
| Paired new-best (chosen) | 0.65 / 0.67 / 0.75 | 0.62 / 0.71 / 0.63 | ≈ 0 at all stages |
| Paired child percentile | 0.68 / 0.72 / 0.86 | **0.45 / 0.54 / 0.50** | ≈ 0; fails in Signal |
| Paired "child beats parent" | **0.31 / 0.31 / 0.31** | 0.52 / 0.50 / 0.53 | a plausible-looking score that is *systematically wrong* |

**Meta arms at 100 calls** (`scripts/run_meta.py`: gain vs stock on the same 200 seeds, in stock-SD units, ±95% CI):

| | PRISM W0 | PRISM W1 | PRISM W2 | Signal W0 | Signal W1 | Signal W2 |
|---|---|---|---|---|---|---|
| Oracle fixed policy (ceiling) | +4.62 | +2.42 | +1.47 | +0.11 | −0.04 | −0.07 |
| `evox_log` | +1.00 ±.28 | +0.47 ±.19 | +0.36 ±.19 | +0.02 | −0.10 | −0.21 |
| `trace_exp22` | +1.27 ±.32 | +0.69 ±.20 | +0.60 ±.19 | +0.08 | −0.03 | −0.26 |
| `evox_paired` | +0.54 ±.23 | +0.17 ±.17 | +0.22 ±.16 | +0.15 | +0.05 | −0.05 |
| `trace_paired` | +0.59 ±.22 | +0.27 ±.16 | +0.21 ±.16 | +0.10 | +0.06 | −0.08 |
| **A2 PrioritySearch, 24 iterations (knobs)** | **+3.80** | **+1.66** | **+1.73** | 0.00 | 0.00 | −0.07 |
| A2 PrioritySearch, 8 iterations (knobs) | +2.57 | +1.60 | +1.17 | 0.00 | 0.00 | −0.07 |

## Limitations (read before trusting a number)

- The simulator is fitted to 549 logged children from the EXP22 strict runs (`data/deltas.json`, produced by `EXP22/diagnosis/extract_deltas.py`). Children are independent given the parent's score bin, and parent wear-out is an assumption swept over W0 (none), W1 (×0.8 per reuse) and W2 (×0.5).
- The offline proposer is a feedback-blind random local/global mutator. Offline results measure score and selection mechanics, never LLM reasoning.
- Signal's replayed deltas include rare large jumps, so absolute Signal levels are optimistic. Its "no headroom" conclusion is consistent across all three worlds.
- A4–A6 are *declared* (specs compile; plans are persisted) but their engines are stubs that raise `NotImplementedError`. They need live SkyDiscover kernel work.

## Layout and commands

`src/world.py` replay world · `src/policies.py` knob/code surfaces + validator · `src/search.py` step, scores, summaries ·
`src/traced.py` traced selection module · `src/mock_llm.py` offline proposer · `src/meta.py` online arms ·
`src/control_plane.py` registrations + one raw spec per alternative · `configs/<Ak>/<arm>/` raw, normalized and resolved plans.

From `experiments/recursive_opt/EXP23` (Python from the EXP22 venv = the frozen Trace worktree):

```sh
PY="../EXP22/.venv/bin/python -I"
$PY -m unittest discover -s tests -v            # 12 offline tests
$PY scripts/compile_configs.py                  # compile all 22 specs -> configs/
$PY scripts/validate_scores.py                  # ~3 min
$PY scripts/run_meta.py --workers 16            # ~10 min, A0/A1/A2/A7
$PY scripts/sensitivity.py                      # many meta-steps; cheaper amortization
$PY scripts/run_spec.py A2/priority_search-code --holdout 20   # any spec via execute_plan
# A3 (live, paid): export OPENROUTER_API_KEY per AGENTS.md, then
$PY scripts/run_spec.py A3/live_llm-trace_paired-stagnation --live
```

The live profile uses EXP22's `z-ai/glm-5.3-flash` for comparability. The repository's AGENTS.md
names `deepseek/deepseek-v4-flash-0731` as the Experiment-0 default, so choose one explicitly before any paid run.

## Live (1+2)×4 schedule — harness ready, live run pending a valid key

`scripts/live_schedule.py` runs every simulator alternative over 3 blocks of 4 steps: a stock
block, then 2 × (policy update, block), for 12 solution steps and 2 optimizer calls per run. One worker process runs
per (task, arm, seed). Arms: `evox_log`, `trace_exp22` (A0), `evox_paired`, `trace_paired` (A1),
`priority_search` (A2: 3 trainer steps = 2 proposals over 4-step episodes), and `llm_blind` (the A3 control:
same LLM, no feedback). Twelve steps cannot rank arms, so every proposed policy is also scored offline
(200 seeds × 100 steps, paired vs stock): `policy_value_sd` measures what the LLM proposed.

- `--mode mock` passes: 24 workers, exactly 12 steps and 2 proposals per run (test `ScheduleTests`).
  No paired challenger was promoted: 2 challenger steps per block almost never yield a new best.
- `--mode live --provider {deepinfra,novita,inference-net}` pins `z-ai/glm-5.3-flash` with
  `allow_fallbacks: false` and logs latency, served provider, tokens, and cost per call (never prompts or keys).
  Expected size: ~72 calls at 3 seeds, prompts ≤ 9k chars, well under $0.20.
- Not run: the local key source returned HTTP 401 (expired). A4–A6 still need the live SkyDiscover
  kernels. A7 cannot be tested in 12 steps.

## EvoX as a meta-optimizer: source-level analysis (2026-09-29)

Verified against SkyDiscover `3f7a611` and this repository's Trace, and cross-checked with EXP22 logs.
Paths are relative to `skydiscover/optimize/`.

### EvoX meta level, mapped to Trace concepts

| Trace concept | EvoX equivalent | Verified detail |
|---|---|---|
| Loop (Trainer) | `CoEvolutionController.run_discovery` (`search/evox/controller.py:153`) | One solution iteration at a time. Trigger: stagnation (`:252`), i.e. no best-score gain > 0.01 absolute (`:48`) for `switch_interval` = 10% of the horizon (`:47`; 10 at 100 iterations). On trigger: score the pending policy, archive it, make **one** proposal (`max_iterations=1`, up to 3 attempts with validator errors fed back), then **hot-swap** the solution database. The population is migrated, not reset (`:482`); a database that fails at runtime is rolled back (`:535`). Online, not episodic. |
| Archive / parent selection | `SearchStrategyDatabase.sample` | Parent = highest-scoring strategy; context = 2 other random strategies (`search/evox/config/search.yaml`). No UCB, no explicit exploration. |
| Guide (score) | `LogWindowScorer` + strategy validator | Best-score gain over the window × (1 + log(1 + start)) / √`switch_interval`. The divisor is constant, not the real window length (`controller.py:646`). Scoring is deferred to the next trigger. The validator only gates validity (score 0). |
| Guide (text feedback) | A **guide LLM** + formatters (`context_builder/evox/builder.py`) | Up to 3 extra LLM calls per proposal. They summarize population state, the problem and scoring formula (read from evaluator code), and the 2 context strategies (code + start/end window stats → signals / what helped / insight). Comparable to an LLM-judge Guide that writes the feedback. |
| Trace of past examples | `get_statistics().recent_solution_stats.execution_trace` (`search/base_database.py`) | **Per decision**: iteration, child score, parent (label, id, score), contexts (label, id, score); plus score trajectory, stagnation count and reuse ratios. Filtered to the last `horizon` entries. |
| Optimizer memory | The strategy archive | Stateless LLM calls; one "score improved/declined" line versus the previous strategy. |
| Surface | The whole EVOLVE block: `EvolvedProgramDatabase` (`add`, `sample`, arbitrary state) | Full rewrite (no diff), ≤ 60,000 chars. Arbitrary Python under a validator contract (class name, inheritance, `sample` signature and return types, metric preservation, migration). |
| Levers of the surface | 4 levers | (1) Parent choice. (2) Context programs, which and how many (default 4, `config.py:590`). (3) **Variation label**: DIVERGE/REFINE text, generated once by an LLM from the task and evaluator, **injected into the solution prompt**. (4) Population management (`add`, state, Pareto front). |
| Horizon | Stagnation window | At least `switch_interval` iterations, extended while improving; 3–5 proposals per 100-iteration run in practice. |

### The optimizer EvoX meta-optimizes, compared with OptoPrimeV2 and OPRO

| Aspect | SkyDiscover solution operator (driven by the EvoX policy) | OptoPrimeV2 | OPRO / OPROv2 |
|---|---|---|---|
| Input | Parent program + score + metric breakdown + evaluator artifacts, top-3 previous attempts (`context_builder/default/builder.py:400`), context programs (code + scores), score trend, label | Execution graph: #Code, #Variables, #Inputs, #Others, #Outputs, #Feedback | Past (variables, feedback) pairs |
| Output | SEARCH/REPLACE diffs (PRISM: `diff_based_generation: true`) or full rewrite | Full new value per variable (XML) | Full new value |
| History | Population; **the policy chooses** what enters the prompt | FIFO memory of (variables, feedback), **size 0 by default** (`optimizers/optoprime_v2.py:436`) | OPRO: unbounded; OPROv2: FIFO of 5 |
| Population / selection | Yes, programmable: this is what is meta-optimized | No (the trainer, e.g. PrioritySearch, owns the archive) | No |
| Diversity instruction | DIVERGE/REFINE labels | None | None |
| Repair | 3 attempts with errors fed back | 1 call per step | 1 call |
| What the meta level changes | The sampling and prompt-context assembly policy. Not the templates, system prompt or temperature. | `recursive_opt` O1 = `LevelConfig` (`levels.py:71`): a discrete choice of optimizer, trainer, guide, batch, memory and trace settings, **evaluated by rerunning the whole inner optimization** (`MetaLevel`, `levels.py:291`); `CodeArtifactLevel` can invent code. | Same |

EvoX and `recursive_opt` optimize different objects. EvoX rewrites, online, the equivalent of a trainer's
selection code (PrioritySearch's exploit/explore) plus the prompt composition of a fixed LLM operator.
Trace O1 chooses, episodically, which components and hyperparameters to use. Neither optimizes
OptoPrimeV2's own prompt.

### Corrections to earlier claims

| Earlier claim | Status | Evidence |
|---|---|---|
| The meta score is biased by timing and often zero; EvoX reuses the best-scored parent | Correct | `controller.py:646`, `search_strategy_db.py`; real windows 5.3 → 0 → 0 |
| EXP22 meta-optimized EvoX, not Trace | Correct | Single node and identity graph (`EXP22/src/kernel.py`) |
| EvoX makes a proposal in one call | Incomplete | Up to 3 attempts with errors (`run_discovery(retry_times=3)`); EXP22 Trace had one attempt |
| Traced per-decision evidence is new relative to EvoX | **Wrong** | EvoX already feeds a per-decision trace, summarized by an LLM. EXP22 Trace had the same raw data, undigested, in a 286,000-char dump. |
| The policy only chooses the parent (simulator; EXP23 `select_parent` surface) | **Wrong** | It also chooses the context and the label (see below) |
| Guide-LLM summaries run in real EvoX runs | Inferred | EvoX run 1 logs 10 calls under an unmapped role (`trace_meta_or_preflight`), consistent with ~2 summaries per proposal; not proven directly |

**Labels.** All 3 jumps in the best EvoX run (X = 29.91) came from labelled children: 22.61→26.75
(DIVERGE, iteration 14), then →27.15 and →29.91 (REFINE, iterations 38 and 43). All 5 evolved policies in that run use
labels; the stock policy never does. No Trace run ever used a label: EXP22 used none (0 of 5 policies, 0 of 79 children), and
EXP23 excluded them by construction, including TRACE-CURRENT and TRACE-PAIRED. The simulator and both EXP23 Trace arms therefore
removed the lever EvoX evidently exploited, so the comparison with X was not fair. Causality is not established: there is
one run, and the second EvoX run used 15 labels without any gain.

### Can the control plane v2 express EvoX?

Not by composing standard levels; only as an opaque custom engine (what EXP22 and EXP23 do). `execute_plan` runs levels
**sequentially, each to completion**, and passes only final outputs through `bindings`. EvoX needs four things this model
lacks: a meta level that acts **during** the lower level's run on a trigger; **shared mutable state** between levels (the
population); **deferred, online evaluation** over a non-stationary window; and hot-swapping with rollback. The native O1
is episodic (each candidate reruns O0), which is a different algorithm (the amortized A2/A5).

Transfers directly: the proposer (OptoPrimeV2/OPRO with memory); the archive and best-parent choice (PrioritySearch memory,
once a deferred-scoring adapter exists); the guide LLM (the existing `llm_roles.feedback` role); validated code surfaces;
and the knowledge store, for learning across runs.

Missing for a true v2 equivalent:
1. A co-evolution execution mode: an engine that runs the inner level step by step and exposes a trigger hook to the
   upper level, with declarative `trigger`, `window_scorer`, `archive_selection`, `meta_optimizer` and `feedback_composer`.
2. A deferred-evaluation contract.
3. Declared shared state between levels.
4. Hot-swap with rollback.
5. At O0, a population-conditioned operator (context programs, diffs, retry with errors, **labels**). None of this exists in
   OptoPrimeV2/PrioritySearch; it must be written, or SkyDiscover's operator registered as an O0 engine.
