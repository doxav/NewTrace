# EXP25 — protocol (written before the runs, 2026-09-30)

## Question

On Signal Processing (no known optimum), within 100 solution calls: how do the best Trace configuration from EXP24,
the Trace configuration that produced EXP23's PRISM record, and stock SkyDiscover EvoX compare?

## Metric facts established before the runs (`scripts/probe_signal_metric.py`, `results/analysis/signal_metric_probe.json`)

The stock score skips failed signals and accepts any non-empty output length. A hand-written program returning only 5
samples scores **0.799** (initial program 0.499; EXP22 runs 0.54–0.61; SkyDiscover reports 0.72–0.76). A non-causal
smoother scores 0.514. The **valid score** (`signal/whitebox.py`) recomputes the stock aggregation over all 5 signals,
counting a signal only when its output has the documented length `len(x) - 20 + 1` and finite values; an invalid or
failed signal contributes the initial program's per-signal metrics. Causality (no look-ahead) is measured on every
valid signal and reported, not enforced.

## Arms (3 seeds each: 42, 43, 44; one worker per run; nine runs in parallel)

| Arm | Engine | Guide | O0 extras |
|---|---|---|---|
| `trace_exp24` | native coevolution, Trace proposer (`evox_preset` + `proposer: trace`) | valid score | per-signal feedback, compile check + per-signal fallback projection |
| `trace_exp23` | same engine and proposer | stock score | none (EXP23 configuration) |
| `evox_stock` | stock SkyDiscover `CoEvolutionController` (EXP22 audited kernel) | stock score, stock cascade evaluator | stock |

Common: `z-ai/glm-5.3-flash` via OpenRouter, Novita only, reasoning low, temperature 0.7; exactly 100 solution attempts
(native `strict_budget`; stock controller caps retries by the remaining budget); transient transport errors retried up
to one hour per call. Native arms evaluate with the stock-exact per-signal evaluator (no stage-1 cascade); the stock
arm keeps SkyDiscover's cascade.

## Endpoints

- Primary: best valid score among all candidates evaluated in the run (every arm re-scored with the same evaluator).
- Secondary: valid score of the program each run returns (its own best); best stock score and whether it is valid;
  causal fraction of the returned program; best-valid-so-far curve; wasted attempts; policy switches; provider cost.

## Limits

Three runs per arm; no significance test. The five signals are both feedback and score (no holdout). The EXP24 arm
changes guide, feedback and projection together. The stock arm differs from native arms in evaluator plumbing
(cascade) and prompt wording, as in EXP22/EXP23.
