# EXP26 — protocol (written before the runs, 2026-10-06)

## Question

On Signal Processing, does prompt fidelity in label generation close EXP25's gap between stock EvoX and
native Trace coevolution? EXP25 found SciPy in ~70% of EvoX candidates and 1 of 498 Trace candidates,
with two unseparated mechanisms: label **content** (M1) and label **selection** (M2). Rationale and
evidence for every choice below: [docs/DESIGN_ANALYSIS.md](docs/DESIGN_ANALYSIS.md).

## Arms (3 seeds each: 42, 43, 44)

| Arm | Engine | Labels |
|---|---|---|
| `native` | native coevolution, EXP25 `trace_exp24` configuration | native generator (as EXP25) |
| `native_pkg` | same | native generator + `label_packages` from stock's own `get_available_packages()` |
| `native_stocklabels` | same | produced by stock's own `_generate_variation_operators()`, one draw per seed, injected via `labels` |
| `evox_stock` | stock SkyDiscover `CoEvolutionController` (EXP22 audited kernel), unchanged | stock; now persisted |

The EXP25 `trace_exp24` configuration is held fixed for every native arm: valid-score guide, per-signal
white-box feedback, compile check and per-signal fallback projection, Trace proposer. Only the label
source varies.

Stock labels for `native_stocklabels` are generated before the campaign by calling the stock controller's
own probe and generator (`scripts/gen_stock_labels.py`). A draw is rejected and redrawn if it is empty or
equals stock's default templates (silent fallback); every draw, accepted or not, is kept.

Common (as EXP25): `z-ai/glm-5.3-flash` via OpenRouter, Novita only, reasoning low, temperature 0.7;
exactly 100 solution attempts; transient transport errors retried up to one hour per call. Native arms
evaluate with EXP25's stock-exact per-signal evaluator (`../EXP25/signal/whitebox.py`, unchanged).
Scheduling: three waves, one per seed, each running all four arms concurrently, so endpoint load is shared
across arms rather than confounded with them.

## Endpoints

- **Primary (P1):** best valid score among all candidates evaluated in the run (where EXP25's gap appeared).
- **Co-primary (P2):** best valid score among strictly causal candidates (`causal_fraction == 1`).
- **Mechanism variables:** share of candidates importing SciPy; share of mutations using DIVERGE (native
  arms); whether labels name SciPy; label text of every arm.
- Secondary: returned program's valid score and causality, best-valid-so-far at 25/50/100, wasted
  attempts, policy switches, provider cost.

## Decision rules (fixed before the runs)

EvoX's EXP25 range was 0.679–0.723 on P1. With three runs per arm no significance is claimed; these
rules decide what to do next, not whether an effect is proven.

| rule | condition | conclusion |
|---|---|---|
| R1 | `native_stocklabels` median P1 ≥ 0.68 **and** SciPy share ≥ 30% | label content (M1) explains the gap |
| R2 | `native_pkg` median P1 within 0.02 of `native_stocklabels` | the compact port is adequate as default |
| R3 | `native_stocklabels` median P1 < 0.63 **and** DIVERGE share < 20% | content insufficient; label selection (M2) implicated |
| R4 | `native_stocklabels` median P1 < 0.63 **and** DIVERGE share ≥ 20% | gap lies elsewhere (solution-operator template, cascade) |
| R5 | all arms' P2 medians within 0.03 | EvoX's lead is look-ahead only; run EXP27 with a causal guide |

`evox_stock` must itself reproduce EXP25 (median P1 ≥ 0.68) for R1–R4 to be read; otherwise the
campaign is a reproduction failure and is reported as such.

## Limits

Three runs per arm; no significance test. The five signals are both feedback and score (no holdout).
`native_stocklabels` uses one label draw per seed, not the draw the matching `evox_stock` run made.
`evox_stock` keeps SkyDiscover's cascade evaluator and solution-operator prompt; only label content is
equalised. Transcripts and run data are local (`experiments/**/*.jsonl` is ignored) and summarised in
RESULTS.md.
