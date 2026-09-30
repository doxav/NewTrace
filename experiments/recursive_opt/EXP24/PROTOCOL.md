# EXP24 — protocol (written before the clean runs, 2026-09-30)

## Question

EXP23's best PRISM result (30.8766) is a metric exploit. With a valid guide, per-case white-box feedback and a
repair projection at O0, (1) how fast does each arm reach the proven PRISM optimum, and (2) does the meta level
(policy co-evolution) add anything over a fixed stock policy with the same O0?

## Fixed facts established before the runs (pilot, `results/pilot_20260930T064616/`)

- PRISM's stock score averages KVPR over solved cases only; crashing on hard cases raises it (`scripts/prism_validity.py`).
- Exact optimum on all 50 cases: **26.2559717495** (`scripts/prism_exact.py`, `results/analysis/prism_exact_optimum.json`).
  Any higher stock score is an exploit; final score cannot rank methods.

## Arms (identical O0 and budget; `scripts/run_prism.py`)

| Arm | Meta level |
|---|---|
| `fixed` | Stock uniform policy, never changed (trigger `never`) — control |
| `llm_rewrite` | EvoX-equivalent (`evox_preset`: stagnation trigger, LogWindowScorer, best-parent archive, LLM rewrite) |
| `trace` | Same, with the persistent OptoPrimeV2 proposer |

Shared O0: SkyDiscover-equivalent diff operator with generated diverge/refine labels; guide `guided_score` =
`valid_score` (1 / mean KVPR over all 50 cases, a failed case counted at the initial program's KVPR, + success rate);
per-case feedback artifact; projections `compile_check` + `fallback_wrapper` (baseline = initial placement function).

## Budget and runtime

100 solution LLM calls per run, enforced exactly (`strict_budget`). Model `z-ai/glm-5.3-flash` via OpenRouter, provider
Novita only (no fallback), reasoning effort low, temperature 0.7. Transient transport errors are retried for up to one
hour per call and do not consume solution budget. Seeds 42, 43, 44 per arm (policy RNG and archive seed); LLM sampling
is not seeded. Nine runs execute in parallel, one worker each.

## Endpoints

- Primary: solution calls until the first child at the exact optimum (not reached = censored at 100).
- Secondary: best valid score, wasted attempts (failed generation/parse/evaluation), children at the optimum, policy
  deployments, provider-reported cost, transport retries.

## Analysis and limits

`scripts/analyze.py` per arm; report every run, medians only when all runs reached the optimum. Three runs per arm
detect only large differences; no significance test is claimed. PRISM is saturated, so this experiment measures
speed and robustness, not final quality. The 50 feedback cases are the scored cases (no holdout exists in PRISM).
