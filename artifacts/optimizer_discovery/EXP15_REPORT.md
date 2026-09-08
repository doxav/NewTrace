# EXP-15 — controlled optimizer-program discovery

**Status: confirmatory execution in progress. No confirmatory scientific conclusion yet.**

Research question: does iterative optimizer-code generation through recursive_opt
produce a deployable black-box policy with better held-out normalized anytime regret
than its unchanged starting policy and equal-response-budget independent generation?
H15-B (A2 versus A1) is central. This is not a FunSearch/OpenEvolve comparison.

## Provenance and design

Scientific baseline: 90651f47b9c3c2570e33b0aa64ca648c4a7c765c.
Branch: codex/phase1-optimizer-discovery-exp15. A rollback branch preserves the base.
The sole initial untracked user probe remains outside the experiment. Phase-0
specifications, requests, responses and negative evidence were not overwritten.

Initial protocol: 13b88440. Pilot implementation: 81cf12c0. Final scientific freeze:
0643691b, before any confirmatory request. The freeze records 89 source/config files,
24 semantic task fingerprints and the Python/distribution/platform environment.

| Item | Frozen configuration |
|---|---|
| Artifact | OptimizerProgramV0: `propose(history, bounds, seed) -> x` |
| Families | shifted Sphere, anisotropic axis-aligned Quadratic, shifted/scaled Rosenbrock |
| Dimensions / bounds | 2 and 4; [-5,5] in each coordinate |
| Splits | train 6, validation 6, ID holdout 12; balanced family/dimension strata |
| Inner budget | 32 actual objective evaluations per trajectory |
| Outer seeds | 11, 23, 37, 41, 53; one paired local seed per task/outer seed |
| A0 | unchanged stochastic exploration/incumbent-perturbation seed |
| A1 | eight independent completed responses per outer seed |
| A2 | eight training-feedback updates through production Trace/PrioritySearch |
| Selection | seed plus all candidates valid on every train/validation task; lowest validation AUC, earliest index; seed index -1 |
| Holdout barrier | every outer-seed selection and validation-selected representative frozen first |
| Model | OpenRouter, deepseek/deepseek-v4-flash-0731; no replacement or provider pinning |
| Settings | temperature .6, top_p 1, max_tokens 8000, native low reasoning, timeout 300s, concurrency 1, cache false, empty retries zero |
| Transport | initial attempt plus three transient retries with delays 2/4/8s; separate attempt receipts |
| Normalization | mean excess value on 128 deterministic independent uniform reference points per task |
| Primary | mean normalized best-so-far regret across all 32 evaluations, equal stratum weights; lower better |
| Uncertainty | 10,000 paired outer-seed bootstrap draws, seed1515, percentile 95% interval |
| Target | normalized regret .01; nonattainment censored, B+1 reporting convention |
| Deployment failure | permanent seed fallback with actual accumulated history and remaining budget |

Each generative arm receives 17,280 logical search objective allocations and 1,920
holdout allocations. All arms' total is 40,320 search/deployment allocations, with
shared deterministic caching explicitly audited; normalization preparation is
separate. Equalized resource is completed response slots and allocated task budgets,
not realized tokens or money. Completed empty/truncated/invalid outputs consume slots.

The scalar Control Plane fix before the pilot prevents empty typed-invalid metric
dictionaries from crashing redundant scalar projection. Internal rejection ranking
is never reported as scientific regret. A narrow registered engine adapter delegates
to the existing production engine; it does not implement another search framework.
The only model calls are declared proposal slots, with deterministic training feedback.

## Engineering pilot and verification

The separate pilot retained all four responses (two eligible candidates, two source
failures) and one transport retry. Both validation pools selected the seed. Independent
fixed-policy diagnostics established performance headroom; ten identical local
replays were exact. See [pilot report](exp15/PILOT_REPORT.md), including measured
cost/latency, routing metadata and unknown billing on the timed-out attempt.
No scientific setting changed because of pilot arm performance.

Baseline: 559 passed, 2 skipped. Final affected regression: 581 passed, 2 skipped.
Broader offline suite: 766 passed, 3 existing optional skips, one existing warning.
New/interface-specific suites: 47 passed. Commands and exclusions:
[verification record](exp15/VERIFICATION.md). Final raw-evidence checks follow after
confirmation, including full manifest/source/slot/chronology integrity and recomputation.

## Confirmatory results

Pending completion of every registered slot and global validation selection.
No partial holdout analysis is permitted. Raw requests, responses, attempts and
training evaluations are being persisted incrementally under exp15/raw/.

## Interpretation and limitations

No present claim supports H15-A or H15-B. The preregistered paired interval will
classify positive signal, negative signal, no detectable difference, or inconclusive.
Five outer seeds yield fragile uncertainty; sample mean ordering alone is insufficient.
The primary estimand includes the shared deployment fallback, with candidate validity
reported separately. Invalidity is never assigned an artificial regret value.

Neither this experiment nor a positive outcome establishes literature novelty,
additional recursion-depth benefit or amortization. Historical H3 concerned a
different signature-bound task interface; its evidence and retractions remain intact.
Provider routing and LLM generation are stochastic. Reference tasks are modest and
ID-only; there is no OOD claim.

The subprocess boundary provides API separation and a sanitized credential environment,
not an operating-system security sandbox. The AST protocol check does not prove
filesystem confinement against adversarial code. No Project-1 infrastructure was built.

## Portable next step

The engine-independent evaluator and replaceable launcher are documented in
[PORTABILITY.md](exp15/PORTABILITY.md). Patrick can integrate a source-producing
FunSearch/OpenEvolve adapter under a new preregistered equal-budget comparison using
the identical artifact, task generator, metrics, selection and fallback rules.
He is not being asked to adopt recursive_opt infrastructure. The brief will not be sent.
