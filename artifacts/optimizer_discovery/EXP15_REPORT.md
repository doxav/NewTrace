# EXP-15 — controlled optimizer-program discovery

**Status: valid confirmatory experiment completed. Positive signal versus the seed;
the central comparison against independent generation is inconclusive.**

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

The registered transformations use z_i=(x_i−shift_i)/scale_i, shifts uniform in
[-2,2], coordinate scales uniform in [0.75,1.5] and amplitude 10^U(−1,1).
Sphere is amplitude·sum(z_i²), so coordinate scaling introduces modest anisotropy.
Quadratic additionally weights each square by 10^U(0,3). Rosenbrock uses y_i=1+z_i
and amplitude·sum(100·(y_(i+1)−y_i²)²+(1−y_i)²). Every known feasible optimum is
x=shift, value zero. Exact seed derivations, aggregation equations and reference
normalization are in the frozen [preregistration](PREREG_EXP15.md) and
[manifest](exp15_manifest.json).

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
Latest broader offline suite: 772 passed, 3 existing optional skips. The earlier
766-test run also reported one pre-existing SyntaxWarning.
New/interface-specific suites include interrupted production-A2 resume, corrected
attempt-timing association and path-independent source export. Commands and exclusions:
[verification record](exp15/VERIFICATION.md). The final integrity audit verifies
all 80 unique response IDs, actual settings, raw source hashes, production Trace
blocks, equal allocations, rotated sequential requests and selection chronology.
All cached trajectory metrics and the complete aggregate recompute exactly.

The original 3,000-token completion allowance was a starting design choice, not
evidence that this reasoning model could reliably finish code within that budget.
The prospectively selected 8,000-token/low-reasoning configuration passed Phase 0's
ten fresh fixture requests; those used at most 2,331 completion tokens. That result
did not establish readiness for longer iterative prompts. EXP-15 preserves its
frozen 8,000-token cap even when reasoning consumes the allowance without usable
code. A user-authorized larger cap would require a separate registered protocol;
it is not applied midway through these confirmatory comparisons.

## Runtime observation and reporting correction

R15-LATENCY-01 is a reporting-only defect. A manual diagnostic initially paired the
start of slot 11/A1/06's first attempt with the 275.127-second duration of its second
attempt. That wrongly attributed the entire 2,132.109-second difference to a clock
or suspension gap. The original two diagnostic summaries are retained and explicitly
superseded by exp15/raw/latency_observation_correction.json.

Correct accounting: attempt one ended in a transport timeout after 372.852 measured
seconds; attempt two completed after 275.127 seconds. Total measured attempt time is
647.979 seconds, versus 2,407.236 wall-clock seconds for the slot. The remaining
1,759.257 seconds include backoff, scheduling and the observed system suspension;
they are not assigned to model inference. The completed response had no code after
8,000 reasoning tokens and consumed exactly one scientific slot. Its first attempt
has unknown remote completion/billing. The first A1 block has eight completed
responses and nine transport attempts.

A tested reporting helper now matches each response with its actual attempt ID.
Original requests, responses, sources, evaluations and frozen settings are unchanged.
No primary metric or scientific selection changed, and no experiment was invalidated
by this descriptive reporting repair. The live process was never manually restarted.

R15-EXPORT-PATH-01 is also reporting-only. Re-exporting with an absolute path after
an earlier relative-path invocation produced different path labels in index.json.
The immutable writer refused the overwrite. A failing test reproduced this; the
reporting helper now uses canonical repository-relative labels (absolute labels
outside the repository). The existing index and all source exports remain byte-for-byte
identical to commit 6ffdd524, and both invocation forms pass. No frozen scientific
source changed. The initial failed audit had already verified the scientific values;
the complete audit was repeated successfully after this repair.

Large production Trace records were archived losslessly as XZ after generation
completed. Exact JSON hashes and original gzip hashes remain alongside each archive.
The first oversized gzip entered one raw-data commit before being replaced in the
working tree; Git history preserves it. Final file-size checks were not relaxed.
These reporting/storage corrections require no scientific invalidation or replacement
experiment. No completed negative or invalid response was rerun.

The completed 1,463,776-byte event stream is retained as a 45,361-byte gzip archive,
with exact-byte hashes and a tested temporary restoration path for the unchanged
analyzer. This also passes the 500 KiB file gate. Recompute the delivered evidence
with `python -m artifacts.optimizer_discovery.reporting artifacts/optimizer_discovery/exp15/raw`;
the adapter verifies exact equality with the preserved aggregate and makes no model
or candidate calls. The original scientific evaluator and runner remain frozen.

## Confirmatory results

Normalized anytime regret, lower is better. Each row first averages the fixed
family/dimension strata; the replication unit is the outer seed.

| Outer seed | A0 seed | A1 independent | A2 recursive | A2−A0 | A2−A1 |
|---|---:|---:|---:|---:|---:|
| 11 | 0.120232 | 0.104444 | 0.120232 | 0.000000 | +0.015787 |
| 23 | 0.162396 | 0.151787 | 0.145691 | −0.016705 | −0.006096 |
| 37 | 0.164886 | 0.126687 | 0.164886 | 0.000000 | +0.038199 |
| 41 | 0.133280 | 0.086700 | 0.079068 | −0.054212 | −0.007632 |
| 53 | 0.117101 | 0.139123 | 0.073525 | −0.043576 | −0.065598 |
| Mean | **0.139579** | **0.121748** | **0.116680** | **−0.022899** | **−0.005068** |
| Median | 0.133280 | 0.126687 | 0.120232 | −0.016705 | −0.006096 |

| Preregistered contrast | Mean paired delta | Paired bootstrap 95% interval | Interpretation |
|---|---:|---:|---|
| H15-A: A2−A0 | −0.0228985262 | [−0.0424560606, −0.0033409918] | positive signal; supported within EXP-15 |
| H15-B: A2−A1 | −0.0050677685 | [−0.0377275162, +0.0245505610] | inconclusive; added feedback value not established |

A2 beats A1 in three pairs and loses in two. A2 retains the seed in two pairs;
the remaining three improve over it on this holdout panel. These are conditional
results for the sampled tasks and budget, with fragile uncertainty at n=5.
The sample mean ordering does not establish superiority over independent search.

Secondary deployment metrics also retain unfavorable outcomes:

| Mean over outer seeds | A0 | A1 | A2 |
|---|---:|---:|---:|
| Final normalized regret | 0.019441 | **0.010110** | 0.012166 |
| Target attainment at regret ≤0.01 | 37/60 (61.7%) | **49/60 (81.7%)** | 46/60 (76.7%) |
| Capped/censored target evaluations | 21.200 | **16.017** | 17.383 |
| Holdout candidate-valid trajectories | 60/60 | 60/60 | 60/60 |
| Deployment fallback trajectories | 0/60 | 0/60 | 0/60 |

Nonattainment remains null in the raw hitting-time field. The capped mean uses 33
for nonattainment; it is not an observed hitting time. A1 has better sample means
on these final-regret/target metrics despite A2's slightly lower primary mean AUC.

## Generation validity and resources

| Confirmation | A1 | A2 |
|---|---:|---:|
| Completed responses / allocated slots | 40/40 | 40/40 |
| Missing code after length finish | 9 | 7 |
| Syntax errors | 0 | 4 |
| Additional execution-invalid candidates | 0 | 1 |
| Eligible generated candidate slots | 31/40 (77.5%) | 28/40 (70.0%) |
| Seed selected | 0/5 | 2/5 |
| Prompt tokens | 17,557 | 118,738 |
| Completion tokens, including reasoning | 215,174 | 221,722 |
| Reasoning tokens, subset of completion | 188,629 | 193,655 |
| Total tokens | 232,731 | 340,460 |
| Response-reported cost, USD | 0.042630795124 | 0.044193322216 |

The one execution-invalid program, outer23/A2/slot02, uses unseeded global
randomness for its first point. All twelve allocated train/validation trajectories
fail deterministic replay before an objective call. Its source remains preserved.
There are no recorded candidate-execution timeout or exception failures. Two eligible generated
responses reproduce the seed's exact source; eligible slots are not a count of
distinct new algorithms. Every pool contains eligible generated programs, so neither
seed selection resulted from an entirely empty eligible-generation pool.

Inner recorded proposal validity is 11,904/12,012 for A1 (99.10%) and
10,752/10,896 for A2 (98.68%). These denominators include source-screen failure
events and many points per valid trajectory; they must not obscure the candidate
invalidity rates of 22.5% and 30.0%.

There are **80 completed model responses, 81 transport attempts, one transport
timeout and zero replacement calls for completed invalid responses**. Total reported
usage is **573,191 tokens**, of which 436,896 are completion tokens including
382,284 reasoning tokens. Cost is **USD 0.086824117340** from response usage;
the 80 later provider receipts sum to USD 0.086824096. Both originals remain;
their difference is about 2.13×10⁻⁸ USD. The timed-out attempt has unknown remote
completion/billing. The separate pilot adds four responses, five attempts,
26,788 tokens and USD 0.005411162024; it is excluded from all scientific estimates.

Equal response limits did not yield equal tokens: A2 used longer prompts. Fifteen
upstream provider identities were observed under the common routing policy,
including Baidu for 39/80 responses. See [resource audit](exp15/resource_audit.json).
Median completed-response duration is 37.387 seconds, maximum 437.455 seconds;
300 seconds is the configured client timeout, not a demonstrated wall-clock ceiling.
Summed measured transport-attempt durations are 6,285.940 seconds. Provider timing
is secondary and includes uncontrolled routing/latency variation.

| Objective accounting | A1 | A2 |
|---|---:|---:|
| Logical search allocation | 17,280 | 17,280 |
| Logical completed search calls, including reused results | 13,824 | 12,672 |
| Logical unused search allocation | 3,456 | 4,608 |
| Logical holdout allocation | 1,920 | 1,920 |

A0 adds 1,920 holdout allocations. Total scientific allocation is 40,320;
32,256 logical values are available and 8,064 allocations are unused. Unused
allocations were not recycled. Shared deterministic execution used **28,800 actual
objective calls**: 11,904 train, 11,904 validation and 4,992 holdout. There were
57,624 subprocess executions and 2,382.437 seconds of actual shared execution time.
The extra 24 subprocesses are the rejected program's twelve replay pairs.

The production path recorded 40 Trace updates, 3,090 evaluation requests, 2,070
cache hits and 1,020 unique cached trajectories, including 120 invalid trajectories.
Trace feedback/selection reevaluations reused complete frozen cache keys; they did
not create extra candidate proposals or objective opportunities. Unique unused
allocations are 3,840, distinct from logical unused allocations. Shared normalization
preparation uses 3,072 reference evaluations. Post-run integrity auditing separately
used 4,608 reference evaluations, including the audit interrupted by the export
path-label issue; these supplied no search information or candidate opportunities.

## Selected programs and lineage

All five A2 sources are exported unchanged in [selected/index.json](exp15/selected/index.json).
Gzip decompression yields the exact evaluated optimizer.py bytes. Both source and
compressed-file hashes are recorded. The full sources for A1 also remain in each
outer seed's selection.json and original response record.

| A2 outer seed | Selected zero-based slot | Ancestry from seed | Exact source SHA-256 |
|---|---:|---|---|
| 11 | −1 | unchanged seed | 5b74e5a3fe2fc90fcb42a603b058fa7befbf65f38775caa77acd30cdf619b640 |
| 23 | 0 | seed → 0 | 8c1079d841f13ddc1a6c739447b3bfd94477fbee5daf147f41c30611edae8ded |
| 37 | −1 | unchanged seed | 5b74e5a3fe2fc90fcb42a603b058fa7befbf65f38775caa77acd30cdf619b640 |
| **41, representative** | **5** | **seed → 4 → 5** | **1684f91acdc36c0ca6aac70afeb9cc2c4eed7ab847926d5880590e059266abb7** |
| 53 | 6 | seed → 3 → 4 → 6 | 3a87caf393828d41396dda16eafb703bca52cd2dcb437ddfe8d86bd90e9904b4 |

The representative was frozen on validation AUC 0.0509190046 before any holdout.
Slot4 introduced a regularized full quadratic fit. Slot5 adds Halton exploration,
a predicted-improvement gate and randomized use of perturbed quadratic stationary
points, plus occasional heavy-tailed local steps. It clips proposals to bounds and
reduces exploration/local scales with history length. It does not check that a
quadratic stationary point is a minimum. Its training AUC was worse than its parent
(0.104562 versus 0.083489), yet it remained available for validation selection as
registered; retaining search-rejected attempts mattered to the selection procedure.

The seed23 program scores uniform candidates through their nearest observation and
a positive distance penalty; its source comment calling that penalty an exploration
bonus is inaccurate for minimization. Seed53 uses Halton initialization, stagnation
counts, incumbent/top-quartile bases and directional/Gaussian perturbations. Its
directional noise uses absolute units while other steps scale by bound width.
Exact attempted diffs are in the per-seed lineage files; descriptive inspections
are under exp15/inspection/. No candidate was manually repaired or reformatted.

## Interpretation and limitations

H15-A receives a positive signal under the preregistered rule. H15-B is inconclusive:
this experiment has not established that iterative feedback adds value over equal-
proposal independent generation. Five outer seeds yield fragile uncertainty; sample
mean ordering alone is insufficient. There is no claim of statistical significance.
The primary estimand includes the shared deployment fallback, with candidate validity
reported separately. Invalidity is never assigned an artificial regret value.

Neither this experiment nor a positive outcome establishes literature novelty,
additional recursion-depth benefit or amortization. Historical H3 concerned a
different signature-bound task interface; its evidence and retractions remain intact.
Provider routing and LLM generation are stochastic. Reference tasks are modest and
ID-only; there is no OOD claim.
Optima lie in the central [-2,2] sub-box of [-5,5]; conclusions are conditional on
that task distribution. Uncertainty conditions on the frozen twelve-instance panel.

The frozen cache lookup/accounting expects ordinary JSON trajectory records. All
1,020 actual cache files satisfy that expectation (none required compression).
A future version supporting much larger captured output must handle compressed
cache records consistently before freezing its own run; this edge was not exercised
and did not affect EXP-15. No frozen evaluator was changed after seeing outcomes.

The subprocess boundary provides API separation and a sanitized credential environment,
not an operating-system security sandbox. The AST protocol check does not prove
filesystem confinement against adversarial code. No Project-1 infrastructure was built.

## Portable next step

The engine-independent evaluator and replaceable launcher are documented in
[PORTABILITY.md](exp15/PORTABILITY.md). Patrick can integrate a source-producing
FunSearch/OpenEvolve adapter under a new preregistered equal-budget comparison using
the identical artifact, task generator, metrics, selection and fallback rules.
He is not being asked to adopt recursive_opt infrastructure. The brief will not be sent.

Machine-readable results: [exp15_results.json](exp15_results.json). Raw evidence:
exp15/raw/. Final checks: [final_integrity.json](exp15/final_integrity.json).
The exact next engine integration should be preregistered under a new legitimate ID;
EXP-15 and its now-observed holdout must remain preserved as a completed experiment.

## Commit provenance

The following 24 commits precede the final report/ledger delivery commit, whose SHA
is reported in the delivery response. The scientific freeze remains 0643691b.

| Commit | Change |
|---|---|
| 13b88440 | exp15: register pilot and draft confirmatory design |
| c5353169 | exp15: implement tested deterministic benchmark and deployment fallback |
| fc31bb66 | exp15: keep scalar Trace ranking operational after typed-invalid proposals |
| 81cf12c0 | exp15: integrate budgeted independent and registered production Trace arms |
| 5d7a6b00 | exp15: verify environment and evidence integrity before confirmation |
| 15af6e2a | exp15: preserve separate pilot evidence and feasibility assessment |
| 0643691b | exp15: freeze confirmatory protocol after the separate pilot |
| 339ff237 | exp15: record frozen run status and broader offline verification |
| b2ea8821 | exp15: document frozen methods while confirmation runs |
| 87567435 | exp15: audit interrupted production search resume without extra proposals |
| a95763e0 | exp15: distinguish observed system suspension from model call latency |
| 4003f25d | exp15: preserve first independent-arm response block and accounting audit |
| 79963997 | exp15: correct descriptive latency attribution across transport attempts |
| e18baf61 | exp15: prepare tested source exports and draft Patrick methods brief |
| 5c16c8ad | exp15: preserve first paired search and validation selection |
| bebc83a1 | exp15: archive complete Trace evidence losslessly within repository limits |
| f40d5c44 | exp15: record complete offline regression for final reporting tools |
| 0ab209ad | exp15: preserve second paired search and validation selection |
| 9f6bb040 | exp15: describe first selected policies without altering evaluated source |
| 896db848 | exp15: preserve third paired search and selection |
| 3aa687e7 | exp15: preserve fourth paired search and selected program |
| 6ffdd524 | exp15: preserve all proposals and frozen validation-selected artifacts |
| 50d8837d | exp15: verify portable exports and lossless completed-event replay |
| cfb4b41f | exp15: preserve complete confirmation and verified raw-result aggregates |
