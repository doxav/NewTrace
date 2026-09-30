# EXP-16 / P1 — prospective production validation of instrument corrections

Status: draft until the exact machine-checkable freeze exists. This document and
all execution/analysis sources are sealed before the first P1 generation. This is
the final prospective exploratory efficacy diagnostic of EXP-16; it does not
replace EXP-15 or turn the investigation into a search for a favorable outcome.

## Questions and arms

At eight completed generation responses per arm, does training-informed production
search outperform independent generation on new instances? Does evaluated feedback
add value beyond repeatedly editing a training-selected parent? Does exploring two
parents per round change that outcome?

| Arm | Generator information and production schedule |
|---|---|
| A0 | Unchanged handwritten seed; no generative calls |
| I | Eight fresh independent requests containing the seed and common invariant information |
| C | Eight one-parent production updates; current parent code, no evaluated feedback text |
| R | Same schedule, with actual propagated current-parent training Trace feedback |
| W | Two production parents per round, four update rounds, eight responses total; same rich feedback |

The first width-two round may start from copies of the same seed. Parent identities,
actual source hashes, lineage, archive outcomes and all callbacks are retained.
No claim of distinct algorithms is inferred from a class alias. This uses the
existing Control Plane/Trace/PrioritySearch implementation, with a narrow tested
owner/slot adapter. It does not test an additional level of nested recursion or a
complete memory of rejected candidate programs.

Every arm has identical invariant information: optimizer API, constraints, generic
task description, seed source, explicit anytime selection objective and reserved
identifier clarification. R/W additionally receive indexed raw progress, incumbent
coordinates, improvement events and best-so-far curves for the current source,
plus its aggregate training AUC. This combined instrument correction is broader
than F1's raw-only feedback factor. Optima, parameters, normalization constants,
task identities, family labels and split labels never enter the candidate API.
No individual normalized/raw per-task score pair is supplied to generation.

## Frozen tasks, randomness and metric

Keep EXP-15's shifted Sphere, anisotropic axis-aligned Quadratic and Rosenbrock,
dimensions2/4, parameterization, bounds[-5,5], central shift range[-2,2], B32 and
128-point independently seeded uniform reference normalization. B1 demonstrated
usable headroom under this task; no family is removed or added based on A2 results.
No OOD arm or task is part of this diagnostic.

Use `fresh_tasks('P1', split, replicates)` with train4, validation2, audit2 replicates
per family/dimension stratum:24/12/12 instances. Use two local optimizer seeds per
instance/outer seed, shared across all candidates and arms. Derive each with
`local_seed('P1', int(sha256(canonical_json([outer, local_index]))[:15],16), task)`,
local_index0/1, as implemented in the frozen owner. Task, generation, local and
outer seeds are separate namespaces. Outer seeds:
`[16411,16423,16437,16441,16453,16467]`.

Primary: mean normalized anytime regret, lower better. At each t, normalized regret
is `(best_observed(t)-known_optimum)/reference_mean_excess`; AUC is its arithmetic
mean over all32 evaluations. Constants remain host-only. The unchanged benchmark
implements the numerical tolerance and rejects larger optimum inconsistencies.
Average trajectories within each of the six family/dimension strata, then equally
average strata. Outer seeds, not tasks or time points, are the replication units.
Also retain final normalized regret, target0.01 attainment, capped/censored
evaluations-to-target, validity, fallback, source complexity and execution time.

The S1 fixed-bank study motivates24 training instances ×2 local seeds. Its measured
selection improvement is not a projected recursive-feedback effect. Validation
and audit each use12 independent instances ×2 locals. P1-E1 checks the longer
training prompts and production path before P1 begins. Pilot programs/results
cannot enter these pools or prompts.

## Budgets, order and failure handling

Use the exact G1 reliability-selected configuration: OpenRouter,
`deepseek/deepseek-v4-flash-0731`, max_tokens32000, temperature0.6,top_p1,
`extra_body={"reasoning":{"effort":"low"}}`, timeout300s, cachefalse,
empty retries0, wrapper max_retries1, client num_retries0. Live concurrency1,
common default routing. No provider/model substitutions. Request seed hint uses
the frozen namespace/outer/slot derivation and excludes arm, without assuming
deterministic provider generation. Max total prompt text524288 characters.

Generation order rotates whole arms by outer-seed index through `[I,C,R,W]`.
Sequential dependencies are preserved. Generation is completed for all six seeds
before any validation; every selection is frozen before any audit. No early
audit outcome is visible during later generation or selection.

There are6×4×8=192 completed-response allocations. Invalid/empty/truncated responses
consume their slot. No repair, retry of a completed response, manual source editing,
extra critique call or early stopping. Retain transport attempts separately under
the bounded2/4/8s retry policy. Preserve uncertain remote completion/billing and
resume only unfinished slots after checking immutable request/code/config state.

Every arm's final pool contains the seed (index−1) and all eight raw proposals.
Every pool entry receives48 train and24 validation trajectory allocations, without
recycling invalid candidates' unused calls. Final eligibility requires every train
and validation trajectory to finish validly. Select by minimum validation AUC,
then earliest index (seed wins exact ties). No eligible replacement means seed
selection, reported explicitly. The overall representative is the selected R
program with the smallest validation AUC, tied by registered outer-seed order;
freeze it with all selections before audit.

Ordinary candidate invalidity remains typed, with partial observations and no
invented numeric objective score. A completed production search whose final
exploration artifact is invalid may still have a valid seed-inclusive selection
pool. This must not be classified as infrastructure failure. Unknown engine,
trusted-seed, evaluator or fallback defects stop the affected execution.

During audit, permanently switch a failing selected program to the common seed
for the remaining budget, retaining actual history and local seed. No reset,
extra objective calls or extreme-score imputation. Primary results describe this
deployment procedure; candidate-only validity and fallback are separate outputs.

Offline workers8, supported by T1's exact outcome comparisons; hard per-proposal
timeout2s and deterministic replay remain unchanged. This is a sanitized subprocess
API boundary, not an OS filesystem sandbox. Source protocol violations are invalid.

Logical upper allocations:216 seed-inclusive pool entries ×72 trajectories
=15552 train/validation trajectories,497664 objective calls; plus720 audit
trajectories,23040 objective calls. Without deployment failures, deterministic
replay allocates1041408 candidate subprocess executions. A failing generated
deployment may add one or two failed subprocess attempts before permanent fallback,
at most1152 additional attempts across the576 audit trajectories of the four
generative arms. The resulting upper bound is1042560 with successful trusted
infrastructure, before cache savings and invalid early termination. Shared
normalization preparation is separate.
Complete cache keys include source/task/local seed/outer/split/budget/deployment/
timeout/seed fallback/evaluator version and code hash. Report actual physical calls,
logical allocations, cache accesses/hits and unused invalid allocations separately.

## Analysis, integrity and interpretation

Freeze the analysis implementation before audit. Report every outer-seed value,
per-arm mean/median and paired signed contrasts R−I (central),R−C,W−R,R−A0.
Use the existing frozen EXP-15 paired bootstrap algorithm, random seed, number of
resamples and percentile interpolation, included by hash in the P1 freeze.
Resample outer seeds jointly; negative AUC deltas favor the first arm. Its interval
labels are descriptive positive/negative signal, no detectable difference (all
deltas zero), or inconclusive. All contrasts are exploratory at n=6, with fragile
intervals and no multiple-comparison guarantee or significance claim.

Report both favorable and unfavorable contrasts with actual tokens/cost; equal
response allocations are not equal realized token costs. Include source/execution
invalidity, seed selection despite eligible alternatives, fallback, every response
and transport attempt, metadata coverage, selected source hashes and lineage.
Descriptive prefix search curves use training/validation only and their best-of-N
monotonicity is by construction. They do not independently establish future gains.

Persist exact raw request/response/source bytes and safe receipts incrementally.
Verify hashes, completed slot IDs, chronological generation/selection/audit barriers,
manifest equality and recomputed aggregates. Record wall/monotonic/boot clocks so
user host suspension is distinguishable from provider latency. No completed
scientific evidence is overwritten. Defects affecting semantics require an explicit
new version; reporting-only repairs require tests and provenance.

No subsequent efficacy stage is triggered merely because R loses or remains
inconclusive. The final inquiry must distinguish demonstrated instrument defects,
validated engineering corrections, fixed-bank selection gains, prospective search
effects and remaining hypotheses. No novelty, amortization or nested-depth claim.

## Resource-accounting clarifications before freeze

The installed client forwards timeout300 to HTTPX connect/read/write/pool
operations. A loopback test through the actual project/OpenRouter adapter confirms
that continuing response bytes can keep a request alive longer than300 seconds.
This is not an absolute generation deadline, and the parameter is not silently
dropped. Preserve active/elapsed durations and any suspension separately.

The48 unique P1 tasks imply6144 reference objective calls for one complete unique
normalization design. Physical reconstructions across separate processes/resumes
are not fully instrumented and must not be mislabeled as exactly6144 physical
calls. Candidate objective allocations/cached outcomes are counted separately.
For P1-E1 only24 training instances are evaluated; its unique preparation design
is3072 reference calls, with the same reconstruction caveat.

Measured feasibility before the first P1 request: F1 completed24 responses in26
transport attempts; the completed-response active durations sum13,912.446s,
mean579.685s, median276.014s, maximum4,229.334s. The maximum is a retained
provider-ended error, not a host suspension and not discarded from this estimate.
F1 reported321,691 total tokens and $0.079774660676; unresolved timeout billing
is additional unknown usage. P1-E1 completed2 responses in2 attempts, with
472.360s summed request-active time,127,274 total tokens and $0.003472328.
Both141,807-character prompts were transmitted; one generated source passed all48
training trajectories, while the other was syntactically invalid. The strict
production accounting, training-only boundary and client-free resume gate passed.
Both callbacks used the same seed parent after the first invalid source; this does
not demonstrate reliability after many accepted replacements.

Linear192-response scenarios based on those two observed mean latencies are12.60h
(E1) and30.92h(F1), plus approximately1.86h offline evaluation from T1's eight-worker
measurement before cache/invalidity savings. These are scenarios, not a confidence
interval or upper bound; provider tails, transport failures and suspension may add
time. The corresponding reported-usage extrapolations are12.22M/2.57M total tokens
and $0.333/$0.638 respectively. E1 contains only long R prompts and F1 mostly short
prompts, with heterogeneous upstream routing; neither is an unbiased forecast for
the four-arm mixture, and reported costs are not a billing cap. Reserve sufficient
local runtime, preserve checkpoints, and keep the registered six replications and
eight response slots. No resource or scientific design change is made from F1
efficacy outcomes. The supplementary fixed B2 reference adds144 offline trajectories
and no generative responses, as specified below.

## Prospective supplementary initialization reference

Before any main P1 generation, separately register the fixed B2 optimizer under
`production/BASELINE_CONTROL_PROTOCOL.md`. Its exact source hash is
`958fbb12279a15966bf1ffa45ddffc7cd5f946ca800a3fe2b8e132a150c5c190`.
This reference changes only the original seed's empty-history proposal to the box
midpoint. Its development evidence comes from public B1 diagnostic fixtures.

The separate control freeze must bind this main P1 freeze before generation starts.
Only after all primary generation, selection and audit are complete and the primary
integrity checks pass may the control evaluate its 144 fixed audit trajectories:
the same six outer seeds, 12 audit instances, two local seeds and B32. It reuses
the original seed as the common permanent deployment fallback. No primary seed,
prompt, candidate pool, selection, comparison, cache or raw evidence is changed.

Allocate 4,608 additional objective calls and 9,216 successful-replay subprocesses,
at most 9,504 subprocesses including failed proposals before trusted fallback.
Reference reconstruction and interrupted uncheckpointed work are counted separately.
There are no additional model responses. Store all evidence outside the primary
run tree. Compare B2−A0 and R−B2 using the already declared paired procedure, as
supplementary exploratory references with all six seeds retained. This cannot
replace the central R−I comparison or make it a confirmatory test against B2.
The control runs regardless of the signs of the primary results.
