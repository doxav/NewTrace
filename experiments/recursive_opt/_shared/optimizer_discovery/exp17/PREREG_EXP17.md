# EXP-17 — confirmation of training-selected parent rewriting

Status: FINAL CONFIRMATORY PROTOCOL, finalized after the disjoint EXP17-E1 pilot;
source/configuration freeze occurs before any confirmatory response.
Date: 2026-09-10. Base: 13ebda2242e1c18022591737b113030ca2ce2da2 plus the
preserved, completed EXP-16 working-tree sources. Earlier results are not pooled.

## Fixed scientific question

H17: at eight completed DeepSeek proposals, does rewriting a parent selected on
TRAIN (C) improve held-out normalized anytime regret over independent generation
(I)? C receives the current source but no explicit scores or trajectory text.
The current-parent choice itself uses training information. This is the only
confirmatory primary contrast. A positive result would support this search policy,
not explicit rich feedback, novel optimization algorithms or recursion depth.

The confirmatory allocation is fixed at **46 new paired outer seeds**, eight
completed responses per arm (736 total), 32 objective calls per trajectory.
Seeds are integers 17001 through 17046 inclusive. Each adjacent pair uses an order
and its reverse, rotating the base between pairs: IC, CI, CI, IC, then repeating.
This gives 23 occurrences of each precedence. Seeds/instances from EXP-15/16 are not
reused. No early stopping or sample-size change based on outcomes is allowed.

The planning effect is 0.02 AUC. Forty-six pairs come from a normal approximation
using historical C−I paired SD 0.048137, two-sided alpha 0.05 and nominal power
0.80. This is conditional planning, not guaranteed power for the bootstrap.

## Shared task, deployment and generation

OptimizerProgramV0, original exact seed, source screening, timeouts, deterministic
double replay and permanent common seed fallback remain unchanged from EXP-16.
Candidates see history, legal bounds and a local seed only. The host keeps objective
parameters, instance/split identity, optima and normalization private. The subprocess
is not an operating-system security sandbox. No new production dependency.

New namespace `EXP17-CONFIRM-v1`: 24 TRAIN, 12 validation and 12 audit instances,
balanced across shifted Sphere/anisotropic Quadratic/Rosenbrock × dimensions 2/4.
Two local seeds per instance, shared across arms/candidates. Generators, parameters,
reference normalization and seed derivation reuse the frozen EXP-16 instrument.
The independent fixed midpoint-first seed B2 is an additional audit control; it is
not added to candidate pools or substituted for their starting source. A0 is retained.

Each arm allocates seed+8 sources × (48 TRAIN +24 validation) trajectories per outer
seed, including invalid slots. A0/B2/I/C each allocate 24 audit trajectories per
seed. Total logical allocation: 64,032 trajectories, 2,049,024 objective calls.
Actual calls, cache hits, invalid early termination and unused allocations remain
separate; unused budgets are not recycled. Immediate TRAIN recording after each
response makes provenance complete but does not enter independent prompts.

Provider OpenRouter, exact model `deepseek/deepseek-v4-flash-0731`, temperature0.6,
top_p1, max_tokens32000, native extra_body.reasoning.effort=low, timeout300s,
generation concurrency1, cachefalse, empty_response_retries0, client retries0.
Reuse the tested project client and bounded slot transport policy (2/4/8-second
delays, at most three retries). Completed empty/invalid/error/truncated responses
consume slots. Ambiguous interrupted transport is retained and reconciled under
the existing resume rules. Request seeds do not establish deterministic generation.
Local evaluation uses eight workers and a two-second per-proposal timeout.

No model/route change, added critique call, manual candidate repair or hidden
validation access is allowed. Model cost/time settings and exact prompts are
hashed before confirmation; all provider metadata and unknown billing are retained.

## Selection, protected audit and analysis

All generation must finish before validation. A source is eligible only if every
TRAIN and validation trajectory completes validly. Include the seed in every pool.
Select minimum validation AUC, then earliest proposal index (seed=-1). Preserve
every candidate, including search-rejected candidates. Freeze all selections and
the C representative (minimum validation AUC, then registered outer order) before
any audit. Final pools and audit use the same evaluator.

Primary AUC and aggregation equations, target0.01, numerical tolerance, censoring,
family/dimension weights and fallback are exactly the existing benchmark definitions.
Use the outer seed as replication unit. Report every seed, mean/median, final
regret, target attainment/censoring, source/execution invalidity, seed selection,
fallback, objective/LLM allocations, actual tokens and provider-reported costs.

Primary C−I: paired bootstrap10,000, Random1515, percentile2.5/97.5 with the existing
linear interpolation. Entire interval below0: positive signal; above0: negative
signal; all deltas0: no detectable difference; otherwise inconclusive. Report effect
relative to the 0.02 planning threshold without redefining this decision rule.
C−A0, I−A0, C−B2 and I−B2 are registered descriptive secondary contrasts, not
additional confirmatory discoveries. No multiplicity-adjusted secondary claim.

## Engineering pilot and freeze

EXP17-E1 uses namespace `EXP17-E1-v1`, outer17901, I/C with two responses each,
same full panels/settings and separate task seeds. It tests actual production C,
equal slots, prompt isolation, valid seed/fallback infrastructure, response/source
identity, checkpoint/resume and cost/runtime. At least one generated candidate must
execute validly for the generation-readiness gate; poor candidate performance is
not a defect or a reason to replace a response. Pilot comparisons do not select
settings to favor C. Prior benchmark headroom is already established.

Before confirmation: finish tests, pilot, implemented analysis, machine-checkable
preflight and exact source archive; freeze this protocol and manifest. Record any
engineering amendments before execution. A defect changing confirmatory semantics
requires a new version/ID with preserved affected evidence. Reporting-only repairs
need tests/provenance and cannot change scientific values or decisions.

There is no hard monetary blocker indicated: EXP-16 I/C extrapolates about USD1.03
and 5.89M reported tokens, with approximately37.29 active request hours before
pilot, retries and local evaluation. These are estimates, not spending guarantees.
Preserve state through host suspension. Do not stop because C loses.

## Final engineering decision before confirmation

EXP17-E1 completed its exact four real responses and all required TRAIN/validation
allocations. All four sources parsed; three generated candidates completed every
required trajectory validly. All responses ended with stop, none with length.
The actual production C callbacks, source identities, protected split barriers and
complete no-new-work resume passed. No audit was performed. Exact evidence is in
engineering/engineering_results.json and engineering/resource_summary.json.

The first independent source had48 invalid TRAIN trajectories, while its separate
24 validation trajectories completed validly. It remains ineligible: validation
success does not erase training failures. A separately registered eight-call
proposal-only check did not reproduce the first selected failure input; it showed
additional candidate CPU cost, without isolating the system cause. Runtime state
and the two-second bound are limitations of the declared deployment procedure.
These ordinary recorded failures do not justify replacement responses or tuning
the experiment. Worker/timeout rules and all scientific settings remain unchanged.

Pilot usage:28,892 total reported tokens and USD0.0055736, four completed responses
and four transport attempts. Scaling provider-reported generation durations gives
approximately45.06h for736 responses; a fragile observed local-execution scaling
adds about19.94h before queueing behind EXP-18, retries or suspension. This is a
resource estimate from four outputs, not a deadline or efficacy prediction. It
does not constitute a resource blocker or justify reducing the46 registered pairs.

The only draft amendment corrected prose to the already implemented balanced
arm order. Both draft versions remain preserved. No comparative pilot performance
was used to change the design. All46 outer seeds, eight responses per arm and the
registered primary C−I interpretation are retained, regardless of outcome.
