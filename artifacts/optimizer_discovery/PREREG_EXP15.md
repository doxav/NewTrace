# EXP-15 — pilot protocol and draft confirmatory design

Registered before implementation/pilot outcomes. Baseline 90651f47. Scientific
question: does iterative training feedback improve deployed optimizer-program
regret-AUC against the unchanged seed (H15-A) and independent best-of-eight (H15-B)?
H15-B is central. No claim of novelty, recursion-depth benefit or amortization.
The machine-readable exp15_manifest.json is authoritative for numeric settings.

## Fixed design and permitted pilot amendments

One handwritten seed (exact source/hash in manifest) mixes 25% uniform exploration
with incumbent Gaussian perturbations, decreasing step scale with history length.
Freeze it before comparative pilot outcomes. Pilot outer seed 701, two proposals
per arm; confirmation outer seeds 11/23/37/41/53, eight per arm (80 slots total).
Pilot and confirmation task domains are separate. No OOD. Pilot may inform only
engineering feasibility, structural headroom/saturation and symmetric generation
settings; preserve all revisions and pilot evidence. Never choose settings by A2's
comparative success. Final confirmation freeze must precede all confirmatory calls.

## Benchmark and metrics

Bounds [-5,5]^d; d in {2,4}; families Sphere, axis-aligned Quadratic, Rosenbrock.
Each instance independently draws shift coordinates uniform [-2,2], coordinate
scales uniform [0.75,1.5], and amplitude 10^U(-1,1). Let z_i=(x_i-shift_i)/scale_i.
Sphere: amplitude*sum(z_i^2). Quadratic: amplitude*sum(w_i*z_i^2), with weights
10^U(0,3). Rosenbrock uses y_i=1+z_i and amplitude*sum(100*(y_(i+1)-y_i^2)^2+
(1-y_i)^2). Known feasible optimum is x=shift, value zero. Distinctness checks
compare semantic task parameters, not IDs. Six train, six validation, twelve ID
holdout instances, balanced over family/dimension. Seeds use SHA256 and separate
domains for phase, split, instance, normalization, local randomness and requests.

Each trajectory has B=32 objective calls. One local seed per outer-seed/task pair,
shared by every candidate and arm. Normalization s_j is mean objective excess on
128 independently seeded uniform reference points, shared preparation charged
separately. No candidate receives objective source, family, parameters, optimum,
normalization or split. Regret r_j(t)=(best_so_far-f*)/s_j; AUC_j=mean over t=1..B.
No upper clipping. Negative normalized regret within 1e-12 rounds to zero; larger
negativity is a benchmark defect. s_j must be finite/positive. Aggregate instances
within each of six family/dimension strata, then average strata equally. Aggregate
tasks before paired inference across outer seeds. Report final regret and target
0.01 attainment; nonattainment is censored at B, with B+1 only a labeled convention.

## Validity and deployment

Reuse OptimizerProgramV0, fresh subprocesses, exact signature, legal finite output,
two-run deterministic proposal replay, sanitized environment and 2s timeout. Apply
the same declared AST screen to every source: only listed stdlib imports; reject
relative imports, dunder introspection and named external-I/O/dynamic-execution
builtins. Document its precise implementation before confirmation. This is a
protocol screen, not a security sandbox; filesystem/network confinement is absent.

An invalid response consumes its slot. Train/validation trajectories stop on
invalidity and retain partial observations without numeric metric imputation.
Evaluate every allocated train and validation task, including after another task
fails; do not recycle unused allocations. Final eligibility requires all twelve
trajectories valid. Both final pools include seed at index -1. Select minimum
validation AUC, breaking ties by earliest proposal index. If none qualifies, seed
remains selected. Holdout failures permanently switch that trajectory to seed,
continuing the actual history and remaining objective budget. Record the failure,
fallback, actual evaluated values, and candidate-only validity. Seed/fallback/
benchmark failure is infrastructure failure, not scientific underperformance.

## Arms, production integration and information control

A0 is seed. A1 receives only the identical invariant contract, generic benchmark
description, seed and restrictions in fresh requests. A2 additionally receives its
current source and deterministic bounded training-only feedback. Use production
Control Plane's trainable source component, evaluator and PrioritySearch/Trace
update path; a narrow Optimizer adapter may convert one Trace proposal into one
registered model request. No bespoke recursive search loop, LLM critique or repair.
Only training results drive the production search incumbent. Feedback exposes
per-task raw observed x/value samples (first/last two), validity and raw best values;
no hidden constants, task identities or normalization constants. The host may use
normalized training AUC for ranking, but that scalar is not supplied to the model.

No validation/holdout dataset is supplied to the production generation plan.
Validation occurs after both arms finish generation for an outer seed. Freeze all
ten final selections and the representative A2 source before any holdout evaluation.
Representative is minimum validation AUC across outer seeds, ties by listed seed
order. Phase-1 task labels are host-side only.

## Budgets, persistence and uncertainty

Use selected Phase-0 generation settings exactly. Eight completed responses per arm,
including truncated/empty/invalid responses; only transport failures receive three
2/4/8s retries. Disable SDK and empty-output retries. Resume uncompleted slots only.
An in-flight request with uncertain remote completion is recorded and reconciled;
never silently reissue it. Cache complete deterministic evaluations by source hash,
semantic task, phase/split, local seed, B, deployment mode and evaluator/code version.
Record every logical request/cache hit, actual objective call, unused allocation,
and subprocess proposal/replay separately. Additional internal Trace evaluations
must be cache hits and never add scientific opportunities. Raw evidence is immutable
JSON (including exact evaluated source), with separate presentation exports if needed.

Predeclare A1 then A2 at even outer-seed indices and reverse at odd indices. Keep
LLM calls serial. Same routing policy; actual tokens/costs are not equalized and must
be reported. Persist upstream provider metadata when available.

Paired differences A2-A0 and A2-A1: negative favors A2. Use 10,000 paired bootstrap
resamples of the five outer-seed deltas, random.Random(1515), linearly interpolated
2.5/97.5 percentiles. Entire interval below zero: positive signal; above zero:
negative signal; all deltas zero: no detectable difference; otherwise inconclusive.
Small-n intervals are fragile; no statistical-significance claim. Report every seed,
means, medians, deltas, validity, fallback, target censoring, calls and costs.

## Pilot, freeze and completion

Before live pilot, test benchmark optima/splits, metrics, accounting, AST screen,
fallback, source identity, persistence/resume, production A2 and split isolation.
Fixed diagnostic policies (seed, uniform, midpoint) assess headroom without selecting
on A2>A1. Re-evaluate one pilot seed trajectory ten times for an observed repeatability
range, separate from search. Pilot exercises both real generative arms and at least
two iterative A2 prompts. Estimate full objective calls, subprocesses, tokens, cost
and runtime. Commit a final manifest with code/prompt/environment hashes and a
machine-checkable preflight. No confirmatory semantics may change thereafter.

Completion accepts any valid comparative result, including fallback-dominated
search. Retain all slots/seeds, raw source hashes, lineage, frozen selections, full
holdout outcomes, recomputable aggregates and a <=2-page Patrick brief. No merge,
push, PR or contact. Preserve historical conclusions and original Phase-0 evidence.
