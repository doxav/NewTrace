# EXP-16 — draft independent validation through production search

This is a draft, not permission for unregistered calls. Freeze the final design,
source/environment hashes and all tasks/settings before this stage begins.

Purpose: validate proposed feedback and evaluation changes in actual iterative
search, and separate feedback information from the production parent-selection
mechanism. Do not assume a favorable F1 result. Use new task and outer-seed
namespaces; do not use F1 validation to choose individual parents or candidates.

Candidate generative arms for the final preregistration:

- I: independent generation from the common unchanged handwritten seed;
- C: production iterative parent selection, current code but no evaluated feedback;
- R: same production schedule with explicit anytime instruction and rich feedback;
- W: same rich feedback with production search breadth (two parents per round),
  four update rounds rather than eight one-parent rounds, at eight responses total.

Every arm receives the same explicit objective, validity rules and invariant task
information. C distinguishes a benefit from evaluated feedback content from a
benefit of repeatedly editing training-selected code. R−I remains the practical
comparison; R−C isolates feedback information; W−R explores search breadth.
No additional critique, repair or model calls. All invalid responses consume slots.

The narrow `trace_schedule.py` adapter calls the existing versioned evaluator,
Control Plane, and PrioritySearch. It changes only declared schedule fields.
Unit tests must verify actual callback counts and parent lineage, not class labels.
Initial tests establish eight callbacks for schedules 8×1, 4×2 proposals,
2×4 proposals, and 4×2 parents. These are engineering tests, not live efficacy.

Intended outer replication: six paired outer seeds; eight responses per arm.
Exact seeds and rotated arm order will be registered before generation. Keep B32
unless new structural evidence justifies a change. Choose task/local-seed panel
replication using S1 measurement precision and cost, not comparative arm wins.
Choose the cap using G1's registered generation-reliability criterion. Any prompt
amendment for execution feasibility applies symmetrically.

All training, validation and independent audit allocations are common across arms.
Freeze every selection before any audit evaluation. Include the seed in all final
pools; retain all candidates even when production search rejects their ancestry.
Use exact-source artifacts, typed invalidity and common permanent seed fallback.
Audit every response, actual/allocated objective call, cache hit, provider attempt,
source hash and selection chronology. Offline evaluation parallelism may be used;
live generation remains sequential.

Primary analysis uses paired outer-seed deployment regret-AUC, not task instances
as independent replications. Report uncertainty, validity, fallback, final regret,
target attainment, tokens/cost, and cumulative training/validation best-by-slot
curves as search-efficiency diagnostics. Do not generalize an outer-search-speed
observation to amortization unless upstream training costs and future task reuse
are measured directly.

This stage may confirm benefit, identify a negative interaction, or show that the
instrument corrections still do not establish feedback superiority. All outcomes
must inform the final recommended rerun; no additional stage is triggered merely
to reverse an unfavorable result.

## Choices supported by the completed diagnostics (still draft)

S1 supports 24 training instances ×2 local seeds for a stable training ranking
within its fixed bank. Use 12 separate validation instances ×2 local seeds as a
common final selection panel, and 12 separate audit instances ×2 local seeds,
subject to the measured T1 execution feasibility. This is 48 training and 24
validation trajectories allocated per candidate. Every arm has the seed plus
eight proposal slots. Before cache savings or early invalid termination, six
outer seeds ×four arms ×nine pool entries ×72 trajectories =15,552 allocated
train/validation trajectories, or497,664 objective evaluations and995,328
deterministic-replay subprocess executions. The independent audit adds
6×5 policies×24 trajectories =720 trajectories,23,040 objective evaluations
and46,080 subprocess executions. Normalization reference evaluations and any
preflight/pilot evaluations are accounted separately. Repeated Trace cache reads
do not consume fresh scientific allocations or permit more model responses.

Use the unchanged three objective families and B32. B1 establishes headroom
without changing the scientific target; no family is removed because A2 performs
poorly on it. Omit OOD in this production diagnostic. All local seeds and task
instances come from a new P1 namespace.

The intended six outer seeds are16411,16423,16437,16441,16453,16467. Four arm
orders rotate by outer index. Live calls remain sequential. A2-style rich arms
receive the actual propagated current-parent Trace feedback, including its source
hash, indexed progress, incumbent and improvement events. An aggregate training
AUC scalar may be included to align the model with the canonical ranking; no
individual normalization constant, optimum or normalized/raw per-task pair is
provided. This is a combined instrument correction beyond the raw-only F1 factor,
and must be stated as such. It does not test a complete archive of rejected
programs or additional nested meta-optimization depth.

Before this run, perform a separately registered two-response engineering check
of the R production path at the final panel and prompt sizes. It must use new
engineering seeds/tasks and must not inspect P1 audit evaluations. Its purpose
is request-size/timeout/recorder/schedule feasibility, never choosing a candidate
for P1 or changing settings because a proposed optimizer wins.

Predeclare paired outer-seed bootstrap contrasts R−I (central),R−C,W−R andR−A0.
All contrasts are exploratory at n=6; intervals are fragile and multiple outcomes
are reported together. Keep each signed delta and invalid/fallback outcome.
A negative delta means lower regret-AUC. Do not treat task observations as outer
replications. This is the final prospective efficacy diagnostic in EXP-16: an
unfavorable outcome triggers analysis and an honest conditional recommendation,
not another search stage designed to reverse it.
