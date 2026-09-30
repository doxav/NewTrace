# EXP-16 / P1 — frozen analysis specification

This fragment and `analysis.py` must be included in `S.prepare(extra_frozen_paths=...)`
before the first P1 live response. It supplements `PROTOCOL_P1.md`; it does not
introduce a new experimental arm, outcome-dependent rule, or confirmatory claim.

## Replication, aggregation and interpretation

The six registered outer seeds are the replication units. For every seed retain
all five deployment arms, A0/I/C/R/W. A trajectory's preserved regret curve has
32 entries; its arithmetic mean is normalized anytime AUC. Average trajectories
within each family/dimension stratum, then average the six strata equally, using
the unchanged `benchmark.aggregate`. Lower AUC is better. Do not clip large regret
or assign numeric train/validation scores to invalid trajectories.

Report all seed values, mean and median AUC per arm, and the four signed paired
contrasts R−I (central), R−C, W−R and R−A0. Reuse `exp15.paired` exactly: draw 10,000
bootstrap resamples of the paired outer-seed differences with Python
`random.Random(1515)`, each of size six with replacement, compute each mean, sort,
and interpolate linearly at percentile positions `(10000−1)×p/100`, p=2.5 and97.5.
Use the same outer-seed ordering and resampling algorithm in every contrast.
An interval wholly below zero is a descriptive positive signal for the first arm;
wholly above zero is a negative signal; all exactly zero differences mean no
detectable difference; otherwise the result is inconclusive. These are exploratory
small-sample labels, without a multiple-comparison guarantee or significance claim.
Do not suppress contradictory outcomes or convert an interval crossing zero into
proof of no effect. No post-hoc sample-size or metric adjustment is permitted.

Final regret, target0.01 attainment, target hitting times with nonattainment retained
as `None`, B+1 capped/censored summaries, candidate/trajectory validity, fallback,
source bytes, AST node counts and execution time are secondary/descriptive outputs.
A parser resource failure yields unknown AST complexity while retaining the exact
source size and invalid response. An invalid program is not dropped from the
response, eligibility or allocated evaluation denominators. Generated execution
validity uses generated candidates only; the trusted seed is excluded from that
denominator but included in seed-inclusive logical evaluation allocations.

## Selection and source integrity

Recompute eligibility from every train and validation row. Recompute each final
choice from the seed at index−1 plus all eight proposal slots, taking minimum
validation AUC then earliest index. Verify the saved source and SHA256 against the
response and evaluated rows. Count seed-index selection, seed-source selection,
zero eligible generated replacements, and seed selection despite eligible generated
alternatives separately. Recompute the R representative by minimum selected
validation AUC, then registered outer-seed order. Never use audit values for this.

Retain every request parent hash and response source hash as lineage edges. Any
prefix curve uses the same eligible pool restricted to response indices already
available at that prefix, with the seed always included. Its monotonic improvement
is a property of best-of-N validation selection and is not an independent estimate
of generalization or gains from a larger future response budget.

The common seed fallback completes failed audit trajectories using real accumulated
history and the remaining budget. Deployment AUC retains those real values;
candidate-only failures and fallback frequency remain visible alongside it.
Trusted-seed/fallback/evaluator failures cannot be relabeled as candidate outcomes.

## Resource accounting and completeness

Require every registered seed, arm, response slot, pool candidate and task/local-seed
trajectory. Refuse duplicate provider response IDs, source/hash mismatches, changed
validation decisions, missing cache records, unallocated physical cache rows, extra
completed response paths and unreconciled in-flight requests. Verify generation,
selection and audit chronology using the production owner's frozen checks. Analysis
never calls the objective, candidate subprocess, normalization reconstruction, or LLM.
Metric checks recompute AUC, final regret and target attainment from preserved
normalized curves; they do not independently reconstruct normalization constants
from raw objective observations and the reference design.

Distinguish logical allocation from physical cache execution. Every seed-inclusive
pool entry has its own registered train/validation allocation, even if its source
is duplicated or invalid. Report objective calls actually made, unused allocation,
subprocess executions and invalid trajectories for those logical references, then
report corresponding totals across unique persisted cache records by split.
Cache accesses/hits/misses are separate; repeated misses or missing miss events are
surfaced rather than silently reconciled. Audit uses the I owner as a technical
delegate for all five arms. That event owner is not scientific cost attribution.

Unique normalization designs require 48×128=6,144 reference evaluations. Physical
reference reconstruction across processes or resumes was not fully instrumented;
6,144 is the unique design count, not a claim of exact physical invocation count.
Similarly, physical cache totals cover completed persisted evaluations; work lost
before checkpoint persistence may incur additional unknown objective/subprocess
cost. Retain that limitation and any observed interrupted-execution evidence.

Count 192 completed response allocations separately from all transport attempts,
transport failures and uncertain remote completion/duplicate-billing attempts.
Report prompt/completion/reasoning/total tokens and monetary cost for each arm and
overall with both reported sums and missing-response coverage. A missing value is
unknown, not zero. Response usage has priority; matching safe provider receipts may
fill missing fields only, never add a second copy of the same response's cost.
Preserve per-field provenance. Native reasoning tokens, provider-reported cost and
routing identities remain labeled; do not infer unreported tokens or cost from
prices. Keep any transport-failure usage separate, since remote completion or
billing may be uncertain. Equal response counts are not equal realized token costs.

Retain phase and per-slot timing clocks and transport durations. Host suspension
can affect realtime and monotonic clocks differently; no raw timing sum is a clean
provider-latency estimate without inspecting those records.

## Reproducibility

`python -m artifacts.optimizer_discovery.investigation16.production.analysis ROOT`
reads the frozen completed run and writes an immutable `analysis_results.json`
(or `.json.gz` when required). Rerunning it with unchanged inputs must reproduce the
same JSON values. Optional later provider receipts may change usage coverage only;
write a distinct named reporting revision rather than overwrite completed output.
No scientific values, selections, primary contrasts or interpretations change when
only receipt coverage changes.

The tests use synthetic evidence only. They cover missing/unfavorable/invalid rows,
all six outer seeds, source/hash corruption, candidate versus deployment validity,
fallback without budget reset, seed selection despite an eligible alternative,
unknown cost coverage, receipt identity and double counting, compressed path
handling, extra physical evaluations, and prohibition of objective evaluation from
the analysis layer. Freeze tests and implementation hashes with the run provenance.
