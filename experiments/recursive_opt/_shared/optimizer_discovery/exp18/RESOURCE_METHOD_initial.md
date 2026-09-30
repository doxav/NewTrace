# EXP18 resource-accounting method

This is a read-only accounting plan, separate from scientific analysis. The long
memory pilot is still pending completion. The short pilot has a completed launch
marker and a passing engineering proof. Final feasibility projections require
both stages; no comparative pilot efficacy is inspected or used here.

## Fixed allocations

The frozen pilots and main driver use B=32, two local seeds per task, 24 TRAIN,
12 validation and 12 audit instances, eight evaluation workers per process,
one shared model-call lock, and a 32,000-token completion cap. A seed-inclusive
candidate pool therefore allocates 48 TRAIN and 24 validation trajectories per
candidate position. The common seed occupies one position in every arm.

| Scope | Completed-response allocation | TRAIN trajectories | Validation trajectories | Audit trajectories | Logical trajectories | Objective allocations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Long M pilot: one outer seed, M ×10 | 10 | 528 | 264 | 0 | 792 | 25,344 |
| Short pilot: one outer seed, L/P/PM ×2 | 6 | 432 | 216 | 0 | 648 | 20,736 |
| EXP18 pilots combined | 16 | 960 | 480 | 0 | 1,440 | 46,080 |
| Main: six outer seeds, L/M/P/PM ×16 | 384 | 19,584 | 9,792 | 864 | 30,240 | 967,680 |

Main formula: `6 × 4 × (16+1) × (48+24) + 6 × (4+2) × 24`.
The two extra audit policies are A0 and fixed B2; they are not additional search
arms. Every search arm has 96 response slots and 7,488 logical trajectories
including its selected-policy audit. The two controls add 288 audit trajectories.
No audit is executed in either pilot. Together with EXP17-E1's four responses,
432 trajectories and 13,824 allocations, all planned engineering pilots total
20 responses, 1,872 trajectories and 59,904 objective allocations.

The main configured completion-cap sum is 12,288,000 tokens, excluding prompts
and transport retries. It is a configuration product, not a realized token or
cost estimate. Generation settings and the exact main seed/order grid remain
those in `driver.main_config()`; this document changes no registered field.

## Logical versus physical work

The cache key includes namespace, exact source hash, task identity, split, outer
seed, local seed, budget, deployment flag, timeout, seed hash and evaluator
version/hash. It excludes arm, allowing identical sources to reuse results across
arms. It does not allow reuse across pilot namespaces or outer seeds.

If every generated source is distinct and differs from the seed, maximum
distinct TRAIN/validation trajectories are
`outer_count × (1 + arm_count × slots) × 72`.
The one shared seed is counted once per outer seed. Main audit adds at most
`6 × 6 × 24` keys; seed-retention and duplicate selections reduce this further.

| Scope | Maximum distinct persisted trajectories | Associated objective allocations |
| --- | ---: | ---: |
| Long pilot | 792 | 25,344 |
| Short pilot | 504 | 16,128 |
| EXP18 pilots combined | 1,296 | 41,472 |
| Main | 28,944 | 926,208 |

These bounds describe the completed cache schedule, not interrupted unpersisted
work. Invalid source/early execution failure retains its allocation and reduces
actual calls; those unused evaluations are not recycled. Deployment fallback
retains actual history and does not increase the objective budget.

Count physical work once per authenticated cache key, using the existing pure
`production.analysis._allocation`: `objective_calls`,
`unused_objective_allocation`, `subprocess_executions`, `execution_s` and typed
invalid trajectory counts. Verify `calls + unused == allocated` for every row.
Report globally and by split. Cache event owners do not provide fair per-arm
attribution of shared seed or duplicate-source work.

Normal successful proposals execute twice for deterministic replay. A failed
first execution may use one subprocess; a failed replay uses two. Audit fallback
can add up to two failed-candidate executions per trajectory, in addition to the
two-per-evaluated-point seed path. Consequently an upper bound on recorded
subprocesses for the main completed cache schedule is
`28,080 × 64 + 864 × 66 = 1,854,144`. Pilot bounds are 50,688 and 32,256.
This is not a wall-time bound. Count recorded subprocesses directly for results.

## Completed-pilot extraction

1. Require the exact registered freeze/configuration, completed launch marker,
   saved passing engineering proof, generation barrier and selection barrier.
   Authenticate source freeze and chronology through existing read-only checks.
   Do not call an engineering runner or metadata collector from this audit.
2. Enumerate exactly each registered outer/arm/slot; read response, request,
   completed attempt records and available provider receipts. Reject duplicate
   plain/gzip artifacts, request/response identity mismatch and missing slots.
   Include invalid, empty and length-finished responses in all resource totals.
3. Read each physical cache artifact once, authenticate full key and row hashes,
   retain invalid partial rows, and require zero audit keys. Count cache-hit/miss
   events separately from logical allocations and unique physical keys. Report
   orphan cache rows or repeated miss events; do not treat event count as a new
   evaluation budget.
4. Reuse pure `_known_usage` for per-field sums and reporting coverage. Preserve
   response-versus-receipt provenance. Read only allocation/validity/resource
   fields from candidate pools; no AUC, winner ranking or efficacy contrast is
   needed. Request/context lengths may be described by arm and slot to audit
   prompt growth, without analyzing optimization performance.
5. Save `resource_summary.json` immutably with input hashes, proof/freeze hashes,
   exact allocation, physical counts, clocks, token/cost coverage and caveats.
   Verify all input hashes before and after extraction. This step evaluates no
   objective and executes no candidate or model.

## Usage, routing and time caveats

- A completed response consumes one proposal slot even if invalid or truncated.
  Preserve every transport attempt separately, plus unresolved started records
  and any reconciliation evidence. Four attempts is the per-invocation bound;
  an explicit later resume can add attempts to an unfinished slot. Do not call
  four times the slot count an absolute lifetime retry bound.
- Existing `_known_usage` keeps response counters when present, filling missing
  prompt/completion counts from matching `tokens_prompt/tokens_completion`,
  reasoning from `native_tokens_reasoning`, and cost from `total_cost`. Native
  tokenizer counts are distinct metadata; do not mix them silently with response
  counts. Missing counters remain unknown, not zero. Report receipts' coverage.
- Reasoning tokens are reported separately and are not added to completion or
  total tokens. Total cost is the reported charge; upstream inference cost is
  separate metadata and must not be added to it. Missing response `total_tokens`
  is not silently reconstructed by the existing helper. Transport failures can
  incur billing not reported in completed-response cost totals.
- Provider receipt IDs must match response IDs. Metadata GET failures are
  auxiliary requests, not generative proposal slots; count their recorded
  failures separately. Do not fetch missing receipts in this offline audit.
- `attempt.wall_s` includes waiting for the shared generation lock, provider
  transport and client handling. Per-slot generation timing also includes retry
  backoff and persistence. Phase generation timing additionally includes local
  TRAIN evaluation and production processing. These durations overlap; do not
  sum them as independent costs.
- Use provider `generation_time / 1000` for the documented millisecond field,
  with the same pinned provenance as
  [EXP17's engineering report](../exp17/ENGINEERING_REPORT.md). Preserve raw
  metadata and coverage; do not infer provider time from queue-inclusive wall
  time or assume a reported time is a strict end-to-end bound. Routing may differ
  between calls under the identical routing policy.
- Report wall, monotonic and boottime phase clocks, detected resets and estimated
  suspension separately. Shared generation queueing is not separately timed;
  subtracting provider duration does not isolate queueing from network or client
  overhead. Independently running studies may overlap local evaluation: eight
  workers per process is not a global limit of eight workers.
- `execution_s` is elapsed trajectory evaluation time, not CPU time. In the
  frozen evaluator it is captured before metric/normalization computation and
  excludes cache persistence, event writes and thread-pool orchestration.
  Summing it over concurrent workers does not equal experiment elapsed time.
- Unique normalization reference design size is 128 points per active task:
  4,608 distinct TRAIN/validation reference values per pilot namespace and 6,144
  across main TRAIN/validation/audit. Process-local caching, process restarts and
  concurrent first misses can repeat physical reference computations; their
  actual count is not fully instrumented. Do not report 6,144 pilot evaluations
  merely because audit tasks are present in the freeze. Numerical verification
  later performs explicitly separate integrity-math calls.

## Main projections after both pilots complete

Do not scale the pooled 16-response pilot average by 384: it overweights M
(10 responses versus two for each other arm), whereas the main has 96 per arm.
For tokens, cost and provider generation duration, report the transparent
arm-balanced calculation `96 × sum(pilot arm mean)` with coverage and sample
counts. Also retain each individual request's size/usage, particularly late M
slots. The short L/P/PM pilot observes only depth two; long M observes depth ten,
while the main reaches sixteen. The seven-source archive cap does not make later
contexts constant-size: typed metadata keeps growing, parent/source sizes vary,
and source omissions can change realized exposure. No pilot validates the full
main depth's realized latency or cost.

For local work, project unique cache rows separately by TRAIN, validation and
audit, multiplying by observed per-row `execution_s` and dividing by an explicit
effective worker-occupancy assumption. Use all eligible and invalid source costs;
do not select the fast or successful subset. Audit has no direct pilot sample, so
state any borrowed cost assumption. Eight continuously occupied workers is an
optimistic scheduling model, not observed guaranteed throughput.

EXP17-E1's four responses already demonstrate the fragility of source-cost mixes:
28,892 reported tokens, USD 0.0055736, 881.569 provider-generation seconds and
3,404.153 summed trajectory seconds. These are context for feasibility, not a
substitute for EXP18's arm/depth-specific evidence. Combine provider and local
projections only as clearly labeled planning scenarios, excluding unknown
queueing, retry billing, suspension and unpersisted work. No projected efficacy,
seed reduction or outcome-driven design amendment follows from this audit.

Pilot freeze hashes checked while preparing this method:

- M: `b8995156cacb2ea2b6821c165a7783116d431913887be27eda33fffd7491fe98`.
- Short: `3f2d24303ec0623c53f13a03a4da9bf632d1c07fd5e7dd35cb1a7bf18ea594ab`.

The allocation table was independently recomputed from `driver.main_config()`
and both registered engineering configurations using integer arithmetic only.
