# Independent prospective review: EXP-17 and EXP-18

Review date: 2026-09-10. This is a design review, not the final registration or
an execution report. It reads completed EXP-16 evidence and makes no model,
candidate, or scientific objective calls. The root experiment owner remains
responsible for the final manifest, tests, engineering pilot, freezes and runs.
The [supporting calculations](design_review_calculations.json) preserve the input
file hashes and all numerical planning assumptions. No earlier result is changed.

**Recommendation: execute a fresh 46-pair C-versus-I confirmation, and a separate
six-pair exploratory factorial test of archive memory and parent selection.**
Neither experiment should be adapted in response to a favorable or unfavorable
audit result from the other. This respects the newly authorized scope without
making a positive result the condition for completion.

## EXP-17: the precise confirmation

Use I, eight independent responses, and C, eight production updates from the
training-selected parent without explicit evaluation feedback text. Keep their
invariant information, original source seed, model settings, validity rules,
candidate allocations and validation selection identical. Preserve A0 and B2 as
fixed deployment controls. B2 is neither the seed nor an extra selection-pool
candidate for I/C. The original trusted seed remains the common fallback; changing
the fallback to B2 would alter the deployment estimand.

Use 46 new outer seeds and a new frozen 24 TRAIN / 12 validation / 12 audit panel,
with two common local optimizer seeds per instance and outer seed. Keep B=32,
the three families, dimensions 2/4, six equal family/dimension weights, and the
same independent 128-point normalization reference design procedure. Do not pool
EXP-16's six post-hoc C−I pairs with these new observations.

The single confirmatory primary contrast is the paired mean **C−I in audit
normalized anytime AUC**. Negative favors C. Reuse the exact frozen percentile
bootstrap implementation: 10,000 outer-pair resamples, Python `Random(1515)`,
interpolated 2.5/97.5 percentiles, same registered outer-seed order. Record a
positive signal if the interval lies wholly below zero, negative if wholly above,
inconclusive otherwise, with all-zero results explicitly reported. Crossing zero
does not demonstrate equivalence. A zero-sided sign or a favorable sample ordering
is not enough to claim superiority.

Register C−A0, I−A0, C−B2, I−B2 and B2−A0 as descriptive secondary contrasts,
alongside final regret, attainment/censoring, invalidity, fallback, tokens and cost.
Their intervals are individual descriptive intervals, not simultaneous confidence
bounds. A single confirmatory contrast needs no multiplicity correction for a
family of one; this does not license inferential claims from whichever secondary
contrast looks most favorable. If additional confirmatory claims are desired,
their multiplicity procedure must be frozen before any execution.

The planning effect of interest is **0.02 AUC**. Distinguish detecting any benefit
from establishing a benefit larger than 0.02. An upper confidence bound below zero
supports a favorable direction under the registered procedure. An upper bound
below −0.02 would support an effect exceeding that practical threshold. Power
planned for a true effect of −0.02 against zero does not imply 80% power to prove
the effect is more favorable than −0.02.

The EXP-16 post-hoc C−I sample SD is approximately 0.048137. Under a normal,
independent-pair approximation with stable future variance, a two-sided 5% test
and nominal 80% power at |delta|=0.02 gives **46 total new pairs**. The approximate
power at n=46 is 80.46%; the approximate normal interval half-width is 0.01391.
These are conditional planning calculations, not a guarantee of bootstrap power,
coverage, service stability, or a future gain. The variance estimate comes from
only six pairs. Effect scenarios 0.005/0.01/0.02/0.03 yield 728/182/46/21 pairs.
Do not shrink the 46 after seeing any new comparative results.

All 46 pairs share one new audit panel. They replicate generation and local-seed
variation conditional on that frozen panel; they do not create 46 independent
benchmark panels. Generalization claims must retain that scope. Instances and
trajectory points must not be counted as independent outer replications. Frozen
additional panels could broaden the claim, but would be a separate registered
design choice, not an addition after observing the primary outcome.

Balance whole-arm order with 23 I→C and 23 C→I sequences, arranged in 23 adjacent
inverse pairs. Freeze which order starts each block using a separate ordering
namespace. This prevents early/late precedence from being concentrated in one
arm. Slot-by-slot interleaving would further reduce temporal separation, but is
not necessary if it would require a second engine or destabilize the existing
production scheduler. Never condition the order on observed quality or provider.

## EXP-18: separate mechanisms, explicit interventions

The reviewed final direction uses a 2×2 factorial. All four arms share the current
parent source and the same compact, deterministic current-parent TRAIN aggregate
AUC plus typed validity/count information. No arm receives the P1 rich trace by
default. Each receives 16 completed responses at the same model and evaluation
settings. Use six new paired outer seeds and an entirely separate frozen panel.

| Arm | Parent policy | Additional historical memory |
|---|---|---|
| L | Scalar best TRAIN AUC | None |
| M | Scalar best TRAIN AUC | Up to seven recent distinct prior attempts |
| P | Uniform sampling from a per-instance TRAIN Pareto frontier | None |
| PM | Same per-instance Pareto policy | Same historical-memory rule |

Calling the common baseline L avoids implying it is identical to C in EXP-17:
it now receives compact explicit current-parent feedback and has N=16. EXP-18
estimates the incremental effect of the archive and the joint effect of frontier
construction plus its sampling policy. It does not isolate every possible memory
encoding or every possible Pareto strategy.

The three principal exploratory paired estimands, calculated within each outer
seed before aggregation, are:

- memory: `((M − L) + (PM − P)) / 2`;
- Pareto parent policy: `((P − L) + (PM − M)) / 2`;
- interaction: `PM − P − M + L`.

Also register the simple effects M−L, PM−P, P−L and PM−M, plus combined PM−L.
Always report these when interpreting an interaction: a mean main effect can hide
opposite effects under the other factor. The entire factorial remains exploratory
at six outer seeds. Use the same frozen paired bootstrap scheme and explicit
small-sample limitations; do not make a familywise significance claim from several
unadjusted intervals. Add A0/B2 as fixed audit controls and report their comparisons
descriptively. Without an N=16 I arm, **EXP-18 cannot establish that memory or
Pareto beats equal-budget independent generation**. EXP-17's N=8 I is not an
equal-budget comparator for that claim.

An example six-sequence schedule is below. Every unordered arm pair has 3/3
precedence, and every arm occupies each position one or two times. The calculation
JSON records the same L/M/P/PM labels.

```text
L  M  PM P
P  PM M  L
M  P  L  PM
PM L  P  M
L  P  M  PM
PM M  P  L
```

Do not use EXP-17 audit outcomes to alter EXP-18 prompts, memory formatting,
frontier policy, or budget, or vice versa. Prefer freezing both designs and their
analysis code before opening either audit. At a minimum maintain complete
experiment-specific generation→selection→audit barriers and record chronology.

## Memory: what must be demonstrated

Adding shared current-parent feedback to all factorial arms is a useful correction
to a potential confound: M−L then measures the archive increment, rather than
simultaneously introducing all explicit performance information. Its interpretation
still includes historical code, outcome information, formatting and longer prompts
as one declared intervention. Isolating historical code from historical outcomes
would require another control and is not necessary for this bounded experiment.

The proposed cap is seven recent distinct prior attempt sources and 65,536
characters for the complete archive section. Freeze chronological ordering,
deduplication, handling of responses without source, parent-source duplication,
oversized sources and explicit omissions before the pilot. Do not silently clip
source and represent it as a complete program with the original hash. Preserve
all prior attempts in evidence even when the prompt cap excludes them. A seven-slot
history is not necessarily seven distinct sources, seven valid sources or seven
actual rejections.

Every included entry must align exact source or explicitly identified excerpt,
slot, response hash, parent hash, outcome, typed failure and only already available
TRAIN evidence. Numeric TRAIN AUC exists only for a fully valid panel. A missing
or invalid panel stays typed, without an invented penalty. The word "rejected"
requires a recorded search decision; "not the current parent" is insufficient for
an archive/frontier policy. Preserve actual decision provenance on the host.

Do not expose task IDs, families, hidden parameters, optima, normalization constants
or per-instance normalized/raw score pairs to the generator. Aggregate TRAIN AUC
may be exposed under the declared schema. The per-instance Pareto score vector can
remain host-only, influencing parent selection without adding a different vector
feedback prompt to P/PM.

Build archive snapshots only from the same arm/outer seed and prior completed
slots whose TRAIN results were available at the registered cutoff. Reconstruct
the same snapshot after resume; do not accidentally include later evaluations or
another arm's cache. This uses production callbacks and existing evaluations,
without extra LLM criticism, repair, or objective work. Missing expected receipts
are a checkpoint defect, not permission to fabricate history.

At N=16, nine requests can have at least seven previous responses, compared with
one at N=8. That is a availability ceiling, not a promise of seven complete examples.
Report memory entry counts, distinct sources, typed rejected entries, omitted
entries/characters, context size and the number of actual live requests exercising
each occupancy level. Do not force invalid proposals merely to fill a rejection
archive.

## Pareto: make the policy real and reproducible

Use one dimension per TRAIN instance: the mean normalized AUC across its two local
seeds. All candidates already incur the same complete 24×2 TRAIN allocation, so
this adds no evaluation budget. Exclude invalid panels from the eligible frontier
without converting them into extreme score vectors. Include the trusted seed.
Use only already completed same-arm attempts. Retain every proposal, including
dominated candidates, in the final validation selection pool.

Freeze minimization direction, finite-score checks, the exact dominance predicate
and tolerance, duplicate-source handling, tied vectors, candidate IDs, and the
uniform sampling RNG namespace. Sorting candidates by stable slot/hash before
sampling prevents dictionary order from changing results. Frontier membership must
use the 24-dimensional vector; a scalar AUC archive given a Pareto class name is
not this intervention. Equivalent vectors and repeated sources must not silently
receive multiple probability mass unless explicitly registered.

A 24-objective frontier may contain most candidates. Uniform frontier sampling
therefore combines avoiding dominance with a potentially broad parent distribution.
It is not guaranteed to preserve enough pressure for the aggregate objective, and
it is not the same strategy as sampling instance champions proportional to wins.
Keep the selected policy fixed and report frontier size, non-dominated fraction,
distinct parent hashes, source lineages and how often a scalar-best parent was
chosen. An empty/singleton frontier yields a legitimate limited intervention, not
grounds to rerun the model or add proposals.

## Engineering pilot and gates

Use the actual existing Trace/Control Plane generation path and no new search
framework. Unit tests can script candidate responses; all genuinely live pilot
calls must be individually identified. Test accounting through real callbacks,
archive-on/off, unchanged parent after an invalid child, scalar/Pareto divergence
on a known two-instance fixture, invalid exclusion, frontier ties/duplicates,
source/hash alignment, protected split access, bounded omission, local replay and
resume without extra completed calls. Scripted 16-step paths are especially useful
for deterministic memory occupancy, frontier exercise and repeated crash/resume.

The owner has selected a **20-response engineering allocation**. EXP-17-E1 runs
I2/C2 on a separate full 24×2 TRAIN panel and 12×2 validation panel, with validation
only after all pilot generation. EXP-18-E1 runs M10/L2/P2/PM2 in separate frozen
engineering namespaces, each on its full 24×2 TRAIN panel. There is no replacement
for a completed invalid output and no comparison of pilot quality to choose the
protocol. Audit values are not part of this gate.

M10 can naturally accumulate seven prior responses for its last three requests;
report the actual count after source deduplication, source absence and omission.
PM2 exercises combined archive/frontier wiring but does not validate the natural
long-memory PM regime. Long PM contexts and 16-step resume behavior are tested with
explicitly scripted fixtures. That is a real limitation to report, not a claim
that seven live PM examples were observed. The shared memory formatter and its
identical configuration should be verified across M/PM.

Gates concern actual production callbacks and request settings, fixed budgets,
checkpoint/resume, memory receipt identity, frontier correctness on deterministic
fixtures and whatever live frontier variation occurs. The owner also requires at
least one valid generated replacement across the engineering search; this is a
minimal executability check, not a statistically estimated validity-rate threshold
or a requirement for each arm to win. All live outputs, including poor or invalid
ones, remain evidence. A pilot defect requires a documented engineering fix and
another separately allocated relevant check; ordinary invalid generation must not
be relabeled an infrastructure bug.

EXP-17 may start after its own pilot and immutable base-adapter freeze. Subsequent
EXP-18 extensions must be separate modules, leaving all frozen EXP-17 dependencies
unchanged. A common live-call coordinator must enforce concurrency one across both
experiments and their pilots, even if offline development/evaluation proceeds in
parallel. A protocol or preflight file existing is not evidence that this coordinator
or a suspended process is actually running; inspect the live process and completed
slot evidence before any resume.

The pilot must measure serialized prompt feasibility under the unchanged 32,000
completion ceiling, low reasoning, 300-second configured request timeout, eight
local workers, sequential live calls and ordinary common provider routing. The
timeout is not a proven hard total wall-clock deadline: P1 contains request spans
well over 300 seconds. Preserve attempt, monotonic and host-suspension timing.
Do not select a prompt or frontier policy because it happened to win pilot AUC.

## Resource feasibility from observed P1 evidence

The counts below assume the original seed is a logical pool member in every
generative arm, every pool entry has 48 TRAIN and 24 validation trajectories, and
each final deployment arm has 24 audit trajectories per outer seed. They are
allocations, not promises of realized objective work; invalidity and complete-key
caching reduce actual work and never create extra proposal slots.

| Resource | EXP-17: 46×2×8 | EXP-18: 6×4×16 |
|---|---:|---:|
| Completed-response slots | 736 | 384 |
| Logical TRAIN trajectories | 39,744 | 19,584 |
| Logical validation trajectories | 19,872 | 9,792 |
| Logical audit trajectories, including A0/B2 | 4,416 | 864 |
| Total logical trajectories | 64,032 | 30,240 |
| Allocated objective calls, B=32 | 2,049,024 | 967,680 |
| Physical objective bound, only shared-seed cache savings | 1,943,040 | 926,208 |
| Matching deterministic replay subprocess bound | 3,886,080 | 1,852,416 |
| Unique reference-design objective values | 6,144 | 6,144 |

The physical bounds assume each generated source is unique and valid, no deployment
fallback or interrupted/retried execution, and one shared seed evaluation per
outer/task/local seed. Actual fallback, resume or verification overhead must be
accounted separately. Reference-design reconstruction frequency also needs its own
counter rather than assuming unique points equal physical evaluations. Reuse of
the cache is valid only with the complete frozen source/task/local-seed/B/evaluator
key; references from another arm must not leak into its prompts.

Linear scaling of P1's I+C generation-slot timing gives **37.29 active hours** for
EXP-17 requests, before local work outside those slot spans, pilot, extra retries,
integrity verification and host suspension. The same-arm reported usage extrapolates
to about **5.89 million total tokens and USD 1.0292**. These are historical-rate
scenarios, not current prices or provider guarantees. The current model may route
differently, long completions occur and transport-failure billing can be unknown.
Budget runtime in days, not a short interactive session, and checkpoint every slot.

EXP-18's 384 responses lie between historical C-like and R-like aggregate context
scenarios: roughly 3.63–25.42 million total tokens and USD 0.672–1.464 under those
past rates. This wide range is illustrative, not an actual memory-prompt bound;
the pilot should replace it with measured prompt size and context-specific usage.
Historical C/R per-response active timing corresponds to approximately 19.37–28.32
request hours for 384 responses, again excluding local work and future drift.
No affordability evidence here warrants reducing EXP-17 below its 46-pair target.

## Completion and interpretation

Before live execution, freeze exact prompts/formatters, source code/environment,
seeds/splits, normalization, proposal and allocation schedules, parent/archive
policy, deployment fallback, tie-breaking, analysis algorithms and resume rules.
Require a machine-checkable preflight against these hashes. Every completed empty,
truncated, unparsable or poor response consumes its registered slot. Preserve all
transport attempts and ambiguity about remotely completed requests. Do not add
automatic uncounted repair or empty-response retry calls.

Freeze every final selection and representative from validation before any audit
opens. Audit failures switch permanently to the original seed with actual history
and the remaining objective budget; report candidate-only validity and fallback.
Verify every source/hash, all 46 or six outer seeds, every registered slot and the
exact model/settings. Recompute metrics/contrasts from preserved rows, independently
audit numerical objectives, and update the reports even for negative findings.

EXP-17 can provide new evidence for or against the C−I signal under its precise
benchmark and budget. EXP-18 can identify which additional mechanism is worth a
later confirmation, or show that neither helps. Neither implies algorithmic
novelty, an advantage from recursion depth, successful transfer to other APIs,
amortization, an operating-system security sandbox, or success of every possible
recursive_opt meta-optimization strategy.
