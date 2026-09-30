# EXP17-E1 engineering pilot

The registered four-response I/C pilot passed its engineering gate. All responses
and allocations remain preserved, including one ineligible program. This is
readiness evidence, not a comparison of optimization performance.

| Check | Observed evidence |
| --- | --- |
| Exact model responses | Four completed, four transport attempts; no replacement or transport retry |
| Source parsing/screening | Four valid sources; all responses ended with `stop` |
| Full TRAIN and validation eligibility | Three of four generated candidates eligible |
| Actual production C | Two update callbacks, authenticated source parents and Trace records |
| Split protection | Global generation/selection barriers verified; no audit cache or evaluation |
| Resume | Completed run replay needs no client or new evaluator work; preserved bytes identical |
| Behavior | Four complete TRAIN-valid sources including the seed have four distinct point-trajectory signatures; this does not establish headroom |
| Usage | 2,980 prompt, 25,912 completion, 28,892 total reported tokens; reasoning counter22,261 retained separately, not added to total |
| Cost | USD0.0055736, reported for all four responses |

The logical seed-inclusive schedule allocates432 trajectories and13,824 objective
calls. Source reuse gives360 unique cached trajectories,11,520 physical objective
allocations,11,036 actual objective calls,484 unused allocations and22,143 actual
candidate subprocess executions. TRAIN contains240 unique trajectories, of which
48 are invalid; all120 validation trajectories are valid. Invalid TRAIN eligibility
is not repaired by success on separate validation instances.

The first independent program is the ineligible source. It repeatedly rebuilds a
GP covariance matrix and factorization for400 acquisition points. Its48 TRAIN
trajectories contain25 timeout and23 nondeterministic labels. The latter label can
also mean a failed second execution in the unchanged replay runner; it does not
alone prove stochastic source behavior. All24 separate validation trajectories of
that source completed validly. A registered diagnostic on one exact failed input
also succeeded subsequently. These observations show runtime dependence and extra
computation, but do not isolate the system cause or replace the original failures.
See [the bounded diagnostic](engineering_diagnostics/timeout_01/REPORT.md).

Generation lasted1,759.61 monotonic seconds and selection99.42s, with no detected
suspension. Shared provider-lock queueing is included in request wall times.
The sum of provider-reported generation duration is881.569s; the metadata field's
millisecond unit is documented in the [official OpenRouter SDK](https://raw.githubusercontent.com/OpenRouterTeam/typescript-sdk/v0.9.11/src/models/operations/getgeneration.ts).
Routing identities were OpenInference, DeepInfra and Inceptron under the same
unmodified routing policy. No route was selected for an arm.

Scaling the four responses to736 yields5.316M reported tokens, aboutUSD1.026 and
45.06h of reported provider generation. A separate local-execution calculation,
using the observed source-cost mix and eight occupied workers, adds approximately
19.94h. These are fragile projections from four outputs, not bounds or deadlines.
They exclude further queueing, retries, host suspension and incomplete physical
normalization accounting. The expensive first source makes a large contribution
to this local estimate. No efficacy projection is made from this pilot.

The unchanged benchmark already has independently measured headroom in EXP-16;
new behavioral diversity is not used as a substitute for that evidence. No model,
cap, timeout, worker count, task, seed, metric, selection, budget or analysis change
was made after the pilot. EXP-17 retains46 paired outer seeds and8 responses per
arm. The only draft amendment corrected the wording of the already implemented
balanced order; both earlier versions remain preserved.

Evidence: [gate](engineering/engineering_results.json),
[resources and formulas](engineering/resource_summary.json),
[behavior signatures](engineering/behavior_diversity.json),
[frozen sources](engineering/source_snapshot.json),
[confirmation manifest](exp17_manifest.json).
