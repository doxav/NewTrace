# EXP-18 engineering readiness

Both separately registered pilots passed. This establishes readiness to run the
fixed four-arm study; it does not establish a performance benefit from memory or
Pareto parent selection. No completed response was replaced or manually repaired,
and no audit trajectory was evaluated.

| Observation | Long memory pilot | Short L/P/PM pilot |
| --- | ---: | ---: |
| Completed responses / transport attempts | 10 / 10 | 6 / 6 |
| Sources passing parsing and screening | 10 | 5 |
| Generated candidates eligible on all TRAIN and validation trajectories | 10 | 5 |
| Length responses without source | 0 | 1 |
| Current-parent contexts authenticated | 10 | 6 |
| Audit trajectories | 0 | 0 |
| Reported prompt tokens | 46,885 | 7,235 |
| Reported completion tokens | 75,337 | 60,276 |
| Reported total tokens | 122,222 | 67,511 |
| Reasoning tokens, included component | 66,450 | 50,930 |
| Reported cost, USD | 0.028116664 | 0.021010768 |
| Unique cached trajectories | 792 | 504 |
| Actual objective calls | 25,344 | 13,824 |
| Unused physical objective allocations | 0 | 2,304 |
| Candidate subprocess executions | 50,688 | 27,648 |

Every response has usage and a matching provider receipt. Reasoning tokens are not
added again to total tokens. The short pilot's empty-source response exhausted
32,000 completion tokens; it remains an invalid allocated slot. A larger cap does
not guarantee code. No setting was changed in reaction to that response.

The long pilot's final context contains all nine prior attempts and eight distinct
nonparent sources. It includes seven complete earlier programs and explicitly omits
one by the registered source-count limit, with no character-budget omission. The
memory serialization has 21,852 characters. The previous request also contained
seven complete earlier sources and returned usable code. These are observed
context-exposure and generation-feasibility checks, not optimization efficacy.

The short pilot authenticated six Pareto decisions, of which two are terminal
choices without a subsequent generation. It recorded zero non-scalar parent
updates. Scripted tests separately demonstrate a nontrivial frontier and an actual
non-scalar stored parent reaching the production callback. The live pilot therefore
does not demonstrate that natural frontier diversity will occur. The main study
will retain and report realized frontier and parent exposure.

Both complete pilots pass their resume checks with new objective calls, candidate
evaluation and live-client creation blocked. Source identities, original memory
availability cutoffs, TRAIN receipts and global split barriers are authenticated.
The long pilot reaches nine prior attempts naturally; the short PM pilot does not
establish reliability for long PM histories. Main-study failures remain outcomes
of the declared procedure, with candidate validity and deployment fallback reported
alongside performance.

The logical seed-inclusive pilot schedule allocates 1,440 trajectories and 46,080
objective evaluations. Its physical cache contains 1,296 distinct trajectories,
39,168 actual objective calls and 2,304 unused allocations. No unused allocation
produced an extra search opportunity. Reference-normalization reconstruction and
integrity calculations are separate from these scientific trajectory counts.

Provider-reported generation durations sum to 2,119.109 seconds for M and 1,189.693
seconds for L/P/PM. The provider field is milliseconds, as documented in the
[official OpenRouter SDK](https://raw.githubusercontent.com/OpenRouterTeam/typescript-sdk/v0.9.11/src/models/operations/getgeneration.ts).
Observed monotonic generation phases were 4,742.24 and 3,058.49 seconds, including
queueing for the common single-call lock and local work. Selection took 151.12 and
78.01 seconds. No suspension was detected in these phase clocks. Overlapping pilot
wall times must not be added as a sequential runtime estimate.

For the main study, each arm has 96 responses. Equal-weighting the four pilot arm
means gives a resource scenario of about 4.414 million reported tokens, USD 1.278
and 21.51 hours of provider generation. A separate calculation distinguishing
the shared seed and each arm's generated trajectories at eight occupied workers
gives about 3.73 hours of local execution, or 25.24 hours combined. These are
fragile scenarios, not bounds or deadlines. Three arms have only two pilot outputs;
M has ten, main histories extend to sixteen slots, cached seed work has a different
weight, and audit controls have no live-pilot timing sample. Queueing behind EXP-17,
retries, host load and suspension add uncertainty. Nothing in these resource
observations justifies reducing the registered sample or changing the design.

The unchanged benchmark has separately established headroom in EXP-16. No pilot
arm comparison was used to select tasks, prompts, budgets or settings. EXP-18 keeps
six fresh paired outer seeds, L/M/P/PM with sixteen responses each, the exact
original seed, 24 TRAIN / 12 validation / 12 audit instances and two local seeds
per instance, at 32 objective evaluations per trajectory. Every comparison is
exploratory, with no independent sixteen-response arm and no pooled EXP-17/16 claim.
The subprocess boundary remains an API boundary, not an OS security sandbox.

Evidence: [long gate](engineering_memory/engineering_results.json),
[short gate](engineering_short/engineering_results.json),
[long resource record](engineering_memory/resource_summary.json.gz),
[short resource record](engineering_short/resource_summary.json),
[resource projection](engineering_resource_projection.json),
[resource methods](RESOURCE_METHOD.md), [independent review](readiness_audit_01.md),
and [final protocol](PREREG_EXP18.md).
