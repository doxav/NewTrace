# EXP-18 — prospective memory and instance-Pareto diagnostic

Status: FINAL REGISTERED EXPLORATORY PROTOCOL, after both disjoint engineering
pilots and before any main-study response. Finalized 2026-09-10, from base
13ebda2242e1c18022591737b113030ca2ce2da2 and the preserved EXP-16 working tree.
This is separate from EXP-17 and EXP-16. No outcomes are pooled or used to alter
EXP-17's confirmatory C−I comparison.

Six paired outer seeds18011,18023,18037,18041,18053,18067; namespace
`EXP18-MECHANISMS-v1`. Each of four arms receives16 completed responses, total384.
Common original seed, model/settings (32,000 tokens/low), benchmark family,
24 TRAIN/12 validation/12 audit instances ×2 local seeds, B32, source screening,
eight local workers, selection and permanent deployment fallback follow EXP-17.
New instances and seeds are disjoint. A0 and fixed B2 are audit controls only.

The exact machine-readable configuration is `driver.main_config()` and will be
bound to the final manifest and source archive before main execution. The main
study is a registered exploratory mechanism test; its six paired replications do
not constitute a confirmatory superiority study. No main response has been
requested during this final protocol preparation.

## Shared fixed instrument and allocated resources

Each arm allocates the seed plus all 16 response slots to 48 TRAIN and 24 validation
trajectories per outer seed. A0, B2 and the four selected arm deployments each
allocate 24 audit trajectories per seed. This gives 19,584 TRAIN, 9,792 validation
and 864 audit trajectories: 30,240 logical trajectories and 967,680 objective-call
allocations. An invalid or empty response consumes its slot and keeps its unused
evaluation allocation; neither resource is recycled into another proposal.

Caching the identical seed across arms reduces the unique-source schedule to at
most 28,944 distinct trajectories and 926,208 objective allocations. This estimate
allows unique selected sources on audit; further duplicates can reduce it. It is
not a count of actual objective calls or candidate executions. Report those,
partial invalid trajectories, cache reuse, reference preparation and integrity
recomputation separately. Two successful deterministic replay executions per
valid optimizer point proposal are required; failure can change the realized
execution count.

The frozen generator, feasible optima, arm-independent normalization, stable local
seed derivation and equal family/dimension weights are the EXP-16 instrument used
by EXP-17. Regret-AUC is lower-is-better, target normalized regret is 0.01, and
nonattainment is censored with the explicitly labelled B+1 reporting convention.
Generated code sees only black-box history, bounds and its local seed. Source
screening and the sanitized subprocess boundary remain mandatory; that boundary
does not provide operating-system security isolation.

Use OpenRouter model `deepseek/deepseek-v4-flash-0731`, temperature 0.6, top_p 1,
max_tokens 32000, native `extra_body={"reasoning":{"effort":"low"}}`, request
timeout 300 seconds, cache false, empty-response retries zero and client retries
zero. The common generation lock permits one active model call across the studies.
The slot wrapper allows at most three transport retries per explicit invocation
batch, with 2/4/8-second delays. An explicitly reviewed later resume may add another
bounded batch for an unfinished slot; report every attempt across all batches.
A completed empty, length, error or invalid response is never replaced. Request
seeds do not establish deterministic generation. Preserve routing metadata and
reported usage/cost without exposing credentials to candidate processes.

Resume must verify the frozen sources/configuration and complete slots, preserve
original context availability cutoffs and execute only unfinished work. An unmatched
started-attempt record after interruption blocks further requests until remote
completion is reconciled. Recorded transient transport failures use the declared
bounded retry policy while retaining uncertainty and possible duplicate billing.
The operational helper stops on errors and does not retry a
scientific stage automatically. A semantic defect requires a new version/ID with
affected evidence preserved; a reporting-only repair must prove unchanged values
and decisions and retain the original report.

## Two intervention factors

All arms get the invariant prompt, exact current-parent source and the same compact
current-parent TRAIN summary: aggregate normalized AUC only when the entire panel
is valid, allocated/completed/valid trajectory counts and typed status counts.
No raw per-task values paired with normalization constants, no validation, no hidden
parameters and no LLM-generated critique are included.

L: scalar TRAIN-best parent, no prior-attempt archive.
M: scalar parent plus deterministic prior-attempt archive.
P: uniformly sampled nondominated TRAIN-instance parent, no archive text.
PM: nondominated parent plus the same archive rule.

These controls isolate the increment of the historical archive from the introduction
of immediate score feedback. They do not estimate the effect of individual archive
fields separately. P changes the complete parent-selection policy, including
frontier filtering and uniform sampling; it is not a claim about filtering alone.

Memory uses at most seven most recent distinct earlier nonempty sources excluding
the already displayed current parent, plus all previous-slot typed summaries.
Same arm/outer and prior completion only. Source hashes, response/receipt times,
parent identity and TRAIN receipt hashes must align. Source is shown whole or
explicitly omitted, never silently truncated. Exact serialized memory budget:
65,536 characters. All omitted/duplicate/no-code/invalid counts remain recorded.
No cache-miss evaluation is allowed to fill a historical context. All arms record
the same allocated TRAIN panel immediately after each completed response.

The Pareto vector contains24 means, each averaging the two local trajectories for
one TRAIN instance. Only complete TRAIN-valid programs enter parent selection.
Include the seed. Strict componentwise nondominance, no floating tolerance;
source and equal-vector duplicates retain the earliest index. Uniform choice from
the frontier uses a deterministic SHA256-derived local Random seed, independently
of model RNG and holdout. Record the frontier, chosen/scalar parent and whether
the choice will feed a subsequent proposal. Use the real production PrioritySearch
path with a narrow parent-selection override; do not substitute a second loop.

## Order, analysis and interpretation

Whole-arm orders, paired with their reverses:
18011 L/M/P/PM;18023 PM/P/M/L;
18037 M/PM/L/P;18041 P/L/PM/M;
18053 P/L/M/PM;18067 PM/M/L/P.
Thus every pairwise precedence is balanced3/3. Single generation concurrency.
Sixteen sequential single-parent updates in every arm; no width/depth tradeoff.

Freeze every selection and each arm's validation-selected representative before
any audit. Per-seed deployment AUC is the same metric/weights as EXP-17.
All generation across all six outer seeds precedes validation. Eligibility requires
every TRAIN and validation trajectory to be valid; the unchanged seed is always in
the final pool. Select minimum validation AUC, then seed index -1 or earliest slot.
Choose each arm's representative by minimum validation AUC, then registered outer
seed order. Audit failures permanently switch to the original seed with actual
history and the remaining objective budget, without resetting that history or
imputing a poor numeric value for a failed proposal.

Registered exploratory contrasts: M−L, P−L, PM−P, PM−M;
memory main effect=((M−L)+(PM−P))/2;
Pareto main effect=((P−L)+(PM−M))/2;
interaction=PM−P−M+L. Bootstrap the six outer seeds jointly,10,000 draws/seed1515.
All contrasts are exploratory, with no multiplicity guarantee or superiority claim
from ordered means. Report all seeds/invalidities/fallback and realized exposure
to distinct prior sources/nontrivial frontiers. No independentN16 arm is included,
so this study cannot establish a memory advantage over equal-budget independentN16.
The fixed percentile bootstrap uses Random1515 and the existing linear
interpolation at 2.5/97.5 percentiles. An interval wholly below zero is an
exploratory positive signal, wholly above zero a negative signal, all paired
deltas zero no detectable difference, and otherwise inconclusive. These labels do
not provide simultaneous coverage across contrasts. PM−L, arm−B2 and B2−A0 are
descriptive secondary contrasts in the frozen analysis. Report mean, median and
every paired outer-seed value; task instances are not independent outer replicates.
For the interaction, its sign describes departure from additivity: a negative
interaction means a lower combined AUC than the additive expectation. The generic
signal label for that contrast does not establish that PM beats any particular
arm; report the simple contrasts alongside it.

## Engineering checks before scientific execution

Separate live engineering namespaces: EXP18-E1-M-v1, outer18901, M10 responses;
EXP18-E1-SHORT-v1, outer18903, L/P/PM two responses each. Combined with EXP17-E1,
the planned engineering allocation is20 completed model responses. The long M
pilot tests naturally accumulated history after seven prior attempts; invalid and
duplicate outputs may reduce distinct usable examples and remain visible.
The short pilot exercises actual P/PM production paths; scripted integration tests
must additionally demonstrate a nontrivial frontier and an actual non-scalar parent.

Test first: provenance, invalid/empty output, honest omissions, no hidden data,
all allocated TRAIN receipts before reuse, determinism, no extra calls/evaluations,
real stored parent identity, global barriers and crash/resume with identical memory
snapshots. Freeze exact source/feedback/selection/analysis code after engineering.
Poor outputs are not replaced; no outcome-driven prompt/metric/seed change.
The mechanism study completes for positive, null, negative or ordinary generation
failure outcomes. Engineering failures must be fixed and documented before freeze.

## Final engineering decision and amendments

Both registered engineering gates passed without replacing a completed response.
The M pilot completed all ten responses, all with usable source and stop finish;
all ten generated candidates completed every TRAIN and validation trajectory
validly. Its final request contains nine prior attempt records, eight distinct
nonparent sources, seven complete sources shown and one explicit source-limit
omission. There was no character-budget omission. The ninth response had already
successfully produced code after seven complete prior-source examples.

The short L/P/PM pilot completed six responses, five fully eligible generated
programs and one length response without usable source. That response consumed
32000 completion tokens and remains invalid, with its allocation retained. Six
current contexts and six Pareto decisions were authenticated, including two unused
terminal choices. There were zero naturally selected non-scalar parent updates.
The registered scripted tests exercise a nontrivial frontier and an actual
non-scalar parent through production; this does not replace reporting the absence
of that natural exposure in the small live pilot.

Both pilots had zero audit trajectories. Their complete replay passed with new
objective calls, candidate evaluation and live-client creation explicitly blocked.
No comparative pilot AUC was used to select settings. Benchmark headroom comes
from the separate EXP-16 diagnostic policies; source diversity or successful
generation is not substituted for performance headroom.

The two pilots used16 completed responses and16 transport attempts:189733 total
reported tokens and USD0.049127432. Reasoning tokens are reported separately as a
component, never added again to the total. Their distinct physical cache schedules
contain1296 trajectories,39168 actual objective calls,2304 unused allocations
and78336 candidate subprocess executions. The seed-inclusive logical schedule is
1440 trajectories and46080 objective allocations. The only invalid trajectory
allocation is the short pilot's preserved empty-source response.

For resource planning, equal-weight the four pilot arm means because the main
study allocates96 responses to each arm. The resulting scenario is approximately
4.414M reported tokens, USD1.278 and21.51h of reported provider generation. A
separate arm/seed-stratified eight-worker local scenario is approximately3.73h,
giving about25.24h when added to the provider scenario.
These are fragile extrapolations, not limits, deadlines or efficacy forecasts:
L/P/PM have only two pilot responses each, M has ten, later main prompts may grow,
audit controls lack a live-pilot timing sample, and queueing behind EXP-17,
transport retries, host load and suspension add uncertainty. No resource blocker
or outcome-based reduction of the six pairs or16 slots is justified.

Preserve the original draft at protocol_versions/PREREG_EXP18_draft_01.md and the
unchanged PILOT_PROTOCOL.md, both with SHA256
6bdf05d95269fffac4dc51609e3168ba702cc7b3e44ed9c2c33c0cdae13d7f33.
The final prose makes existing settings, allocations, replay/retry semantics and
interaction interpretation explicit and adds the engineering decision. No model,
route, prompt, seed, benchmark, timeout, worker count, memory/Pareto behavior,
selection, budget or analysis implementation changed from the pilots.
See ENGINEERING_REPORT.md, both engineering_results.json records, resource
summaries and engineering_resource_projection.json for exact evidence.
