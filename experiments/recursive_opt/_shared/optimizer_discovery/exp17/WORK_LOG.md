# EXP-17/18 continuation work log

## 2026-09-10 — new authorized objective and rollback point

The previous goal completed EXP-16 with a negative rich-feedback result. The new
goal explicitly authorizes executing the recommended C−I rerun and the uncovered
attempt-memory/Pareto mechanisms, then revising the analysis documents. All previous
evidence stays separate. No new efficacy result is inferred from implementation.

Starting SHA is 13ebda2242e1c18022591737b113030ca2ce2da2. The prior working tree
contains completed, uncommitted EXP-16 research and one preexisting user probe.
Created branch codex/exp17-parent-selection-exp18-memory-pareto without changing
HEAD. baseline/state.json and preserved_files.json hash 87,070 existing artifacts;
initial_tracked.patch and authored_sources.zip provide an additional rollback
record. No .env, credential or dependency directory was included. The user's probe
hash remains ca8a08e5c2eca14c1004e2af3241ae8d9ebf2909d386eee9e1e2611f5b5e84e8.

Read AGENTS.md, CLAUDE.md, current ledgers, completed EXP-16 design/analysis,
source owners and rejected-attempt design. EXP-17 and EXP-18 were free in the ledger.
Recorded both draft protocols/manifests before new model calls. Independent design
review confirms 46 C−I pairs/736 responses and 384 separate factorial responses,
plus 20 planned engineering responses. Neither replication nor budgets depend on
new comparative outcomes. Provider request time alone extrapolates about37.29h for
EXP-17; local evaluations and retries add time. The protocols preserve the exact
model/settings, 32k cap, eight local workers and global protected-audit barriers.

## Test-first adapter implementation

The new Study subclasses the existing experiment owner, reuses its invariant
prompts, actual feedback integrity, independent generator and production Trace
scheduler, and extends registered-arm barriers without modifying frozen EXP-16
or production files. Evaluation caching is extracted into a narrow successor helper
with exact original keys/evaluator calls and read-only authenticated access.

The first study/driver tests failed on the missing modules; later guard tests
failed on the missing exact-confirmatory-grid check and missing EXP-18 context/
parent-decision barriers. Those failures are preserved in baseline/tdd_*.log.
The first combined driver run also collected the newly added grid test before its
implementation; its single failure is the same intended missing-method behavior.
The passing current owner/driver/cache suite is **33 tests in20.91s**, saved as
baseline/regression_01.log. A final read-only audit-replay refinement follows the
independent review and will be rerun before freeze. Black py313 and Ruff pass.

The cache helper separately passes20 new tests plus16 affected frozen-owner tests
(36 total,44.09s), including exact evaluator arguments, invalid unused allocation,
all complete keys, gzip, concurrent deduplication and no read-only side effects.
The analysis implementation has an independent actual-subprocess integration test
with four scripted model responses, including one empty response; all new stages
will still use real DeepSeek generation in their live runs.

EXP-18 pure memory projection preserves typed invalid/empty attempts, exact source
identity, closed TRAIN metadata and explicit whole-source omissions;41 tests pass.
Its source limit is seven recent distinct previous programs beyond the displayed
parent, with65,536 characters in the registered run. No assumed seven-example
threshold or success rate is claimed. Immediate TRAIN receipts are allocated in all
arms so historical context needs no extra evaluations.

An independent runtime check showed that nominal Pareto configuration at width one
would retain the scalar parent. The new selector therefore reuses production
nondominance ranking but explicitly samples a frontier parent through an overridden
PrioritySearch.explore hook. A scoped trainer alias uses the existing resolver and
is removed afterward; no production dependency or source modification. The selector
and trainer have32 tests, including actual Control Plane callbacks and identical
parent decisions on replay. Full mechanism-owner integration remains in progress.

The shared live client is the tested EXP-16 constructor. A local file lock prevents
simultaneous generative calls across successor processes; any inter-study queueing
is included in attempt wall time, so wall time remains secondary rather than a pure
provider-latency measure. Proposal identities/budgets and each study's arm order are
unchanged by queueing. No new live request has started at this work-log checkpoint.

## Final pre-pilot regressions and process ownership

The final general offline command was:
`/tmp/phase0-venv/bin/python -m pytest -q -rs --disable-socket --allow-hosts=127.0.0.1,localhost tests/unit_tests --ignore=tests/unit_tests/test_recursive_opt_review_regression.py`
It passed **844 tests**, with **3 existing optional skips**, in230.82s. The excluded
review-regression module requires a live external Trace-Bench/provider backend;
it is not part of the bounded optimizer-generation study. Skips are missing Graphviz
and two optional graph/telemetry backends. Full output is preserved in
baseline/full_unit_regression_01.log. The final shared study/driver suite passed
14tests in20.62s; the lease test passed again after a formatting-only adjustment.
Black and Ruff are reused from the existing miniconda environment; they are not
installed in the isolated test venv. An initial attempt to invoke nonexistent
venv formatter executables failed, followed by successful existing-tool checks.

Every mutating CLI action now holds a nonblocking per-run process lease. This
prevents accidental duplicate writers while retaining the separate shared
provider-call lock. A retained lock file is not itself proof that a process lives.
The analysis suite passed55 tests, including real objective/subprocess integration.
EXP-18 mechanisms passed102 tests; its driver/shared-driver regression passed24.
The independent numeric verifier passed26 tests and recomputes actual objective
observations, including partial invalid rows, B2 and fallback, without candidate or
model execution. All these are engineering tests, not live scientific results.

No successor model calls have begun at this checkpoint. The remaining pre-freeze
archive check authenticates complete ZIP contents and handles a missing metadata
record without overwriting the original archive. Draft protocols and budgets are
unchanged. Available local disk is approximately2.9TiB; no resource reduction is
indicated.

## Engineering freezes and first live launches

The independent pre-live audit identified a missing *early* exact pilot-grid guard
in EXP-17. Its new test failed because live_client was reached, then passed after
the guard was placed before client creation. The combined driver/archive/EXP-18
driver suite passes33tests in46.50s. Source archives now verify every member and
recover only absent metadata for a complete authenticated ZIP; corrupt archives
are retained and rejected. No generated response existed during those repairs.

The first EXP-17 draft is preserved inprotocol_versions/PREREG_EXP17_draft_01.md.
Draft02 corrects prose to the existing orderIC,CI,CI,IC,… (still23/23); neither the
implemented order nor scientific allocations changed. The exact numeric verifier
and tests are included in both drivers' source freezes. Separate pilot protocol
copies permit final-confirmation documentation without altering pilot evidence.

EXP17-E1 froze103files at
d96e853c5c0170c0f3a4e4ea4755d93615361688fc92e6e81c622678aad45ecf,
ZIP3a7affd9008847d5d96eadd8a47b5e62e20155c4af0c77b2661a2abf3a3132b7.
Started real four-response pilot asPID669562/session77827; launch_01.json and
live_01.log reside inengineering/. It has no planned audit evaluations.

EXP18-E1-M froze113files at
b8995156cacb2ea2b6821c165a7783116d431913887be27eda33fffd7491fe98,
ZIPd78a5ef7f5ab6a4e0167a0696cb4f15fc34b69880033e8fbebea8880e20d95eb.
Started ten-response natural-memory pilot asPID675087/session33671 with its own
launch record/log. EXP18-E1-SHORT is prepared at
3f2d24303ec0623c53f13a03a4da9bf632d1c07fd5e7dd35cb1a7bf18ea594ab,
ZIP0ca9c2a6c0782dac72c9a35e0dcb3bb5c10299b0d8410fbfa1634cd067b6b41e.
Its six responses have not started at this checkpoint. All model calls share one
exclusive provider lock; metadata-only GET requests are separately accounted.

Independent identity-only reconstruction confirms new main/pilot instances are
mutually disjoint and disjoint from EXP-15 and P1. Logical allocations are
2,049,024 objective calls forEXP-17;967,680 forEXP-18;59,904 across the20 pilot
responses' TRAIN/validation schedule. Actual calls remain measured separately.

The preservation scan passes:87,069 prior files byte-identical and the only change
among87,070 inventoried files is a two-line insertion intoRESEARCH_LOG, preserving
all prior lines. Assessment and user probe match the baseline. The private scan
covers211 successor files and453 archive members with no credential/.env leakage;
it is a dated snapshot, not a claim about future live files. See
baseline/preservation_pre_live.json.

## Operational/reporting checks and bounded timeout diagnostic

All three pilot processes are now live: short L/P/PM usesPID719594/session87195,
withengineering_short/launch_01.json andlive_01.log. Calls remain serialized.
No main-study response or audit evaluation has begun. Main analyses, a small
ordered CLI pipeline and a separate program-inspection exporter are implemented
before outcomes, with no changes to the already frozen pilot sources.

Independent review found and fixed two reporting-helper defects before any real
export: ambiguous JSON/gzip records and output paths/symlinks overlapping input
evidence. Tests now reject both before any write. The operational helper records
each childPID, preserves numbered immutable journals and stops on failure. It does
not replace completed proposals or automate scientific retries. Late provider
metadata can require a separately named reporting revision; targeted numeric
verification after analysis is documented inRUN_OPERATIONS.md. The final combined
pipeline/inspection suite passes34 tests; Black/Ruff/diff checks pass.

The first EXP17-E1 independent source is statically valid but all48 TRAIN
trajectories are invalid (25timeout,23nondeterministic labels from the unchanged
runner). Its exact source rebuilds GP covariance/Cholesky for each of400 internal
candidate points. This is a source-level cost observation, not proof of the
system cause of any particular timeout. The runner's nondeterministic label also
covers an invalid second replay; the existing evidence does not distinguish those
subcases. No completed response or candidate was replaced or manually edited.

A separate diagnostic was registered before eight proposal-only calls on the
unchanged source and trusted seed, using empty history and the exact first
lexicographically selected failing history. All8 calls/16 subprocess executions
are valid; zero objective/LLM calls. On that12-observation history, candidate child
CPU averages0.2188s versus0.0614s for the seed (3.56×), and elapsed calls take
0.230–0.331s versus approximately0.098s. The original failure did NOT reproduce.
Host load is uncontrolled; this is neither proof of filesystem/OS isolation nor
proof that the entire candidate trajectory would now be valid. No timeout or
worker setting was changed. Seeengineering_diagnostics/timeout_01/REPORT.md;
original52 registered evidence files remain byte-identical. A first direct-script
invocation failed at import before any diagnostic execution; the unchanged module
invocation then completed the sole registered sequence.

## EXP-17 confirmation frozen and launched

EXP17-E1 is complete:4 responses/4transport attempts, all statically valid,3fully
eligible generated programs, no length finish. All required allocations, actual
Ccallbacks, chronology and no-new-work resume pass. Full counts and limitations
are inENGINEERING_REPORT.md andengineering/resource_summary.json. Four complete
TRAIN-valid sources including the seed exhibit four distinct behavior signatures;
this is not used as performance-headroom evidence. No pilot AUC comparison changed
the registered design.

The final confirmation keeps46 paired seeds17001–17046, I/C×8responses, B32,
24TRAIN/12validation/12audit instances with2local seeds,32k/low/concurrency1,
8local workers/2s timeout and the same primary C−I analysis. Both earlier drafts
are preserved. Final protocol SHAfb3b0023d3b30b0eb05bfff3c6c56c02ac60350fceeebe4a91dbc38b13e90d96.
Canonical main freeze SHAe520ba38428cf30cecefb76787f73454f4b985143c9ae46a7b9f48e36167c980;
103-file ZIP SHAe0a91cd10f69f2a1a5252786a7c2602e4991f91677599af3c09798fa18c0c33a.
exp17_manifest.json separately records the protocol, freeze, source archive,
operational/reporting sources and pilot hashes; its canonical JSON digest is
3a60db399ea1558efb9ed367c7a148bb81760ef7cf28b94eea1ae6e18cfc24d9.

Independent final preflight passes, with actual objective/candidate/live-client
creation blocked during resume checks. It confirms the exact736-response grid,
23/23order, no audit and unchanged pilot records.40guard/archive/pipeline tests
pass. baseline/pre_confirmation_review.json explicitly separates its prelaunch
zero-request observation from postlaunch report-writing; no exact prelaunch time
is invented. Its stable-scope private scan reports no actual credential or .env
leakage; deliberate fake credentials inside an archived detection test are
classified explicitly rather than concealed as a heuristically clean scan.

Started the EXP-17 pipeline asparentPID819810, generationchildPID820068,
session77879, at the timestamp inrun/runtime/launch_001/launch_started.json.
The exact helper SHA is6a75fc8fc725f8419904cd79dda88141167e1a2a5d9aa7b1b74da33b4f73bbe2.
It executes generation→receipts→selection→audit→analysis→numeric verification,
stopping on an error. Progress journals andpipeline_01.log are durable. A restart
must inspect both parent/child state and unfinished provider attempts. Completed
responses and negative/invalid outcomes are never replaced. EXP-18 pilots continue
under their separate freezes and the shared single-provider-call lock.

Resource planning now uses the completed E1 counters:~45.06h reported provider
generation plus~19.94h local scaling, before other-study queueing/retries/suspension;
aboutUSD1.03 and5.316Mreported tokens. This four-output extrapolation is fragile and
not a deadline, upper bound or performance forecast. The46pairs are not reduced.
OpenRouter's official SDK v0.9.11 documentsgeneration_time in milliseconds;
ENGINEERING_REPORT links the exact primary source. No new dependency was installed.

The short EXP-18 pilot has one completed length response consuming32000completion
tokens with no usable source. It remains an invalid allocated slot, without a
replacement.32k does not guarantee usable code, and no cap or prompt is changed in
response to this failure. EXP-18 final gating still requires both complete pilots.

## Continued EXP-18 readiness review, 2026-09-10

The short L/P/PM pilot has terminated successfully (session87195, exit0). Its
six responses include five fully eligible generated programs and the preserved
length/no-source response. All504 cache trajectories are retained. Independent
guarded replay authenticates six current contexts and six Pareto decisions,
including two unused terminal choices. No live non-scalar parent choice occurred;
scripted production-path coverage remains distinct from natural pilot exposure.
No audit trajectories were evaluated. See exp18/readiness_audit_01.md.

The long memory pilot reached nine completed responses. Its final registered
request's snapshot has all nine prior attempts, eight distinct nonparent sources,
seven full prior sources included and one explicitly omitted by the source-count
limit. It has no character-budget omission. This is actual context exposure, not
evidence that memory improves optimization. Final execution/selection/resume gates
remain pending. The original EXP-18 draft is now separately preserved byte-for-byte
in exp18/protocol_versions/PREREG_EXP18_draft_01.md; the frozen pilot protocol is
unchanged.

The independent short-pilot resource reconstruction reports67,511 total tokens,
USD0.021010768,13,824 actual objective calls,2,304 unused allocations and27,648
candidate subprocesses. Reasoning tokens are retained as a component, not added
again. The measured generation phase includes shared-lock queueing. Separate
resource methods account for the unbalanced pilot arm/depth sample rather than
pooling it as representative of the four-arm main study. EXP-17 confirmation
continues under its existing freeze, without inspecting interim efficacy.

The historical-document map in exp18/DOCUMENT_UPDATE_MAP.md identifies updates
for after actual results, with dated links preserving EXP-16 evidence, retractions
and the historical scope of older hypotheses. No historical scientific files,
production sources or frozen implementations were modified by these reviews.

## EXP-18 final protocol and source freeze

Both pilots are terminal with passing gates. The long stage completed 10 responses,
all 10 eligible; the short stage completed 6 with 5 eligible. A separate guarded
authentication of both gates passed with benchmark objective calls, candidate
evaluation and live-client creation blocked. The two resource summaries preserve
16 attempts, 189,733 total tokens, USD 0.049127432, 39,168 actual objective calls
and 78,336 subprocess executions. Neither pilot used audit. The matching immutable
resource projection and original method version remain preserved.

The final resource projection balances the four arm means and distinguishes seed
work from generated positions. It gives about 4.414M tokens, USD 1.278, 21.51h of
provider generation and a separately qualified 3.73h local scenario at eight
occupied workers. The earlier rough pooled local scaling of 4.65h is superseded
for planning by this explicit allocation calculation, without changing any
scientific setting or excluding invalid costs. Both are fragile timing scenarios;
neither is a deadline or an efficacy prediction.

PREREG_EXP18 now records the completed engineering decision and spells out existing
allocation, selection, fallback, model, retry and interaction semantics. The initial
draft and frozen PILOT_PROTOCOL remain unchanged. Retries are bounded per explicit
invocation; an unmatched interrupted start requires reconciliation. The interaction
is departure from additivity, not a simple PM-versus-arm superiority contrast.
No implementation or scientific design changed from the pilots.

Preparation completed successfully in session95695. The main freeze is
ce88c4b2d9530456e63b497f7bbde7608fbf3d8aca441c13c086d77b6beeea8a,
with 113 source files and ZIP hash
c29b8ba5a5538d6af62bc5696a8ef2841838016906e1384c563138bdf671a4eb.
The final protocol hash is
0f84fe008dff9d0795de0e061e157142a16dc39b80d34e2000b8e53ed49dd07c.
The external exp18_manifest.json canonical digest is
46c04ada0a40bf40e1d9f73209766798adb00530f7a4bdf9c2e1a3bc3edef67d.
It binds the exact configuration, arm orders, protocol, source archive, both pilot
proofs/resources and operational/reporting helpers. Main request count was zero
at persistence. An initial ad hoc manifest-construction call used the wrong
signature of the pure arm_order helper and failed before any persistence; the
corrected call completed without changing that helper or scientific evidence.

Independent final prelaunch review is pending. Main EXP-18 generation is not
launched until that dated review has been saved and passes. EXP-17 continues with
its original frozen sources and protected audit.

The final EXP-18 review is now saved PASS at exp18/pre_launch_review.json, SHA256
31a876358576a97036cf9d6e634cc9b332534120d92ff0e48932da114c093669.
Its checks completed at 1789046307451488783 ns, before any main request. It verifies
the exact 384-slot grid/order, all 113 frozen sources and archive/manifest/protocol
bindings, both guarded pilot gates, and stable private scans of 7,041 files,
339 ZIP members and nine gzip payloads without credential, .env or private-source
path leakage. All 103 EXP-17 frozen files and the user probe remain unchanged;
the original preservation inventory retains every prior line, with only the
declared new ledger rows inserted.

Launched the EXP-18 pipeline in session1799, parent PID1005713, generation child
PID1010153. Its durable runtime/launch_001/launch_started.json timestamp is
1789046389168141465 ns. The child process record and pipeline_01.log are preserved.
The reviewed prelaunch record was authenticated again immediately before launch.
At the first postlaunch check, EXP-18 has one pending request and no response;
EXP-17 has eight completed responses. Their one shared generation lock remains in
use. Both pipelines must finish their complete registered studies; no efficacy
conclusion or completion claim is made from engineering readiness or partial runs.

The new EXP-18 ledger row now links the frozen protocol and manifest and records
main execution in progress. No old experiment result, assessment conclusion or
retraction was rewritten. No commit, push, PR or external message was made.

## Recorded network interruption and unchanged explicit resume

The first main launches stopped at approximately14:00 UTC on2026-09-10, before
generation completed: EXP-17 retained11 responses and EXP-18 retained2. The common
production wrapper reported an incomplete response allocation; preserved underlying
attempts identify a connection timeout followed by temporary DNS resolution errors.
EXP-17 C/17001/slot03 has one recorded failed attempt. EXP-18 L/18011/slot02 has
two: timeout then DNS. Every started attempt has a matching failure record; neither
pending slot has a response. Both pipeline parents and generation children exited.

The frozen inner retry layer recognizes temporary DNS errors, while the outer
classifier does not. Consequently the recorded DNS attempts have transient=false
and stop that invocation's retry batch. This operational limitation is documented,
not changed in the frozen sources. A public OpenRouter status check reported chat
operational; it did not establish the pending upstream request's status. The
elapsed request clocks include shared-lock waiting and are not pure provider
generation durations. Comparing current boottime/monotonic separation with both
pending-slot starts showed no additional suspension beyond numerical noise.

Local DNS subsequently resolved openrouter.ai, and a credential-free public models
request returnedHTTP200 with the exact frozen model listed. Available failed-attempt
records contain no provider generation IDs, so the existing receipt collector
cannot reconcile these by local slot ID. Possible remote completion or duplicate
billing remains explicitly true, especially for the timeout; unknown usage is not
reported as zero. No completed response is replaced.

The pre-resume inventory in exp18/operational_status_checks/incident_001.json
preserves all3273 existing run files. The independent resume audit is
exp18/operational_status_checks/resume_audit_01.md, SHA
ccd3204f1a26122bb2ffca4506e1c55aca1a5b6a2461b69dfedae910ec3da78d.
It confirms unchanged freezes, matched failure records and safe explicit resume
under the existing protocol after network recovery. All3273 hashes were checked
again before launch, with no active old pipeline/driver and no existing second log
or journal. No scientific semantics changed and no completed evidence was invalidated.

Explicit second launches use new non-overwriting logs pipeline_02.log and numbered
runtime/launch_002 journals. EXP-17 session68473 has parentPID1154868 and generation
childPID1155207, journal start1789050585270332316ns. EXP-18 session79823 has
parentPID1154991 and childPID1159714, journal start1789050610780377570ns.
They resumed exactly at EXP-17 attempt2 and EXP-18 attempt3 for the pending slots,
with the original11/2 responses still present. The one shared generation lock and
all source, task, prompt, seed, budget, selection and analysis settings remain fixed.

## Second recorded transport interruption: EXP-17 only

Both previously interrupted slots completed on resume: EXP-17 C/17001/slot03
returned a valid source at attempt2, while EXP-18 L/18011/slot02 returned
missing_source with finish_reason=length at attempt3 and completion_tokens=32000.
The latter consumes its slot and is not replaced. The original3273 pre-resume run
files remained byte-identical after those completions.

EXP-17 launch002 subsequently stopped with exit1 at1789052353309741459ns after
C/17001/slot04 attempt1 returned a recorded incomplete chunked read. There is no
response or provider generation ID for that attempt. All started records match
attempt records; the12 completed responses remain intact. EXP-18 launch002 stays
active and is not restarted. The frozen outer classifier records this wrapped
transport error as transient=false. Its classification is preserved, and no
retry source or scientific setting is changed.

The new incident_002.json in exp18/operational_status_checks records4354 EXP-17
run-file hashes, the failure and successful credential-free DNS/public-model-list
checks. Its SHA256 is db6e38471a3c099a527ffe8a54fa5bca0c3023311a0d1edcb3b972d66ec5fa37.
All4354 hashes and the original EXP-17 main freeze authenticated before a proposed
explicit resume. Remote completion and billing of this incomplete response remain
unknown, not zero. No completed proposal is regenerated.

The installed-client audit at operational_status_checks/client_timeout_audit_01.md,
SHA256 8b19d15db9d87ec2eb61bdb9483ad0d02308b4f987b5755f62d866c06119792c,
confirms that HTTPX receives300 seconds for each connect/read/write/pool operation,
not an overall deadline. Shared-lock waiting is outside that timeout. The actual
OpenRouter path uses LiteLLM's HTTPX handler, not the OpenAI SDK retry loop.
No hidden retries or dropped timeout are established. The documented conditional
LiteLLM process-global retry override was not found configured in the reviewed
path; it is not evidence of extra requests in these runs. No invalidation follows
from elapsed-time observations or this read-only audit.

Independent resume audit02, SHA256
9f3611b8191f6e514ce2b51e64ceeb9afd36ddbc067e2ab9954022d8a2bb3189,
confirms the existing guard permits EXP-17 slot04 attempt2 after the recorded
transport failure. EXP-18 launch002 then also ended at1789052698825016852ns with
the same incomplete-chunked-read error on L/18011/slot03 attempt1. Its three
completed responses are preserved. Root applied the same unchanged guard review:
all starts have matching attempt records, the pending slot has no response or
provider ID, and both recorded pipeline/driver processes are gone.

The separate incident_003.json preserves1123 EXP-18 files and authenticates its
original main freeze, with successful DNS4records/publicmodelsHTTP200/exactmodel
checks. SHA256:393041c983cafbf18cae78e3e9579b28db2f8611cd59d725ff4287bddbe53772.
An initial ad hoc guarded-preflight command targeted a nonexistent direct
exp18.driver.live_client attribute and failed before persistence or any live call;
the corrected guard targeted the existing shared driver factory and passed.
No implementation source was edited for that operator-command error.

Both interrupted run inventories were rechecked unchanged, all starts matched,
no study process remained, and third logs/journals did not exist. Explicit third
pipeline launches now run in session70763(EXP-17) and session25140(EXP-18), using
new pipeline_03.log files and launch003 journals. EXP-17 parentPID1235795 and
childPID1236170 started at1789052951163877146ns. The original model, settings,
requests, sources and all evaluation/selection semantics stay fixed; no completed
candidate is replaced. Both pending slots resume at attempt2. These interruptions
remain transport failures with unknown remote billing, not scientific outcomes.

EXP-18 third-launch parentPID1235848 and childPID1240622 are recorded at
1789052975885900165ns. EXP-17 slot04 completed on attempt2, bringing its count to13.
Postrestart verification confirms all4354+1123 interrupted-state files remain
byte-identical. The readonly monitor watches third-launch parent command identities
and phase-presence markers; it stops on either pipeline exit for operator review.

EXP-18 L/18011/slot03 completed on attempt2 at1789054211451777411ns with
finish_reason=length and missing_source. Reported prompt/completion/total tokens
are2196/32000/34196, reasoning_tokens32000, costUSD0.008211615664, wall_s1201.505641.
This response consumes its slot without replacement. Both interrupted slots from
second launches are now locally resolved, and all4354+1123 preserved run-file
hashes still match after these completions. The main counts are13(EXP-17) and
4(EXP-18); neither study has completed generation or reached selection/audit.

## Observed host suspension during third launches

The readonly monitor advanced from2026-09-10T17:26:51UTC to19:13:03UTC with the
same22(EXP-17)/12(EXP-18) completed-response counts and both pipeline parents alive.
The independent CLOCK_BOOTTIME minus CLOCK_MONOTONIC comparison against both
pending generation-start records increased by6317.531 seconds, confirming about
1h45m18s additional system suspension. Exact clocks are preserved separately in
exp18/operational_status_checks/suspension_001.json. Pending identities were
EXP-17 C/17002/slot06 and EXP-18 L/18011/slot12. No operator interruption, new
request, source change or restart was performed for this observation. Existing
calls must first produce their recorded completion/failure; wall-clock suspension
must remain distinguishable from model/evaluator processing time and lock wait.

After that suspension, EXP-17 C/17002/slot06 returned a valid stop response on
attempt1, bringing its total to23. EXP-18 L/18011/slot12 recorded a timeout on
attempt1(transient=true) and entered its existing bounded automatic retry path.
The timeout remains transport failure with possible remote completion/billing;
no operator restart was issued. The temporal association with suspension does not
establish the complete network cause. The frozen response/attempt wall_s field is
measured with time.monotonic(), so it includes shared-lock waiting but excludes
system suspension; present that clock meaning explicitly in final timing reports.

At2026-09-10T20:40UTC the third launches remain active with30/736(EXP-17) and
20/384(EXP-18) completed responses. EXP-18's initial L block is complete and M is
running; EXP-17 is finishing I for outer17002. No generation-completion, selection
or audit analysis is available. A further EXP-17 I/17002/slot05 timeout was handled
by the unchanged automatic retry path and then completed; no manual restart.
Current preserved transport-failure counts are3(EXP-17) and4(EXP-18). Keep every
failure's possible remote completion/billing uncertainty alongside known completed
response usage. The previous suspended EXP-18 L/slot12 completed on attempt2 with
protocol_violation(stop), so its completed response remains invalid and unedited.

A read-only main-run mechanism-coverage check is saved in
exp18/operational_status_checks/memory_exposure_001.json, SHA256
56894c1322cf5374b8ecd1c8a118cb9174158d2b859bf1f5c865fc4173a31348.
EXP-18 M/18011/slot11 includes11 earlier slots,9 distinct nonparent nonempty
sources, and7 complete source examples. Two sources are omitted by the registered
count limit and none by the character budget. Memory text is52801 characters,
including45703 source characters. Its SHA matches and the exact memory text occurs
in the preserved request. No performance value, validation or audit outcome was
used. This demonstrates actual context exposure, not an efficacy benefit.

The00:15UTC integrity checkpoint on2026-09-11 authenticated both original freezes,
all file hashes in the three earlier incident inventories(3273,4354,1123 checks,
with overlap), and the unchanged user probe. Both third-launch parent commands
still match their recorded PIDs. Counts were49(EXP-17)/38(EXP-18) completed responses
and4/7 recorded transport failures. The checkpoint, with no scientific execution or
metric-value inspection, is exp18/operational_status_checks/integrity_checkpoint_001.json,
SHA2561987bfba4f879b449ccc8b9e3a02a427a55d7a7653b5151deba5ea2efc623b06.
An initial ad hoc reader assumed the newer incident-schema file field for incident001
and failed before persistence; the corrected reader explicitly handles its original
nested inventory. No frozen source or evidence was changed by that operator error.

The independent P/18011 exposure audit covers only consumed slots00–05. Five of
six requests used a multi-member frontier; slots01 and05 used parents different
from the scalar best. Frozen frontier/dedup/uniform-draw reconstruction, persisted
TRAIN receipt provenance, exact parent-source/request/context alignment and timing
all matched. All46 input hashes remained unchanged. It neither replays Python
object identity after the fact nor certifies raw evaluator numerics; those remain
the tested-hook and final-numeric-check scopes respectively. No metric values,
inter-arm efficacy or validation/audit outcomes are reported. The mechanism was
exercised; this is not evidence of benefit. Report:exp18/operational_status_checks/
pareto_exposure_001.md, SHA256
 d1fa6dde2d364459c22e4629d5ab6970351ac4bb88a54c6e5cab7bed3efbe9b4.

A further read-only operational-accounting review is saved in
exp18/operational_status_checks/final_accounting_checklist_001.md, SHA256
711ce7cacb070fafaac706982567e9fd67e39789346dcdd0b24f11609bca0554.
It found no established scientific accounting defect in the documented incidents.
The final report must supplement the frozen analysis with metadata-GET coverage,
unknown failed-request billing, deduplicated attempt identities, and clock labels.
In particular, attempt_wall_s_reported sums monotonic client time including shared
lock queueing; it is neither civil elapsed time nor provider compute time. Its
values and frozen field remain unchanged. The five failures enumerated by that
review are a historical incident subset, not the final live failure count. Twelve
source/incident hashes were verified without calls, tests, or efficacy inspection.

At02:36UTC on2026-09-11 EXP-18 reached64/384 completed responses: all four arms
of outer18011 have completed generation. EXP-17 had75/736 responses. Both third
launches remain active; no final generation, selection, audit or efficacy analysis
is available. The02:32 raw-attempt count was4(EXP-17)/7(EXP-18) transport failures,
unchanged from00:15, with all earlier failures retained.

The independent joint-mechanism report
exp18/operational_status_checks/pm_joint_exposure_001.md, SHA256
 d30c5d02f0755854efd7d2567b14985d7674ce9ab1687db9b3d4004e2d2a4113,
checks only consumed PM/18011 slots00–14. Requests08–14 each contain seven full
historical sources. Detailed checks of08 and09 authenticate cutoffs, provenance,
source exclusion/deduplication, frontier reconstruction and the saved request.
Slot08 excludes its current parent7 from historical code. Slot09 combines seven
sources with the seed selected from a four-member frontier while scalar-best
origin7 differs; two older sources are explicitly omitted by the source-count
limit. All82 input hashes remained unchanged. No efficacy values, validation or
audit outcomes were inspected, and no tests or scientific calls were made. This
confirms joint mechanism exposure, not performance benefit or a learning threshold.

At03:26:01UTC on2026-09-11 the unchanged counts79(EXP-17)/67(EXP-18) were checked
against /proc/locks without process intervention. EXP-18 generation PID1240622
held the shared generation flock(inode196345921), with wchan
poll_schedule_timeout.constprop.0; EXP-17 PID1236170 was its queued waiter with
wchan locks_lock_inode_wait. Pending requests were18023/PM/slot03 and
17005/C/slot07, each on its first in-flight attempt. The03:11 attempt census still
had4/7 recorded transport failures. This observation supports a provider wait plus
serialized queueing, not a mutual flock deadlock. It does not establish remote
progress, tokens, billing or a total timeout guarantee. No request was interrupted,
replaced, or manually resumed; frozen retry and generation settings are unchanged.

The pending EXP-18 PM/18023/slot03 response completed on its original attempt1,
completed_ns=1789097314554105343, finish_reason=stop, source_status=valid.
Its client monotonic duration was2005.738422773s; reported tokens were
4158prompt/8415completion/12573total,
including7345reasoning tokens, reported costUSD0.003437467704.
The response SHA256 isedae6a42930ee6ae20c764c91b153be1cc10ed6869852f4b068754a26c00579b.
No retry or intervention was required. Source validity is not a statement of
trajectory eligibility or performance. This resolves the03:26 pending operational
observation while preserving its uncertainty at that earlier time.

The second integrity checkpoint completed at2026-09-11T06:15:00.990895+00:00, with113/736
EXP-17 and100/384 EXP-18 responses. Both original canonical freezes, their
103/113 exact source-file hashes, both frozen protocol hashes, all earlier incident
inventories(3273/4354/1123 checks, overlapping) and the user-probe hash passed.
Both third-launch process commands matched; generation/selection/audit/analysis
phase files remain absent. The recorded transport-failure totals are now5/9;
the existing automatic policy continued without an operator restart. This static
checkpoint does not claim an environment reconstruction or absence of hidden
access merely from file presence. No scientific calls or metric inspection.
Artifact:exp18/operational_status_checks/integrity_checkpoint_002.json, SHA256
0aae26abe7deb3bbff3d0e12da95caae939e2034ce9643df9fe306c2adec874f.

A second host-suspension observation is saved as
exp18/operational_status_checks/suspension_002.json, SHA256
6a683cce5f050016d853700338233fc7ce281564a59e7bd9a693ada795f58153.
The monitor jumped06:31:49UTC to07:13:42UTC on2026-09-11. CLOCK_BOOTTIME minus
CLOCK_MONOTONIC increased by2458.312117s since both pending generation starts:
EXP-17 I/17008/slot04 and EXP-18 M/18023/slot07. Count the shared host pause once,
not once per study. Both third launches remained active with116/103 responses;
no operator interruption, restart or protocol change. Await the existing recorded
response/failure paths; a pause alone establishes no remote billing or progress.

Both pending slots from suspension_002 subsequently completed normally(stop) with
source_status=valid; this is source readiness, not trajectory eligibility or gain.
EXP-17 I/17008/slot04 completed attempt1 at1789111190995321604ns, monotonic
404.801332650s, response SHA256
50686e64127f5ce68e81dea3657eef3497471f737877cf036d78f76076d2bdc7.
EXP-18 M/18023/slot07 retained attempt1 transport_failure(transient=true,
possible remote completion/billing=true, monotonic764.379450610s), then completed
attempt2 under the existing automatic retry policy at1789111307162674717ns,
monotonic218.876620443s, response SHA256
25ebc31fef33dab0bd699f9367e9434d88335ee3d98be14a37ddd48855e726cd.
No operator restart or replacement of a completed response occurred. The failure
is temporally associated with the suspension; this alone does not isolate its
full cause, provider state or billable usage.

Both third launches ended at07:54UTC on2026-09-11, before any complete pipeline
stage. EXP-17 exit1 at1789113283530382359ns retains120 responses and8 transport
failures; pending C/17008/slot00 has a timeout then DNS failure, both recorded.
EXP-18 exit1 at1789113268706965072ns retains108 responses and11 transport
failures; pending M/18023/slot12 has a recorded DNS failure. The top-level
production-allocation error is the wrapper symptom, not a failed optimizer result.
There are no unmatched starts or usable pending provider generation IDs. Both
parent/child PID pairs were absent and no other scientific driver process was
found. Unknown remote completion/billing remain unknown; completed responses
cannot be replaced. The frozen outer DNS classifier remains unchanged.

At08:06:02UTC, a credential-free DNS lookup returned4 records and the public
OpenRouter model list returnedHTTP200 with the exact registered model present.
This is connectivity/model-list evidence, not proof of generation health. Original
main grids, all engineering gates and source ZIPs passed with scientific evaluation,
live-client creation and private-key loading blocked. Immutable incident records:
- exp18/operational_status_checks/incident_004.json:38318 EXP-17 file hashes,
  SHA2569a936fc3ad343d8728c41680729745e6b1119f13a74c388512bc6132b484b7b8;
- exp18/operational_status_checks/incident_005.json:28458 EXP-18 file hashes,
  SHA2567fc71870b14909c6c384f03ee856f5ee098ac68902106750a25299a1c09e58c0.
At08:10UTC both inventories were rechecked unchanged, no launch004/log04 existed,
and helper SHA remained6a75fc8fc725f8419904cd79dda88141167e1a2a5d9aa7b1b74da33b4f73bbe2.
The existing recorded-failure resume policy permits next attempts3(EXP-17 pending
slot00) and2(EXP-18 pending slot12); preserve every earlier attempt and response.

Independent resume_audit_03.md was completed before the fourth launches, SHA256
 a5f702423118bcd39661aaa584ddbf59f07ebfd629d6184c34a786a88d97dfea.
It authenticated247 matched starts/attempts(228 completed responses,19 historical
transport failures), both grids and the exact pending EXP-18 memory context at its
original cutoff1789112858276439989ns. The65448-character memory text and messages
reconstructed exactly from all12 prior slots and readonly TRAIN caches. No new
candidate or objective execution, no manual source/context edit or invalidation.

At1789114382430108454ns the final operator check authenticated that report, both
complete incident004/005 inventories, no active scientific writers, absent fourth
launch paths, and recovered DNS. Explicit fourth launches then used fresh
pipeline_04.log files with shell noclobber and unchanged helper/drivers:
- EXP-17 parent3381330, child3381532, started1789114422525355575ns;
- EXP-18 parent3381735, child3388799, started1789114488285597634ns.
Both launch004 journals bind their original freezes. Global generation serialization
continues. Keep earlier logs/journals/attempts unchanged and verify the original
incident inventories after resumed slots complete.

Operational monitor correction only: the new fourth-launch watcher checks the
actual generation_frozen.json barrier filename. The earlier ad hoc watcher and
checkpoint002 presence list queried generation_complete.json instead. Their
partial response counts and separate frozen-driver/audit checks already established
that generation was incomplete; no scientific gate, value, selection or execution
used this display field. Preserve the earlier observations and this clarification;
no frozen implementation or preregistration was changed.

The fourth-launch resume was verified after both previously pending slots completed:
exp18/operational_status_checks/resume_verification_004.json, SHA256
93e9c956f8b44061bee7926c9bd3af2baf054b71fb8faf807409c872074b829f.
All38318/28458 incident004/005 file hashes stayed unchanged. EXP-17 C/17008/slot00
completed at attempt3(stop,source_status=valid), response SHA256
8f9607ad01c51323081ed1581cfdc216791a797477f6c9d8e0d51b3437745b77;
EXP-18 M/18023/slot12 completed at attempt2(stop,source_status=valid), response SHA256
8cee17af27491def4e1e3a0ee6329502486c268c9eaac69c12fca5f62d4f894d.
Reported completed usage is9055total tokens/USD0.002503745 and28826total tokens/
USD0.003157955 respectively. Earlier failed-request expenditure remains unknown.
Both fourth-launch parents remain active; no completed source was replaced or
edited. These operational source-validity checks do not report task performance.

A third host-suspension observation is saved in
exp18/operational_status_checks/suspension_003.json, SHA256
0f83672cf37256235d99a86b383c03c71b1bc060bd1dcf4625ef3892b9e49fb1.
The monitor jumped09:54:40UTC to11:26:08UTC on2026-09-11. Both pending starts
(EXP-17 C/17009/slot00 and EXP-18 L/18023/slot12) show5432.773377s additional
boottime-minus-monotonic time. This is one shared host pause. Fourth-launch parents
remained active with136/736 and124/384 responses. No operator interruption, restart
or source/configuration change; await normal completion/failure processing and
preserve remote uncertainty if a request fails.

The two slots pending during suspension_003 completed without an operator restart.
EXP17 17009/C/slot_00: attempt2, finish_reason=stop, source_status=valid,
completed_ns=1789126388867424754, response SHA256
b68327497ee9f4b0058cd41cd325b2baf2d947f94cffd269a3331a8f2aff9620.
Prior recorded transport failures in that slot:1; their remote-usage uncertainty is retained.
EXP18 18023/L/slot_12: attempt1, finish_reason=stop, source_status=valid,
completed_ns=1789126346902756736, response SHA256
091baa2ae7414626732035921649f1916930d63e360843b78a84b96a27aa1432.
Prior recorded transport failures in that slot:0; their remote-usage uncertainty is retained.
These are completion/source-validity observations, not trajectory eligibility or efficacy.

Fourth launches stopped during generation at11:57:36.493908UTC(EXP-17) and
11:58:27.510379UTC(EXP-18) on2026-09-11, retaining140/129 responses respectively.
EXP-17 pending C/17009/slot04 recorded one SSL UNEXPECTED_EOF_WHILE_READING
failure(transient=false). EXP-18 pending M/18037/slot01 recorded connection reset
(transient=true), then SSL unexpected EOF(transient=false). Every start has a
corresponding attempt; no pending response/provider ID or usage is available.
Both fourth-launch parents and children were absent. Preserve all completed
responses and unknown remote expenditure; the allocation-wrapper error is a
transport-stop symptom, not a scientific result.

A credential-free verified-HTTPS model-list check at12:04:22UTC returnedHTTP200,
4 DNS records and the exact model present. No TLS, classifier, routing, model or
scientific parameter was changed. Original main and engineering gates/sourceZIPs
passed with execution/client/secret-loading blocked; current inventories were
stable before/after checking:
- incident_006.json:51222 EXP-17 file hashes,140responses,10transportfailures,
  SHA2565115523048799c8ba8f1f272c248c587967e9fb9bbd4f0c9e320df13089549b9;
- incident_007.json:40894 EXP-18 file hashes,129responses,13transportfailures,
  SHA25639618234491014883aa5dfe2d2cf3c3ee503aad0ea7410d0d0c723fa879d1b8f.
Both records are under exp18/operational_status_checks. Supported next attempts
are2(EXP-17 pending slot04) and3(EXP-18 pending slot01) after final preservation
and no-writer checks. These counts include earlier failures; do not sum overlapping
incident snapshots to obtain experiment totals.

Independent differential resume audit04 passed, SHA256
51aea0a256ed61793a1a174a5a9e70b1e1a02c178639bcc80069bf1c8ff05a70.
The final operator check at1789128855429579834ns authenticated all51222/40894
incident006/007 file hashes and audit04, found no active scientific module
processes, confirmed absent launch005 paths and recovered DNS(four records).
Explicit fifth launches used new pipeline_05.log files with shell noclobber and
the unchanged pipeline helper. EXP17 parent3798347/child3798564 started at
1789128876520035036ns; EXP18 parent3799201/child3805304 started at
1789128909272087462ns. Their launch005 journals bind the original freezes and
engineering gates. Next pending indices are2(EXP17 C/17009/slot04) and
3(EXP18 M/18037/slot01), subject only to the existing bounded transport policy.
No completed response was replaced; prior unknown remote usage remains unknown.

Fifth-launch resume verification is saved as
exp18/operational_status_checks/resume_verification_005.json, SHA256
9ae21d1286ffe807ce2ec1a44f6379a093230275db941d23f77e1d20c6afc642.
All51222/40894 incident006/007 file hashes stayed unchanged after both pending
slots completed. EXP17 C/17009/slot04 completed at attempt2(stop,source_status=valid),
response SHA256714d10a7e0f8dad696d8402fe39056e47ff03ea02d3fc9c155cde76302f47db4;
EXP18 M/18037/slot01 completed at attempt3(stop,source_status=valid), response
SHA256f2eee623f33c80e2d0a36cea83bfb10acf74c91691a85ceea864e8e379bb658d.
Reported completed usage is6551total tokens/USD0.00148715 and11865total tokens/
USD0.00168115 respectively. Earlier failed-request expenditure remains unknown.
The resume preserved contiguous attempts and all completed evidence. Source-status
checks do not establish candidate trajectory eligibility or efficacy. Both parents
remain active; no frozen scientific or production file changed.

Read-only process check at2026-09-11T15:11:31.582209UTC, after response counts
remained161/150 for several minutes: EXP17 driver3798564 held the shared generation
flock and its main thread was in poll_schedule_timeout; EXP18 driver3805304 waited
in locks_lock_inode_wait and was the lock's queued waiter. Pending slots were
EXP17 C/17011/slot01 and EXP18 PM/18037/slot06, both started_1 with no recorded
attempt yet. This is consistent with one outstanding client operation plus the
registered shared queue, not evidence of a mutual-lock deadlock. The existing
per-operation timeout interpretation remains applicable. No request/process was
interrupted, duplicated or restarted, and no remote outcome was inferred.

The long EXP17 C/17011/slot01 request then completed at attempt1, timestamp
1789139555140136568ns, response SHA256
e41aa6603fb39ef6a65e1533c18178497498049d92b4aeb27fbecd512d7c28ec.
It returned finish_reason=length and source_status=missing_source, with32000
completion/reasoning tokens,760prompt tokens,32760total, reportedUSD0.005158,
and1342.6773337879858s client wall_s. This is a completed invalid generation that
consumes its allocated slot, not an infrastructure defect or a request eligible
for replacement. No operator intervention occurred and all settings remain frozen.
The observation does not estimate the causal effect of a larger token ceiling.

Fifth launches stopped at2026-09-11T15:45:29.255496UTC(EXP17) and
15:45:29.358918UTC(EXP18), retaining165/154 completed responses. Main sessions
returnedexit1 and all four recorded parent/child PIDs were absent. Pending slots:
- EXP17 C/17011/slot05: attempt1 timeout(transient=true), attempt2 DNS(false);
- EXP18 PM/18037/slot10: attempt1 DNS(false).
Every start has a recorded failure; no pending response or available provider ID.
Unknown remote completion/billing remains explicit. An additional EXP17 timeout
at I/17010/slot07 was automatically retried earlier in launch005 and retained;
it is included in the full historical count, not an additional pending slot.

A credential-free public model-list check at17:07:21.076126UTC returnedHTTP200
with verified TLS, four DNS records and the exact model present. This check proves
connectivity/list availability only. Original main/engineering/source-ZIP gates
passed with evaluation, slot completion, clients and credential loading blocked.
Whole-run file inventories were stable before/after these checks. Saved records:
- exp18/operational_status_checks/incident_008.json:68454 EXP17 files,
  165responses,13historical transport failures,nextattempt3, SHA256
  21611db4e116cef98f68f0d8667cbddea21e90ac323114ef5895247cd5b3e9e1;
- exp18/operational_status_checks/incident_009.json:53699 EXP18 files,
  154responses,14historical transport failures,nextattempt2, SHA256
  146bafff26050e6d622f366ed9495286e725dafd0be26f782f211f86c86367ca.
No model, classifier, timeout, routing, TLS or scientific setting changed. Final
no-writer/preservation checks must precede another explicit launch; preserve the
civil-time interruption/resume gap separately from measured active runtime.

Differential resume audit05 passed, SHA256
ed4e0ab3a9b7c7e286d46a80951c242b5a67c9d52cf10db5463f7a8f1aae6b3a.
Final check at1789146603169755671ns authenticated all68454/53699 incident008/009
file hashes and audit05, found no active scientific module process, confirmed
absent sixth-launch paths and DNS(four records). Explicit sixth launches used
new pipeline_06.log files with shell noclobber and the unchanged helper:
- exp17: parent224642, child224798, started1789146619690694592ns;
- exp18: parent225402, child232639, started1789146670912530858ns;
Both launch006 journals bind the original freezes and engineering gates. Pending
slots retain next indices3(EXP17 C/17011/slot05) and2(EXP18 PM/18037/slot10),
subject only to the existing bounded transport policy. Completed responses and
prior uncertain remote usage are preserved. Recheck incident inventories after
both pending responses arrive; source validity alone does not prove efficacy.

Sixth-launch resume verification is saved as
exp18/operational_status_checks/resume_verification_006.json, SHA256
8f65957b44e87ffcb102cfcd50f1cd4215ce2efa29e20501a8e79f21bd8a7112.
All68454/53699 incident008/009 file hashes remain unchanged after both pending
slots completed. EXP17 C/17011/slot05 completed at expectedattempt3(stop),
response SHA256edcad2a368ef6f266a9f120a9312faef15e4b84628c60de097ec9a5bf605b0c8,
with source_status=syntax_error. This completed invalid source consumes its slot
and is preserved without repair or replacement. EXP18 PM/18037/slot10 completed
at expectedattempt2(stop,static source_status=valid), response SHA256
efe3731bf55f0b488dc2df947d12af21bda4d6d5fbd8c22d2b6ab578a8a1958a.
Reported completed usage is11259total tokens/USD0.00299892 and21968total tokens/
USD0.002887743516 respectively; failed-attempt remote usage remains unknown.
Both sixth-launch parents remain active. Source-status checks are distinct from
trajectory eligibility and no comparative efficacy was inspected.

A fourth documented host-suspension observation is saved in
exp18/operational_status_checks/suspension_004.json, SHA256
44103cf831d98839f30225977885c82611528c33d61195637daf914ce86a39ec.
Monitor timestamps jumped17:25:07UTC to19:45:04UTC on2026-09-11. Both pending
slot generation clocks(EXP17 C/17011/slot07 and EXP18 PM/18037/slot12) show
8341.920029s additional CLOCK_BOOTTIME-minus-CLOCK_MONOTONIC time. This is one
shared observed host pause; do not count it twice or claim the observations are
a complete suspension census. Both sixth-launch parents remain active with
167/736 and156/384 completed responses. No operator restart/interruption or
scientific change occurred. Await ordinary completion/transport handling and
retain any uncertainty about remote execution or billing.

Both slots pending during suspension_004 completed without an operator restart.
exp17 17011/C/slot_07: attempt2, finish_reason=stop,
source_status=valid, completed_ns=1789156342509304611, response SHA256
59db2cd05a1c265699ee439ac19f472f96c932152161e7ecfb4cc348ec489590. Prior recorded transport failures in slot:1.
exp18 18037/PM/slot_12: attempt1, finish_reason=stop,
source_status=valid, completed_ns=1789156161111139329, response SHA256
9a7ffd29315a0f9c9191195c1be6955d6ca70c4a5756ec4743d1d88d8ad30c91. Prior recorded transport failures in slot:0.
Any prior failed-request remote usage remains unknown. These observations concern
transport completion and static validity, not execution eligibility or efficacy.

EXP18 launch006 alone stopped at1789158787201539688ns(2026-09-11), retaining160
responses. Pending L/18037/slot00 recorded one incomplete-chunked-read failure,
transient=false, wall_s2028.624880131014, possible remote completion/billing unknown.
Its started_1 matches attempt_1; parent225402/child232639 were absent. EXP17 launch006
remained active at171responses and was not interrupted or inventoried in this check.
EXP18-only original main/engineering/source-ZIP gates passed with execution/client/
credential loading blocked; before/after whole-run inventory stayed identical.
Credential-free public model-list check at20:35:27.860489UTC returnedHTTP200,
verified TLS, exact model present and four DNS records. Saved incident_010.json
under exp18/operational_status_checks contains64350 file hashes,160responses,
15historical transport failures,nextattempt2; SHA256
21edd382ee53b68065c83ae990ceb23bc2f9dc13f459adcae1236658b659c241.
Independent EXP18-only resume_audit_06.md, SHA256
de75e1b4935f2857a2021ead3fb13551fb2f6a89c8e9ba082c6ed7acc211bad9,
authenticates the unchanged seed-parent context and original cutoff. Its finalized
record predates the root inventory result, which is documented separately here.
Final check1789159114320742103ns reauthenticated all64350files and audit06, found
no EXP18 writer, confirmed absent launch007 paths/DNS4 and active EXP17 parent.
An explicit EXP18 seventh launch then used pipeline_07.log with noclobber and
unchanged helper, parent406340. Preserve all lock files and shared concurrency1;
no EXP17 restart, completed-response replacement, or scientific setting change.

EXP18 launch007 journal confirms parent406340/child416536,
started1789159270461037332ns, original freezece88c4b2d9530456e63b497f7bbde7608fbf3d8aca441c13c086d77b6beeea8a.
EXP17 remains on launch006 with parent224642/child224798.

EXP18-only seventh-launch resume verification is saved in
exp18/operational_status_checks/resume_verification_007.json, SHA256
5cae663fe47bdbf9aa471ca9949b629be95e9f95061367054fb96f974d02df57.
All64350 incident010 files remain unchanged. L/18037/slot00 completed at expected
attempt2(stop,static source_status=valid), response SHA256
c415245a6672d678d45a49b2eff558b9d5e8ee91c6a3179ffb5881e2ced63311,
completed_ns1789159552823912015. Reported completed usage is8381total tokens/
USD0.00140554. The original incomplete-response failure and its unknown remote
usage remain preserved. EXP17 continued independently on launch006; both parents
remain active. No inference about trajectory eligibility or efficacy follows from
this static response/preservation check.

2026-09-12 continuation: the original EXP17 launch006 and EXP18 launch007
remain active, with381/736 and367/384 completed responses at08:45:42UTC.
No operator restart, request replacement, scientific edit, or commit occurred
in this continuation. A read-only process check during the slower interval
found EXP17 holding the shared generation flock and EXP18 subsequently waiting
on that lock. These observations do not identify upstream compute duration.
The main generation/selection/audit/analysis barriers remain absent.
An independent read-only method review authenticated both configs and seven
scientific dependencies against the freezes. Reporting must distinguish the
reused EXP15 benchmark constants from the successor config/tasks: the legacy
benchmark_manifest also contains unused old model/seed/split fields.
The implemented successor local seeds/panels/model remain those in freeze.config
and freeze.tasks; no design change is implied by this reporting clarification.

EXP18 main generation sealed at1789215050264585317ns(2026-09-12 12:10:50UTC).
All384 responses are present. Launch007 generate/receipts stages returned0 at
1789215581348379252/1789215582021103124ns; validation selection then began.
Independent generation_integrity_001.md(SHA256
4af792df6e4cc997589bd4f4e27c66fa76f2048a7cac377e1c3497fe03929975) passed:
384 unique slots/responses,399 matched attempts(15transport failures),2244sealed
files,113frozen sources,3045stable scientific/attempt inputs. All384prompt
contexts reconstruct from TRAIN-only projections;192memories and204Pareto
decision identities/timing/draws authenticated. No performance ranking or
validation/audit/cache values were inspected. Pareto dominance and trajectory
numerics were explicitly outside this stage audit, pending final verification.
Static failures23 and candidates with any invalid TRAIN trajectory48 remain
represented. This is generation integrity, not final eligibility or efficacy.
EXP17 launch006 remains active with401/736responses at12:24:49UTC.

Infrastructure recovery after2026-09-12 16:34UTC: both existing pipelines
returned1 due to trusted inspect.getsource worker construction. EXP17's outer
schedule guard masked the same OSError; this is not a candidate validity score.
All467/384 responses and481/399 matched attempts are retained; no unmatched
request or started attempt remains. EXP18 retains21selections and no audit.
incident_011.json(SHA256fa49d0bd46f6a281e2d3a21cca17a990c1eef905b319e7f2f1c6299302895e9a)
authenticated169334EXP17files/166406EXP18files before/after main+engineering+ZIP
gates with execution/client/key loading blocked. Original103/113source freezes
pass; inspect.getsource of allthree worker functions currently succeeds.
HEAD independently changed to7e701b40485b9880ccfbb64c1faaabb22401c294, checkpoint
dated16:34:22UTC, changing only RESEARCH_LOG and assessment. No root commit was
issued; preserve this documentary checkpoint and scientific baseline13ebda224.
Source mtime/ctime changed near the incident, with original bytes now matching;
the author/mechanism and any causal link are not established.
EXP17 raw17030/C/slot02 has its completed response but0/48TRAIN cache rows;
seed+slots00/01 have144authenticated rows. Production prefix replay reproduces
requests/parents/feedback0..2 with all execution and writes blocked.
EXP18 raw18067/M/slot11 is the first incomplete validation panel,0/24rows;
all prior completed rows/selections are immutable. Same frozen resume may fill
missing keys; it must not replace completed outcomes or responses.
No per-observation checkpoint exists for72interrupted trajectory allocations.
Any lost physical prefix work remains unknown, not zero. Conservative one-pass
ceilings are2304objective calls and4608subprocess launches, not measured values
and not additional logical scientific allocations. Subsequent unstarted panels
are excluded from those bounds. Final resource reporting must disclose this.
Final prelaunch check1789235982033396569ns verified all335740incident files and worker source
hashes unchanged, no active scientific driver/pipeline, and absent fresh
EXP17 launch007/EXP18 launch008 paths. Resume the unchanged helper explicitly,
with separate fresh logs, preserving all lock files and concurrency1.

Explicit unchanged infrastructure resumes: EXP17 launch007 parent683033/child683183,
started1789236007950488249ns; EXP18 launch008 parent683924,
started1789236043172530217ns. The original freezes remain bound.
resume_verification_008.json SHA256
c82af099b574e9fa63954ebeca36c024aed8ee377067a9c92578bf3f15bbf3fd
reauthenticated all169334/166406 prior incident011 run files after launch;
all old responses, caches and21selections remain byte-identical.
Independent local_evaluation_resume_audit_001.md SHA256
ed336380bfdab3cdb1f4e06c6e8ce387f024e2a23d110703bda539054e4b0153
and infrastructure_source_read_001.md SHA256
e5f0dd576754ad65ffb25a8dbf40475b568b028c0d1f344f8b0a85e8414b4dd4
record exact prefix replay, uncertain lost work, and causal limits.
A root read-only exact-key check authenticated all48recovered TRAIN rows for
EXP17 17030/C/slot02 source118281f0b2fa30fdbd45df99f52a2c72f8cb1e8665f2dea520a4dec8f534c4ae;
EXP18 18067/M/slot11 still had0/24validation rows at this check, before its
selection recovery completed. No comparative numeric outcome was inspected.

The user requested a plain-language explanation of FUTURE_DESIGN, tasks, surfaces,
parents/TRAIN, recursion, traces and CurriculumBuffer while work continued.
Added new exp18/GUIDE_EXPERIENCES.md, final SHA256
dd47108ed2c80cea7afb34fa0a2827d8aa1cee3d57b857ff8634c4fb2184d2a3.
It distinguishes task sampling from recent-program memory, real Trace propagation
from graph/internal/log data not shown to the model, and elementary numerical
diagnostics from untested matrix/long-horizon applications. No historical document,
frozen protocol or main scientific result was changed to answer this clarification.
The guide is pedagogical, not a replacement for the still-pending final reports.

EXP18 completed its unchanged launch008 pipeline on2026-09-12 at19:15UTC,
return0. All384responses,24selections and864audittrajectories are present.
Generation seal preceded all validation; selection seal preceded audit.
analysis_results.json.gz SHA256
1654ab1ff03bf1d2713bb3ec6bfe870f603cc8ae6e77b32579a074d20f94925a.
numeric_verification/attempt_001.json.gz statusPASS, completed1789240520824535852ns:
27000cache rows including2068invalid rows verified; original sources and172580
input files unchanged. No candidate subprocess or model call in numerical audit.
Independent result review SHA256
7e0c9d15a88a68661441583fc7dd45c42f2c64f3f3ec1367c32c655337b50ee8
matches all6seeds,6arms,13contrasts,registered bootstrap and selections.
RESOURCE_REVIEW.md SHA256
f469bcddc42e6a5ecfee21eb3cf57543035b790679d95986d5c3ceae2ddeb394
matches226resource fields. Known completed responses costUSD0.905998544736,
5599563total tokens;15failed transport attempts retain unknown billing.
Persisted804709physical objective calls exclude unknown interrupted work;
incident011 limitations remain applicable and are not imputed aszero.
All7registered mechanism contrasts are inconclusive. No audit fallback occurred.
The root invoked the existing program_inspection helper once after completion,
exporting24selected exact artifacts and130files to disjoint exp18/programs;
inspection report SHA256
77f5272d149c37e75b418556f34883727a5896ff795ae999eda8b75b52add1ca.
No new scientific candidate, model response or protocol change was introduced.
EXP17 launch007 remains active; no EXP17 efficacy has been inspected. Its full
736response study and final documentation remain required. No root commit made.

EXP18 reporting completed while EXP17 generation continues: REPORT.md,
PROGRAM_REVIEW.md, PATRICK_BRIEF.md and French paired-results figure with
reproducible rendering script. The brief remains unsent. Independent review
matches all resource/interpretation statements; root inspected the figure and
reports. Existing program-inspection export authenticated24exact selected sources.
Full mechanism review confirms192memory requests and192consumed Pareto choices,
including real historical-code exposure and non-scalar parent choices; no gain
is inferred from exposure alone. No candidate/model/objective calls in these reviews.
Guide clarified the supplied CurriculumBuffer's fixed-pool history bypass and
the requirement for the training caller to detect and record success-after-fail;
updated guide SHA256e26f18bf11041df16c491eb36b234a9640740261827cf233b294a3b4d2ac8bce.
Current REPORT SHA2567102a09eb1229a158c554db83ec3e5259932469e0f4ebbc4e2c66fcb0a150e5a;
PROGRAM_REVIEW SHA256e97543f847f41e38d386c0f3df489dd54a725c3bd35adcc3bd2013174f2cf5f3;
PATRICK_BRIEF SHA2567dd8083b9e09a43475d5c3836ba95e917fd74beb19fd733c7ae5c32c369d7302.
Ledger now reports completedEXP18 and in-progressEXP17, with no interim17efficacy.
Assessment adds dated27pending/28completed, preserving historical0–26 evidence
byte-for-byte. Independent ledger review confirms all16older experiment rows,
historical H1/H2/H3/H5/H15-A/B and oldsections6–12 unchanged. Root gitdiffcheckPASS;
11newreport/figurefiles passed relative-link/whitespace/recognizable-key screening.
This is not the still-pending full-run credential/preservation/regression audit.

Root checked the final French figure visually and the standalone presentation
script with `/tmp/phase0-venv/bin/python -m black --check --target-version py313
artifacts/optimizer_discovery/exp18/figures/render_mechanisms.py` and
`/tmp/phase0-venv/bin/python -m ruff check
artifacts/optimizer_discovery/exp18/figures/render_mechanisms.py`: bothPASS;
gitdiffcheckPASS. Explicit Black target matches the current Python3.13 after its
default future-target warning; no safety check was disabled and no file changed.
Final French PNG SHA2564f13b0b3f8cac847b0cbfb90ebecc200b7f18561e020ffc2248511e6f1f9657b;
SVG SHA25605958451b20e39e9a46b7f7ea6f9e53ac5640ee2f11f8fbb4a082c769c5d79c7.
Both render identically on repeat from the exact sealed analysis, without fitting
or resampling. These checks do not execute candidates or add scientific proposals.
