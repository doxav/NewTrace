# EXP-16 execution log

Starting HEAD: `13ebda2242e1c18022591737b113030ca2ce2da2`.
Working branch: `codex/investigation-feedback-exp16`.
No commit, push, merge, PR or external message has been created for this inquiry.
The initial HEAD is the rollback point; all implementation so far is in new
research files. The research registry gains only a new EXP-16 row.

Preexisting untracked user file `artifacts/probe_2026/probe_aa_results.json` retains
SHA256 `ca8a08e5c2eca14c1004e2af3241ae8d9ebf2909d386eee9e1e2611f5b5e84e8`.
The original goal-objective attachment was reread; it describes the completed
Phase 0. The active user goal is the subsequent causal investigation, not a rerun
of Phase 0 or a revision of EXP-15's outcomes.

## Test-first records

G1 tests initially failed collection because the new generation module did not
exist. After implementing the separate recorder/task/request functions:

```
/tmp/phase0-venv/bin/python -m pytest -q tests/unit_tests/test_investigation16_generation.py tests/unit_tests/test_recursive_exp15.py tests/unit_tests/test_recursive_optimizer_benchmark.py tests/unit_tests/test_recursive_optimizer_program.py
```

Result: **56 passed in 14.28 s**. Black (`--target-version py313`) and Ruff pass on
the new G1 module and test. G1 preparation then preserved 24 exact requests and
the source/protocol/environment freeze before the first real call.

F1 tests initially failed collection because the new intervention module did not
exist. After implementation: **3 passed in 2.43 s**, with Black/Ruff passing.
An independent review and additional boundary tests precede its live freeze.
These unit clients are mocks; the live diagnostic calls are real OpenRouter calls.

Historical, feedback and selection audit commands/results are recorded in their
respective subdirectories. Agent test runs do not authorize unrelated edits.

## Active stages

- G1: registered 24-response cap probe, real calls sequential, immutable evidence
  in `generation/raw/`; request-order clarification in `generation/ORDER_NOTE.md`.
- B1: 384 separate fixed-policy central/wide-shift trajectories, two offline workers.
- S1: 1,584 separate fixed-policy selection/replication trajectories, four offline
  workers; every panel-size selection frozen before independent audit evaluation.
- F1: 24-response objective/feedback intervention implemented, awaiting completion
  of G1's registered cap choice and independent review before its own freeze.

Offline parallelism is candidate evaluation only. Live generation concurrency is 1.
Final recommendations and a production iterative/search-strategy validation remain
required; none is claimed complete by these progress records.

## Completed offline diagnostics

B1 completed all 384 trajectories without invalidity. S1 completed all 1,584
trajectories, with 1,800 overlapping panel selections sealed before the separate
audit. Their fixed-policy results establish usable headroom and improved selection
precision; neither is a causal estimate of recursive feedback. S2 recomputed all
N1–N8 prefixes from train/validation only and shows the two latest selected A2
replacements arrived at responses 6 and 7. All five historical outer seeds and
all invalid responses are retained.

F1 independent review and focused boundary tests now report 28 passed in 3.26 s.
Before any F1 generation, the replacement task namespace F1-R1 and the exact
recorder amendment are documented in its protocol. A separate T1 offline
throughput test is frozen to select feasible evaluation parallelism. The P1
production owner remains under implementation and test; no P1 calls or freeze
have occurred.

## User-reported host suspension

The user reported pausing the host during G1's final response slot. On resumption,
the same tracked process remained alive, with the first 23 completed responses
intact and the final slot carrying its existing started-attempt record. No
completed response was replaced. Wall-clock duration across the suspension is
not interpretable as model/provider latency. T1 timing is being checked separately
for overlap with this interruption; any contaminated timing evidence is preserved.

## G1 complete; F1 frozen and running

G1 completed24 unique responses and25 transport attempts, including the final
slot's timeout after host suspension and successful second attempt. Exact requests,
source hashes, source extraction, budgets and24 provider receipts pass. Reliability
counts8/12 eligible at8000 versus10/12 at32000, length1 versus0, select32000 under
the prospective rule. All32000 responses stopped below8000 tokens, so the result
is not a deterministic rescue demonstration. Reported159782 tokens and0.0207313
USD exclude unknown timeout billing.

F1 preparation completed36 fixed-parent training trajectories on the fresh F1-R1
namespace. Freeze digest9ee3466976a6de41ce885aee3c7b4d081256f46323bec78585011c120f99428c;
24 requests, cap32000, maximum actual request24485 characters. Tracked live process
1688782 / exec session46550 runs the preserved sequential schedule. First response
completed after538.3 monotonic seconds with26594 completion tokens and unparseable
source; this is retained, without replacement. Live timeout300 is therefore not
claimed a proven absolute wall-clock limit; its installed-client semantics are
being audited separately without changing the frozen stage.

T1's one serial reference measurement overlapped suspension by8407.79 seconds.
All scientific values remain identical. Only that timing condition was repeated
under T1-R1, preserving both records. Corrected speedups are3.07/5.49/7.29 for
4/8/16 workers. P1 chooses8 workers and24 training instances ×2 local seeds.

P1 driver tests began with the missing-module collection failure; initial3 tests
passed, then the max-prompt test failed at262144 as intended and passed at524288.
A legal synthetic48-trajectory full-improvement trace reaches276612 characters
before instructions/source, motivating this resource allowance before any P1
call. Independent review is tightening engineering gate and gzip/receipt integrity.
The production owner/Trace adapter reports28 targeted and68 affected regression
tests passing; ordinary invalid final exploration artifacts now remain scientific
search failures with a seed-inclusive final pool, not infrastructure errors.

## Runtime timeout semantics validated offline

The actual installed project/OpenRouter/LiteLLM/HTTPX path was exercised against
a loopback server, with external transport blocked and no real provider/key.
A0.5-second request timeout reaches connect/read/write/pool; periodic bytes permit
a1.204-second completed response, while an inactive response raises ReadTimeout
after0.515s. Both use one HTTP request. The default300 is thus an operation/read
inactivity timeout, not an absolute request deadline. No frozen code was modified.
This explains why F1's538.325-second response is compatible with the actual
request settings; it is not evidence the timeout parameter was dropped.

## P1-E1 frozen; metadata accounting caveat

P1-E1 is frozen at4be570f58e0522d76d17e9f67b9bb1338eed060dc800319c5b083f23fa1abe75,
with24train instances ×2local seeds,8offline workers,2R responses,32000completion
cap and524288maximum request characters. NoE1call has started; live concurrency
remains1 whileF1 finishes. Source snapshot contains93 exact files including all
production opto Python dependencies, zipSHA256
ea11026cc21bf4fe5c589522ba97f57899f363e34dd5803dbbdc7430b2bbf04a.
Roundtrip and actual-key scan pass.

The latest broad offline suite reports812passed,3existing optional skips in116.67s.
The unchanged external-backend module exclusion and temporary exclusion of the
then-under-review new driver are explicit in the recorded command. Driver later
passes19tests independently; combined analysis/owner/driver/Trace reports66passed.

F1 rich response16301 reports completion32000 but reasoning33818. Its DeepInfra
receipt distinguishes tokens_prompt4717/native6494 and tokens_completion33818/
native32000, with native_reasoning33818. Preserve all fields verbatim. Different
accounting/tokenizer conventions are a possible explanation, not established
semantics. Do not add reasoning to completion/total, infer a cap violation from
this ratio, or treat a cross-provider reasoning/completion ratio as a verified
partition of a common token budget. No scientific objective value is affected.

## Complete offline verification and F1 transport resume

Final unit command passes831 tests with3 existing optional skips in129.47s.
Thirteen additional research test modules pass individually,117 tests total.
Combined coverage is948 passing tests. Two combined-collection failures (duplicate
test basenames, then module/evidence-directory collisions under importlib mode)
are retained; isolated test processes resolve collection without dropping tests
or changing frozen sources. Exact commands/logs are inVERIFICATION.md and
verification_isolated/results.json.

F1 stopped after five completed responses: slot16302/Aanytime_sparse had a timeout
followed by temporary DNS failure. The inner transport classifier identifies that
DNS error as transient; the outer measurement classifier does not. This mismatch
was reproduced from the preserved error text, without changing frozen code, in
runtime/dns_classifier_reproduction.json. DNS resolution recovered; F1 preflight
and all five completed source/request hashes passed. The original runner resumed
only unfinished work, tracked as process1821389/session12357 with resume_01.log.
The recovered slot completed with15651 completion tokens and all6 training
trajectories valid. Earlier timeout billing remains unknown; no completed response
was replaced or added as a replacement for poor output.

A second clock observation shows5577.196s additional host suspension from the
BOOTTIME-minus-MONOTONIC offset increase. F1 clock_observation_02.json preserves
the observation. Calendar duration is not interpreted as model latency.

P1-E1's registered fixed seed panel was prepared while F1 generation continued:48
valid training trajectories, zero LLM calls, native feedback138591 characters,
22.668s active evaluation/preparation. These cache entries are reused by the two
future R updates; they do not add scientific candidate allocations. No validation
or audit task was evaluated. Context preparation time is separate from the later
generation-phase timing.

Before the full P1 freeze, its draft resource bound was clarified:1041408 is the
no-fallback subprocess allocation; at most1152 additional failed candidate
subprocesses may precede permanent deployment fallback across four generative
arms, giving1042560 with successful trusted infrastructure. Objective budgets
remain unchanged. P1-E1 has no deployment audit and its frozen protocol is intact.

The post-audit numeric verifier was independently implemented and reviewed in
production/verify_numerics.py (sha256 fa07e8c59190d0caa810c051168520f89edfc62c02f3f124e59455b69500edf7).
Ten new tests pass, with26 unchanged analysis tests:36 passed. The combined
non-overlapping verification total is958 passed,3 existing optional skips.
No P1/E1 evaluation was opened. The helper requires all completion barriers and
structural checks before rebuilding reference designs/objective observations;
its extra integrity calls are separate from scientific allocations. It does not
change any frozen source or primary metric.

A metadata-only TCP observation during pending F1 16303/anytime_rich preserved
175 incoming bytes over15.053s, five data segments, without endpoint addresses
or packet bodies (runtime/f1_active_socket_01.json). This establishes continuing
network activity, not generation-token progress. The same request remains pending;
no response was replaced. This is consistent with the separately verified
operation/inactivity timeout rather than an absolute generation deadline.

F1 reached12/24 completed responses; metadata-only checkpoint is preserved in
feedback_experiment/progress_metadata_12.json. The two latest completions
(16303/rich and16303/code) took1267.276s and804.632s active time, with5837 and5777
completion tokens respectively. Both receipts name Morph. Their raw
generation_time fields are1266796 and804221; unit conversion is not silently
assumed in primary reporting. This is descriptive endpoint metadata, not a
controlled provider comparison. No routing/model/setting change was made.
All12 completed provider receipts were collected by GET; no additional generation.

B2 was registered before its single counterfactual policy was evaluated on the
public B1 fixtures: only the empty-history proposal changes to the bounds midpoint.
Freeze6eae835547be85c560a4daa4c2bd3f714b7f16600307ef4657e3af628647f3db;
source958fbb12279a15966bf1ffa45ddffc7cd5f946ca800a3fe2b8e132a150c5c190.
All96 variants valid;96 controls reused. AUC central .192300→.027569 and broad
.167906→.055169, with26 losing pairs and nonuniform final-regret effects retained.
Independent review46fec532666d17529a4b23181f32e83de8b61761cd194c71b34b5b9b62d0cdb0
reconstructed6144 observations and192 metrics;9216 additional integrity objective
calls are separate. MainP1 seed remains unchanged. A supplemental fixed-baseline
control is being specified before mainP1 freeze, to evaluate this already fixed
policy only after the primary P1 audit completes, without modifying primary pools,
prompts, selections or raw evidence. No new generative search stage is added.

The supplementary B2 protocol/helper is ready after independent review. Seventeen
new tests pass; 43 pass together with the unchanged 26 primary-analysis tests.
Resume verifies every existing checkpoint before running missing jobs, refuses to
replace a missing row after final results exist, and links each row to its started
attempt journal. The main P1 draft now declares this separate prospective reference
and its 144 trajectories before its freeze. No main P1 generation or supplementary
control evaluation has started. The helper must be frozen after the main freeze
and before main generation; its audit access remains blocked until primary completion.

A metadata-only OpenRouter review is preserved in runtime/ROUTING_OPTIONS.md and
ROUTING_PUBLIC_METADATA.json. Two anonymous public GETs found 29 endpoint objects,
with no non-null public throughput/latency measurements. Available routing controls
therefore do not justify naming a faster provider from this snapshot. No generation,
credential, fitness inspection or frozen request change was involved. The current
F1/E1 routing is unchanged; any later intervention needs its own symmetric pilot.
Provider receipts now cover all 15 completed F1 responses, collected by GET only.

A second sanitized socket observation during F1 16304/anytime_rich recorded 175
incoming bytes and five data segments over 15.055 active seconds, with zero detected
suspension (runtime/f1_active_socket_02.json). The request had already exceeded
33 minutes. This verifies continuing connection activity only; it does not show
useful token progress or identify the upstream provider before its response receipt.
No address, header, body or credential was captured. The request remains preserved
as in flight; no duplicate or replacement generation was issued.

B2's paired-outcome figure now preserves all 96 pairs in PNG/PDF, with explicit
logarithmic axes and all losses/ties visible. Raw hashes and means match the
independent audit; no candidate/objective/model execution occurred. A separate
analytic note proves the first-objective raw expectation ratio a²/(a²+b²) under
the stated independent symmetric quadratic model, with explicit limits on applying
it to finite deterministic fixtures, normalized regret, AUC or Rosenbrock.

The public EXP-15 first-point audit reuses S0 and all 180 persisted trajectories:
no arm has any exact full midpoint initialization (0/60 each). The first AUC term
contributes −0.004887411 to A2−A0 and −0.002772778 to A2−A1; the latter differs only
at outer seed 53. All row/source/file hashes and selections were checked. The
decomposition is descriptive, not causal mediation. Evidence digest:
e9e4293076443bbc697368aa1d1d28d6d9dc2882f621d09f68eccd862dd0f673.
No new objective/candidate/model evaluation or F1/P1/E1 efficacy access occurred.

F1's sixteenth response, 16304/anytime_rich, completed after 4229.334 active seconds
with `finish_reason=error`, null final content and 18186 reported completion/reasoning
tokens. Its receipt names Morph and reports total/upstream cost zero. This is a
provider-ended generation failure, not a syntactically invalid optimizer or a
token-cap exhaustion. Under the frozen returned-response rule it consumes its slot;
no replacement was made. The read-only analysis will annotate this origin separately
without changing the primary deployment/fallback metric or frozen raw status.
Receipt ID: gen-1788959588-rNlO6GxQ5YdHztz7POE1. F1 then progressed to17/24 with a
source-valid 16305/anytime_code response. No validation has been opened.

F1 subsequently completed all24 registered responses and opened its validation
barrier only after their hashes/identities and training panels were verified.
While F1 performs offline validation, P1-E1 has started its two registered live R
updates: session63836, Python process2235223, canonical log
production_engineering/run_01.log. The E1 freeze passes; no main P1 call has begun.
This preserves generation concurrency1: F1 has no remaining generation requests.

F1 completed validation and its first independent final audit passes all24 responses,
24 receipts and60 evaluation artifacts. All registered contrasts are retained;
rich minus sparse is unfavorable, while rich minus code remains inconclusive.
The final F1 report is being completed without modifying frozen scientific values.

P1-E1 completed exactly2 real R responses in2 attempts. The strict gate passed:
one generated source is eligible across48 training trajectories, the other is
syntactically invalid, and client-free replay preserves all completed evidence.
Both prompts contain141807 characters and58435 reported input tokens. Request-active
durations sum472.360s, reported total tokens127274 and cost$0.003472328.
An independent guarded replay forbidding key loading, network, generation and
objective/candidate evaluation passes with all1277 E1 files byte-identical.

Before any main generation, the P1 protocol added measured F1/E1 resource scenarios
(12.60–30.92h linear generation scenarios, not interval or bound) without changing
the six replications or any scientific factor based on efficacy. Main freeze:
113f2eb03abfbb3f8e80e8ce5946beecd6b039fef28968b96171475a21d57e8e.
The supplementary fixed B2 reference was then registered against that freeze:
f0aeb8b78d56745c2e54b02462877952bdda15ec4e184fc0346e13d0bb0e288f.
The exact97-file source archive passed roundtrip/hash and actual-credential checks;
ZIP SHA f5120ad690b842f6fd2ca1558d00c5c2e84f723984dbbecd336e60c76fc546eb.
Main P1 generation started afterward in session73099, canonical production_run/run_01.log.
All192 allocated responses and later globally gated validation/audit remain required.

The completed F1 report and final_verification.json are now available; an independent
statistical reviewer reproduced all four paired contrasts exactly. Root report and
decision matrix now retain the unfavorable rich-versus-sparse finding and distinguish
it from the passing engineering gate. E1 also exposes a concrete memory limitation:
after the rejected syntax error, the next prompt repeats the preserved seed context
without that rejected code/error. This is a declared current-parent-only limitation,
not a retroactive P1 implementation change or proof that memory would improve efficacy.

A one-shot orchestration process now waits for the current generation PID2275932
to finish (identity includes its process start time). It will require the complete
generation barrier and192 responses, then invoke the existing frozen CLIs for
receipts, selection, audit, analysis, post-audit numeric verification, and the fixed
B2 control/run recomposition, stopping at the first failed command. It issues no
generation call or retry. Exact argv, freeze bindings and stage clocks are retained
in production_run/post_generation_schedule.json and pipeline_steps; canonical log
pipeline_01.log, exec session67882. This automates already registered phase transitions
without changing scientific semantics. Root review/reporting and final integrity
work remain required after the pipeline finishes.

A manifest-only order audit, independently checked before any P1 validation,
records central I/R precedence3/3 but C-before-R and R-before-W5/1. This is a
potential temporal-service confounding limitation of secondary contrasts, not
observed drift, an invalidation or a reason to amend the running protocol.
production/ORDER_AUDIT.md and order_audit.json preserve all six pair counts;
W−R is described as changing breadth and number of rounds together.

history/REJECTION_MEMORY_DESIGN.md records a bounded source-only future intervention:
explicit aligned prior-slot context at proposal_from_trace, separate from the
actual propagated Trace text; same-arm/outer scope, strict already-computed TRAIN
reads and immutable request snapshots. Existing archive/FIFO names do not guarantee
rejected-child feedback reaches the next parent. N8 offers seven prior responses
only at its last call; N16 could provide nine such calls before deduplication and
missing evidence. This is unvalidated efficacy, not an amendment or extra live
stage. No P1 response/evaluation was read for this design review.

The next goal turn confirms both process handles live: generation PID2275932 and
post-generation orchestrator PID2318356. The previous turn made concrete progress
(F1 completion, engineering gate, P1/control freeze and live start), not merely a
status restatement. P1 reached nine completed responses with the first I arm sealed.
first_arm_accounting_check.json verifies its eight unique response IDs, exact frozen
requests/settings/source identities and nine seed-inclusive TRAIN allocations
(432 trajectories/13824 objective calls allocated), without computing fitness.

The Deep Research and Spreadsheets skills are applied to final synthesis/artifact
delivery; the already explicit scope/audience and repository evidence format need
no new clarification. A supporting data workbook is being authored from completed
diagnostics only, with P1 omitted until its audit completes. No frozen code changes.
COMPLETION_AUDIT.md maps the original inquiry's requirements to current evidence
and explicitly leaves prospective execution, final interpretation and integrity
open. research/LITERATURE_MECHANISMS.md adds four primary-source mechanism reviews,
with no extrapolation of published gains or changes to running P1.

On this continuation the main generation and post-generation orchestrator remain
live. P1 reached25 completed response files; I/C/R of the first outer seed each
retain eight responses. A read-only clock checkpoint on the in-flight W slot01
separates305.675 active seconds from4039.230 seconds of host suspension. The
observer issued no retry or replacement and inspected no response content or
fitness. Record: runtime/p1_resume_observation_1788979878227542151.json.

The completed-study workbook now exports17 sheets from204 hashed source files,
with332 independently checked formula results and738 numeric comparisons.
All26 previews were inspected, including negative contrasts and B2 losses.
report_data/README.md preserves the first failed Boolean COUNTIFS check and its
explicit1/0 representation repair; raw scientific inputs were not changed.
The final export uses locally available Liberation Sans and corrects Guide widths
and block-label formatting. Node syntax check and build_03 both pass. No P1 data,
model calls, candidate executions or objective evaluations enter this export.
The visual-review record initially failed because of a relative/absolute path
comparison before writing; resolving the report directory corrected that metadata
operation. The exact workbook hash and26 preview hashes are in
report_data/visual_review.json. An independent ZIP/XML/data review is requested.

The suspended W/slot01 subsequently completes under the existing bounded transport
policy: attempt1 is a retained transient TransportRetryError after468.239 active
seconds, with possible remote completion/billing unknown; attempt2 completes in
155.962 seconds. Slot timing records626.210 active seconds and4039.230 seconds
of suspension. It contributes one completed response, not a replacement of a poor
completed response. At this checkpoint P1 has26 responses and28 transport starts.
The observer did not issue a request, edit the protocol or inspect fitness.

The independent workbook review is now complete and PASS: all557 formula caches,
204 source hashes,26 table bounds and4259 projected source cells match. The saved
standard-library verifier is report_data/verify_workbook.py (SHA-256
a6d011366f8e5476a7fb8e0751bd0573e7053d17e824f6253f0cab15f82c1b7e),
with its exact command and limitations in independent_workbook_review.md.
Root confirmed the unchanged export hash and a passing git diff --check.
P1 has subsequently reached27 completed responses and30 started attempts;
generation and the waiting post-generation orchestrator remain live. No global
generation barrier exists yet, so no P1 result or final gain projection is claimed.
The remaining work is still the full192-response study, global selection/audit,
numeric integrity and B2 prospective control, followed by final synthesis and
ledger/assessment updates. Workbook completion does not narrow that objective.

## Final completion — 2026-09-10

P1 finished all 192 unique model-response slots and 194 transport attempts.
The tracked post-generation pipeline completed its seven registered stages,
including global selection, audit, frozen analysis, numeric verification and the
separately registered B2 control. No completed poor response was replaced.
The host suspension is retained separately from active time. No new live stage
was introduced after outcomes became available.

The central R−I delta is +0.045257, bootstrap interval [+0.003176,+0.085126],
a negative exploratory signal. R−C is also negative; W−R inconclusive; R−A0
positive. B2 reaches mean AUC 0.031633 versus A0 0.179329, confirming the fixed
initial-point intervention on the new audit panel. C's apparent advantage over I
is explicitly post hoc and motivates a fresh question, not a claimed P1 victory.
All six seeds, every slot, invalidity and selected seed remain visible.

Independent numeric and chronology review passed; the full raw analysis recomposes
exactly and an independent arithmetic route matches all six contrasts. The primary
verifier checked 447,105 objective/reference values without executing candidates.
The final B2 review added 6,144 mathematical integrity checks, separately counted.
Program inspection verified 192 sources, 144 actual native feedback envelopes and
6,912 projections; selected-source exports are byte-identical. It documents absent
rejected-attempt memory, repeated current-parent panels, actual lineage depths and
W's exercised width. None of those remaining mechanisms is called a proven solution.

REPORT.md, DECISION_MATRIX.md, RESEARCH_LOG §12 and assessment §26 are synchronized
with the completed evidence. FUTURE_DESIGN.md independently verifies variance and
sample-size scenarios; no numerical feedback gain is projected. A ten-line Patrick
brief and the precise source/evaluator integration point are saved, not sent.
The common seed is not silently replaced and prior H15/H3/retractions stay intact.

The final production subset passes 93 tests in 73.32s. Four additional inspection
tests bring the accumulated non-overlapping total to 998, alongside the same three
optional baseline skips; the prior broad run was 831 passed/3 skipped. Formatting,
lint and diff checks pass. No frozen production behavior changed after the broad
run. New reporting/scanner checks do not inflate that test count.

The five-sheet P1 workbook is separate from the earlier diagnostic workbook. Its
138 formulas, all 36 per-seed results, 24 search pools, six contrasts, nine source
hashes and ten visual previews pass independent checks. The earlier unchanged
builder was accidentally rerun during agent resumption and replaced only its
derived XLSX/preview checks. Both historical and current hashes are documented:
the independent r2 review passes 557 formulas/204 sources/4,259 cells, and all 26
old preview hashes remain identical. No scientific artifact was changed. This
reporting event is explicit and requires no scientific invalidation.

The final private scan passes 87,067 files and 345 compressed members, including
gzip and XLSX/ZIP contents, with no key, .env contents or secret-source value
detected. Only the two ledgers differ among tracked files; staged changes are
empty, HEAD remains 13ebda2242e1c18022591737b113030ca2ce2da2 and the user's preexisting
probe hash is unchanged. No commit, push, merge, PR, new production dependency or
external message was created. Closure text is written after the scan using only
public findings. Full commands and scope are in VERIFICATION.md.

The final reporting-only aggregation records G1/F1/P1-E1/P1 totals separately:
242 completed responses, 247 transport attempts, 7,588,972 reported tokens and
USD 0.59217334461 known cost, excluding uncertain transport billing. Its source
hashes are in runtime/final_reporting_summary.json. This does not pool the
scientific outcomes or include EXP-15's historical expenditure.

The investigation is complete with a valid negative rich-feedback result,
validated engineering/measurement improvements and bounded future hypotheses.
There is no unresolved external execution blocker. Establishing a future positive
feedback gain remains an empirical question, not unfinished work to chase by
changing this experiment.
