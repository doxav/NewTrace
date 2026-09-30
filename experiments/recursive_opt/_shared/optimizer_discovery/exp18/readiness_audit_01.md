# EXP-18 readiness audit 01

Independent read-only review on 2026-09-10. This is a readiness record, not the final
protocol or permission to bypass an unfinished pilot. No EXP-17 confirmatory
efficacy data were inspected, no live generation was invoked, and no frozen file
was changed. Only this review file is added.

**Status: pending the complete long-memory pilot and final main freeze.** No new
semantic implementation blocker was identified in the reviewed sources. A passing
short pilot does not replace the missing memory gate.

## Observed state and authenticated identities

Snapshot at 12:51:17.392 UTC (`1789044677392531187` ns):

| Stage | Responses present | Gate | Generation/selection barriers | Audit export |
| --- | ---: | --- | --- | --- |
| Long M engineering | 8/10 | Pending | Neither present | Absent |
| Short L/P/PM engineering | 6/6 | Present | Both present | Absent |
| EXP-18 main | 0 allocated by execution | Not prepared | Not applicable | Not applicable |

Both pilot source/environment/configuration preflights passed. Each freezes 113
files. Main task reconstruction was checked without objective evaluation: its
48 instances are disjoint from both pilot panels.

- Memory freeze: `b8995156cacb2ea2b6821c165a7783116d431913887be27eda33fffd7491fe98`.
- Short freeze: `3f2d24303ec0623c53f13a03a4da9bf632d1c07fd5e7dd35cb1a7bf18ea594ab`.
- Current main configuration digest: `d44f6a9dcd56da6b2c8f5ee010b9139488ab5b8bb2a59e7952ff634b6c2764df`.
- Draft and preserved pilot protocol byte hash:
  `6bdf05d95269fffac4dc51609e3168ba702cc7b3e44ed9c2c33c0cdae13d7f33`.

At 12:52:22.717 UTC (`1789044742717273279` ns), `require_pilot("short")` also passed
with objective evaluation, candidate evaluation and live-client creation explicitly
blocked. It authenticated six completed responses, five eligible generated sources,
six current-parent contexts, six Pareto decisions, two unused terminal decisions,
and zero audit cache rows. Its canonical evidence digest is
`d5464a435d6af43b591803c8f3692626b25cc9429fd4eeb0779cf646128cf37d`.

The short pilot recorded **zero naturally selected non-scalar parents**. This is
allowed by the registered engineering design, which separately requires scripted
coverage of a nontrivial frontier and an actual non-scalar parent. It must not be
described as a live demonstration of that behavior.

## Scientific and implementation review

The exact main grid remains six outer seeds
`[18011,18023,18037,18041,18053,18067]`, four arms `L/M/P/PM`, and 16 responses per
arm: **384 completed-response slots**. The six orders in `driver.main_config()`
match the protocol and give every pairwise precedence exactly three occurrences
in each direction. Generation uses the common sequential provider path; local
evaluation retains eight workers, two local seeds and budget 32.

Each arm receives the same current-parent source and compact TRAIN summary. Only
M/PM add prior-attempt memory. P/PM replace scalar parent selection with uniform
selection from the nondominated 24-instance TRAIN frontier. The production
PrioritySearch update path remains in use. The parent selected by the override is
checked against the actual source reaching the Trace callback and model request.

Historical contexts require already recorded, causally available TRAIN receipts.
They cannot fill a missing evaluation cache entry. Invalid and empty responses
remain in the prior-slot record; only complete valid panels receive numeric AUC.
Source exposure is limited to seven recent distinct nonparent sources and an exact
65,536-character memory serialization budget, with explicit omissions. The smaller
default argument in the pure helper is not the study setting: the owner supplies
the registered limit explicitly.

The shared global barriers defer validation until all generation finishes and defer
audit until all selections and all arm representatives are frozen. Final selection
still includes the original seed. A0 and B2 remain audit controls. The primary
deployment metric uses the same permanent seed fallback and actual remaining
objective budget; invalidity and fallback remain separate reporting fields.

The main allocation is 19,584 TRAIN trajectories, 9,792 validation trajectories and
864 audit trajectories: **30,240 logical trajectories / 967,680 objective-call
allocations**. Actual work, cache reuse and invalid unused allocations must remain
separate. Under valid unique generated sources and the shared seed cache, the
previously calculated physical upper estimate is 926,208 objective calls and
1,852,416 deterministic replay subprocess executions, before separately accounted
reference/integrity work. This is a feasibility estimate, not a measured pilot cost.

The paired six-seed factorial analysis matches the protocol: memory and Pareto main
effects average the two appropriate simple contrasts; interaction is PM−P−M+L.
The shared bootstrap uses the same outer-seed draws for each contrast. All EXP-18
contrasts are exploratory, without simultaneous coverage or a confirmatory
superiority claim. There is no independent arm at N=16. The study therefore cannot
establish an archive benefit over independent N=16 generation, or attribute a
Pareto-policy result to frontier filtering alone.

## Exact requirements before the main freeze

1. Finish the remaining registered M slots, preserving every existing response.
   Authenticate the natural slot-9 context against all nine preceding slots,
   timestamps, source identities and TRAIN receipts. Report the actual number of
   distinct complete sources exposed; nine preceding attempts do not guarantee
   seven usable source examples. Do not replace poor, empty or duplicate responses.
2. Complete the memory selection barrier and `engineering_results.json`. The
   registered stage gate requires at least one eligible generated candidate,
   authenticated evidence, no audit cache access and a no-new-work replay.
   Independently re-authenticate both pilot gates after their writers finish.
3. Preserve the original `PILOT_PROTOCOL.md` and draft manifest. Finalize the main
   protocol with pilot engineering outcomes, actual memory/frontier exposure,
   invalidity, usage and measured feasibility. State that short-pilot non-scalar
   exposure was zero. Preserve and explain any engineering amendment; do not choose
   settings from comparative pilot efficacy.
4. Prepare the main freeze only after both gates pass. Bind the exact grid, order,
   fresh task panels, benchmark, original seed/B2, memory/Pareto adapters, shared
   production/evaluator/analysis code, numeric verifier, tests and environment.
   Authenticate every ZIP member and the archive metadata against that freeze.
   Shared EXP-17 frozen files must remain byte-identical while its run continues.
5. Produce the final EXP-18 manifest binding protocol, canonical freeze, archive,
   both pilot evidence digests and the reviewed operational/reporting helper hashes.
   Check its correspondence to `driver.main_config()` and actual frozen settings
   before the first main request. A separate main prelaunch review should record
   its timestamp and zero-request state before orchestration starts.

The CLI enforces the exact grid and both pilot gates before main generation, and
main preparation also requires both gates. It does **not** independently interpret
the prose status of the protocol or require an external summary manifest. Those
final preregistration checks remain explicit responsibilities of the freeze review;
passing a CLI configuration check alone does not establish that they happened.

## Coverage and remaining limits

Reviewed tests cover closed memory schemas, chronological provenance, missing
receipts, bounded complete-source inclusion, actual four-arm Trace integration,
non-scalar parent selection, restoration of scoped engine bindings, identical
resume contexts, full allocation accounting and both main-entry pilot guards.
The scripted non-scalar test checks the actual production callback parent, not
only a selector return value. The factorial test verifies the per-seed algebra.

Existing recorded regressions report 102 EXP-18 mechanism tests, 24 driver/shared
driver tests, and a later 33-test driver/archive regression. These counts come from
the preserved work log and `exp17/baseline/regression_freeze_drivers.log`; they were
not rerun as part of this read-only review. Direct checks here were pure grid/task
reconstruction, both frozen pilot preflights, and the guarded completed short-pilot
replay. No new objective calls or candidate/model execution occurred.

The long live history check is in M; PM has only a short live pilot. Pure memory
tests and the common implementation cover larger archives, but the short pilot does
not establish long PM prompt reliability or guarantee diverse Pareto frontiers in
main execution. These are declared exposure/interpretation limits, not reasons to
rerun unfavorable responses. Six outer seeds give fragile exploratory uncertainty.
The subprocess boundary still does not provide operating-system security isolation.

Operational resume retains the already documented limits: inspect a surviving child
before resuming; preserve ambiguous remote-request evidence; do not overwrite a
saved analysis if late provider metadata changes reporting. Use the targeted numeric
verifier when only post-analysis verification remains. No new orchestration layer
or scientific design change is recommended by this review.
