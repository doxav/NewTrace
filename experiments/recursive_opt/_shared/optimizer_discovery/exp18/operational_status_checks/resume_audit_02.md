# EXP-17 interrupted-generation resume audit 02

Read-only snapshot: 2026-09-10 15:04:56.844957 UTC
(`1789052696844957450` ns). Recorded-process existence check:
`1789052715068233306` ns. This audit made no model, provider-metadata or network
calls, evaluated no candidate/objective, and changed no frozen source or evidence.
EXP-18 remains under the root operator's separate monitoring; it was not inspected
or interrupted here.

**Conclusion:** an explicit resume of the recorded, incomplete EXP-17 slot is
permitted by the existing frozen resume policy, subject to the operational checks
below. This interruption does not invalidate the 12 completed responses and is
not a scientific result or a failed optimizer candidate.

## Observed interruption and intact response accounting

The second EXP-17 pipeline launch recorded exit code 1 at
`1789052353309741459` ns. Its generation step failed before any pipeline stage
completed. Recorded parent PID 1154868 and child PID 1155207 were absent from
`/proc` at this audit's process check. This is a point-in-time observation; the
root operator owns the final no-active-writer check before launching again.

Pending slot `17001/C/slot_04` contains its immutable request, original generation
start, propagated feedback, `started_1.json` and `attempt_1.json`. It has no ordinary
or compressed response, provider receipt or provider metadata attempt file.

The saved attempt has:

- status `transport_failure`, error type `TransportRetryError`;
- the message marker `incomplete chunked read`, describing the connection closing
  before the response body was complete;
- `transient=false` and `possible_remote_completion_or_duplicate_billing=true`;
- no usage/cost/token fields, provider identifier fields or recognizable `gen-...`
  generation identifier in its sanitized error/log text.

Across EXP-17, all **14 started records have corresponding attempt records**:
12 completed responses and two recorded transport failures. There are no unmatched
starts. The earlier pending C slot 03 has completed; the new interruption is C slot
04. No generation, selection or audit completion barrier exists.

The 103-file source/configuration/environment preflight passed against original
freeze `e520ba38428cf30cecefb76787f73454f4b985143c9ae46a7b9f48e36167c980`,
with objective evaluation, candidate evaluation, live-client creation and secret
loading patched to raise if called. No efficacy values were inspected.

## Why the wrapper stopped

The frozen slot wrapper calls
[`is_transient_provider_error`](../../../../../../opto/features/recursive_opt/measurement.py)
after a returned exception. Its narrow marker list does not match the preserved
incomplete-chunk message or treat the generic wrapped `TransportRetryError` type
as automatically transient. A pure classifier replay on the recorded error text
returned false, reproducing the saved classification. The wrapper therefore
recorded the attempt and stopped instead of automatically requesting a retry.

The inner project retry helper can recognize protocol exceptions through their
causal chain, and `max_retries=1` permits one call before `TransportRetryError`.
The original exception chain is not persisted, so this audit does not reconstruct
its exact runtime class or the network component that closed the connection.
`transient=false` is the frozen classifier's decision, not proof that this network
failure is permanent. Retain the original flag; do not change a classifier or
retry setting during these frozen studies.

Recorded `wall_s=1515.5473279790021` includes any shared generation-lock queue.
As established in [client_timeout_audit_01.md](client_timeout_audit_01.md), the
HTTPX timeout is not a total wall-clock deadline. This duration establishes
neither hidden retries nor where the time was spent.

## Resume guard and remaining uncertainty

The frozen call chain is `exp17.study.Study._proposal` through
`investigation16.evidence_io.complete_slot` to
`investigation16.generation.complete_slot` (lines 105–151). Exact request
persistence first authenticates the saved request; completed responses are reused.
An unmatched started record blocks new calls pending reconciliation. Here the
pending start has a recorded failure return, so the existing guard permits a new
explicit invocation to continue **slot 04 at attempt 2**, retaining attempt 1.
The invocation has the same bounded initial-plus-three-retry policy with 2/4/8-second
delays. A second pipeline launch does not reset evidence or completed slot counts.
The regenerated request must match the original parent, feedback, settings and
identity. Previously completed responses cannot be replaced for any reason related
to their quality.

An incomplete response body does not prove that the upstream generation did not
finish or incur charges. The existing
`investigation16.production_driver.collect_receipts` (lines 70–119) queries provider
metadata by the provider ID of a completed persisted response. It cannot reconcile
this pending slot using its local slot ID alone. No such provider ID is available
in the preserved failure evidence. The audit did not query provider metadata.
If separate activity evidence can reliably identify this request, preserve that
correlation; otherwise keep remote completion, tokens and billing unknown. Do not
record zero usage/cost or invent a recovered response. The previously documented
recorded-failure resume policy permits continuation while retaining this uncertainty;
an unmatched future start would still require reconciliation before another call.

## Conditions for the operator's explicit resume

1. Preserve both launch journals, all completed responses, the original slot-04
   request/feedback and every attempt. Keep lock files and confirm no EXP-17 writer
   remains; do not interfere with the active EXP-18 process or shared lock.
2. Confirm connectivity through non-generative checks and authenticate the existing
   confirmation and engineering gates. Retain the exact model, routing, settings,
   frozen sources and proposal order. No model smoke call is required by this audit.
3. Record the absence of a usable failed-request provider ID and preserve possible
   remote completion/duplicate billing. Reassess if new response or unmatched-start
   evidence appears before the new invocation.
4. Explicitly invoke the existing EXP-17 pipeline, allowing its shared provider
   lock to serialize calls. Do not add an external automatic restart loop.
5. Verify that the 12 existing response hashes remain unchanged and new attempt
   numbering is contiguous. Report both failed attempts separately from completed
   response slots. No unused proposal slot or completed response is replaced.

The root operator separately reports a successful credential-free DNS lookup and
public model-list HTTP 200 with the exact model present, no remaining EXP-17 process,
a passing original confirmation gate, and 4,354 unchanged EXP-17 files. Its saved
[incident_002.json](incident_002.json) has SHA-256
`db6e38471a3c099a527ffe8a54fa5bca0c3023311a0d1edcb3b972d66ec5fa37`,
verified here. Those checks are attributed to that operational record, not rerun
by this audit. This incident alone requires neither a semantic repair nor a new
experiment ID. Both the interruption and unknown provider expenditure remain part
of the final report.

## Integrity anchors

The response collection digest uses `benchmark.digest` over relative response paths
mapped to the SHA-256 of their exact bytes.

| Artifact | SHA-256 / digest |
| --- | --- |
| 12 completed EXP-17 responses | `c80644385cec7cd363fe7cc20cddbf0e77eacffba9a116a2a17472f3d8e3945c` |
| Slot-04 request | `afcf61fac61ae1541dffc56f82c1ecec91ab15a93da3ac8ad50c39d7bc2c89fd` |
| Slot-04 propagated feedback | `f5cd108b9208ebf086e9d1dcd9c6c34c10b13400a588d553dfcb3475fd2e91b1` |
| Slot-04 generation start | `01e89390fe5e1043ed970e875086754a072dc35b9b6ba4f881bff8e91090f056` |
| Slot-04 started attempt 1 | `92515cf593a9efa9e910b04202ddc380ba2081d356b92ba1c6c3a0a06635787d` |
| Slot-04 failed attempt 1 | `77461778ab482553760f551ce180ba0cf6ccf0912a38006d3d6ad6626c2001e3` |
