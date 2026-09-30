# P1 driver review before any production-stage freeze

The driver is ready for the **P1-E1 engineering check** after the shared regression
suite passes. This is not a claim that the two live engineering responses already
passed, or that the larger efficacy diagnostic is authorized by a simulated test.
`prepare(engineering=False)` still requires complete passing real engineering
evidence. No live requests, prospective evaluations, or frozen F1 files were
changed during this review.

The review reproduced and fixed these defects with unit tests:

1. **Incomplete engineering proof accepted.** A result containing only
   `passed=true`, the freeze hash and an empty response-hash map previously opened
   the gate. It now requires the exact two R/16601 slots, settings and source
   integrity, lineage, actual propagated feedback, registered callback schedule,
   three seed-inclusive training allocations, and unchanged semantic evidence.
2. **Stale passing evidence survived cache changes.** Changing a training row or
   its split to validation previously went undetected by the gate. Raw requests,
   responses, propagated feedback, Trace, schedules, allocations and training
   caches are now included in the engineering evidence hashes. The scope check
   rejects other arms or any validation/audit cache/barrier. The known seed must
   be eligible and at least one generated program must complete the training panel.
3. **Compressed evidence was discovered but not readable.** The existing EXP-15
   reader accepts a logical `.json` path and transparently finds its compressed
   counterpart; it does not decode a directly supplied physical `.json.gz` path.
   The driver now normalizes discovered paths without changing the old reader.
   A complete simulated production run tests actual compressed responses and
   caches, integrity gating, and resume. The shared search module independently
   received the equivalent fix for its cache scan.
4. **Provider receipts lacked identity validation.** A receipt for another
   generation could previously be persisted. Both newly fetched and cached
   receipts now require the response ID. Tests cover all I/C/R/W paths, compressed
   storage, whitelist removal of a synthetic private field, and separately
   recorded transport errors whose messages contain a synthetic credential.
5. **A completed engineering result skipped receipt recovery.** Re-entering the
   completed stage now collects any missing receipts before returning through the
   full gate. It does not issue replacement generation calls. Two ordinary invalid
   responses remain preserved as a failed engineering check on every resume.

The driver reuses `S.verify_arm_responses()` and the production saved-Trace checks
instead of creating another search loop or duplicating the response validator.
The full source/config/environment freeze remains authoritative. Replay runs with
no callable client and compares semantic records before and after; routine cache
access events are not mistaken for new evaluations. Receipt files and retrieval
attempts are separately append-only metadata, so later receipt recovery does not
invalidate the scientific evidence snapshot.

The parent's separately tested prompt limit correction is **524,288 characters**
for both P1-E1 and P1, documented before either stage freezes. The driver enforces
the same value through its registered configuration. No generation setting or
scientific selection rule was chosen from new comparative performance outcomes.

The tests use the `UNIT-P1-DRIVER` namespace and synthetic evaluation helpers with
real production Control Plane/PrioritySearch scheduling. No candidate is executed
on prospective engineering or P1 tasks by these unit tests. Both valid and invalid
completed responses consume their slots. The subprocess execution boundary and
credential sanitation remain those of the existing evaluator; no stronger OS
confinement claim is introduced.

Verification commands, final results and source hashes are recorded in
`driver_review_verification.json`. The initial test run also exposed and corrected
a test-fixture directory omission. A gzip stress fixture was adjusted to put its
large text outside a short fenced program, so it tests packaging rather than
violating the intentionally bounded prompt size. These were synthetic unit
fixtures; no model response or scientific outcome was altered.
