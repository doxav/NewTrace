# EXP-17 / EXP-18 interrupted-generation resume audit 01

Read-only inspection on 2026-09-10 at 14:24:33.231 UTC
(`1789050273231869994` ns), with process/failure-record checks at 14:25:28.748 UTC.
No generation, objective evaluation, candidate execution, receipt lookup, network
probe, source edit or resume was performed by this audit. The root operator is
separately checking current DNS/HTTPS connectivity.

**Conclusion:** this is a recorded transport interruption, not an invalidation of
the existing scientific responses. Explicit resume is supported by the unchanged
frozen protocol after the operational prerequisites below. Both studies remain
incomplete; these partial runs are not scientific negative results.

## Preserved state

| Study | Completed responses | Uncompleted slot | Recorded failed attempts | Next attempt index |
| --- | ---: | --- | --- | ---: |
| EXP-17 | 11 | `17001/C/slot_03` | 1: DNS; recorded transient=false | 2 |
| EXP-18 | 2 | `18011/L/slot_02` | 1: timeout, transient=true; 2: DNS, transient=false | 3 |

All 12 EXP-17 and four EXP-18 `started_*.json` records have corresponding attempt
records. Neither pending slot has `response.json` or a compressed response. No
generation-completion, selection or audit barrier exists for either main study.
The original per-slot requests remain present. There is no basis for replacing
any of the 13 completed responses.

Both first pipeline launches recorded failure exit code 1 at the generation stage,
with zero completed stages. Their recorded generation-child PIDs, 820068 and
1010153, no longer existed at the process check. This supports the root operator's
separate observation that all children had exited; it is not a substitute for a
fresh process/lease check immediately before resume.

Both exact main preflights passed while objective evaluation, candidate evaluation,
live-client creation and credential loading were explicitly blocked:

- EXP-17 freeze: `e520ba38428cf30cecefb76787f73454f4b985143c9ae46a7b9f48e36167c980`.
- EXP-18 freeze: `ce88c4b2d9530456e63b497f7bbde7608fbf3d8aca441c13c086d77b6beeea8a`.

These checks authenticated the frozen sources/configuration/environment. No
confirmatory objective values or candidate behavior were inspected.

## Root cause of the stop

There are two existing transport classifiers:

1. `opto/utils/auto_retry.py::_is_retryable` traverses exception chains and includes
   the message marker `temporary`. It recognizes the recorded DNS failure. With
   the configured `max_retries=1`, its loop makes **one attempt**, then raises
   `TransportRetryError`; that parameter does not grant a hidden second call.
2. `opto/features/recursive_opt/measurement.py::is_transient_provider_error`, used
   by the frozen slot wrapper, matches a narrower list against the exception's
   type/text. It includes timeout markers but does not include `temporary failure
   in name resolution`, general `temporary`, or an explicit rule for the wrapped
   transport exception. It does not traverse the exception chain.

Pure classifier replay on the recorded sanitized errors reproduced the flags:
both classifiers returned true for the timeout; the inner classifier returned
true and the slot classifier false for each DNS error. Therefore the slot wrapper
preserved the failure and stopped immediately at DNS instead of consuming its
remaining automatic retry allowance. The pipeline then stopped, as designed.

The stored transient=false is a classifier result, not proof that this DNS fault
is permanent. Do not rewrite those records to true. A future offline engineering
change could reconcile the classifiers, but changing frozen transport behavior
during these runs is unnecessary and is not recommended by this audit.

The failure-record `wall_s` values are approximately 1419.203, 1562.294 and 0.011
seconds. The slot timer starts before entering the shared provider lock, so it can
include time queued behind the other study. These measurements are not isolated
provider-request durations and do not prove that hidden model retries occurred or
that the configured 300-second request timeout was changed.

## Why unchanged-protocol resume is possible

`investigation16/generation.py::complete_slot` first authenticates the exact saved
request through immutable persistence. If a response exists, it returns that
response. If a started record has no corresponding attempt record, it stops for
remote-completion reconciliation. Otherwise an explicit invocation starts a new
bounded batch after the existing attempt count.

Here every started attempt has a recorded transport failure or completed-response
attempt. The two pending slots have known local failure returns, rather than
unmatched in-flight records. Under the registered policy, a fresh explicit
invocation can therefore continue at EXP-17 attempt 2 and EXP-18 attempt 3. Each
invocation permits an initial attempt and at most three transport retries with
2/4/8-second delays; an additional explicit resume does not erase earlier attempts.

The original request identity, parent, settings, model and prompt remain fixed.
For EXP-18, the existing slot-02 current-parent context and its original availability
cutoff must also remain fixed. Production Trace reconstruction reuses earlier
responses and authenticated TRAIN receipts; it is not permission to generate a
new version of an already completed proposal. Global validation/audit barriers
remain in force.

## Unknown remote completion and billing

Every failed attempt already has
`possible_remote_completion_or_duplicate_billing=true`. Preserve this flag for
all three failures. In particular, a client timeout does not prove the upstream
generation never completed or was never billed. Even the DNS failures should not
be relabelled as certainly unbilled based only on the local exception.

No provider generation identifier was found in the failed-attempt identifier
fields or recognizable generation-ID text in their sanitized error/log fields.
The existing receipt collector queries by the IDs of persisted completed responses;
it cannot retrieve a pending slot by the experiment's local slot ID alone.

Use any available read-only provider activity/metadata to reconcile these time
windows if it can be correlated reliably. Record the source and outcome of that
check. If no correlation is available, retain the uncertainty explicitly; do not
invent an ID, a zero cost, a zero-token count or a recovered response. The frozen
protocol permits explicit continuation after recorded transport failures while
preserving this uncertainty. An unmatched started record on a future interruption
would still require reconciliation before another attempt.

Before resume there are 13 completed responses and three recorded failed attempts,
16 locally observed request attempts in total. The failed attempts do not consume
completed-response slots. Report their unknown tokens/cost separately from known
completed-response usage. Equal completed-response allocation is not a claim of
equal realized model compute or billing. A subsequently discovered billing record
does not justify selecting a different response after observing its quality.

## Explicit-resume prerequisites

1. Preserve the interrupted journals, requests, responses, context, source/cache
   evidence and attempt files. Keep all existing pipeline launch records and locks.
   Do not delete a lock file to bypass an active process. Confirm that no surviving
   pipeline/driver/candidate worker owns the run and that DNS/HTTPS access has
   recovered. A local/provider reachability check must not add a generative smoke
   call outside the registered slots.
2. Re-run the existing main and pilot gates through the normal CLI entry point.
   Keep the frozen model, routing policy, token limit, timeout, worker count and
   exact requests. No package, retry-classifier or scientific source change is
   necessary for the present interruption.
3. Document the available remote-metadata reconciliation and remaining billing
   uncertainty. The current matched failure records satisfy the implemented resume
   guard; if any new unmatched started record or completed response appears,
   reassess that record before issuing another request.
4. Invoke the existing pipeline explicitly, creating a new operational launch:

   ```bash
   /tmp/phase0-venv/bin/python -m artifacts.optimizer_discovery.exp17.run_pipeline exp17
   /tmp/phase0-venv/bin/python -m artifacts.optimizer_discovery.exp17.run_pipeline exp18
   ```

   Use these only after the preceding checks. The shared generation lock maintains
   one active provider call. Internal arm order and pending request identities
   remain frozen. Do not add an external automatic retry loop around the pipeline.
5. After continuation, verify that all 13 original response hashes remain unchanged,
   new attempt numbering is contiguous, and completed responses still consume
   exactly one slot each. Preserve interruption/queue time in secondary runtime
   reporting. A further recorded transport failure should stop and be documented
   under the same policy; it does not authorize replacements or a model change.

This incident alone requires no result invalidation or new experiment ID: no
completed response, evaluation, selection, metric or proposal allocation was
changed. A later semantic repair or irreconcilable identity mismatch would need
its own assessment under the frozen defect policy.

## Integrity anchors for the operator's preservation record

The completed-response collection digests below use `benchmark.digest` over the
mapping from each relative response path to the SHA-256 of its exact stored bytes.

| Artifact | SHA-256 / collection digest |
| --- | --- |
| EXP-17 completed responses | `9d88281704b5503045e74f453f3a9fbf3101a6be5b5f9f3e001f56037276d6a3` |
| EXP-18 completed responses | `97b9023613b22233f6e1f7a4629930c4eff1d7dfd7d1c0cb24463ea45782d7ee` |
| EXP-17 pending request bytes | `65a5eb681f93abcd0cddb6abf6fa7958efd03ed9cfb2b1ba48742799c8cd90bf` |
| EXP-17 DNS attempt bytes | `bac9eb9cc4b7cabe502d426e832e1520e573611ce93673ef71ee4e71abe65305` |
| EXP-18 pending request bytes | `ca66740a4acf99dba8cbdf0b6ae7f2b3dd732dfca3dd034fb594e4bb54419d09` |
| EXP-18 pending context bytes | `d34b52f8d347e886084c8fe3895695c5bf9dabc735fe1bd1afb77c78e8534e26` |
| EXP-18 timeout attempt bytes | `bd4cf51ffd948ac7393eda4262293c97898f390d624ad21b75d2864fb82466aa` |
| EXP-18 DNS attempt bytes | `4f22e2c52d0c1bb0170fc56e6b2da2fcd7f0140376843547d266e3bb78a0e08c` |
