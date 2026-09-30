# EXP-18-only differential resume audit 06

Read-only check: 2026-09-11T20:35:51.471731+00:00 to
2026-09-11T20:35:52.456349+00:00.

**PASS for the existing explicit-resume interpretation: next attempt 2 for
`18037/L/slot_00`**, subject to the operator's EXP-18 inventory, engineering/archive,
connectivity and no-active-EXP-18-writer checks. No new blocker was found compared
with the same incomplete-response-body interruption in
[resume audit 02](resume_audit_02.md). EXP-17 remains under the operator's separate
monitoring; this audit neither inspected/interfered with its run nor requires it
to stop. Keep the shared provider lock and both studies' lock files intact.

EXP-18 launch 006 recorded exit code 1 at `1789158787201539688` ns during generation,
with zero completed pipeline stages. Its recorded parent 225402 and child 232639
were absent from `/proc` at this snapshot. The operator reports 160 completed
responses; this differential audit does not repeat that full response census.

The pending slot has exactly matched `started_1.json` / `attempt_1.json` and no
ordinary/compressed response, provider receipt or metadata lookup directory.
The saved failure is `TransportRetryError`, contains the incomplete-chunked-read
marker, and has `transient=false`. Pure outer-classifier replay reproduced false.
Its `possible_remote_completion_or_duplicate_billing=true` flag remains intact;
there are no usable provider-ID fields, recognizable generation IDs in sanitized
error/log text, or reported usage/token/cost fields.

The original main configuration gate authenticated all 113 frozen files against
freeze `ce88c4b2d9530456e63b497f7bbde7608fbf3d8aca441c13c086d77b6beeea8a`.
Request namespace/outer/arm/slot, exact model/settings, original seed-parent hash
and propagated-feedback hash match. `MechanismStudy.messages` reconstructed the
saved context and messages exactly using original cutoff
`1789156757307307936` ns. The current parent is the unchanged seed (`current_index=-1`);
L correctly has no prior-attempt memory. Cached TRAIN receipt authentication was
read-only, and persistence was replaced by an exact-existing-bytes comparator.
All six pending files were unchanged; no missing evidence was filled and no new
objective/candidate evaluation, model/network/metadata call, key load, source edit,
process action or commit occurred. No comparative efficacy was inspected.

The frozen `generation.complete_slot` guard authenticates the original request,
reuses any completed response, blocks unmatched starts, and otherwise appends a
new explicitly invoked bounded batch after recorded attempts. This matched local
failure therefore permits attempt 2 without replacing completed proposals or
changing the registered retry policy. Preserve the original request/context and
all completed responses. No model, timeout, routing, TLS or classifier change is
needed for this interruption.

Recorded `wall_s=2028.624880131014` can include waiting for the shared generation
lock; the existing phase-timeout qualification in
[client_timeout_audit_01.md](client_timeout_audit_01.md) applies. Duration alone
does not establish extra retries, a particular network cause or remote completion.
An incomplete body does not prove nonexecution or zero billing. Receipt lookup
requires a provider generation ID, unavailable in this failed record; keep unknown
remote completion and expenditure explicit. Reassess any newly discovered unmatched
start or conflicting response before another call. This interruption alone
invalidates no scientific response and requires no new experiment ID.

| Pending evidence | SHA-256 |
| --- | --- |
| Request | `e49a5b7d1d2964b355327c1f58376f1dc20be1b2598db75cc572926e6c64a014` |
| Generation start | `6c08fdf1e271ad3cb0ec2aee0c1b79c4443c3a51f73b187148702f2148315a83` |
| Propagated feedback | `14815dffecf586712a9236dee882a1ed836c19b26e23b6bfbcce070f743f98c2` |
| Context bytes | `c97585cb190ebbc26bcb75227817f3ebf1eda5383afcf18a7db2a5df6dade292` |
| Started attempt 1 | `eec2d2ad82b1702847fd66dc6a6db30df478c58aacab42b64407d98e97e305ce` |
| Failed attempt 1 | `896747cb7e8572942078738a63f3e91c4c979def0ebccb0b48a5c26b21c57735` |

Unchanged relevant sources: `investigation16/generation.py`
`6a5a66b9b12fd4f4217668a5803bd92ad660a5e35d78a42cd45ec7882dd6d24c`;
`exp18/study.py`
`22c653204145fe77a2a8619a86182edb0ce1d8f83b886a9b730f06e2fc5670cf`.
No regression suite was rerun for this read-only differential audit.
