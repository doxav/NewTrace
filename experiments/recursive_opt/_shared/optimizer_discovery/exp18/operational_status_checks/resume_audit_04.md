# EXP-17 / EXP-18 differential resume audit 04

Read-only check interval: 2026-09-11T12:05:12.350534+00:00
to 2026-09-11T12:05:12.711337+00:00.
Nanosecond anchors: `1789128312350533953`–`1789128312711336831`.

**Decision: the new SSL EOF failures do not change the existing recorded-failure
resume interpretation.** After the root operator completes the current incident
inventories and final connectivity/no-writer checks, the unchanged protocol permits
EXP-17 to continue at attempt **2** and EXP-18 at attempt **3** in their existing
pending slots. No new scientific blocker was found and no launch was performed.

This is an incremental review of the new interruptions, not another full raw-data
audit. It relies on [audit 03](resume_audit_03.md), whose SHA-256 remains
`a5f702423118bcd39661aaa584ddbf59f07ebfd629d6184c34a786a88d97dfea`,
and the root operator's new incident-006/007 preservation checks. It inspects only
the new pending attempts, their immutable request/context identities, main freeze
gates, recorded process status and completion-file counts. It does not reauthenticate
all historical raw responses, cache files or starts; those broader current-run
checks remain with the root operator.

## New observed state

| Item | EXP-17 | EXP-18 |
| --- | --- | --- |
| Completed response files | 140 | 129 |
| Pending slot | `17009/C/slot_04` | `18037/M/slot_01` |
| Pending started / attempt records | 1 / 1 | 2 / 2 |
| New recorded failures | attempt 1: SSL unexpected EOF | attempt 1: connection reset; attempt 2: SSL unexpected EOF |
| Saved transient flags | false | true, false |
| Next attempt index | 2 | 3 |
| Fourth launch exit code / completed stages | 1 / 0 | 1 / 0 |
| Exit timestamp, ns | `1789127856493907523` | `1789127907510379311` |
| Recorded parent / child PID | 3381330 / 3381532 | 3381735 / 3388799 |

All pending starts have their corresponding failure records, with contiguous
attempt numbers. Both pending slots lack ordinary or compressed responses,
provider receipts and metadata lookup directories. All three pending failures
are recorded as `TransportRetryError`, preserve
`possible_remote_completion_or_duplicate_billing=true`, and contain neither a
provider-ID field nor a recognizable generation ID in their sanitized error/log
text. No usage/token/cost fields are present.

All four recorded processes were absent from `/proc` during this check. Neither
main study has a global generation, selection or audit-completion barrier.
The response-file counts above are directory counts, not newly recomputed
scientific results or a repeat of the prior full response-integrity audit.

## Interpretation of the SSL EOF

The new errors contain `UNEXPECTED_EOF_WHILE_READING`; the earlier EXP-18 attempt
contains the connection-reset marker. Pure replay of the frozen outer classifier
reproduced false for both SSL errors and true for the connection reset. The narrow
marker list in `measurement.is_transient_provider_error` explains why the wrapper
recorded the SSL failure and stopped rather than taking another automatic attempt.
The stored flags must remain unchanged.

This observation does not establish the remote component responsible, whether the
request reached generation, whether the response completed remotely, or whether it
was billed. It also does not establish a certificate-validation defect. **Keep TLS
verification enabled and retain the existing client, model, routing, timeout and
classifiers.** Disabling TLS or rewriting retry classification is neither needed
nor justified by this audit.

The pending wall times are approximately 324.642 seconds for EXP-17 and 24.243 /
27.118 seconds for EXP-18. The existing queue/timeout qualification from
[client_timeout_audit_01.md](client_timeout_audit_01.md) still applies; these totals
do not establish hidden retries, time spent at a particular network layer, or
remote completion/noncompletion.

## Immutable request and EXP-18 memory reconstruction

Both pending requests match their original freeze digest, exact model/settings,
registered namespace/outer/arm/slot identity, shared invariant prompt and saved
propagated-feedback hash. The unchanged main gates passed with objective evaluation,
candidate evaluation, client creation, slot completion and credential loading blocked:

| Study | Original freeze digest | Authenticated frozen files |
| --- | --- | ---: |
| EXP-17 | `e520ba38428cf30cecefb76787f73454f4b985143c9ae46a7b9f48e36167c980` | 103 |
| EXP-18 | `ce88c4b2d9530456e63b497f7bbde7608fbf3d8aca441c13c086d77b6beeea8a` | 113 |

EXP-18's early M slot-01 retains original snapshot
`1789127853220014131` ns and `before_slot=1`. Its memory contains the one completed
prior attempt, with the registered seven-source / 65,536-character limits; saved
memory text is 3,201 characters. `MechanismStudy.messages` reproduced both saved
messages and the context exactly from the saved parent/feedback and existing TRAIN
receipts. The original cutoff was reused, not advanced to the present time.

Persistence was replaced with an exact-existing-bytes comparator, and the existing
cache path was read-only. Missing evidence would have raised; no new evaluation,
request or evidence write was permitted. All five EXP-17 and eight EXP-18 pending
files were byte-identical before and after the check. TRAIN data was used only for
immutable context authentication; no comparative efficacy values were exposed.

## Safe continuation under the existing guard

The frozen `generation.complete_slot` first authenticates exact request persistence,
returns an existing response, blocks unmatched starts, then numbers a new explicitly
invoked bounded batch after the existing attempts. Here the pending starts have
recorded local failure returns, so the next indices are 2 and 3. The initial attempt
plus at most three transport retries and 2/4/8-second backoff remain unchanged for
that invocation. Earlier attempts remain evidence; no completed proposal is replaced.

Retain unknown remote completion, tokens and billing for all three new failures.
The receipt collector needs a provider generation ID, and none is available in the
pending records. Do not infer zero billing or nonexecution from TLS EOF or connection
reset. If independent provider activity can be correlated reliably, preserve the
correlation; otherwise the uncertainty remains explicit. A new unmatched start or
conflicting response discovered before continuation would require reassessment.

Before an explicit existing pipeline launch, the root operator must finish its
incident-006/007 inventories, confirm no active writer owns either run, and check
connectivity without a generative smoke request. Keep lock files and the shared
provider-call serialization. After continuation, verify preserved response and
pending request/context hashes and contiguous attempt numbering. No automatic outer
restart loop, TLS bypass, source change or completed-response replacement follows
from this decision.

These remain incomplete studies. The new transport interruptions alone invalidate
no scientific response and require no new experiment ID. Ordinary generated failures
and unfavorable results remain retained independently of transport failures.

## Integrity anchors

| Pending evidence | SHA-256 |
| --- | --- |
| EXP-17 request | `92b6979b82b2d120812af23f03210ef6c18c56bccbb8fb58131ae08e6be0aabd` |
| EXP-17 feedback | `cb7cb6c15642b5b98b71f41349ac3798c1dc484a0a4d5fe92dffd09d36284e91` |
| EXP-17 SSL attempt 1 | `09361db2d0941fc4d27cf48e88a756348970e49142a3334c5cdb07df9e481336` |
| EXP-18 request | `c10034d16df7461a9818fe6008176ae2295ba5ee72b9fff2b5a95055f5099a28` |
| EXP-18 context bytes | `c581fdd12c31fdb00d871af7a1e6bc1e00e1d07f83cd3f37c0da1863e708965f` |
| EXP-18 context canonical digest | `5e04b71ff41798071bc919dc0a93fe8b22e2fe902833a0931d963f6f832ce060` |
| EXP-18 memory text | `9dbffcd59f2ba9bf120326d7b289a56776729fa805888e570191848ae10c0656` |
| EXP-18 feedback | `14815dffecf586712a9236dee882a1ed836c19b26e23b6bfbcce070f743f98c2` |
| EXP-18 connection-reset attempt 1 | `1bfed44e484c2f78589905cd4814922a42bc80fd1cffd2b6c461c887b0057f80` |
| EXP-18 SSL attempt 2 | `689e1eebff8eb122fc15339f88b44c04c4cbde17853074e6352a734ebf90b4af` |

Relevant sources retain the same hashes recorded in audit 03:

| Source | SHA-256 |
| --- | --- |
| `investigation16/generation.py` | `6a5a66b9b12fd4f4217668a5803bd92ad660a5e35d78a42cd45ec7882dd6d24c` |
| `exp17/study.py` | `f16cfd1eac80f1aa27584e6f654ff5c43abe46aeb57265fd67496bc4aeb48b5b` |
| `exp18/study.py` | `22c653204145fe77a2a8619a86182edb0ce1d8f83b886a9b730f06e2fc5670cf` |
| `opto/features/recursive_opt/measurement.py` | `e667909e92d9fbe43baf088f4e61ed976755f9e9db9213b6ca0e8f6eb7b4a842` |

No model/network call, candidate/objective execution, process intervention or frozen
edit occurred. No regression suite was rerun for this read-only differential review.

Operator completion update received after the initial record: the root operator
reports successful credential-free DNS (four records) and verified-TLS public
model-list HTTP 200 with the exact model present at 12:04:22 UTC. It also reports
passing original main/engineering/source-archive gates under execution blockers,
and unchanged whole-run snapshots before and after its checks. These checks are
attributed to the operator and were not repeated by this differential audit.

| Operator incident | Files preserved | Completed responses | Historical transport failures | Reported SHA-256 |
| --- | ---: | ---: | ---: | --- |
| [incident_006.json](incident_006.json) | 51,222 | 140 | 10 | `5115523048799c8ba8f1f272c248c587967e9fb9bbd4f0c9e320df13089549b9` |
| [incident_007.json](incident_007.json) | 40,894 | 129 | 13 | `39618234491014883aa5dfe2d2cf3c3ee503aad0ea7410d0d0c723fa879d1b8f` |

The remaining fresh no-writer/hash check is owned by the operator immediately
before its explicit fifth launch. No additional scientific blocker was found.
