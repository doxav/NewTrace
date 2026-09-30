# EXP-17 / EXP-18 interrupted-generation resume audit 03

Read-only checks ran on 2026-09-11 from 08:08:29.275179 to 08:08:35.290904 UTC
(`1789114109275179142`–`1789114115290903985` ns). This follows
[resume audit 01](resume_audit_01.md) and [resume audit 02](resume_audit_02.md).
The earlier description of EXP-18 as active in audit 02 is historical; its second
launch subsequently stopped and the root operator documented that in incident 003.

**Decision: the existing explicit-resume policy supports continuing both pending
slots without changing scientific semantics.** EXP-17 continues at attempt 3;
EXP-18 continues at attempt 2. No unmatched start, changed frozen identity, missing
historical context, or conflicting evidence was found. The root operator must
complete its final preservation, connectivity and no-active-writer checks before
launching. This audit did not launch, stop or attach to a process.

No model, provider metadata or network call was made. Candidate/objective execution,
live-client creation and credential loading were blocked during the checks.
Historical TRAIN evidence was authenticated only to reconstruct the saved EXP-18
prompt; efficacy values and candidate behavior were not exposed or compared.
No frozen source, scientific record or installed package was edited.

## Exact interruption state

| Item | EXP-17 | EXP-18 |
| --- | --- | --- |
| Completed responses / unique provider response IDs | 120 / 120 | 108 / 108 |
| Saved requests | 121 | 109 |
| Started records / recorded attempts | 128 / 128 | 119 / 119 |
| Recorded completed attempts / transport failures | 120 / 8 | 108 / 11 |
| Unmatched started records | 0 | 0 |
| Pending slot | `17008/C/slot_00` | `18023/M/slot_12` |
| Pending recorded failures | attempt 1: timeout; attempt 2: DNS | attempt 1: DNS |
| Next attempt under unchanged guard | 3 | 2 |
| Third pipeline launch exit | 1, generation stage | 1, generation stage |
| Launch completion, UTC | 07:54:43.530382 | 07:54:28.706965 |
| Recorded parent / child PID | 1235795 / 1236170 | 1235848 / 1240622 |

All four recorded processes were absent from `/proc` at the audit snapshot.
Both launch journals record zero completed pipeline stages. No global generation,
selection or audit-completion barrier exists in either study.

Both pending slots have no ordinary or compressed `response.json`. Their failure
records have no usable provider identifier fields or recognizable `gen-...` ID in
the sanitized error/log text; their directories contain no provider receipt or
metadata lookup record. All three pending failure records retain
`possible_remote_completion_or_duplicate_billing=true` and have no reported
usage/token/cost fields. Their `TransportRetryError` types and transient flags are
preserved. Pure replay of the outer classifier reproduced timeout=true and
DNS=false for the corresponding saved error text.

Across each study, attempt numbers are contiguous within every slot. Completed
responses were reparsed with the existing frozen response verifier, checking their
exact source hashes and identities; each maps to its completed attempt and matching
provider ID. Every existing request belongs to the exact registered schedule
prefix and retains its frozen model, settings, invariant prompt and slot identity.
All non-independent requests retain the recorded propagated-feedback hash.
The next scheduled slots are exactly the pending slots above, not replacement
slots selected after observing outcomes.

## Frozen implementation and historical preservation

The original main configuration gates passed with forbidden execution operations
patched to raise:

| Study | Original freeze digest | Frozen files authenticated |
| --- | --- | ---: |
| EXP-17 | `e520ba38428cf30cecefb76787f73454f4b985143c9ae46a7b9f48e36167c980` | 103 |
| EXP-18 | `ce88c4b2d9530456e63b497f7bbde7608fbf3d8aca441c13c086d77b6beeea8a` | 113 |

The unchanged `require_confirmation` / `require_main` paths reconstructed source,
environment, task, seed, order and configuration identities. All 4,354 files in
the earlier EXP-17 incident-002 inventory and all 1,123 files in the earlier
EXP-18 incident-003 inventory remain byte-identical, including their original
completed responses and then-pending request/context evidence. Raw evidence was
also hashed before and after this audit: all 979 current EXP-17 and 1,101 current
EXP-18 raw files were unchanged. This does not replace the root operator's broader
current-run inventories.

The root operator separately reported credential-free DNS success (four records),
public model-list HTTP 200 with the exact model present at 08:06:02 UTC, and
successful original main/engineering/source-archive checks with calls blocked.
[incident_004.json](incident_004.json) contains its EXP-17 preservation inventory of
38,318 files; its SHA-256 was independently verified as
`9a936fc3ad343d8728c41680729745e6b1119f13a74c388512bc6132b484b7b8`.
The analogous incident-005 inventory and final writer check were still being
prepared by the root operator when this audit record was written. Their completion
is an operational prerequisite, not a permission to alter existing evidence.

## EXP-18 long-memory replay identity

The pending M slot-12 `current_context.json` preserves its original cutoff
`1789112858276439989` ns (2026-09-11 07:47:38.276440 UTC), rather than using the
current wall clock. Its memory snapshot has `before_slot=12` and all 12 completed
prior slots from the same arm and outer seed. Every prior response and TRAIN
availability timestamp precedes that original cutoff.

The audit called only `MechanismStudy.messages` for this pending slot, with the
exact saved parent source and propagated feedback. The inherited historical TRAIN
receipt path read existing cache entries in read-only mode. Immutable persistence
was replaced by an exact-existing-bytes comparator, so reconstruction could not
write, fill missing evidence, or silently change a record. Missing cache entries
would have raised; candidate/objective execution and slot completion were blocked.

The reconstructed context and both prompt messages matched their saved originals
exactly. Memory retained its frozen limits of seven source entries and 65,536
characters; the actual saved memory text is 65,448 characters. This proximity to
the cap does not justify trimming or rewriting a pending prompt. Whole-source
omissions and all earlier attempt summaries remain as originally constructed.
The audit found no context overflow or replay defect.

| Pending memory identity | Digest |
| --- | --- |
| Exact context-file bytes | `cd0e151151058501e996a7cf42fc1be07d8cac7667482efe5ae8fc370eecf07b` |
| Canonical context object | `41f953f1d92607936941865d5c9d943f92f766793abd95440ed2ab63c162324b` |
| Exact memory text | `0a0a540a26135e0135d3d1acbba9054270cd7487f2586d8031bdf9ad99d29665` |

For all saved EXP-18 request contexts, the audit additionally checked arm/seed/slot,
parent and propagated-feedback identities. Every memory-bearing context retained
its own original snapshot, complete strict-prior slot range, text hash and causal
TRAIN/response timestamps. It did not regenerate historical candidate programs or
recompute comparative search results.

## Why explicit resume remains supported

The frozen path is `Study._proposal` → `evidence_io.complete_slot` →
`generation.complete_slot` (lines 105–151). Immutable request persistence first
requires exact identity. An existing response is returned; an unmatched start
blocks another call pending reconciliation. A matched, recorded transport failure
leaves the slot incomplete and permits a new explicitly invoked bounded batch.
Attempt numbering starts after existing attempts, giving 3 and 2 here. Each
invocation retains the initial attempt plus at most three transport retries with
2/4/8-second delays; no prior attempt is erased and no completed response receives
a replacement because it is poor or invalid.

The saved DNS=false result follows the existing outer classifier's narrower text
markers. It does not establish that DNS failure is permanent. Preserve the flags
and existing classifiers. Neither these DNS errors nor the preceding timeout
require an in-run model/settings/source repair. As documented in
[the timeout audit](client_timeout_audit_01.md), queued lock time and HTTPX phase
semantics prevent inferring hidden retries from wall-clock duration alone.

A local timeout or DNS error does not establish that the remote request was
unexecuted or unbilled. The existing receipt collector needs a provider generation
ID from a completed response; the local experimental slot ID alone is insufficient.
No such identity is available in these three pending failures. Preserve their
unknown remote completion and expenditure. Correlate separately available provider
activity only if identity can be established reliably; do not invent zero costs,
zero tokens or a recovered response. Any newly discovered unmatched start or
conflicting response must be assessed before another call.

The two studies currently have 228 completed response slots and 19 recorded
transport failures, 247 local attempts in total. Retain all of them. Equal completed
response allocation remains distinct from realized token expenditure, provider
billing and elapsed time. The partial studies are incomplete, not scientific
negative results. No scientific invalidation or new experiment ID is warranted by
these recorded interruptions alone.

## Operator's next action and post-resume checks

After the root operator has saved both current-run inventories and completed its
fresh writer/connectivity check, use the existing explicit pipeline invocations.
Keep lock files, frozen requests, sources, contexts, settings, arm order and all
completed responses. Do not add an automatic outer restart loop or unregistered
model smoke call. The shared provider lock continues to serialize EXP-17/EXP-18.

After continuation, verify the 120 and 108 pre-existing response hashes, prior
attempts, pending request/context bytes and contiguous next-attempt numbering.
Preserve any further failure and its uncertainty. No classifier, memory cap,
proposal allocation or scientific design change is authorized by this audit.

## Integrity anchors

Collection digests below use `benchmark.digest` over relative response paths mapped
to the SHA-256 of their exact stored bytes.

| Evidence | SHA-256 / collection digest |
| --- | --- |
| 120 EXP-17 completed responses | `bec9447b1400d6004364fd333084dd9a03ba42296fee6ac0b0f6ff6baa2d6714` |
| 108 EXP-18 completed responses | `1dd8e8be9f732e09debd25729488c627f271bd01c331bde1361ebdbe9195524f` |
| EXP-17 pending request | `a79d8f44c08af68c53d7c3eefa77f1083b0119d014e7c2f5225012e66d1442e9` |
| EXP-17 pending propagated feedback | `a77dcb15aa7fb5c11ce0f0cb29882e3dafd1c75a94f2f7fd6878d3a870b92f99` |
| EXP-17 timeout attempt 1 | `dba6c3830c069ffdf04a51b91a98bce67e14567de3a2a958fc3879a9b1d7f8ad` |
| EXP-17 DNS attempt 2 | `03ab7feb9aeebb8b179bd8227e970b501d544c2c6cc7902de3baec66d7815ee7` |
| EXP-18 pending request | `ee98175e78098eaaf0d5a950d2f3153a177876ff9fbae356a92634cbb1a4587c` |
| EXP-18 pending propagated feedback | `a56a468cfa3652fce85f290c138873e8dc8905c4551b4bf9a8dc529896c63e5b` |
| EXP-18 DNS attempt 1 | `5ed627a5d3aee23162c2a9cb32123fa3ccb307b6a989b878ba1d92e0d4468b4f` |

| Relevant frozen source | SHA-256 |
| --- | --- |
| `investigation16/generation.py` | `6a5a66b9b12fd4f4217668a5803bd92ad660a5e35d78a42cd45ec7882dd6d24c` |
| `investigation16/evidence_io.py` | `fe0ca40d7c26c05598f38a76fdab2679375349dfad2008af3417c31bfdee3888` |
| `exp17/study.py` | `f16cfd1eac80f1aa27584e6f654ff5c43abe46aeb57265fd67496bc4aeb48b5b` |
| `exp18/study.py` | `22c653204145fe77a2a8619a86182edb0ce1d8f83b886a9b730f06e2fc5670cf` |
| `exp18/memory_projection.py` | `0dc1a6d414b60176e737f8d914561cfcf01edcbb47d4e186e92b414e71210391` |
| `exp17/driver.py` | `be19f5bc162e1f4891c81a2e68825bb3873171363c2ebfe0b11b8201b38b0236` |
| `exp18/driver.py` | `62da3ad70c1d05d5ab5c9bd1ead7cfee72b5801eaf5176cc929e06387483a154` |
| `opto/utils/auto_retry.py` | `b7547b14979648ccc17dda76f0113b44b6802e63f0043160a67697eeb68b09bb` |
| `opto/features/recursive_opt/measurement.py` | `e667909e92d9fbe43baf088f4e61ed976755f9e9db9213b6ca0e8f6eb7b4a842` |

Verification used a read-only Python inspection with the existing frozen gate,
response and message-reconstruction functions and explicit execution blockers.
No regression suite was rerun because no implementation changed.

Operator update received after the initial audit record: incident 005 is now
saved with 28,458 EXP-18 run-file hashes and 108 completed responses. Its
SHA-256 was checked as
`7fc71870b14909c6c384f03ee856f5ee098ac68902106750a25299a1c09e58c0`.
The root operator reports unchanged pre/post inventories and passing main,
both pilot and source-archive gates under execution blockers. The remaining
fresh no-active-writer/hash check is owned by the operator immediately before
its explicit launch. No additional scientific blocker was identified.
