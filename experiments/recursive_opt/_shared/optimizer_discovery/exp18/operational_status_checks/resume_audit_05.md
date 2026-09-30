# Differential resume audit 05: recorded DNS interruptions

Read-only check: 2026-09-11T17:07:32.853961+00:00 to
2026-09-11T17:07:34.630350+00:00.

**PASS for the existing explicit-resume interpretation**, subject to the root
operator's current inventory, engineering/archive, connectivity and no-writer
checks. No new blocker was found compared with [audit 03](resume_audit_03.md).
This review covers only the two pending slots and original main freeze gates;
it does not repeat the full raw-evidence census or inspect efficacy. The root
operator reports 165 completed EXP-17 and 154 completed EXP-18 responses and both
fifth launches stopped during generation.

| Study / pending slot | Recorded failures | Matched starts / attempts | Next attempt |
| --- | --- | ---: | ---: |
| EXP-17 `17011/C/slot_05` | 1: timeout, transient=true; 2: DNS, transient=false | 2 / 2 | 3 |
| EXP-18 `18037/PM/slot_10` | 1: DNS, transient=false | 1 / 1 | 2 |

Both slots have no ordinary/compressed response, provider receipt or metadata
lookup directory. All three failures are `TransportRetryError` records with
`possible_remote_completion_or_duplicate_billing=true`, no usable provider ID
fields or recognizable generation IDs in sanitized error/log text, and no reported
usage/token/cost fields. Pure outer-classifier replay reproduced the saved flags.
The DNS/timeout distinction matches the already documented narrow classifier;
keep the flags, model, settings and classifiers unchanged.

Both original main gates authenticated the complete frozen source/configuration
identities under execution blockers: EXP-17, 103 files, freeze
`e520ba38428cf30cecefb76787f73454f4b985143c9ae46a7b9f48e36167c980`;
EXP-18, 113 files, freeze
`ce88c4b2d9530456e63b497f7bbde7608fbf3d8aca441c13c086d77b6beeea8a`.
The pending requests match their registered identities, model/settings, invariant
prompt, prior-source parent and propagated feedback. Both sets of saved messages
reconstructed exactly.

For EXP-18 PM, `MechanismStudy.messages` reused original context cutoff
`1789141160467229768` ns and all ten completed prior attempts. The 36,448-character
memory and original context matched exactly. Prior response/TRAIN receipt and seed
receipt timestamps precede that cutoff. The existing `select_pareto_parent` path
reconstructed the saved slot-10 decision from authenticated historical TRAIN
vectors and confirmed the same actual parent hash:
`9fb3f6fe655cde9e27fbde2fce9307564f8f0b34f7a5ffb5405e0c7388210653`.
This was an identity check, not a comparison of experimental outcomes.

The cache was read-only and persistence was replaced by an exact-existing-bytes
comparator. Missing evidence would have raised rather than trigger new work.
Seven checked files per study, including the PM parent decision, remained
byte-identical. No objective, candidate, model, network or metadata call, key load,
process intervention, frozen edit or commit occurred.

The unchanged `generation.complete_slot` guard authenticates the original request,
reuses an existing response and blocks unmatched starts. Here all pending starts
have recorded failure returns, so an explicit invocation may continue at indices
3 and 2 under the same bounded retry policy. It must preserve every previous
attempt and completed response. The three failed requests may have completed
remotely or incurred charges; absent IDs do not establish nonexecution or zero
billing. Retain unknown usage and any available reliable reconciliation evidence.
These interruptions alone require neither scientific invalidation nor a new ID.

## Integrity anchors

| Evidence | SHA-256 |
| --- | --- |
| EXP-17 pending request | `f606c7fa481452f3303b0c6a9632a36f8bfb36f9a696570519e3c45626f061d8` |
| EXP-17 propagated feedback | `cb3e6ed2acaab7f256a2560fd3a2d54ce5ebf07312255772a5875a85f5df61e7` |
| EXP-17 timeout attempt 1 | `7117bfed3b37858d932953aad2ff67360d81c1c2bc7f464014512fe2206032e2` |
| EXP-17 DNS attempt 2 | `eb65ebc7d11474ca202d6a0f84a6a1755f220811eafac3c03f9969e8c30aaa6d` |
| EXP-18 pending request | `5e2c5dd4c88634186285186f6ddce38409033550169e55eb2c26ae7e8fb43f75` |
| EXP-18 propagated feedback | `4c3a5f140885c6ffb33c54a03c0086abd1e63c08044e432ff890454badb9d99d` |
| EXP-18 context bytes | `e16182191acb8ad7890404af4b94205625464081785bcd59a1bbd673593e0bf7` |
| EXP-18 memory text | `70ea62fc909ccbae2721fca5da86af2b307b8568727d98804855fec5a2179f61` |
| EXP-18 parent decision | `507d1de64f86347352d31e6701ef1922714dbfd4876965ead2cb4336cf4c53ff` |
| EXP-18 DNS attempt 1 | `19f7292a504bc1b6fdb239045084fb45e01788dfa0e4dfb023a5a5e09c4a2781` |

Relevant unchanged sources: `generation.py`
`6a5a66b9b12fd4f4217668a5803bd92ad660a5e35d78a42cd45ec7882dd6d24c`;
`exp18/study.py`
`22c653204145fe77a2a8619a86182edb0ce1d8f83b886a9b730f06e2fc5670cf`;
`exp18/pareto_selection.py`
`bb76350204c75a4d96aa4ef46082f347874926ee5b656f00c698d45c270de7b3`.
No regression suite was rerun for this read-only differential audit.

The root operator subsequently supplied its completed checks: original main and
engineering gates, source archives and whole-inventory before/after comparisons
passed under execution/key-load blockers. Credential-free DNS returned four records
and verified-TLS public model-list HTTP 200 included the exact model at
2026-09-11 17:07:21.076126 UTC. Both inventories have no unmatched starts and
preserve original evidence. These checks are attributed to the operator and were
not repeated by this differential review.

| Operator snapshot | Files | Completed responses | Historical transport failures | Reported SHA-256 |
| --- | ---: | ---: | ---: | --- |
| [incident_008.json](incident_008.json) | 68,454 | 165 | 13 | `21611db4e116cef98f68f0d8667cbddea21e90ac323114ef5895247cd5b3e9e1` |
| [incident_009.json](incident_009.json) | 53,699 | 154 | 14 | `146bafff26050e6d622f366ed9495286e725dafd0be26f782f211f86c86367ca` |

The final fresh no-writer/hash check remains the operator's responsibility
immediately before its explicit sixth launch. No additional scientific blocker
was identified by the independent pending-slot review.
