# EXP-18 sealed generation integrity audit 001

**PASS within the generation-only scope: no integrity anomaly detected.**
Checks ran from 2026-09-12T12:16:59.687998+00:00 to
2026-09-12T12:17:53.349119+00:00.
The generation barrier completed at
2026-09-12T12:10:50.264585+00:00
(`1789215050264585317` ns). This is a generation-stage integrity result, not scientific
completion of EXP-18 or evidence of optimizer superiority.

## Coverage and preserved failures

Exactly six outer seeds `[18011, 18023, 18037, 18041, 18053, 18067]` × four arms
`[L, M, P, PM]` × 16 slots are represented: **384 requests and completed responses**,
with unique slot IDs and unique provider response IDs. No missing, duplicate or
extra request/response path was found. All source hashes and static source statuses
were authenticated by reparsing exact preserved response content through the frozen
response verifier. No generated program was executed or edited.

There are **399 matched started/attempt records: 384 completed attempts and 15
transport failures**. Every slot has contiguous attempt numbering, exactly one
completed attempt, and a completed response matching that final attempt's provider
ID. Earlier failures remain preserved with unknown-remote-completion/duplicate-billing
flags. Nine are recorded transient=true and six transient=false. All completed
responses consume their slots, including poor or invalid output; no replacement
or omitted invalid slot was found.

Static screening retained 361 valid sources, 17 missing sources, four syntax errors
and two protocol violations. Parsing extracted source from 367 responses; 17 were
unparsable. Finish reasons are 368 `stop`, 15 `length`, and one `error`. The completed
response with finish_reason=error remains a completed scientific proposal slot,
separate from the 15 transport failures.

| Arm | Responses | Statically valid source | Static rejection | All 48 TRAIN trajectories valid | At least one invalid TRAIN trajectory |
| --- | ---: | ---: | ---: | ---: | ---: |
| L | 96 | 85 | 11 | 82 | 14 |
| M | 96 | 91 | 5 | 91 | 5 |
| P | 96 | 91 | 5 | 81 | 15 |
| PM | 96 | 94 | 2 | 82 | 14 |
| Total | 384 | 361 | 23 | 336 | 48 |

These are typed validity counts, not comparisons of optimization performance.
Among the 48 candidates not valid on every TRAIN trajectory, 23 have statically
rejected source and 25 have statically accepted source. The audit does not diagnose
those execution failures or infer validation eligibility. Validation was not read.
All 408 TRAIN receipts (384 proposals plus 24 seed pools) retain 48 allocated and
observed trajectory records and 48 row hashes each: 19,584 logical trajectory
allocations. Status counts agree with their recorded valid-trajectory counts.
This is not an estimate of actual objective calls or physical cache work.

## Frozen requests and causal prompt data

The exact main gate authenticated the original 113 frozen source files and main
configuration against freeze
`ce88c4b2d9530456e63b497f7bbde7608fbf3d8aca441c13c086d77b6beeea8a`.
Every request matches its registered model, namespace, outer seed, arm, slot and
freeze digest. Actual generation settings match the frozen request derivation:
OpenRouter `deepseek/deepseek-v4-flash-0731`, temperature 0.6, top_p 1.0,
max_tokens 32000, native extra_body.reasoning.effort=low, timeout 300 and
num_retries 0; request seeds follow the frozen stable derivation. Client/routing
implementation remained covered by the source freeze. This audit did not make
provider calls or inspect new routing/billing receipts.

All **384 contexts and complete prompt messages reconstruct exactly** from the
frozen invariant instruction, the actual prior-source parent, compact TRAIN
feedback and, where registered, prior-attempt memory. Each parent is the earliest
matching source occurrence among the seed and strictly preceding same-arm slots.
Its source/receipt/feedback hashes match. All response completions precede the
barrier; their TRAIN observations occur after response completion and before the
barrier. The original context cutoffs precede the first started attempt and remain
fixed across retries/resumes.

For M and PM, all **192 memory contexts** reconstruct exactly under the frozen
seven-source / 65,536-character policy. They include **1,440 strict-prior attempt
references**, including **86 appearances of statically invalid prior sources**.
All prior response and TRAIN-availability timestamps precede the original context
cutoff, and every required prior slot is retained exactly once. Source omissions,
limits, hashes and compact text match the preserved context. L and P correctly
have no prior-attempt memory.

Prompt reconstruction used only the permitted closed TRAIN projections, recorded
source artifacts and frozen generic invariant. No validation/audit result fields
or values entered the reconstructed prompt data. Generic instruction wording or
model-authored source text was not treated as evidence of leaked results by a
keyword scan; the check is exact data provenance. No objective values were printed,
ranked or compared between candidates/arms for this audit.

## Pareto chronology and identity, without ranking efficacy

All **204 P/PM decisions** are preserved: 192 used decisions plus 12 explicitly
unused terminal decisions. Each archive contains the seed and every completed
same-arm slot strictly before its decision slot. Archive hashes match the sealed
TRAIN receipt vectors, treated as opaque data for hashing. Invalid-source/receipt
positions remain represented. Used archive receipts precede the corresponding
request's original cutoff; terminal archive receipts precede the generation seal.

The registered frontier indices are nonempty, unique, within the prior archive and
linked to TRAIN-valid source identities. The frozen deterministic uniform draw
reproduces the recorded chosen member of that frontier. Used decisions match the
actual propagated parent hash in the request; all decisions identify the production
`PrioritySearch.explore` hook and correct used/terminal status.

**Boundary:** dominance ranks, scalar-best values and front membership were not
recomputed by comparing performance values. This review certifies recorded archive
provenance, timing, deterministic draw and parent identity. It relies on the frozen,
previously tested ranking implementation for mathematical nondominance; it does
not add an efficacy analysis or a new test of that ranking algorithm.

## Stable evidence and explicit reading boundary

The generation barrier names exactly **2,244 mandatory scientific files** and their
canonical hashes. Every hash passed at the start and end of the check. Including
that barrier, the freeze/identity files, and 798 started/attempt files, **3,045
scientific inputs remained byte-identical** throughout the audit. New operational
journals or provider receipts were intentionally outside this stability inventory.

A read whitelist restricted project record access to generation evidence, freeze
files and the registered started/attempt paths. No cache, validation-result,
selection-result, holdout/audit-result, provider-metadata or runtime record was read.
In particular, `verify_chronology` and analysis readers were not invoked because
they can open later-stage evidence once it appears. TRAIN receipts and row hashes
were authenticated against the seal, without opening trajectory caches or
recomputing metrics. Thus this review does not certify cache-to-row numeric
correctness or later split-access timing; those remain separate checks.

Objective/candidate evaluation, live-client/key loading, transport execution,
receipt collection, production persistence, metric aggregation and Pareto ranking
were explicitly blocked during the audit. No heavy test, scientific call, production
mutation, frozen edit or commit was performed. No regression suite was rerun because
no implementation changed. Only this report was written.

| Integrity anchor | SHA-256 / canonical digest |
| --- | --- |
| [Generation barrier bytes](../../../../EXP18/results/run/generation_frozen.json) | `7a8186436ea6933ae36b78025b3aad157db9dee7d7221d3704441e9ffedb9d53` |
| Generation barrier canonical object | `3a4adf50973e749681c305307ac068d3c50d7304784a6dd95370991d02f09e7d` |
| 2,244 sealed files plus barrier/freeze/identity byte-hash mapping | `ab1e74c8ff23e07c33a1bf6d5a3899b51415ff2bfdec00693644396809a68842` |
| 798 started/attempt byte-hash mapping | `8dcdfc50d0e204bd43bb93d8645d5f9cdc9ac269975314edb6383d42a88ab8c0` |

Collection digests use `benchmark.digest` over relative logical paths mapped to
SHA-256 hashes of their exact physical JSON/gzip bytes. No anomaly requiring a
protocol or implementation change was found in the examined scope.
