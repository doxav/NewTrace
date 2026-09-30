# EXP18 resource and invalidity review

Offline review completed 2026-09-12, after all 384 responses, all 24 selections and the audit were preserved. **PASS: independent accounting agrees with the frozen analysis on 226 checked scalar or structured fields.** No candidate, objective, model, metadata request or test was executed, no EXP17 performance was inspected, and no run artifact or frozen implementation was changed. Numerical objective verification is a separate pipeline step managed by the study owner; this report does not claim its completion.

## Completed proposals, tokens and known cost

The frozen allocation was six outer seeds × four arms × 16 completed responses = **384 slots**, using `deepseek/deepseek-v4-flash-0731`, a 32,000-token completion limit, the registered low-reasoning configuration and one live generation at a time. There are 384 unique response IDs and 384 unique slot IDs. All **399** started records match retained attempts: 384 successful attempts and **15 transport failures**, with no unfinished slot or unmatched start. The same model identifier appears in every response.

| Arm | Completed slots | Transport attempts | Failed attempts | Matching receipts |
|---|---:|---:|---:|---:|
| L | 96 | 101 | 5 | 96/96 |
| M | 96 | 103 | 7 | 96/96 |
| P | 96 | 98 | 2 | 96/96 |
| PM | 96 | 97 | 1 | 96/96 |
| Total | 384 | 399 | 15 | 384/384 |

All five usage fields are reported by every completed response. No receipt fallback or zero substitution was needed. Known USD amounts below follow the frozen response-usage authority.

| Arm | Prompt tokens | Completion tokens | Reasoning component | Total tokens | Known response cost USD |
|---|---:|---:|---:|---:|---:|
| L | 190,561 | 914,016 | 763,845 | 1,104,577 | 0.237342534028 |
| M | 1,072,759 | 778,771 | 615,187 | 1,851,530 | 0.261415000300 |
| P | 164,760 | 766,259 | 650,604 | 931,019 | 0.165487488904 |
| PM | 1,001,071 | 711,366 | 540,909 | 1,712,437 | 0.241753521504 |
| Total | 2,429,151 | 3,170,412 | 2,570,545 | 5,599,563 | 0.905998544736 |

Reasoning tokens are included in completion tokens and must not be added again. Prompt and completion counts match the receipts' `native_tokens_prompt` and `native_tokens_completion` exactly for all 384 responses. The receipts' non-native `tokens_prompt`/`tokens_completion` fields differ for 383 responses and sum to 1,940,773/3,059,726; they must not silently replace the frozen counters. The reasoning receipts agree exactly. Receipt `total_cost` sums to USD 0.905998501, differing from response cost by approximately USD 0.000000043736 (maximum per-response difference below USD 0.000000001). Both original representations remain preserved.

All 15 failed attempts have missing usage and retain `possible_remote_completion_or_duplicate_billing=true`. Thus USD 0.905998544736 is **known completed-response cost**, not verified total spend. These flags are uncertainty, not proof that 15 additional generations completed or were billed. Exact failed-attempt paths, typed errors, transient classifications and durations are in [the data companion](resource_review_data.json.gz); repeated incident snapshots were not counted as new failures. There are 384 matching provider receipts, 22 recorded route names, and **zero saved metadata-GET failures**. Per-arm routing counts are retained in the companion. No metadata was recollected for this review.

## Invalidity and deployment fallback

Every generated candidate remains represented in its allocated TRAIN and VALIDATION schedule. In total, **23/384 sources (5.99%) are statically invalid**: 17 missing sources, four syntax errors and two protocol violations. Another 25 statically valid candidates become ineligible during execution. Overall **336/384 generated candidates are eligible; 48/384 (12.50%) are ineligible**.

| Arm | Source invalid | Eligible generated | Ineligible generated | Invalid generated trajectories |
|---|---:|---:|---:|---:|
| L | 11/96 (11.46%) | 82/96 | 14/96 (14.58%) | 972/6,912 (14.06%) |
| M | 5/96 (5.21%) | 91/96 | 5/96 (5.21%) | 360/6,912 (5.21%) |
| P | 5/96 (5.21%) | 81/96 | 15/96 (15.62%) | 952/6,912 (13.77%) |
| PM | 2/96 (2.08%) | 82/96 | 14/96 (14.58%) | 648/6,912 (9.38%) |

The trajectory denominator is the allocated 96 candidates × (48 TRAIN + 24 VALIDATION trajectories) per arm; it includes logical reuse of cached rows. It excludes the extra unchanged-seed selection-pool member. Typed final statuses for those invalid generated trajectories are:

| Arm | Missing source | Syntax error | Protocol violation | Exception | Nondeterministic | Timeout |
|---|---:|---:|---:|---:|---:|---:|
| L | 576 | 144 | 72 | 180 | 0 | 0 |
| M | 216 | 72 | 72 | 0 | 0 | 0 |
| P | 288 | 72 | 0 | 520 | 72 | 0 |
| PM | 144 | 0 | 0 | 504 | 0 | 0 |

There are 2,932 logical invalid generated trajectories and **2,068 distinct physical invalid cache rows**. Of the latter, **1,204 retain partial observations totaling 6,885 actual objective calls**. These calls are included in physical resources. Invalid trajectories have `metrics=null`; their partial work was neither dropped nor given an invented regret score. No final `timeout` status appears, but the frozen `nondeterministic` label can also contain a failed second replay, so zero timeout labels do not prove every replay completed successfully. Proposal-status counters include typed source rejection without a child launch; they are distinct from model slots and subprocess counts.

Each arm selected a generated replacement for all six outer seeds: zero seed-index selections, zero selections of seed-identical source, and zero searches without an eligible replacement. On the protected audit, **all 864 trajectories** completed validly: 144 per deployment arm A0/B2/L/M/P/PM. Candidate-invalid trajectories and fallback events are **0/864**, including **0/576** for the four generative arms. All seeds and failed searches/candidates remain in the analysis. Deployment success does not erase TRAIN/VALIDATION invalidity.

## Allocated and persisted physical evaluation work

The registered logical schedule totals **30,240 trajectories and 967,680 objective allocations**. Each generative arm receives 7,344 TRAIN/VALIDATION trajectories including the unchanged seed, plus 144 audit trajectories: 7,488 trajectories and 239,616 objective allocations. A0 and B2 each add 144 audit trajectories and 4,608 allocations. Shared sources reuse the same complete frozen cache key; physical work is attributed globally and by split, not assigned to whichever arm happened to access the cache first.

| Accounting | Split | Trajectories | Objective allocation | Recorded objective calls | Unused allocation | Recorded subprocesses |
|---|---|---:|---:|---:|---:|---:|
| logical | train | 19,584 | 626,688 | 568,739 | 57,949 | 1,138,375 |
| logical | validation | 9,792 | 313,344 | 284,354 | 28,990 | 569,159 |
| logical | audit | 864 | 27,648 | 27,648 | 0 | 55,296 |
| physical | train | 17,424 | 557,568 | 518,051 | 39,517 | 1,036,999 |
| physical | validation | 8,712 | 278,784 | 259,010 | 19,774 | 518,471 |
| physical | audit | 864 | 27,648 | 27,648 | 0 | 55,296 |

Logical rows total 880,741 recorded objective values, 86,939 unused allocations and 1,762,830 recorded subprocess executions; reused values are counted once per logical allocation in that view. The **distinct physical cache totals are 27,000 trajectories, 864,000 allocations, 804,709 objective calls, 59,291 unused allocations and 1,610,766 subprocess executions**. Their identity set exactly equals the union of all allocated panels. No invalid early-termination allocation was recycled into an additional proposal.

The event journal has **140,832 cache accesses = 113,832 hits + 27,000 misses**. Every physical row has its miss event; no repeated miss or event referencing a missing cache row was found. There are 384 `trace_update` and 28 `trace_update_replay` events. Replay and cache-hit events are not additional completed model responses or physical cache evaluations.

Normalization preparation has 48 frozen tasks × 128 reference points = **6,144 unique design evaluations**, accounted separately from search objective calls. Repeated physical reconstruction across processes, retries or caches was not exhaustively instrumented. Later numerical-integrity recomputation is also separate work; it must not be added to the optimizer's scientific budget or silently omitted from a claim about all host computation.

**Interrupted physical work remains unknown.** Incident 011 interrupted outer 18067/M/slot 11 during VALIDATION before any of that panel's 24 rows were persisted. Its unchanged resume reused every completed result and filled the missing keys. Lost partial observations and child counts were not journaled. For this EXP18 incident alone, the conservative one-attempt ceiling is **768 objective calls and 1,536 subprocesses**, not measured additional expenditure. The actual extra counts and seconds remain unknown; do not impute zero or add the ceilings as observations. The persisted physical totals above must carry this qualification. See [the incident report](operational_status_checks/infrastructure_source_read_001.md).

## Elapsed time, queueing and suspension

The frozen field `attempt_wall_s_reported` sums **monotonic client-attempt duration**, including waiting for the shared generation lock. It is neither civil elapsed time nor provider compute time. Attempt sums exclude retry backoff between attempts; whole-slot clocks retain backoff, persistence and explicit-resume gaps. Generation phase clocks also include TRAIN evaluation and production processing.

| Arm | Summed client-attempt seconds | Included failed-attempt seconds | Reported provider-generation seconds |
|---|---:|---:|---:|
| L | 36728.812 | 4830.645 | 19395.360 |
| M | 33263.447 | 3123.335 | 15983.294 |
| P | 30328.068 | 777.583 | 15682.884 |
| PM | 26094.146 | 363.309 | 13448.923 |
| Total | 126414.473 | 9094.873 | 64510.461 |

Provider `generation_time` is converted from its documented millisecond field as specified in [RESOURCE_METHOD.md](RESOURCE_METHOD.md#L128), with 384/384 coverage. The raw duration and latency summaries remain in the data companion. Subtracting provider time from client time does not isolate queueing: network/client overhead and other unmeasured intervals remain. The forwarded 300-second timeout is not a strict total slot deadline, as documented in [the existing timeout audit](operational_status_checks/client_timeout_audit_01.md).

Despite their `_ns` field names, values under `elapsed_s` are seconds:

| Phase | Civil seconds | Monotonic seconds | Boottime seconds | Estimated suspension seconds |
|---|---:|---:|---:|---:|
| audit | 564.300 | 564.300 | 564.300 | 0.000 |
| generation | 168631.606 | 143056.137 | 168628.246 | 25572.109 |
| selection | 22746.207 | 22746.207 | 22746.207 | 0.000 |

No phase reports a clock reset. Generation includes approximately **25,572.109 seconds (7.103 hours)** of boottime-minus-monotonic suspension. Suspension records 001–004 are supporting observations of shared host pauses; observations from both studies must not be added as separate pauses or added again to these phase clocks. Selection includes the documented local source-read interruption and resume gap. The civil interval from first generation start to audit completion is 193168.099 seconds (53.658 hours); analysis and numerical verification occur later.

Summed physical trajectory `execution_s` is 107,891.467 seconds. It is elapsed per-trajectory time summed across concurrent workers, not CPU time or experiment wall time; the frozen timer ends before metric/normalization calculation and cache/event persistence. Phase, slot, attempt, provider and trajectory durations overlap and must not be summed into a purported end-to-end total.

## Verification and provenance

This independent read-only review checked response/slot uniqueness; contiguous successful/failed attempts and matching starts; receipt identities and token provenance; every pool's eligibility counts; full logical and physical allocation/status sums; all audit fallback flags; partial invalid observations; exact cache-identity coverage; event hits/misses; and phase clocks. The resulting resource/invalidity fields agree with `analysis_results.json.gz` on **226 comparisons**. Floating sums were compared at relative tolerance 1e-12 and absolute tolerance 1e-9; counts and structured identities were exact.

The input snapshots were hashed before and after the read-only work without changes. Their file-count inventories describe provenance, not scientific allocations. The companion records canonical SHA-256 digests of the sorted path-to-file-hash maps, named evidence hashes, the complete resource tables, all failed attempts, routing/coverage details and the analysis comparison. No resource-accounting mismatch was found. Known limitations are remote billing uncertainty, unpersisted interrupted work, incompletely instrumented normalization reconstruction, and overlapping time measures; none is silently converted to zero.

Data companion exact gzip SHA-256: `61a646011b2e35dac9ca81cd754d18e3008b9d120834d6bd677130a3c4d83fea`.

Paths below are relative to this EXP18 directory; hashes are exact stored bytes, including gzip where named.

| Evidence | SHA-256 |
|---|---|
| `run/freeze.json` | `9abb465cda1ea55c9cce7e5e784621a9354033c1f6f4c82960f2b3ce917354be` |
| `run/generation_frozen.json` | `7a8186436ea6933ae36b78025b3aad157db9dee7d7221d3704441e9ffedb9d53` |
| `run/selections_frozen.json` | `ded3281b54b46055dca32088bf00216c8c8e1a17a4363d82b73ff762701bb038` |
| `run/audit_results.json.gz` | `dff97a377f5db8cfda1eb22bb761f4be91c79cc88a4d1eed44d94a0797b964eb` |
| `run/analysis_results.json.gz` | `1654ab1ff03bf1d2713bb3ec6bfe870f603cc8ae6e77b32579a074d20f94925a` |
| `operational_status_checks/infrastructure_source_read_001.md` | `e5f0dd576754ad65ffb25a8dbf40475b568b028c0d1f344f8b0a85e8414b4dd4` |
| `../exp17/analysis.py` | `204e43f8f62de913538804d506ebcc1a040102f84e85ddee8b8ae30a3e08bb04` |
| `../investigation16/production/analysis.py` | `a5b2c759b7ce2c491c76820acc7ed1c12ef0138a538cc658dca6bead3097656e` |
| `../evidence.py` | `23320a4a44598a363d58623f95a88762fb6c6a4fef4ed09afd510461719fe36a` |

The freeze file-byte hash above is distinct from canonical manifest digest `ce88c4b2d9530456e63b497f7bbde7608fbf3d8aca441c13c086d77b6beeea8a`. Source authorities: shared analysis resource accounting (`../exp17/analysis.py:233–335`), usage fallback/provenance (`../investigation16/production/analysis.py:255–300`), validity descriptions (`../evidence.py:52–116`), and evaluator/cache persistence (`../benchmark.py:232–305`, `../exp17/evaluation_cache.py:157–205`).
