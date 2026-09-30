# EXP-17 local-evaluation interruption: bounded preparatory resume audit

Recorded 2026-09-12. This audit is scoped to the EXP-17 generation frontier after launch 006 stopped. It does not reopen validation, holdout or comparative efficacy. No model, credential, network, objective or generated-program execution occurred. Production Trace was replayed in memory using authenticated TRAIN cache rows, with execution and scientific writes blocked. No pipeline was launched or interrupted.

**Recommendation: preserve all 467 completed responses and resume the unchanged frozen implementation.** The interruption occurred after a completed model response and before its TRAIN panel was recorded. It is an infrastructure interruption, not an invalid generated candidate and not an incomplete provider request. Reuse completed caches; evaluate only still-missing work under the original rules. Existing replay guards must remain active and any actual identity divergence must stop resumption.

## Evidence and frontier

At check `1789235425379317927` ns, EXP-17 had 467 requests and 467 responses, with 481 started-attempt records, all matched to recorded outcomes. No request lacked its response and no started attempt was unmatched. The global generation barrier was absent. The last launch journal records return code 1 at `1789230877977586732` ns. Parent 224642 and child 224798 were absent; a targeted process-module scan also found no EXP-17 scientific writer at `1789235501064993844` ns. Root must retain its final no-writer check immediately before restart.

The retained `raw/17030/C/trace_attempt_001.json.gz` records `OSError: could not get source code`, status `error`, and two completed update callbacks. Its byte SHA-256 is `67c900ac9242d1afbe572f142a3a610a25ea3bbc1e6582463a77d9a6f5fbe197`. Hume's complementary traceback inspection places the failure in inspection of trusted worker source, before candidate execution. This audit does not establish why source inspection failed at that instant.

Only `17030/C/slot_02` among the 467 completed responses lacks a TRAIN receipt. Its attempt 1 is recorded completed, finish reason `stop`, source statically valid, response completion `1789230876265475928` ns. Static validity is not trajectory validity. Exact byte identities:

- Request: `37fd0b92f8fc43f3bfcfa7928a4d770836df594c20b093152ef06f01c9c43a16`.
- Response: `f8b8e706e77abaf25c88a8691f501a412d09299af2ec59b453b227d70657d0b6`.
- Attempt 1: `169afe8faf67cdb7ad06f7e53ceb08822979c21d2e7834e88d745e5b48f7a771`.
- Started 1: `47dab47d61831cf5e69223ebf4f37f1379ca2d738102e78b9e54e7d83f02e1de`.
- Exact source: `118281f0b2fa30fdbd45df99f52a2c72f8cb1e8665f2dea520a4dec8f534c4ae`.

Targeted cache verification at `1789235803843424520` ns constructed the frozen 48 TRAIN keys for each source, without executing the evaluator:

| Source position in 17030/C | Authenticated cache rows | Expected | TRAIN receipt |
| --- | ---: | ---: | --- |
| Seed | 48 | 48 | Present; exact source/response identities and row hashes verified |
| Slot 00 | 48 | 48 | Present; exact source/response identities and row hashes verified |
| Slot 01 | 48 | 48 | Present; exact source/response identities and row hashes verified |
| Slot 02 | 0 | 48 | Absent |

All 144 existing rows passed the frozen full cache-key and row-hash checks, including source, task, local seed, budget and evaluator identity. Their byte hashes were stable on recheck. Digest of the ordered-path/hash mapping: `f2e345e92293f4997d47d0db7825ac7ef5ab20335c703239e9de6330e2b54420`. No `.pending` files were present in the three inspected slot directories. A missing row is uncompleted persisted work, not an invented numerical or invalidity outcome. Unpersisted execution overhead must not be silently asserted to be zero.

## Actual production replay, without new scientific work

At `1789235907277906215` ns, the existing `trace_schedule.generate_recursive` and Control Plane reconstructed the prefix from the frozen seed using cache-only panel reads. Completed-response handling was replaced only in this audit process by exact request comparison and reading the preserved response. Existing evidence writes became identity checks; events and the diagnostic failed Trace result stayed in memory. Objective/candidate/client calls and output writes were blocked.

The resulting parent, feedback and full request identities matched all three preserved requests. Two callbacks completed. After returning the unchanged slot-02 response, the replay stopped at the first cache miss, as deliberately required by the read-only panel guard. It did not request a replacement response, create a receipt, or evaluate a missing trajectory. The next request was never reached.

Canonical JSON request digests (distinct from byte hashes):

- Slot 00: `54813c6dc42759efb596877f6bce788769259c4b9645608e8a77c4e14255cb9b`.
- Slot 01: `c384b9f701a8caaad88ef7c3654f328b2b50c70733ce05941f1baf51fdbac456`.
- Slot 02: `52332c6cf55bde42e21d8f9b172050fe31f6e286241ef0f5121b8edfc7f71b69`.

The slot-02 parent remained `559f3343d70f8e20bbe6ff648f76c2cb04aff14ae8e7f80995eb397d6b91f140`. Twelve existing generation/receipt records compared exactly and remained byte-stable; their path/hash mapping digest is `5207d3bb6e6bd549b9bc324fa49b8bd44b171e0b37cf55c50a4a854abf3b53f1`. No real replay guard mismatch occurred.

## Frozen identity, provenance and limits

`driver.require_confirmation()` passed with evaluator, client and write calls blocked. Original freeze `e520ba38428cf30cecefb76787f73454f4b985143c9ae46a7b9f48e36167c980` still authenticates all 103 scientific source files and the exact 46-pair registered configuration. Relevant byte hashes:

- `exp17/study.py`: `f16cfd1eac80f1aa27584e6f654ff5c43abe46aeb57265fd67496bc4aeb48b5b`.
- `exp17/evaluation_cache.py`: `3e1485c08d06f324111239bc2a71d7c7dae3d974da869ccaf66855abbd63ca9d`.
- `investigation16/trace_schedule.py`: `a1ad02d77d3b5b665f9e64b6e1894b222350a016211aaa678b7d4757c33b5732`.

Root separately reports incident 011 SHA `fa49d0bd46f6a281e2d3a21cca17a990c1eef905b319e7f2f1c6299302895e9a`: 169,334 EXP-17 and 166,406 EXP-18 file hashes unchanged, original main/engineering/archive gates passing, and current trusted-function source inspection passing. These full-inventory findings are attributed to root, not redundantly repeated here.

Git HEAD is now `7e701b40485b9880ccfbb64c1faaabb22401c294` (checkpoint timestamp 2026-09-12 16:34:22 UTC). The diff from baseline `13ebda2242e1c18022591737b113030ca2ce2da2` changes only `artifacts/RESEARCH_LOG.md` and `artifacts/recursive_opt_assessment.md`. Preserve that checkpoint and all user work. The commit's proximity to the interruption and the observed source mtimes do not establish causation. Current source identity does not prove every transient filesystem state during the earlier error.

The frozen cache records a row only after its evaluator call returns; an uncaught trusted infrastructure exception does not create a typed invalid-candidate row. The completed slot remains consumed. On normal resume, Trace reconstructs its incumbent from the completed cache prefix, authenticates the already-recorded slot-02 request, reuses its exact response, and completes its missing TRAIN allocation. Only after that may it advance to the next uncompleted response slot. Preserve the original failed Trace attempt and all earlier transport uncertainty/usage; this event provides no basis to reclassify or replace any prior response or scientific outcome. No changed scientific semantics or new experiment ID is indicated by the verified frontier; a future actual semantic mismatch must be investigated rather than bypassed.
