# Local source-read interruption: EXP17 launch 006 and EXP18 launch 007

Read-only diagnostic prepared on 2026-09-12 after both processes exited. This report does not change the frozen protocols, execute candidate code or objectives, call a model, or authorize a scientific replacement. No comparative performance is analyzed here. One incidental seed-score line appeared during initial diagnostic log extraction; it was not used in any inference or decision.

## Finding and boundaries

EXP18 selection stopped because the host could not extract trusted worker source with `inspect.getsource`: `OSError: could not get source code`. Its traceback reaches `optimizer_program.py:152–154`, before the current worker files are written and before `subprocess.Popen` at line 165. This is an infrastructure exception, not a candidate invalidity, candidate replay mismatch, or a serialization exception. EXP17's preserved Trace result contains the same underlying `OSError`; its outer exception, `production schedule did not consume the response allocation`, is a consequence of the interrupted callback, not evidence that the model response budget was exhausted incorrectly.

The current source-read operation succeeds for `_finite_number`, `_point_status`, and `_worker` without executing any candidate. Both full source freezes currently match. These facts establish present readiness of the unchanged implementation; they do not identify who or what caused the earlier read failure, or prove the historical contents of a file at every worker launch.

## Preserved stopping state

| Item | EXP17 | EXP18 |
|---|---|---|
| Launch | `launch_006`, generate returned 1 | `launch_007`, generate and receipts returned 0; select returned 1 |
| Launch finished UTC | 2026-09-12 16:34:37.977587 | 2026-09-12 16:34:42.180203 |
| Finished Unix ns | 1789230877977586732 | 1789230882180203536 |
| Completed model responses | 467 | 384; full response grid complete |
| First incomplete local panel | outer 17030, arm C, slot 02, TRAIN | outer 18067, arm M, slot 11, VALIDATION |
| Source SHA-256 | `118281f0b2fa30fdbd45df99f52a2c72f8cb1e8665f2dea520a4dec8f534c4ae` | `fbb88cadfeb6f902a55d72213c1dc44ce1b7e16b2abbd1c8e0885b5140c37ef6` |
| Cache rows for that panel | 0/48; TRAIN receipt absent | 0/24 |
| Immediately preceding panels | seed and slots 00/01 each have 48/48 authenticated TRAIN rows and matching receipts | seed and slots 00–10 each have 24/24 VALIDATION rows |
| Global barriers | generation, selections and audit absent | generation frozen; 21 per-arm selections and pools persisted; global selections and audit absent |

EXP17 has 467 matching requests/responses and 481 paired transport-start/attempt records, with no pending request. Its failed Trace attempt records two completed callbacks after three persisted responses in the current arm. The callback writes the response before constructing the TRAIN receipt (`exp17/study.py:412–420`); an exception in that receipt prevents the callback from being counted complete. Thus slot 02 must reuse its existing response on resume. The remaining 269 response slots are the original unfinished allocation, not replacements.

EXP18's remaining per-arm selections are M, P and PM for outer 18067. Within M, panels after slot 11 were not reached by the sequential selection loop. Their missing rows are unfinished original work and must not be counted as incident retries merely because they are absent. Existing 21 selections are skipped unchanged by `select_all` (`exp17/study.py:607–650`).

Parent and child PIDs 224642, 224798, 406340 and 3558407 were absent at the diagnostic process check. A fresh lease/process check remains the launcher owner's responsibility before resumption.

## What was durable, what can repeat, and how to account for it

`exp17/evaluation_cache.py:157–205` serializes each complete evaluation only after `B.evaluate` returns. It authenticates an existing row against the complete frozen cache key, row hash and row identity. Per-key locks prevent duplicate execution within that process. Completed valid and typed-invalid rows both remain authoritative and must be reused; none may be discarded in pursuit of a different outcome. Other running futures can complete and persist while a failing future causes the panel to unwind, so the final on-disk inventory, not the first exception time alone, determines what is reusable.

The evaluator keeps trajectory history, proposal attempts and execution counts in memory (`benchmark.py:232–305`). It records an attempt after `propose_point` returns and evaluates the objective only after a valid proposal. Ordinary invalid candidates return typed rows with partial observations. The uncaught host source-read exception instead bypasses the evaluator's return and therefore bypasses row persistence. There is no per-observation or per-trajectory-start journal. Earlier valid points in an interrupted trajectory may already have consumed objective calls and child processes, although the failing `_execute_once` itself did not launch its current child. A failure during the second replay may also follow one successful child for the current proposal.

Consequently, restarting a missing cache key starts the same fixed trajectory from empty history and can repeat its lost prefix. This is a technical retry of an unfinished allocation. It does not create a new candidate, new source, new local seed, larger trajectory budget, or choice between persisted outcomes. The completed response/source and frozen logical allocation remain unchanged. There is no numeric result to replace for those missing keys.

For this single interruption, the potentially affected scope is bounded by 48 TRAIN keys plus 24 VALIDATION keys. Their actual number of started trajectories, lost objective calls, child launches and execution seconds is **unknown**, not zero. Under the inspected code path, each key is submitted at most once to this failed panel invocation, with B=32 and at most two child launches per proposal. A deliberately conservative ceiling is therefore:

| Potential unpersisted work | EXP17 | EXP18 | Combined ceiling |
|---|---:|---:|---:|
| Trajectory attempts | 48 | 24 | 72 |
| Objective calls | 1,536 | 768 | 2,304 |
| Candidate subprocess launches | 3,072 | 1,536 | 4,608 |

These ceilings are not measurements, not a budget increase, and not additional observations in the analysis. They assume only the identified single panel attempt per key; a broader undiscovered incident would require a new accounting inventory. The specific failing trajectory necessarily stopped before a complete B-point result, but the conservative bound avoids inferring unlogged per-worker histories. Runtime has no analogous precise measured value for the lost trajectories.

Final reporting must state persisted physical work **plus unknown interrupted work**, retaining the incident and optional ceilings separately. Do not add the ceiling as if it were realized cost, silently report the persisted subtotal as exhaustive physical expenditure, count all later unstarted panels as lost work, or recycle unused allocations into extra proposals. Logical scientific allocations remain the frozen grid. This local exception occurred after completed model responses and creates no new uncertain remote generation or duplicate billing; earlier provider incidents retain their own separate accounting.

## Source provenance and limitations

The two process stops were 4.202616804 seconds apart. The current `optimizer_program.py` mtime and ctime are both 1789230901505808770 ns (16:35:01.505809 UTC); those for `benchmark.py` are both 1789230901344808540 ns (16:35:01.344809 UTC), after both stops. These timestamps are consistent with a filesystem write or metadata-affecting operation; they do not identify its author or demonstrate causation. Current hashes match the frozen bytes:

| Current trusted source | SHA-256 |
|---|---|
| `opto/features/recursive_opt/optimizer_program.py` | `4d8bc35541943d2430536adb46470671f3433f1749eb7bae60af516610c1361c` |
| `artifacts/optimizer_discovery/benchmark.py` | `77d1dbdf56dc88926a42716e311b6f4686bded73bec5ed320f8b78c58d0419cf` |
| `artifacts/optimizer_discovery/exp17/evaluation_cache.py` | `3e1485c08d06f324111239bc2a71d7c7dae3d974da869ccaf66855abbd63ca9d` |
| `artifacts/optimizer_discovery/exp17/study.py` | `f16cfd1eac80f1aa27584e6f654ff5c43abe46aeb57265fd67496bc4aeb48b5b` |

HEAD is `7e701b40485b9880ccfbb64c1faaabb22401c294`, commit `checkpoint`, dated 2026-09-12 18:34:22 +0200. The root audit reports its diff from the preceding checkpoint changes only the research ledger and assessment. It was not created by the root agent. Preserve it; its temporal proximity does not prove it caused source rewriting or the exceptions.

The full root integrity inventory in `incident_011.json` checked 169,334 EXP17 and 166,406 EXP18 evidence files before/after read-only gates, with no differences, and confirmed source/archive and prerequisite gates while model/candidate/key operations were blocked. The narrow two-minute pre-stop cache check found 72 completed EXP18 rows of status valid and no captured worker-error marker; EXP17 had no completed cache rows in that window. This is limited diagnostic evidence, not a proof of historical source immutability or an efficacy assessment.

## Safe unchanged resume conditions

1. Preserve both failed launch records, the failed Trace attempt, all responses, receipts, cache rows and existing selections, together with the source/read-only inventories. Preserve the unrelated checkpoint commit.
2. Confirm no live process owns the pipeline lease; rerun the existing frozen preflight and use the unchanged existing CLI. Do not alter timeout, worker source construction, prompts, budgets, candidate source or failure rules during this resumption.
3. EXP17: reuse all 467 responses, replay the same production prefix against existing caches, finish slot 02's missing TRAIN keys/receipt, then continue only genuinely unfinished slots. The independent read-only prefix audit reconstructed slots 00/01/02 with identical requests, parents and feedback, stopping at the expected missing-cache guard before evaluation.
4. EXP18: keep generation complete, reuse cached TRAIN/VALIDATION rows and the 21 finished selections, finish the missing M panel and remaining original selections. Keep audit inaccessible until the full selection barrier exists.
5. Carry the unknown interrupted-work disclosure into final resources. Do not infer a scientific loss from these infrastructure errors or replace ordinary invalid scientific outcomes.
6. A read failure with unchanged scientific semantics does not itself require redesign or invalidation. If later evidence establishes changed worker/evaluator semantics or corrupted scientific data during prior execution, apply the frozen defect policy to the affected evidence rather than treating current matching bytes as retrospective proof.

No candidate, objective, model call, heavy test, experiment launch or frozen-file edit was performed for this report. The decision and action to resume belong to the root agent.

## Evidence hashes

Paths below are relative to `artifacts/optimizer_discovery/`; hashes refer to exact stored bytes, including gzip where applicable.

| Evidence | SHA-256 |
|---|---|
| `exp17/run/pipeline_06.log` | `137e3785017b7ae9dec7c8d6b837cba76db9140c05fbd9c0230c8eb14d0f945c` |
| `exp17/run/runtime/launch_006/launch_finished.json` | `9d2ac895498f2c69df847cbf78bcadabc5b4481f621e4c5b1f8a0085d94d1107` |
| `exp17/run/raw/17030/C/trace_attempt_001.json.gz` | `67c900ac9242d1afbe572f142a3a610a25ea3bbc1e6582463a77d9a6f5fbe197` |
| `exp18/run/pipeline_07.log` | `6f1d11f779b3e11809975869c5a5cebbbc1676f6772ce423916e5f4799331368` |
| `exp18/run/runtime/launch_007/launch_finished.json` | `304f99e807adee338e6818628de6f072ca10cb7f0b833f8ac0b7961d318edfa7` |
| `exp18/run/generation_frozen.json` | `7a8186436ea6933ae36b78025b3aad157db9dee7d7221d3704441e9ffedb9d53` |
| `exp18/operational_status_checks/incident_011.json` | `fa49d0bd46f6a281e2d3a21cca17a990c1eef905b319e7f2f1c6299302895e9a` |
| `exp18/operational_status_checks/local_evaluation_resume_audit_001.md` | `ed336380bfdab3cdb1f4e06c6e8ce387f024e2a23d110703bda539054e4b0153` |

Frozen manifest file-byte hashes: EXP17 `4cfe9a5d9b696830992a9fc29700acfcb64b1051a981f25fe6c19b4848fa214b`; EXP18 `9abb465cda1ea55c9cce7e5e784621a9354033c1f6f4c82960f2b3ce917354be`. These are distinct from their canonical manifest digests.
