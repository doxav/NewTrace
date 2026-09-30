# F1 engineering review before live execution

The implementation is ready for **preparation and freeze after G1 completes**.
This review did not prepare or run scientific F1. All acceptance-run model
responses and evaluation panels were explicit unit fixtures, on the separate
`F1_UNIT_REVIEW` namespace. No model call was made by this review.

The fixed-parent four-condition design separates objective instruction from
evaluation information. It is a mechanism study, not an iterative-search result.
The parent, task panel, local optimizer seed, proposal slot budget, token cap and
model settings are common within a block. Rich-versus-sparse changes several
declared information fields together; it does not identify which individual
field caused any future effect. Six blocks and four contrasts remain exploratory.

## Defects found and corrected before F1 freeze

| Finding | Resolution and evidence |
|---|---|
| A response file with `completed=False` or the wrong model could open validation. | Verify exact model, completion flag, positive completion timestamp, unique provider response ID, raw-content extraction, source hash and source status for all 24 slots. Negative tests reject the corrupted last response before any evaluation. |
| A missing response after validation could issue a new model call on resume. | Audit all responses and the complete barrier before loading credentials or constructing a client. After the barrier, `run` only resumes validation/analysis. An integration test completes all 24 synthetic slots and then resumes with zero additional calls. |
| The manifest could be edited without changing an embedded code hash. | Seal the manifest using `freeze_sha256.json`; recompute the cap from the G1 reliability-only fields; check G1 result identity; reconstruct the six parent blocks and all 24 shuffled requests from the preserved training contexts. Even a recomputed manifest checksum cannot authorize a changed cap, parent, task namespace or request schedule. |
| Actual request files and trajectory provenance were not rechecked. | Match every request against its frozen form; check every source, semantic task identity, local seed, stratum, objective budget, validity/fallback state, actual call count and metric. Seal all training, validation and control evaluation files for analysis. |
| Interrupting preparation could reevaluate existing contexts or strand a valid unsealed manifest. | Reuse verified existing context rows. `prepare` may seal a fully reverified manifest only when no request or validation has started. The live preflight never accepts an unsealed manifest. |
| The old generic redactor changed benign `task-specific` text. | `evidence_io.py` preserves exact finite JSON, checks the active credential privately and rejects recognizable credentials without echoing them. Tests cover benign source/request identity, immutable writes, compression and synthetic-key rejection. |

The exact recorder reuses the already tested `generation.complete_slot` transport,
retry and resume implementation. A nonblocking lock protects a temporary recorder
binding in the current process, restored after success or failure. Concurrent
direct `G.complete_slot` calls in that same process are unsupported. G1 runs in a
separate process and its files, behavior and evidence remain frozen. A two-thread
unit test confirms the second adapter request is rejected before a slot is made.

## Preserved review failure and namespace amendment

The first negative test for altered request metadata did not mock evaluation.
Since the defect was real, validation began on the old F1 tasks using the unchanged
seed. The test was interrupted using session **95134**, without inspecting any
performance value or using one for a choice. One completed panel file (70,956
bytes) was preserved byte-for-byte as compressed evidence; the interrupted
partial work was not reconstructed. The original file path and both original-data
identity and preserved artifact path are recorded in `review_failure/index.json`.

The old F1 validation panel is therefore not claimed unseen. Before any live F1
call, the task and local-seed namespace was amended to **F1-R1**, with unchanged
reporting stage and paths. `PROTOCOL_F1.md` records this amendment. Unit fixtures
now use `F1_UNIT_REVIEW` and prohibit real evaluation by default. No G1 request or
result was changed, and no F1 scientific result was invalidated because none
existed.

## Static validity gate clarification

Independent inspection of G1 `16002/AL_32000/slot_00` confirms source hash
`068c006eaa6ba610ccad85d15245f7786f365789a7dd09ba36d97ed77ebc0fd2`.
The identifier `dir` occurs once as `Store` and once as `Load` on line 122, as a
comprehension variable. The static rule rejects all occurrences, even without a
call to the built-in. A separate synthetic comprehension is rejected under that
name and accepted under `direction`. The original generated program was neither
edited nor executed for this diagnostic; `identifier_gate.json` preserves it.

The program stays invalid under G1's declared rule. This is not evidence of
attempted introspection. F1 now tells **all four conditions** the existing reserved
identifier rule explicitly. The evaluator remains unchanged. Future investigations
must distinguish protocol-screen invalidity from execution invalidity.

## Search strategy review

`num_candidates=1` plus `use_best_candidate_to_explore=True` always uses the
incumbent. Switching only `use_best_candidate_to_explore=False` while retaining
mean priority need not create a different strategy: explored parents are re-added
to the archive by `validate` and `update_memory`.

An informative production comparison should verify actual parent hashes, not
strategy labels. A width-two search can branch while preserving eight total
responses by reducing rounds and auditing initialization and final evaluation
budgets. Root's prospective P1 design follows this direction. The `time` priority
currently returns negative creation time; its larger values favor older candidates,
despite the adjacent comment describing newer ones. No recency benefit should be
claimed from that configuration without a distinct test or adapter.

## Verification

Combined audit, rich-feedback, exact-recorder, F1-integrity, original F1/G1 tests
and the real EXP-15 production integration test: **58 passed in 8.94s** before a
test-only cleanup of thread exception propagation. The exact-recorder suite was
then rerun: **7 passed in 1.79s**. That cleanup uses `Future.result` to propagate
thread failures without broad exception handling. Ruff passes; Black formatting
was applied to the changed test. No old production or frozen experiment file was
modified. No production dependency was introduced.

Preparation still requires complete G1 results, followed by F1 preflight and freeze.
Generation and evaluation efficacy are unproven until the live F1 and subsequent
independent production studies execute. This engineering GO is not a projected
scientific gain.
