# EXP-16 / F1 — objective instruction and feedback information

Prospective exploratory mechanism study. Exact requests and code/environment hashes
will be frozen before the first F1 call. No EXP-15 result is changed. The model
and non-cap settings remain G1's DeepSeek/OpenRouter configuration. The cap will
be set by the registered G1 reliability rule, before examining F1 outcomes.

Six blocks, generation request seeds 16301–16306, four conditions per block:

1. `legacy_code`: original generic objective instruction; improve a fixed parent,
   without evaluated feedback.
2. `anytime_code`: identical request plus an explicit description of the actual
   anytime objective; no evaluated feedback.
3. `anytime_sparse`: condition 2 plus original EXP-15 sparse current-parent feedback.
4. `anytime_rich`: condition 2 plus deterministic indexed trajectory feedback with
   incumbent coordinates, improvement events and best-so-far values. No hidden
   task labels/parameters/optima/normalization constants.

The even-indexed three blocks use the unchanged seed as parent; the other three
use EXP-15's validation-selected representative from outer 41, exact source hash
`1684f91acdc36c0ca6aac70afeb9cc2c4eed7ab847926d5880590e059266abb7`.
Parents are fixed before this study, never chosen using F1 results. All four
conditions receive the same parent and local seed within each block. All requests
are independent fixed-parent calls. This is not a replacement for an iterative
production search test and cannot establish a recursion-depth benefit.

Use a prespecified shuffled condition order for each block, RNG seed 163000.
No resampling of order after inspecting results. Same request-seed hint within a
block; outer LLM determinism is not assumed. Capture actual provider receipts.

Tasks: `fresh_tasks('F1-R1', 'train', 1)` and separate
`fresh_tasks('F1-R1', 'validation', 1)`: six balanced family/dimension instances in
each panel, EXP-15 transformations/ranges. B=32. Local randomness from
`local_seed('F1-R1', block, task)` is shared by all conditions and parents.
Freeze both panels before calls. Only training data enter prompts. Freeze all
responses before any F1 validation evaluation. No response influences another
request. No manual repair or replacement. The common budget is 24 responses.

Retain every training invalidity and partial trajectory as typed evidence. For
validation, measure each proposed deployment policy with the same permanent
unchanged-seed fallback used in EXP-15; do not omit failed candidates or reset
the objective budget. Also evaluate each fixed parent and seed on these same
validation tasks. This is a one-proposal intervention estimand, not best-of-N.

Predeclared contrasts, lower AUC better:

- anytime_code − legacy_code: objective-instruction intervention;
- anytime_sparse − anytime_code: sparse evaluation-information intervention;
- anytime_rich − anytime_sparse: added trajectory-information intervention;
- anytime_rich − anytime_code: combined feedback-information intervention.

Report every block, parent type, source/execution invalidity, deployment fallback,
actual tokens/cost, mean/median paired delta and the same paired percentile
bootstrap implementation as EXP-15 (seed/resamples retained). With n=6 and three
blocks per parent class, intervals are descriptive and fragile. Four contrasts are
exploratory; no significance claim and no selecting a winner by the most favorable
contrast. A full production pilot must independently test any recommended change.

Engineering tests must establish exact pair invariants, objective-instruction
placement, current-source integrity, hidden-data exclusion, deterministic feedback,
no validation before all 24 responses, invalid-response accounting, and freeze
integrity. Preserve failed probes and protocol amendments. Later studies use new
stages and task/outer seed namespaces.

## Engineering amendments before any F1 live call

F1-R1 replaces the initially proposed F1 task and local-seed namespace. A failed
unit integrity test accidentally reached real evaluation of the handwritten seed
on the old F1 validation panel before its missing request-integrity check was
implemented. The test was interrupted; no performance value was inspected or used
for any decision, and no model call was made. One completed validation-panel file
was preserved verbatim, compressed and hashed under
`feedback/review_failure/`; partial interrupted work was not reconstructed. The
old F1 validation panel is therefore **not claimed unseen**. F1-R1 is a fresh
deterministic namespace, chosen without examining those outcomes. Tests now use
`F1_UNIT_REVIEW` tasks and forbid real evaluation by default.

All four conditions receive one identical clarification of the existing static
validity rule: reserved built-in names such as `dir` cannot be used anywhere,
including as ordinary local variables. This makes the declared conservative AST
screen explicit; it does not change the validator or repair any G1 candidate.
G1's `16002/AL_32000/slot_00` used `dir` as a comprehension variable and remains
typed invalid under its frozen rule. Its invalidity is not evidence of attempted
introspection or unsafe execution.

F1 uses an exact immutable recorder, `evidence_io.py`. The prior generic redactor
matched the benign substring `sk-specific` inside `task-specific`, which changed
persisted instructions. The new recorder rejects actual active or recognizable
credentials without printing them, and otherwise preserves exact request/source
bytes and hashes. It reuses G1's transport/slot code with a temporary process-local
recorder binding, restored after each call; the separate live G1 process and its
frozen files remain untouched. Log/error redaction remains available for logs.

Before execution, `freeze_sha256.json` seals the manifest. Preflight checks code,
environment, G1-result hash and reliability-only cap decision; recreates every
request from the frozen parents and training contexts; and verifies the six-by-four
unique response schedule. Responses require exact model identity, completion flag,
unique ID and source/hash correspondence. All training allocations are verified
before opening validation. The barrier seals complete response hashes and forbids
new generation, including on resume. Analysis checks every source/task/local seed,
budget, preserved metric and evaluation-file hash. These controls were implemented
and tested before freeze and before any F1 generation.
