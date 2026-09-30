# EXP-18 production and information adapters

This is an engineering description, not evidence of an optimization gain.
No frozen EXP-16 file or `opto/` source is changed by these adapters.

## Why a named Pareto setting is insufficient

The production `PrioritySearch.explore` method inserts `_best_candidate` first
when `use_best_candidate_to_explore=True`. At one parent it never calls the heap's
`pop`. Its `exploit` method uses scalar priority. Even `ParetoHeapMemory.pop` breaks
frontier ties by a scalar mean/weighted sum; it does not use the configured
`random_seeded` tie-break. A read-only runtime check with two nondominated vectors
and twelve different configured seeds selected the scalar-best vector twelve
times. Thus changing the objective mode alone would not exercise the proposed
parent policy.

## Actual intervention

`pareto_selection.py` reuses production `pareto_rank` after negating per-instance
regret AUC. Its input contains the trusted seed at index -1 and **every completed
slot** below the current update index, including typed invalid records. Each
vector has one mean per TRAIN instance, averaging the registered local seeds.
Vectors must be complete, finite and nonnegative. Invalid records are retained
and excluded from parent selection; they never receive an invented numeric score.

The archive is sorted by proposal index. Repeated source hashes and exactly equal
vectors have a single sampling position represented by their earliest index.
Selection samples uniformly from the full nondominated frontier using:

```python
seed = int(sha256(compact_json([
    "EXP18-PARETO-UNIFORM-V1", namespace, outer_seed, next_slot
])).hexdigest(), 16)
chosen = random.Random(seed).randrange(frontier_size)
```

The Python environment is frozen. This intervention includes frontier filtering
**and** uniform exploration of its members. It is not an isolated estimate of the
effect of dominance filtering. A large frontier can approach uniform exploration
of the archive; frontier size and non-scalar selections must be reported.

`pareto_trainer.py` creates a subclass overriding only `PrioritySearch.explore`.
It returns an actual archived `ModuleCandidate`, preserving the production
candidate/source identity used by the next backward/step callback. It does not
pop the archive, evaluate source, manufacture a replacement module, or run an
additional search loop. All TRAIN-valid archive sources must exist in production
memory. Missing invalid sources are permitted and remain visible in the receipt
archive. The scalar `exploit` method still maintains the engine's displayed/final
incumbent; common external validation selection is unchanged.

## Compatibility binding and provenance

The canonical engine accepts a trainer resource, but `optimize.resolve_trainer`
currently requires a registered string name. The factory therefore derives the
stable runtime alias `EXP18Pareto_` plus the first 24 hex characters of the SHA256
of compact JSON `[namespace, outer_seed, proposal_slots]`.

`registered_pareto_trainer` temporarily registers that alias in
`opto.trainer.algorithms`, rejects collisions and removes it in `finally`.
`MechanismStudy._pareto_engine_binding` scopes the EXP-16 schedule's engine
callback to an adapter that passes this alias in `resources['trainer']` to the
existing `control._run_module_engine`. It preserves the actual registered slot
optimizer. The prior callback and engine registry entry are restored on success
or error. Concurrent bindings are rejected. Generation remains concurrency one.

The inherited raw plan retains `trainer='PrioritySearch'` and
`engine='exp16_trace_v1'`. The actual injected alias, selected/scalar parent,
frontier membership and archive digest are recorded in
`parent_decisions/slot_XX.json`. These records are required in the generation
barrier. This explicit resource binding must not be mistaken for an unchanged
scalar parent policy.

## Compact feedback and memory

Every L/M/P/PM candidate is evaluated on the same allocated TRAIN panel. The
base EXP-17 owner persists the TRAIN receipt before returning a completed update.
`MechanismStudy.feedback` projects that already available receipt to aggregate
AUC only for a completely valid panel, allocated/observed/valid counts and typed
status counts. The compact projection is propagated through real Trace. The
proposal callback verifies its source and content and includes that exact
propagated string in the model request.

M and PM additionally receive the pure `memory_projection.build_memory` output.
It covers all earlier same-arm/outer slots, preserves invalid/empty responses, and
shows at most seven prior distinct nonempty sources excluding the current parent,
subject to the registered 65,536-character budget. Code is included whole or
explicitly omitted. Source differences are not labeled as search rejection.
Host timestamps, receipt hashes, instance vectors, normalization details and
benchmark parameters are not included in the projected prompt.

Every request first preserves `slot_XX/current_context.json`. Rebuilding a prompt
uses that original snapshot timestamp and only slot indices strictly below its
cutoff, even if later response files exist. A missing context after a request or
response exists is an error. Historical cache reads are authenticated and read
only; they cannot trigger replacement evaluations.

## Schedule and replay

All arms use one parent and one proposal for each update. For N responses the
production schedule runs N+1 iterations, including initialization. It calls
`explore` initially and after each update. The last decision is explicitly marked
`will_generate=false`; the final production sample reads an already evaluated
parent through the same cache. It creates no additional proposal or objective
allocation.

The next-slot counter tracks **consumed production callbacks**, not how many
responses already exist on disk. It resets to zero when the Trace graph is
rebuilt. Replayed responses advance it without a new model call. Tests rebuild
the real Control Plane/Trace path while retaining later completed responses and
verify identical contexts, request/source bytes and parent decisions, with no new
model response or objective evaluation.

The integration tests also exercise a deliberately conflicting TRAIN fixture:
the generated specialist has a worse scalar mean than the seed but belongs to
the frontier. A real Trace proposal callback receives that non-scalar parent's
exact source. This is engineering coverage, not a scientific optimization result.

## Driver and engineering gates

`driver.py` reuses the EXP-17 process lease, provider-wide generation lock, live
client, source ZIP snapshot and complete engineering replay verifier. Its only
registered live grids are M10 at outer 18901 (`engineering_memory`), L/P/PM with
two responses each at outer 18903 (`engineering_short`), and the exploratory
main grid of six outer seeds × four arms × sixteen responses (`run`, 384 slots).
The main whole-arm order reversals balance every pair's precedence 3/3.

Preparation freezes all four EXP-18 implementation modules, this driver, the
shared driver and analysis, and relevant tests. The initial protocol is copied
once to `PILOT_PROTOCOL.md`, so later main-protocol edits preserve pilot evidence.

Both pilot gates require complete production callbacks, a valid trusted seed,
at least one eligible generated replacement per pilot, immutable slot/context
evidence and a complete resume with no client or extra objective evaluations.
The long-memory gate authenticates slot 9 with all nine prior-slot summaries;
it records the number of distinct sources actually shown and does not demand
seven usable sources by silently replacing invalid or duplicate responses.
Every P/PM decision is reconstructed from its TRAIN archive, including the unused
terminal decision. Any pilot audit cache row is rejected even when no audit
result export exists. Main preparation and generation require both gates; the
generation action checks them before creating the live client.
