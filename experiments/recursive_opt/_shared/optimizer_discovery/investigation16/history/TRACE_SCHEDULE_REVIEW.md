# Review of the prospective EXP-16 production schedule adapter

This is an engineering review before a new live search pilot. Its scripted
clients and scalar feedback are unit fixtures, never scientific model evidence.
The adapter delegates to `E._trace_engine` / `control._run_module_engine` and the
real production `PrioritySearch`; it does not introduce another search loop.

## Defect reproduced and fixed before live use

The initial implementation wrote `trace.json` even when an engine invocation
stopped after only two of its eight response slots. Resume correctly preserved
the two completed responses and generated the remaining six, but then failed
because immutable `trace.json` could not be replaced with a successful trace.
The same overwrite conflict existed after a crash between successful trace
serialization and writing `generation_complete.json`.

`raw/trace_resume_red.txt` records **2 failing and 3 passing** tests before the
fix. The narrow adapter now:

- writes unsuccessful engine evidence to numbered `trace_attempt_001.json`, etc.;
- writes canonical `trace.json` only after all allocated callbacks completed and
  the production result is valid;
- recovers a missing completion marker from a valid canonical trace only after
  checking every allocated response file and the immutable schedule;
- records reconstructed callbacks as `trace_update_replay`, so resumed callbacks
  do not inflate the scientific response count;
- continues to reject ambiguous started requests without completion receipts.

No EXP-15, Phase-0 or production-library file changed. Only the new
`investigation16/trace_schedule.py` adapter changed. No live result needed to be
invalidated for this defect.

## What the new tests establish

The controlled unit evaluator gives each valid generated source a distinct,
monotonically improving score. This makes parent choice observable while keeping
the real Trace propagation, production scheduling, response serialization and
resume machinery active.

- Greedy scheduling uses the previous valid improvement as the next parent.
- Two proposals per parent share a parent within each pair.
- Two parents per round use different source hashes by the second round when
  the available candidates have different ranks; a configured width alone does
  not ensure diversity on a real plateau.
- One deliberately empty response consumes its slot, remains invalid, and does
  not prevent eight allocated responses or produce a replacement request.
- Every generated source, including the invalid one, reaches the training
  evaluator; the fixture asserts that no validation split is requested.
- Generation settings remain identical except the registered request-seed hint.
- A partial resume retains previous response bytes and reaches eight total
  successful unit responses. Two replayed callbacks are counted separately.
- An ambiguous in-flight request causes no new client call.
- A completed trace can recover its final marker without regenerating a response.

The root's real subprocess tests additionally exercise the same schedule through
the actual optimizer benchmark at B=2 for (parents, proposals) = (1,1), (1,2),
(1,4), (2,1), each with eight unit-generated response slots.

## Budget counters must be interpreted correctly

The native engine reserves `iterations * num_candidates`, which is not the
number of completed LLM responses. The new scripted runs allocate eight responses:

| Schedule | Native reserved | Native proposed counter | Native evaluated counter | Completed responses |
|---|---:|---:|---:|---:|
| 1 parent × 1 proposal | 9 | 17 | 17 | 8 |
| 1 parent × 2 proposals | 5 | 13 | 13 | 8 |
| 2 parents × 1 proposal | 10 | 7 | 18 | 8 |

These native counters describe different internal objects and visits. In
particular the `proposed` counter is derived from retained trainer memory, and
the injected owner client is outside native role-client accounting. The owner
must therefore continue to report actual immutable response slots, transport
attempts, real tokens, logical evaluation allocations, unique objective calls,
and cache hits. It must not relabel native reserved/proposed values as the LLM
proposal budget.

## Scientific design choices still required

The old EXP-15 `E.proposal` supplies feedback from the immediately preceding
response, even when the production parent is fixed for a batch. Thus batch2
means a shared parent with potentially updated feedback, unless the new owner
explicitly freezes feedback at the round boundary. Register that choice before
using the labels "batch" or "independent siblings".

The new owner must also provide its own frozen model/prompt request builder:
unchanged `E.proposal` would reuse EXP-15's 8000-token setting and its old
non-gzip-aware response existence check. The newer `G.complete_slot` already
supports immutable gzip-aware response replay. The schedule adapter itself
intentionally delegates request design and provider evidence to its owner.

Changing `num_candidates` changes actual parent scheduling under the controlled
fixture; it does not establish that increased breadth improves benchmark quality.
Keep parent hashes and realized diversity in the live result, and compare all
proposed schedules at equal response and candidate-evaluation allocations.

Tests and verification commands:

```bash
/tmp/phase0-venv/bin/python -m pytest -q tests/unit_tests/test_investigation16_history_trace_review.py tests/unit_tests/test_investigation16_trace_schedule.py
/tmp/phase0-venv/bin/python -m black --check --target-version py313 artifacts/optimizer_discovery/investigation16/trace_schedule.py tests/unit_tests/test_investigation16_history_trace_review.py
/tmp/phase0-venv/bin/python -m ruff check artifacts/optimizer_discovery/investigation16/trace_schedule.py tests/unit_tests/test_investigation16_history_trace_review.py
```

Final result: **11 tests passed in 48.49 s**, preserved in
`raw/trace_review_green.txt`. The adapter must stop
changing before the live pilot's source/configuration freeze.
