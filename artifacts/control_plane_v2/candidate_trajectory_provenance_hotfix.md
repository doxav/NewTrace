# Candidate-trajectory provenance hotfix

## Confirmed gap

Experiment 0 completed its frozen 40-unit main matrix, but the episode-export
audit found no `metadata.candidate_trajectory` records in any persisted Trace or
GEPA level result. The scientific results remain valid; only proposal lineage
needed for later episode construction was absent.

Both engines already expose the required data after optimization:

- Trace's returned trainer retains candidate artifacts and aggregate evaluation
  scores in `trainer_result.memory.memory`.
- GEPA 0.1.4's documented `GEPAResult.to_dict()` retains candidates, parent
  indices, validation aggregate scores, and `best_idx`.

The missing seam was therefore persistence in the recursive-opt engine
adapters, not proposal generation, evaluation, or selection.

## Tests-first causal proof

Before the runtime edit, the new real-Trace and actual-GEPA 0.1.4 tests failed
with `KeyError: 'candidate_trajectory'`. The audit's independent malformed-row
cases already passed, proving it rejects absence of artifact, relation,
evaluation, or selected/rejected status separately.

After the edit, the tests require:

- a nonempty trajectory for real Trace and real GEPA proposal paths;
- artifact, parent/seed relation, evaluation, and status on every row;
- the selected improved artifact to be represented;
- no extra optimizer-provider call;
- Trace record count to match its existing accounted proposal count; and
- persisted Trace trajectory to resume byte-semantically unchanged with zero
  additional provider calls.

## Minimal runtime change

Only `opto/features/recursive_opt/spec.py` changes in production. Four code
lines are added or replaced: capture GEPA's existing result history, initialize
the optional Trace trainer result, and attach one trajectory comprehension to
each engine's metadata. Two separator-only blank lines were removed so the
frozen 8,850-line runtime footprint remains unchanged. No helper, public API,
engine algorithm, evaluator, budget, model, data, objective, or selection rule
changes.

The old 40-run results cannot be retroactively assigned missing lineage. They
remain the authoritative scientific matrix and valid evidence of the provenance
gap. Future Trace/GEPA optimized runs under this runtime will persist the
trajectory required by the existing episode-export audit.

## Pre-CI identity

- `runtime_tree_sha256=89156f8abc7d198d439aaa923b9c154df48b14a97480363a29a25e34900d2877`
- `registry_sha256=f1a94b02c94607c2d22e6f10bce25b10d9b642814915ec01c72765280be26fa4`
- runtime footprint: 8,850 lines
- exact network-blocked required matrix: 270 passed, zero skipped
- complete unit suite: 510 passed, 3 unrelated optional skips, 1 existing
  LangGraph deprecation warning
- offline Experiment-0 A/B/C/D contract: passed
- Ruff, JSON validation, diff check, and tracked credential scan: passed
- readiness remains false until the required external CI job is observed green.
