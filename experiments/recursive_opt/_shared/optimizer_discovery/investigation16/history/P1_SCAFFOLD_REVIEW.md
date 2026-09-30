# Prospective P1 orchestration review

This is engineering evidence with scripted model responses and mostly synthetic
unit evaluation values. It is not a live P1 result or a preregistration. No live
request, final P1 freeze, commit or change to EXP-15/Phase-0 evidence was made by
this task.

## Implemented boundary

`../search_experiment.py` owns configuration, exact recording, common evaluation
caching and the generation/selection/audit phase barriers. C/R/W updates remain
inside the existing Control Plane, versioned EXP-15 training evaluator and actual
PrioritySearch trainer through `../trace_schedule.py`. The independent arm uses
fresh fixed prompts. No second iterative search loop or dependency was added.

The optional owner hook receives `self.trace_graph.user_feedback` from the actual
backward pass. R/W prompts preserve this exact propagated string. Their canonical
payload is checked against the permitted raw training projection, including source
identity. C receives its current production-selected source with no evaluated
feedback. All four arms receive the same explicit anytime objective, seed and
execution instructions. The optional single aggregate training AUC defaults off;
enabling it is an explicit joint correction beyond F1's raw-only factor contrast.
Per-task normalized values, normalizers, optima and hidden parameters are excluded.

Training feedback contains the current parent's complete registered panel, with
deterministic raw progress summaries. It does not append a separate previous
attempt. Width two means two **production-selected exploration parents**; these
may include an invalid source. It is not an assertion that both parents are valid
or that a particular alternative search framework has been implemented.

Every completed response consumes one slot. The common seed plus every response
gets the same logical training and validation allocation, including early-invalid
trajectories. Exact physical evaluator calls are recoverable from unique cache
rows, independently of logical allocations and production cache-hit events. Cache
keys include namespace, exact source, task identity, split, outer/local seed,
budget, deployment mode, timeout, seed hash and evaluator version/source hash.
Gzip rows are checked with the same reader and an exact row hash.

Global generation completion precedes all validation; all selections precede any
audit trajectory. Every barrier verifies the complete registered record set and
hashes. Response IDs, exact model, parser output, source hashes, settings, parent
lineage and prompt content are rechecked against frozen configuration. Preflight
reconstructs tasks, arm order and panel sizes, and checks source/environment hashes.
The shared EXP-15 benchmark manifest, metric/bootstrap settings and seed are frozen
as dependencies; this does not substitute EXP-15 instances for the new P1 namespace.

All selected policies use the existing permanent seed deployment fallback with
actual accumulated history and no objective-budget reset. Audit aggregation can
be recomputed from preserved cache rows and refuses a mismatch. This preserves
the existing subprocess contract; it does not establish OS filesystem confinement.

## Defects isolated before live execution

1. The first actual-Trace hook test failed because the old adapter called only
   `owner.proposal`; after the optional hook, all eight callbacks receive the
   actual sentinel through production propagation.
2. The initial prospective parser accepted scalar JSON only. Native invalid
   evaluation feedback is instead propagated as `ID [0]: ['{...JSON...}']`.
   The new parser accepts strictly one string in that literal list using
   `ast.literal_eval`, then JSON validation. It rejects arbitrary expressions,
   unexpected elements, wrong source identity and extra hidden fields. The exact
   propagated representation remains preserved. This was an adapter parser
   defect, not evidence that PrioritySearch reduced the response allocation.
3. After fixing the envelope, the all-invalid width-two search completed **eight
   actual callbacks**, but returned a typed invalid final exploration artifact.
   The old completion gate misclassified this ordinary search failure as an
   infrastructure failure. Completion now accepts only a valid result or the
   exact typed tuple `status=invalid`, `valid=false`, `error=invalid_program`
   at both result and evaluation levels, with all callbacks complete. It records
   `search_completed_final_artifact_invalid` and preserves the raw invalid result.
   The seed-inclusive validation pool selects the trusted seed. Unknown engine
   errors still fail. All-invalid and identical-source W tests both retain eight
   callbacks; no trainer or parent-ranking change was needed.
4. Review of the separate driver exposed a physical/logical gzip-path mismatch:
   the existing reader accepts a logical `.json` path and locates `.json.gz`
   itself. A chronology scan passed the physical gzip filename, producing a
   UnicodeDecodeError in a new failing test. The scan now converts it back to
   the logical path. Exact source and evaluation bytes are unchanged; the frozen
   EXP-15 reader was not edited.

Preserved actual synthetic evidence:
`raw/p1_all_invalid_red_trace.json.gz` and
`raw/p1_native_invalid_envelope.json` (writer may compress by size). The raw trace
contains eight callbacks and the original invalid final result before the gate
correction. Earlier parser failure was diagnosed from its failed unit trace;
its exact problematic envelope is preserved here.

## Context and runtime limits

A declared synthetic 48-trajectory B32 panel with 32 improving observations per
trajectory produces **276,612 characters** in native Trace JSON, versus 255,907
for compact JSON, before invariant text/current source. This demonstrates that
262,144 characters is not a sufficient general engineering bound for the proposed
24-task × two-local-seed training panel. Register a larger bound (for example
524,288) and check actual context/token feasibility in the separate P1-E1 live
engineering test before P1. This is a representation-size diagnostic, not a
benchmark evaluation or a favorable-result design choice.

Phase and slot clocks preserve wall, monotonic and, where available, Linux
boottime readings. Boottime minus monotonic elapsed time estimates suspension;
negative deltas flag a reset. Suspension or wall-clock adjustments must not be
attributed to provider latency. These readings do not change the frozen live
slot recorder or its transport-retry behavior.

## Handoff and remaining work

Public entry points: `configuration(max_tokens=...)`,
`prepare(root, config, protocol, extra_frozen_paths=[...])`, `preflight(root)`,
`run_generation(root, client=...)`, `select_all(root)`, `run_audit(root)` and
`verify_chronology(root)`. `verify_arm_responses(root, outer, arm,
completed_before_ns=...)` reuses the same strict request/source/lineage checks for
a single arm without requiring a global four-arm barrier. A separate R-only two-response engineering run can call
`SearchExperiment(root, 'R', client=...).generate(outer)` after its own explicit
freeze; it must not call the four-arm generation coordinator or audit.

The parent task owns final protocol decisions, live-client creation, bounded
P1-E1/P1 execution, provider metadata collection, analysis/reporting and commits.
The source is not a claim that the larger prompt, new feedback or width-two search
will improve held-out performance. Only the prospectively frozen live comparison
can test that claim.

Validation with `/tmp/phase0-venv/bin/python`:

```text
-m pytest -q tests/unit_tests/test_investigation16_search_experiment.py
  tests/unit_tests/test_investigation16_history_trace_review.py
  tests/unit_tests/test_investigation16_trace_schedule.py
  tests/unit_tests/test_recursive_exp15.py
  tests/unit_tests/test_recursive_optimizer_program.py
68 passed in 57.83s

After the final single-arm verification helper extraction:
-m pytest -q tests/unit_tests/test_investigation16_search_experiment.py
15 passed in 20.48s

After the additional gzip scan regression:
-m pytest -q tests/unit_tests/test_investigation16_search_experiment.py
  tests/unit_tests/test_investigation16_history_trace_review.py
  tests/unit_tests/test_investigation16_trace_schedule.py
28 passed in 55.14s
```

Black `--check --target-version py313`, Ruff checks for the four changed
implementation/test files, and `git diff --check` pass. No skips in these test
commands. Exact prospective implementation hashes at handoff:

- `search_experiment.py`:
  `9050d74c9a41aea1822cc353fba6b829c8cda0b5de9ffbd54646e4cd7aa12e3e`
- `trace_schedule.py`:
  `a1ad02d77d3b5b665f9e64b6e1894b222350a016211aaa678b7d4757c33b5732`
