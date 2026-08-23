# Proposal-gate semantics correction

## Confirmed experiment-runner defect

The post-reflection-fix GEPA arm successfully called the guarded optimizer,
received a textual reflection, produced a changed candidate, and evaluated that
candidate. GEPA then rejected it because its scalar objective was worse than
P0, so the final selected artifact correctly remained P0.

The Experiment-0 runner nevertheless reported `proposal_nontrivial=false`
because it required both a proposed candidate and a final artifact different
from P0. That implementation merged two distinct events:

1. whether the optimizer exercised its proposal/evaluation path;
2. whether selection accepted the proposal over the seed.

Retaining P0 after evaluating a worse candidate is not proposal failure. It is
the expected behavior of the validation/search selection step.

## Corrected protocol

The versioned experiment runner now records:

- `proposal_path_exercised`, an infrastructure gate requiring optimizer calls,
  optimizer tokens, proposed candidates, and evaluated candidates;
- `selection_changed`, a result diagnostic recording whether the selected
  artifact differs from P0.

Persistence/resume testing depends only on its infrastructure prerequisites.
It no longer depends on selection outcome or candidate acceptance. The pilot
retains three distinct gates: `trace_real_proposal`, `gepa_real_proposal`, and
`optimized_artifact_differs`.

This correction restores Prompt 18-R3's original distinction. It does not
change the task, data, P0, model profiles, objective, invalidity constraint,
holdout policy, paired design, or scientific success criteria. The failed
post-fix report, stop decision, prior attempts, and pre-reflection-fix evidence
remain immutable historical protocol-debugging evidence.
