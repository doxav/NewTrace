# Prompt 18-R3C pilot stop decision

## Proposal-gate correction

The Experiment-0 runner now separates infrastructure from selection outcome:

- `proposal_path_exercised` requires optimizer calls, optimizer tokens, at
  least one proposed candidate, and at least one evaluated candidate;
- `selection_changed` records whether the final optimized artifact differs
  from P0 and is not a micro-smoke gate;
- persistence/resume runs whenever its explicit infrastructure prerequisites
  pass;
- pilot gates separately record `trace_real_proposal`,
  `gepa_real_proposal`, and `optimized_artifact_differs`.

The frozen recursive-opt control plane was not changed. Its runtime digest is
`37072c1364a02c277a677bf43ad8132a32a9f233488c80cd2b6bf1a7e344f33e`
and its registry digest is
`f1a94b02c94607c2d22e6f10bce25b10d9b642814915ec01c72765280be26fa4`.
The amended Experiment-0 source digest is
`32803d7a5c824a27930243a625ca5915c4892bfa2d275aecfc7d7311e7dc7100`;
the Experiment-0 registry remains
`18a7efd58f3d265a723fa62efb89d2dc07082afbcdf16c9470e6d4eb93a77121`.

The R3B stop report, post-fix live report, post-fix attempts, and provisional
pre-reflection-fix evidence remain unchanged as historical protocol-debugging
evidence.

## Corrected live micro-smoke

The fresh GSM8K 1/1/1 A/B/C micro-smoke passed under the amended experiment
lock. Task, samples, P0, model profiles, objective, seed, and budgets were
unchanged.

| arm | accuracy | forward-token ratio | forward calls/tokens | optimizer calls/tokens | proposed/evaluated | selection changed | resume |
|---|---:|---:|---:|---:|---:|---|---|
| A fixed | 1.0 | 1.218434 | 2 / 965 | 0 / 0 | 0 / 0 | false | pass |
| B Trace | 1.0 | 0.936869 | 16 / 4,256 | 1 / 5,442 | 2 / 5 | false | pass |
| C GEPA | 1.0 | 1.138889 | 10 / 2,974 | 1 / 1,214 | 4 / 4 | false | pass |

Both optimized arms have `proposal_path_exercised=true`. GEPA again produced
and evaluated a changed proposal while selection retained P0; this is now
correctly accepted as valid micro infrastructure evidence. Holdout isolation,
usage reconciliation, exact provider/model/request parameters, source locks,
and output persistence/resume all passed.

## Cost forecast

OpenRouter model metadata retrieved on 2026-08-23 records the frozen model at
`$0.08` per million prompt tokens and `$0.18` per million completion tokens.
The measured micro usage projects:

- complete 24-run pilot: 3,804,320 tokens and `$0.4155584`;
- full-pool main design: 36,958,306.67 tokens and `$4.0359419`.

The pilot forecast is complete. Main execution remains unauthorized because
the preregistration has no acknowledged main-run monetary ceiling.

## Pilot stop gate

The frozen three-seed A/B/C/D pilot began and stopped after 6 of 24 planned
runs, at seed `0`, candidate budget `6`, arm `B`.

The seed-0/budget-4 A/B/C/D quartet completed. Both Trace and GEPA exercised
real proposal paths; GEPA selected a changed artifact, independently satisfying
the observed selection-change condition for the completed quartet. Seed
0/budget-6 A then completed. On B's sixth optimizer call the run ended with:

```text
TypeError: argument of type 'NoneType' is not iterable
```

The failed arm recorded 6 optimizer calls, 38 candidate evaluations, 84 forward
calls, 72,212 accounted tokens, no budget overrun, and no holdout leakage. Its
canonical result is invalid and cannot satisfy proposal, reconciliation,
persistence/resume, or selected-artifact safety gates. The aggregate pilot
therefore records:

- `all_planned_runs_completed=false`;
- `trace_real_proposal=false` across all attempted Trace runs;
- `no_selected_invalid_artifact=false`;
- `outputs_and_resume=false`;
- overall `passed=false`.

No retry, profile change, gate change, remaining pilot arm, main run, or episode
export was attempted. This is not a new recursive-opt control-plane diagnosis;
it is a live pilot execution stop under the frozen Experiment-0 protocol.

`BLOCKED_EXPERIMENT_RUNNER`
