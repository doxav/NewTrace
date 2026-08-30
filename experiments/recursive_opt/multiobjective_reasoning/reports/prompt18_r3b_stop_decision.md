# Prompt 18-R3B stop decision

## Control-plane repair and lock

- implementation commit: `c1bf6a296d159b92d09ac29ee90556d2c1997a5d`
- runtime tree: `37072c1364a02c277a677bf43ad8132a32a9f233488c80cd2b6bf1a7e344f33e`
- registry: `f1a94b02c94607c2d22e6f10bce25b10d9b642814915ec01c72765280be26fa4`
- required implementation CI: run `32650995105`, job `97222458541`, success
- evidence-state CI: run `32651342206`, job `97223324883`, success
- Experiment-0 source: `40e31f1d50d2dd89f6983a9e3a2e20b0c858d208f5ec6c2a394172eb9292b87e`
- Experiment-0 registry: `18a7efd58f3d265a723fa62efb89d2dc07082afbcdf16c9470e6d4eb93a77121`

The old locks and the original failed C arm remain historical evidence. The new
v3 lock is `control_plane_lock_after_gepa_reflection_fix.json`. The selected
GSM8K task, 16/12/24 pools, eligibility result, P0, profiles, objective, and
budgets were not changed or rerun.

## Causal repair result

The GEPA 0.1.4 reflection protocol defect is fixed. The installed-GEPA offline
test and live C arm both exercised:

```text
GEPA text prompt
  -> production GEPA adapter
  -> guarded optimizer chat client
  -> OpenRouter response
  -> exact textual content returned to GEPA
```

The live C arm made one optimizer call using the recorded low-effort request
parameters, attributed 1,584 optimizer tokens once, produced and evaluated a
changed reflection proposal, and did not expose holdout during optimization.
There was no recurrence of `AttributeError: 'str' object has no attribute
'get'`.

## Live micro rerun

| arm | accuracy | token ratio | forward calls/tokens | optimizer calls/tokens | proposed/evaluated | selected artifact |
|---|---:|---:|---:|---:|---:|---|
| A fixed | 1.0 | 1.202020 | 2 / 952 | 0 / 0 | 0 / 0 | P0, unchanged |
| B Trace | 1.0 | 0.984848 | 18 / 4,990 | 1 / 5,193 | 3 / 6 | changed |
| C GEPA | 1.0 | 1.058081 | 8 / 2,033 | 1 / 1,584 | 3 / 3 | P0, unchanged |

The first post-fix attempt stopped in B when the unchanged Trace optimizer
again exhausted the 8,192-token completion allowance in reasoning and returned
no final content. That attempt and the subsequent resume-cache replay are
preserved under `reports/post_fix_micro_attempts/attempt1_trace_reasoning_exhaustion/`.
No setting changed before the one fresh retry shown above.

A and B are behaviorally compatible with the provisional pre-fix A/B evidence:
both remained valid at accuracy 1.0, A remained P0, and B produced a changed
candidate through the real Trace path. Token differences are provider variation.

## Blocking gate

GEPA proposed the changed analysis instruction shown in the GEPA console and
evaluated it. GEPA rejected it because its one-example subsample scalar score
was `-0.06016260162601627`, below P0's `-0.057723577235772365`, so the selected
artifact remained P0.

The frozen live runner consequently reports:

- `proposal_nontrivial=false`, because its implementation requires the selected
  optimized artifact to differ from P0;
- `output_persistence_and_resume=false`, because resume is attempted only after
  every preceding check passes;
- overall `passed=false`.

Changing that gate after observing the proposal outcome would be a post-hoc
protocol change. The report is preserved as generated. The reflection adapter
is causally validated, but the Experiment-0 micro stop gate is not green.

## Decision

No cost forecast, pilot, main experiment, ablation, or optimizer-efficacy
comparison was run after the failed gate. Holdout use remained limited to each
arm's permitted final one-unit evaluation.

`BLOCKED_RETURN_TO_CONTROL_PLANE`
