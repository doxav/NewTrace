# Scientific versus infrastructure gate correction

## Confirmed causal chain

Before this correction, `live._execute_arm()` set `checks["run_succeeded"]`
only when `result.status == "success" and result.valid`. The shared
`_infrastructure_checks_pass()` included that value. `run_main_experiment()`
stopped immediately whenever those checks failed.

The frozen preregistration instead says
`stop_on_first_infrastructure_failure = true`; it does not say to stop on the
first scientific failure. The main analyzer already evaluates the frozen
safety outcome independently as `all(run.invalid_rate == 0)`.

Consequently, the first main A result (`status=invalid`,
`evaluation.status=constraint_failed`, `invalid_rate=1/24`) was a completed,
interpretable scientific safety failure that the experiment runner incorrectly
classified as an infrastructure failure.

## Falsification evidence from the prior P0 baseline

The persisted P0 holdout baseline result has SHA-256
`fcabf0bee08d97d442060b392defa173952bcaa22c97f290369ea5aea1a65b06`.
It used the same P0 artifact, forward model/profile, and 24 frozen holdout
examples and recorded:

- `status=success`, `valid=true`, `accuracy=1.0`, `invalid_rate=0.0`;
- `gsm8k:test:915`: expected `23`, answer `23`;
- `gsm8k:test:35`: expected `9`, answer `9`.

The stopped main realization recorded `gsm8k:test:915` as non-extractable and
`gsm8k:test:35` as the valid but wrong answer `0`. Its persisted result has
SHA-256
`7df98c0849122fc3e571ea10bc42975afe6f3e2c60bdeb1b5bad2278aa5f750d`.
This proves that P0 is not deterministically invalid. It supports only the
conclusion that one stochastic provider realization violated the unchanged
frozen safety criterion. The raw final provider text was not persisted, so no
specific formatting cause is inferred.

## Corrected semantics

The experiment runner now separates:

- `execution_completed`: canonical success, or canonical invalid result whose
  evaluation status is `invalid` or `constraint_failed`;
- `scientific_feasible`: the unchanged `result.valid` value;
- `safety_passed`: the unchanged `invalid_rate == 0.0` criterion;
- `selection_changed`: the unchanged selected-artifact diagnostic.

Only `execution_completed` is an infrastructure gate. Missing evaluations,
provider/runtime errors, budget exhaustion, provenance/accounting failures,
holdout leakage, proposal-path failures, and persistence/resume failures remain
blocking infrastructure failures.

No task, sample ID, artifact, parser, model/profile, request parameter,
objective, constraint, seed, budget, arm, pairing, or statistical rule changed.
The old 1/40 stopped matrix remains immutable evidence that its P0 realization
failed safety. No B, C, or D main outcome was observed before this amendment.
