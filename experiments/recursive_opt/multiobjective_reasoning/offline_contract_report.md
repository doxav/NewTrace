# Experiment 0 offline contract report

> **Status update (superseded blocker).** The GEPA contract blocker described below was
> fixed in `52a7b0bd` ("fix GEPA 0.1.4 public evaluator contract"): the production
> evaluator now returns the public two-item `(score, side_info)` pair, and the required
> CI job observed that fix green (run `32583433295`). See
> `artifacts/control_plane_v2/gepa_014_contract_hotfix.md`.
>
> **Correction, 2026-08-30.** An earlier annotation here said Experiment 0 had not run.
> That is now out of date: a frozen main run was authorised on 2026-08-24 and **started**,
> then stopped on its first unit (seed 0, budget 6, arm A) with
> `hard constraints not satisfied` — accuracy 22/24, invalid rate 1/24. See
> `reports/prompt18_r3_main_stop_decision.md`. The blocked state described *below* is
> historical and refers to the pre-fix tree.
>
> Two further conditions must be cleared before restarting from Phase 0:
> 1. the validity fixes D1/D3/D4/D6 changed the runtime tree again, so the recorded CI
>    run no longer certifies the current source (`prompt18_readiness.json` now reports
>    `ready_for_prompt_18: false` with explicit blockers);
> 2. the arm-comparability gate added with D3 must be satisfied by the experiment
>    design — see `artifacts/recursive_opt_assessment.md`.

Status at the time of writing: **BLOCKED — RETURN_TO_CONTROL_PLANE**

No Experiment 0 optimizer run or provider call was started. Phase 0 readiness evidence is valid, but the first real installed-GEPA boundary probe invalidated the GEPA arm before the experiment package was implemented.

## Frozen identity

- Git HEAD: `4b5bd3cc855b60430ec1b223a1db4319882c086f`
- Runtime tree SHA-256: `6315c6fc23d7f4e51effeb936f0b8c5938a36d821dd7b85346f7e2d8407ef07c`
- Readiness registry SHA-256: `f1a94b02c94607c2d22e6f10bce25b10d9b642814915ec01c72765280be26fa4`
- GEPA: `0.1.4`
- Required workflow run: `32569922958`, completed/success
- Required job: `97023835508`, `recursive-opt v2 offline (required)`, completed/success

## Passing preflight evidence

- Readiness matrix: 200 passed, zero skipped.
- Broader recursive-opt regression: 305 passed, two accepted optional telemetry skips.
- Installed LangGraph probe: 7 passed.
- Provider calls: 0.
- Monetary cost: USD 0.

The two skips and their exact classifications are recorded in `preflight_skips.json`.

## Blocking invariant failure

The production GEPA adapter defines its evaluator in `opto/features/recursive_opt/spec.py` as a three-item return:

```python
return (score, candidate, {"evaluation": info, "scores": info["metrics"]})
```

The installed `gepa.optimize_anything` 0.1.4 public evaluator contract accepts either a scalar score or a two-item `(score, side_info)` result. Its `EvaluatorWrapper` therefore fails while evaluating the seed candidate:

```text
ValueError: too many values to unpack (expected 2)
```

Production-path probe result:

```text
evaluator_mode=output
status=error
valid=False
error=ValueError: too many values to unpack (expected 2)
budget={'optimizer_llm_calls': 0, 'eval_llm_calls': 0,
        'candidates': 1, 'candidates_reserved': 1,
        'candidates_proposed': 1, 'candidates_evaluated': 1,
        'evaluator_runs': 1, 'total_tokens': 0}
```

This proves GEPA cannot produce a real proposal through the current production control-plane path. Candidate reporting also labels a seed evaluation as proposed/evaluated before any optimizer/reflection call, so it cannot support the required mechanistic reconciliation in this failed run.

## Minimal reproducer

```python
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

def evaluator(candidate, *, example, opt_state):
    return 0.0, candidate, {"evaluation": {"valid": True}}

config = GEPAConfig(
    engine=EngineConfig(
        max_candidate_proposals=1,
        max_metric_calls=8,
        display_progress_bar=False,
        parallel=False,
        capture_stdio=False,
    ),
    reflection=ReflectionConfig(reflection_lm=lambda _prompt: "```correct```"),
)
optimize_anything(
    seed_candidate={"planner": "wrong"},
    evaluator=evaluator,
    dataset=[{"id": "train"}],
    valset=[{"id": "validation"}],
    objective="Produce the expected value.",
    config=config,
)
```

The traceback terminates at `gepa/optimize_anything.py` in `wrapped_evaluator`:

```text
score, side_info = result
ValueError: too many values to unpack (expected 2)
```

## Why the green tests did not cover this boundary

- `test_22_gepa_externalizes_holdout` injects a fake `gepa_optimize` function that explicitly destructures the control plane's three-item tuple. It does not execute installed `optimize_anything`.
- `test_22b_real_gepa_public_contract_without_provider_calls` invokes `OptimizeAnythingAdapter.evaluate` directly. That lower-level adapter accepts a raw three-item evaluator result, bypassing the top-level `EvaluatorWrapper` that enforces the public two-item contract.

## Result validity and required next action

- Phase 0 lock/readiness observations: valid.
- Fixed/Trace/GEPA/Trace-no-validation experiment results: not run; none exist.
- Dataset selection, preregistration, live smoke, pilot, main run, statistics, and episode export: not started.
- Prompt 19 dataset readiness: false.

The next action is a separate control-plane hardening task. It must align the production GEPA evaluator with the installed top-level API, add a no-provider test that calls real `optimize_anything` end to end, verify reflection-client response compatibility, and rerun the full readiness/CI lock. This experiment must then restart from Phase 0 under the new source digests.
