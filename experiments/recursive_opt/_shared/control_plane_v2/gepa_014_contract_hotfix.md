# GEPA 0.1.4 public evaluator contract hotfix

## Pre-patch reproduction

- Branch/HEAD inspected: `recursive_opt` at `82d37beca437135a56e76502f1595d63fec4bf84`.
- Installed distribution: `gepa==0.1.4` in the `humanllm` environment.
- `optimize_anything` signature: `(seed_candidate: str | dict[str, str] | None = None, *, evaluator: Callable[..., Any] | None = None, batch_evaluator: Callable[..., Any] | None = None, dataset: list[DataInst] | None = None, valset: list[DataInst] | None = None, objective: str | None = None, background: str | None = None, config: GEPAConfig | None = None) -> GEPAResult`.
- `EvaluatorWrapper` signature: `(evaluator_fn: Callable[..., Any], single_instance_mode: bool, capture_stdio: bool = False, str_candidate_mode: bool = False, raise_on_exception: bool = True) -> None`.
- `EvaluatorWrapper.__call__` signature: `(self, candidate: dict[str, str], example: object | None = None, **kwargs: Any) -> tuple[float, Any, dict[str, Any]]`.

The requested `from gepa.gepa_launcher import EvaluatorWrapper` probe raised `ModuleNotFoundError: No module named 'gepa.gepa_launcher'`: the published 0.1.4 wheel defines and uses `EvaluatorWrapper` in `gepa.optimize_anything`. Repeating the probe against that actual wrapper produced the exact contract result:

```text
prescribed_import_error: ModuleNotFoundError: No module named 'gepa.gepa_launcher'
gepa_version: 0.1.4
actual_wrapper_import: gepa.optimize_anything
bad_result: ValueError: too many values to unpack (expected 2)
good_result: {'score': 1.0, 'output': None, 'side_info': {'diagnostic': 'x'}}
```

The public contract is `score | (score, side_info)`. `EvaluatorWrapper` converts it to the internal `(score, output, side_info)` triple, with `output is None`, for `OptimizeAnythingAdapter`.

## Root cause

Before `82d37bec`, `_run_gepa_engine` passed a public evaluator to `optimize_anything` that returned `(score, candidate, side_info)`. GEPA 0.1.4's public wrapper tried to unpack that tuple as `(score, side_info)` and failed. The installed-GEPA test directly instantiated `OptimizeAnythingAdapter` with the raw 3-tuple evaluator, so it tested the internal adapter contract rather than the public `optimize_anything` evaluator contract and gave a false green signal.

## Hotfix and verification

The nested public callback now returns exactly `float(score), info`; it does not
return the candidate and does not populate GEPA's reserved `side_info["scores"]`.
The callback annotation matches the public pair. Production runtime remains
8,850 physical lines (`spec.py`: 2,732), a net delta of zero lines.

Tests now preserve the actual layering:

```text
recursive-opt public evaluator pair
  -> EvaluatorWrapper
  -> internal (score, output, side_info) triple
  -> OptimizeAnythingAdapter
```

The actual public `optimize_anything()` smoke uses one seed, one train example,
one validation example, `max_metric_calls=1`, `max_candidate_proposals=0`, a
local deterministic reflection callable, no provider string or key, and blocked
external sockets. It terminates after the seed validation evaluation without
reaching reflection.

The weighted regression holds accuracy constant and changes
`forward_token_ratio` from 0.5 to 1.5. The lower ratio produces the higher
projected scalar score. Both public side-info mappings retain raw metrics under
`metrics`, neither contains `scores`, and an invalid candidate with an
arbitrarily high accuracy still receives `-1_000_000_000_000.0` with
`valid is False`.

Provider-free/network-blocked local results in conda `humanllm`:

- focused GEPA seam: `4 passed, 41 deselected in 1.59s`;
- control-plane v2: `45 passed in 10.20s`;
- final hardening: `21 passed in 2.61s`;
- recursive spec: `47 passed in 2.89s`;
- objectives/vector/multi-objective: `89 passed in 2.92s`;
- all recursive unit files: `224 passed, 2 skipped, 1 warning in 16.78s`;
- complete unit suite: `487 passed, 3 skipped, 1 warning in 39.26s`;
- clean-kernel notebook: `1 passed, 44 deselected in 4.47s`;
- Ruff on both changed Python files: `All checks passed!`;
- `git diff --check` and `git diff --cached --check`: passed.

The two common skips require optional graph/telemetry backends. The
complete-suite-only third skip requires the Graphviz `dot` executable. No GEPA
test skipped. The warning is the existing LangGraph serializer-default
deprecation. No provider or paid call occurred.

Production-computed authoritative digests:

- `runtime_tree_sha256=5b460d771ca0b0f9bd914b2c8330860e6f5771a8447d40e50db0d554986e0642`
- `registry_sha256=f1a94b02c94607c2d22e6f10bce25b10d9b642814915ec01c72765280be26fa4`

Required CI remains unclaimed until the pushed hotfix is observed green.
Accordingly, `required_gepa_ci=false` and `ready_for_prompt_18=false` at this
pre-push checkpoint.

## Required CI outcome

- Hotfix commit: `52a7b0bd86b21975e2de09cec0a957b04e835312`.
- GitHub Actions run: `32583433295`.
- Required job: `recursive-opt v2 offline (required)` (`97056076300`).
- Result: `completed / success` in 1m07s.
- URL: <https://github.com/doxav/NewTrace/actions/runs/32583433295>.

The required job installed `.[gepa]` and passed the public-contract test through
the actual GEPA 0.1.4 package. The readiness gate is therefore promoted to
`required_gepa_ci=true` and `ready_for_prompt_18=true`.
