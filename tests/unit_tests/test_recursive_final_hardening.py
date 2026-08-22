"""Causal tests for the Prompt 17.7 experiment-validity hardening."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from opto.features.recursive_opt import register_dataset, register_evaluator
from opto.features.recursive_opt import spec as S
from opto.optimizers.optimizer import Optimizer
from opto.trace import bundle, node
from opto.trace.modules import Module
from opto.trainer.objectives import EvaluationResult


_FORWARDS: list[Any] = []
_EVALUATED_OUTPUTS: list[Any] = []
_PROVIDER_REQUESTS: list[tuple[str, dict[str, Any]]] = []


@bundle()
def _traced_output(value: Any, example: Any, nonce: int) -> dict[str, Any]:
    """Create a stochastic output node causally downstream of one parameter."""
    return {"value": value, "example": example, "nonce": nonce}


class _CountingWorkflow(Module):
    """Count each workflow call and expose its exact traced output."""

    def __init__(self, value: str, forward_client: Any = None) -> None:
        self.value = node(value, name="value", trainable=True)
        self.forward_client = forward_client
        self.last_output: Any = None

    def forward(self, example: Any) -> Any:
        """Execute the optional forward client and one traced operation."""
        if self.forward_client is not None:
            self.forward_client(messages=[{"role": "user", "content": "forward"}])
        _FORWARDS.append(example)
        self.last_output = _traced_output(self.value, example, len(_FORWARDS))
        return self.last_output


class _ScriptedOptimizer(Optimizer):
    """Apply one deterministic update through the real Trace trainer."""

    def __init__(self, parameters: list[Any], **_kwargs: Any) -> None:
        super().__init__(parameters)

    def _step(self, *_args: Any, **_kwargs: Any) -> dict[Any, str]:
        """Set every exposed target to the expected value."""
        return {parameter: "correct" for parameter in self.parameters}


class _FeedbackOptimizer(Optimizer):
    """Expose the standard graph propagator for a direct feedback assertion."""

    def _step(self, *_args: Any, **_kwargs: Any) -> dict[Any, Any]:
        """Return no update because this test inspects feedback only."""
        return {}


class _FakeResponse:
    """Provider response with deterministic usage."""

    usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


class _FakeClient:
    """Record role requests and optionally fail to exercise fallback order."""

    def __init__(self, model: str, *, fail: bool = False) -> None:
        self.model = model
        self.fail = fail

    def __call__(self, *_args: Any, **kwargs: Any) -> _FakeResponse:
        """Record exact kwargs and return deterministic usage."""
        _PROVIDER_REQUESTS.append((self.model, copy.deepcopy(kwargs)))
        if self.fail:
            raise RuntimeError("configured fake-provider failure")
        return _FakeResponse()


def _build_counting(level: Mapping[str, Any], resources: Mapping[str, Any]) -> Module:
    """Build the counting workflow from its validated value and role client."""
    return _CountingWorkflow(
        str(level["module"]["config"]["value"]),
        resources.get("llm_clients", {}).get("forward"),
    )


def _snapshot_counting(module: Module) -> dict[str, Any]:
    """Snapshot the counting workflow's sole parameter."""
    return {"value": module.parameters()[0].data}


def _restore_counting(module: Module, artifact: Mapping[str, Any]) -> None:
    """Restore the counting workflow's sole parameter."""
    module.parameters()[0]._set(artifact["value"])


def _validate_counting(value: Mapping[str, Any]) -> None:
    """Validate counting config/artifacts shared shape."""
    if set(value) != {"value"} or not isinstance(value["value"], str):
        raise ValueError("counting values require one string 'value'")


def _output_evaluator(
    output: Any, example: Any, _context: Mapping[str, Any]
) -> EvaluationResult:
    """Score the exact output node supplied by the output evaluator contract."""
    _EVALUATED_OUTPUTS.append(output)
    data = output.data
    expected = example.get("expected", data["value"])
    return EvaluationResult(
        valid=True,
        status="ok",
        metrics={"accuracy": float(data["value"] == expected)},
        trace={"nonce": data["nonce"]},
    )


def _legacy_evaluator(
    module: Module, dataset: Any, _context: Mapping[str, Any]
) -> EvaluationResult:
    """Exercise explicitly labelled module-mode compatibility."""
    output = module(dataset[0])
    return EvaluationResult(
        valid=True, status="ok", metrics={"accuracy": float(output.data["value"] == "correct")}
    )


def _dataset(split: str, _config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Resolve one exact example for provenance and resume tests."""
    return [{"split": split, "expected": "correct"}]


@pytest.fixture(autouse=True)
def _contracts() -> Any:
    """Register stable hardening contracts and clear observable events."""
    entry = S.ModuleRegistryEntry(
        build=_build_counting,
        snapshot=_snapshot_counting,
        restore=_restore_counting,
        validate_artifact=_validate_counting,
        capabilities=frozenset({"trace_module", "json_snapshot"}),
        validate_config=_validate_counting,
    )
    S.register_module("tests.module.counting@1", entry)
    register_evaluator("tests.evaluator.output@1", _output_evaluator, mode="output")
    register_evaluator("tests.evaluator.legacy@1", _legacy_evaluator, mode="legacy_module")
    register_dataset("tests.dataset.hardening@1", _dataset)
    _FORWARDS.clear()
    _EVALUATED_OUTPUTS.clear()
    _PROVIDER_REQUESTS.clear()
    yield


def _spec(engine: str = "fixed") -> dict[str, Any]:
    """Return one complete offline counting-workflow spec."""
    engine_config: dict[str, Any]
    if engine == "trace":
        engine_config = {
            "optimizer": "OptoPrimeV2",
            "trainer": "PrioritySearch",
            "iterations": 1,
            "num_candidates": 1,
            "validation_gate": True,
        }
    elif engine == "gepa_optimize_anything":
        engine_config = {"engine": {"max_candidate_proposals": 1, "max_metric_calls": 4}}
    else:
        engine_config = {}
    return {
        "schema_version": S.SCHEMA_VERSION,
        "kind": S.SPEC_KIND,
        "runtime": {"offline": True, "seed": 7},
        "levels": [
            {
                "id": "level-a",
                "surface": {"kind": "module", "targets": ["value"]},
                "module": {
                    "ref": "tests.module.counting@1",
                    "config": {"value": "wrong"},
                    "inputs": {},
                },
                "engine": {"name": engine, "config": engine_config},
                "objective": {
                    "evaluator_ref": "tests.evaluator.output@1",
                    "intent": "Produce the expected value.",
                    "metrics": {
                        "accuracy": {
                            "direction": "maximize",
                            "source": "evaluation.metrics.accuracy",
                            "aggregate_examples": "mean",
                        }
                    },
                    "selection": {"mode": "scalar", "score_key": "accuracy"},
                },
                "datasets": {
                    "train": [{"expected": "correct"}],
                    "validation": [{"expected": "correct"}],
                    "holdout": [{"expected": "correct"}],
                },
            }
        ],
    }


def _fake_gepa(**kwargs: Any) -> Any:
    """Evaluate one seed candidate and return it as the GEPA best candidate."""
    kwargs["evaluator"](
        kwargs["seed_candidate"], example=kwargs["dataset"][0], opt_state={}
    )
    return SimpleNamespace(best_candidate=kwargs["seed_candidate"])


@pytest.mark.parametrize("engine", ["fixed", "trace", "gepa_optimize_anything"])
def test_each_engine_forwards_exactly_once_per_evaluated_example(engine: str) -> None:
    """Fixed, Trace, and GEPA must never evaluate by rerunning the workflow."""
    raw = _spec(engine)
    resources: dict[str, Any] = {}
    if engine == "trace":
        raw["runtime"]["test_mode"] = True
        resources["optimizer"] = _ScriptedOptimizer
    elif engine == "gepa_optimize_anything":
        raw["runtime"]["test_mode"] = True
        resources.update({"gepa_optimize": _fake_gepa, "gepa_config": object()})

    result = S.run_spec(raw, resources=resources)

    assert len(_FORWARDS) == len(_EVALUATED_OUTPUTS) > 0
    assert all(output.data["nonce"] == index for index, output in enumerate(_EVALUATED_OUTPUTS, 1))
    assert result.engine == engine


def test_scored_output_is_exact_trace_anchor_and_propagates_feedback() -> None:
    """Evaluation and optimizer feedback share the actual stochastic output node."""
    raw = _spec("trace")
    plan = S.compile_plan(raw)
    objective = S.compile_objective(
        plan.units[0].levels[0].spec["objective"], capabilities={"scalar"}
    )
    module = _CountingWorkflow("wrong")
    wrapped = S._EvaluatedModule(
        module,
        S._evaluator_entry("tests.evaluator.output@1"),
        objective,
        {},
        S._BudgetGuard(plan.spec["budget"]),
        set(),
        [],
    )

    result = wrapped.forward({"expected": "wrong"})

    assert _EVALUATED_OUTPUTS == [module.last_output]
    assert result.data["trace"]["nonce"] == module.last_output.data["nonce"]
    assert module.last_output in result.parents
    assert module.value in result.parameter_dependencies
    _FeedbackOptimizer([module.value]).backward(result, "preserve exact scored behavior")
    assert module.value.feedback


def test_forward_role_calls_and_tokens_equal_workflow_forwards() -> None:
    """One forward request and its tokens are charged exactly once."""
    raw = _spec("fixed")
    raw["runtime"]["test_mode"] = True
    raw["llm_profiles"] = {
        "forward": {"provider": "fake", "model": "fake/forward", "max_tokens": 8}
    }
    raw["levels"][0]["llm_roles"] = {"forward": "forward"}

    result = S.run_spec(
        raw,
        resources={
            "llm_factory": lambda profile, _role: _FakeClient(profile["resolved_model"])
        },
    )

    assert len(_PROVIDER_REQUESTS) == len(_FORWARDS) == 1
    assert result.usage["forward"]["calls"] == 1
    assert result.usage["forward"]["total_tokens"] == 5
    assert result.budget["accounted"]["total_tokens"] == 5


def test_explicit_legacy_evaluator_is_nonportable_compatibility() -> None:
    """Legacy module-mode evaluators run only when explicitly nonportable."""
    raw = _spec("fixed")
    raw["levels"][0]["module"]["config"]["value"] = "correct"
    raw["levels"][0]["objective"]["evaluator_ref"] = "tests.evaluator.legacy@1"
    with pytest.raises(ValueError, match="requires an output evaluator"):
        S.compile_plan(raw)

    raw["runtime"]["test_mode"] = True
    result = S.run_spec(raw)

    assert result.valid is True
    assert result.portable is False and result.promotable is False
    assert len(_FORWARDS) == 1


def test_canonical_trace_ignores_all_hidden_behavior_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conflicting demo environment controls cannot affect canonical v2."""
    optimize_module = importlib.import_module("opto.features.recursive_opt.optimize")
    captures: list[dict[str, Any]] = []

    def fake_train(**kwargs: Any) -> Any:
        """Capture the fully resolved trainer call without provider work."""
        captures.append(kwargs)
        return SimpleNamespace(memory=SimpleNamespace(memory=[]))

    monkeypatch.setattr(optimize_module, "_train_returning_trainer", fake_train)
    monkeypatch.setattr(S.time, "monotonic", lambda: 100.0)
    raw = _spec("trace")
    raw["levels"][0]["engine"]["config"].update(
        {"optimizer_kwargs": {"explicit_optimizer": 1}, "trainer_kwargs": {"explicit_trainer": 2}}
    )
    environments = [
        {
            "RECURSIVE_OPT_TRAINER": "FirstConflict",
            "RECURSIVE_OPT_OPTIMIZER": "FirstOptimizer",
            "RECURSIVE_OPT_ITERATIONS": "91",
            "RECURSIVE_OPT_NUM_CANDIDATES": "81",
            "RECURSIVE_OPT_OPTIMIZER_KWARGS": '{"hidden": 1}',
            "RECURSIVE_OPT_TRAINER_KWARGS": '{"hidden": 2}',
            "RECURSIVE_OPT_LLM_PROFILES": '{"hidden": {"backend": "LiteLLM"}}',
            "RECURSIVE_OPT_MODEL": "hidden/first",
            "TRACE_LITELLM_MODEL": "hidden/trace-first",
        },
        {
            "RECURSIVE_OPT_TRAINER": "SecondConflict",
            "RECURSIVE_OPT_OPTIMIZER": "SecondOptimizer",
            "RECURSIVE_OPT_ITERATIONS": "92",
            "RECURSIVE_OPT_NUM_CANDIDATES": "82",
            "RECURSIVE_OPT_OPTIMIZER_KWARGS": '{"other": 3}',
            "RECURSIVE_OPT_TRAINER_KWARGS": '{"other": 4}',
            "RECURSIVE_OPT_LLM_PROFILES": '{"other": {"backend": "LiteLLM"}}',
            "RECURSIVE_OPT_MODEL": "hidden/second",
            "TRACE_LITELLM_MODEL": "hidden/trace-second",
        },
    ]
    results = []
    fingerprints = []
    for environment in environments:
        _FORWARDS.clear()
        _EVALUATED_OUTPUTS.clear()
        for name, value in environment.items():
            monkeypatch.setenv(name, value)
        fingerprints.append(S.normalize_spec(raw)["fingerprint"])
        results.append(S.run_spec(raw).to_dict())

    resolved = [
        {
            "algorithm": call["algorithm"],
            "optimizer": call["optimizer"],
            "iterations": call["num_steps"],
            "candidates": call["num_candidates"],
            "optimizer_kwargs": call["optimizer_kwargs"],
            "explicit_trainer": call["explicit_trainer"],
        }
        for call in captures
    ]
    assert fingerprints[0] == fingerprints[1]
    assert resolved[0] == resolved[1] == {
        "algorithm": "PrioritySearch",
        "optimizer": "OptoPrimeV2",
        "iterations": 1,
        "candidates": 1,
        "optimizer_kwargs": {"explicit_optimizer": 1},
        "explicit_trainer": 2,
    }
    assert results[0] == results[1]


def test_request_params_are_fingerprinted_manifested_and_causal(tmp_path: Path) -> None:
    """Changing request params changes only identity and exact provider kwargs."""
    raw = _spec("fixed")
    raw["runtime"]["test_mode"] = True
    raw["outputs"] = {"directory": str(tmp_path)}
    raw["llm_profiles"] = {
        "primary": {
            "provider": "fake",
            "model": "fake/primary",
            "fallbacks": ["fallback"],
            "temperature": 0.2,
            "max_tokens": 8,
            "request_params": {"reasoning": {"enabled": True}, "top_p": 0.8},
        },
        "fallback": {
            "provider": "fake",
            "model": "fake/fallback",
            "request_params": {"reasoning": {"enabled": False}},
        },
    }
    raw["levels"][0]["llm_roles"] = {"forward": "primary"}

    def factory(profile: Mapping[str, Any], _role: str) -> _FakeClient:
        """Fail the primary so both per-fallback request controls are observed."""
        return _FakeClient(profile["resolved_model"], fail=profile["model"] == "fake/primary")

    normalized = S.normalize_spec(raw)
    result = S.run_spec(raw, resources={"llm_factory": factory})
    manifest = json.loads(
        (tmp_path / result.plan_fingerprint / "resolved_execution_plan.json").read_text()
    )
    changed = copy.deepcopy(raw)
    changed["llm_profiles"]["primary"]["request_params"]["reasoning"]["enabled"] = False
    changed_normalized = S.normalize_spec(changed)

    assert normalized["fingerprint"] != changed_normalized["fingerprint"]
    assert normalized["llm_profiles"]["primary"]["model"] == changed_normalized["llm_profiles"]["primary"]["model"]
    assert [request[1]["reasoning"]["enabled"] for request in _PROVIDER_REQUESTS] == [True, False]
    assert _PROVIDER_REQUESTS[0][1]["temperature"] == 0.2
    assert _PROVIDER_REQUESTS[0][1]["max_tokens"] == 8
    roles = manifest["units"][0]["levels"][0]["llm_roles"]
    assert roles["forward"]["request_params"]["reasoning"]["enabled"] is True
    assert normalized["llm_profiles"]["fallback"]["request_params"] == {"reasoning": {"enabled": False}}


@pytest.mark.parametrize(
    "request_params",
    [
        {"model": "hidden"},
        {"provider": "hidden"},
        {"api_key": "secret"},
        {"credential_ref": "env:HIDDEN"},
        {"base_url": "https://hidden.invalid"},
        {"nested": [{"token": "secret"}]},
    ],
)
def test_request_params_reject_identity_and_secret_overrides(
    request_params: dict[str, Any],
) -> None:
    """Request parameters cannot bypass normalized provider identity."""
    raw = _spec()
    raw["llm_profiles"] = {
        "forward": {"provider": "fake", "model": "fake/exact", "request_params": request_params}
    }
    raw["levels"][0]["llm_roles"] = {"forward": "forward"}
    with pytest.raises(ValueError, match="may not override|secret value"):
        S.normalize_spec(raw)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_request_params_reject_nonfinite_numbers(value: float) -> None:
    """Request controls remain strict JSON even for Python float edge cases."""
    raw = _spec()
    raw["llm_profiles"] = {
        "forward": {
            "provider": "fake",
            "model": "fake/exact",
            "request_params": {"top_p": value},
        }
    }
    raw["levels"][0]["llm_roles"] = {"forward": "forward"}
    with pytest.raises(ValueError, match="finite JSON numbers"):
        S.normalize_spec(raw)


def test_code_provenance_manifest_and_registry_change_invalidate_resume(tmp_path: Path) -> None:
    """Source changes behind an identical ref force reevaluation, not stale resume."""
    raw = _spec("fixed")
    raw["runtime"]["resume"] = True
    raw["outputs"] = {"directory": str(tmp_path)}
    raw["levels"][0]["datasets"] = {
        split: {"ref": "tests.dataset.hardening@1", "split": split, "config": {}}
        for split in ("train", "validation", "holdout")
    }
    first = S.run_spec(raw)
    root = tmp_path / first.plan_fingerprint
    first_manifest = json.loads((root / "resolved_execution_plan.json").read_text())
    first_provenance = first_manifest["code_provenance"]
    old_entry = S._EVALUATOR_REGISTRY["tests.evaluator.output@1"]

    def changed_evaluator(
        output: Any, _example: Any, _context: Mapping[str, Any]
    ) -> EvaluationResult:
        """Return a changed score while retaining the identical versioned ref."""
        _EVALUATED_OUTPUTS.append(output)
        return EvaluationResult(valid=True, status="ok", metrics={"accuracy": 0.0})

    S._EVALUATOR_REGISTRY["tests.evaluator.output@1"] = S._EvaluatorEntry(
        changed_evaluator, "output"
    )
    _EVALUATED_OUTPUTS.clear()
    try:
        second = S.run_spec(raw)
    finally:
        S._EVALUATOR_REGISTRY["tests.evaluator.output@1"] = old_entry

    second_provenance = json.loads((root / "resolved_execution_plan.json").read_text())[
        "code_provenance"
    ]
    assert _EVALUATED_OUTPUTS
    assert second.evaluation.metrics["accuracy"] == 0.0
    assert first_provenance["runtime_tree_sha256"] == second_provenance["runtime_tree_sha256"]
    assert first_provenance["registry_sha256"] != second_provenance["registry_sha256"]
    for kind in ("modules", "evaluators", "datasets", "codecs", "engines"):
        assert first_provenance["entries"][kind]
        for record in first_provenance["entries"][kind]:
            assert set(record) == {
                "ref",
                "python_module",
                "qualified_name",
                "source_file",
                "source_sha256",
                "package_version",
                "implementation_sha256",
            }
            assert len(record["source_sha256"]) == len(record["implementation_sha256"]) == 64


def test_candidate_accounting_separates_reservations_and_observations() -> None:
    """Planned capacity is never reported as observed candidate evaluation."""
    raw = _spec("gepa_optimize_anything")
    raw["runtime"]["test_mode"] = True
    result = S.run_spec(
        raw, resources={"gepa_optimize": _fake_gepa, "gepa_config": object()}
    )
    accounted = result.budget["accounted"]

    assert accounted["candidates"] == accounted["candidates_reserved"] == 1
    assert accounted["candidates_proposed"] == 1
    assert accounted["candidates_evaluated"] == 1
    assert accounted["evaluator_runs"] > accounted["candidates_evaluated"]


def test_required_workflow_has_gepa_dependency_and_hardening_matrix() -> None:
    """The required offline job installs and executes its complete GEPA contract."""
    workflow = Path(".github/workflows/recursive-opt-v2.yml").read_text()
    required_job = workflow.split("jobs:", 1)[1]

    assert "python -m pip install -e '.[gepa]'" in required_job
    assert "tests/unit_tests/test_recursive_final_hardening.py" in required_job
    assert "if: github.event_name == 'workflow_dispatch'" not in required_job


def test_readiness_uses_source_digests_without_sha_environment() -> None:
    """A checkout verifies readiness provenance without a self-referential SHA."""
    readiness = json.loads(
        Path("artifacts/control_plane_v2/prompt18_readiness.json").read_text()
    )
    smoke = json.loads(
        Path("artifacts/control_plane_v2/golden_specs/uc4_positive.normalized.json").read_text()
    )
    provenance = S.compile_plan(smoke).code_provenance

    assert readiness["schema_version"] == "prompt18-readiness/v2"
    assert "final_sha" not in readiness
    assert readiness["verified_runtime_tree_sha256"] == provenance["runtime_tree_sha256"]
    assert readiness["verified_registry_sha256"] == provenance["registry_sha256"]
    assert readiness["gates"]["required_gepa_ci"] is True
    assert readiness["ready_for_prompt_18"] is True
    assert readiness["blockers"] == []
    assert readiness["required_ci_run"] == {
        "id": 32583433295,
        "job_id": 97056076300,
        "job": "recursive-opt v2 offline (required)",
        "head_sha": "52a7b0bd86b21975e2de09cec0a957b04e835312",
        "status": "completed",
        "conclusion": "success",
        "url": "https://github.com/doxav/NewTrace/actions/runs/32583433295",
    }
