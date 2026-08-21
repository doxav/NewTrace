"""Offline contract tests for the recursive-opt v2alpha control plane."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from opto.features.recursive_opt import spec as S
from opto.features.recursive_opt.memory import KnowledgeCard, MemoryLite
from opto.trace.modules import Module
from opto.trainer import objectives as O


def _portable_spec() -> dict[str, Any]:
    """Return the smallest portable v2alpha run specification."""
    return {
        "schema_version": "recursive-opt/v2alpha",
        "kind": "recursive_optimization",
        "module": {
            "ref": "recursive_opt.module.reasoning_workflow@1",
            "config": {"components": {"planner": "answer carefully"}},
        },
        "objective": {
            "intent": "Answer correctly with low latency.",
            "metrics": ["accuracy", "latency_ms"],
            "directions": {"accuracy": "maximize", "latency_ms": "minimize"},
            "selection": {"mode": "weighted", "weights": {"accuracy": 1.0}},
        },
        "llm_profiles": {"main": {"provider": "openrouter"}},
        "llm_roles": {"forward": "main", "judge": "main"},
    }


def test_normalize_spec_materializes_defaults_and_is_json_round_trippable() -> None:
    normalized = S.normalize_spec(_portable_spec())

    assert normalized["schema_version"] == "recursive-opt/v2alpha"
    assert normalized["kind"] == "recursive_optimization"
    assert set(S.CANONICAL_SPEC_BLOCKS).issubset(normalized)
    assert len(normalized["fingerprint"]) == 64
    assert json.loads(json.dumps(normalized))["fingerprint"] == normalized["fingerprint"]
    assert normalized["llm_profiles"]["main"]["model"] == "deepseek/deepseek-v4-flash-0731"
    assert normalized["llm_profiles"]["main"]["resolved_model"] == (
        "openrouter/deepseek/deepseek-v4-flash-0731"
    )
    assert normalized["llm_profiles"]["main"]["api_key_ref"] == "env:OPENROUTER_API_KEY"
    assert set(normalized["llm_roles"]) == {"forward", "optimizer", "feedback", "judge"}

    with pytest.raises(TypeError):
        normalized["budget"]["candidates"] = 1


def test_normalize_spec_fingerprint_is_stable_across_order_and_round_trip() -> None:
    raw = _portable_spec()
    reordered = {key: raw[key] for key in reversed(raw)}

    first = S.normalize_spec(raw)
    second = S.normalize_spec(reordered)
    third = S.normalize_spec(json.loads(json.dumps(first)))

    assert first["fingerprint"] == second["fingerprint"] == third["fingerprint"]


def test_normalize_spec_rejects_unknown_keys_except_namespaced_extensions() -> None:
    with pytest.raises(ValueError, match="unknown spec keys"):
        S.normalize_spec({**_portable_spec(), "typo": True})

    bad_nested = _portable_spec()
    bad_nested["objective"]["implicit_average"] = True
    with pytest.raises(ValueError, match="objective"):
        S.normalize_spec(bad_nested)

    extended = _portable_spec()
    extended["extensions"] = {"acme.audit": {"ticket": "CP-2"}}
    assert S.normalize_spec(extended)["extensions"]["acme.audit"]["ticket"] == "CP-2"

    unnamespaced = _portable_spec()
    unnamespaced["extensions"] = {"audit": {}}
    with pytest.raises(ValueError, match="namespace"):
        S.normalize_spec(unnamespaced)


def test_normalize_spec_rejects_callables_secrets_and_arbitrary_import_refs() -> None:
    callable_spec = _portable_spec()
    callable_spec["module"]["config"]["builder"] = lambda: None
    with pytest.raises(TypeError, match="callable"):
        S.normalize_spec(callable_spec)

    secret_spec = _portable_spec()
    secret_spec["llm_profiles"]["main"]["api_key"] = "not-portable"
    with pytest.raises(ValueError, match="secret"):
        S.normalize_spec(secret_spec)

    import_spec = _portable_spec()
    import_spec["module"]["ref"] = "some_package.module:builder"
    with pytest.raises(ValueError, match="versioned registry ref"):
        S.normalize_spec(import_spec)


def test_migrate_legacy_spec_preserves_supported_behavioral_controls() -> None:
    legacy = {
        "families": {"reasoning": ["fake:reasoning"]},
        "levels": [
            {
                "id": "reason",
                "surface": "config",
                "family": "reasoning",
                "targets": ["batch_size"],
                "fixed": {"batch_size": 4},
                "iterations": 2,
            }
        ],
        "budget": {"candidates": 5},
        "memory_root": "./memory",
        "reuse_priors": True,
        "trainer_kwargs": {"num_threads": 1},
        "run_id": "legacy-run",
    }

    migrated = S.migrate_legacy_spec(legacy)
    normalized = S.normalize_spec(legacy)

    assert migrated["schema_version"] == "recursive-opt/v2alpha"
    assert normalized["surface"]["levels"][0]["id"] == "reason"
    assert list(normalized["module"]["config"]["families"]["reasoning"]) == ["fake:reasoning"]
    assert normalized["budget"]["candidates"] == 5
    assert normalized["runtime"]["memory_root"] == "./memory"
    assert normalized["runtime"]["reuse_priors"] is True
    assert normalized["runtime"]["run_id"] == "legacy-run"


def test_normalize_spec_rejects_a_stale_fingerprint() -> None:
    normalized = json.loads(json.dumps(S.normalize_spec(_portable_spec())))
    normalized["budget"]["candidates"] = 99

    with pytest.raises(ValueError, match="fingerprint"):
        S.normalize_spec(normalized)


def test_module_registry_builds_snapshots_and_restores_multi_component_module() -> None:
    normalized = S.normalize_spec(_portable_spec())
    module = S.build_module(normalized)

    assert isinstance(module, Module)
    assert {parameter.name.split(":")[0] for parameter in module.parameters()} == {"planner"}

    initial = S.snapshot_module(normalized, module)
    assert initial == {"components": {"planner": "answer carefully"}}

    S.restore_module(normalized, module, {"components": {"planner": "verify every step"}})
    assert S.snapshot_module(normalized, module)["components"]["planner"] == "verify every step"
    assert module({"question": "2+2"})["components"]["planner"] == "verify every step"

    with pytest.raises(ValueError, match="component keys"):
        S.restore_module(normalized, module, {"components": {"critic": "new key"}})


def test_compile_plan_resolves_refs_and_expands_experiment_axes() -> None:
    raw = _portable_spec()
    raw["experiment"] = {
        "seeds": [3, 7],
        "arms": [
            {"id": "fixed", "engine": {"name": "fixed"}},
            {"id": "trace", "engine": {"name": "trace"}},
        ],
        "matrix": {"budget.candidates": [2, 4]},
    }

    plan = S.compile_plan(raw)
    explanation = plan.explain()

    assert len(plan.units) == 8
    assert explanation["execution_units"] == 8
    assert explanation["engines"] == ["fixed", "trace"]
    assert {unit.seed for unit in plan.units} == {3, 7}
    assert {unit.spec["budget"]["candidates"] for unit in plan.units} == {2, 4}
    assert all(len(unit.spec["fingerprint"]) == 64 for unit in plan.units)

    with pytest.raises(TypeError):
        plan.units[0].spec["engine"]["name"] = "other"


def test_compile_plan_rejects_unregistered_module_ref() -> None:
    raw = _portable_spec()
    raw["module"]["ref"] = "acme.module.missing@1"

    with pytest.raises(ValueError, match="unregistered module ref"):
        S.compile_plan(raw)


def test_normalize_evaluation_result_adapts_legacy_shapes_without_sentinels() -> None:
    negative = O.normalize_evaluation_result(-0.25)
    vector = O.normalize_evaluation_result({"accuracy": 0.8, "latency_ms": 40.0})
    verbal = O.normalize_evaluation_result((0.7, "The answer needs a citation."))
    invalid = O.normalize_evaluation_result({
        "valid": False,
        "status": "invalid",
        "metrics": {"score": 1.0},
        "feedback": "constraint failed",
    })

    assert negative.valid is True
    assert negative.metrics == {"score": -0.25}
    assert vector.metrics == {"accuracy": 0.8, "latency_ms": 40.0}
    assert verbal.feedback == "The answer needs a citation."
    assert invalid.valid is False
    assert invalid.metrics["score"] == 1.0
    assert set(negative.usage) == {"forward", "optimizer", "feedback", "judge"}

    with pytest.raises(ValueError, match="ObjectiveConfig"):
        O.to_scalar_score(vector.metrics, None)


def test_compile_objective_supports_scalar_weighted_and_pareto_capabilities() -> None:
    scalar = S.compile_objective(
        S.normalize_spec(_portable_spec())["objective"],
        capabilities={"scalar", "weighted", "pareto"},
    )
    assert scalar["config"].mode == "weighted"
    assert scalar["config"].minimize == frozenset({"latency_ms"})

    raw = _portable_spec()
    raw["objective"]["selection"] = {
        "mode": "pareto",
        "pareto_metrics": ["accuracy", "latency_ms"],
        "tie_break": "lexicographic",
    }
    pareto = S.compile_objective(
        S.normalize_spec(raw)["objective"],
        capabilities={"scalar", "weighted", "pareto"},
    )
    assert pareto["config"].mode == "pareto"

    with pytest.raises(ValueError, match="does not support objective mode 'pareto'"):
        S.compile_objective(
            S.normalize_spec(raw)["objective"],
            capabilities={"scalar", "weighted"},
        )


def test_hard_constraints_filter_candidates_before_weighted_selection() -> None:
    raw = _portable_spec()
    raw["objective"]["hard_constraints"] = [
        {"metric": "latency_ms", "op": "<=", "value": 100.0}
    ]
    compiled = S.compile_objective(
        S.normalize_spec(raw)["objective"],
        capabilities={"scalar", "weighted", "pareto"},
    )
    fastest_valid = O.normalize_evaluation_result(
        ({"accuracy": 0.7, "latency_ms": 20.0}, "fast and acceptable")
    )
    highest_but_infeasible = O.normalize_evaluation_result(
        ({"accuracy": 0.99, "latency_ms": 150.0}, "too slow")
    )

    selected = O.select_evaluation_result(
        [highest_but_infeasible, fastest_valid],
        compiled["config"],
        compiled["hard_constraints"],
    )

    assert selected.feedback == "fast and acceptable"


def test_llm_roles_materialize_overrides_and_preflight_exact_models() -> None:
    raw = _portable_spec()
    raw["llm_roles"]["optimizer"] = {
        "profile": "main",
        "temperature": 0.2,
        "max_tokens": 512,
    }
    normalized = S.normalize_spec(raw)

    assert normalized["llm_roles"]["forward"]["profile"] == "main"
    assert normalized["llm_roles"]["forward"]["resolved_model"] == (
        "openrouter/deepseek/deepseek-v4-flash-0731"
    )
    assert normalized["llm_roles"]["optimizer"]["temperature"] == 0.2
    assert normalized["llm_roles"]["feedback"] is None

    level_roles = S.resolve_llm_roles(
        normalized,
        {"forward": None, "judge": {"profile": "main", "temperature": 0.0}},
    )
    assert level_roles["forward"] is None
    assert level_roles["judge"]["temperature"] == 0.0

    checked: list[str] = []
    S.preflight_llm_profiles(normalized, checker=checked.append)
    assert checked == ["openrouter/deepseek/deepseek-v4-flash-0731"]


def test_runtime_llm_usage_is_collected_once_by_role() -> None:
    from opto.features.recursive_opt.runmode import track_llm_usage

    class _Response:
        usage = {
            "prompt_tokens": 11,
            "completion_tokens": 5,
            "total_tokens": 16,
            "cost_usd": 0.003,
        }

    class _FakeLLM:
        def __call__(self, *_args: Any, **_kwargs: Any) -> _Response:
            return _Response()

    usage: dict[str, dict[str, float | int]] = {}
    llm = track_llm_usage(_FakeLLM(), "optimizer", usage)

    llm(messages=[])
    llm(messages=[])
    result = O.normalize_evaluation_result({"metrics": {"score": 1.0}, "usage": usage})

    assert result.usage["optimizer"] == {
        "calls": 2,
        "prompt_tokens": 22,
        "completion_tokens": 10,
        "total_tokens": 32,
        "cost_usd": pytest.approx(0.006),
    }
    assert result.usage["forward"]["total_tokens"] == 0

    with pytest.raises(ValueError, match="role"):
        track_llm_usage(_FakeLLM(), "other", usage)


def test_knowledge_retrieval_is_promoted_only_scoped_and_rollback_safe(tmp_path: Any) -> None:
    memory = MemoryLite(root=str(tmp_path))
    promoted = KnowledgeCard(
        claim="Short verification improves arithmetic accuracy.",
        scope={"family": "reasoning", "runtime": "trace"},
        preconditions=["arithmetic task"],
        recommended_action="Verify the final numeric result.",
        evidence_refs=["run:1"],
        counterevidence_refs=[],
        support=3,
        uncertainty=0.1,
        status="promoted",
        runtime_compatibility={"engines": ["trace"]},
        supersedes=[],
    )
    candidate = KnowledgeCard(
        claim="Use a longer chain of thought.",
        scope={"family": "reasoning", "runtime": "trace"},
        preconditions=[],
        recommended_action="Add more steps.",
        evidence_refs=["run:2"],
        counterevidence_refs=[],
        support=1,
        uncertainty=0.8,
        status="candidate",
        runtime_compatibility={"engines": ["trace"]},
        supersedes=[],
    )
    other_scope = KnowledgeCard(
        claim="Use packing heuristics.",
        scope={"family": "combinatorial", "runtime": "trace"},
        preconditions=[],
        recommended_action="Sort items.",
        evidence_refs=["run:3"],
        counterevidence_refs=[],
        support=4,
        uncertainty=0.2,
        status="promoted",
        runtime_compatibility={"engines": ["trace"]},
        supersedes=[],
    )
    promoted_record = memory.record_artifact(
        "knowledge", "reasoning", "knowledge_card", promoted, 0.9
    )
    memory.record_artifact("knowledge", "reasoning", "knowledge_card", candidate, 0.8)
    memory.record_artifact("knowledge", "combinatorial", "knowledge_card", other_scope, 0.7)

    raw = _portable_spec()
    raw["knowledge"] = {"scope_fields": ["family", "runtime"], "top_k": 2}
    retrieved = S.retrieve_knowledge(
        S.normalize_spec(raw), memory, {"family": "reasoning", "runtime": "trace"}
    )

    assert [record.artifact_id for record in retrieved] == [promoted_record.artifact_id]
    assert retrieved[0].content["recommended_action"] == "Verify the final numeric result."

    memory.update_artifact_status(
        promoted_record.artifact_id,
        "rolled_back",
        reason="negative transfer on validation",
    )
    assert S.retrieve_knowledge(
        S.normalize_spec(raw), memory, {"family": "reasoning", "runtime": "trace"}
    ) == []
    assert MemoryLite(root=str(tmp_path)).artifact_history("reasoning", "knowledge_card")[0].status == (
        "rolled_back"
    )


def test_causal_dependency_requires_typed_binding_or_explicit_ordering_only() -> None:
    raw = _portable_spec()
    raw["surface"] = {
        "kind": "recursive_levels",
        "levels": [
            {"id": "level_a", "surface": "config"},
            {"id": "level_b", "surface": "config", "depends_on": ["level_a"]},
        ],
    }
    with pytest.raises(ValueError, match="decorative dependency"):
        S.normalize_spec(raw)

    raw["surface"]["levels"][1]["ordering_only"] = True
    assert S.normalize_spec(raw)["surface"]["levels"][1]["ordering_only"] is True

    raw["surface"]["levels"][1].pop("ordering_only")
    raw["bindings"] = [{
        "from": "level_a.outputs.artifact",
        "to": "module.inputs.prior",
        "codec": "recursive_opt.codec.artifact_to_prior@1",
    }]
    assert S.compile_plan(raw).units


def test_binding_counterfactual_changes_input_and_records_lineage() -> None:
    raw = _portable_spec()
    raw["bindings"] = [{
        "from": "level_a.outputs.artifact",
        "to": "module.inputs.prior",
        "codec": "recursive_opt.codec.artifact_to_prior@1",
    }]
    spec = S.normalize_spec(raw)

    first_inputs: dict[str, Any] = {}
    first_lineage = S.apply_bindings(
        spec,
        {"level_a": {"outputs": {"artifact": {"artifact_id": "a1", "content": "first"}}}},
        first_inputs,
    )
    second_inputs: dict[str, Any] = {}
    S.apply_bindings(
        spec,
        {"level_a": {"outputs": {"artifact": {"artifact_id": "a2", "content": "second"}}}},
        second_inputs,
    )

    assert first_inputs["prior"] != second_inputs["prior"]
    assert first_lineage == [{
        "from": "level_a.outputs.artifact",
        "to": "module.inputs.prior",
        "codec": "recursive_opt.codec.artifact_to_prior@1",
        "artifact_id": "a1",
    }]

    with pytest.raises(TypeError, match="mapping artifact"):
        S.apply_bindings(
            spec,
            {"level_a": {"outputs": {"artifact": "untyped"}}},
            {},
        )


def test_holdout_access_is_capability_gated_by_phase() -> None:
    access = S.DatasetAccess({
        "train": ["train-1"],
        "validation": ["val-1"],
        "holdout": ["holdout-1"],
    })

    for phase in ("fit", "proposal", "induction", "candidate_selection"):
        with pytest.raises(PermissionError, match="holdout"):
            access.read("holdout", phase=phase)
    assert access.read("train", phase="fit") == ("train-1",)
    assert access.read("holdout", phase="final_evaluation") == ("holdout-1",)
    assert access.read("holdout", phase="promotion") == ("holdout-1",)
    assert access.read("holdout", phase="report") == ("holdout-1",)

    with pytest.raises(ValueError, match="phase"):
        access.read("train", phase="unknown")


def test_same_spec_runs_through_fixed_and_trace_engine_contracts() -> None:
    raw = _portable_spec()
    raw["datasets"] = {
        "train": [{"question": "2+2", "answer": "4"}],
        "validation": [{"question": "3+3", "answer": "6"}],
        "holdout": [{"question": "4+4", "answer": "8"}],
    }
    raw["experiment"] = {
        "arms": [
            {"id": "fixed", "engine": {"name": "fixed"}},
            {"id": "trace", "engine": {"name": "trace"}},
        ]
    }

    def fit(module: Module, access: Any, context: dict[str, Any]) -> None:
        assert access.read("train", phase="fit")
        S.restore_module(
            context["spec"], module, {"components": {"planner": "verify every step"}}
        )

    def evaluator(module: Module, dataset: Any, _context: dict[str, Any]) -> Any:
        planner = next(iter(S.snapshot_module(raw, module)["components"].values()))
        score = 0.9 if "verify" in planner else 0.6
        return O.EvaluationResult(
            valid=True,
            status="ok",
            metrics={"accuracy": score, "latency_ms": 25.0},
            feedback="kept natural-language feedback",
            trace={"evaluated": len(dataset)},
            usage={"judge": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}},
        )

    results = S.run_spec(raw, resources={"fit": fit, "evaluator": evaluator})

    assert isinstance(results, tuple) and len(results) == 2
    by_engine = {result.engine: result for result in results}
    assert set(by_engine) == {"fixed", "trace"}
    assert all(isinstance(result, S.RunResult) and result.valid for result in results)
    assert by_engine["trace"].evaluation.metrics["accuracy"] == pytest.approx(0.9)
    assert by_engine["fixed"].evaluation.metrics["accuracy"] == pytest.approx(0.6)
    assert by_engine["trace"].evaluation.feedback == "kept natural-language feedback"
    assert by_engine["trace"].evaluation.trace == {"evaluated": 1}
    assert by_engine["trace"].usage["judge"]["total_tokens"] == 4
    json.dumps([result.to_dict() for result in results])


def test_trace_engine_fault_injection_cannot_read_holdout_during_fit() -> None:
    raw = _portable_spec()
    raw["datasets"] = {"train": [1], "validation": [2], "holdout": [3]}

    def leaking_fit(_module: Module, access: Any, _context: dict[str, Any]) -> None:
        access.read("holdout", phase="fit")

    result = S.run_spec(
        raw,
        resources={
            "fit": leaking_fit,
            "evaluator": lambda _module, _dataset, _context: 1.0,
        },
    )

    assert isinstance(result, S.RunResult)
    assert result.valid is False
    assert result.status == "error"
    assert result.evaluation.error.startswith("PermissionError: holdout is inaccessible")


def test_gepa_optimize_anything_contract_uses_canonical_module_and_evaluator() -> None:
    raw = _portable_spec()
    raw["module"]["config"]["components"]["critic"] = "check briefly"
    raw["engine"] = {"name": "gepa_optimize_anything"}
    raw["datasets"] = {
        "train": [{"id": "train"}],
        "validation": [{"id": "validation"}],
        "holdout": [{"id": "holdout"}],
    }
    calls: dict[str, Any] = {}

    def evaluator(module: Module, dataset: Any, _context: dict[str, Any]) -> Any:
        components = S.snapshot_module(raw, module)["components"]
        accuracy = 0.9 if "improved" in components["planner"] else 0.5
        return O.EvaluationResult(
            valid=True,
            status="ok",
            metrics={"accuracy": accuracy, "latency_ms": 20.0},
            feedback=f"evaluated {dataset[0]['id']}",
            trace={"candidate": components},
            usage={"judge": {"total_tokens": 2}},
        )

    def fake_optimize_anything(**kwargs: Any) -> Any:
        calls.update(kwargs)
        score, info = kwargs["evaluator"](kwargs["seed_candidate"], kwargs["dataset"][0])
        assert isinstance(score, float)
        assert info["valid"] is True
        assert info["metrics"] == {"accuracy": 0.5, "latency_ms": 20.0}
        assert info["feedback"] == "evaluated train"
        return SimpleNamespace(
            best_candidate={"planner": "improved verification", "critic": "check briefly"}
        )

    result = S.run_spec(
        raw,
        resources={"evaluator": evaluator, "gepa_optimize": fake_optimize_anything},
    )

    assert isinstance(result, S.RunResult) and result.valid
    assert result.engine == "gepa_optimize_anything"
    assert calls["seed_candidate"] == {
        "planner": "answer carefully",
        "critic": "check briefly",
    }
    assert calls["dataset"] == [{"id": "train"}]
    assert calls["valset"] == [{"id": "validation"}]
    assert calls["test_set"] == [{"id": "holdout"}]
    assert result.artifact["components"]["planner"] == "improved verification"
    assert result.evaluation.metrics["accuracy"] == pytest.approx(0.9)
    assert result.metadata["objective_projection"] == "weighted"


def test_same_spec_passes_fixed_trace_and_gepa_engine_contract() -> None:
    raw = _portable_spec()
    raw["datasets"] = {
        "train": [{"split": "train"}],
        "validation": [{"split": "validation"}],
        "holdout": [{"split": "holdout"}],
    }
    raw["experiment"] = {
        "arms": [
            {"id": "fixed", "engine": {"name": "fixed"}},
            {"id": "trace", "engine": {"name": "trace"}},
            {"id": "gepa", "engine": {"name": "gepa_optimize_anything"}},
        ]
    }

    def evaluator(_module: Module, dataset: Any, context: dict[str, Any]) -> Any:
        return O.EvaluationResult(
            valid=True,
            status="ok",
            metrics={"accuracy": 0.75, "latency_ms": 10.0},
            feedback=f"{context['engine']} evaluated {dataset[0]['split']}",
        )

    def fake_gepa(**kwargs: Any) -> Any:
        score, info = kwargs["evaluator"](kwargs["seed_candidate"], kwargs["dataset"][0])
        assert score == pytest.approx(0.75)
        assert info["valid"] is True
        return SimpleNamespace(best_candidate=kwargs["seed_candidate"])

    results = S.run_spec(
        raw,
        resources={"evaluator": evaluator, "gepa_optimize": fake_gepa},
    )

    assert isinstance(results, tuple) and len(results) == 3
    assert {result.engine for result in results} == {
        "fixed", "trace", "gepa_optimize_anything",
    }
    assert all(isinstance(result, S.RunResult) and result.valid for result in results)
    assert {result.evaluation.metrics["accuracy"] for result in results} == {0.75}
    assert all(result.module_ref == "recursive_opt.module.reasoning_workflow@1" for result in results)


def test_gepa_rejects_pareto_objective_at_compile_time() -> None:
    raw = _portable_spec()
    raw["engine"] = {"name": "gepa_optimize_anything"}
    raw["objective"]["selection"] = {"mode": "pareto"}

    with pytest.raises(ValueError, match="does not support objective mode 'pareto'"):
        S.compile_plan(raw)


def test_gepa_is_pinned_as_optional_dependency() -> None:
    from pathlib import Path

    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'gepa = ["gepa==0.1.4"]' in pyproject


def test_fake_graph_adapter_is_deterministic_serializable_and_registry_backed() -> None:
    from opto.features.graph import GraphAdapter, GraphExecutor, GraphModule
    from opto.trace import node

    class FakeGraphExecutor(GraphExecutor):
        """Deterministic graph backend used by the portable graph contract."""

        def __init__(self) -> None:
            self.offset = node(1, name="offset", trainable=True)

        @property
        def capabilities(self) -> frozenset[str]:
            """Declare the backend features available to the adapter."""
            return frozenset({"deterministic", "fake"})

        def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
            """Add the current offset to the input value."""
            return {"answer": state["value"] + self.offset.data}

        def parameters(self) -> tuple[Any, ...]:
            """Expose the single trainable graph parameter."""
            return (self.offset,)

        def snapshot(self) -> dict[str, Any]:
            """Return JSON state for the fake backend."""
            return {"offset": self.offset.data}

        def restore(self, artifact: dict[str, Any]) -> None:
            """Restore the fake backend after validating its artifact."""
            if set(artifact) != {"offset"} or not isinstance(artifact["offset"], int):
                raise ValueError("fake graph artifact must contain integer offset")
            self.offset._set(artifact["offset"])

    executor = FakeGraphExecutor()
    adapter = GraphAdapter(executor, input_key="value", output_key="answer")
    module = adapter.as_module()

    assert isinstance(module, GraphModule)
    assert module.forward(2).data == 3
    assert module.parameters() == [executor.offset]
    assert {"deterministic", "fake", "snapshot", "restore", "trace_module"}.issubset(
        adapter.capabilities
    )

    artifact = adapter.snapshot()
    json.dumps(artifact)
    executor.offset._set(4)
    adapter.restore(artifact)
    assert module.forward(2).data == 3

    raw = _portable_spec()
    raw["module"] = {
        "ref": "recursive_opt.module.graph@1",
        "config": {
            "executor_ref": "fake.graph@1",
            "input_key": "value",
            "output_key": "answer",
            "input_codec": "graph.codec.state@1",
            "output_codec": "graph.codec.output_key@1",
        },
    }
    resources = {"graph_executors": {"fake.graph@1": executor}}
    registered = S.build_module(raw, resources)
    registered_artifact = S.snapshot_module(raw, registered)
    executor.offset._set(7)
    S.restore_module(raw, registered, registered_artifact)

    assert registered.forward(2).data == 3
    assert json.loads(json.dumps(registered_artifact)) == registered_artifact


def test_langgraph_adapter_optional_smoke_contract() -> None:
    pytest.importorskip("langgraph")
    from langgraph.graph import END, START, StateGraph

    from opto.features.graph import LangGraphAdapter

    def increment(state: dict[str, Any]) -> dict[str, Any]:
        return {"answer": state["value"] + state["offset"]}

    def build_graph(increment: Any, offset: int = 1) -> Any:
        del offset
        graph = StateGraph(dict)
        graph.add_node("increment", increment)
        graph.add_edge(START, "increment")
        graph.add_edge("increment", END)
        return graph

    adapter = LangGraphAdapter(
        graph_factory=build_graph,
        function_targets={"increment": increment},
        graph_knobs={"offset": 2},
        input_key="value",
        output_key="answer",
    )

    assert adapter.as_module().forward(3).data == 5
    artifact = adapter.snapshot()
    json.dumps(artifact)
    adapter.graph_knobs["offset"]._set(5)
    adapter.restore(artifact)
    assert adapter.as_module().forward(3).data == 5


def _notebook_spec_literals() -> dict[str, dict[str, Any]]:
    """Load the notebook's checked-in normalized golden specifications."""
    from pathlib import Path

    root = Path("artifacts/control_plane_v2/golden_specs")
    return {
        "UC4_SPEC": json.loads(
            (root / "uc4_positive.normalized.json").read_text(encoding="utf-8")
        ),
        "UC14_SPEC": json.loads(
            (root / "uc14_negative.normalized.json").read_text(encoding="utf-8")
        ),
    }


def test_notebook_is_a_strict_spec_only_control_plane_client() -> None:
    import ast
    from pathlib import Path

    notebook = json.loads(
        Path("examples/recursive_opt_use_cases.ipynb").read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    tree = ast.parse(code)
    forbidden_calls = {
        "compile_level", "optimize", "optimize_config_numeric", "_final_eval",
        "MemoryLite", "OptoPrime", "CodeArtifactLevel", "MetaLevel",
    }
    allowed_calls = {
        "Path", "read_text", "loads", "normalize_spec", "explain_spec", "run_spec", "display",
    }
    call_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }

    assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree))
    assert not any(isinstance(node, (ast.For, ast.AsyncFor, ast.While)) for node in ast.walk(tree))
    assert call_names.isdisjoint(forbidden_calls)
    assert call_names <= allowed_calls
    assert "os.environ" not in code
    assert set(_notebook_spec_literals()) == {"UC4_SPEC", "UC14_SPEC"}
    assert "uc4_positive.normalized.json" in code
    assert "uc14_negative.normalized.json" in code


def test_notebook_uc4_positive_and_uc14_negative_controls() -> None:
    specs = _notebook_spec_literals()

    uc4 = S.run_spec(specs["UC4_SPEC"])
    uc14 = S.run_spec(specs["UC14_SPEC"])

    assert isinstance(uc4, S.RunResult) and uc4.valid is True
    assert uc4.evaluation.metrics == {"accuracy": 1.0}
    assert isinstance(uc14, S.RunResult) and uc14.valid is False
    assert uc14.status == "invalid"
    assert uc14.evaluation.status == "constraint_failed"


def test_notebook_executes_from_a_clean_offline_kernel() -> None:
    from pathlib import Path

    import nbformat
    from nbclient import NotebookClient

    path = Path("examples/recursive_opt_use_cases.ipynb")
    notebook = nbformat.read(path, as_version=4)
    executed = NotebookClient(
        notebook,
        timeout=60,
        kernel_name="python3",
        allow_errors=False,
    ).execute(cwd=str(Path.cwd()))

    assert all(
        output.get("output_type") != "error"
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    )


def test_historical_migration_report_is_complete_and_source_immutable() -> None:
    import hashlib
    from pathlib import Path

    report = json.loads(
        Path("artifacts/control_plane_v2/migration_report.json").read_text(
            encoding="utf-8"
        )
    )
    entries = report["entries"]
    categories = {
        "replayable", "migrated_replayable", "historical_only", "invalid",
        "missing_dependency", "local_nonportable",
    }

    assert report["scope"]["tracked_files"] == len(entries) == 85
    assert set(report["summary"]) == categories
    assert sum(report["summary"].values()) == len(entries)
    assert report["summary"]["replayable"] == 46
    assert report["summary"]["migrated_replayable"] == 16
    assert report["summary"]["missing_dependency"] == 23

    for entry in entries:
        source = Path(entry["source"])
        assert source.exists()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == entry["sha256"]
        assert entry["classification"] in categories
        if entry["classification"] != "migrated_replayable":
            continue
        migrated = json.loads(
            Path(entry["migrated_path"]).read_text(encoding="utf-8")
        )
        assert json.loads(json.dumps(S.normalize_spec(migrated))) == migrated
        assert migrated["fingerprint"] == entry["fingerprint"]


def test_run_result_accounts_budget_from_canonical_role_usage() -> None:
    raw = _portable_spec()
    raw["engine"] = {"name": "fixed"}
    raw["budget"] = {
        "optimizer_llm_calls": 2,
        "eval_llm_calls": 1,
        "candidates": 1,
        "wall_time_s": 30,
        "on_exceed": "fail",
    }

    def evaluator(_module: Module, _dataset: Any, _context: dict[str, Any]) -> Any:
        return {
            "metrics": {"accuracy": 1.0, "latency_ms": 1.0},
            "usage": {
                "optimizer": {"calls": 2, "total_tokens": 8},
                "judge": {"calls": 1, "total_tokens": 3},
            },
        }

    result = S.run_spec(raw, resources={"evaluator": evaluator})

    assert isinstance(result, S.RunResult)
    assert result.budget["accounted"]["optimizer_llm_calls"] == 2
    assert result.budget["accounted"]["eval_llm_calls"] == 1
    assert result.budget["accounted"]["candidates"] == 1
    assert result.budget["accounted"]["evaluation_runs"] == 1
    assert result.budget["exceeded"] == ()


def test_resume_reuses_the_same_fingerprinted_result_idempotently() -> None:
    raw = _portable_spec()
    raw["runtime"] = {"resume": True, "offline": True}
    calls = {"evaluator": 0}
    result_store: dict[str, S.RunResult] = {}

    def evaluator(_module: Module, _dataset: Any, _context: dict[str, Any]) -> float:
        calls["evaluator"] += 1
        return 1.0

    resources = {"evaluator": evaluator, "result_store": result_store}
    first = S.run_spec(raw, resources=resources)
    second = S.run_spec(raw, resources=resources)

    assert isinstance(first, S.RunResult) and isinstance(second, S.RunResult)
    assert first.to_dict() == second.to_dict()
    assert calls["evaluator"] == 1
    assert len(result_store) == 1
