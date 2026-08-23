"""Provider-free causal contract for all primary Experiment 0 arms."""

from __future__ import annotations

import copy
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from opto.features.recursive_opt import spec as control_plane
from opto.trainer.algorithms.algorithm import Trainer

from .components import FORWARD_EVENTS, clear_forward_events
from .datasets import all_expected_answers, dataset_manifest, dataset_manifest_v2
from .evaluator import EVALUATOR_EVENTS, clear_evaluator_events
from .registration import assert_strict_output_evaluator, register_experiment_components
from .specs import INITIAL_ARTIFACT, build_spec


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def model_dump(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class _FakeResponse:
    def __init__(self, content: str, prompt: str, *, model: str) -> None:
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(content) // 4)
        self.usage = _Usage(
            prompt_tokens,
            completion_tokens,
            prompt_tokens + completion_tokens,
        )
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
        self.model = model
        self._hidden_params = {
            "custom_llm_provider": "offline-fake",
            "model_id": model,
            "response_cost": 0.0,
            "cache_hit": False,
        }


class _FakeClient:
    def __init__(self, role: str, model: str, *, harmful: bool = False) -> None:
        self.role = role
        self.model = model
        self.harmful = harmful
        self.requests: list[dict[str, Any]] = []
        self.expected = all_expected_answers()

    def _optimizer_update(self, prompt: str) -> str:
        names = re.findall(r'<variable\s+name="([^"]+)"', prompt)
        if not names:
            names = ["analysis_instruction", "answer_instruction"]
        value = "HARMFUL INVALID" if self.harmful else "OFFLINE IMPROVED"
        return "\n".join(
            ["<reasoning>deterministic offline proposal</reasoning>"]
            + [
                f"<variable><name>{name}</name><value>{value} {name}</value></variable>"
                for name in names
            ]
        )

    def _forward_content(self, prompt: str) -> str:
        if "Analysis from the first stage:" not in prompt:
            return "deterministic local analysis"
        if "HARMFUL INVALID" in prompt:
            return "no parseable final answer"
        for question, expected in self.expected.items():
            if question in prompt:
                return f"FINAL: {expected}"
        return "FINAL: 0"

    def _gepa_reflection(self) -> str:
        """Return a deterministic proposal for GEPA's canonical chat prompt."""
        value = "HARMFUL INVALID" if self.harmful else "OFFLINE GEPA IMPROVED"
        return f"```\n{value}\n```"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            raise TypeError("offline provider accepts only keyword chat messages")
        messages = kwargs.get("messages")
        if not isinstance(messages, list) or not all(
            isinstance(message, Mapping) for message in messages
        ):
            raise TypeError("offline provider requires messages=[chat mappings]")
        prompt = "\n".join(str(message.get("content", "")) for message in messages)
        is_gepa_reflection = (
            self.role == "optimizer"
            and "## Optimization Goal" in prompt
            and "## Current Component" in prompt
        )
        kind = "gepa_reflection" if is_gepa_reflection else self.role
        self.requests.append({"kind": kind, "prompt": prompt, "kwargs": copy.deepcopy(kwargs)})
        content = (
            self._forward_content(prompt)
            if self.role == "forward"
            else self._gepa_reflection()
            if is_gepa_reflection
            else self._optimizer_update(prompt)
        )
        return _FakeResponse(content, prompt, model=self.model)


class _Factory:
    def __init__(self, *, harmful: bool = False) -> None:
        self.harmful = harmful
        self.clients: list[_FakeClient] = []

    def __call__(self, profile: Mapping[str, Any], role: str) -> _FakeClient:
        client = _FakeClient(role, str(profile["resolved_model"]), harmful=self.harmful)
        self.clients.append(client)
        return client


class _ScriptedHarmfulTrainer(Trainer):
    """Install a known-invalid prompt so the outer validation gate is causal."""

    def train(self, *_args: Any, **_kwargs: Any) -> "_ScriptedHarmfulTrainer":
        for parameter in self.agent.parameters():
            parameter._set(f"HARMFUL INVALID {parameter.py_name}")
        self.memory = SimpleNamespace(memory=[{"kind": "scripted-harmful"}])
        return self


def _without_engine(spec: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(spec))
    value.pop("fingerprint", None)
    for level in value["levels"]:
        level["engine"] = {"name": "<declared-arm>", "config": "<declared-arm>"}
    return value


def _run_one(
    engine: str,
    output_root: Path,
    *,
    validation_gate: bool = True,
    harmful: bool = False,
    resources: Mapping[str, Any] | None = None,
    task: str = "object_counting",
    baseline_tokens: Mapping[str, int] | None = None,
    split_limits: Mapping[str, int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    clear_forward_events()
    clear_evaluator_events()
    factory = _Factory(harmful=harmful)
    raw = build_spec(
        task=task,
        engine=engine,
        seed=7,
        output_directory=output_root,
        proposals=1,
        validation_gate=validation_gate,
        offline=True,
        test_mode=True,
        baseline_tokens=baseline_tokens,
        split_limits=split_limits,
    )
    result = control_plane.run_spec(
        raw, resources={"llm_factory": factory, **dict(resources or {})}
    )
    forward_ids = {id(event["output"]) for event in FORWARD_EVENTS}
    exact_output = all(event["output_identity"] in forward_ids for event in EVALUATOR_EVENTS)
    manifest = (
        dataset_manifest_v2()
        if task.startswith("bbeh_") or task == "gsm8k" and baseline_tokens
        else dataset_manifest()
    )
    holdout_ids = {
        sample["id"]
        for sample in manifest["tasks"][task]["samples"]
        if sample["split"] == "holdout"
    }
    holdout_during_fit = any(
        event["sample_id"] in holdout_ids and event["phase"] != "final_evaluation"
        for event in EVALUATOR_EVENTS
    )
    observations = {
        "raw": raw,
        "normalized": control_plane.normalize_spec(raw),
        "forward_count": len(FORWARD_EVENTS),
        "evaluator_count": len(EVALUATOR_EVENTS),
        "exact_output": exact_output,
        "holdout_during_fit": holdout_during_fit,
        "provider_requests": sum(len(client.requests) for client in factory.clients),
        "provider_request_kinds": [
            request["kind"] for client in factory.clients for request in client.requests
        ],
        "factory_roles": [client.role for client in factory.clients],
    }
    return result, observations


def run_offline_contract() -> dict[str, Any]:
    register_experiment_components()
    assert_strict_output_evaluator()
    package_root = Path(__file__).resolve().parent
    repository_root = package_root.parents[2]
    lock = json.loads(
        (package_root / "control_plane_lock_after_empty_text_retry.json").read_text(
            encoding="utf-8"
        )
    )
    locked_control = lock["control_plane"]
    readiness_spec = json.loads(
        (
            repository_root
            / "artifacts/control_plane_v2/golden_specs/uc4_positive.normalized.json"
        ).read_text(encoding="utf-8")
    )
    readiness_provenance = control_plane.compile_plan(readiness_spec).code_provenance
    baseline_path = package_root / "baseline_token_manifest.json"
    eligibility_path = package_root / "reports/task_eligibility_v2.json"
    if baseline_path.exists() and eligibility_path.exists():
        baseline_manifest = json.loads(baseline_path.read_text(encoding="utf-8"))
        task = str(baseline_manifest["selected_task"])
        baseline_tokens = {
            str(row["sample_id"]): int(row["baseline_forward_tokens"])
            for row in baseline_manifest["samples"]
        }
        split_limits: Mapping[str, int] | None = {
            "train": 4,
            "validation": 4,
            "holdout": 8,
        }
    else:
        task = "object_counting"
        baseline_tokens = {
            sample["id"]: 10_000
            for sample in dataset_manifest()["tasks"][task]["samples"]
        }
        split_limits = None
    with tempfile.TemporaryDirectory(prefix="recursive_exp0_offline_") as tmp:
        root = Path(tmp)
        configs = {
            "A": ("fixed", True),
            "B": ("trace", True),
            "C": ("gepa_optimize_anything", True),
            "D": ("trace", False),
        }
        runs: dict[str, Any] = {}
        observations: dict[str, Any] = {}
        for arm, (engine, gate) in configs.items():
            runs[arm], observations[arm] = _run_one(
                engine,
                root / "arms",
                validation_gate=gate,
                task=task,
                baseline_tokens=baseline_tokens,
                split_limits=split_limits,
            )

        base_specs = [_without_engine(observations[arm]["normalized"]) for arm in configs]
        comparable_base = all(spec == base_specs[0] for spec in base_specs[1:])
        provenance = {
            arm: control_plane.compile_plan(observations[arm]["raw"]).code_provenance
            for arm in configs
        }
        runtime_digests = {value["runtime_tree_sha256"] for value in provenance.values()}

        import opto.trainer.algorithms as trace_algorithms

        scripted_trainer = "Experiment0ScriptedHarmfulTrainer"
        setattr(trace_algorithms, scripted_trainer, _ScriptedHarmfulTrainer)
        try:
            harmful_gated, gated_obs = _run_one(
                "trace",
                root / "harmful-gated",
                validation_gate=True,
                harmful=True,
                resources={"trainer": scripted_trainer},
                task=task,
                baseline_tokens=baseline_tokens,
                split_limits=split_limits,
            )
            harmful_ungated, ungated_obs = _run_one(
                "trace",
                root / "harmful-ungated",
                validation_gate=False,
                harmful=True,
                resources={"trainer": scripted_trainer},
                task=task,
                baseline_tokens=baseline_tokens,
                split_limits=split_limits,
            )
        finally:
            delattr(trace_algorithms, scripted_trainer)

        resume_factory = _Factory()
        resume_raw = build_spec(
            task=task,
            engine="fixed",
            seed=19,
            output_directory=root / "resume",
            offline=True,
            test_mode=True,
            baseline_tokens=baseline_tokens,
            split_limits=split_limits,
        )
        resume_raw["runtime"]["resume"] = True
        clear_forward_events()
        first = control_plane.run_spec(resume_raw, resources={"llm_factory": resume_factory})
        first_calls = len(FORWARD_EVENTS)
        clear_forward_events()
        second = control_plane.run_spec(resume_raw, resources={"llm_factory": resume_factory})
        resume_calls = len(FORWARD_EVENTS)

    assertions = {
        "same_normalized_base_spec": comparable_base,
        "same_module_ref": len({run.module_ref for run in runs.values()}) == 1,
        "same_evaluator_ref": all(
            obs["normalized"]["levels"][0]["objective"]["evaluator_ref"]
            == "recursive_experiments.evaluator.exact_reasoning@1"
            for obs in observations.values()
        ),
        "same_dataset_refs": len(
            {
                json.dumps(obs["normalized"]["levels"][0]["datasets"], sort_keys=True)
                for obs in observations.values()
            }
        )
        == 1,
        "same_initial_artifact": all(
            run.level_results[0]["artifact"] == INITIAL_ARTIFACT for run in [runs["A"]]
        ) and all(
            obs["normalized"]["levels"][0]["module"]["artifact"] == INITIAL_ARTIFACT
            for obs in observations.values()
        ),
        "same_objective": len(
            {
                json.dumps(obs["normalized"]["levels"][0]["objective"], sort_keys=True)
                for obs in observations.values()
            }
        )
        == 1,
        "same_role_configuration": len(
            {
                json.dumps(obs["normalized"]["levels"][0]["llm_roles"], sort_keys=True)
                for obs in observations.values()
            }
        )
        == 1,
        "same_runtime_tree_sha256": runtime_digests
        == {locked_control["runtime_tree_sha256"]},
        "same_registry_sha256": (
            readiness_provenance["registry_sha256"]
            == locked_control["registry_sha256"]
            and all(
                observation["normalized"]["runtime"]["strict_refs"]
                for observation in observations.values()
            )
        ),
        "one_forward_per_evaluator": all(
            obs["forward_count"] == obs["evaluator_count"] for obs in observations.values()
        ),
        "evaluator_receives_exact_output": all(
            obs["exact_output"] for obs in observations.values()
        ),
        "holdout_inaccessible_during_optimization": not any(
            obs["holdout_during_fit"] for obs in observations.values()
        ),
        "usage_attributed_once": all(
            int(run.usage["forward"].get("calls", 0))
            == int(run.budget["accounted"]["eval_llm_calls"])
            for run in runs.values()
        ),
        "output_persistence_and_resume": first.to_dict() == second.to_dict()
        and first_calls > 0
        and resume_calls == 0,
        "gepa_does_not_receive_holdout": not observations["C"]["holdout_during_fit"],
        "trace_real_candidate": runs["B"].artifact != INITIAL_ARTIFACT,
        "gepa_real_candidate_proposed": "gepa_reflection"
        in observations["C"]["provider_request_kinds"],
        "validation_gate_rejects_harmful": harmful_gated.artifact == INITIAL_ARTIFACT,
        "ungated_harmful_candidate_is_observable": harmful_ungated.artifact != INITIAL_ARTIFACT,
        "candidate_accounting_consistent": all(
            run.budget["accounted"]["candidates"]
            == run.budget["accounted"]["candidates_reserved"]
            and run.budget["accounted"]["candidates_proposed"] >= 1
            and run.budget["accounted"]["candidates_evaluated"] >= 1
            for arm, run in runs.items()
            if arm in {"B", "C", "D"}
        ),
    }
    return {
        "schema_version": "recursive-opt-offline-contract/v1",
        "task": task,
        "assertions": assertions,
        "passed": all(assertions.values()),
        "runtime_tree_sha256": locked_control["runtime_tree_sha256"],
        "registry_sha256": locked_control["registry_sha256"],
        "plan_registry_sha256": {
            arm: provenance[arm]["registry_sha256"] for arm in configs
        },
        "arms": {
            arm: {
                "engine": run.engine,
                "valid": run.valid,
                "artifact": dict(run.artifact),
                "metrics": dict(run.evaluation.metrics),
                "usage": json.loads(json.dumps(run.usage)),
                "budget": json.loads(json.dumps(run.budget)),
                "observations": {
                    key: value
                    for key, value in observations[arm].items()
                    if key not in {"raw", "normalized"}
                },
            }
            for arm, run in runs.items()
        },
        "harmful_gate": {
            "gated_artifact": dict(harmful_gated.artifact),
            "ungated_artifact": dict(harmful_ungated.artifact),
            "gated_observations": {
                key: value for key, value in gated_obs.items() if key not in {"raw", "normalized"}
            },
            "ungated_observations": {
                key: value for key, value in ungated_obs.items() if key not in {"raw", "normalized"}
            },
        },
    }
