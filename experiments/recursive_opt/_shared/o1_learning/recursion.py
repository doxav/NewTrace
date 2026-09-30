"""Bounded genuine O2 -> O1 -> O0 nesting through the existing control plane."""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping

from experiments.recursive_opt._shared.o1_learning import study
from experiments.recursive_opt._shared.o1_learning.analysis import extra_variants
from opto.trainer.objectives import EvaluationResult

LOWER: dict[str, dict[str, Any]] = {}
META: dict[str, dict[str, Any]] = {}
EVENTS: list[dict[str, Any]] = []
CONFIGS = {item["id"]: item for item in study.variants() + extra_variants()}


def utility(report: dict[str, Any]) -> float:
    """Prefer the frozen hitting-time criterion; accuracy only breaks ties."""
    capped = report["first_hit"] if report["first_hit"] is not None else 5
    return -float(capped) + report["selection"]["accuracy"] / 100


def evaluate_meta(
    output: Any, example: Any, context: Mapping[str, Any]
) -> EvaluationResult:
    """A meta evaluation executes the next production optimization level on development only."""
    value = getattr(output, "data", output)
    choice = value["components"]["choice"]
    tier = context["inputs"]["tier"]
    if tier == 1:
        if choice not in CONFIGS:
            return EvaluationResult(
                False, "invalid", feedback=f"choose one of {list(CONFIGS)}"
            )
        cached = choice in LOWER
        if not cached:
            parent_client, parent_records = study.CLIENT, list(study.RECORDS)
            try:
                LOWER[choice] = study.run(
                    {**CONFIGS[choice], "max_tokens": 16000}, 19023, "recursive_O0", 4
                )
            finally:
                study.CLIENT = parent_client
                study.RECORDS[:] = parent_records
        report = LOWER[choice]
        value_score = utility(report)
        details = {
            "curve": report["validation_curve"],
            "hit": report["first_hit"],
            "O0_calls": report["calls"],
        }
        valid = report["valid"]
    else:
        if choice not in ["OptoPrimeV2", "OPROv2"]:
            return EvaluationResult(
                False, "invalid", feedback="choose OptoPrimeV2 or OPROv2"
            )
        cached = choice in META
        if not cached:
            META[choice] = run_meta(1, choice)
        report = META[choice]
        value_score = report["utility"]
        details = {
            "O1_selected_configuration": report["selected"],
            "O1_calls": report["calls"],
            "utility": value_score,
        }
        valid = report["valid"]
    event = {
        "tier": tier,
        "choice": choice,
        "cache_hit": cached,
        "details": details,
        "utility": value_score,
        "valid": valid,
    }
    EVENTS.append(event)
    study.persist(
        study.ROOT / "raw" / "recursion_events" / f"event_{len(EVENTS):03}.json", event
    )
    return EvaluationResult(
        valid,
        "ok" if valid else "invalid",
        {"score": value_score} if valid else {},
        json.dumps(event),
        artifacts=event,
    )


def meta_spec(tier: int, optimizer: str) -> dict[str, Any]:
    """Declare optimizer choice at O2 or learning-configuration choice at O1."""
    base = study.specification(
        {**study.variants()[0], "max_tokens": 16000}, 19023, calls=1
    )
    level = base["levels"][0]
    level["id"] = f"O{tier}"
    level["module"] = {
        "ref": "recursive_opt.module.reasoning_workflow@1",
        "config": {
            "components": {"choice": "standard" if tier == 1 else "OptoPrimeV2"}
        },
        "inputs": {"tier": tier},
    }
    level["datasets"] = {
        "train": [{"development_seed": 19023}],
        "validation": [],
        "holdout": [],
    }
    level["engine"]["config"].update(
        optimizer=optimizer,
        trainer="PrioritySearch",
        iterations=2,
        trainer_kwargs={"batch_size": 1, "num_threads": 1, "test_frequency": None},
        optimizer_kwargs={
            "log": False,
            "max_tokens": 16000,
            "objective": f'Choose a better setting from {list(CONFIGS) if tier==1 else ["OptoPrimeV2","OPROv2"]}. Each evaluation really executes the next optimization level. Maximize utility: fewer O0 proposals to validation accuracy 0.9, then higher accuracy as tie-break. Preserve exact choice spelling. No access to final test.',
        },
    )
    level["objective"] = {
        "evaluator_ref": "exp19.evaluator.meta@1",
        "intent": "Minimize O0 calls to a fixed quality target; report additional meta costs separately.",
        "metrics": {
            "score": {
                "direction": "maximize",
                "source": "evaluation.metrics.score",
                "aggregate_examples": "mean",
            }
        },
        "selection": {"mode": "scalar", "score_key": "score"},
        "trace_config": {
            "mode": "internal",
            "detail": "full",
            "credit_horizon": "full",
            "max_nodes": 24,
            "max_chars": 3000,
        },
    }
    return base


def run_meta(tier: int, optimizer: str) -> dict[str, Any]:
    """Run one genuine nested optimization and retain its exact real-call count."""
    path = study.ROOT / "raw" / f"recursive_O{tier}" / optimizer
    client = study.RecordingClient(path)
    spec = meta_spec(tier, optimizer)
    study.persist(path / "spec.json", spec)
    parent_client, parent_records = study.CLIENT, list(study.RECORDS)
    try:
        result = study.S.run_spec(
            spec, resources={"llm_factory": lambda profile, role: client}
        )
    finally:
        study.CLIENT = parent_client
        study.RECORDS[:] = parent_records
    raw = result.to_dict()
    study.persist(path / "production.json", raw)
    choice = result.artifact.get("components", {}).get(
        "choice", "standard" if tier == 1 else "OptoPrimeV2"
    )
    summary = {
        "tier": tier,
        "optimizer": optimizer,
        "selected": choice,
        "valid": result.valid,
        "calls": client.calls,
        "usage": client.usage,
        "utility": result.evaluation.metrics.get("score", -6),
        "error": result.error,
    }
    study.persist(path / "result.json", summary)
    return summary


def execute() -> dict[str, Any]:
    """Audit three real nested levels, with a maximum of 15 completed responses."""
    study.register()
    if "exp19.evaluator.meta@1" not in study.S._EVALUATOR_REGISTRY:
        study.S.register_evaluator("exp19.evaluator.meta@1", evaluate_meta)
    path = study.ROOT / "recursion_result.json"
    if path.exists():
        return json.loads(path.read_text())
    outer = run_meta(2, "OptoPrimeV2")
    selected_o1 = META.get(outer["selected"])
    choice = (
        selected_o1["selected"] if selected_o1 and selected_o1["valid"] else "standard"
    )
    result = {
        "O2": outer,
        "O1": META,
        "O0": {
            key: {
                "calls": value["calls"],
                "utility": utility(value),
                "valid": value["valid"],
            }
            for key, value in LOWER.items()
        },
        "selected_config": copy.deepcopy(CONFIGS.get(choice, CONFIGS["standard"])),
        "actual_calls": outer["calls"]
        + sum(row["calls"] for row in META.values())
        + sum(row["calls"] for row in LOWER.values()),
        "cache_hits": sum(event["cache_hit"] for event in EVENTS),
        "events": EVENTS,
        "interpretation": "single-development-seed recursion feasibility/selection diagnostic; not evidence that depth improves generalization",
    }
    study.persist(path, result)
    return result
