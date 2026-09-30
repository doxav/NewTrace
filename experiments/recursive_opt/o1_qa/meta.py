"""EXP21 genuine O1 -> O0 and O2 -> O1 -> O0 production evaluations."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping

from opto.features.recursive_opt.budget import BudgetExceeded
from opto.trace.nodes import GRAPH
from opto.trainer.algorithms.priority_search import PrioritySearch
from opto.trainer.objectives import EvaluationResult

from . import axes as X, campaign as C, task

ACTIVE: X.Journal | None = None
CONTEXT: dict[str, Any] = {}
EVENTS: list[dict[str, Any]] = []
EVALUATOR = "exp21.evaluator.meta@1"


def config_name(config: dict[str, Any]) -> str:
    """Reuse a screened policy exactly, otherwise identify its canonical combined configuration."""
    return next(
        (v["id"] for v in X.variants() if v["config"] == config),
        "auto_" + task.digest(config)[:16],
    )


def evaluate_configuration(config: dict[str, Any]) -> dict[str, Any]:
    """Execute and score real lower-level learning on development questions only."""
    X.validate_config(config)
    root, name = X.ROOT / "development", config_name(config)
    jobs = [
        {"root": str(root), "seed": seed, "name": name, "config": config, "mode": "dev"}
        for seed in X.DEV_SEEDS
    ]
    missing = [
        j
        for j in jobs
        if not (root / "reports" / str(j["seed"]) / f"{name}.json").exists()
    ]
    if missing:
        X.run_jobs(missing, workers=2)
    reports = [X.read(root / "reports" / str(s) / f"{name}.json") for s in X.DEV_SEEDS]
    return {
        "utility": statistics.mean(v["primary"] for v in reports),
        "details": {
            "lower_config": config,
            "lower_name": name,
            "seeds": X.DEV_SEEDS,
            "curves": [v["curve"] for v in reports],
            "lower_responses_allocated": len(X.DEV_SEEDS) * X.CALLS,
            "new_child_chains": len(missing),
            "cache_hits": len(jobs) - len(missing),
        },
    }


def inner_job(args: dict[str, Any]) -> dict[str, Any]:
    """Execute a nested O1 in a fresh process so parent Trace state remains isolated."""
    from experiments.recursive_opt._shared.optimizer_discovery.phase0 import _load_key

    _load_key()
    return run_meta(
        Path(args["root"]),
        tier=1,
        optimizer=args["optimizer"],
        memory_size=args["memory_size"],
        calls=3,
    )


def evaluate_meta(
    output: Any, example: Any, context: Mapping[str, Any]
) -> EvaluationResult:
    """An O1 evaluation runs O0; an O2 evaluation invokes an actual production O1 engine."""
    value = getattr(output, "data", output)["components"]
    tier = CONTEXT["tier"]
    try:
        if tier == 1:
            choice = X.decode_meta(value)
        else:
            if (
                set(value) != {"optimizer", "memory_size"}
                or value["optimizer"] not in ["OptoPrimeV2", "OPROv2"]
                or type(value["memory_size"]) is not int
                or value["memory_size"] not in [0, 3]
            ):
                raise ValueError(
                    "O2 may only choose the registered O1 optimizer and memory size"
                )
            choice = dict(value)
    except (ValueError, TypeError, KeyError):
        event = {
            "tier": tier,
            "valid": False,
            "choice": value,
            "slot": ACTIVE.calls,
            "reason": "outside frozen configuration domain",
        }
        EVENTS.append(event)
        C.retain(ACTIVE.directory / "evaluations" / f"{len(EVENTS):04d}.json", event)
        return EvaluationResult(
            False,
            "invalid_configuration",
            feedback="Use exactly the frozen configuration domain and native types.",
        )
    key = task.digest(choice)
    path = ACTIVE.directory / "fitness_cache" / f"{key}.json"
    cache_hit = path.exists()
    if cache_hit:
        assessed = X.read(path)
    elif tier == 1:
        assessed = evaluate_configuration(choice)
        C.retain(path, assessed)
    else:
        root = (
            Path(CONTEXT["family_root"])
            / f"O1_{choice['optimizer']}_memory{choice['memory_size']}"
        )
        with ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            report = pool.submit(inner_job, {"root": str(root), **choice}).result()
        assessed = {
            "utility": report["utility"],
            "details": {
                "O1_path": str(root),
                "O1_selected_config": report["selected_config"],
                "O1_completed_responses": report["completed_responses"],
            },
        }
        C.retain(path, assessed)
    event = {
        "tier": tier,
        "valid": True,
        "choice": choice,
        "slot": ACTIVE.calls,
        "cache_hit": cache_hit,
        **assessed,
    }
    EVENTS.append(event)
    C.retain(ACTIVE.directory / "evaluations" / f"{len(EVENTS):04d}.json", event)
    return EvaluationResult(
        True,
        "ok",
        {"score": assessed["utility"]},
        json.dumps(event, ensure_ascii=False),
        artifacts=event,
    )


class MetaTrainer(PrioritySearch):
    """Production meta-search with a strict completed-response cap, including empty outputs."""

    def propose(self, samples: Any, verbose: bool = False, **kwargs: Any) -> Any:
        """Never replace a completed bad response; let the canonical engine handle candidates."""
        if ACTIVE.calls >= ACTIVE.limit:
            return []
        try:
            return super().propose(samples, verbose=verbose, **kwargs)
        except BudgetExceeded:
            if ACTIVE.calls != ACTIVE.limit:
                raise
            return []
        except RuntimeError as error:
            if "no final textual content after 2 metered attempts" not in str(error):
                raise
            return []


def specification(
    tier: int, optimizer: str, memory_size: int, calls: int
) -> dict[str, Any]:
    """Declare a meta-policy as trainable data whose objective executes lower learning."""
    value = task.specification(
        [{"development_suite": "EXP21"}], seed=21031, curriculum=False, calls=calls
    )
    level = value["levels"][0]
    components = (
        dict(X.BASE) if tier == 1 else {"optimizer": "OptoPrimeV2", "memory_size": 0}
    )
    level["id"] = f"O{tier}"
    level["surface"]["targets"] = list(components)
    level["module"] = {
        "ref": "recursive_opt.module.reasoning_workflow@1",
        "config": {"components": components},
        "inputs": {},
    }
    level["engine"]["config"].update(
        optimizer=optimizer,
        trainer="PrioritySearch",
        iterations=calls + 1,
        optimizer_kwargs={
            "log": False,
            "memory_size": memory_size,
            "objective": (
                "Optimize the LEARNING CONFIGURATION, not question-answer text. Each evaluation actually runs the lower optimization via the production control plane on separate development examples. Maximize average answer exact-match of validation-selected prefix policies over 0..6 lower optimizer responses, averaged over two development seeds. No final TEST access. All calls including failed outputs count. "
                + (
                    "Change the named learning parameters using EXACTLY these allowed native values: "
                    + json.dumps(X.DOMAINS)
                    + ". You may combine choices from multiple axes. Keep the reader and optimizer model identities fixed."
                    if tier == 1
                    else "Choose optimizer from OptoPrimeV2 or OPROv2 and integer memory_size from 0 or 3. Each choice executes three O1 proposals, whose evaluations run O0. Consider measured fitness, not the optimizer name."
                )
            ),
        },
        trainer_kwargs={
            "batch_size": 1,
            "num_threads": 1,
            "test_frequency": None,
            "selection_score_window": "latest_train_batch",
        },
    )
    level["objective"] = {
        "evaluator_ref": EVALUATOR,
        "intent": "Improve lower-level learning speed under a fixed lower response budget; discovery costs reported separately.",
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
            "max_nodes": 12,
            "max_chars": 3000,
        },
    }
    return value


def run_meta(
    root: Path,
    *,
    tier: int,
    optimizer: str,
    memory_size: int,
    calls: int,
    factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run bounded genuine optimization of optimizer settings and preserve all evaluated choices."""
    global ACTIVE, CONTEXT, EVENTS
    X.register()
    import opto.trainer.algorithms as algorithms

    algorithms.EXP21MetaTrainer = MetaTrainer
    if EVALUATOR not in task.S._EVALUATOR_REGISTRY:
        task.S.register_evaluator(EVALUATOR, evaluate_meta)
    spec = specification(tier, optimizer, memory_size, calls)
    C.retain(root / "spec.json", spec)
    if (root / "result.json").exists():
        return X.read(root / "result.json")
    GRAPH.clear()
    ACTIVE = X.Journal(root, root / "unused_reader_cache", 21031, calls)
    CONTEXT = {"tier": tier, "family_root": str(root.parent)}
    EVENTS = []

    def clients(profile: Any, role: str) -> Any:
        """Every meta-level call uses the same metered DeepSeek profile."""
        return ACTIVE.client(profile, role, factory(profile, role) if factory else None)

    production = task.S.run_spec(
        spec, resources={"llm_factory": clients, "trainer": "EXP21MetaTrainer"}
    ).to_dict()
    C.retain(root / "production.json", production)
    if production.get("error") or ACTIVE.calls != calls:
        raise RuntimeError(
            "incomplete meta run; preserve evidence and diagnose infrastructure"
        )
    eligible = [e for e in EVENTS if e["valid"]]
    if not eligible:
        raise RuntimeError("trusted initial meta configuration failed")
    selected = max(eligible, key=lambda e: (e["utility"], -e["slot"]))
    config = (
        selected["choice"] if tier == 1 else selected["details"]["O1_selected_config"]
    )
    result = {
        "tier": tier,
        "optimizer": optimizer,
        "memory_size": memory_size,
        "completed_responses": ACTIVE.calls,
        "selected_config": config,
        "selected_choice": selected["choice"],
        "utility": selected["utility"],
        "events": EVENTS,
        "production_valid": production["valid"],
        "scope": "bounded configuration optimization through actual nested lower runs; not a claim of novel optimizer code or free recursion",
    }
    C.retain(root / "result.json", result)
    return result


def main() -> None:
    """Execute O1 first, then the registered O2 treatment without any confirmatory data."""
    from experiments.recursive_opt._shared.optimizer_discovery.phase0 import _load_key

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["O1", "O2"])
    args = parser.parse_args()
    X.verify()
    _load_key()
    root = X.ROOT / "meta"
    if args.stage == "O1":
        result = run_meta(
            root / "O1_OptoPrimeV2_memory0",
            tier=1,
            optimizer="OptoPrimeV2",
            memory_size=0,
            calls=3,
        )
    else:
        result = run_meta(
            root / "O2", tier=2, optimizer="OptoPrimeV2", memory_size=0, calls=2
        )
    print(
        json.dumps(
            {
                k: result[k]
                for k in ["tier", "completed_responses", "utility", "selected_config"]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
