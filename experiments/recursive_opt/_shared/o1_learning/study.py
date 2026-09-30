"""EXP-19 task adapter and bounded O1 screen using the production control plane."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

from opto.features.recursive_opt import spec as S
from opto.features.recursive_opt.runmode import _response_usage, make_live_llm
from opto.trace import bundle, node
from opto.trainer.objectives import EvaluationResult

ROOT = Path(__file__).parent
RULES = dict(zip("ABCDEF", ["0001", "0111", "0110", "1001", "1101", "1110"]))
START = {name: "0000" for name in RULES}
MODEL = "deepseek/deepseek-v4-flash-0731"
TARGET = 0.9
RECORDS: list[dict[str, Any]] = []
CLIENT: Any = None
RECORD_LOCK = threading.Lock()


def digest(value: Any) -> str:
    """Hash canonical JSON for configurations and deterministic instance identity."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def persist(path: Path, value: Any) -> None:
    """Refuse to overwrite evidence; write one exclusive JSON record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, default=str, allow_nan=False)
        handle.write("\n")


def predict(rules: Mapping[str, str], tree: Any) -> int:
    """Execute a bounded Boolean expression under explicit four-bit truth tables."""
    if isinstance(tree, int):
        return tree
    name, left, right = tree
    table = rules[name]
    if not isinstance(table, str) or len(table) != 4 or set(table) - {"0", "1"}:
        raise ValueError("truth table must be exactly four binary digits")
    return int(table[2 * predict(rules, left) + predict(rules, right)])


def panels(seed: int) -> dict[str, list[dict[str, Any]]]:
    """Generate disjoint primitive/train and composed validation/test expressions."""
    rng = random.Random(seed)

    def tree(depth: int) -> Any:
        if depth == 0:
            return rng.randrange(2)
        return [rng.choice(list(RULES)), tree(depth - 1), tree(depth - 1)]

    train = [[name, a, b] for name in RULES for a in range(2) for b in range(2)]
    rng.shuffle(train)
    used = {digest(item) for item in train}
    result = {"train": train}
    for split, count, depth in [("validation", 24, 2), ("test", 48, 3)]:
        result[split] = []
        while len(result[split]) < count:
            item = tree(depth)
            if digest(item) not in used:
                used.add(digest(item))
                result[split].append(item)
    return {
        name: [{"tree": item, "expected": predict(RULES, item)} for item in values]
        for name, values in result.items()
    }


def score(rules: Mapping[str, str], panel: list[dict[str, Any]]) -> float:
    """Compute deterministic exact-match accuracy, without replacing invalidity."""
    return statistics.mean(
        predict(rules, item["tree"]) == item["expected"] for item in panel
    )


def first_hit(curve: list[float | None], target: float = TARGET) -> int | None:
    """Return observed proposal count to target, preserving nonattainment as None."""
    return next(
        (
            index
            for index, value in enumerate(curve)
            if value is not None and value >= target
        ),
        None,
    )


@bundle()
def apply_rule(table: str, left: int, right: int) -> int:
    """Apply table indexed by 2*left+right, in order 00,01,10,11."""
    if len(table) != 4 or set(table) - {"0", "1"}:
        raise ValueError("truth table must contain four binary digits")
    return int(table[2 * left + right])


@bundle()
def decode_rules(text: str) -> dict[str, str]:
    """Decode the alternative single-parameter JSON policy representation."""
    value = json.loads(text)
    if set(value) != set("ABCDEF"):
        raise ValueError("policy must define A through F")
    return value


@bundle()
def pack_prediction(prediction: int, configuration: dict[str, str]) -> dict[str, Any]:
    """Retain the actual prediction and exact parameter state for replay."""
    return {"prediction": prediction, "configuration": configuration}


class BooleanPolicy(S._ComponentModule):
    """A reusable interpreter with trainable tables and real per-operation Trace nodes."""

    def forward(self, inputs: Any) -> Any:
        """Evaluate composition through traced lookups; never execute generated code."""
        item = getattr(inputs, "data", inputs)
        rules = (
            decode_rules(self.components["rules"])
            if "rules" in self.components
            else self.components
        )
        configuration = (
            json.loads(self.components["rules"].data)
            if "rules" in self.components
            else {k: v.data for k, v in self.components.items()}
        )

        def execute(tree: Any) -> Any:
            if isinstance(tree, int):
                return node(tree)
            name, left, right = tree
            return apply_rule(rules[name], execute(left), execute(right))

        return pack_prediction(execute(item["tree"]), configuration)


def evaluate(output: Any, example: Any, context: Mapping[str, Any]) -> EvaluationResult:
    """Provide a fixed score and separately configurable training explanation."""
    value = getattr(output, "data", output)
    result = float(value["prediction"] == example["expected"])
    explanation = f"expected {example['expected']}, observed {value['prediction']} for {example['tree']}"
    if context["inputs"].get("feedback") == "scalar":
        explanation = f"accuracy={result}"
    with RECORD_LOCK:
        RECORDS.append(
            {
                "calls": CLIENT.calls if CLIENT else 0,
                "configuration": value["configuration"],
                "phase": context["phase"],
                "score": result,
                "tree_hash": digest(example["tree"]),
            }
        )
    return EvaluationResult(True, "ok", {"accuracy": result}, explanation)


def register() -> None:
    """Register only a task adapter; reuse the production trainer and module persistence."""
    if "exp19.module.boolean@1" not in S._MODULE_REGISTRY:
        S.register_module(
            "exp19.module.boolean@1",
            S.ModuleRegistryEntry(
                build=lambda spec, resources: BooleanPolicy(
                    spec["module"]["config"]["components"], spec["module"]["inputs"]
                ),
                snapshot=S._snapshot_components,
                restore=S._restore_components,
                validate_artifact=S._validate_component_artifact,
                validate_config=S._validate_component_config,
                capabilities=frozenset(
                    {"multi_component", "json_snapshot", "trace_module"}
                ),
            ),
        )
        S.register_evaluator("exp19.evaluator.boolean@1", evaluate)


def variants() -> list[dict[str, Any]]:
    """Freeze one-factor engineering contrasts; combined choices come from development only."""
    base = {
        "batch_size": 3,
        "mode": "internal",
        "detail": "summary",
        "credit_horizon": "episode",
        "surface": "multi",
        "feedback": "localized",
        "goal": "generic",
        "optimizer": "OptoPrimeV2",
        "trainer": "SequentialSearch",
    }
    changes = [("standard", "baseline", {})]
    changes += [(f"batch{n}", "batch", {"batch_size": n}) for n in [1, 5, 7]]
    changes += [
        (
            f"curriculum{n}",
            "batch",
            {
                "batch_size": n,
                "curriculum": {"history_size": min(4, n - 1), "success_threshold": 1.0},
            },
        )
        for n in [3, 5, 7]
    ]
    changes += [
        (mode, "trace", {"mode": mode}) for mode in ["otel", "sysmon", "hybrid"]
    ]
    changes += [
        ("trace_full", "trace", {"detail": "full", "credit_horizon": "full"}),
        ("trace_step", "trace", {"credit_horizon": "step"}),
    ]
    changes += [
        ("joint", "surface", {"surface": "joint"}),
        ("scalar", "feedback", {"feedback": "scalar"}),
        ("goal_explicit", "goal", {"goal": "explicit"}),
        ("opro", "optimizer", {"optimizer": "OPROv2"}),
        ("priority", "trainer", {"trainer": "PrioritySearch"}),
    ]
    return [
        {"id": name, "axis": axis, **base, **change} for name, axis, change in changes
    ]


def specification(
    config: Mapping[str, Any], seed: int, *, calls: int = 4, live: bool = True
) -> dict[str, Any]:
    """Build the entire run from one canonical dict, keeping test out of fitting."""
    data = panels(seed)
    components = {"rules": json.dumps(START)} if config["surface"] == "joint" else START
    trainer_kwargs = {
        "batch_size": config["batch_size"],
        "num_threads": 1,
        "test_frequency": None,
    }
    if "curriculum" in config:
        trainer_kwargs["curriculum"] = config["curriculum"]
    optimizer_kwargs = {"log": False, "max_tokens": config.get("max_tokens", 8000)}
    if config["goal"] == "explicit":
        optimizer_kwargs["objective"] = (
            "Infer six unknown Boolean truth tables from observations. Change only bits constrained by feedback, preserve previously correct bits. Tables use order 00,01,10,11. Maximize accuracy on composed expressions."
        )
    level = {
        "id": "O0",
        "surface": {"kind": "module", "targets": ["*"]},
        "module": {
            "ref": "exp19.module.boolean@1",
            "config": {"components": components},
            "inputs": {"feedback": config["feedback"]},
        },
        "engine": {
            "name": "trace" if live else "fixed",
            "config": {
                "optimizer": config["optimizer"],
                "trainer": config["trainer"],
                "iterations": calls + 1,
                "num_candidates": 1,
                "optimizer_kwargs": optimizer_kwargs,
                "trainer_kwargs": trainer_kwargs,
            },
        },
        "objective": {
            "evaluator_ref": "exp19.evaluator.boolean@1",
            "intent": "Maximize exact-match accuracy on Boolean expressions.",
            "metrics": {
                "accuracy": {
                    "direction": "maximize",
                    "source": "evaluation.metrics.accuracy",
                    "aggregate_examples": "mean",
                }
            },
            "selection": {"mode": "scalar", "score_key": "accuracy"},
            "trace_config": {
                "mode": config["mode"],
                "detail": config["detail"],
                "credit_horizon": config["credit_horizon"],
                "semantic_names": ["forward", "execute", "apply_rule", "evaluate"],
                "max_nodes": 48,
                "max_chars": 4000,
            },
        },
        "datasets": {
            "train": data["train"],
            "validation": data["validation"],
            "holdout": [],
        },
        "llm_roles": {"optimizer": "deepseek" if live else None},
    }
    return {
        "schema_version": S.SCHEMA_VERSION,
        "kind": S.SPEC_KIND,
        "runtime": {"seed": seed, "offline": not live, "test_mode": True},
        "llm_profiles": {
            "deepseek": {
                "provider": "openrouter",
                "model": MODEL,
                "temperature": 0.6,
                "max_tokens": config.get("max_tokens", 8000),
                "request_timeout_s": 300,
                "transport_max_attempts": 4,
                "transport_base_delay_s": 2,
                "request_params": {
                    "top_p": 1.0,
                    "extra_body": {"reasoning": {"effort": "low"}},
                },
            }
        },
        "budget": {"optimizer_llm_calls": calls, "on_exceed": "raise"},
        "levels": [level],
    }


class RecordingClient:
    """Persist real responses and exact prompts; never load credentials into artifacts."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.calls = 0
        self.transport_events: list[dict[str, Any]] = []
        self.client = make_live_llm(
            "openrouter/" + MODEL,
            max_retries=4,
            base_delay=2,
            request_timeout_s=300,
            allow_env_overrides=False,
            cache=False,
            empty_response_retries=0,
            budget_resource=None,
            retry_event_callback=self._record_transport_event,
        )
        self.usage: list[dict[str, Any]] = []

    def _record_transport_event(self, event: str, failure_kind: str | None) -> None:
        """Retain transport lifecycle labels without persisting exception text."""
        record = {"event": event, "failure_kind": failure_kind}
        persist(self.directory / f"transport_{len(self.transport_events)}.json", record)
        self.transport_events.append(record)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        index = self.calls
        request = {"args": args, "kwargs": kwargs}
        persist(self.directory / f"request_{index}.json", request)
        start = time.monotonic()
        response = self.client(*args, **kwargs)
        self.calls += 1
        usage = _response_usage(response)
        provider_usage = response.model_dump().get("usage", {})
        if provider_usage.get("cost") is not None:
            usage["cost_usd"] = provider_usage["cost"]
        usage["reasoning_tokens"] = (
            provider_usage.get("completion_tokens_details") or {}
        ).get("reasoning_tokens")
        self.usage.append(usage)
        persist(
            self.directory / f"response_{index}.json",
            {
                "response": response.model_dump(),
                "usage": usage,
                "wall_s": time.monotonic() - start,
            },
        )
        return response

    def __deepcopy__(self, memo: dict[int, Any]) -> RecordingClient:
        memo[id(self)] = self
        return self


def run(
    config: Mapping[str, Any], seed: int, phase: str, calls: int = 4
) -> dict[str, Any]:
    """Execute a registered production run and rescore preserved candidates on development."""
    global CLIENT
    path = ROOT / "raw" / phase / f"{config['id']}_{seed}"
    if (path / "result.json").exists():
        return json.loads((path / "result.json").read_text())
    if path.exists():
        raise RuntimeError(f"unfinished evidence requires reconciliation: {path.name}")
    register()
    RECORDS.clear()
    CLIENT = RecordingClient(path)
    spec = specification(config, seed, calls=calls)
    persist(path / "spec.json", spec)
    start = time.monotonic()
    try:
        result = S.run_spec(
            spec, resources={"llm_factory": lambda profile, role: CLIENT}
        )
        raw = result.to_dict()
    except Exception as error:
        raw = {
            "valid": False,
            "error_type": type(error).__name__,
            "error": S._safe_error(error),
        }
    persist(path / "production.json", raw)
    persist(path / "evaluations.json", RECORDS)
    candidates = {digest(START): {"configuration": START, "calls": 0}}
    for row in RECORDS:
        candidates.setdefault(digest(row["configuration"]), row)
    panel = panels(seed)["validation"]

    def assess(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        key, row = item
        try:
            value = score(row["configuration"], panel)
            return {
                "hash": key,
                "configuration": row["configuration"],
                "calls": row["calls"],
                "accuracy": value,
                "valid": True,
            }
        except (ValueError, KeyError, TypeError, IndexError):
            return {
                "hash": key,
                "configuration": row["configuration"],
                "calls": row["calls"],
                "accuracy": None,
                "valid": False,
            }

    with ThreadPoolExecutor(max_workers=8) as executor:
        assessed = list(executor.map(assess, candidates.items()))
    eligible = [row for row in assessed if row["valid"]]
    selected = max(eligible, key=lambda row: (row["accuracy"], -row["calls"]))
    curve = [
        max(row["accuracy"] for row in eligible if row["calls"] <= count)
        for count in range(CLIENT.calls + 1)
    ]
    report = {
        "phase": phase,
        "config": dict(config),
        "seed": seed,
        "valid": raw["valid"],
        "calls": CLIENT.calls,
        "usage": CLIENT.usage,
        "wall_s": time.monotonic() - start,
        "candidate_results": assessed,
        "selection": selected,
        "validation_curve": curve,
        "first_hit": first_hit(curve),
        "actual_trainer_evaluations": len(RECORDS),
        "extra_selection_evaluations": len(assessed) * len(panel),
        "error": raw.get("error"),
    }
    persist(path / "result.json", report)
    return report


def main() -> None:
    """Run one bounded live pilot or one preregistered individual contrast."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="standard")
    parser.add_argument("--seed", type=int, default=19001)
    parser.add_argument("--phase", default="pilot")
    parser.add_argument("--calls", type=int, default=4)
    args = parser.parse_args()
    from experiments.recursive_opt._shared.optimizer_discovery.phase0 import _load_key

    _load_key()
    config = next(item for item in variants() if item["id"] == args.variant)
    report = run(config, args.seed, args.phase, args.calls)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ["valid", "calls", "validation_curve", "first_hit", "error"]
            }
        )
    )


if __name__ == "__main__":
    main()
