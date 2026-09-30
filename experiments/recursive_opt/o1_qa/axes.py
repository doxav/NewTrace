"""EXP21: bounded learning-configuration contrasts using the existing Trace engine."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import multiprocessing
import random
import re
import statistics
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

from opto.trace.nodes import GRAPH
from opto.trainer.objectives import EvaluationResult
from opto.trainer.loader import CurriculumBuffer
from opto.features.recursive_opt.levels import is_invalid_score

from . import campaign as C, campaign_analysis as A, prepare, task

ROOT = prepare.ROOT / "experiments/recursive_opt/_shared/o1_learning/exp21"
MODULE = "exp21.module.hotpot@1"
EVALUATOR = "exp21.evaluator.hotpot@1"
DEV_SEEDS = [21011, 21023]
TEST_SEEDS = [21111, 21123, 21137, 21141, 21153, 21171]
CALLS = 6
WORKERS = 10
COUNTS = {
    "pilot_train": 24,
    "pilot_selection": 12,
    "dev_train": 36,
    "dev_selection": 16,
    "dev_probe": 24,
    "train": 60,
    "validation": 24,
    "test": 48,
}
BASE = {
    "batch_size": 6,
    "history_size": 0,
    "trace_mode": "internal",
    "trace_detail": "full",
    "credit_horizon": "full",
    "surface": "all",
    "feedback": "localized",
    "goal": "default",
    "optimizer": "OptoPrimeV2",
    "memory_size": 0,
    "trainer": "priority",
}
DOMAINS = {
    "batch_size": [4, 6, 12],
    "history_size": [0, 1, 2],
    "trace_mode": ["internal", "otel", "sysmon", "hybrid"],
    "trace_detail": ["summary", "full"],
    "credit_horizon": ["step", "full"],
    "surface": ["all", "code", "prompt", "knobs", "knobs_then_all"],
    "feedback": ["localized", "scalar", "compact"],
    "goal": ["default", "explicit", "minimal"],
    "optimizer": ["OptoPrimeV2", "OPROv2"],
    "memory_size": [0, 3],
    "trainer": ["priority", "latest"],
}
AXES = {
    "batch": ["batch_size", "history_size"],
    "trace": ["trace_mode", "trace_detail", "credit_horizon"],
    "surface": ["surface"],
    "feedback": ["feedback"],
    "goal": ["goal"],
    "optimizer": ["optimizer", "memory_size"],
    "trainer": ["trainer"],
}


def read(path: Path) -> Any:
    """Read a scientific record without silently defaulting missing evidence."""
    return json.loads(path.read_text())


def validate_config(config: Mapping[str, Any]) -> None:
    """Accept only the preregistered native types and finite categorical domain."""
    if set(config) != set(BASE):
        raise ValueError(
            "learning configuration must specify exactly the frozen fields"
        )
    for key, values in DOMAINS.items():
        if type(config[key]) is not type(values[0]) or config[key] not in values:
            raise ValueError(f"configuration field outside frozen domain: {key}")


def variants() -> list[dict[str, Any]]:
    """Enumerate every one-axis contrast before any live result is observed."""
    changes = [("standard", "baseline", {})]
    changes += [(f"batch{n}", "batch", {"batch_size": n}) for n in (4, 12)]
    changes += [(f"curriculum{n}", "batch", {"history_size": n}) for n in (1, 2)]
    changes += [
        (f"trace_{m}", "trace", {"trace_mode": m}) for m in ("otel", "sysmon", "hybrid")
    ]
    changes += [
        ("trace_summary", "trace", {"trace_detail": "summary"}),
        ("trace_step", "trace", {"credit_horizon": "step"}),
    ]
    changes += [
        (f"surface_{m}", "surface", {"surface": m})
        for m in ("code", "prompt", "knobs", "knobs_then_all")
    ]
    changes += [
        (f"feedback_{m}", "feedback", {"feedback": m}) for m in ("scalar", "compact")
    ]
    changes += [(f"goal_{m}", "goal", {"goal": m}) for m in ("explicit", "minimal")]
    changes += [
        ("optimizer_opro", "optimizer", {"optimizer": "OPROv2"}),
        ("optimizer_memory3", "optimizer", {"memory_size": 3}),
    ]
    changes += [("trainer_latest", "trainer", {"trainer": "latest"})]
    return [
        {"id": name, "axis": axis, "config": {**BASE, **delta}}
        for name, axis, delta in changes
    ]


def targets(config: Mapping[str, Any], responses: int) -> list[str]:
    """Activate the registered surface; staged release occurs after two responses."""
    surface = config["surface"]
    if surface == "all" or surface == "knobs_then_all" and responses >= 2:
        return list(task.INITIAL)
    return {
        "code": ["ranker_source"],
        "prompt": ["answer_instruction"],
        "knobs": ["top_k", "bridge_expansion"],
        "knobs_then_all": ["top_k", "bridge_expansion"],
    }[surface]


def identity_keys(row: dict[str, Any]) -> set[str]:
    """Exclude IDs, normalized questions and supporting-title pairs across experiments."""
    return {
        "id:" + row["id"],
        "q:" + " ".join(row["question"].lower().split()),
        "support:" + task.digest(sorted({p[0] for p in row["supporting_facts"]})),
    }


@lru_cache(maxsize=1)
def panels() -> dict[str, list[dict[str, Any]]]:
    """Reconstruct new balanced splits, excluding every EXP20 pilot/train/validation/test row."""
    excluded = set().union(
        *(identity_keys(r) for rows in prepare.panels().values() for r in rows)
    )
    remaining = [r for r in prepare.load_rows() if not identity_keys(r) & excluded]
    return task.split_rows(remaining, COUNTS, seed=21001)


def stable_request(request: dict[str, Any]) -> dict[str, Any]:
    """Normalize random telemetry IDs only; retain all labels, sources and values."""
    result = copy.deepcopy(request)
    for message in result["messages"]:
        text = message["content"]
        for identity, label in re.findall(r'"id": "([^"]+)", "label": "([^"]+)"', text):
            text = text.replace(json.dumps(identity), json.dumps(label))
        message["content"] = text
    return C.canonical_request(result)


def replay_identity(request: dict[str, Any]) -> dict[str, Any]:
    """Compare replay data, ignoring renderer order and truncated process identities."""
    value = stable_request(request)
    for message in value["messages"]:
        message["content"] = re.sub(
            r'("id": ")[0-9]+(?=\.\.\.\(skipped due to length limit\))',
            r"\1<process-id>",
            message["content"],
        )
        text = message["content"]
        start = re.search(r"^# Inputs\n", text, re.M)
        end = re.search(r"^# Feedback\n", text, re.M)
        if start is None or end is None or start.start() >= end.start():
            continue
        graph = text[start.start() : end.start()]
        pattern = r'<node name="[^"]+"[^>]*>.*?</node>'
        blocks = sorted(re.findall(pattern, graph, re.S))
        remainder = re.sub(pattern, "", graph, flags=re.S)

        def decoration(value: str) -> str:
            """Remove only escaped renderer headings and normalize their line separators."""
            value = re.sub(r"(?:\\n)+# Others\\n", r"\\n", value)
            return re.sub(r"(?:\\n|\n)+", r"\\n", value)

        # Keep every full node (including multiplicity), code/instructions before
        # the graph, and feedback values. This is never submitted as a new prompt.
        message["content"] = json.dumps(
            [
                text[: start.start()],
                blocks,
                decoration(remainder),
                decoration(text[end.start() :]),
            ]
        )
    return value


class Journal(C.Journal):
    """Retain EXP20 transport/cache semantics with external trace replay normalization."""

    def client(
        self, profile: Mapping[str, Any], role: str, provider: Any = None
    ) -> Any:
        """Wrap only optimizer serialization, leaving all live calls and receipts metered."""
        original = super().client(profile, role, provider)
        if role != "optimizer":
            return original
        journal = self

        class Client:
            def __call__(self, **kwargs: Any) -> Any:
                """Pass a deterministic serialized request to the existing journal."""
                request = stable_request(kwargs)
                folder = journal.directory / "optimizer" / f"{journal.calls + 1:02d}"
                previous = sorted(folder.glob("**/request.json"))
                if previous:
                    recorded = read(previous[0])["request"]
                    if replay_identity(recorded) == replay_identity(request):
                        request = recorded
                return original(**request)

            def __deepcopy__(self, memo: dict[int, Any]) -> Any:
                """Share the actual response budget across candidate copies."""
                return self

        return Client()


def activate(module: Any) -> None:
    """Apply native trainable flags to the exact current source nodes."""
    while not isinstance(module, task.QAPolicy):
        module = module.module
    allowed = targets(
        {"surface": module.surface_mode}, C.ACTIVE.calls if C.ACTIVE else 0
    )
    for name in task.INITIAL:
        getattr(module, name).trainable = name in allowed


class Policy(C.RecordedPolicy):
    """Keep raw evaluated sources while applying a declared dynamic surface schedule."""

    def __init__(
        self, artifact: Mapping[str, Any], reader: Any, surface_mode: str
    ) -> None:
        """Bind the schedule from canonical module inputs instead of process-global configuration."""
        super().__init__(artifact, reader)
        self.surface_mode = surface_mode

    def forward(self, example: Any) -> Any:
        """Activate surface before constructing the current execution graph."""
        activate(self)
        return super().forward(example)


class ValidCurriculum(CurriculumBuffer):
    """Use the existing curriculum transition algorithm after removing typed invalidity sentinels."""

    def observe(self, scores: Mapping[int, float]) -> None:
        """Invalid programs are missing observations, never wrong-answer learning examples."""
        super().observe(
            {
                index: value
                for index, value in scores.items()
                if not is_invalid_score(value)
            }
        )


class Trainer(C.CampaignTrainer):
    """Production PrioritySearch with recording hooks, not a second optimization loop."""

    def train(self, *args: Any, parent_rule: str = "priority", **kwargs: Any) -> Any:
        """Resolve the parent-selection rule from canonical trainer kwargs."""
        if parent_rule not in {"priority", "latest"}:
            raise ValueError("parent_rule must be priority or latest")
        self.parent_rule = parent_rule
        return super().train(*args, **kwargs)

    def sample(self, agents: Any, verbose: bool = False, **kwargs: Any) -> Any:
        """Install the typed observation adapter before the first curriculum observation."""
        loader = self.train_sampler.loader
        if loader.curriculum is not None and not isinstance(
            loader.curriculum, ValidCurriculum
        ):
            if loader.curriculum.events or loader.curriculum.failed:
                raise RuntimeError("curriculum adapter must precede all observations")
            loader.curriculum = ValidCurriculum(
                history_size=loader.curriculum.history_size,
                success_threshold=loader.curriculum.success_threshold,
            )
        return super().sample(agents, verbose=verbose, **kwargs)

    def propose(self, samples: Any, verbose: bool = False, **kwargs: Any) -> Any:
        """Refresh the native surface and label proposal age for the latest-parent variant."""
        activate(self.agent)
        result = super().propose(samples, verbose=verbose, **kwargs)
        for candidate in result:
            candidate.exp21_slot = C.ACTIVE.calls
        return result

    def compute_exploitation_priority(self, candidate: Any) -> float:
        """Compare TRAIN scores or always advance to the latest valid proposal, as declared."""
        value = super().compute_exploitation_priority(candidate)
        if (
            self.parent_rule == "latest"
            and value is not None
            and math.isfinite(value)
            and value >= 0.0
        ):
            return float(getattr(candidate, "exp21_slot", 0))
        return value

    def compute_exploration_priority(self, candidate: Any) -> float:
        """Use the same declared parent rule for the production exploration queue."""
        if self.parent_rule == "latest":
            return self.compute_exploitation_priority(candidate)
        return super().compute_exploration_priority(candidate)


def evaluate(output: Any, row: Any, context: Mapping[str, Any]) -> EvaluationResult:
    """Change only feedback information, never the exact-match scientific metric."""
    value = getattr(output, "data", output)
    result = C.evaluate(output, row, context)
    if not result.valid or context.get("phase") != "fit":
        return result
    mode = context.get("inputs", {}).get("feedback", "localized")
    if mode == "localized":
        return result
    from dataclasses import replace

    if mode == "scalar":
        feedback = f"answer_EM={result.metrics['accuracy']}"
    else:
        feedback = json.dumps(
            {
                "correct": bool(result.metrics["accuracy"]),
                "expected": row["answer"],
                "observed": value["answer"],
                "format_valid": value["format_valid"],
                "support_recall": result.metrics["support_recall"],
            },
            ensure_ascii=False,
        )
    return replace(result, feedback=feedback)


def register() -> None:
    """Register only the experiment module/evaluator and narrow production trainer hooks."""
    from . import resume

    resume.install(ROOT)
    C.register()
    import opto.trainer.algorithms as algorithms

    algorithms.EXP21Trainer = Trainer
    if MODULE not in task.S._MODULE_REGISTRY:
        task.S.register_module(
            MODULE,
            task.S.ModuleRegistryEntry(
                build=lambda level, resources: Policy(
                    level["module"]["config"],
                    resources["llm_clients"].get("forward"),
                    level["module"]["inputs"]["surface_mode"],
                ),
                snapshot=task.snapshot,
                restore=task.restore,
                validate_artifact=C.validate_record,
                validate_config=task.validate_artifact,
                capabilities=frozenset(
                    {"multi_component", "json_snapshot", "trace_module"}
                ),
            ),
        )
        task.S.register_evaluator(EVALUATOR, evaluate)


def specification(
    config: Mapping[str, Any], train: list[dict[str, Any]], *, seed: int, calls: int
) -> dict[str, Any]:
    """Map each tunable learning choice into the canonical control-plane dict."""
    validate_config(config)
    value = task.specification(train, seed=seed, curriculum=False, calls=calls)
    value["budget"].update(eval_llm_calls=2000, evaluator_runs=2000)
    level = value["levels"][0]
    level["module"]["ref"] = MODULE
    level["module"]["inputs"] = {
        "feedback": config["feedback"],
        "surface_mode": config["surface"],
    }
    level["surface"]["targets"] = targets(config, 0)
    level["objective"]["evaluator_ref"] = EVALUATOR
    level["objective"]["trace_config"].update(
        mode=config["trace_mode"],
        detail=config["trace_detail"],
        credit_horizon=config["credit_horizon"],
    )
    engine = level["engine"]["config"]
    engine["trainer"] = "EXP21Trainer"
    engine["optimizer"] = config["optimizer"]
    engine["optimizer_kwargs"] = {"memory_size": config["memory_size"], "log": False}
    goals = {
        "explicit": "Improve answer exact-match on new HotpotQA questions. Diagnose retrieval omissions versus answer formatting or reasoning using TRAIN feedback. Change only enabled variables; preserve their native types. Prefer concise general rules, never memorized question-answer pairs. Output complete executable ranker code when changing it, preserving rank(question, passages).",
        "minimal": "Improve exact-match accuracy. Change as few enabled components as justified by the observed failures. Preserve correct behavior and legal native parameter types; never memorize question-answer pairs.",
    }
    if config["goal"] in goals:
        engine["optimizer_kwargs"]["objective"] = goals[config["goal"]]
    trainer = engine["trainer_kwargs"]
    trainer["batch_size"] = config["batch_size"]
    trainer["parent_rule"] = config["trainer"]
    if config["history_size"]:
        trainer["curriculum"] = {
            "history_size": config["history_size"],
            "success_threshold": 1.0,
        }
    return value


def fit(
    root: Path,
    config: dict[str, Any],
    train: list[dict[str, Any]],
    *,
    seed: int,
    name: str,
    calls: int = CALLS,
    factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run one bounded production chain; reuse raw completed responses on interruption."""
    register()
    folder = root / "chains" / str(seed) / name
    expected = {
        "config": config,
        "train_hash": task.digest(train),
        "seed": seed,
        "calls": calls,
    }
    C.retain(folder / "identity.json", expected)
    if (folder / "result.json").exists():
        return read(folder / "result.json")
    GRAPH.clear()
    C.ACTIVE = Journal(folder, root / "reader_cache" / str(seed), seed, calls)
    spec = specification(config, train, seed=seed, calls=calls)
    C.retain(folder / "spec.json", json.loads(json.dumps(spec)))
    C.retain(folder / "artifacts" / f"{task.digest(task.INITIAL)}.json", task.INITIAL)

    def clients(profile: Any, role: str) -> Any:
        """Connect the role-specific provider through the canonical metered path."""
        return C.ACTIVE.client(
            profile, role, factory(profile, role) if factory else None
        )

    production = task.S.run_spec(
        spec, resources={"llm_factory": clients, "trainer": "EXP21Trainer"}
    ).to_dict()
    terminal_candidate_failure = (
        production.get("error") == "invalid candidate evaluation"
        and production.get("evaluation", {}).get("valid") is False
        and production.get("evaluation", {}).get("status") == "invalid"
    )
    if (
        C.ACTIVE.failure
        or C.ACTIVE.calls != calls
        or (production.get("error") and not terminal_candidate_failure)
    ):
        C.retain(
            folder / "incomplete.json",
            {
                "calls": C.ACTIVE.calls,
                "failure": C.ACTIVE.failure,
                "production": production,
            },
        )
        raise RuntimeError(
            "incomplete production run; preserve completed evidence before resuming"
        )
    records = [read(p) for p in sorted((folder / "evaluations").glob("*.json"))]
    report = {
        "config": config,
        "seed": seed,
        "name": name,
        "optimizer_responses": C.ACTIVE.calls,
        "actual_train_evaluations": len(records),
        "reader_logical_calls": C.ACTIVE.reader_calls,
        "candidate_count": len(list((folder / "artifacts").glob("*.json"))),
        "batch_events": C.ACTIVE.batches,
        "curriculum_events": production["metadata"]["curriculum_events"],
        "unique_questions_in_feedback": len(
            {i for e in C.ACTIVE.batches for i in e["row_ids"]}
        ),
        "production": production,
    }
    C.retain(folder / "result.json", report)
    return report


def assess_panel(
    root: Path,
    seed: int,
    artifact: dict[str, Any],
    rows: list[dict[str, Any]],
    split: str,
    *,
    deployment: bool = False,
) -> dict[str, Any]:
    """Serialize only identical artifact panels; different candidate evaluations remain parallel."""
    lock_path = root / "evaluation_locks" / str(seed) / split / task.digest(artifact)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return A.evaluate_panel(
            root, seed, artifact, rows, split, deployment=deployment
        )


def select(
    root: Path,
    seed: int,
    name: str,
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    calls: int,
) -> dict[str, Any]:
    """Reuse the engine-independent evaluator; validation never reaches generation."""
    menu = A.candidate_menu(root, seed, name)
    rows = []
    for key, item in menu.items():
        training = assess_panel(root, seed, item["artifact"], train, "train_complete")
        selection = assess_panel(root, seed, item["artifact"], validation, "validation")
        rows.append(
            {
                "hash": key,
                "first_slot": item["first_slot"],
                "valid": training["valid"] and selection["valid"],
                "accuracy": selection["accuracy"],
                "train": training,
                "validation": selection,
            }
        )
    value = {
        "candidates": menu,
        "evaluations": rows,
        "prefixes": A.select_prefixes(rows, calls),
    }
    C.retain(root / "selection" / str(seed) / f"{name}.json", value)
    return value


def measure(
    root: Path,
    seed: int,
    selected: dict[str, Any],
    rows: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    """Score frozen selected programs with shared seed fallback; retain every prefix."""
    if split == "test":
        gate = read(root / "test_gate.json")
        allowed = {
            h for s in gate["selections"][str(seed)].values() for h in s["prefixes"]
        }
        if not set(selected["prefixes"]) <= allowed:
            raise ValueError("unfrozen program cannot access TEST")
    evaluated = {
        h: assess_panel(
            root,
            seed,
            selected["candidates"][h]["artifact"],
            rows,
            split,
            deployment=True,
        )
        for h in dict.fromkeys(selected["prefixes"])
    }
    curve = [evaluated[h]["accuracy"] for h in selected["prefixes"]]
    return {
        "curve": curve,
        "primary": statistics.mean(curve),
        "final": curve[-1],
        "first_target_prefix": next((i for i, v in enumerate(curve) if v >= 0.7), None),
        "evaluations": evaluated,
    }


def freeze_test(
    root: Path,
    names: list[str],
    seeds: list[int],
    selections: dict[str, Any],
    *,
    calls: int,
) -> None:
    """Require every configuration, seed, source and prefix before any TEST evaluation."""
    if set(selections) != set(map(str, seeds)) or any(
        set(v) != set(names) for v in selections.values()
    ):
        raise ValueError("complete configuration/seed selections required")
    for per_seed in selections.values():
        for selected in per_seed.values():
            if len(selected["prefixes"]) != calls + 1 or any(
                h not in selected["candidates"] for h in selected["prefixes"]
            ):
                raise ValueError("complete prefix selections required")
            for key, item in selected["candidates"].items():
                if key != task.digest(item["artifact"]):
                    raise ValueError("source hash mismatch")
    C.retain(
        root / "test_gate.json",
        {"names": names, "seeds": seeds, "calls": calls, "selections": selections},
    )
    if not (root / "test_gate_time.json").exists():
        C.retain(
            root / "test_gate_time.json",
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "hash": task.digest(read(root / "test_gate.json")),
            },
        )


def decode_meta(components: Mapping[str, Any]) -> dict[str, Any]:
    """Validate native learning-policy components without an intermediate JSON string."""
    config = dict(components)
    validate_config(config)
    return config


def job(args: dict[str, Any]) -> dict[str, Any]:
    """Run one isolated seed/configuration task; outer concurrency cannot corrupt Trace globals."""
    from experiments.recursive_opt._shared.optimizer_discovery.phase0 import _load_key

    _load_key()
    root, seed, name = Path(args["root"]), args["seed"], args["name"]
    data = panels()
    mode = args["mode"]
    calls = args.get("calls", CALLS)
    if mode in {"dev", "pilot", "fit"}:
        train_name = {"dev": "dev_train", "pilot": "pilot_train", "fit": "train"}[mode]
        fit(root, args["config"], data[train_name], seed=seed, name=name, calls=calls)
        if mode == "fit":
            return {"seed": seed, "name": name, "fit": "complete"}
        selected = select(
            root,
            seed,
            name,
            data[train_name],
            data["dev_selection" if mode == "dev" else "pilot_selection"],
            calls,
        )
        result = measure(
            root,
            seed,
            selected,
            data["dev_probe" if mode == "dev" else "pilot_selection"],
            "dev_probe" if mode == "dev" else "pilot_measurement",
        )
        C.retain(root / "reports" / str(seed) / f"{name}.json", result)
        return {
            "seed": seed,
            "name": name,
            "primary": result["primary"],
            "final": result["final"],
        }
    if mode == "select":
        select(root, seed, name, data["train"], data["validation"], calls)
        return {"seed": seed, "name": name, "selection": "complete"}
    if mode == "test":
        selected = read(root / "test_gate.json")["selections"][str(seed)][name]
        result = measure(root, seed, selected, data["test"], "test")
        C.retain(root / "reports" / str(seed) / f"{name}.json", result)
        return {
            "seed": seed,
            "name": name,
            "primary": result["primary"],
            "final": result["final"],
        }
    raise ValueError("unknown study job mode")


def run_jobs(jobs: list[dict[str, Any]], workers: int = WORKERS) -> None:
    """Use bounded isolated processes; report progress and preserve failed job identities."""
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        pending = {pool.submit(job, item): item for item in jobs}
        errors = []
        for future in as_completed(pending):
            item = pending[future]
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as error:
                errors.append(
                    {
                        "job": item,
                        "error_type": type(error).__name__,
                        "error": str(error)[:200],
                    }
                )
                print(
                    json.dumps(
                        {
                            "failed": item["name"],
                            "seed": item["seed"],
                            "type": type(error).__name__,
                        }
                    ),
                    flush=True,
                )
        if errors:
            path = (
                ROOT
                / f"failures_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
            )
            C.retain(path, errors)
            raise RuntimeError(
                f"{len(errors)} incomplete jobs; completed evidence retained"
            )


def sources() -> dict[str, str]:
    """Freeze existing source dependencies plus the narrow EXP21 adapters."""
    names = list(A.source_hashes()) + [
        str(Path(__file__).relative_to(prepare.ROOT)),
        "experiments/recursive_opt/o1_qa/meta.py",
        "experiments/recursive_opt/o1_qa/resume.py",
        "opto/optimizers/opro_v2.py",
        "opto/trace/io/telemetry_session.py",
        "opto/trace/io/sysmonitoring.py",
        "opto/trace/io/otel_adapter.py",
    ]
    return {
        n: hashlib.sha256((prepare.ROOT / n).read_bytes()).hexdigest()
        for n in names
        if (prepare.ROOT / n).exists()
    }


def freeze() -> None:
    """Seal protocol and source bytes before any scientific call in this stage."""
    config = {
        "experiment": "EXP21",
        "variants": variants(),
        "base": BASE,
        "domains": DOMAINS,
        "source_hashes": sources(),
        "panels": {
            n: {"count": len(v), "ids": [r["id"] for r in v], "hash": task.digest(v)}
            for n, v in panels().items()
        },
        "dev_seeds": DEV_SEEDS,
        "test_seeds": TEST_SEEDS,
        "responses": CALLS,
        "workers": WORKERS,
        "models": A.profiles(),
        "primary": "mean exact-match over validation-selected prefix policies 0..6; higher better",
        "target": 0.7,
        "bootstrap": "paired outer-seed bootstrap 10000 draws seed20099; exploratory, no unadjusted significance claims across axes",
        "budget": "same completed optimizer responses; batch variation intentionally changes reader/evaluation budget, report separately",
        "axis8": "genuine O2 chooses O1 optimizer/memory; each O1 evaluates configuration by running production O0 on DEV only",
        "meta": {
            "O1_responses": 3,
            "O2_responses": 2,
            "outer_optimizers": ["OptoPrimeV2", "OPROv2"],
            "outer_memory": [0, 3],
        },
        "selection": "one best nonbaseline configuration per axis on mean DEV probe AUC, registry-order ties; include baseline; combine only positive DEV deltas; ablate each active axis; all remain exploratory until fresh confirmation",
        "confirmation": "baseline, seven per-axis winners, combined, automatic O1 and recursive O2 configurations; deduplicate identical configs, preserve aliases; six new seeds; all selections freeze before TEST",
        "failure": "invalid source typed and ineligible; completed empty response consumes a slot; common initial-policy TEST fallback; infrastructure failure blocks affected job, never silently scored",
        "resume": "immutable requests/responses; ambiguous remote completion blocks replacement; only explicit transport rejections retried",
        "scope": "finite systematic menu, not a global optimum; hyperparameter discovery, not optimizer algorithm invention; no Project1 or independent-generation comparison",
    }
    C.retain(ROOT / "manifest.json", config)
    archive = ROOT / "sources.zip"
    if not archive.exists():
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
            for name in config["source_hashes"]:
                z.write(prepare.ROOT / name, name)


def verify() -> None:
    """Refuse changed code or data under an existing scientific manifest."""
    manifest = read(ROOT / "manifest.json")
    amendment = ROOT / "engineering_amendment.json"
    if (ROOT / "engineering_amendment_v2.json").exists():
        amendment = ROOT / "engineering_amendment_v2.json"
    if (ROOT / "engineering_amendment_v3.json").exists():
        amendment = ROOT / "engineering_amendment_v3.json"
    if (ROOT / "engineering_amendment_v4.json").exists():
        amendment = ROOT / "engineering_amendment_v4.json"
    if (ROOT / "engineering_amendment_v5.json").exists():
        amendment = ROOT / "engineering_amendment_v5.json"
    if (ROOT / "engineering_amendment_v6.json").exists():
        amendment = ROOT / "engineering_amendment_v6.json"
    if (ROOT / "engineering_amendment_v7.json").exists():
        amendment = ROOT / "engineering_amendment_v7.json"
    if (ROOT / "engineering_amendment_v8.json").exists():
        amendment = ROOT / "engineering_amendment_v8.json"
    expected = (
        read(amendment)["source_hashes"]
        if amendment.exists()
        else manifest["source_hashes"]
    )
    if sources() != expected:
        raise ValueError(
            "source drift requires a prospective amendment before live collection"
        )
    for name, rows in panels().items():
        if task.digest(rows) != manifest["panels"][name]["hash"]:
            raise ValueError("split drift")


def main() -> None:
    """Dispatch bounded stages, keeping development and confirmation evidence separate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "pilot", "screen"])
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--pilot-retest", action="store_true")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
        print(
            json.dumps(
                {"frozen": str(ROOT / "manifest.json"), "variants": len(variants())}
            )
        )
        return
    verify()
    if args.stage == "pilot":
        names = [
            "standard",
            "trace_hybrid",
            "surface_knobs_then_all",
            "optimizer_opro",
            "trainer_latest",
            "curriculum2",
        ]
        jobs = [
            {
                "root": str(ROOT / "pilot"),
                "seed": 21003,
                "name": v["id"],
                "config": v["config"],
                "mode": "pilot",
                "calls": 2,
            }
            for v in variants()
            if v["id"] in names
        ]
        if args.pilot_retest:
            jobs = [
                {**j, "root": str(ROOT / "pilot_retest"), "seed": 21005}
                for j in jobs
                if j["name"] == "trainer_latest"
            ]
    else:
        jobs = [
            {
                "root": str(ROOT / "development"),
                "seed": s,
                "name": v["id"],
                "config": v["config"],
                "mode": "dev",
            }
            for s in DEV_SEEDS
            for v in variants()
        ]
        random.Random(21009).shuffle(jobs)
    run_jobs(jobs, args.workers)


if __name__ == "__main__":
    main()
