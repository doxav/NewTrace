"""Successor study ownership reusing the frozen benchmark and production search.

The adapter generalizes registered arms and barriers; it does not implement a
search algorithm. C continues through the exact EXP-16 Trace/Control Plane path.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.evidence import environment
from experiments.recursive_opt._shared.optimizer_discovery.exp17 import evaluation_cache as K
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import generation as G
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import search_experiment as S

ROOT = Path(__file__).resolve().parent
B2_PATH = ROOT.parent / "investigation16/benchmark/b2/optimizer.py"


def configuration(
    *,
    experiment: str,
    namespace: str,
    arms: list[str],
    outer_seeds: list[int],
    slots: int = 8,
    budget: int = 32,
    local_replicates: int = 2,
    task_replicates: dict[str, int] | None = None,
    workers: int = 8,
) -> dict[str, Any]:
    """Build a draft allocation with explicit arms and the unchanged benchmark rules."""
    config = S.configuration(
        namespace=namespace,
        outer_seeds=outer_seeds,
        slots=slots,
        budget=budget,
        local_replicates=local_replicates,
        workers=workers,
        max_tokens=32000,
        task_replicates=task_replicates or {"train": 4, "validation": 2, "audit": 2},
        include_aggregate_auc=True,
        max_prompt_chars=524288,
    )
    config.update(
        {
            "experiment": experiment,
            "arms": list(arms),
            "schedules": {arm: [1, 1] for arm in arms if arm != "I"},
            "analysis_kind": "exp17" if experiment == "EXP-17" else "exp18",
            "representative_arm": (
                "C" if "C" in arms else "PM" if "PM" in arms else arms[0]
            ),
            "representative": "minimum_registered_arm_validation_auc_then_outer_order",
            "cache_version": "successor-evaluation-v1",
            "audit_controls": ["A0", "B2"],
            "model_config": {
                key: value
                for key, value in S._generation_settings(
                    config, outer_seeds[0], 0
                ).items()
                if key != "seed"
            },
            "memory_max_sources": 7,
            "memory_max_chars": 65536,
        }
    )
    validate_config(config)
    return config


def confirmatory_config() -> dict[str, Any]:
    """Return the registered 46-pair C-versus-I allocation on a new namespace."""
    return configuration(
        experiment="EXP-17",
        namespace="EXP17-CONFIRM-v1",
        arms=["I", "C"],
        outer_seeds=list(range(17001, 17047)),
    )


def validate_config(config: dict[str, Any]) -> None:
    """Reject unsupported identities, unequal schedules and unimplemented settings."""
    arms = config.get("arms", [])
    allowed = (
        {"I", "C"} if config.get("experiment") == "EXP-17" else {"L", "M", "P", "PM"}
    )
    if (
        config.get("experiment") not in {"EXP-17", "EXP-18"}
        or not isinstance(arms, list)
        or not arms
        or len(set(arms)) != len(arms)
        or not set(arms) <= allowed
        or not isinstance(config.get("namespace"), str)
        or not config["namespace"]
    ):
        raise ValueError("unregistered successor experiment or arms")
    for key in ("slots", "budget", "workers", "local_replicates", "max_prompt_chars"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError("invalid allocation: " + key)
    seeds = config.get("outer_seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or config["workers"] > 8
        or config["slots"] % 2
        or config["schedules"] != {arm: [1, 1] for arm in arms if arm != "I"}
    ):
        raise ValueError("invalid seeds, workers or response schedule")
    fixed = {
        "model": G.MODEL,
        "max_tokens": 32000,
        "generation_concurrency": 1,
        "timeout_s": 2,
        "cache_version": "successor-evaluation-v1",
        "audit_controls": ["A0", "B2"],
        "include_aggregate_auc": True,
        "selection": "eligible_min_validation_auc_then_seed_minus_one_or_slot",
        "fallback": "permanent_common_seed_actual_history_no_budget_reset",
        "bootstrap": B.MANIFEST["bootstrap"],
    }
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError("unsupported scientific setting")
    counts = config.get("task_replicates", {})
    if set(counts) != set(S.SPLITS) or any(
        type(n) is not int or n < 1 for n in counts.values()
    ):
        raise ValueError("all splits require registered positive stratum counts")
    expected_model = {
        key: value
        for key, value in S._generation_settings(config, seeds[0], 0).items()
        if key != "seed"
    }
    if config.get("model_config") != expected_model:
        raise ValueError("model settings differ from the tested compatibility path")
    if config.get("representative_arm") not in arms:
        raise ValueError("representative must use a registered arm")
    if config.get("analysis_kind") != (
        "exp17" if config["experiment"] == "EXP-17" else "exp18"
    ):
        raise ValueError("analysis does not match the registered experiment")
    if config["experiment"] == "EXP-18" and (
        config.get("memory_max_sources") != 7 or config.get("memory_max_chars") != 65536
    ):
        raise ValueError("memory limits differ from the registered mechanism")
    arm_order(config)


def arm_order(config: dict[str, Any]) -> dict[str, list[str]]:
    """Use adjacent inverse orders, or an explicitly registered balanced order list."""
    arms, seeds = config["arms"], config["outer_seeds"]
    supplied = config.get("arm_orders")
    if supplied is not None:
        if len(supplied) != len(seeds) or any(
            sorted(row) != sorted(arms) for row in supplied
        ):
            raise ValueError("arm orders must contain each registered arm exactly once")
        return {str(seed): list(row) for seed, row in zip(seeds, supplied)}
    result = {}
    for index, seed in enumerate(seeds):
        rotation = (index // 2) % len(arms)
        order = arms[rotation:] + arms[:rotation]
        result[str(seed)] = list(reversed(order)) if index % 2 else order
    return result


def prepare(
    root: Path,
    config: dict[str, Any],
    protocol: Path,
    *,
    extra_frozen_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Freeze exact sources, new task identities and controls before experimental calls."""
    validate_config(config)
    if E.exists(root / "freeze.json"):
        saved = preflight(root)
        if saved["config"] != config or saved["protocol_sha256"] != B.source_hash(
            protocol.read_text()
        ):
            raise RuntimeError("existing freeze differs from requested study")
        return saved
    tasks = {
        split: G.fresh_tasks(
            config["namespace"], split, config["task_replicates"][split]
        )
        for split in S.SPLITS
    }
    ids = [B.task_identity(task) for panel in tasks.values() for task in panel]
    if len(ids) != len(set(ids)):
        raise RuntimeError("new split identities overlap")
    old = E.read(ROOT.parent / "investigation16/production_run/freeze.json")
    # The predecessor receipt retains its original paths. A new freeze records
    # current source locations and hashes without rewriting that receipt.
    legacy_root = B.ROOT.parents[3] / "artifacts/optimizer_discovery"
    files = []
    for name in old["files"]:
        path = Path(name)
        if path.is_relative_to(legacy_root):
            path = B.ROOT / path.relative_to(legacy_root)
        files.append(path)
    files += list(Path("opto").rglob("*.py"))
    files += [
        Path(__file__),
        Path(K.__file__),
        B2_PATH,
        protocol,
        *(extra_frozen_paths or []),
    ]
    b2 = B2_PATH.read_text()
    frozen = {
        "schema": "optimizer_successor.study.v1",
        "created_ns": time.time_ns(),
        "config": config,
        "tasks": tasks,
        "seed_source": B.SEED_SOURCE,
        "seed_sha256": B.source_hash(B.SEED_SOURCE),
        "fixed_controls": {"B2": {"source": b2, "source_sha256": B.source_hash(b2)}},
        "protocol_sha256": B.source_hash(protocol.read_text()),
        "files": {
            str(path.resolve()): B.source_hash(path.read_text())
            for path in sorted(set(files))
        },
        "environment": environment(),
        "arm_order": arm_order(config),
        "panels": {
            split: {
                "tasks": len(tasks[split]),
                "trajectories": len(tasks[split]) * config["local_replicates"],
            }
            for split in S.SPLITS
        },
        "benchmark_manifest": B.MANIFEST,
        "local_seed_derivation": old["local_seed_derivation"],
    }
    I.persist(root / "freeze.json", frozen)
    I.persist(root / "freeze_sha256.json", {"sha256": B.digest(frozen)})
    return frozen


def preflight(root: Path) -> dict[str, Any]:
    """Refuse source, environment, task, configuration or control drift on resume."""
    frozen = E.read(root / "freeze.json")
    if B.digest(frozen) != E.read(root / "freeze_sha256.json")["sha256"]:
        raise RuntimeError("study freeze digest mismatch")
    config = frozen["config"]
    validate_config(config)
    for name, expected in frozen["files"].items():
        if B.source_hash(Path(name).read_text()) != expected:
            raise RuntimeError("study frozen source changed: " + name)
    tasks = {
        split: G.fresh_tasks(
            config["namespace"], split, config["task_replicates"][split]
        )
        for split in S.SPLITS
    }
    if (
        frozen["tasks"] != tasks
        or frozen["panels"]
        != {
            split: {
                "tasks": len(panel),
                "trajectories": len(panel) * config["local_replicates"],
            }
            for split, panel in tasks.items()
        }
        or frozen["environment"] != environment()
        or frozen["seed_source"] != B.SEED_SOURCE
        or frozen["seed_sha256"] != B.source_hash(B.SEED_SOURCE)
        or frozen["arm_order"] != arm_order(config)
        or frozen["benchmark_manifest"] != B.MANIFEST
        or frozen["fixed_controls"]["B2"]["source"] != B2_PATH.read_text()
        or frozen["fixed_controls"]["B2"]["source_sha256"]
        != B.source_hash(B2_PATH.read_text())
    ):
        raise RuntimeError("study freeze reconstruction mismatch")
    return frozen


def barrier_paths(root: Path, name: str) -> list[Path]:
    """Enumerate every mandatory phase record from the registered arm/slot grid."""
    config = E.read(root / "freeze.json")["config"]
    paths = []
    for outer in config["outer_seeds"]:
        for arm in config["arms"]:
            directory = root / "raw" / str(outer) / arm
            if name == "selections_frozen.json":
                paths.append(directory / "selection.json")
            else:
                paths += [
                    directory / item
                    for item in (
                        "generation_complete.json",
                        "allocations_train.json",
                        "seed_train_receipt.json",
                    )
                ]
                if arm != "I":
                    paths += [directory / "trace.json", directory / "schedule.json"]
                if arm in {"P", "PM"}:
                    paths += [
                        directory / f"parent_decisions/slot_{slot:02d}.json"
                        for slot in range(config["slots"] + 1)
                    ]
                for slot in range(config["slots"]):
                    names = ["request", "response", "train_receipt"]
                    if arm != "I":
                        names.append("propagated_feedback")
                    if config["experiment"] == "EXP-18":
                        names.append("current_context")
                    paths += [
                        directory / f"slot_{slot:02d}/{item}.json" for item in names
                    ]
    return paths


def verify_barrier(root: Path, name: str) -> dict[str, Any]:
    """Require complete immutable phase evidence before a protected split is opened."""
    if not E.exists(root / name):
        raise RuntimeError(
            "all generation must finish first"
            if name.startswith("generation")
            else "all selections must freeze first"
        )
    value = E.read(root / name)
    expected = {str(path.relative_to(root)) for path in barrier_paths(root, name)}
    if set(value["hashes"]) != expected:
        raise RuntimeError("phase barrier omits registered evidence")
    for relative, digest in value["hashes"].items():
        if B.digest(E.read(root / relative)) != digest:
            raise RuntimeError("phase barrier evidence changed")
    return value


class Study(S.SearchExperiment):
    """Generalized study owner; benchmark, Trace feedback and scheduling are inherited."""

    def __init__(self, root: Path, arm: str, *, client: Any = None) -> None:
        """Bind a registered arm without changing any previous experiment constructor."""
        self.frozen = preflight(root)
        self.config = self.frozen["config"]
        if arm not in self.config["arms"]:
            raise ValueError("unregistered study arm")
        self.run_root, self.root, self.arm = root, root / "raw", arm
        self.phase = self.config["namespace"] + ":" + arm
        self.seeds, self.slots = self.config["outer_seeds"], self.config["slots"]
        self.budget, self.client = self.config["budget"], client

    def panel(
        self, source: str, outer: int, split: str, *, deployment: bool = False
    ) -> list[dict[str, Any]]:
        """Apply registered split barriers before using the common frozen evaluator."""
        if split == "validation":
            verify_barrier(self.run_root, "generation_frozen.json")
        if split == "audit":
            verify_barrier(self.run_root, "selections_frozen.json")
        if deployment != (split == "audit"):
            raise ValueError("deployment fallback is restricted to audit")
        return K.evaluate_panel(self, source, outer, split, deployment=deployment)

    def cached_panel(self, source: str, outer: int, split: str) -> list[dict[str, Any]]:
        """Read previously available TRAIN rows without creating new evaluations."""
        if split != "train":
            raise ValueError("generation archive accepts TRAIN only")
        return K.evaluate_panel(self, source, outer, split, read_only=True)

    def messages(
        self, outer: int, slot: int, parent: str, feedback: str | None
    ) -> list[dict[str, str]]:
        """Reproduce exactly the shared EXP-16 I/C information treatment."""
        result = [{"role": "user", "content": S._invariant(self.config)}]
        if self.arm != "I":
            result.append(
                {
                    "role": "user",
                    "content": "Improve the current optimizer.\nCURRENT SOURCE:\n"
                    + parent,
                }
            )
        return result

    def _proposal(
        self, outer: int, slot: int, parent: str, feedback: str | None
    ) -> dict[str, Any]:
        """Record one response and its allocated TRAIN evidence before another update."""
        if outer not in self.seeds or not 0 <= slot < self.slots:
            raise ValueError("unregistered proposal slot")
        messages = self.messages(outer, slot, parent, feedback)
        if (
            sum(len(message["content"]) for message in messages)
            > self.config["max_prompt_chars"]
        ):
            raise ValueError("prompt exceeds its registered size limit")
        request = {
            "slot_id": f"{self.config['namespace']}_{outer}_{self.arm}_{slot:02d}",
            "model": G.MODEL,
            "outer": outer,
            "arm": self.arm,
            "slot": slot,
            "parent_sha256": B.source_hash(parent),
            "settings": S._generation_settings(self.config, outer, slot),
            "messages": messages,
            "trace_feedback_sha256": (
                B.source_hash(feedback) if feedback is not None else None
            ),
            "freeze_sha256": B.digest(self.frozen),
        }
        directory = self.root / str(outer) / self.arm / f"slot_{slot:02d}"
        if not E.exists(directory / "response.json") and not callable(self.client):
            raise RuntimeError("unfinished generation requires an explicit client")
        start = S._start_clock(directory, "generation")
        response = I.complete_slot(directory, request, self.client)
        S._verify_response(request, response)
        if not E.exists(directory / "generation_timing.json"):
            end = S.clock_snapshot()
            I.persist(
                directory / "generation_timing.json",
                {"start": start, "end": end, **S.elapsed_clocks(start, end)},
            )
        self.training_receipt(outer, slot)
        return response

    def training_receipt(self, outer: int, index: int) -> dict[str, Any]:
        """Preserve complete host TRAIN provenance, including invalid prior attempts."""
        directory = self.root / str(outer) / self.arm
        path = directory / (
            "seed_train_receipt.json"
            if index == -1
            else f"slot_{index:02d}/train_receipt.json"
        )
        response = (
            None
            if index == -1
            else E.read(directory / f"slot_{index:02d}/response.json")
        )
        source = B.SEED_SOURCE if response is None else response["source"]
        existing = E.read(path) if E.exists(path) else None
        rows = (
            self.cached_panel(source, outer, "train")
            if existing
            else self.panel(source, outer, "train")
        )
        valid = all(row["valid"] for row in rows)
        vector = (
            [
                B.aggregate(
                    rows[start : start + self.config["local_replicates"]], "auc"
                )
                for start in range(0, len(rows), self.config["local_replicates"])
            ]
            if valid
            else None
        )
        value = {
            "arm": self.arm,
            "outer": outer,
            "slot": index,
            "split": "train",
            "source_sha256": B.source_hash(source),
            "response_sha256": None if response is None else B.digest(response),
            "observed_ns": existing["observed_ns"] if existing else time.time_ns(),
            "allocated_trajectories": len(rows),
            "observed_trajectories": len(rows),
            "valid_trajectories": sum(row["valid"] for row in rows),
            "status_counts": dict(
                sorted(Counter(row["status"] for row in rows).items())
            ),
            "aggregate_auc": B.aggregate(rows, "auc") if valid else None,
            "auc_by_instance": vector,
            "row_hashes": [B.digest(row) for row in rows],
        }
        I.persist(path, value)
        return value

    def generate(self, outer: int) -> None:
        """Retain the seed receipt, then use inherited independent/production scheduling."""
        self.training_receipt(outer, -1)
        super().generate(outer)


def owner_type(config: dict[str, Any]) -> type[Study]:
    """Import the registered mechanism extension only for EXP-18 runs."""
    if config["experiment"] == "EXP-18":
        from experiments.recursive_opt._shared.optimizer_discovery.exp18.study import MechanismStudy

        return MechanismStudy
    return Study


def verify_chronology(root: Path) -> dict[str, int]:
    """Verify every response identity, prompt and protected evaluation timestamp."""
    frozen = preflight(root)
    config = frozen["config"]
    generation = verify_barrier(root, "generation_frozen.json")
    ids = set()
    for outer in config["outer_seeds"]:
        for arm in config["arms"]:
            owner = owner_type(config)(root, arm)
            directory = root / "raw" / str(outer) / arm
            parents = {frozen["seed_sha256"]: B.SEED_SOURCE}
            for slot in range(config["slots"]):
                folder = directory / f"slot_{slot:02d}"
                request, response = E.read(folder / "request.json"), E.read(
                    folder / "response.json"
                )
                S._verify_response(request, response)
                feedback = (
                    None
                    if arm == "I"
                    else E.read(folder / "propagated_feedback.json")["text"]
                )
                parent = parents.get(request["parent_sha256"])
                if (
                    parent is None
                    or request["slot_id"]
                    != f"{config['namespace']}_{outer}_{arm}_{slot:02d}"
                    or request["outer"] != outer
                    or request["arm"] != arm
                    or request["slot"] != slot
                    or response["id"] in ids
                    or response["completed_ns"] >= generation["completed_ns"]
                    or request["settings"]
                    != S._generation_settings(config, outer, slot)
                    or request["freeze_sha256"] != B.digest(frozen)
                    or request["messages"]
                    != owner.messages(outer, slot, parent, feedback)
                    or request["trace_feedback_sha256"]
                    != (B.source_hash(feedback) if feedback is not None else None)
                ):
                    raise RuntimeError("response chronology or frozen prompt mismatch")
                receipt = E.read(folder / "train_receipt.json")
                if (
                    receipt["response_sha256"] != B.digest(response)
                    or not response["completed_ns"]
                    <= receipt["observed_ns"]
                    < generation["completed_ns"]
                ):
                    raise RuntimeError(
                        "TRAIN receipt chronology or source identity mismatch"
                    )
                ids.add(response["id"])
                parents[response["source_sha256"]] = response["source"]
    selection = (
        verify_barrier(root, "selections_frozen.json")
        if E.exists(root / "selections_frozen.json")
        else None
    )
    if selection:
        for relative in selection["hashes"]:
            if (
                not generation["completed_ns"]
                < E.read(root / relative)["selected_ns"]
                < selection["completed_ns"]
            ):
                raise RuntimeError("selection chronology mismatch")
    for path in (root / "cache").glob("*.json*"):
        if not path.name.endswith((".json", ".json.gz")):
            continue
        record = E.read(path.with_suffix("") if path.suffix == ".gz" else path)
        split, timestamp = record["key"]["split"], record["computed_clock"]["wall_ns"]
        if split == "validation" and timestamp <= generation["completed_ns"]:
            raise RuntimeError("validation occurred during generation")
        if split == "audit" and (
            selection is None or timestamp <= selection["completed_ns"]
        ):
            raise RuntimeError("audit occurred before all selections froze")
    return {"completed_responses": len(ids), "unique_response_ids": len(ids)}


def run_generation(root: Path, *, client: Any) -> None:
    """Execute the frozen order and seal all responses before validation is possible."""
    frozen = preflight(root)
    if E.exists(root / "generation_frozen.json"):
        verify_chronology(root)
        return
    start = S._start_clock(root, "generation")
    owners = {
        arm: owner_type(frozen["config"])(root, arm, client=client)
        for arm in frozen["config"]["arms"]
    }
    for outer in frozen["config"]["outer_seeds"]:
        for arm in frozen["arm_order"][str(outer)]:
            owners[arm].generate(outer)
    end = S.clock_snapshot()
    I.persist(
        root / "generation_frozen.json",
        {
            "completed_ns": end["wall_ns"],
            "hashes": {
                str(path.relative_to(root)): B.digest(E.read(path))
                for path in barrier_paths(root, "generation_frozen.json")
            },
            "clocks": {"start": start, "end": end, **S.elapsed_clocks(start, end)},
        },
    )
    verify_chronology(root)


def select_all(root: Path) -> None:
    """Apply the unchanged validation-only rule to every registered seed-inclusive pool."""
    frozen = preflight(root)
    verify_barrier(root, "generation_frozen.json")
    if E.exists(root / "selections_frozen.json"):
        verify_chronology(root)
        return
    start = S._start_clock(root, "selection")
    config, selections = frozen["config"], {}
    for outer in config["outer_seeds"]:
        for arm in config["arms"]:
            owner = owner_type(config)(root, arm)
            directory = root / "raw" / str(outer) / arm
            if not E.exists(directory / "selection.json"):
                pool = owner.pool(outer, arm)
                for candidate in pool:
                    train = owner.panel(candidate["source"], outer, "train")
                    validation = owner.panel(candidate["source"], outer, "validation")
                    eligible = all(row["valid"] for row in [*train, *validation])
                    candidate.update(
                        {
                            "train": train,
                            "validation": validation,
                            "eligible": eligible,
                            "validation_auc": (
                                B.aggregate(validation, "auc") if eligible else None
                            ),
                        }
                    )
                if not pool[0]["eligible"]:
                    raise RuntimeError("trusted seed failed candidate eligibility")
                best = min(
                    (item for item in pool if item["eligible"]),
                    key=lambda item: (item["validation_auc"], item["index"]),
                )
                I.persist(directory / "pool.json", pool)
                I.persist(
                    directory / "selection.json",
                    {
                        "selected_ns": time.time_ns(),
                        **{
                            key: best[key]
                            for key in (
                                "index",
                                "source",
                                "source_sha256",
                                "validation_auc",
                            )
                        },
                    },
                )
            selections[f"{outer}/{arm}"] = E.read(directory / "selection.json")
    representatives = {}
    for arm in config["arms"]:
        outer = min(
            config["outer_seeds"],
            key=lambda seed: (
                selections[f"{seed}/{arm}"]["validation_auc"],
                config["outer_seeds"].index(seed),
            ),
        )
        representatives[arm] = {
            "outer": outer,
            "selection": selections[f"{outer}/{arm}"],
        }
    chosen = representatives[config["representative_arm"]]
    end = S.clock_snapshot()
    hashes = {
        str(path.relative_to(root)): B.digest(E.read(path))
        for path in barrier_paths(root, "selections_frozen.json")
    }
    I.persist(
        root / "selections_frozen.json",
        {
            "completed_ns": end["wall_ns"],
            "hashes": hashes,
            "selection_hashes": hashes,
            "representative_outer": chosen["outer"],
            "representative": chosen["selection"],
            "representatives": representatives,
            "clocks": {"start": start, "end": end, **S.elapsed_clocks(start, end)},
        },
    )
    verify_chronology(root)


def run_audit(root: Path) -> dict[str, Any]:
    """Evaluate all selected deployments and fixed controls after the global barrier."""
    frozen = preflight(root)
    verify_chronology(root)
    selection = verify_barrier(root, "selections_frozen.json")
    completed = E.exists(root / "audit_results.json")
    config, per_seed = frozen["config"], {}
    start = S._start_clock(root, "audit")
    for outer in config["outer_seeds"]:
        owner = owner_type(config)(root, config["arms"][0])
        values = {}
        for arm in [*config["audit_controls"], *config["arms"]]:
            source = (
                B.SEED_SOURCE
                if arm == "A0"
                else (
                    frozen["fixed_controls"]["B2"]["source"]
                    if arm == "B2"
                    else E.read(root / "raw" / str(outer) / arm / "selection.json")[
                        "source"
                    ]
                )
            )
            rows = (
                K.evaluate_panel(
                    owner, source, outer, "audit", deployment=True, read_only=True
                )
                if completed
                else owner.panel(source, outer, "audit", deployment=True)
            )
            if not all(row["valid"] and row["metrics"] is not None for row in rows):
                raise RuntimeError("trusted deployment infrastructure failed")
            values[arm] = {
                "source_sha256": B.source_hash(source),
                "rows": rows,
                "auc": B.aggregate(rows, "auc"),
                "final_regret": B.aggregate(rows, "final_regret"),
                "fallback_trajectories": sum(row["fallback_used"] for row in rows),
            }
        per_seed[str(outer)] = values
    if E.exists(root / "audit_results.json"):
        saved = E.read(root / "audit_results.json")
        if saved["per_seed"] != per_seed:
            raise RuntimeError("recomputed audit differs from completed evidence")
        return saved
    end = S.clock_snapshot()
    result = {
        "schema": "optimizer_successor.audit.v1",
        "per_seed": per_seed,
        "selections_frozen_ns": selection["completed_ns"],
        "completed_ns": end["wall_ns"],
        "clocks": {"start": start, "end": end, **S.elapsed_clocks(start, end)},
    }
    I.persist(root / "audit_results.json", result)
    verify_chronology(root)
    return result
