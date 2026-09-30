"""Prospective P1 owners around the common evaluator and production Trace search.

This module does not register or start a scientific run on import. A caller must
prepare a separate protocol, freeze it explicitly, then supply a client. Candidate
execution retains the Phase-0 subprocess guarantee, not OS filesystem confinement.
"""

from __future__ import annotations

import ast
import json
import math
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.evidence import environment
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import feedback_experiment as F
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import generation as G
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import trace_schedule as T
from experiments.recursive_opt._shared.optimizer_discovery.investigation16.feedback import rich_feedback as R

ARMS = ("I", "C", "R", "W")
SPLITS = ("train", "validation", "audit")
_CACHE_LOCKS: dict[str, threading.Lock] = {}
_LOCK_GUARD = threading.Lock()


def configuration(
    *,
    max_tokens: int,
    namespace: str = "P1",
    outer_seeds: list[int] | None = None,
    slots: int = 8,
    budget: int = 32,
    task_replicates: dict[str, int] | None = None,
    local_replicates: int = 2,
    workers: int = 4,
    include_aggregate_auc: bool = False,
    max_prompt_chars: int = 262144,
) -> dict[str, Any]:
    """Build a configurable draft; the caller must independently register its choices."""
    result = {
        "namespace": namespace,
        "outer_seeds": (
            outer_seeds
            if outer_seeds is not None
            else [16411, 16423, 16437, 16441, 16453, 16467]
        ),
        "slots": slots,
        "budget": budget,
        "task_replicates": (
            task_replicates if task_replicates is not None else dict.fromkeys(SPLITS, 2)
        ),
        "local_replicates": local_replicates,
        "workers": workers,
        "include_aggregate_auc": include_aggregate_auc,
        "max_prompt_chars": max_prompt_chars,
        "max_tokens": max_tokens,
        "model": G.MODEL,
        "arms": list(ARMS),
        "schedules": {"C": [1, 1], "R": [1, 1], "W": [2, 1]},
        "generation_concurrency": 1,
        "timeout_s": 2,
        "feedback": "current_parent_actual_trace_only",
        "selection": "eligible_min_validation_auc_then_seed_minus_one_or_slot",
        "representative": "minimum_R_validation_auc_then_outer_seed_order",
        "cache_version": "p1-evaluation-v1",
        "fallback": "permanent_common_seed_actual_history_no_budget_reset",
        "bootstrap": B.MANIFEST["bootstrap"],
    }
    _validate_config(result)
    return result


def _validate_config(config: dict[str, Any]) -> None:
    """Reject unsupported settings before any request, evaluation or cache access."""
    if (
        not isinstance(config.get("namespace"), str)
        or not config["namespace"]
        or config.get("model") != G.MODEL
        or config.get("arms") != list(ARMS)
        or config.get("generation_concurrency") != 1
        or config.get("max_tokens") not in (8000, 32000)
        or type(config.get("include_aggregate_auc")) is not bool
    ):
        raise ValueError(
            "unsupported prospective search identity or generation settings"
        )
    for name in ("slots", "budget", "local_replicates", "workers", "max_prompt_chars"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    seeds = config.get("outer_seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
        or config["slots"] % 2
        or config["workers"] > 8
        or config.get("timeout_s") != 2
        or config.get("schedules") != {"C": [1, 1], "R": [1, 1], "W": [2, 1]}
    ):
        raise ValueError("invalid seed, schedule or evaluation resource configuration")
    counts = config.get("task_replicates")
    if (
        not isinstance(counts, dict)
        or set(counts) != set(SPLITS)
        or any(type(count) is not int or count < 1 for count in counts.values())
    ):
        raise ValueError(
            "every split needs a positive number of six-stratum replicates"
        )
    expected = {
        "feedback": "current_parent_actual_trace_only",
        "selection": "eligible_min_validation_auc_then_seed_minus_one_or_slot",
        "representative": "minimum_R_validation_auc_then_outer_seed_order",
        "cache_version": "p1-evaluation-v1",
        "fallback": "permanent_common_seed_actual_history_no_budget_reset",
        "bootstrap": B.MANIFEST["bootstrap"],
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("unimplemented scientific configuration")


def clock_snapshot() -> dict[str, int | None]:
    """Capture wall and process-independent clocks so host suspension can be diagnosed."""
    return {
        "wall_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "boottime_ns": (
            time.clock_gettime_ns(time.CLOCK_BOOTTIME)
            if hasattr(time, "CLOCK_BOOTTIME")
            else None
        ),
    }


def elapsed_clocks(
    start: dict[str, int | None], end: dict[str, int | None]
) -> dict[str, Any]:
    """Report suspension-sensitive elapsed clocks without attributing pauses to the model."""
    deltas = {
        key: (
            None
            if start[key] is None or end[key] is None
            else (end[key] - start[key]) / 1e9
        )
        for key in start
    }
    boot, monotonic = deltas["boottime_ns"], deltas["monotonic_ns"]
    return {
        "elapsed_s": deltas,
        "suspend_s_estimate": (
            None
            if boot is None or monotonic is None or min(boot, monotonic) < 0
            else max(0.0, boot - monotonic)
        ),
        "clock_reset_detected": any(
            value is not None and value < 0 for value in deltas.values()
        ),
    }


def _start_clock(root: Path, name: str) -> dict[str, int | None]:
    """Preserve the first phase start clock across interruptions and process resume."""
    path = root / (name + "_started.json")
    if not E.exists(path):
        I.persist(path, clock_snapshot())
    return E.read(path)


def _arm_order(config: dict[str, Any]) -> dict[str, list[str]]:
    """Reconstruct the fixed Latin-style arm rotation without consulting results."""
    return {
        str(seed): list(ARMS[index % 4 :] + ARMS[: index % 4])
        for index, seed in enumerate(config["outer_seeds"])
    }


def prepare(
    root: Path,
    config: dict[str, Any],
    protocol: Path,
    *,
    extra_frozen_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Freeze an explicitly supplied protocol/configuration without evaluating audit tasks."""
    _validate_config(config)
    if E.exists(root / "freeze.json"):
        saved = preflight(root)
        if (
            saved["config"] != config
            or saved["protocol_sha256"] != B.source_hash(protocol.read_text())
            or saved["extra_frozen_paths"]
            != [str(path.resolve()) for path in extra_frozen_paths or []]
        ):
            raise RuntimeError("existing freeze differs from requested protocol")
        return saved
    tasks = {
        split: G.fresh_tasks(
            config["namespace"], split, config["task_replicates"][split]
        )
        for split in SPLITS
    }
    identities = [B.task_identity(task) for panel in tasks.values() for task in panel]
    if len(identities) != len(set(identities)):
        raise RuntimeError("prospective split instances must be disjoint")
    files = [Path(module.__file__) for module in (B, E, G, I, F, T, R)] + [
        Path(__file__),
        protocol,
        *(extra_frozen_paths or []),
    ]
    files += [
        Path("opto/features/recursive_opt/optimizer_program.py"),
        Path("opto/trainer/algorithms/priority_search.py"),
        B.ROOT / "exp15_manifest.json",
    ]
    frozen = {
        "schema": "investigation16.production_search.v1",
        "created_ns": time.time_ns(),
        "config": config,
        "tasks": tasks,
        "seed_source": B.SEED_SOURCE,
        "seed_sha256": B.source_hash(B.SEED_SOURCE),
        "protocol_sha256": B.source_hash(protocol.read_text()),
        "files": {
            str(path.resolve()): B.source_hash(path.read_text()) for path in files
        },
        "extra_frozen_paths": [
            str(path.resolve()) for path in extra_frozen_paths or []
        ],
        "environment": environment(),
        "arm_order": _arm_order(config),
        "panels": {
            split: {
                "tasks": len(tasks[split]),
                "trajectories": len(tasks[split]) * config["local_replicates"],
            }
            for split in SPLITS
        },
        "benchmark_manifest": B.MANIFEST,
        "local_seed_derivation": "G.local_seed(namespace, int(sha256([outer, replicate])[:15],16), task)",
    }
    I.persist(root / "freeze.json", frozen)
    I.persist(root / "freeze_sha256.json", {"sha256": B.digest(frozen)})
    return frozen


def preflight(root: Path) -> dict[str, Any]:
    """Compare exact configuration, source, environment and task identities before resume."""
    frozen = E.read(root / "freeze.json")
    if B.digest(frozen) != E.read(root / "freeze_sha256.json")["sha256"]:
        raise RuntimeError("prospective freeze digest mismatch")
    _validate_config(frozen["config"])
    for path, expected in frozen["files"].items():
        if B.source_hash(Path(path).read_text()) != expected:
            raise RuntimeError("prospective freeze source mismatch: " + path)
    if frozen["environment"] != environment() or frozen["seed_sha256"] != B.source_hash(
        B.SEED_SOURCE
    ):
        raise RuntimeError("prospective freeze environment or seed mismatch")
    config = frozen["config"]
    reconstructed = {
        split: G.fresh_tasks(
            config["namespace"], split, config["task_replicates"][split]
        )
        for split in SPLITS
    }
    panels = {
        split: {
            "tasks": len(reconstructed[split]),
            "trajectories": len(reconstructed[split]) * config["local_replicates"],
        }
        for split in SPLITS
    }
    if (
        frozen["tasks"] != reconstructed
        or frozen["arm_order"] != _arm_order(config)
        or frozen["panels"] != panels
        or frozen["benchmark_manifest"] != B.MANIFEST
    ):
        raise RuntimeError("prospective freeze reconstruction mismatch")
    return frozen


def _verify_barrier(root: Path, name: str) -> dict[str, Any]:
    """Verify every immutable child record referenced by a completed phase barrier."""
    path = root / name
    if not E.exists(path):
        raise RuntimeError(
            "all generation must finish first"
            if name.startswith("generation")
            else "all selections must freeze first"
        )
    value = E.read(path)
    config = E.read(root / "freeze.json")["config"]
    expected = set()
    for outer in config["outer_seeds"]:
        for arm in ARMS:
            prefix = f"raw/{outer}/{arm}/"
            if name == "selections_frozen.json":
                expected.add(prefix + "selection.json")
            else:
                expected.update(
                    prefix + suffix
                    for suffix in ("generation_complete.json", "allocations_train.json")
                )
                if arm != "I":
                    expected.update(
                        prefix + suffix for suffix in ("trace.json", "schedule.json")
                    )
                expected.update(
                    prefix + f"slot_{slot:02d}/{kind}.json"
                    for slot in range(config["slots"])
                    for kind in ("request", "response")
                )
    if set(value["hashes"]) != expected:
        raise RuntimeError("phase barrier omits registered evidence")
    for relative, expected in value["hashes"].items():
        if B.digest(E.read(root / relative)) != expected:
            raise RuntimeError("phase barrier evidence changed")
    return value


def _verify_response(request: dict[str, Any], result: dict[str, Any]) -> None:
    """Reparse immutable provider content and verify response identity and typed source status."""
    try:
        parsed = E.parse_program(result["content"])
        parse_status = "parsed"
    except ValueError:
        parsed, parse_status = "", "unparsable"
    if (
        result.get("completed") is not True
        or not isinstance(result.get("id"), str)
        or not result["id"]
        or result.get("model") not in {G.MODEL, "openrouter/" + G.MODEL}
        or request.get("model") != G.MODEL
        or type(result.get("completed_ns")) is not int
        or type(result.get("attempt")) is not int
        or result["attempt"] < 1
        or parsed != result["source"]
        or parse_status != result["parse_status"]
        or B.source_status(parsed) != result["source_status"]
        or B.source_hash(parsed) != result["source_sha256"]
    ):
        raise RuntimeError(
            "completed response identity, parsing or source integrity failure"
        )


def _generation_settings(
    config: dict[str, Any], outer: int, slot: int
) -> dict[str, Any]:
    """Share and reconstruct exact model settings across independent and recursive arms."""
    return {
        "temperature": 0.6,
        "top_p": 1.0,
        "max_tokens": config["max_tokens"],
        "extra_body": {"reasoning": {"effort": "low"}},
        "timeout": 300,
        "seed": int(B.digest([config["namespace"], "generation", outer, slot])[:8], 16)
        & 0x7FFFFFFF,
        "num_retries": 0,
    }


def _invariant(config: dict[str, Any]) -> str:
    """Build the identical objective, seed and execution instruction for every arm."""
    return (
        E.INVARIANT.replace(
            "budget is 32 objective", f"budget is {config['budget']} objective"
        )
        + B.SEED_SOURCE
        + "\n"
        + F.VALIDITY_CLARIFICATION
        + "\nSELECTION OBJECTIVE:\n"
        + R.ANYTIME_OBJECTIVE
    )


def verify_arm_responses(
    root: Path, outer: int, arm: str, *, completed_before_ns: int | None = None
) -> dict[str, Any]:
    """Audit one complete arm, including R-only engineering runs without a global barrier."""
    frozen = preflight(root)
    if arm not in ARMS or outer not in frozen["config"]["outer_seeds"]:
        raise ValueError("unregistered response audit arm or seed")
    response_ids: set[str] = set()
    count = 0
    directory = root / "raw" / str(outer) / arm
    known_parents = {B.source_hash(B.SEED_SOURCE): B.SEED_SOURCE}
    for slot in range(frozen["config"]["slots"]):
        request = E.read(directory / f"slot_{slot:02d}/request.json")
        response = E.read(directory / f"slot_{slot:02d}/response.json")
        _verify_response(request, response)
        config = frozen["config"]
        if (
            request["slot_id"] != f"{config['namespace']}_{outer}_{arm}_{slot:02d}"
            or request["outer"] != outer
            or request["arm"] != arm
            or request["slot"] != slot
            or request["freeze_sha256"] != B.digest(frozen)
            or request["settings"] != _generation_settings(config, outer, slot)
            or request["messages"][0] != {"role": "user", "content": _invariant(config)}
            or len(request["messages"]) != (1 if arm == "I" else 2)
            or request["parent_sha256"] not in known_parents
        ):
            raise RuntimeError(
                "request differs from the frozen configuration or lineage"
            )
        if arm != "I":
            feedback = E.read(directory / f"slot_{slot:02d}/propagated_feedback.json")[
                "text"
            ]
            expected = (
                "Improve the current optimizer.\nCURRENT SOURCE:\n"
                + known_parents[request["parent_sha256"]]
            )
            if arm in {"R", "W"}:
                expected += "\nTRAINING FEEDBACK:\n" + feedback
            if request["trace_feedback_sha256"] != B.source_hash(feedback) or request[
                "messages"
            ][1] != {
                "role": "user",
                "content": expected,
            }:
                raise RuntimeError("request differs from propagated training feedback")
        if response["id"] in response_ids or (
            completed_before_ns is not None
            and response["completed_ns"] >= completed_before_ns
        ):
            raise RuntimeError("response identity or generation chronology violation")
        response_ids.add(response["id"])
        known_parents[response["source_sha256"]] = response["source"]
        count += 1
    return {"completed_responses": count, "response_ids": sorted(response_ids)}


def verify_chronology(root: Path) -> dict[str, int]:
    """Prove all registered responses preceded generation/selection barriers and audit calls."""
    frozen = preflight(root)
    generation = _verify_barrier(root, "generation_frozen.json")
    response_ids: set[str] = set()
    count = 0
    for outer in frozen["config"]["outer_seeds"]:
        for arm in ARMS:
            audited = verify_arm_responses(
                root, outer, arm, completed_before_ns=generation["completed_ns"]
            )
            if response_ids.intersection(audited["response_ids"]):
                raise RuntimeError("duplicate provider response across arms or seeds")
            response_ids.update(audited["response_ids"])
            count += audited["completed_responses"]
    if E.exists(root / "selections_frozen.json"):
        selections = _verify_barrier(root, "selections_frozen.json")
        for relative in selections["hashes"]:
            selected = E.read(root / relative)
            if (
                not generation["completed_ns"]
                < selected["selected_ns"]
                < selections["completed_ns"]
            ):
                raise RuntimeError("selection chronology violation")
        for path in (root / "cache").glob("*.json*"):
            if path.name.endswith(".pending"):
                continue
            cached = E.read(path.with_suffix("") if path.suffix == ".gz" else path)
            if (
                cached["key"]["split"] == "audit"
                and cached["computed_clock"]["wall_ns"] <= selections["completed_ns"]
            ):
                raise RuntimeError(
                    "audit trajectory preceded the global selection freeze"
                )
    return {"completed_responses": count, "unique_response_ids": len(response_ids)}


class SearchExperiment(E.Experiment):
    """One arm's prompts and evaluator ownership; search updates remain in production Trace."""

    persist = staticmethod(I.persist)

    def __init__(self, root: Path, arm: str, *, client: Any = None) -> None:
        """Bind a frozen prospective run without invoking EXP-15's constructor or proposals."""
        if arm not in ARMS:
            raise ValueError("unknown prospective arm")
        self.frozen = preflight(root)
        self.config = self.frozen["config"]
        self.run_root = root
        self.root = root / "raw"
        self.arm = arm
        self.phase = self.config["namespace"] + ":" + arm
        self.seeds = self.config["outer_seeds"]
        self.slots = self.config["slots"]
        self.budget = self.config["budget"]
        self.client = client

    def event(self, value: dict[str, Any]) -> None:
        """Record exact isolated events; parallel offline workers cannot interleave JSON."""
        I.persist(
            self.run_root / "events" / (uuid.uuid4().hex + ".json"),
            {"arm": self.arm, "time_ns": time.time_ns(), **value},
        )

    def _panel_inputs(self, outer: int, split: str) -> list[tuple[dict[str, Any], int]]:
        """Derive the same ordered task/local-seed panel for every candidate and arm."""
        if outer not in self.seeds or split not in SPLITS:
            raise ValueError("unregistered outer seed or split")
        return [
            (
                task,
                G.local_seed(
                    self.config["namespace"],
                    int(B.digest([outer, replicate])[:15], 16),
                    task,
                ),
            )
            for task in self.frozen["tasks"][split]
            for replicate in range(self.config["local_replicates"])
        ]

    def panel(
        self, source: str, outer: int, split: str, *, deployment: bool = False
    ) -> list[dict[str, Any]]:
        """Apply the common evaluator with an exact, complete, split-separated shared cache."""
        if split == "validation":
            _verify_barrier(self.run_root, "generation_frozen.json")
        if split == "audit":
            _verify_barrier(self.run_root, "selections_frozen.json")
        if deployment != (split == "audit"):
            raise ValueError("deployment fallback is restricted to the final audit")

        def evaluate_one(item: tuple[dict[str, Any], int]) -> dict[str, Any]:
            """Evaluate each unique frozen cache key once and validate identity on every reuse."""
            task, local = item
            key = {
                "namespace": self.config["namespace"],
                "source_sha256": B.source_hash(source),
                "task_identity": B.task_identity(task),
                "split": split,
                "outer": outer,
                "local_seed": local,
                "budget": self.budget,
                "deployment": deployment,
                "timeout_s": self.config["timeout_s"],
                "seed_sha256": self.frozen["seed_sha256"],
                "evaluator_version": self.config["cache_version"],
                "evaluator_sha256": self.frozen["files"][
                    str(Path(B.__file__).resolve())
                ],
            }
            digest = B.digest(key)
            path = self.run_root / "cache" / (digest + ".json")
            with _LOCK_GUARD:
                lock = _CACHE_LOCKS.setdefault(str(path.resolve()), threading.Lock())
            with lock:
                hit = E.exists(path)
                if not hit:
                    row = B.evaluate(
                        source,
                        task,
                        local,
                        budget=self.budget,
                        deployment=deployment,
                        seed_source=B.SEED_SOURCE,
                        timeout_s=self.config["timeout_s"],
                    )
                    I.persist(
                        path,
                        {
                            "key": key,
                            "row": row,
                            "row_sha256": B.digest(row),
                            "computed_clock": clock_snapshot(),
                        },
                    )
                cached = E.read(path)
                row = cached["row"]
                if (
                    cached["key"] != key
                    or cached["row_sha256"] != B.digest(row)
                    or any(
                        row[field] != key[field]
                        for field in (
                            "source_sha256",
                            "task_identity",
                            "local_seed",
                            "budget",
                        )
                    )
                ):
                    raise RuntimeError("cached trajectory integrity failure")
            self.event(
                {
                    "event": "evaluation_cache",
                    "outer": outer,
                    "split": split,
                    "key": digest,
                    "hit": hit,
                }
            )
            return row

        with ThreadPoolExecutor(max_workers=self.config["workers"]) as pool:
            return list(pool.map(evaluate_one, self._panel_inputs(outer, split)))

    def feedback(self, source: str, outer: int) -> dict[str, Any]:
        """Expose only current-parent raw training progress and an optional aggregate scalar."""
        rows = self.panel(source, outer, "train")
        bounds = [
            [[-5.0, 5.0]] * task["dimension"]
            for task, _ in self._panel_inputs(outer, "train")
        ]
        payload = R.build_feedback(source, rows, bounds)
        if self.config["include_aggregate_auc"]:
            payload["aggregate_training_auc"] = (
                B.aggregate(rows, "auc") if all(row["valid"] for row in rows) else None
            )
        R.serialize_feedback(payload, max_chars=self.config["max_prompt_chars"])
        return payload

    def proposal_from_trace(
        self, outer: int, slot: int, parent: str, feedback: str
    ) -> dict[str, Any]:
        """Consume actual propagated Trace user feedback, never a reconstructed prompt panel."""
        if self.arm == "I":
            raise ValueError("independent generation has no Trace callback")
        I.persist(
            self.root
            / str(outer)
            / self.arm
            / f"slot_{slot:02d}/propagated_feedback.json",
            {"text": feedback},
        )
        if not feedback.startswith("ID [0]: "):
            raise RuntimeError("unexpected production feedback envelope")
        text = feedback[len("ID [0]: ") :].strip()
        if text.startswith("["):
            native_invalid = ast.literal_eval(text)
            if (
                not isinstance(native_invalid, list)
                or len(native_invalid) != 1
                or not isinstance(native_invalid[0], str)
            ):
                raise RuntimeError("unexpected native invalid-feedback envelope")
            text = native_invalid[0]
        payload = json.loads(text)
        allowed = {"schema", "current"} | (
            {"aggregate_training_auc"}
            if self.config["include_aggregate_auc"]
            else set()
        )
        if set(payload) != allowed or payload["current"][
            "source_sha256"
        ] != B.source_hash(parent):
            raise RuntimeError("propagated feedback does not match the current parent")
        # This equality is an integrity check only. The prompt includes the exact
        # propagated string below, rather than serializing a replacement payload.
        if payload != self.feedback(parent, outer):
            raise RuntimeError(
                "propagated feedback differs from the permitted training projection"
            )
        return self._proposal(outer, slot, parent, feedback)

    def _proposal(
        self, outer: int, slot: int, parent: str, feedback: str | None
    ) -> dict[str, Any]:
        """Build one symmetric invariant request and retain every completed response slot."""
        if outer not in self.seeds or not 0 <= slot < self.slots:
            raise ValueError("unregistered proposal slot")
        messages = [{"role": "user", "content": _invariant(self.config)}]
        if self.arm != "I":
            content = "Improve the current optimizer.\nCURRENT SOURCE:\n" + parent
            if self.arm in {"R", "W"}:
                if feedback is None:
                    raise RuntimeError(
                        "rich generation requires actual production Trace feedback"
                    )
                content += "\nTRAINING FEEDBACK:\n" + feedback
            messages.append({"role": "user", "content": content})
        if (
            sum(len(message["content"]) for message in messages)
            > self.config["max_prompt_chars"]
        ):
            raise ValueError("prompt exceeds the registered size limit")
        settings = _generation_settings(self.config, outer, slot)
        request = {
            "slot_id": f"{self.config['namespace']}_{outer}_{self.arm}_{slot:02d}",
            "model": G.MODEL,
            "outer": outer,
            "arm": self.arm,
            "slot": slot,
            "parent_sha256": B.source_hash(parent),
            "settings": settings,
            "messages": messages,
            "trace_feedback_sha256": (
                B.source_hash(feedback) if feedback is not None else None
            ),
            "freeze_sha256": B.digest(self.frozen),
        }
        directory = self.root / str(outer) / self.arm / f"slot_{slot:02d}"
        if not E.exists(directory / "response.json") and not callable(self.client):
            raise RuntimeError("unfinished generation requires an explicit client")
        start = _start_clock(directory, "generation")
        result = I.complete_slot(directory, request, self.client)
        _verify_response(request, result)
        if not E.exists(directory / "generation_timing.json"):
            end = clock_snapshot()
            I.persist(
                directory / "generation_timing.json",
                {"start": start, "end": end, **elapsed_clocks(start, end)},
            )
        return result

    def generate(self, outer: int) -> None:
        """Generate independently or delegate the entire search schedule to production Trace."""
        directory = self.root / str(outer) / self.arm
        if self.arm == "I":
            for slot in range(self.slots):
                self._proposal(outer, slot, B.SEED_SOURCE, None)
            if not E.exists(directory / "generation_complete.json"):
                I.persist(
                    directory / "generation_complete.json",
                    {"slots": self.slots, "completed_ns": time.time_ns()},
                )
        else:
            parents, proposals = self.config["schedules"][self.arm]
            T.generate_recursive(
                self, outer, parents_per_round=parents, proposals_per_parent=proposals
            )
        candidates = self.pool(outer, self.arm)
        allocations = []
        for candidate in candidates:
            if B.source_hash(candidate["source"]) != candidate["source_sha256"]:
                raise RuntimeError("candidate pool source integrity failure")
            rows = self.panel(candidate["source"], outer, "train")
            allocations.append(
                {
                    "index": candidate["index"],
                    "source_sha256": candidate["source_sha256"],
                    "trajectories": len(rows),
                    "allocated_objective_calls": len(rows) * self.budget,
                    "valid_trajectories": sum(row["valid"] for row in rows),
                }
            )
        I.persist(
            directory / "allocations_train.json",
            {"candidate_slots": len(candidates), "rows": allocations},
        )


def run_generation(root: Path, *, client: Any) -> None:
    """Execute the preregistered arm rotation, then freeze every response and train allocation."""
    frozen = preflight(root)
    if E.exists(root / "generation_frozen.json"):
        _verify_barrier(root, "generation_frozen.json")
        verify_chronology(root)
        return
    started = _start_clock(root, "generation")
    owners = {arm: SearchExperiment(root, arm, client=client) for arm in ARMS}
    hashes = {}
    for outer in frozen["config"]["outer_seeds"]:
        for arm in frozen["arm_order"][str(outer)]:
            owners[arm].generate(outer)
            directory = root / "raw" / str(outer) / arm
            paths = [
                directory / "generation_complete.json",
                directory / "allocations_train.json",
            ]
            if arm != "I":
                paths += [directory / "trace.json", directory / "schedule.json"]
            paths += [
                directory / f"slot_{slot:02d}/{name}.json"
                for slot in range(frozen["config"]["slots"])
                for name in ("request", "response")
            ]
            for path in paths:
                hashes[str(path.relative_to(root))] = B.digest(E.read(path))
    ended = clock_snapshot()
    I.persist(
        root / "generation_frozen.json",
        {
            "completed_ns": ended["wall_ns"],
            "hashes": hashes,
            "clocks": {
                "start": started,
                "end": ended,
                **elapsed_clocks(started, ended),
            },
        },
    )
    verify_chronology(root)


def select_all(root: Path) -> None:
    """Apply a common validation-only seed-inclusive rule and freeze every choice before audit."""
    frozen = preflight(root)
    _verify_barrier(root, "generation_frozen.json")
    if E.exists(root / "selections_frozen.json"):
        _verify_barrier(root, "selections_frozen.json")
        verify_chronology(root)
        return
    started = _start_clock(root, "selection")
    hashes, selections = {}, {}
    for outer in frozen["config"]["outer_seeds"]:
        for arm in ARMS:
            owner = SearchExperiment(root, arm)
            directory = root / "raw" / str(outer) / arm
            if not E.exists(directory / "selection.json"):
                candidates = owner.pool(outer, arm)
                for candidate in candidates:
                    train = owner.panel(candidate["source"], outer, "train")
                    validation = owner.panel(candidate["source"], outer, "validation")
                    eligible = all(row["valid"] for row in [*train, *validation])
                    candidate.update(
                        {
                            "eligible": eligible,
                            "train": train,
                            "validation": validation,
                            "validation_auc": (
                                B.aggregate(validation, "auc") if eligible else None
                            ),
                        }
                    )
                if not candidates[0]["eligible"]:
                    raise RuntimeError("trusted seed failed candidate eligibility")
                best = min(
                    (candidate for candidate in candidates if candidate["eligible"]),
                    key=lambda candidate: (
                        candidate["validation_auc"],
                        candidate["index"],
                    ),
                )
                I.persist(directory / "pool.json", candidates)
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
            selected = E.read(directory / "selection.json")
            if B.source_hash(selected["source"]) != selected["source_sha256"]:
                raise RuntimeError("selected source hash mismatch")
            relative = str((directory / "selection.json").relative_to(root))
            hashes[relative] = B.digest(selected)
            selections[f"{outer}/{arm}"] = selected
    representative = min(
        frozen["config"]["outer_seeds"],
        key=lambda outer: (
            selections[f"{outer}/R"]["validation_auc"],
            frozen["config"]["outer_seeds"].index(outer),
        ),
    )
    ended = clock_snapshot()
    I.persist(
        root / "selections_frozen.json",
        {
            "completed_ns": ended["wall_ns"],
            "hashes": hashes,
            "selection_hashes": hashes,
            "representative_outer": representative,
            "representative": selections[f"{representative}/R"],
            "clocks": {
                "start": started,
                "end": ended,
                **elapsed_clocks(started, ended),
            },
        },
    )
    verify_chronology(root)


def run_audit(root: Path) -> dict[str, Any]:
    """Evaluate every selected deployment policy after the global selection barrier."""
    frozen = preflight(root)
    _verify_barrier(root, "generation_frozen.json")
    barrier = _verify_barrier(root, "selections_frozen.json")
    verify_chronology(root)
    started = _start_clock(root, "audit")
    per_seed = {}
    for outer in frozen["config"]["outer_seeds"]:
        owner = SearchExperiment(root, "I")
        values = {}
        for arm in ("A0", *ARMS):
            source = (
                B.SEED_SOURCE
                if arm == "A0"
                else E.read(root / "raw" / str(outer) / arm / "selection.json")[
                    "source"
                ]
            )
            rows = owner.panel(source, outer, "audit", deployment=True)
            if not all(row["valid"] and row["metrics"] is not None for row in rows):
                raise RuntimeError(
                    "deployment or trusted fallback infrastructure failed"
                )
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
            raise RuntimeError("recomputed audit differs from preserved results")
        return saved
    ended = clock_snapshot()
    result = {
        "schema": "investigation16.production_audit.v1",
        "selections_frozen_ns": barrier["completed_ns"],
        "completed_ns": ended["wall_ns"],
        "per_seed": per_seed,
        "clocks": {"start": started, "end": ended, **elapsed_clocks(started, ended)},
    }
    if not all(
        math.isfinite(value["auc"])
        for values in per_seed.values()
        for value in values.values()
    ):
        raise RuntimeError("audit aggregation contains nonfinite outcomes")
    I.persist(root / "audit_results.json", result)
    verify_chronology(root)
    return result
