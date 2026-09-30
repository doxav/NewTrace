"""Inspect sealed P1 evidence with standard-library reads; never execute candidates.

No project evaluator, task generator, LLM client or subprocess runner is imported.
All objective values used here were already saved before the pipeline barrier.
"""

from __future__ import annotations

import ast
import difflib
import gzip
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1] / "production_run"
OUT = Path(__file__).resolve().parent


def digest(value: Any) -> str:
    """Match the frozen canonical JSON digest without importing the evaluator."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def source_hash(source: str) -> str:
    """Hash the exact UTF-8 source, without formatting or execution."""
    return hashlib.sha256(source.encode()).hexdigest()


def lineage(rows: list[dict[str, Any]], seed_hash: str) -> list[dict[str, Any]]:
    """Retain all earlier origins when identical sources make object ancestry ambiguous."""
    origins: dict[str, list[tuple[int, int, int]]] = {seed_hash: [(-1, 0, 0)]}
    result = []
    for index, row in enumerate(rows):
        if row["index"] != index or row["parent_sha256"] not in origins:
            raise ValueError("lineage parent must belong to an earlier source")
        parents = origins[row["parent_sha256"]]
        low = 1 + min(parent[1] for parent in parents)
        high = 1 + max(parent[2] for parent in parents)
        result.append(
            {
                **row,
                "parent_origin_slots": [parent[0] for parent in parents],
                "depth_range": [low, high],
            }
        )
        origins.setdefault(row["source_sha256"], []).append((index, low, high))
    return result


def feedback_payload(text: str) -> dict[str, Any]:
    """Decode the actual native Trace envelope using only literal/JSON parsing."""
    if not text.startswith("ID [0]: "):
        raise ValueError("unexpected feedback envelope")
    text = text[len("ID [0]: ") :].strip()
    if text.startswith("["):
        items = ast.literal_eval(text)
        if (
            not isinstance(items, list)
            or len(items) != 1
            or not isinstance(items[0], str)
        ):
            raise ValueError("unexpected invalid feedback envelope")
        text = items[0]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError("feedback must be a JSON object")
    return value


def verify_projection(task: dict[str, Any], row: dict[str, Any]) -> tuple[int, int]:
    """Reconstruct initial/incumbent/improvement events and the full raw anytime curve."""
    allowed = {
        "valid",
        "status",
        "budget",
        "bounds",
        "observed_evaluations",
        "initial",
        "incumbent",
        "improvements",
        "best_so_far_curve",
        "raw_anytime_mean",
        "diagnostics",
    }
    if set(task) - allowed:
        raise ValueError("undeclared fields in training feedback")
    observations = row["observations"]
    events = []
    curve = []
    for index, observation in enumerate(observations, 1):
        if not events or observation["value"] < events[-1]["value"]:
            events.append({"evaluation": index, **observation})
        curve.append({"evaluation": index, "value": events[-1]["value"]})
    expected = {
        "initial": events[0] if events else None,
        "incumbent": events[-1] if events else None,
        "improvements": events[1:],
        "best_so_far_curve": curve,
        "observed_evaluations": len(observations),
        "valid": row["valid"],
        "status": row["status"],
    }
    for key, value in expected.items():
        if task[key] != value:
            raise ValueError(f"feedback {key} differs from saved observations")
    return len(curve), len(events)


def behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize realized geometry; do not infer unobserved candidate branch choices."""
    initials: dict[int, set[tuple[float, ...]]] = defaultdict(set)
    boundary = repeated = midpoint = total = 0
    improvement_counts = []
    for row in rows:
        observations = row["observations"]
        points = [tuple(observation["x"]) for observation in observations]
        if points:
            initials[len(points[0])].add(points[0])
            midpoint += all(value == 0 for value in points[0])
        boundary += sum(
            any(value in (-5.0, 5.0) for value in point) for point in points
        )
        repeated += len(points) - len(set(points))
        total += len(points)
        best = float("inf")
        improvements = 0
        for index, observation in enumerate(observations):
            if observation["value"] < best:
                improvements += index > 0
                best = observation["value"]
        improvement_counts.append(improvements)
    return {
        "trajectories": len(rows),
        "observations": total,
        "boundary_points": boundary,
        "repeated_points": repeated,
        "midpoint_initials": midpoint,
        "distinct_initial_points_by_dimension": {
            str(dim): len(points) for dim, points in initials.items()
        },
        "mean_strict_improvements_after_first": statistics.mean(improvement_counts),
        "candidate_invalid": sum(not row["candidate_valid"] for row in rows),
        "fallback_trajectories": sum(row["fallback_used"] for row in rows),
    }


def _auc(rows: list[dict[str, Any]]) -> float | None:
    """Aggregate saved metrics by stratum; invalidity stays absent."""
    if any(not row["valid"] for row in rows):
        return None
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[row["stratum"]].append(row["metrics"]["auc"])
    return statistics.mean(statistics.mean(values) for values in groups.values())


def inspect(root: Path = ROOT) -> dict[str, Any]:
    """Audit completed source/prompt evidence and emit derived inspection metadata."""
    checked: dict[str, str] = {}

    def read(relative: str) -> Any:
        """Read a logical JSON record and remember exact physical input bytes."""
        path = root / relative
        if not path.exists():
            path = path.with_suffix(path.suffix + ".gz")
        raw = path.read_bytes()
        checked[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)

    completion = read("pipeline_complete.json")
    frozen = read("freeze.json")
    frozen_hash = read("freeze_sha256.json")["sha256"]
    if (
        digest(frozen) != frozen_hash
        or completion["primary_freeze_sha256"] != frozen_hash
    ):
        raise ValueError("P1 completion/freeze integrity mismatch")
    selection_barrier = read("selections_frozen.json")
    audit_start = read("audit_started.json")
    audit = read("audit_results.json")
    if (
        not selection_barrier["completed_ns"]
        < audit_start["wall_ns"]
        < audit["completed_ns"]
    ):
        raise ValueError("selection/audit chronology violation")
    seed_source = frozen["seed_source"]
    seed_hash = source_hash(seed_source)
    assert seed_hash == frozen["seed_sha256"]
    summary: dict[str, Any] = {
        "schema": "investigation16.program_inspection.v1",
        "freeze_sha256": frozen_hash,
        "representative_outer": selection_barrier["representative_outer"],
        "selected_R": {},
        "searches": {},
        "feedback": [],
        "executed_candidates": 0,
        "objective_calls": 0,
        "llm_calls": 0,
    }
    invariant = None
    selected = {}
    for outer in frozen["config"]["outer_seeds"]:
        for arm in frozen["config"]["arms"]:
            directory = f"raw/{outer}/{arm}"
            pool = read(directory + "/pool.json")
            by_hash = {item["source_sha256"]: item for item in pool}
            if len(pool) != 9 or [item["index"] for item in pool] != list(range(-1, 8)):
                raise ValueError("candidate pool differs from registered slots")
            for item in pool:
                if source_hash(item["source"]) != item["source_sha256"]:
                    raise ValueError("pool source integrity mismatch")
            chosen_path = directory + "/selection.json"
            chosen = read(chosen_path)
            if digest(chosen) != selection_barrier["hashes"][chosen_path]:
                raise ValueError("selected record differs from barrier")
            expected = min(
                (item for item in pool if item["eligible"]),
                key=lambda item: (item["validation_auc"], item["index"]),
            )
            if any(
                chosen[key] != expected[key]
                for key in ("index", "source", "source_sha256", "validation_auc")
            ):
                raise ValueError("selected record differs from validation-only rule")
            records = []
            sources = {seed_hash: seed_source}
            train_signatures = []
            for slot in range(8):
                request = read(directory + f"/slot_{slot:02d}/request.json")
                response = read(directory + f"/slot_{slot:02d}/response.json")
                candidate = pool[slot + 1]
                if (
                    response["source"] != candidate["source"]
                    or response["source_sha256"] != candidate["source_sha256"]
                ):
                    raise ValueError("response/pool source mismatch")
                if request["freeze_sha256"] != frozen_hash:
                    raise ValueError("request belongs to another freeze")
                if invariant is None:
                    invariant = request["messages"][0]
                if (
                    request["messages"][0] != invariant
                    or "Optimize anytime performance:" not in invariant["content"]
                ):
                    raise ValueError(
                        "invariant anytime instruction differs across requests"
                    )
                parent_hash = request["parent_sha256"]
                parent = sources[parent_hash]
                record = {
                    "index": slot,
                    "parent_sha256": parent_hash,
                    "source_sha256": candidate["source_sha256"],
                    "eligible": candidate["eligible"],
                    "train_auc": _auc(candidate["train"]),
                    "parent_train_auc": _auc(by_hash[parent_hash]["train"]),
                    "request_chars": sum(
                        len(message["content"]) for message in request["messages"]
                    ),
                    "response_record": directory + f"/slot_{slot:02d}/response.json",
                }
                if arm == "I":
                    if len(request["messages"]) != 1 or parent_hash != seed_hash:
                        raise ValueError(
                            "independent request received evolving context"
                        )
                else:
                    text = read(
                        directory + f"/slot_{slot:02d}/propagated_feedback.json"
                    )["text"]
                    payload = feedback_payload(text)
                    if set(payload) != {
                        "schema",
                        "current",
                        "aggregate_training_auc",
                    } or set(payload["current"]) != {"source_sha256", "valid", "tasks"}:
                        raise ValueError(
                            "unexpected history/hidden fields in current feedback"
                        )
                    if (
                        payload["current"]["source_sha256"] != parent_hash
                        or source_hash(text) != request["trace_feedback_sha256"]
                    ):
                        raise ValueError(
                            "parent and propagated feedback are misaligned"
                        )
                    tasks = payload["current"]["tasks"]
                    rows = by_hash[parent_hash]["train"]
                    if len(tasks) != 48 or len(rows) != 48:
                        raise ValueError(
                            "feedback loses registered training trajectories"
                        )
                    coverage = [
                        verify_projection(task, row) for task, row in zip(tasks, rows)
                    ]
                    if payload["aggregate_training_auc"] != _auc(rows):
                        raise ValueError(
                            "parent aggregate feedback differs from saved metrics"
                        )
                    expected_content = (
                        "Improve the current optimizer.\nCURRENT SOURCE:\n" + parent
                    )
                    if arm in {"R", "W"}:
                        expected_content += "\nTRAINING FEEDBACK:\n" + text
                    if request["messages"][1:] != [
                        {"role": "user", "content": expected_content}
                    ]:
                        raise ValueError(
                            "actual request is not the declared current-parent prompt"
                        )
                    summary["feedback"].append(
                        {
                            "outer": outer,
                            "arm": arm,
                            "slot": slot,
                            "visible_to_llm": arm in {"R", "W"},
                            "parent_sha256": parent_hash,
                            "text_sha256": source_hash(text),
                            "chars": len(text),
                            "trajectories": len(tasks),
                            "valid_trajectories": sum(task["valid"] for task in tasks),
                            "curve_points": sum(item[0] for item in coverage),
                            "observation_coordinates_exposed": sum(
                                item[1] for item in coverage
                            ),
                        }
                    )
                sources[candidate["source_sha256"]] = candidate["source"]
                records.append(record)
                if all(row["valid"] for row in candidate["train"]):
                    train_signatures.append(
                        digest(
                            [
                                [
                                    row["task_identity"],
                                    row["local_seed"],
                                    [o["x"] for o in row["observations"]],
                                ]
                                for row in candidate["train"]
                            ]
                        )
                    )
            records = lineage(records, seed_hash)
            width = 2 if arm == "W" else 1
            summary["searches"][f"{outer}/{arm}"] = {
                "records": records,
                "selected_slot": chosen["index"],
                "distinct_generated_sources": len(
                    {item["source_sha256"] for item in pool[1:]}
                ),
                "valid_training_policies": len(train_signatures),
                "distinct_valid_training_point_signatures": len(set(train_signatures)),
                "parent_counts": dict(
                    Counter(record["parent_sha256"] for record in records)
                ),
                "distinct_parents_per_round": [
                    len(
                        {
                            record["parent_sha256"]
                            for record in records[start : start + width]
                        }
                    )
                    for start in range(0, 8, width)
                ],
            }
            if arm == "R":
                selected[outer] = chosen
                chain = []
                cursor = chosen["index"]
                while cursor >= 0:
                    record = records[cursor]
                    if len(record["parent_origin_slots"]) != 1:
                        raise ValueError("selected source has ambiguous ancestry")
                    parent_slot = record["parent_origin_slots"][0]
                    old = pool[parent_slot + 1]["source"]
                    new = pool[cursor + 1]["source"]
                    diff = list(
                        difflib.unified_diff(
                            old.splitlines(),
                            new.splitlines(),
                            fromfile=f"slot_{parent_slot}",
                            tofile=f"slot_{cursor}",
                            lineterm="",
                        )
                    )
                    chain.append(
                        {
                            "slot": cursor,
                            "parent_slot": parent_slot,
                            "source_sha256": record["source_sha256"],
                            "train_auc": record["train_auc"],
                            "parent_train_auc": record["parent_train_auc"],
                            "added_lines": sum(
                                line.startswith("+") and not line.startswith("+++")
                                for line in diff
                            ),
                            "removed_lines": sum(
                                line.startswith("-") and not line.startswith("---")
                                for line in diff
                            ),
                            "diff": "\n".join(diff) + "\n",
                        }
                    )
                    cursor = parent_slot
                source = chosen["source"]
                tree = ast.parse(source)
                saved_rows = audit["per_seed"][str(outer)]["R"]["rows"]
                if any(
                    row["source_sha256"] != chosen["source_sha256"]
                    for row in saved_rows
                ):
                    raise ValueError("audited source differs from selected source")
                summary["selected_R"][str(outer)] = {
                    **chosen,
                    "source_record": chosen_path + ":source",
                    "source_bytes": len(source.encode()),
                    "source_lines": len(source.splitlines()),
                    "ast_nodes": sum(1 for _ in ast.walk(tree)),
                    "lineage": chain[::-1],
                    "audit_behavior": behavior(saved_rows),
                    "validation_behavior": behavior(expected["validation"]),
                }
    representative = min(
        selected,
        key=lambda outer: (
            selected[outer]["validation_auc"],
            frozen["config"]["outer_seeds"].index(outer),
        ),
    )
    if (
        representative != summary["representative_outer"]
        or selected[representative] != selection_barrier["representative"]
    ):
        raise ValueError("representative differs from frozen validation-only rule")
    for path, expected_hash in checked.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_hash:
            raise ValueError("input evidence changed during inspection")
    summary["input_file_sha256"] = checked
    summary["input_files_unchanged"] = len(checked)
    return summary


if __name__ == "__main__":
    data = inspect()
    path = OUT / "program_inspection.json"
    encoded = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
    if path.exists() and path.read_bytes() != encoded:
        raise RuntimeError("refusing to replace nonidentical inspection evidence")
    path.write_bytes(encoded)
    print(
        json.dumps(
            {
                "path": str(path),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "inputs": data["input_files_unchanged"],
            }
        )
    )
