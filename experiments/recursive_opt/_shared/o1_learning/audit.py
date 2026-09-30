"""Recompute EXP19-S3 evidence without generating proposals or changing selections."""

from __future__ import annotations

import collections
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.o1_learning import analysis, study


def read(path: Path) -> Any:
    """Read preserved JSON evidence without modifying it."""
    return json.loads(path.read_text())


def replay_stage(
    final: dict[str, Any], prefix_file: str, phase: str, budget: int, arm_count: int
) -> dict[str, Any]:
    """Recompute every frozen prefix and check its exact persisted artifact identity."""
    root = study.ROOT
    seeds = final["seeds"]
    frozen = read(root / prefix_file)
    assert len(frozen) == arm_count * len(seeds)
    recomputed: dict[str, dict[int, float]] = {}
    hitting: dict[str, dict[int, int | None]] = {}
    selected = []
    for row in frozen:
        seed, arm = row["seed"], row["arm"]
        path = root / "raw" / phase / f"{arm}_{seed}" / "result.json"
        raw = read(path)
        eligible = {
            candidate["hash"]: candidate
            for candidate in raw["candidate_results"]
            if candidate["valid"]
        }
        assert len(row["prefixes"]) == budget + 1
        for count, candidate in enumerate(row["prefixes"]):
            assert candidate["hash"] == study.digest(candidate["configuration"])
            assert eligible[candidate["hash"]] == candidate
            assert candidate["calls"] <= count
        curve = [
            study.score(candidate["configuration"], study.panels(seed)["test"])
            for candidate in row["prefixes"]
        ]
        saved = next(
            item
            for item in final["per_seed"]
            if item["seed"] == seed and item["arm"] == arm
        )
        assert curve == saved["test_curve"]
        recomputed.setdefault(arm, {})[seed] = curve[-1]
        hitting.setdefault(arm, {})[seed] = study.first_hit(curve)
        selected.append(
            {
                "seed": seed,
                "arm": arm,
                "path": str(path.relative_to(root)),
                "hash": row["prefixes"][-1]["hash"],
            }
        )
    return {
        "paired_values": recomputed,
        "hitting_times_censored": hitting,
        "selected_artifacts": selected,
        "means": {
            arm: statistics.mean(values.values()) for arm, values in recomputed.items()
        },
        "medians": {
            arm: statistics.median(values.values())
            for arm, values in recomputed.items()
        },
    }


def audit() -> dict[str, Any]:
    """Require complete results, verify selected artifacts, and account for every response."""
    root = study.ROOT
    if not (root / "s3_results.json").exists():
        raise RuntimeError("S3 results are incomplete: audit must not open TEST")
    final = read(root / "s3_results.json")
    seeds = final["seeds"]
    s3 = replay_stage(final, "s3_prefix_selections_frozen.json", "paired_s3", 8, 3)
    recomputed, hitting, selected = (
        s3["paired_values"],
        s3["hitting_times_censored"],
        s3["selected_artifacts"],
    )
    for label, left, right in [
        ("O1_minus_standard", "O1_selected", "standard"),
        ("recursive_minus_standard", "recursive_selected", "standard"),
        ("recursive_minus_O1", "recursive_selected", "O1_selected"),
    ]:
        assert (
            analysis.paired(recomputed[left], recomputed[right], seeds=seeds)
            == final[label]
        )
    s4 = None
    if (root / "s4_results.json").exists():
        result4 = read(root / "s4_results.json")
        s4 = replay_stage(
            result4, "s4_prefix_selections_frozen.json", "selection_s4", 4, 2
        )
        assert (
            analysis.paired(
                s4["paired_values"]["train_only_fit"],
                s4["paired_values"]["standard_B6"],
                seeds=result4["seeds"],
            )
            == result4["train_only_minus_standard"]
        )
        protocol4 = read(root / "stage_s4_protocol.json")
        assert (
            hashlib.sha256((root / "selection_diagnostic.py").read_bytes()).hexdigest()
            == protocol4["source_sha256"]
        )
    response_paths = sorted(root.rglob("response_*.json"))
    totals: dict[str, dict[str, Any]] = {}
    identifiers = []
    empty = []
    finishes: collections.Counter[str] = collections.Counter()
    providers: collections.Counter[str] = collections.Counter()
    caps: collections.Counter[int] = collections.Counter()
    for path in response_paths:
        evidence = read(path)
        response = evidence["response"]
        usage = response.get("usage") or {}
        request = read(path.with_name(path.name.replace("response_", "request_")))
        settings = request["kwargs"]
        assert response["model"] == study.MODEL
        assert settings["temperature"] == 0.6 and settings["top_p"] == 1.0
        assert settings["extra_body"] == {"reasoning": {"effort": "low"}}
        assert settings["max_tokens"] in {8000, 16000}
        caps[settings["max_tokens"]] += 1
        identifiers.append(response["id"])
        choice = response["choices"][0]
        finishes[choice["finish_reason"]] += 1
        providers[response.get("provider", "unknown")] += 1
        if not (choice["message"].get("content") or "").strip():
            empty.append(str(path.relative_to(root)))
        phase = path.parent.parent.name
        if "s3/raw" in path.as_posix():
            phase = "s3_" + phase
        bucket = totals.setdefault(
            phase,
            {
                "responses": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "reported_cost_usd": 0.0,
                "cost_missing": 0,
            },
        )
        bucket["responses"] += 1
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            bucket[key] += usage.get(key) or 0
        bucket["reasoning_tokens"] += (
            usage.get("completion_tokens_details") or {}
        ).get("reasoning_tokens") or 0
        bucket["reported_cost_usd"] += usage.get("cost") or 0.0
        bucket["cost_missing"] += usage.get("cost") is None
    assert len(identifiers) == len(
        set(identifiers)
    ), "duplicate completed provider response"
    missing = [
        str(path.relative_to(root))
        for path in root.rglob("request_*.json")
        if not path.with_name(path.name.replace("request_", "response_")).exists()
    ]
    allocations = []
    for path in sorted(root.rglob("result.json")):
        row = read(path)
        if "actual_trainer_evaluations" not in row:
            continue
        production = read(path.with_name("production.json"))
        accounted = (
            production.get("budget", {}).get("accounted", {}).get("evaluator_runs")
        )
        records = read(path.with_name("evaluations.json"))
        assert row["actual_trainer_evaluations"] == len(records)
        allocations.append(
            {
                "path": str(path.relative_to(root)),
                "valid": row["valid"],
                "calls": row["calls"],
                "completed_task_scores": len(records),
                "accounted_evaluator_attempts": accounted,
                "attempts_without_task_score": (
                    None if accounted is None else accounted - len(records)
                ),
                "selection_evaluations": row["extra_selection_evaluations"],
                "invalid_execution_observations": [
                    observation
                    for observation in production.get("metadata", {}).get(
                        "menu_observations", []
                    )
                    if not observation["valid"]
                ],
                "menu_evidence": production.get("metadata", {}).get("menu_evidence"),
                "curriculum_events": production.get("metadata", {}).get(
                    "curriculum_events", []
                ),
            }
        )
    source_manifest = read(root / "sources_S3_manifest.json")
    repair = read(root / "s3_reporting_repair.json")
    for name, digest in source_manifest["hashes"].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        assert actual == digest or (
            name.endswith("experiments/recursive_opt/_shared/o1_learning/analysis.py")
            and actual == repair["after_sha256"]
        ), name
    return {
        "experiment": "EXP-19",
        "S4": s4,
        "recomputation_matches": True,
        "paired_values": recomputed,
        "hitting_times_censored": hitting,
        "means": {
            arm: statistics.mean(values.values()) for arm, values in recomputed.items()
        },
        "medians": {
            arm: statistics.median(values.values())
            for arm, values in recomputed.items()
        },
        "selected_artifacts": selected,
        "usage_by_phase": totals,
        "completed_responses": len(response_paths),
        "unique_response_ids": len(set(identifiers)),
        "empty_responses": empty,
        "finish_reasons": dict(finishes),
        "providers": dict(providers),
        "completion_caps": dict(caps),
        "unresolved_requests": missing,
        "transport_events": [
            read(path) for path in sorted(root.rglob("transport_*.json"))
        ],
        "transport_limit": "four attempts per request; S1/S2 events not captured, exact realized attempts unavailable there",
        "task_accounting": allocations,
        "freeze_precedes_result_write": (root / "s3_prefix_selections_frozen.json")
        .stat()
        .st_mtime
        <= (root / "s3_results.json").stat().st_mtime,
        "selection_order_evidence": "freeze_and_score persists every prefix before entering TEST loop; unit test verifies this boundary",
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
