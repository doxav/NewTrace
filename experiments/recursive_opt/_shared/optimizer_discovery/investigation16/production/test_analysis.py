"""Synthetic completeness and accounting tests; no objective or model execution."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import generation as G
from experiments.recursive_opt._shared.optimizer_discovery.investigation16.production import analysis as A


def bundle(seeds: list[int] | None = None) -> dict[str, Any]:
    """Build a complete small experiment including one invalid slot per pool."""
    seeds = seeds or [16411]
    tasks = {
        split: G.fresh_tasks("TEST-P1-ANALYSIS", split, 1)
        for split in ("train", "validation", "audit")
    }
    config = {
        "namespace": "TEST-P1-ANALYSIS",
        "outer_seeds": seeds,
        "arms": ["I", "C", "R", "W"],
        "slots": 2,
        "budget": 2,
        "local_replicates": 1,
        "timeout_s": 2.0,
        "cache_version": "p1-evaluation-v1",
    }
    sources = [
        B.SEED_SOURCE,
        "def propose(history, bounds, seed):\n    return [0.0 for _ in bounds]\n",
        "invalid candidate (",
    ]
    frozen = {
        "config": config,
        "tasks": tasks,
        "seed_sha256": B.source_hash(sources[0]),
        "seed_source": sources[0],
        "benchmark_manifest": B.MANIFEST,
        "files": {str(Path(B.__file__).resolve()): "frozen-evaluator-hash"},
    }
    result: dict[str, Any] = {
        "freeze": frozen,
        "audit": {"per_seed": {}},
        "pools": {},
        "selections": {},
        "slots": {},
        "cache": {},
        "events": [],
        "chronology": {"verified": True},
    }

    def panel(
        source: str, outer: int, split: str, score: float, valid: bool = True
    ) -> list[dict[str, Any]]:
        """Create fake deterministic trajectories and matching physical cache records."""
        rows = []
        for task in tasks[split]:
            local = G.local_seed(
                config["namespace"], int(B.digest([outer, 0])[:15], 16), task
            )
            row = {
                "valid": valid,
                "candidate_valid": valid,
                "status": "valid" if valid else "syntax_error",
                "source_sha256": B.source_hash(source),
                "task_identity": B.task_identity(task),
                "stratum": f"{task['family']}/{task['dimension']}",
                "local_seed": local,
                "budget": 2,
                "observations": [{"x": [0.0] * task["dimension"], "value": score}]
                * (2 if valid else 0),
                "proposal_attempts": [
                    {
                        "evaluation_index": index,
                        "status": "valid",
                        "policy": "candidate",
                        "stdout": "",
                        "stderr": "",
                    }
                    for index in range(2 if valid else 0)
                ],
                "fallback_used": False,
                "objective_calls": 2 if valid else 0,
                "unused_objective_allocation": 0 if valid else 2,
                "subprocess_executions": 4 if valid else 0,
                "execution_s": 0.1,
                "metrics": (
                    {
                        "auc": score,
                        "curve": [score, score],
                        "final_regret": score,
                        "attained": score <= 0.01,
                        "target_evaluations": 1 if score <= 0.01 else None,
                        "capped_target_evaluations": 1 if score <= 0.01 else 3,
                    }
                    if valid
                    else None
                ),
            }
            key = {
                "namespace": config["namespace"],
                "source_sha256": row["source_sha256"],
                "task_identity": row["task_identity"],
                "split": split,
                "outer": outer,
                "local_seed": local,
                "budget": 2,
                "deployment": split == "audit",
                "timeout_s": config["timeout_s"],
                "seed_sha256": frozen["seed_sha256"],
                "evaluator_version": config["cache_version"],
                "evaluator_sha256": frozen["files"][str(Path(B.__file__).resolve())],
            }
            digest = B.digest(key)
            entry = {"key": key, "row": row, "row_sha256": B.digest(row)}
            if digest in result["cache"]:
                assert result["cache"][digest] == entry
            else:
                result["cache"][digest] = entry
                result["events"].append(
                    {
                        "event": "evaluation_cache",
                        "arm": "I",
                        "outer": outer,
                        "split": split,
                        "key": digest,
                        "hit": False,
                    }
                )
            rows.append(copy.deepcopy(row))
        return rows

    for outer in seeds:
        result["audit"]["per_seed"][str(outer)] = {}
        for arm in config["arms"]:
            candidates = []
            for index, source in zip([-1, 0, 1], sources):
                valid = index != 1
                candidates.append(
                    {
                        "index": index,
                        "source": source,
                        "source_sha256": B.source_hash(source),
                        "train": panel(
                            source, outer, "train", 0.2 if index == -1 else 0.1, valid
                        ),
                        "validation": panel(
                            source,
                            outer,
                            "validation",
                            0.2 if index == -1 else 0.1,
                            valid,
                        ),
                        "eligible": valid,
                        "validation_auc": (
                            (0.2 if index == -1 else 0.1) if valid else None
                        ),
                    }
                )
                if index >= 0:
                    name = f"{outer}/{arm}/{index}"
                    result["slots"][name] = {
                        "path": f"raw/{outer}/{arm}/slot_{index:02d}",
                        "request": {
                            "outer": outer,
                            "arm": arm,
                            "slot": index,
                            "slot_id": name,
                            "parent_sha256": frozen["seed_sha256"],
                        },
                        "response": {
                            "completed": True,
                            "id": name,
                            "model": G.MODEL,
                            "source": source,
                            "source_sha256": B.source_hash(source),
                            "source_status": "valid" if valid else "syntax_error",
                            "parse_status": "parsed",
                            "finish_reason": "stop",
                            "usage": {"completion_tokens": 10} if index == 0 else {},
                        },
                        "attempts": [
                            {
                                "status": "transport_failure",
                                "transient": True,
                                "possible_remote_completion_or_duplicate_billing": True,
                            },
                            {"status": "completed", "id": name},
                        ],
                        "metadata": None,
                    }
            result["pools"][f"{outer}/{arm}"] = candidates
            selected = candidates[1]
            result["selections"][f"{outer}/{arm}"] = {
                key: selected[key]
                for key in ("index", "source", "source_sha256", "validation_auc")
            }
        for arm in ["A0", *config["arms"]]:
            source = sources[0] if arm == "A0" else sources[1]
            rows = panel(source, outer, "audit", 0.2 if arm == "A0" else 0.3)
            result["audit"]["per_seed"][str(outer)][arm] = {
                "source_sha256": B.source_hash(source),
                "rows": rows,
                "auc": B.aggregate(rows, "auc"),
                "final_regret": B.aggregate(rows, "final_regret"),
                "fallback_trajectories": 0,
            }
    return result


def test_keeps_all_six_seeds_invalid_slots_and_negative_contrast() -> None:
    """All unfavorable deployment outcomes and failed proposal allocations survive."""
    raw = bundle([16411, 16423, 16437, 16441, 16453, 16467])
    result = A.summarize(raw)
    assert len(result["per_seed"]) == 6
    assert set(result["contrasts"]) == {"R-I", "R-C", "W-R", "R-A0"}
    assert result["contrasts"]["R-A0"]["mean"] == pytest.approx(0.1)
    assert result["contrasts"]["R-A0"]["interpretation"] == "negative signal"
    assert result["arms"]["R"]["generation"]["responses"] == 12
    assert result["arms"]["R"]["generation"]["source_invalid_fraction"] == 0.5
    assert result["arms"]["R"]["candidate_eligibility"]["ineligible_generated"] == 6
    assert result["resources"]["completed_responses"] == 48
    assert result["resources"]["transport_attempts"] == 96
    assert result["resources"]["transport_failures"] == 48
    assert result["resources"]["response_usage"]["cost_usd"]["reported_sum"] is None
    assert (
        result["resources"]["response_usage"]["completion_tokens"]["missing_responses"]
        == 24
    )
    assert result["resources"]["logical"]["train"]["unused_objective_allocation"] > 0
    assert (
        result["resources"]["physical_cache"]["train"]["objective_calls"]
        < result["resources"]["logical"]["train"]["objective_calls"]
    )


@pytest.mark.parametrize(
    "remove", ["seed", "arm", "slot", "candidate", "trajectory", "cache"]
)
def test_refuses_missing_evidence(remove: str) -> None:
    """No aggregation path silently excludes a missing seed, slot or bad row."""
    raw = bundle()
    if remove == "seed":
        raw["audit"]["per_seed"].pop("16411")
    elif remove == "arm":
        raw["audit"]["per_seed"]["16411"].pop("W")
    elif remove == "slot":
        raw["slots"].pop("16411/R/1")
    elif remove == "candidate":
        raw["pools"]["16411/R"].pop()
    elif remove == "trajectory":
        raw["pools"]["16411/R"][2]["train"].pop()
    else:
        raw["cache"].pop(next(iter(raw["cache"])))
    with pytest.raises(ValueError):
        A.summarize(raw)


@pytest.mark.parametrize(
    "tamper",
    ["invalid_metric", "selected_hash", "selection_rule", "audit_auc", "cache_hash"],
)
def test_refuses_semantic_or_integrity_tampering(tamper: str) -> None:
    """Invalid scores cannot be fabricated and selected identities cannot drift."""
    raw = bundle()
    if tamper == "invalid_metric":
        raw["pools"]["16411/R"][2]["train"][0]["metrics"] = {"auc": 0.0}
    elif tamper == "selected_hash":
        raw["selections"]["16411/R"]["source_sha256"] = "wrong"
    elif tamper == "selection_rule":
        raw["selections"]["16411/R"] = {
            key: raw["pools"]["16411/R"][0][key]
            for key in ("index", "source", "source_sha256", "validation_auc")
        }
    elif tamper == "audit_auc":
        raw["audit"]["per_seed"]["16411"]["R"]["auc"] = 0.0
    else:
        raw["cache"][next(iter(raw["cache"]))]["row_sha256"] = "wrong"
    with pytest.raises(ValueError):
        A.summarize(raw)


def test_no_objective_calls_during_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reporting cannot inspect new objective outcomes or reconstruct hidden scales."""
    raw = bundle()

    def forbidden(*args: Any, **kwargs: Any) -> None:
        """Fail if a supposedly pure analysis executes an evaluator."""
        raise AssertionError("unexpected evaluator access")

    monkeypatch.setattr(B, "evaluate", forbidden)
    monkeypatch.setattr(B, "objective", forbidden)
    monkeypatch.setattr(B, "normalization", forbidden)
    assert A.summarize(raw)["resources"]["completed_responses"] == 8


def test_deployment_fallback_preserves_values_and_candidate_failure() -> None:
    """A valid deployment metric must retain a failed candidate and actual history."""
    raw = bundle()
    row = raw["audit"]["per_seed"]["16411"]["R"]["rows"][0]
    identity = (row["source_sha256"], row["task_identity"], row["local_seed"])
    changed = copy.deepcopy(row)
    changed["candidate_valid"] = False
    changed["fallback_used"] = True
    changed["proposal_attempts"] = [
        {"evaluation_index": 0, "status": "exception", "policy": "candidate"},
        *[{**attempt, "policy": "fallback"} for attempt in row["proposal_attempts"]],
    ]
    changed["subprocess_executions"] += 1
    for entry in raw["cache"].values():
        if (
            entry["key"]["split"] == "audit"
            and tuple(
                entry["row"][key]
                for key in ("source_sha256", "task_identity", "local_seed")
            )
            == identity
        ):
            entry["row"] = copy.deepcopy(changed)
            entry["row_sha256"] = B.digest(changed)
    for arm in A.ARMS[1:]:
        raw["audit"]["per_seed"]["16411"][arm]["rows"][0] = copy.deepcopy(changed)
        raw["audit"]["per_seed"]["16411"][arm]["fallback_trajectories"] = 1
    result = A.summarize(raw)
    assert result["arms"]["R"]["auc"]["mean"] == pytest.approx(0.3)
    assert result["arms"]["R"]["deployment"][
        "candidate_invalid_fraction"
    ] == pytest.approx(1 / 6)
    assert result["arms"]["R"]["deployment"]["fallback_fraction"] == pytest.approx(
        1 / 6
    )
    assert result["arms"]["R"]["deployment"]["allocation"]["objective_calls"] == 12


def test_seed_selected_with_eligible_but_worse_generated_alternative() -> None:
    """Selecting the seed is distinct from having no valid generated replacement."""
    raw = bundle()
    candidate_hash = raw["pools"]["16411/R"][1]["source_sha256"]
    for entry in raw["cache"].values():
        if (
            entry["key"]["split"] == "validation"
            and entry["key"]["source_sha256"] == candidate_hash
        ):
            row = entry["row"]
            row["metrics"].update(
                {"auc": 0.4, "curve": [0.4, 0.4], "final_regret": 0.4}
            )
            entry["row_sha256"] = B.digest(row)
    for arm in A.ARMS[1:]:
        candidates = raw["pools"][f"16411/{arm}"]
        for row in candidates[1]["validation"]:
            row["metrics"].update(
                {"auc": 0.4, "curve": [0.4, 0.4], "final_regret": 0.4}
            )
        candidates[1]["validation_auc"] = 0.4
        raw["selections"][f"16411/{arm}"] = {
            key: candidates[0][key]
            for key in ("index", "source", "source_sha256", "validation_auc")
        }
        raw["audit"]["per_seed"]["16411"][arm] = copy.deepcopy(
            raw["audit"]["per_seed"]["16411"]["A0"]
        )
    # The synthetic fixture originally materialized the replaced policy's audit;
    # a real frozen run would never do so. Remove it from this new fixture only.
    removed = {
        key
        for key, entry in raw["cache"].items()
        if entry["key"]["split"] == "audit"
        and entry["key"]["source_sha256"] == candidate_hash
    }
    raw["cache"] = {
        key: entry for key, entry in raw["cache"].items() if key not in removed
    }
    raw["events"] = [event for event in raw["events"] if event["key"] not in removed]
    result = A.summarize(raw)
    assert (
        result["arms"]["R"]["candidate_eligibility"]["seed_index_selection_count"] == 1
    )
    assert (
        result["arms"]["R"]["candidate_eligibility"][
            "seed_selection_despite_eligible_replacement_count"
        ]
        == 1
    )
    assert (
        result["arms"]["R"]["candidate_eligibility"]["no_eligible_replacement_count"]
        == 0
    )
    assert result["contrasts"]["R-A0"]["interpretation"] == "no detectable difference"


def test_receipts_fill_only_missing_usage_without_double_counting() -> None:
    """Usage provenance and missing coverage remain visible when provider receipts arrive."""
    raw = bundle()
    slot = raw["slots"]["16411/R/0"]
    slot["metadata"] = {
        "id": slot["response"]["id"],
        "tokens_completion": 999,
        "total_cost": 0.012,
    }
    result = A.summarize(raw)
    known = result["resources"]["known_usage"]["usage"]
    assert known["completion_tokens"]["reported_sum"] == 40
    assert known["cost_usd"]["reported_sum"] == 0.012
    assert known["cost_usd"]["missing_responses"] == 7
    slot["metadata"]["id"] = "different-response"
    with pytest.raises(ValueError, match="receipt"):
        A.summarize(raw)


def test_logical_json_paths_support_gzip_and_refuse_duplicates(tmp_path: Any) -> None:
    """Compressed raw files map to E.read's logical path exactly once."""
    (tmp_path / "row.json.gz").write_bytes(b"placeholder")
    assert A._json_paths(tmp_path) == [tmp_path / "row.json"]
    (tmp_path / "row.json").write_text("{}")
    with pytest.raises(ValueError, match="compressed"):
        A._json_paths(tmp_path)


def test_refuses_unallocated_physical_cache_rows() -> None:
    """Extra evaluations cannot be concealed inside a valid-looking cache total."""
    raw = bundle()
    entry = copy.deepcopy(next(iter(raw["cache"].values())))
    entry["key"]["outer"] = 99999
    raw["cache"][B.digest(entry["key"])] = entry
    with pytest.raises(ValueError, match="unallocated"):
        A.summarize(raw)


def test_describes_parser_resource_failure_without_dropping_source() -> None:
    """Even a source whose AST parser reaches its depth limit remains reportable."""
    source = "+" * 10000 + "1"
    proposals = [{"source": source, "source_status": "syntax_error", "usage": {}}]
    result = A.safe_describe(proposals, [])
    assert result["responses"] == 1
    assert result["source_complexity"] == [
        {"source_bytes": len(source.encode()), "ast_nodes": None}
    ]


@pytest.mark.parametrize(
    "field",
    [
        "namespace",
        "deployment",
        "timeout_s",
        "seed_sha256",
        "evaluator_version",
        "evaluator_sha256",
    ],
)
def test_rehashed_cache_cannot_change_frozen_execution_fields(field: str) -> None:
    """A fresh digest cannot legitimize a changed evaluator or deployment contract."""
    raw = bundle()
    digest = next(iter(raw["cache"]))
    entry = raw["cache"].pop(digest)
    entry["key"][field] = "modified"
    raw["cache"][B.digest(entry["key"])] = entry
    for event in raw["events"]:
        if event["key"] == digest:
            event["key"] = B.digest(entry["key"])
    with pytest.raises(ValueError, match="frozen cache"):
        A.summarize(raw)


def test_generated_execution_denominator_excludes_trusted_seed() -> None:
    """Source and execution invalidity use generated candidates, not seed-inclusive pools."""
    result = A.summarize(bundle())
    assert result["arms"]["R"]["generation"]["trajectory_invalid_fraction"] == 0.5
    assert result["arms"]["R"]["generation"]["trajectories"] == 24
    assert result["per_seed"]["16411"]["R"]["search"]["train"]["trajectories"] == 18
