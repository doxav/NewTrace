"""Frozen P1 analysis of complete preserved evidence, without evaluating any policy."""

from __future__ import annotations

import argparse
import ast
import math
import statistics
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.evidence import describe
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import generation as G

ARMS = ("A0", "I", "C", "R", "W")
CONTRASTS = (("R", "I"), ("R", "C"), ("W", "R"), ("R", "A0"))
SPLITS = ("train", "validation", "audit")


def safe_describe(
    proposals: list[dict[str, Any]], trajectories: list[dict[str, Any]]
) -> dict[str, Any]:
    """Reuse descriptive statistics while retaining sources with AST resource errors."""
    try:
        return describe(proposals, trajectories)
    except (ValueError, RecursionError, MemoryError):
        sanitized = [{**proposal, "source": ""} for proposal in proposals]
        result = describe(sanitized, trajectories)
        complexity = []
        for proposal in proposals:
            source = proposal["source"]
            try:
                nodes = sum(1 for _ in ast.walk(ast.parse(source)))
            except (SyntaxError, ValueError, RecursionError, MemoryError):
                nodes = None
            complexity.append(
                {"source_bytes": len(source.encode()), "ast_nodes": nodes}
            )
        result["source_complexity"] = complexity
        return result


def _require(condition: bool, message: str) -> None:
    """Refuse incomplete or inconsistent scientific evidence rather than dropping it."""
    if not condition:
        raise ValueError(message)


def _same_number(left: float, right: float) -> bool:
    """Permit only roundoff in arithmetic recomputation of serialized metrics."""
    return (
        math.isfinite(left)
        and math.isfinite(right)
        and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-14)
    )


def _allocation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count every logical or physical trajectory, including partial invalid runs."""
    return {
        "trajectories": len(rows),
        "allocated_objective_calls": sum(row["budget"] for row in rows),
        **{
            field: sum(row[field] for row in rows)
            for field in (
                "objective_calls",
                "unused_objective_allocation",
                "subprocess_executions",
                "execution_s",
            )
        },
        "invalid_trajectories": sum(not row["valid"] for row in rows),
    }


def _deployment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize complete deployment outcomes without concealing candidate failures."""
    result = safe_describe([], rows)
    result.pop("usage")
    result.pop("source_complexity")
    result.update(
        {
            "target_attainment_rate": statistics.mean(
                row["metrics"]["attained"] for row in rows
            ),
            "mean_capped_target_evaluations": statistics.mean(
                row["metrics"]["capped_target_evaluations"] for row in rows
            ),
            "censoring_convention": "B+1 represents nonattainment, not an observed hitting time",
            "fallback_fraction": sum(row["fallback_used"] for row in rows) / len(rows),
            "allocation": _allocation(rows),
        }
    )
    return result


def _validate_row(row: dict[str, Any], budget: int, deployment: bool) -> None:
    """Check accounting and preserved metric arithmetic without new reference calls."""
    _require(row["budget"] == budget, "trajectory budget mismatch")
    calls = row["objective_calls"]
    _require(
        type(calls) is int and 0 <= calls <= budget, "invalid objective accounting"
    )
    _require(
        calls == len(row["observations"]), "observations disagree with objective calls"
    )
    _require(
        row["unused_objective_allocation"] == budget - calls,
        "unused allocation mismatch",
    )
    _require(deployment or not row["fallback_used"], "fallback outside deployment")
    _require(
        not row["fallback_used"] or not row["candidate_valid"],
        "fallback masks candidate validity",
    )
    metric = row["metrics"]
    if not row["valid"]:
        _require(
            not deployment and metric is None,
            "invalid trajectory cannot receive a numeric metric",
        )
        return
    _require(
        calls == budget and metric is not None,
        "valid trajectory must complete its budget",
    )
    curve = metric["curve"]
    _require(
        len(curve) == budget
        and all(math.isfinite(value) and value >= 0 for value in curve),
        "invalid normalized curve",
    )
    _require(
        all(right <= left + 1e-14 for left, right in pairwise(curve)),
        "anytime regret must be nonincreasing",
    )
    _require(
        _same_number(metric["auc"], statistics.mean(curve)),
        "AUC differs from preserved curve",
    )
    _require(
        _same_number(metric["final_regret"], curve[-1]),
        "final regret differs from curve",
    )
    threshold = B.MANIFEST["target"]
    target = next(
        (index + 1 for index, value in enumerate(curve) if value <= threshold), None
    )
    _require(
        metric["target_evaluations"] == target
        and metric["attained"] == (target is not None),
        "target attainment mismatch",
    )
    _require(
        metric["capped_target_evaluations"]
        == (target if target is not None else budget + 1),
        "censored target time mismatch",
    )


def _cache_index(bundle: dict[str, Any]) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Validate unique physical cache artifacts and index their scientific identities."""
    result = {}
    for digest, entry in bundle["cache"].items():
        key, row = entry["key"], entry["row"]
        config, frozen = bundle["freeze"]["config"], bundle["freeze"]
        _require(key["split"] in SPLITS, "unregistered frozen cache split")
        expected_fields = {
            "namespace": config["namespace"],
            "budget": config["budget"],
            "deployment": key["split"] == "audit",
            "timeout_s": config["timeout_s"],
            "seed_sha256": frozen["seed_sha256"],
            "evaluator_version": config["cache_version"],
            "evaluator_sha256": frozen["files"][str(Path(B.__file__).resolve())],
        }
        _require(
            all(key.get(field) == value for field, value in expected_fields.items()),
            "frozen cache execution fields differ",
        )
        _require(
            B.digest(key) == digest and B.digest(row) == entry["row_sha256"],
            "cache hash mismatch",
        )
        identity = tuple(
            key[field]
            for field in (
                "outer",
                "split",
                "source_sha256",
                "task_identity",
                "local_seed",
            )
        )
        _require(identity not in result, "duplicate physical cache identity")
        _require(
            all(
                row[field] == key[field]
                for field in ("source_sha256", "task_identity", "local_seed", "budget")
            ),
            "cache row identity mismatch",
        )
        _validate_row(
            row, bundle["freeze"]["config"]["budget"], key["split"] == "audit"
        )
        result[identity] = entry
    return result


def _panel(
    bundle: dict[str, Any],
    cache: dict[tuple[Any, ...], dict[str, Any]],
    rows: list[dict[str, Any]],
    source_hash: str,
    outer: int,
    split: str,
) -> None:
    """Require exactly the frozen task/local-seed panel, including invalid trajectories."""
    frozen = bundle["freeze"]
    config = frozen["config"]
    expected = {
        (
            B.task_identity(task),
            G.local_seed(
                config["namespace"], int(B.digest([outer, replicate])[:15], 16), task
            ),
        ): f"{task['family']}/{task['dimension']}"
        for task in frozen["tasks"][split]
        for replicate in range(config["local_replicates"])
    }
    actual = [(row["task_identity"], row["local_seed"]) for row in rows]
    _require(
        len(actual) == len(expected) and set(actual) == set(expected),
        "missing or duplicate trajectory in frozen panel",
    )
    for row, identity in zip(rows, actual):
        _require(
            row["source_sha256"] == source_hash
            and row["stratum"] == expected[identity],
            "panel source or stratum mismatch",
        )
        _validate_row(row, config["budget"], split == "audit")
        physical = cache.get((outer, split, source_hash, *identity))
        _require(
            physical is not None and B.digest(row) == physical["row_sha256"],
            "panel does not match preserved physical cache row",
        )


def _known_usage(slots: list[dict[str, Any]]) -> dict[str, Any]:
    """Fill missing response usage only from matching safe provider receipts, retaining provenance."""
    mapping = {
        "prompt_tokens": "tokens_prompt",
        "completion_tokens": "tokens_completion",
        "reasoning_tokens": "native_tokens_reasoning",
        "cost_usd": "total_cost",
    }
    augmented, sources = [], []
    for slot in slots:
        response = slot["response"]
        usage = dict(response["usage"])
        provenance = {
            field: "response" for field, value in usage.items() if value is not None
        }
        metadata = slot.get("metadata")
        if metadata is not None:
            _require(
                metadata.get("id") == response["id"],
                "provider receipt identity mismatch",
            )
            for field, provider_field in mapping.items():
                if (
                    usage.get(field) is None
                    and metadata.get(provider_field) is not None
                ):
                    usage[field] = metadata[provider_field]
                    provenance[field] = "provider_generation_receipt"
        _require(
            all(
                type(value) in (int, float) and math.isfinite(value) and value >= 0
                for value in usage.values()
                if value is not None
            ),
            "invalid reported usage",
        )
        augmented.append({**response, "usage": usage})
        sources.append(
            {"slot": slot["path"], "id": response["id"], "fields": provenance}
        )
    return {
        "usage": safe_describe(augmented, [])["usage"],
        "field_sources": sources,
        "provider_receipts": sum(slot.get("metadata") is not None for slot in slots),
    }


def summarize(bundle: dict[str, Any]) -> dict[str, Any]:
    """Compute all frozen paired contrasts and resource summaries from complete raw evidence."""
    frozen, audit = bundle["freeze"], bundle["audit"]
    config = frozen["config"]
    seeds, slots = config["outer_seeds"], config["slots"]
    _require(
        config["arms"] == list(ARMS[1:]) and seeds and len(seeds) == len(set(seeds)),
        "invalid registered arms or seeds",
    )
    _require(
        frozen["seed_sha256"] == B.source_hash(frozen["seed_source"]),
        "seed source integrity mismatch",
    )
    _require(
        set(audit["per_seed"]) == {str(seed) for seed in seeds},
        "missing or extra outer seed",
    )
    pools = {f"{outer}/{arm}" for outer in seeds for arm in ARMS[1:]}
    expected_slots = {
        f"{outer}/{arm}/{index}"
        for outer in seeds
        for arm in ARMS[1:]
        for index in range(slots)
    }
    _require(
        set(bundle["pools"]) == pools and set(bundle["selections"]) == pools,
        "missing or extra arm pool/selection",
    )
    _require(
        set(bundle["slots"]) == expected_slots,
        "missing or extra completed proposal slot",
    )
    cache = _cache_index(bundle)
    logical: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    descriptions: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS[1:]}
    per_seed, identifiers = {}, set()
    for outer in seeds:
        actual = audit["per_seed"][str(outer)]
        _require(set(actual) == set(ARMS), "missing or extra audit arm")
        per_seed[str(outer)] = {}
        for arm in ARMS:
            value = actual[arm]
            if arm == "A0":
                selected = {
                    "index": -1,
                    "source": frozen["seed_source"],
                    "source_sha256": frozen["seed_sha256"],
                    "validation_auc": None,
                }
                pool_summary = None
            else:
                name = f"{outer}/{arm}"
                candidates = bundle["pools"][name]
                _require(
                    [candidate["index"] for candidate in candidates]
                    == list(range(-1, slots)),
                    "missing or reordered pool candidate",
                )
                for candidate in candidates:
                    source_hash = candidate["source_sha256"]
                    _require(
                        source_hash == B.source_hash(candidate["source"]),
                        "candidate source integrity mismatch",
                    )
                    for split in ("train", "validation"):
                        _panel(
                            bundle, cache, candidate[split], source_hash, outer, split
                        )
                        logical[split].extend(candidate[split])
                    eligible = all(
                        row["valid"]
                        for row in candidate["train"] + candidate["validation"]
                    )
                    expected_auc = (
                        B.aggregate(candidate["validation"], "auc")
                        if eligible
                        else None
                    )
                    _require(
                        candidate["eligible"] == eligible
                        and candidate["validation_auc"] == expected_auc,
                        "candidate eligibility or validation metric mismatch",
                    )
                    if candidate["index"] == -1:
                        _require(
                            eligible and source_hash == frozen["seed_sha256"],
                            "trusted seed missing or invalid",
                        )
                    else:
                        slot = bundle["slots"][f"{outer}/{arm}/{candidate['index']}"]
                        response, request = slot["response"], slot["request"]
                        _require(
                            response["completed"] and response["id"] not in identifiers,
                            "incomplete or duplicate completed response",
                        )
                        identifiers.add(response["id"])
                        _require(
                            response["source_sha256"] == source_hash
                            and response["source"] == candidate["source"],
                            "response source differs from evaluated source",
                        )
                        _require(
                            (request["outer"], request["arm"], request["slot"])
                            == (outer, arm, candidate["index"]),
                            "request slot identity mismatch",
                        )
                        _require(
                            sum(
                                attempt["status"] == "completed"
                                for attempt in slot["attempts"]
                            )
                            == 1,
                            "completed slot requires one retained successful attempt",
                        )
                best = min(
                    (candidate for candidate in candidates if candidate["eligible"]),
                    key=lambda candidate: (
                        candidate["validation_auc"],
                        candidate["index"],
                    ),
                )
                selected = bundle["selections"][name]
                _require(
                    all(
                        selected[field] == best[field]
                        for field in (
                            "index",
                            "source",
                            "source_sha256",
                            "validation_auc",
                        )
                    ),
                    "selection differs from frozen validation rule",
                )
                generated = candidates[1:]
                pool_summary = {
                    "eligible_generated": sum(
                        candidate["eligible"] for candidate in generated
                    ),
                    "ineligible_generated": sum(
                        not candidate["eligible"] for candidate in generated
                    ),
                    "selected_seed_index": selected["index"] == -1,
                    "selected_seed_source": selected["source_sha256"]
                    == frozen["seed_sha256"],
                    "train": _allocation(
                        [row for candidate in candidates for row in candidate["train"]]
                    ),
                    "validation": _allocation(
                        [
                            row
                            for candidate in candidates
                            for row in candidate["validation"]
                        ]
                    ),
                    "prefix_validation_auc": [
                        min(
                            candidate["validation_auc"]
                            for candidate in candidates[: prefix + 1]
                            if candidate["eligible"]
                        )
                        for prefix in range(1, slots + 1)
                    ],
                    "lineage": [
                        {
                            "index": index,
                            "parent_sha256": bundle["slots"][f"{outer}/{arm}/{index}"][
                                "request"
                            ]["parent_sha256"],
                            "source_sha256": candidates[index + 1]["source_sha256"],
                        }
                        for index in range(slots)
                    ],
                }
                descriptions[arm].append(pool_summary)
            _require(
                value["source_sha256"] == selected["source_sha256"],
                "audit source differs from frozen selection",
            )
            rows = value["rows"]
            _panel(bundle, cache, rows, value["source_sha256"], outer, "audit")
            for metric in ("auc", "final_regret"):
                _require(
                    _same_number(value[metric], B.aggregate(rows, metric)),
                    "audit aggregate mismatch",
                )
            _require(
                value["fallback_trajectories"]
                == sum(row["fallback_used"] for row in rows),
                "audit fallback count mismatch",
            )
            logical["audit"].extend(rows)
            arm_rows[arm].extend(rows)
            per_seed[str(outer)][arm] = {
                "auc": value["auc"],
                "final_regret": value["final_regret"],
                "selection": {
                    key: selected[key]
                    for key in ("index", "source_sha256", "validation_auc")
                },
                "selected_source_record": (
                    "freeze.json:seed_source"
                    if arm == "A0"
                    else f"raw/{outer}/{arm}/selection.json:source"
                ),
                "deployment": _deployment(rows),
                "search": pool_summary,
            }
    allocated_cache_identities = {
        (
            outer,
            split,
            candidate["source_sha256"],
            row["task_identity"],
            row["local_seed"],
        )
        for outer in seeds
        for arm in ARMS[1:]
        for candidate in bundle["pools"][f"{outer}/{arm}"]
        for split in ("train", "validation")
        for row in candidate[split]
    } | {
        (
            outer,
            "audit",
            value["source_sha256"],
            row["task_identity"],
            row["local_seed"],
        )
        for outer in seeds
        for value in audit["per_seed"][str(outer)].values()
        for row in value["rows"]
    }
    _require(
        set(cache) == allocated_cache_identities,
        "unallocated or missing physical cache trajectory",
    )
    arms = {}
    for arm in ARMS:
        values = [per_seed[str(outer)][arm]["auc"] for outer in seeds]
        arm_slots = (
            [
                bundle["slots"][f"{outer}/{arm}/{index}"]
                for outer in seeds
                for index in range(slots)
            ]
            if arm != "A0"
            else []
        )
        training_rows = (
            [
                row
                for outer in seeds
                for candidate in bundle["pools"][f"{outer}/{arm}"][1:]
                for split in ("train", "validation")
                for row in candidate[split]
            ]
            if arm != "A0"
            else []
        )
        arms[arm] = {
            "auc": {
                "per_seed": values,
                "mean": statistics.mean(values),
                "median": statistics.median(values),
            },
            "final_regret": {
                "mean": statistics.mean(
                    per_seed[str(outer)][arm]["final_regret"] for outer in seeds
                ),
                "per_seed": [
                    per_seed[str(outer)][arm]["final_regret"] for outer in seeds
                ],
            },
            "deployment": _deployment(arm_rows[arm]),
            "generation": safe_describe(
                [slot["response"] for slot in arm_slots], training_rows
            ),
            "known_usage": _known_usage(arm_slots),
        }
        if arm != "A0":
            data = descriptions[arm]
            arms[arm]["candidate_eligibility"] = {
                "generated_candidates": len(seeds) * slots,
                "eligible_generated": sum(row["eligible_generated"] for row in data),
                "ineligible_generated": sum(
                    row["ineligible_generated"] for row in data
                ),
                "seed_index_selection_count": sum(
                    row["selected_seed_index"] for row in data
                ),
                "seed_source_selection_count": sum(
                    row["selected_seed_source"] for row in data
                ),
                "no_eligible_replacement_count": sum(
                    row["eligible_generated"] == 0 for row in data
                ),
                "seed_selection_despite_eligible_replacement_count": sum(
                    row["selected_seed_index"] and row["eligible_generated"] > 0
                    for row in data
                ),
                "outer_seeds": len(seeds),
            }
    all_slots = list(bundle["slots"].values())
    attempts = [attempt for slot in all_slots for attempt in slot["attempts"]]
    statuses = Counter(attempt["status"] for attempt in attempts)
    _require(
        set(statuses) <= {"completed", "transport_failure"},
        "unknown transport attempt status",
    )
    physical = {
        split: [
            entry["row"]
            for entry in bundle["cache"].values()
            if entry["key"]["split"] == split
        ]
        for split in SPLITS
    }
    cache_events = [
        event for event in bundle["events"] if event["event"] == "evaluation_cache"
    ]
    _require(
        all(event["key"] in bundle["cache"] for event in cache_events),
        "cache event points to missing physical evidence",
    )
    representative_outer = min(
        seeds,
        key=lambda outer: (
            bundle["selections"][f"{outer}/R"]["validation_auc"],
            seeds.index(outer),
        ),
    )
    return {
        "schema": "investigation16.production_analysis.v1",
        "namespace": config["namespace"],
        "outer_seeds": seeds,
        "metric": "normalized anytime regret AUC; lower is better",
        "per_seed": per_seed,
        "arms": arms,
        "contrasts": {
            f"{left}-{right}": E.paired(
                [
                    per_seed[str(outer)][left]["auc"]
                    - per_seed[str(outer)][right]["auc"]
                    for outer in seeds
                ]
            )
            for left, right in CONTRASTS
        },
        "representative": {
            "outer": representative_outer,
            **per_seed[str(representative_outer)]["R"]["selection"],
        },
        "resources": {
            "allocated_proposal_slots": len(expected_slots),
            "completed_responses": len(all_slots),
            "transport_attempts": len(attempts),
            "transport_failures": statuses["transport_failure"],
            "possible_remote_completion_or_duplicate_billing_attempts": sum(
                attempt.get("possible_remote_completion_or_duplicate_billing", False)
                for attempt in attempts
            ),
            "response_usage": safe_describe(
                [slot["response"] for slot in all_slots], []
            )["usage"],
            "known_usage": _known_usage(all_slots),
            "transport_failure_usage": [
                attempt.get("usage")
                for attempt in attempts
                if attempt["status"] == "transport_failure"
            ],
            "finish_reasons": dict(
                Counter(slot["response"].get("finish_reason") for slot in all_slots)
            ),
            "models": dict(Counter(slot["response"]["model"] for slot in all_slots)),
            "provider_receipt_routes": dict(
                Counter(
                    (slot.get("metadata") or {}).get("provider_name", "unknown")
                    for slot in all_slots
                )
            ),
            "trace_event_counts": dict(
                Counter(
                    event["event"]
                    for event in bundle["events"]
                    if event["event"].startswith("trace_update")
                )
            ),
            "attempt_wall_s_reported": sum(
                attempt["wall_s"]
                for attempt in attempts
                if attempt.get("wall_s") is not None
            ),
            "attempts_missing_wall_s": sum(
                attempt.get("wall_s") is None for attempt in attempts
            ),
            "logical": {split: _allocation(rows) for split, rows in logical.items()},
            "physical_cache": {
                split: _allocation(rows) for split, rows in physical.items()
            },
            "cache_accesses": len(cache_events),
            "cache_hits": sum(event["hit"] for event in cache_events),
            "cache_misses": sum(not event["hit"] for event in cache_events),
            "physical_cache_rows": len(bundle["cache"]),
            "cache_rows_without_miss_event": sorted(
                set(bundle["cache"])
                - {event["key"] for event in cache_events if not event["hit"]}
            ),
            "repeated_cache_miss_events": {
                key: count
                for key, count in Counter(
                    event["key"] for event in cache_events if not event["hit"]
                ).items()
                if count > 1
            },
            "normalization_unique_design_calls": sum(
                len(tasks) for tasks in frozen["tasks"].values()
            )
            * frozen["benchmark_manifest"]["normalization_reference_size"],
            "accounting_limits": [
                "Physical cache totals cover completed persisted evaluations; interrupted unpersisted execution may incur unknown additional work.",
                "Reference design calls count unique tasks; physical reconstructions across processes or resumes were not fully instrumented.",
                "Audit cache-event owner I is a technical delegate for all arms; physical cache costs are reported globally and by split, not attributed to that arm.",
                "Missing reported tokens/costs are unknown, not zero; uncertain remote completion may add unreported billing.",
            ],
        },
        "chronology": bundle.get("chronology"),
        "timing": bundle.get("timing", {}),
        "interpretation_limits": [
            "All contrasts are exploratory; outer seeds are the replication units, and small-sample bootstrap intervals are fragile.",
            "Negative contrasts favor the first arm; the central comparison is R-I. Report all four contrasts without outcome-dependent selection.",
            "Primary metrics include common deployment fallback; candidate validity and seed selection are reported separately.",
            "Prefix validation minima are monotone by construction and are not prospective generalization estimates.",
            "This analysis does not establish novelty, additional nested recursion depth, or amortization.",
        ],
    }


def _json_paths(directory: Path, pattern: str = "*.json*") -> list[Path]:
    """Return each logical JSON path once, refusing ambiguous compressed duplicates."""
    physical = [
        path
        for path in directory.glob(pattern)
        if path.name.endswith((".json", ".json.gz"))
    ]
    logical = [
        path.with_suffix("") if path.suffix == ".gz" else path for path in physical
    ]
    _require(
        len(logical) == len(set(logical)), "both plain and compressed evidence exist"
    )
    return sorted(logical)


def read_bundle(root: Path) -> dict[str, Any]:
    """Read a fully completed frozen run and check barriers without evaluator access."""
    from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import search_experiment as S

    frozen = S.preflight(root)
    _require(
        str(Path(__file__).resolve()) in frozen["files"],
        "analysis implementation was not included in pre-execution freeze",
    )
    chronology = S.verify_chronology(root)
    audit = E.read(root / "audit_results.json")
    barrier = E.read(root / "selections_frozen.json")
    _require(
        audit["selections_frozen_ns"] == barrier["completed_ns"],
        "audit selection barrier mismatch",
    )
    result: dict[str, Any] = {
        "freeze": frozen,
        "audit": audit,
        "pools": {},
        "selections": {},
        "slots": {},
        "cache": {},
        "events": [],
        "chronology": chronology,
        "timing": {
            phase: E.read(root / filename).get("clocks")
            for phase, filename in (
                ("generation", "generation_frozen.json"),
                ("selection", "selections_frozen.json"),
                ("audit", "audit_results.json"),
            )
        },
    }
    for outer in frozen["config"]["outer_seeds"]:
        for arm in ARMS[1:]:
            directory = root / "raw" / str(outer) / arm
            name = f"{outer}/{arm}"
            result["pools"][name] = E.read(directory / "pool.json")
            result["selections"][name] = E.read(directory / "selection.json")
            for index in range(frozen["config"]["slots"]):
                slot = directory / f"slot_{index:02d}"
                metadata = slot / "provider_generation.json"
                result["slots"][f"{name}/{index}"] = {
                    "path": str(slot.relative_to(root)),
                    "request": E.read(slot / "request.json"),
                    "response": E.read(slot / "response.json"),
                    "attempts": [
                        E.read(path) for path in _json_paths(slot, "attempt_*.json*")
                    ],
                    "metadata": E.read(metadata) if E.exists(metadata) else None,
                    "generation_timing": (
                        E.read(slot / "generation_timing.json")
                        if E.exists(slot / "generation_timing.json")
                        else None
                    ),
                }
                started = _json_paths(slot, "started_*.json*")
                _require(
                    all(
                        E.exists(
                            path.with_name(path.name.replace("started_", "attempt_"))
                        )
                        for path in started
                    ),
                    "unreconciled request attempt remains",
                )
    expected_response_paths = {
        root / slot["path"] / "response.json" for slot in result["slots"].values()
    }
    _require(
        set(_json_paths(root / "raw", "*/*/slot_*/response.json*"))
        == expected_response_paths,
        "unexpected completed response outside allocated slots",
    )
    result["timing"]["generation_slots"] = {
        name: slot["generation_timing"] for name, slot in result["slots"].items()
    }
    result["cache"] = {path.stem: E.read(path) for path in _json_paths(root / "cache")}
    result["events"] = [E.read(path) for path in _json_paths(root / "events")]
    representative = min(
        frozen["config"]["outer_seeds"],
        key=lambda outer: (
            result["selections"][f"{outer}/R"]["validation_auc"],
            frozen["config"]["outer_seeds"].index(outer),
        ),
    )
    _require(
        barrier["representative_outer"] == representative
        and barrier["representative"] == result["selections"][f"{representative}/R"],
        "representative differs from frozen validation-only rule",
    )
    return result


def main() -> None:
    """Write an immutable aggregate from preserved evidence, never running new tasks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(read_bundle(args.root))
    I.persist(args.output or args.root / "analysis_results.json", result)


if __name__ == "__main__":
    main()
