"""Pure, frozen analysis shared by EXP-17 confirmation and EXP-18 mechanisms."""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import search_experiment as S
from experiments.recursive_opt._shared.optimizer_discovery.investigation16.production import analysis as A


def contrasts(
    per_seed: dict[str, Any], seeds: list[int], kind: str
) -> tuple[dict[str, Any], dict[str, str]]:
    """Apply the registered contrasts to paired outer-seed aggregate outcomes."""
    if kind == "exp17":
        pairs = [
            ("C", "I"),
            ("C", "A0"),
            ("I", "A0"),
            ("C", "B2"),
            ("I", "B2"),
            ("B2", "A0"),
        ]
    elif kind == "exp18":
        pairs = [
            ("M", "L"),
            ("PM", "P"),
            ("P", "L"),
            ("PM", "M"),
            ("PM", "L"),
            *[(arm, "B2") for arm in ("L", "M", "P", "PM")],
            ("B2", "A0"),
        ]
    else:
        raise ValueError("unregistered analysis kind")
    result = {
        f"{left}-{right}": E.paired(
            [
                per_seed[str(seed)][left]["auc"] - per_seed[str(seed)][right]["auc"]
                for seed in seeds
            ]
        )
        for left, right in pairs
    }
    roles = {name: "descriptive_secondary" for name in result}
    if kind == "exp17":
        roles["C-I"] = "confirmatory_primary"
    else:
        formulas = {
            "memory": {"M": 0.5, "L": -0.5, "PM": 0.5, "P": -0.5},
            "pareto": {"P": 0.5, "L": -0.5, "PM": 0.5, "M": -0.5},
            "interaction": {"PM": 1.0, "P": -1.0, "M": -1.0, "L": 1.0},
        }
        for name, weights in formulas.items():
            result[name] = E.paired(
                [
                    sum(
                        weight * per_seed[str(seed)][arm]["auc"]
                        for arm, weight in weights.items()
                    )
                    for seed in seeds
                ]
            )
            roles[name] = "exploratory_factorial"
    return result, roles


def _slot(
    bundle: dict[str, Any],
    outer: int,
    arm: str,
    candidate: dict[str, Any],
    identifiers: set[str],
    request_ids: set[str],
) -> None:
    """Reparse every completed response and verify the exact registered request."""
    frozen = bundle["freeze"]
    index = candidate["index"]
    slot = bundle["slots"][f"{outer}/{arm}/{index}"]
    request, response = slot["request"], slot["response"]
    try:
        S._verify_response(request, response)
    except (RuntimeError, KeyError, TypeError) as error:
        raise ValueError(
            "completed response parsing or source integrity mismatch"
        ) from error
    A._require(
        response["id"] not in identifiers, "duplicate completed response identifier"
    )
    identifiers.add(response["id"])
    A._require(
        isinstance(request.get("slot_id"), str)
        and request["slot_id"]
        and request["slot_id"] not in request_ids,
        "missing or duplicate request slot identity",
    )
    request_ids.add(request["slot_id"])
    A._require(
        (request["outer"], request["arm"], request["slot"]) == (outer, arm, index)
        and request.get("freeze_sha256") == B.digest(frozen),
        "request slot or freeze identity mismatch",
    )
    A._require(
        request["settings"] == S._generation_settings(frozen["config"], outer, index),
        "request generation settings differ from registration",
    )
    A._require(
        request["messages"]
        and request["messages"][0]
        == {"role": "user", "content": S._invariant(frozen["config"])},
        "invariant request information differs",
    )
    A._require(
        response["source"] == candidate["source"]
        and response["source_sha256"] == candidate["source_sha256"],
        "response source differs from evaluated candidate",
    )
    attempts = slot["attempts"]
    successful = [attempt for attempt in attempts if attempt["status"] == "completed"]
    A._require(
        len(successful) == 1
        and successful[0].get("id") == response["id"]
        and attempts[-1]["status"] == "completed"
        and response["attempt"] == len(attempts)
        and all(
            attempt["status"] in {"transport_failure", "completed"}
            for attempt in attempts
        ),
        "completed response requires matching retained transport attempts",
    )


def _pool(
    bundle: dict[str, Any],
    cache: dict[tuple[Any, ...], dict[str, Any]],
    outer: int,
    arm: str,
    logical: dict[str, list[dict[str, Any]]],
    identities: set[tuple[Any, ...]],
    response_ids: set[str],
    request_ids: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the complete candidate pool and its validation-only choice."""
    frozen = bundle["freeze"]
    slots = frozen["config"]["slots"]
    candidates = bundle["pools"][f"{outer}/{arm}"]
    A._require(
        [item["index"] for item in candidates] == list(range(-1, slots)),
        "missing or reordered pool candidate",
    )
    for candidate in candidates:
        digest = candidate["source_sha256"]
        A._require(
            digest == B.source_hash(candidate["source"]),
            "candidate source integrity mismatch",
        )
        for split in ("train", "validation"):
            rows = candidate[split]
            A._panel(bundle, cache, rows, digest, outer, split)
            logical[split].extend(rows)
            identities.update(
                (outer, split, digest, row["task_identity"], row["local_seed"])
                for row in rows
            )
        eligible = all(
            row["valid"]
            for split in ("train", "validation")
            for row in candidate[split]
        )
        expected = B.aggregate(candidate["validation"], "auc") if eligible else None
        A._require(
            candidate["eligible"] == eligible
            and candidate["validation_auc"] == expected,
            "candidate eligibility or validation metric mismatch",
        )
        if candidate["index"] == -1:
            A._require(
                eligible and digest == frozen["seed_sha256"],
                "trusted seed missing or invalid",
            )
        else:
            _slot(bundle, outer, arm, candidate, response_ids, request_ids)
    best = min(
        (item for item in candidates if item["eligible"]),
        key=lambda item: (item["validation_auc"], item["index"]),
    )
    selected = bundle["selections"][f"{outer}/{arm}"]
    A._require(
        all(
            selected[field] == best[field]
            for field in ("index", "source", "source_sha256", "validation_auc")
        ),
        "selection differs from frozen validation rule",
    )
    summary = {
        "eligible_generated": sum(item["eligible"] for item in candidates[1:]),
        "ineligible_generated": sum(not item["eligible"] for item in candidates[1:]),
        "selected_seed_index": selected["index"] == -1,
        "selected_seed_source": selected["source_sha256"] == frozen["seed_sha256"],
        **{
            split: A._allocation([row for item in candidates for row in item[split]])
            for split in ("train", "validation")
        },
        "prefix_validation_auc": [
            min(
                item["validation_auc"]
                for item in candidates[: prefix + 1]
                if item["eligible"]
            )
            for prefix in range(1, slots + 1)
        ],
        "lineage": [
            {
                "index": index,
                "parent_sha256": bundle["slots"][f"{outer}/{arm}/{index}"]["request"][
                    "parent_sha256"
                ],
                "source_sha256": candidates[index + 1]["source_sha256"],
            }
            for index in range(slots)
        ],
    }
    return selected, summary


def _resources(
    bundle: dict[str, Any], logical: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Retain logical allocations, unique physical work, attempts and unknown usage."""
    slots = list(bundle["slots"].values())
    attempts = [attempt for slot in slots for attempt in slot["attempts"]]
    statuses = Counter(attempt["status"] for attempt in attempts)
    events = [
        event for event in bundle["events"] if event["event"] == "evaluation_cache"
    ]
    A._require(
        all(event["key"] in bundle["cache"] for event in events),
        "cache event points to missing evidence",
    )
    misses = Counter(event["key"] for event in events if not event["hit"])
    physical = {
        split: [
            entry["row"]
            for entry in bundle["cache"].values()
            if entry["key"]["split"] == split
        ]
        for split in A.SPLITS
    }
    frozen = bundle["freeze"]
    return {
        "allocated_proposal_slots": len(frozen["config"]["outer_seeds"])
        * len(frozen["config"]["arms"])
        * frozen["config"]["slots"],
        "completed_responses": len(slots),
        "transport_attempts": len(attempts),
        "transport_failures": statuses["transport_failure"],
        "possible_remote_completion_or_duplicate_billing_attempts": sum(
            attempt.get("possible_remote_completion_or_duplicate_billing", False)
            for attempt in attempts
        ),
        "response_usage": A.safe_describe([slot["response"] for slot in slots], [])[
            "usage"
        ],
        "known_usage": A._known_usage(slots),
        "transport_failure_usage": [
            attempt.get("usage")
            for attempt in attempts
            if attempt["status"] == "transport_failure"
        ],
        "finish_reasons": dict(
            Counter(slot["response"].get("finish_reason") for slot in slots)
        ),
        "models": dict(Counter(slot["response"]["model"] for slot in slots)),
        "provider_receipt_routes": dict(
            Counter(
                (slot.get("metadata") or {}).get("provider_name", "unknown")
                for slot in slots
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
        "logical": {split: A._allocation(rows) for split, rows in logical.items()},
        "physical_cache": {
            split: A._allocation(rows) for split, rows in physical.items()
        },
        "cache_accesses": len(events),
        "cache_hits": sum(event["hit"] for event in events),
        "cache_misses": sum(misses.values()),
        "physical_cache_rows": len(bundle["cache"]),
        "cache_rows_without_miss_event": sorted(set(bundle["cache"]) - set(misses)),
        "repeated_cache_miss_events": {
            key: value for key, value in misses.items() if value > 1
        },
        "normalization_unique_design_calls": sum(
            len(tasks) for tasks in frozen["tasks"].values()
        )
        * frozen["benchmark_manifest"]["normalization_reference_size"],
        "accounting_limits": [
            "Cache totals cover completed persisted evaluations; interrupted unpersisted work may add unknown executions.",
            "Unique normalization values do not count repeated physical reference reconstruction unless separately instrumented.",
            "Shared-cache physical work is attributed globally and by split; event-owner labels are not arm resource attribution.",
            "Missing usage is unknown, not zero; transport failures may incur unreported remote billing.",
        ],
    }


def summarize(bundle: dict[str, Any]) -> dict[str, Any]:
    """Validate all raw evidence and aggregate without new scientific execution."""
    frozen, audit = bundle["freeze"], bundle["audit"]
    config = frozen["config"]
    kind, seeds, arms, slots = (
        config["analysis_kind"],
        config["outer_seeds"],
        config["arms"],
        config["slots"],
    )
    expected_arms = {"exp17": ["I", "C"], "exp18": ["L", "M", "P", "PM"]}
    A._require(
        kind in expected_arms and arms == expected_arms[kind],
        "invalid registered analysis arms",
    )
    A._require(
        seeds
        and len(seeds) == len(set(seeds))
        and all(type(seed) is int for seed in seeds)
        and type(slots) is int
        and slots > 0,
        "invalid registered seeds or slots",
    )
    A._require(
        config["bootstrap"] == B.MANIFEST["bootstrap"],
        "analysis bootstrap differs from frozen implementation",
    )
    A._require(config["model"] == S.G.MODEL, "unregistered generation model")
    A._require(
        frozen["seed_sha256"] == B.source_hash(frozen["seed_source"]),
        "seed source integrity mismatch",
    )
    controls = {
        "A0": {"source": frozen["seed_source"], "source_sha256": frozen["seed_sha256"]},
        **frozen["fixed_controls"],
    }
    A._require(
        set(controls) == {"A0", "B2"}
        and all(
            value["source_sha256"] == B.source_hash(value["source"])
            for value in controls.values()
        ),
        "fixed control source integrity mismatch",
    )
    deployment_arms = ["A0", "B2", *arms]
    A._require(
        set(audit["per_seed"]) == {str(seed) for seed in seeds},
        "missing or extra outer seed",
    )
    pool_names = {f"{outer}/{arm}" for outer in seeds for arm in arms}
    A._require(
        set(bundle["pools"]) == pool_names and set(bundle["selections"]) == pool_names,
        "missing or extra arm pool/selection",
    )
    expected_slots = {
        f"{outer}/{arm}/{slot}"
        for outer in seeds
        for arm in arms
        for slot in range(slots)
    }
    A._require(
        set(bundle["slots"]) == expected_slots,
        "missing or extra completed proposal slot",
    )
    cache = A._cache_index(bundle)
    logical: dict[str, list[dict[str, Any]]] = {split: [] for split in A.SPLITS}
    identities: set[tuple[Any, ...]] = set()
    response_ids: set[str] = set()
    request_ids: set[str] = set()
    per_seed: dict[str, Any] = {}
    rows_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in deployment_arms}
    for outer in seeds:
        actual = audit["per_seed"][str(outer)]
        A._require(set(actual) == set(deployment_arms), "missing or extra audit arm")
        per_seed[str(outer)] = {}
        for arm in deployment_arms:
            if arm in controls:
                selected = {**controls[arm], "index": -1, "validation_auc": None}
                search = None
                source_record = (
                    "freeze.json:seed_source"
                    if arm == "A0"
                    else "freeze.json:fixed_controls.B2.source"
                )
            else:
                selected, search = _pool(
                    bundle,
                    cache,
                    outer,
                    arm,
                    logical,
                    identities,
                    response_ids,
                    request_ids,
                )
                source_record = f"raw/{outer}/{arm}/selection.json:source"
            value = actual[arm]
            A._require(
                value["source_sha256"] == selected["source_sha256"],
                "audit source differs from frozen selection",
            )
            rows = value["rows"]
            A._panel(bundle, cache, rows, selected["source_sha256"], outer, "audit")
            A._require(
                all(
                    A._same_number(value[metric], B.aggregate(rows, metric))
                    for metric in ("auc", "final_regret")
                ),
                "audit aggregate mismatch",
            )
            A._require(
                value["fallback_trajectories"]
                == sum(row["fallback_used"] for row in rows),
                "audit fallback count mismatch",
            )
            logical["audit"].extend(rows)
            rows_by_arm[arm].extend(rows)
            identities.update(
                (
                    outer,
                    "audit",
                    selected["source_sha256"],
                    row["task_identity"],
                    row["local_seed"],
                )
                for row in rows
            )
            per_seed[str(outer)][arm] = {
                "auc": value["auc"],
                "final_regret": value["final_regret"],
                "selection": {
                    key: selected[key]
                    for key in ("index", "source_sha256", "validation_auc")
                },
                "selected_source_record": source_record,
                "deployment": A._deployment(rows),
                "search": search,
            }
    A._require(
        set(cache) == identities, "unallocated or missing physical cache trajectory"
    )
    arm_results = {}
    for arm in deployment_arms:
        arm_slots = (
            [
                bundle["slots"][f"{outer}/{arm}/{index}"]
                for outer in seeds
                for index in range(slots)
            ]
            if arm in arms
            else []
        )
        generated_rows = (
            [
                row
                for outer in seeds
                for candidate in bundle["pools"][f"{outer}/{arm}"][1:]
                for split in ("train", "validation")
                for row in candidate[split]
            ]
            if arm in arms
            else []
        )
        aucs = [per_seed[str(outer)][arm]["auc"] for outer in seeds]
        finals = [per_seed[str(outer)][arm]["final_regret"] for outer in seeds]
        value = {
            "auc": {
                "per_seed": aucs,
                "mean": statistics.mean(aucs),
                "median": statistics.median(aucs),
            },
            "final_regret": {"per_seed": finals, "mean": statistics.mean(finals)},
            "deployment": A._deployment(rows_by_arm[arm]),
            "generation": A.safe_describe(
                [slot["response"] for slot in arm_slots], generated_rows
            ),
            "known_usage": A._known_usage(arm_slots),
        }
        if arm in arms:
            searches = [per_seed[str(outer)][arm]["search"] for outer in seeds]
            value["candidate_eligibility"] = {
                "generated_candidates": len(seeds) * slots,
                "eligible_generated": sum(
                    row["eligible_generated"] for row in searches
                ),
                "ineligible_generated": sum(
                    row["ineligible_generated"] for row in searches
                ),
                "seed_index_selection_count": sum(
                    row["selected_seed_index"] for row in searches
                ),
                "seed_source_selection_count": sum(
                    row["selected_seed_source"] for row in searches
                ),
                "no_eligible_replacement_count": sum(
                    row["eligible_generated"] == 0 for row in searches
                ),
                "seed_selection_despite_eligible_replacement_count": sum(
                    row["selected_seed_index"] and row["eligible_generated"] > 0
                    for row in searches
                ),
                "outer_seeds": len(seeds),
            }
        arm_results[arm] = value
    results, roles = contrasts(per_seed, seeds, kind)
    representative_arm = config["representative_arm"]
    A._require(representative_arm in arms, "unregistered representative arm")
    representative = min(
        seeds,
        key=lambda outer: (
            bundle["selections"][f"{outer}/{representative_arm}"]["validation_auc"],
            seeds.index(outer),
        ),
    )
    return {
        "schema": "optimizer_discovery.shared_analysis.v1",
        "namespace": config["namespace"],
        "analysis_kind": kind,
        "outer_seeds": seeds,
        "metric": "normalized anytime regret AUC; lower is better",
        "per_seed": per_seed,
        "arms": arm_results,
        "contrasts": results,
        "contrast_roles": roles,
        "representative": {
            "arm": representative_arm,
            "outer": representative,
            **per_seed[str(representative)][representative_arm]["selection"],
        },
        "resources": _resources(bundle, logical),
        "chronology": bundle.get("chronology"),
        "timing": bundle.get("timing", {}),
        "interpretation_limits": [
            "Outer seeds are replication units conditional on the frozen task panel; task/time points are not extra outer replications.",
            "EXP-17 C-I is the only confirmatory primary; EXP-18 factorial intervals and other contrasts are exploratory/descriptive without a simultaneous-coverage claim.",
            "Negative pairwise contrasts favor the first arm; factorial interaction has its registered algebraic meaning, not an arm-superiority claim.",
            "Primary metrics include common seed fallback; candidate validity, failed searches and seed retention remain separate.",
            "The 0.02 planning effect is not a projected gain or a threshold whose exceedance is implied by an interval below zero.",
            "No novelty, recursion-depth, amortization, external-engine or operating-system sandbox claim is established.",
        ],
    }


def _attempt_paths(slot: Path) -> list[Path]:
    """Preserve numeric attempt order across bounded retry batches and resumes."""
    paths = A._json_paths(slot, "attempt_*.json*")
    try:
        indexed = sorted(
            (int(path.stem.removeprefix("attempt_")), path) for path in paths
        )
    except ValueError as error:
        raise ValueError("invalid transport attempt filename") from error
    A._require(
        [index for index, _ in indexed] == list(range(1, len(indexed) + 1)),
        "missing or duplicate transport attempt index",
    )
    return [path for _, path in indexed]


def read_bundle(root: Path) -> dict[str, Any]:
    """Read complete frozen evidence after the owner's global chronology guards."""
    from experiments.recursive_opt._shared.optimizer_discovery.exp17 import study

    frozen = study.preflight(root)
    A._require(
        str(Path(__file__).resolve()) in frozen["files"],
        "analysis implementation absent from pre-execution freeze",
    )
    chronology = study.verify_chronology(root)
    audit = E.read(root / "audit_results.json")
    barrier = E.read(root / "selections_frozen.json")
    A._require(
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
        for arm in frozen["config"]["arms"]:
            name = f"{outer}/{arm}"
            directory = root / "raw" / name
            result["pools"][name] = E.read(directory / "pool.json")
            result["selections"][name] = E.read(directory / "selection.json")
            for index in range(frozen["config"]["slots"]):
                slot = directory / f"slot_{index:02d}"
                metadata = slot / "provider_generation.json"
                result["slots"][f"{name}/{index}"] = {
                    "path": str(slot.relative_to(root)),
                    "request": E.read(slot / "request.json"),
                    "response": E.read(slot / "response.json"),
                    "attempts": [E.read(path) for path in _attempt_paths(slot)],
                    "metadata": E.read(metadata) if E.exists(metadata) else None,
                    "generation_timing": (
                        E.read(slot / "generation_timing.json")
                        if E.exists(slot / "generation_timing.json")
                        else None
                    ),
                }
                A._require(
                    all(
                        E.exists(
                            path.with_name(path.name.replace("started_", "attempt_"))
                        )
                        for path in A._json_paths(slot, "started_*.json*")
                    ),
                    "unreconciled request attempt remains",
                )
    expected = {
        root / value["path"] / "response.json" for value in result["slots"].values()
    }
    A._require(
        set(A._json_paths(root / "raw", "*/*/slot_*/response.json*")) == expected,
        "unexpected completed response outside allocated slots",
    )
    result["timing"]["generation_slots"] = {
        name: slot["generation_timing"] for name, slot in result["slots"].items()
    }
    result["cache"] = {
        path.stem: E.read(path) for path in A._json_paths(root / "cache")
    }
    result["events"] = [E.read(path) for path in A._json_paths(root / "events")]
    seeds = frozen["config"]["outer_seeds"]
    arm = frozen["config"]["representative_arm"]
    representative = min(
        seeds,
        key=lambda outer: (
            result["selections"][f"{outer}/{arm}"]["validation_auc"],
            seeds.index(outer),
        ),
    )
    A._require(
        barrier["representative_outer"] == representative
        and barrier["representative"]
        == result["selections"][f"{representative}/{arm}"],
        "representative differs from validation-only selection",
    )
    return result


def main() -> None:
    """Persist an immutable summary from completed preserved evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    I.persist(
        args.output or args.root / "analysis_results.json",
        summarize(read_bundle(args.root)),
    )


if __name__ == "__main__":
    main()
