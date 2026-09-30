"""Post-audit numeric verification with separately accounted reference/objective calls.

This helper never executes candidate code or a model. It checks immutable completed
evidence; any mismatch is reported, never repaired or substituted into the experiment.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from experiments.recursive_opt._shared.optimizer_discovery.investigation16.production import analysis as A

SPLITS = ("train", "validation", "audit")
REQUIRED_ARTIFACTS = (
    "generation_frozen.json",
    "selections_frozen.json",
    "audit_results.json",
)


def verify(root: Path) -> dict[str, Any]:
    """Rebuild numerical evidence only after the complete frozen audit and structural checks."""
    if not all(E.exists(root / name) for name in REQUIRED_ARTIFACTS):
        raise RuntimeError(
            "numeric verification requires a completed generation, selection and audit"
        )
    # read_bundle invokes S.preflight and the frozen chronology/barrier verifier.
    # summarize checks source/pool/cache identity and complete registered allocation.
    # Both complete before any objective/reference call or in-memory cache reset.
    bundle = A.read_bundle(root)
    A.summarize(bundle)
    frozen = bundle["freeze"]
    budget = frozen["config"]["budget"]
    tasks = {
        (split, B.task_identity(task)): task
        for split in SPLITS
        for task in frozen["tasks"][split]
    }
    if len(tasks) != sum(len(panel) for panel in frozen["tasks"].values()):
        raise RuntimeError("duplicate task in completed frozen numerical evidence")
    started_ns = time.time_ns()
    failures: list[dict[str, Any]] = []
    scales: dict[tuple[str, str], float] = {}
    reference_size = frozen["benchmark_manifest"]["normalization_reference_size"]
    # This is solely a post-audit verifier's in-memory cache. Clearing it forces
    # an independent rebuild rather than reusing scientific-run normalization.
    B._reference_scale.cache_clear()
    before = B._reference_scale.cache_info()
    for identity, task in sorted(tasks.items()):
        try:
            scale = B.normalization(task)
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("nonpositive reference normalization")
            scales[identity] = scale
        except (ValueError, TypeError, ArithmeticError) as error:
            failures.append(
                {
                    "kind": "reference_reconstruction",
                    "task_identity": identity[1],
                    "error_type": type(error).__name__,
                }
            )
    after = B._reference_scale.cache_info()
    reference_misses = after.misses - before.misses
    observation_calls = 0
    verified = invalid_partial = fallback = 0
    rows: list[dict[str, Any]] = []
    for digest, entry in sorted(bundle["cache"].items()):
        key, row = entry["key"], entry["row"]
        initial_failures = len(failures)
        identity = (key["split"], key["task_identity"])
        if (
            B.digest(key) != digest
            or B.digest(row) != entry["row_sha256"]
            or identity not in tasks
            or any(
                row[field] != key[field]
                for field in ("source_sha256", "task_identity", "local_seed", "budget")
            )
            or row["budget"] != budget
            or type(row["valid"]) is not bool
        ):
            failures.append({"kind": "cache_identity", "cache_key": digest})
            rows.append({"cache_key": digest, "verified": False})
            continue
        observations = row["observations"]
        values = []
        task = tasks[identity]
        for index, observation in enumerate(observations):
            point = observation["x"]
            if (
                not isinstance(point, list)
                or len(point) != task["dimension"]
                or any(
                    type(x) not in (int, float)
                    or not math.isfinite(x)
                    or not -5 <= x <= 5
                    for x in point
                )
            ):
                failures.append(
                    {
                        "kind": "observation_bounds",
                        "cache_key": digest,
                        "evaluation": index + 1,
                    }
                )
                continue
            value = B.objective(task, point)
            observation_calls += 1
            values.append(value)
            if (
                type(observation["value"]) not in (int, float)
                or not math.isfinite(observation["value"])
                or observation["value"] != value
            ):
                failures.append(
                    {
                        "kind": "objective_value",
                        "cache_key": digest,
                        "evaluation": index + 1,
                    }
                )
        if (
            len(observations) > budget
            or row["objective_calls"] != len(observations)
            or row["unused_objective_allocation"] != budget - len(observations)
        ):
            failures.append({"kind": "objective_accounting", "cache_key": digest})
        reconstructed = None
        if row["valid"]:
            if (
                len(observations) != budget
                or len(values) != budget
                or identity not in scales
            ):
                failures.append(
                    {"kind": "incomplete_valid_trajectory", "cache_key": digest}
                )
            else:
                reconstructed = B.metrics(values, scales[identity], budget)
                if row["metrics"] != reconstructed:
                    failures.append({"kind": "normalized_metrics", "cache_key": digest})
        elif row["metrics"] is not None:
            failures.append(
                {"kind": "invalid_trajectory_numeric_score", "cache_key": digest}
            )
        elif len(observations) >= budget:
            failures.append(
                {"kind": "invalid_complete_trajectory", "cache_key": digest}
            )
        if key["split"] != "audit" and row["fallback_used"]:
            failures.append({"kind": "fallback_outside_audit", "cache_key": digest})
        good = len(failures) == initial_failures
        verified += good
        invalid_partial += good and not row["valid"]
        fallback += good and row["fallback_used"]
        rows.append(
            {
                "cache_key": digest,
                "verified": good,
                "valid": row["valid"],
                "observations_recomputed": len(values),
                "fallback_used": row["fallback_used"],
                "recomputed_metrics": reconstructed,
            }
        )
    return {
        "schema": "investigation16.post_audit_numeric_verification.v1",
        "status": "FAIL" if failures else "PASS",
        "started_ns": started_ns,
        "completed_ns": time.time_ns(),
        "frozen_manifest_sha256": B.digest(frozen),
        "helper_sha256": B.source_hash(Path(__file__).read_text()),
        "input_cache_rows_sha256": B.digest(
            {key: entry["row_sha256"] for key, entry in bundle["cache"].items()}
        ),
        "cache_rows_seen": len(bundle["cache"]),
        "cache_rows_verified": verified,
        "invalid_partial_rows_verified": invalid_partial,
        "fallback_rows_verified": fallback,
        "reference_cache_cleared_after_completed_audit_guard": True,
        "reference_cache_misses": reference_misses,
        "reference_cache_hits": after.hits - before.hits,
        "integrity_objective_calls": {
            "observations": observation_calls,
            "reference_design": reference_misses * reference_size,
            "total": observation_calls + reference_misses * reference_size,
        },
        "accounting_scope": "Additional post-audit integrity work; excluded from scientific search/evaluation budgets.",
        "candidate_subprocesses": 0,
        "model_calls": 0,
        "normalization_constants": [
            {"split": split, "task_identity": identity, "scale": scale}
            for (split, identity), scale in sorted(scales.items())
        ],
        "rows": rows,
        "failures": failures,
        "comparison": "Exact deterministic objective and B.metrics recomputation under the frozen implementation/environment; no values repaired.",
    }


def main() -> None:
    """Persist a distinct post-audit verification attempt and exit nonzero on mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = verify(args.root)
    directory = args.root / "numeric_verification"
    index = 1
    while E.exists(directory / f"attempt_{index:03d}.json"):
        index += 1
    path = directory / f"attempt_{index:03d}.json"
    I.persist(path, report)
    print(f"{report['status']}: {path}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
