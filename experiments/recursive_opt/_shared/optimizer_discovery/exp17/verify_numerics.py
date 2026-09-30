"""Read-only post-audit numerical verification for EXP-17 and EXP-18.

No candidate is executed, no provider is contacted and no credential source is
loaded. Objective and reference recomputation is additional integrity work, not
scientific search allocation. Mismatches are preserved, never repaired.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.exp17 import analysis as A
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I

SPLITS = ("train", "validation", "audit")
REQUIRED_ARTIFACTS = (
    "generation_frozen.json",
    "selections_frozen.json",
    "audit_results.json",
)


def _file_hash(path: Path) -> str:
    """Stream exact stored bytes, including compressed evidence, into SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory(root: Path) -> dict[str, str]:
    """Hash every run file except prior integrity reports and Python bytecode."""
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (
            relative.parts[0] == "numeric_verification"
            or "__pycache__" in relative.parts
        ):
            continue
        if path.is_symlink():
            raise RuntimeError("numerical verification requires direct evidence files")
        if path.is_file():
            result[str(relative)] = _file_hash(path)
    return result


def _source_inventory(frozen: dict[str, Any], *, authenticate: bool) -> dict[str, str]:
    """Bind before/after source bytes to the same files checked by study preflight."""
    result = {}
    for name, expected in sorted(frozen["files"].items()):
        path = Path(name)
        if authenticate and B.source_hash(path.read_text()) != expected:
            raise RuntimeError("frozen source changed before numerical reconstruction")
        result[name] = _file_hash(path) if path.is_file() else "missing"
    return result


def _changed(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Report changed, added and deleted identities without copying file contents."""
    return sorted(
        name
        for name in before.keys() | after.keys()
        if before.get(name) != after.get(name)
    )


def _numerics(bundle: dict[str, Any]) -> dict[str, Any]:
    """Rebuild every reference design and every preserved cache observation exactly."""
    frozen = bundle["freeze"]
    budget = frozen["config"]["budget"]
    tasks = {
        (split, B.task_identity(task)): task
        for split in SPLITS
        for task in frozen["tasks"][split]
    }
    if len(tasks) != sum(len(panel) for panel in frozen["tasks"].values()):
        raise RuntimeError("duplicate task in frozen numerical evidence")
    if len({identity for _, identity in tasks}) != len(tasks):
        raise RuntimeError("task identity overlaps protected splits")
    failures: list[dict[str, Any]] = []
    scales: dict[tuple[str, str], float] = {}
    reference_size = frozen["benchmark_manifest"]["normalization_reference_size"]
    # Clear only the verifier process's in-memory memo after all completion and
    # structural guards. Scientific files and cached trajectories stay untouched.
    B._reference_scale.cache_clear()
    before = B._reference_scale.cache_info()
    for identity, task in sorted(tasks.items()):
        try:
            scale = B.normalization(task)
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("nonpositive normalization")
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
    failed_references = len(tasks) - len(scales)
    reference_calls = None if failed_references else reference_misses * reference_size
    observation_calls = verified = invalid_partial = fallback = 0
    controls = {
        "A0": frozen["seed_sha256"],
        **{
            name: value["source_sha256"]
            for name, value in frozen["fixed_controls"].items()
        },
    }
    control_rows = dict.fromkeys(controls, 0)
    rows: list[dict[str, Any]] = []
    for digest, entry in sorted(bundle["cache"].items()):
        key, row = entry["key"], entry["row"]
        initial = len(failures)
        identity = (key["split"], key["task_identity"])
        if (
            B.digest(key) != digest
            or B.digest(row) != entry["row_sha256"]
            or identity not in tasks
            or any(
                row.get(field) != key[field]
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
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or not -5 <= value <= 5
                    for value in point
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
            observation_calls += 1
            try:
                value = B.objective(task, point)
            except (ValueError, TypeError, ArithmeticError) as error:
                failures.append(
                    {
                        "kind": "objective_reconstruction",
                        "cache_key": digest,
                        "evaluation": index + 1,
                        "error_type": type(error).__name__,
                    }
                )
                continue
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
                try:
                    reconstructed = B.metrics(values, scales[identity], budget)
                    if row["metrics"] != reconstructed:
                        failures.append(
                            {"kind": "normalized_metrics", "cache_key": digest}
                        )
                except (ValueError, TypeError, ArithmeticError) as error:
                    failures.append(
                        {
                            "kind": "metric_reconstruction",
                            "cache_key": digest,
                            "error_type": type(error).__name__,
                        }
                    )
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
        good = len(failures) == initial
        verified += good
        invalid_partial += good and not row["valid"]
        fallback += good and row["fallback_used"]
        for name, source_hash in controls.items():
            control_rows[name] += (
                good and key["split"] == "audit" and row["source_sha256"] == source_hash
            )
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
        "cache_rows_seen": len(bundle["cache"]),
        "cache_rows_verified": verified,
        "invalid_partial_rows_verified": invalid_partial,
        "fallback_rows_verified": fallback,
        "fixed_control_cache_rows_verified": control_rows,
        "reference_cache_cleared_after_completed_audit_guard": True,
        "reference_cache_misses": reference_misses,
        "reference_cache_hits": after.hits - before.hits,
        "failed_reference_designs": failed_references,
        "reference_design_calls_upper_bound": reference_misses * reference_size,
        "integrity_objective_calls": {
            "observations": observation_calls,
            "reference_design": reference_calls,
            "total": (
                observation_calls + reference_calls
                if reference_calls is not None
                else None
            ),
        },
        "reference_accounting": (
            "Exact completed reference designs."
            if not failed_references
            else "Failed reference designs may stop early; allocated calls are an upper bound, not observed usage."
        ),
        "normalization_constants": [
            {"split": split, "task_identity": identity, "scale": scale}
            for (split, identity), scale in sorted(scales.items())
        ],
        "rows": rows,
        "failures": failures,
    }


def verify(root: Path) -> dict[str, Any]:
    """Require complete study guards, authenticate all inputs, then verify numerics."""
    if not all(E.exists(root / name) for name in REQUIRED_ARTIFACTS):
        raise RuntimeError(
            "numeric verification requires completed generation, selection and audit"
        )
    started_ns = time.time_ns()
    before_files = _inventory(root)
    # The successor reader lazily imports study.preflight/verify_chronology; the
    # complete analysis checks all registered arms, slots, controls and cache rows.
    bundle = A.read_bundle(root)
    A.summarize(bundle)
    frozen = bundle["freeze"]
    before_sources = _source_inventory(frozen, authenticate=True)
    report = _numerics(bundle)
    after_files = _inventory(root)
    after_sources = _source_inventory(frozen, authenticate=False)
    changed_files = _changed(before_files, after_files)
    changed_sources = _changed(before_sources, after_sources)
    if changed_files:
        report["failures"].append(
            {"kind": "input_files_changed", "paths": changed_files}
        )
    if changed_sources:
        report["failures"].append(
            {"kind": "frozen_source_files_changed", "paths": changed_sources}
        )
    return {
        "schema": "optimizer_successor.post_audit_numeric_verification.v1",
        "status": "FAIL" if report["failures"] else "PASS",
        "started_ns": started_ns,
        "completed_ns": time.time_ns(),
        "frozen_manifest_sha256": B.digest(frozen),
        "helper_sha256": _file_hash(Path(__file__)),
        "input_cache_rows_sha256": B.digest(
            {key: entry["row_sha256"] for key, entry in bundle["cache"].items()}
        ),
        "input_files_sha256": before_files,
        "input_files_seen": len(before_files),
        "input_files_after_sha256": B.digest(after_files),
        "input_files_unchanged": not changed_files,
        "frozen_source_files_sha256": before_sources,
        "frozen_source_files_after_sha256": B.digest(after_sources),
        "frozen_source_files_unchanged": not changed_sources,
        "inventory_exclusions": [
            "numeric_verification reports produced by this helper",
            "Python __pycache__ bytecode",
        ],
        "accounting_scope": "Additional post-audit integrity work; excluded from scientific search/evaluation budgets.",
        "candidate_subprocesses": 0,
        "model_calls": 0,
        "comparison": "Exact deterministic objective and B.metrics recomputation under the frozen implementation/environment; no values repaired.",
        **report,
    }


def verify_and_persist(root: Path) -> tuple[Path, dict[str, Any]]:
    """Save a fresh numbered verification attempt without replacing prior evidence."""
    report = verify(root)
    directory = root / "numeric_verification"
    index = 1
    while E.exists(directory / f"attempt_{index:03d}.json"):
        index += 1
    path = directory / f"attempt_{index:03d}.json"
    I.persist(path, report)
    return path, report


def main() -> None:
    """Persist a distinct integrity attempt and exit nonzero on any mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    path, report = verify_and_persist(args.root)
    print(f"{report['status']}: {path}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
