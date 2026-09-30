"""Read-only numerical and inventory audit; never execute candidates or call a model."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import statistics as st
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B

ROOT = Path(__file__).resolve().parents[4]
HASHES: dict[str, str] = {}


def read(path: Path) -> Any:
    """Read JSON, retaining the exact stored-file hash for provenance."""
    raw = path.read_bytes()
    HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def close(actual: float, expected: float) -> None:
    """Reject a numerical mismatch, allowing only floating summation roundoff."""
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14):
        raise AssertionError(f"numerical mismatch: {actual} versus {expected}")


def verify_rows(rows: list[dict[str, Any]], tasks: dict[str, Any]) -> int:
    """Recompute objective values and every trajectory metric from stored points."""
    count = 0
    for row in rows:
        assert row["valid"] and row["candidate_valid"] and not row["fallback_used"]
        task = tasks[row["task_identity"]]
        values = []
        for obs in row["observations"]:
            value = B.objective(task, obs["x"])
            close(value, obs["value"])
            values.append(value)
        assert len(values) == row["budget"] == row["objective_calls"] == 32
        rebuilt = B.metrics(values, B.normalization(task), row["budget"])
        assert rebuilt == row["metrics"]
        count += len(values)
    return count


def numerical_audits() -> dict[str, Any]:
    """Verify EXP15/16/18 complete holdouts without reading EXP17 outcomes."""
    output = {}
    folder = ROOT / "experiments/recursive_opt/_shared/optimizer_discovery"
    result = read(folder / "exp15_results.json")
    tasks = {B.task_identity(t): t for t in B.make_tasks("confirmation", "holdout")}
    total = 0
    arms: dict[str, list[float]] = defaultdict(list)
    for seed in result["per_seed"]:
        raw = read(folder / f"exp15/raw/{seed['outer_seed']}/holdout.json.gz")
        for arm in result["arms"]:
            total += verify_rows(raw[arm], tasks)
            value = B.aggregate(raw[arm], "auc")
            close(value, seed[arm]["auc"])
            arms[arm].append(value)
    for arm, values in arms.items():
        close(st.mean(values), result["arms"][arm]["mean_auc"])
    output["EXP15"] = {
        "objective_values": total,
        "trajectories": total // 32,
        "mean_auc": {a: st.mean(v) for a, v in arms.items()},
    }
    for name, rel in (
        ("EXP16", "investigation16/production_run"),
        ("EXP18", "exp18/run"),
    ):
        base = folder / rel
        frozen = read(base / "freeze.json")
        tasks = {B.task_identity(t): t for t in frozen["tasks"]["audit"]}
        raw = read(base / "audit_results.json.gz")
        analysis = read(base / "analysis_results.json.gz")
        total = 0
        arms = defaultdict(list)
        for group in raw["per_seed"].values():
            for arm, entry in group.items():
                total += verify_rows(entry["rows"], tasks)
                value = B.aggregate(entry["rows"], "auc")
                close(value, entry["auc"])
                arms[arm].append(value)
        for arm, values in arms.items():
            close(st.mean(values), analysis["arms"][arm]["auc"]["mean"])
        output[name] = {
            "objective_values": total,
            "trajectories": total // 32,
            "mean_auc": {a: st.mean(v) for a, v in arms.items()},
        }
    return output


def inventories() -> dict[str, Any]:
    """Count physical evidence files and compare both worktrees' relevant sources."""
    output: dict[str, Any] = {}
    for rel in ("control_plane_v2", "probe_2026", "optimizer_discovery", "o1_learning"):
        files = size = 0
        for parent, dirs, names in os.walk(ROOT / "artifacts" / rel):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in names:
                files += 1
                size += (Path(parent) / name).stat().st_size
        output[rel] = {"files_excluding_pycache": files, "bytes": size}
    left = ROOT.parent / "Trace-experiment0"
    comparison = {}
    for rel in (
        "opto/features/recursive_opt",
        "opto/trace",
        "opto/trainer",
        "opto/optimizers",
        "experiments/recursive_opt/multiobjective_reasoning",
        "experiments/recursive_opt/_shared/control_plane_v2",
    ):
        maps = []
        for root in (left, ROOT):
            maps.append(
                {
                    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (root / rel).rglob("*")
                    if p.is_file()
                    and "__pycache__" not in p.parts
                    and p.suffix in {".py", ".md", ".json"}
                }
            )
        a, b = maps
        comparison[rel] = {
            "identical": sum(a[k] == b[k] for k in a.keys() & b.keys()),
            "different": sorted(k for k in a.keys() & b.keys() if a[k] != b[k]),
            "only_Trace": sorted(b.keys() - a.keys()),
            "only_experiment0": sorted(a.keys() - b.keys()),
        }
    output["worktree_comparison"] = comparison
    return output


def other_evidence() -> dict[str, Any]:
    """Reconcile partial statuses, historical probes and complete Experiment-0 rows."""
    folder = ROOT / "artifacts"
    f = read(folder / "probe_2026/probe_f_results.json")
    indexed = {(r["arm"], r["seed"]): r["score"] for r in f["rows"]}
    deltas = [
        indexed["standard", s] - indexed["initial", s]
        for s in f["seeds"]
        if indexed["standard", s] is not None and indexed["initial", s] is not None
    ]
    close(st.mean(deltas), f["paired_mean"])
    aa = read(folder / "probe_2026/probe_aa_results.json")
    assert len(aa["order"]) == 12 and all(r["score"] is None for r in aa["order"])
    e17 = folder / "optimizer_discovery/exp17/run"
    responses = list((e17 / "raw").rglob("response.json"))
    pause = read(folder / "optimizer_discovery/exp17/USER_REQUESTED_PAUSE.json")
    assert len(responses) == pause["completed_responses"]
    assert not any(
        (e17 / n).exists()
        for n in (
            "generation_frozen.json",
            "selections_frozen.json",
            "audit_results.json",
            "audit_results.json.gz",
        )
    )
    e0 = (
        ROOT
        / "outputs/recursive_opt/experiment_0/experiment-0-v2/main_after_transport_resilience_fix"
    )
    main = read(e0 / "main.json")
    assert len(main["runs"]) == main["completed_run_count"] == 40
    e0_summary = {}
    for arm in "ABCD":
        rows = [r for r in main["runs"] if r["arm"] == arm]
        assert len(rows) == 10
        e0_summary[arm] = {
            k: st.mean(r["metrics"][k] for r in rows)
            for k in ("accuracy", "forward_token_ratio", "invalid_rate")
        }
    shared = {}
    other = ROOT.parent / "Trace-experiment0" / e0.relative_to(ROOT)
    for name in (
        "main.json",
        "analysis.json",
        "decision.json",
        "episode_trajectory_audit.json",
    ):
        shared[name] = (other / name).read_bytes() == (e0 / name).read_bytes()
    s4 = read(folder / "o1_learning/s4_results.json")
    return {
        "probe_f": {
            "paired_n": len(deltas),
            "paired_deltas": deltas,
            "mean": st.mean(deltas),
        },
        "EXP13": {
            "attempts": len(aa["order"]),
            "valid_scores": 0,
            "found_prompt_chars": len(aa["found_prompt"]),
        },
        "EXP17": {"response_files": len(responses), "audit_exists": False},
        "EXP0": {"means": e0_summary, "shared_primary_files_byte_identical": shared},
        "EXP19_S4": {a: st.mean(v.values()) for a, v in s4["arms"].items()},
        "EXP21_recovery": read(folder / "o1_learning/exp21/recovery_summary.json"),
    }


def main() -> None:
    """Print the complete audit report; redirect stdout to retain a verification record."""
    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "numerical": numerical_audits(),
        "other_evidence": other_evidence(),
        "inventories": inventories(),
        "source_file_sha256": HASHES,
        "scope": "All EXP15/16/18 holdout observations; not all training caches or raw model requests.",
        "live_calls": 0,
        "candidate_executions": 0,
        "status": "PASS",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
