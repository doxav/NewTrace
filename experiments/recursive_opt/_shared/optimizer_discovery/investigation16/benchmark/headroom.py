"""Measure fixed-policy headroom under paired changes to optimum centrality."""

from __future__ import annotations

import argparse
import copy
import gzip
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import generation as G

ROOT = Path(__file__).resolve().parent
LOCAL_SEEDS = [16201, 16202, 16203, 16204]
UNIFORM = '''def propose(history: list[dict], bounds: list[list[float]], seed: int) -> list[float]:
    """Draw a uniform point using deterministic history-dependent randomness."""
    import random
    rng = random.Random(seed + len(history))
    return [rng.uniform(low, high) for low, high in bounds]
'''
MIDPOINT = '''def propose(history: list[dict], bounds: list[list[float]], seed: int) -> list[float]:
    """Always propose the geometric center of the legal bounds."""
    return [(low + high) / 2 for low, high in bounds]
'''


def task_pairs() -> list[dict[str, dict[str, Any]]]:
    """Pair fresh tasks by expanding only their optimum's offset from the center."""
    result = []
    for task in G.fresh_tasks("B1", "train", 2):
        broad = copy.deepcopy(task)
        broad["shift"] = [2.25 * value for value in task["shift"]]
        result.append({"central": task, "broad": broad})
    return result


def early_mass(curve: list[float], count: int) -> float:
    """Return the share of regret-AUC contributed by the first count evaluations."""
    if (
        len(curve) != 32
        or type(count) is not int
        or not 1 <= count <= 32
        or not all(math.isfinite(value) and value >= 0 for value in curve)
    ):
        raise ValueError("Early mass requires a finite 32-evaluation regret curve")
    total = sum(curve)
    return sum(curve[:count]) / total if total > 0 else 0.0


def prepare() -> dict[str, Any]:
    """Freeze all diagnostic inputs and verify existing evidence before any work."""
    source = gzip.decompress(
        (B.ROOT / "exp15/selected/A2_seed_41.py.gz").read_bytes()
    ).decode()
    if B.source_hash(source) != (
        "1684f91acdc36c0ca6aac70afeb9cc2c4eed7ab847926d5880590e059266abb7"
    ):
        raise ValueError("EXP-15 representative source integrity mismatch")
    sources = {
        "seed": B.SEED_SOURCE,
        "uniform": UNIFORM,
        "midpoint": MIDPOINT,
        "representative41": source,
    }
    if any(B.source_status(value) != "valid" for value in sources.values()):
        raise ValueError("A declared fixed policy violates the artifact contract")
    pairs = task_pairs()
    frozen = {
        "stage": "EXP-16/B1 exploratory fixed-policy diagnostic",
        "budget": 32,
        "local_seeds": LOCAL_SEEDS,
        "pairs": pairs,
        "sources": sources,
        "source_hashes": {key: B.source_hash(value) for key, value in sources.items()},
        "normalization": {
            B.task_identity(task): B.normalization(task)
            for pair in pairs
            for task in pair.values()
        },
        "normalization_reference_allocations": 24 * 128,
        "allocated_trajectories": 384,
        "allocated_objective_calls": 12288,
        "script_sha256": B.source_hash(Path(__file__).read_text()),
        "benchmark_sha256": B.source_hash(Path(B.__file__).read_text()),
        "protocol_sha256": B.source_hash((ROOT / "PROTOCOL_B1.md").read_text()),
    }
    E.persist(ROOT / "freeze.json", frozen)
    return frozen


def _jobs(frozen: dict[str, Any]) -> list[dict[str, Any]]:
    """Allocate every paired condition, policy, instance and local seed exactly once."""
    return [
        {
            "condition": condition,
            "policy": policy,
            "local_seed": seed,
            "task_index": index,
            "task": pair[condition],
            "source": source,
            "path": ROOT / "raw" / condition / policy / f"{seed}_{index:02d}.json",
        }
        for index, pair in enumerate(frozen["pairs"])
        for seed in frozen["local_seeds"]
        for condition in ["central", "broad"]
        for policy, source in frozen["sources"].items()
    ]


def _evaluate(job: dict[str, Any]) -> dict[str, Any]:
    """Persist each completed trajectory unchanged; only missing rows may execute."""
    target = job["path"]
    if E.exists(target):
        row = E.read(target)
    else:
        row = B.evaluate(job["source"], job["task"], job["local_seed"], budget=32)
        E.persist(target, row)
    expected = {
        "source_sha256": B.source_hash(job["source"]),
        "task_identity": B.task_identity(job["task"]),
        "local_seed": job["local_seed"],
        "budget": 32,
    }
    if any(row[key] != value for key, value in expected.items()):
        raise ValueError("Existing trajectory does not match its frozen allocation")
    return row


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize all allocated rows, refusing numeric means after any invalid row."""
    result = {
        "n": len(rows),
        "valid": sum(row["valid"] for row in rows),
        "invalid": sum(not row["valid"] for row in rows),
        "objective_calls": sum(row["objective_calls"] for row in rows),
    }
    if result["invalid"]:
        return {**result, "metrics": None}
    metrics = {
        metric: B.aggregate(rows, metric)
        for metric in ["auc", "final_regret", "attained", "capped_target_evaluations"]
    }
    for count in [1, 4, 8]:
        enriched = [
            {
                **row,
                "metrics": {
                    **row["metrics"],
                    "early_mass": early_mass(row["metrics"]["curve"], count),
                },
            }
            for row in rows
        ]
        metrics[f"early_mass_{count}"] = B.aggregate(enriched, "early_mass")
    metrics["mean_curve"] = [
        B.aggregate(
            [
                {**row, "metrics": {"at_t": row["metrics"]["curve"][index]}}
                for row in rows
            ],
            "at_t",
        )
        for index in range(32)
    ]
    return {**result, "metrics": metrics}


def analyze(frozen: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct every metric from the complete frozen schedule, never dropping rows."""
    jobs = _jobs(frozen)
    rows = []
    for job in jobs:
        if not E.exists(job["path"]):
            raise ValueError("Analysis requires every allocated trajectory")
        rows.append(
            {
                **_evaluate(job),
                "condition": job["condition"],
                "policy": job["policy"],
                "task_index": job["task_index"],
            }
        )
    result = {
        "freeze_sha256": B.digest(frozen),
        "total_trajectories": len(rows),
        "objective_calls": sum(row["objective_calls"] for row in rows),
        "subprocess_executions": sum(row["subprocess_executions"] for row in rows),
        "execution_s": sum(row["execution_s"] for row in rows),
        "groups": {},
        "per_local_seed": {},
        "per_stratum": {},
    }
    for condition in ["central", "broad"]:
        result["groups"][condition] = {}
        for policy in frozen["sources"]:
            selected = [
                row
                for row in rows
                if row["condition"] == condition and row["policy"] == policy
            ]
            key = f"{condition}/{policy}"
            result["groups"][condition][policy] = _summarize(selected)
            result["per_local_seed"][key] = {
                str(seed): _summarize(
                    [row for row in selected if row["local_seed"] == seed]
                )
                for seed in frozen["local_seeds"]
            }
            result["per_stratum"][key] = {
                stratum: _summarize(
                    [row for row in selected if row["stratum"] == stratum]
                )
                for stratum in sorted({row["stratum"] for row in selected})
            }
    E.persist(ROOT / "results.json", result)
    return result


def run() -> None:
    """Run the fixed offline schedule with at most two concurrent trajectories."""
    frozen = prepare()
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as executor:
        for index, _ in enumerate(executor.map(_evaluate, _jobs(frozen)), start=1):
            if index % 32 == 0:
                print(
                    f"B1: {index}/384 completed; {time.monotonic() - started:.1f}s",
                    flush=True,
                )
    result = analyze(frozen)
    print(
        f"B1 complete: {result['total_trajectories']} trajectories, "
        f"{result['objective_calls']} objective calls",
        flush=True,
    )


def main() -> None:
    """Choose whether to evaluate the frozen schedule or only reaggregate its evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "analyze"])
    arguments = parser.parse_args()
    if arguments.action == "run":
        run()
    else:
        analyze(prepare())


if __name__ == "__main__":
    main()
