"""Engineering probe of unchanged portable-evaluator scheduling at four worker counts."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import inspect
import json
import os
import resource
import statistics
import time
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.evidence import environment
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import generation as G

ROOT = Path(__file__).resolve().parent


def conditions() -> list[dict[str, int]]:
    """Return the complete, prespecified two-round concurrency schedule."""
    return [
        {"round": round_, "workers": workers}
        for round_, order in enumerate(((4, 1, 16, 8), (16, 8, 4, 1)))
        for workers in order
    ]


def scientific_projection(result: dict[str, Any]) -> dict[str, Any]:
    """Exclude execution duration only; every scientific result field must agree."""
    return {key: value for key, value in result.items() if key != "execution_s"}


def hashes() -> dict[str, str]:
    """Fingerprint exact source, evaluator, helper functions and protocol semantics."""
    files = [
        Path(__file__),
        ROOT / "PROTOCOL_T1.md",
        Path(B.__file__),
        Path(E.__file__),
        Path(inspect.getsourcefile(B.propose_point)),
    ]
    result = {
        str(path.relative_to(G.ROOT.parents[4])): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in files
    }
    for function in (G.fresh_tasks, G.local_seed):
        result[function.__name__] = hashlib.sha256(
            inspect.getsource(function).encode()
        ).hexdigest()
    return result


def prepare() -> dict[str, Any]:
    """Freeze all source/task/randomness inputs before collecting engineering timings."""
    if E.exists(ROOT / "freeze.json"):
        return preflight()
    representative = gzip.decompress(
        (B.ROOT / "exp15/selected/A2_seed_41.py.gz").read_bytes()
    ).decode()
    tasks = G.fresh_tasks("T1", "train", 1)
    old = {
        B.task_identity(task)
        for phase in ("pilot", "confirmation")
        for split in ("train", "validation", "holdout")
        for task in B.make_tasks(phase, split)
    }
    old.update(
        B.task_identity(task)
        for split, count in (("train", 4), ("audit", 2))
        for task in G.fresh_tasks("S1", split, count)
    )
    if {B.task_identity(task) for task in tasks}.intersection(old):
        raise RuntimeError("T1 instance overlaps earlier experiment evidence")
    source_bank = [B.SEED_SOURCE, representative]
    jobs = [
        {
            "id": f"p{policy}_t{index}_s{block}",
            "source_sha256": B.source_hash(source),
            "policy_index": policy,
            "task": task,
            "local_seed": G.local_seed("T1", block, task),
            "budget": 32,
        }
        for policy, source in enumerate(source_bank)
        for index, task in enumerate(tasks)
        for block in (16501, 16502)
    ]
    freeze = {
        "experiment": "EXP-16",
        "stage": "T1",
        "frozen_ns": time.time_ns(),
        "source_bank": source_bank,
        "tasks": tasks,
        "jobs": jobs,
        "conditions": conditions(),
        "hashes": hashes(),
        "environment": environment(),
        "host": {
            "logical_cpus": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "load_at_freeze": os.getloadavg(),
        },
        "expected_trajectories": 192,
        "expected_objective_calls": 6144,
        "reference_objective_calls": 768,
    }
    E.persist(ROOT / "freeze.json", freeze)
    return freeze


def preflight() -> dict[str, Any]:
    """Refuse scheduling changes or modified input code on resumed execution."""
    freeze = E.read(ROOT / "freeze.json")
    if freeze["hashes"] != hashes() or freeze["environment"] != environment():
        raise RuntimeError("T1 frozen source or environment mismatch")
    for job in freeze["jobs"]:
        if (
            B.source_hash(freeze["source_bank"][job["policy_index"]])
            != job["source_sha256"]
        ):
            raise RuntimeError("T1 source hash mismatch")
    return freeze


def directory(condition: dict[str, int]) -> Path:
    """Give each worker-count repetition its own immutable evidence directory."""
    return ROOT / "raw" / f"round_{condition['round']}_workers_{condition['workers']}"


def execute(job: dict[str, Any], source: str, target: Path) -> dict[str, Any]:
    """Evaluate one registered combination or preserve its existing completed outcome."""
    if E.exists(target):
        raw = E.read(target)
        if raw["job"] != job:
            raise RuntimeError("completed T1 trajectory identity changed")
        return raw
    started = time.time_ns()
    try:
        result = B.evaluate(source, job["task"], job["local_seed"], budget=32)
        raw = {
            "job": job,
            "status": "completed",
            "started_ns": started,
            "completed_ns": time.time_ns(),
            "result": result,
        }
    except Exception as error:  # noqa: BLE001 - record every engineering failure
        raw = {
            "job": job,
            "status": "infrastructure_error",
            "started_ns": started,
            "completed_ns": time.time_ns(),
            "error_type": type(error).__name__,
            "error": E._safe(str(error)),
        }
    E.persist(target, raw)
    return raw


def run_condition(freeze: dict[str, Any], condition: dict[str, int]) -> None:
    """Measure a complete scheduling condition while retaining interrupted timing limitations."""
    root = directory(condition)
    if E.exists(root / "timing.json"):
        return
    completed_before = sum(
        E.exists(root / f"{job['id']}.json") for job in freeze["jobs"]
    )
    start = time.monotonic()
    host_cpu = time.process_time()
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=condition["workers"]
    ) as executor:
        futures = [
            executor.submit(
                execute,
                job,
                freeze["source_bank"][job["policy_index"]],
                root / f"{job['id']}.json",
            )
            for job in freeze["jobs"]
        ]
        rows = [future.result() for future in futures]
    children_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    timing = {
        **condition,
        "wall_s": time.monotonic() - start,
        "host_cpu_s": time.process_time() - host_cpu,
        "children_cpu_s": (children_after.ru_utime + children_after.ru_stime)
        - (children_before.ru_utime + children_before.ru_stime),
        "resumed_completed_rows": completed_before,
        "timing_eligible": completed_before == 0,
        "completed_trajectories": len(rows),
        "infrastructure_errors": sum(row["status"] != "completed" for row in rows),
        "load_at_completion": os.getloadavg(),
    }
    E.persist(root / "timing.json", timing)
    print(json.dumps(timing), flush=True)


def analyze(freeze: dict[str, Any]) -> dict[str, Any]:
    """Compare every condition's exact scientific outcomes and summarize paired timings."""
    reference: dict[str, str] = {}
    mismatches = []
    timings = []
    counts = {
        "trajectories": 0,
        "objective_calls": 0,
        "subprocess_executions": 0,
        "invalid_candidates": 0,
        "infrastructure_errors": 0,
    }
    for condition in freeze["conditions"]:
        root = directory(condition)
        timing = E.read(root / "timing.json")
        timings.append(timing)
        for job in freeze["jobs"]:
            raw = E.read(root / f"{job['id']}.json")
            if raw["job"] != job:
                raise RuntimeError("T1 trajectory does not match frozen job")
            counts["trajectories"] += 1
            if raw["status"] != "completed":
                counts["infrastructure_errors"] += 1
                mismatches.append(
                    {**condition, "job": job["id"], "status": raw["status"]}
                )
                continue
            result = raw["result"]
            counts["objective_calls"] += result["objective_calls"]
            counts["subprocess_executions"] += result["subprocess_executions"]
            counts["invalid_candidates"] += not result["candidate_valid"]
            actual = B.digest(scientific_projection(result))
            expected = reference.setdefault(job["id"], actual)
            if actual != expected:
                mismatches.append(
                    {
                        **condition,
                        "job": job["id"],
                        "status": "scientific_result_mismatch",
                    }
                )
    summaries = []
    for workers in (1, 4, 8, 16):
        eligible = [
            row
            for row in timings
            if row["workers"] == workers and row["timing_eligible"]
        ]
        summaries.append(
            {
                "workers": workers,
                "eligible_repetitions": len(eligible),
                "wall_s": [row["wall_s"] for row in eligible],
                "mean_wall_s": (
                    statistics.mean(row["wall_s"] for row in eligible)
                    if eligible
                    else None
                ),
                "mean_children_cpu_s": (
                    statistics.mean(row["children_cpu_s"] for row in eligible)
                    if eligible
                    else None
                ),
            }
        )
    baseline = summaries[0]["mean_wall_s"]
    for summary in summaries:
        summary["speedup_vs_one_worker"] = (
            baseline / summary["mean_wall_s"]
            if baseline and summary["mean_wall_s"]
            else None
        )
    result = {
        "experiment": "EXP-16",
        "stage": "T1",
        "status": "COMPLETE_ENGINEERING_PROBE",
        "freeze_sha256": B.digest(freeze),
        "counts": counts,
        "timings": timings,
        "summaries": summaries,
        "mismatches": mismatches,
        "all_scientific_results_identical": not mismatches
        and counts["trajectories"] == 192,
        "reference_objective_calls": 768,
        "new_model_calls": 0,
    }
    E.persist(ROOT / "results.json", result)
    return result


def main() -> None:
    """Prepare a fixed engineering probe, execute it once, or recompute its summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    action = parser.parse_args().action
    if action == "prepare":
        freeze = prepare()
        print(
            json.dumps({"freeze_sha256": B.digest(freeze), "trajectories": 192}),
            flush=True,
        )
        return
    freeze = preflight()
    if action == "run":
        for task in freeze["tasks"]:
            B.normalization(task)
        for condition in freeze["conditions"]:
            run_condition(freeze, condition)
    print(json.dumps(analyze(freeze)), flush=True)


if __name__ == "__main__":
    main()
