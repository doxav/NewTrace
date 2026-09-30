"""Frozen S1 fixed-bank evaluation and selection diagnostics on independent tasks."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import inspect
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.evidence import environment
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import generation as G
from experiments.recursive_opt._shared.optimizer_discovery.investigation16.statistics.audit import spearman

ROOT = Path(__file__).resolve().parent
BLOCKS = (16101, 16102, 16103, 16104)
SIZES = (1, 2, 4)
MIDPOINT = '''def propose(history, bounds, seed):
    """Return the feasible box midpoint independently of history."""
    return [(low + high) / 2 for low, high in bounds]
'''
UNIFORM = '''def propose(history, bounds, seed):
    """Explore uniformly with deterministic local replay."""
    import random
    rng = random.Random(seed + len(history))
    return [rng.uniform(low, high) for low, high in bounds]
'''


def subset_schedule() -> list[dict[str, list[Any]]]:
    """Freeze paired nested subsamples independently of every policy outcome."""
    rng = random.Random(161516)
    output = []
    for _ in range(200):
        instances = []
        for _ in range(6):
            indices = list(range(4))
            rng.shuffle(indices)
            instances.append(indices)
        locals_ = list(range(4))
        rng.shuffle(locals_)
        output.append({"instances": instances, "locals": locals_})
    return output


def choose(
    scores: np.ndarray,
    valid: np.ndarray,
    draw: dict[str, list[Any]],
    instances: int,
    local_count: int,
) -> dict[str, Any]:
    """Select by balanced training means, preserving typed invalidity and exact ties."""
    if scores.shape != valid.shape or scores.ndim != 4 or scores.shape[1:] != (6, 4, 4):
        raise ValueError("selection requires aligned policy-by-6-by-4-by-4 tensors")
    if instances not in SIZES or local_count not in SIZES:
        raise ValueError("unregistered selection-panel size")
    ranking: list[float | None] = []
    for policy in range(len(scores)):
        means = []
        eligible = True
        for stratum in range(6):
            indices = np.ix_(
                draw["instances"][stratum][:instances], draw["locals"][:local_count]
            )
            if not valid[policy, stratum][indices].all():
                eligible = False
                break
            values = scores[policy, stratum][indices]
            if not np.isfinite(values).all():
                raise RuntimeError("valid trajectory contains a nonfinite metric")
            means.append(float(values.mean()))
        ranking.append(statistics.mean(means) if eligible else None)
    if ranking[0] is None:
        raise RuntimeError("trusted seed failed a registered training trajectory")
    winner = min(
        (i for i, value in enumerate(ranking) if value is not None),
        key=lambda i: (ranking[i], i),
    )
    return {"bank_index": winner, "score": ranking[winner], "ranking": ranking}


def verify_complete(expected: set[str], completed: set[str]) -> None:
    """Block selection/audit when registered evidence is missing or unexpected."""
    if expected != completed:
        raise RuntimeError("incomplete or unexpected fixed-bank trajectory evidence")


def code_hashes() -> dict[str, str]:
    """Identify exact evaluator, runner, protocol and reused diagnostic semantics."""
    paths = [
        Path(__file__).resolve(),
        ROOT / "PROTOCOL_S1.md",
        Path(B.__file__),
        Path(E.__file__),
        G.ROOT / "statistics/audit.py",
        Path(inspect.getsourcefile(B.propose_point)),
    ]
    hashes = {
        str(path.relative_to(G.ROOT.parents[4])): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
    }
    for function in (G.fresh_tasks, G.local_seed):
        hashes[function.__name__] = hashlib.sha256(
            inspect.getsource(function).encode()
        ).hexdigest()
    return hashes


def prepare() -> dict[str, Any]:
    """Freeze bank sources, fresh task identities, subsamples and code before evaluation."""
    target = ROOT / "freeze.json"
    if E.exists(target):
        return preflight()
    bank = [{"source": B.SEED_SOURCE, "provenance": ["unchanged EXP-15 seed"]}]
    for outer in (11, 23, 37, 41, 53):
        selection = E.read(B.ROOT / "exp15/raw" / str(outer) / "selection.json")
        for arm in ("A1", "A2"):
            source = selection[arm]["source"]
            known = next((row for row in bank if row["source"] == source), None)
            if known is None:
                bank.append(
                    {
                        "source": source,
                        "provenance": [f"EXP-15 {outer}/{arm} validation-selected"],
                    }
                )
            else:
                known["provenance"].append(f"EXP-15 {outer}/{arm} validation-selected")
    bank.extend(
        [
            {"source": MIDPOINT, "provenance": ["fixed midpoint diagnostic"]},
            {"source": UNIFORM, "provenance": ["fixed uniform diagnostic"]},
        ]
    )
    if len(bank) != 11:
        raise RuntimeError("registered bank requires exactly eleven unique policies")
    for index, policy in enumerate(bank):
        source = policy.pop("source")
        digest = B.source_hash(source)
        path = ROOT / "sources" / f"{digest}.py.gz"
        payload = gzip.compress(source.encode(), mtime=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != payload:
            raise RuntimeError(
                "existing source export does not match exact selected source"
            )
        path.write_bytes(payload)
        policy.update(
            {
                "bank_index": index,
                "source_sha256": digest,
                "source_path": str(path.relative_to(ROOT)),
            }
        )
    tasks = {
        "train": G.fresh_tasks("S1", "train", 4),
        "audit": G.fresh_tasks("S1", "audit", 2),
    }
    identities = [B.task_identity(task) for panel in tasks.values() for task in panel]
    previous = {
        B.task_identity(task)
        for phase in ("pilot", "confirmation")
        for split in ("train", "validation", "holdout")
        for task in B.make_tasks(phase, split)
    }
    if len(set(identities)) != 36 or set(identities).intersection(previous):
        raise RuntimeError("S1 tasks overlap previous evidence or each other")
    freeze = {
        "experiment": "EXP-16",
        "stage": "S1",
        "frozen_ns": time.time_ns(),
        "code_hashes": code_hashes(),
        "environment": environment(),
        "bank": bank,
        "tasks": tasks,
        "local_blocks": list(BLOCKS),
        "subset_schedule": subset_schedule(),
        "budget": 32,
        "worker_limit": 4,
        "expected_trajectories": 1584,
        "expected_objective_allocations": 50688,
        "reference_objective_calls": 4608,
    }
    E.persist(target, freeze)
    return freeze


def preflight() -> dict[str, Any]:
    """Verify frozen semantics and exact source bytes before run/resume/analysis."""
    freeze = E.read(ROOT / "freeze.json")
    if freeze["code_hashes"] != code_hashes() or freeze["environment"] != environment():
        raise RuntimeError("S1 source/environment freeze mismatch")
    for policy in freeze["bank"]:
        if B.source_hash(load_source(policy)) != policy["source_sha256"]:
            raise RuntimeError("source hash mismatch")
    return freeze


def load_source(policy: dict[str, Any]) -> str:
    """Recover exact evaluated source from its lossless export."""
    return gzip.decompress((ROOT / policy["source_path"]).read_bytes()).decode()


def jobs(freeze: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """Enumerate immutable trajectory IDs with complete scientific cache keys."""
    output = []
    for policy in freeze["bank"]:
        for task_index, task in enumerate(freeze["tasks"][split]):
            for block_index, block in enumerate(BLOCKS):
                key = {
                    "source_sha256": policy["source_sha256"],
                    "task_identity": B.task_identity(task),
                    "split": split,
                    "local_seed": G.local_seed("S1", block, task),
                    "budget": 32,
                    "deployment": split == "audit",
                    "freeze_sha256": B.digest(freeze),
                }
                output.append(
                    {
                        "id": B.digest(key),
                        "key": key,
                        "bank_index": policy["bank_index"],
                        "task_index": task_index,
                        "block_index": block_index,
                        "task": task,
                        "policy": policy,
                    }
                )
    return output


def trajectory_path(split: str, identity: str) -> Path:
    """Locate one immutable result independently of compression choice."""
    return ROOT / "raw" / split / f"{identity}.json"


def checked_result(job: dict[str, Any]) -> dict[str, Any]:
    """Verify complete scientific identity whenever resumed evidence is consumed."""
    raw = E.read(trajectory_path(job["key"]["split"], job["id"]))
    if raw["key"] != job["key"]:
        raise RuntimeError("completed trajectory key changed")
    for key in ("source_sha256", "task_identity", "local_seed", "budget"):
        if raw["result"][key] != job["key"][key]:
            raise RuntimeError("completed trajectory result differs from its key")
    return raw


def execute_job(job: dict[str, Any]) -> dict[str, Any]:
    """Resume a completed result or evaluate exactly its registered local trajectory."""
    path = trajectory_path(job["key"]["split"], job["id"])
    if E.exists(path):
        return checked_result(job)
    started = time.time_ns()
    result = B.evaluate(
        load_source(job["policy"]),
        job["task"],
        job["key"]["local_seed"],
        budget=32,
        deployment=job["key"]["deployment"],
    )
    if job["bank_index"] == 0 and not result["valid"]:
        raise RuntimeError("trusted seed failure is an evaluator defect")
    saved = {
        "key": job["key"],
        "started_ns": started,
        "completed_ns": time.time_ns(),
        "result": result,
    }
    E.persist(path, saved)
    return saved


def run_split(freeze: dict[str, Any], split: str) -> None:
    """Run up to four local evaluations concurrently, retaining all completed evidence."""
    if split == "audit" and not E.exists(ROOT / "selections_frozen.json"):
        raise RuntimeError("audit is blocked until every selection is frozen")
    registered = jobs(freeze, split)
    pending = [
        job for job in registered if not E.exists(trajectory_path(split, job["id"]))
    ]
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(execute_job, job) for job in pending]
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if count % 24 == 0 or count == len(pending):
                print(
                    json.dumps(
                        {
                            "split": split,
                            "new_completed": count,
                            "new_allocated": len(pending),
                            "elapsed_s": time.monotonic() - start,
                        }
                    ),
                    flush=True,
                )
    completed = {
        path.name.split(".", 1)[0] for path in (ROOT / "raw" / split).glob("*.json*")
    }
    verify_complete({job["id"] for job in registered}, completed)


def training_grid(freeze: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Load every registered training result without inventing values for invalidity."""
    scores = np.full((11, 6, 4, 4), np.nan)
    valid = np.zeros(scores.shape, dtype=bool)
    for job in jobs(freeze, "train"):
        path = trajectory_path("train", job["id"])
        if not E.exists(path):
            raise RuntimeError("incomplete training grid")
        result = checked_result(job)["result"]
        index = (
            job["bank_index"],
            job["task_index"] // 4,
            job["task_index"] % 4,
            job["block_index"],
        )
        valid[index] = result["valid"]
        if result["valid"]:
            scores[index] = result["metrics"]["auc"]
    return scores, valid


def freeze_selections(freeze: dict[str, Any]) -> None:
    """Persist every subset decision and source identity before audit evaluation."""
    target = ROOT / "selections_frozen.json"
    if E.exists(target):
        if E.read(target)["freeze_sha256"] != B.digest(freeze):
            raise RuntimeError("selection freeze differs from registered configuration")
        return
    scores, valid = training_grid(freeze)
    selections = []
    for m in SIZES:
        for k in SIZES:
            for draw_index, draw in enumerate(freeze["subset_schedule"]):
                choice = choose(scores, valid, draw, m, k)
                selections.append(
                    {
                        "instances_per_stratum": m,
                        "local_seeds": k,
                        "draw": draw_index,
                        **choice,
                        "source_sha256": freeze["bank"][choice["bank_index"]][
                            "source_sha256"
                        ],
                    }
                )
    E.persist(
        target,
        {
            "freeze_sha256": B.digest(freeze),
            "frozen_ns": time.time_ns(),
            "selections": selections,
        },
    )


def analyze(freeze: dict[str, Any]) -> dict[str, Any]:
    """Describe independent-audit selection performance and separate variance sources."""
    selected = E.read(ROOT / "selections_frozen.json")
    audit: dict[int, list[dict[str, Any]]] = {i: [] for i in range(11)}
    accounting = {
        "objective_calls": 0,
        "unused_allocations": 0,
        "subprocess_executions": 0,
        "invalid_trajectories": {"train": 0, "audit": 0},
        "fallback_trajectories": 0,
    }
    for split in ("train", "audit"):
        registered = jobs(freeze, split)
        verify_complete(
            {job["id"] for job in registered},
            {
                path.name.split(".", 1)[0]
                for path in (ROOT / "raw" / split).glob("*.json*")
            },
        )
        for job in registered:
            raw = checked_result(job)
            row = raw["result"]
            accounting["objective_calls"] += row["objective_calls"]
            accounting["unused_allocations"] += row["unused_objective_allocation"]
            accounting["subprocess_executions"] += row["subprocess_executions"]
            accounting["invalid_trajectories"][split] += not row["candidate_valid"]
            accounting["fallback_trajectories"] += row["fallback_used"]
            if split == "audit":
                if raw["started_ns"] <= selected["frozen_ns"] or not row["valid"]:
                    raise RuntimeError(
                        "audit barrier or deployment infrastructure failed"
                    )
                audit[job["bank_index"]].append(row)
    means = [B.aggregate(audit[i], "auc") for i in range(11)]
    reference = min(means)
    summaries = []
    for m in SIZES:
        for k in SIZES:
            decisions = [
                row
                for row in selected["selections"]
                if row["instances_per_stratum"] == m and row["local_seeds"] == k
            ]
            values = [means[row["bank_index"]] for row in decisions]
            correlations = []
            for row in decisions:
                eligible = [
                    i for i, value in enumerate(row["ranking"]) if value is not None
                ]
                rho = spearman(
                    [row["ranking"][i] for i in eligible], [means[i] for i in eligible]
                )
                if rho is not None:
                    correlations.append(rho)
            summaries.append(
                {
                    "instances_per_stratum": m,
                    "local_seeds": k,
                    "selection_mean_audit_auc": statistics.mean(values),
                    "selection_median_audit_auc": statistics.median(values),
                    "mean_excess_above_best_finite_bank_audit": statistics.mean(values)
                    - reference,
                    "selection_frequency": {
                        str(i): sum(row["bank_index"] == i for row in decisions)
                        for i in range(11)
                    },
                    "mean_rank_correlation_to_audit": (
                        statistics.mean(correlations) if correlations else None
                    ),
                    "paired_draw_audit_values": values,
                }
            )
    scores, valid = training_grid(freeze)
    components = []
    for policy in range(11):
        for stratum in range(6):
            values = scores[policy, stratum]
            complete = bool(valid[policy, stratum].all())
            within = float(values.var(axis=1, ddof=1).mean()) if complete else None
            between = float(values.mean(axis=1).var(ddof=1)) if complete else None
            components.append(
                {
                    "bank_index": policy,
                    "stratum_index": stratum,
                    "all_valid": complete,
                    "within_task_local_variance": within,
                    "variance_of_task_means": between,
                    "raw_task_variance_component": (
                        between - within / 4 if complete else None
                    ),
                }
            )
    result = {
        "experiment": "EXP-16",
        "stage": "S1",
        "status": "COMPLETE_EXPLORATORY_FIXED_BANK",
        "freeze_sha256": B.digest(freeze),
        "selected_before_audit_ns": selected["frozen_ns"],
        "audit_bank_auc": means,
        "best_finite_bank_audit_index": means.index(reference),
        "summaries": summaries,
        "variance_components": components,
        "accounting": accounting,
        "reference_objective_calls": 4608,
        "new_model_calls": 0,
    }
    E.persist(ROOT / "results.json", result)
    return result


def main() -> None:
    """Prepare once, resume immutable local evaluations, or recompute diagnostics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    arguments = parser.parse_args()
    if arguments.action == "prepare":
        freeze = prepare()
        print(
            json.dumps(
                {
                    "freeze_sha256": B.digest(freeze),
                    "bank_size": len(freeze["bank"]),
                    "trajectories": freeze["expected_trajectories"],
                }
            ),
            flush=True,
        )
        return
    freeze = preflight()
    if arguments.action == "run":
        for panel in freeze["tasks"].values():
            for task in panel:
                B.normalization(task)
        run_split(freeze, "train")
        freeze_selections(freeze)
        run_split(freeze, "audit")
    result = analyze(freeze)
    print(
        json.dumps({"status": result["status"], "accounting": result["accounting"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
