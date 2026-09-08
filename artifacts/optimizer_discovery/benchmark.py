"""Engine-independent EXP-15 tasks and portable optimizer deployment evaluation."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import statistics
import time
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import Any

from opto.features.recursive_opt.optimizer_program import propose_point

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "exp15_manifest.json").read_text())
SEED_SOURCE = MANIFEST["seed_source"]
VERSION = "exp15-benchmark/v1"


def digest(value: Any) -> str:
    """Hash canonical JSON without process-dependent Python hashing."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def source_hash(source: str) -> str:
    """Identify exact evaluated UTF-8 source, including whitespace."""
    return hashlib.sha256(source.encode()).hexdigest()


def stable_seed(domain: str, *parts: Any) -> int:
    """Derive a stable integer in an explicitly separated randomness domain."""
    return int(digest(["EXP-15", domain, *parts])[:16], 16) & 0x7FFFFFFF


def task_identity(task: dict[str, Any]) -> str:
    """Identify semantic objective parameters independently of split labels."""
    return digest(
        {
            k: task[k]
            for k in ("family", "dimension", "shift", "scales", "amplitude", "weights")
        }
    )


def make_tasks(phase: str, split: str) -> list[dict[str, Any]]:
    """Reconstruct a balanced split without observing any candidate outcomes."""
    if (
        phase not in ("pilot", "confirmation")
        or split not in MANIFEST["split_replicates"]
    ):
        raise ValueError("unknown benchmark phase or split")
    tasks = []
    for family in MANIFEST["families"]:
        for dimension in MANIFEST["dimensions"]:
            for index in range(MANIFEST["split_replicates"][split]):
                seed = stable_seed("instance", phase, split, family, dimension, index)
                rng = random.Random(seed)
                task = {
                    "family": family,
                    "dimension": dimension,
                    "shift": [rng.uniform(-2, 2) for _ in range(dimension)],
                    "scales": [rng.uniform(0.75, 1.5) for _ in range(dimension)],
                    "amplitude": 10 ** rng.uniform(-1, 1),
                    "weights": (
                        [10 ** rng.uniform(0, 3) for _ in range(dimension)]
                        if family == "quadratic"
                        else [1.0] * dimension
                    ),
                }
                tasks.append(task)
    return tasks


def local_seed(phase: str, outer_seed: int, task: dict[str, Any]) -> int:
    """Share local optimizer randomness across all compared sources and arms."""
    return stable_seed("local", phase, outer_seed, task_identity(task))


def objective(task: dict[str, Any], point: list[float]) -> float:
    """Evaluate a deterministic transformed objective entirely in the trusted host."""
    if len(point) != task["dimension"] or not all(
        type(x) in (int, float) and math.isfinite(x) for x in point
    ):
        raise ValueError("objective point must be finite and dimensionally correct")
    z = [
        (x - shift) / scale
        for x, shift, scale in zip(point, task["shift"], task["scales"])
    ]
    if task["family"] == "rosenbrock":
        y = [1 + x for x in z]
        value = sum(
            100 * (right - left * left) ** 2 + (1 - left) ** 2
            for left, right in pairwise(y)
        )
    elif task["family"] in ("sphere", "quadratic"):
        value = sum(w * x * x for w, x in zip(task["weights"], z))
    else:
        raise ValueError("unknown objective family")
    return float(task["amplitude"] * value)


@cache
def _reference_scale(serialized: str) -> float:
    """Compute shared reference preparation once for each semantic task."""
    task = json.loads(serialized)
    rng = random.Random(stable_seed("normalization", task_identity(task)))
    values = [
        objective(task, [rng.uniform(-5, 5) for _ in range(task["dimension"])])
        for _ in range(MANIFEST["normalization_reference_size"])
    ]
    scale = statistics.mean(values)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("benchmark normalization must be finite and strictly positive")
    return scale


def normalization(task: dict[str, Any]) -> float:
    """Return an arm-independent normalization unavailable to the candidate API."""
    return _reference_scale(json.dumps(task, sort_keys=True))


def metrics(values: list[float], scale: float, budget: int) -> dict[str, Any]:
    """Calculate complete-trajectory anytime regret and censored target attainment."""
    if (
        type(budget) is not int
        or budget <= 0
        or len(values) != budget
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("metrics require a complete budget and positive finite scale")
    curve = []
    best = math.inf
    for value in values:
        if (
            not math.isfinite(value)
            or value / scale < -MANIFEST["negative_regret_tolerance"]
        ):
            raise ValueError(
                "objective regret is inconsistent with the benchmark optimum"
            )
        best = min(best, max(0.0, value / scale))
        curve.append(best)
    hitting = next(
        (i + 1 for i, value in enumerate(curve) if value <= MANIFEST["target"]), None
    )
    return {
        "auc": statistics.mean(curve),
        "curve": curve,
        "final_regret": curve[-1],
        "attained": hitting is not None,
        "target_evaluations": hitting,
        "capped_target_evaluations": hitting if hitting is not None else budget + 1,
    }


def source_status(source: str) -> str:
    """Apply a declared conservative AST protocol screen, not a security sandbox."""
    if not isinstance(source, str) or not source.strip():
        return "missing_source"
    if len(source.encode()) > 65536:
        return "source_size"
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return "syntax_error"
    forbidden = {
        "open",
        "input",
        "eval",
        "exec",
        "compile",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "breakpoint",
        "help",
        "exit",
        "quit",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.alias) and (
            node.name.startswith("_")
            or node.name in {"os", "sys", "builtins", "subprocess", "socket", "pathlib"}
        ):
            return "protocol_violation"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if getattr(node, "level", 0) or any(
                name.split(".")[0] not in MANIFEST["allowed_imports"] for name in names
            ):
                return "protocol_violation"
        if isinstance(node, ast.Name) and (
            node.id in forbidden or node.id.startswith("__")
        ):
            return "protocol_violation"
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return "protocol_violation"
    return "valid"


def evaluate(
    source: str,
    task: dict[str, Any],
    seed: int,
    *,
    budget: int = 32,
    deployment: bool = False,
    seed_source: str = SEED_SOURCE,
    timeout_s: float = 2.0,
) -> dict[str, Any]:
    """Evaluate one policy; optional permanent seed fallback preserves the real budget."""
    if type(seed) is not int or type(budget) is not int or budget <= 0:
        raise ValueError("evaluation requires integer seed and positive integer budget")
    history: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    active = source
    status = source_status(source)
    candidate_valid = status == "valid"
    fallback = False
    started = time.monotonic()
    executions = 0
    while len(history) < budget:
        if status == "valid":
            proposal = propose_point(
                active,
                history,
                [[-5, 5]] * task["dimension"],
                seed,
                timeout_s=timeout_s,
            )
            executions += (
                2 if proposal.valid or proposal.status == "nondeterministic" else 1
            )
            status = proposal.status
            attempts.append(
                {
                    "evaluation_index": len(history),
                    "status": status,
                    "policy": "fallback" if fallback else "candidate",
                    "stdout": proposal.stdout,
                    "stderr": proposal.stderr,
                }
            )
        else:
            proposal = None
            attempts.append(
                {
                    "evaluation_index": len(history),
                    "status": status,
                    "policy": "candidate",
                }
            )
        if status != "valid":
            candidate_valid = False
            if fallback or source == seed_source:
                raise RuntimeError("trusted seed or deployment fallback failed")
            if not deployment:
                break
            active, fallback, status = seed_source, True, source_status(seed_source)
            if status != "valid":
                raise RuntimeError("trusted seed source failed protocol checks")
            continue
        point = proposal.point
        value = objective(task, point)
        if not math.isfinite(value):
            raise RuntimeError("benchmark produced a nonfinite objective value")
        history.append({"x": point, "value": value})
    valid = len(history) == budget
    return {
        "valid": valid,
        "candidate_valid": candidate_valid,
        "status": "valid" if valid else status,
        "source_sha256": source_hash(source),
        "task_identity": task_identity(task),
        "stratum": f'{task["family"]}/{task["dimension"]}',
        "local_seed": seed,
        "budget": budget,
        "observations": history,
        "proposal_attempts": attempts,
        "fallback_used": fallback,
        "objective_calls": len(history),
        "unused_objective_allocation": budget - len(history),
        "subprocess_executions": executions,
        "execution_s": time.monotonic() - started,
        "metrics": (
            metrics([r["value"] for r in history], normalization(task), budget)
            if valid
            else None
        ),
    }


def aggregate(rows: list[dict[str, Any]], metric: str) -> float:
    """Average instances within strata, then give each represented stratum equal weight."""
    if not rows or any(row.get("metrics") is None for row in rows):
        raise ValueError("cannot aggregate absent or invalid trajectories")
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(row["stratum"], []).append(float(row["metrics"][metric]))
    return statistics.mean(statistics.mean(values) for values in groups.values())
