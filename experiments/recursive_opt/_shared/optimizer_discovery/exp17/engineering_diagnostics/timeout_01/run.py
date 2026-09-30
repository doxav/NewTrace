"""Execute the eight preregistered proposal-only timeout diagnostic calls once."""

from __future__ import annotations

import hashlib
import os
import resource
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from opto.features.recursive_opt import optimizer_program as OP

ROOT = Path(__file__).resolve().parent


def digest_file(path: Path) -> str:
    """Identify exact bytes without interpreting source or evaluating an objective."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot() -> dict[str, Any]:
    """Record clocks and CPU counters in this isolated diagnostic parent process."""
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "wall_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "parent_user_s": own.ru_utime,
        "parent_system_s": own.ru_stime,
        "children_user_s": children.ru_utime,
        "children_system_s": children.ru_stime,
        "host_load_1_5_15": list(os.getloadavg()),
        "host_cpu_count": os.cpu_count(),
    }


def delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    """Separate elapsed time from parent and reaped-child CPU consumption."""
    return {
        "wall_s": (after["wall_ns"] - before["wall_ns"]) / 1e9,
        "monotonic_s": (after["monotonic_ns"] - before["monotonic_ns"]) / 1e9,
        **{
            key: after[key] - before[key]
            for key in (
                "parent_user_s",
                "parent_system_s",
                "children_user_s",
                "children_system_s",
            )
        },
    }


def main() -> None:
    """Keep all statuses, with no retries, timeout changes or objective evaluations."""
    protocol = E.read(ROOT / "protocol.json")
    inputs = E.read(ROOT / "inputs.json")
    sources = E.read(ROOT / "sources.json")
    required = {
        "inputs.json": protocol["inputs_sha256"],
        "sources.json": protocol["sources_sha256"],
        "run.py": protocol["run_script_sha256"],
    }
    if any(digest_file(ROOT / name) != expected for name, expected in required.items()):
        raise RuntimeError("diagnostic preregistered artifact hash mismatch")
    originals = {
        name: digest_file(Path(name)) for name in protocol["original_files_sha256"]
    }
    if originals != protocol["original_files_sha256"]:
        raise RuntimeError("diagnostic original evidence or execution source changed")
    if E.exists(ROOT / "started.json") or E.exists(ROOT / "results.json"):
        raise RuntimeError("diagnostic is single-use; no automatic rerun")
    I.persist(
        ROOT / "started.json",
        {
            "started_ns": time.time_ns(),
            "protocol_sha256": digest_file(ROOT / "protocol.json"),
        },
    )
    original_execute = OP._execute_once
    original_spawn = OP.subprocess.Popen
    original_objective = B.objective
    original_evaluate = B.evaluate
    objective_calls = 0
    subprocesses = 0
    inner: list[dict[str, Any]] = []

    def forbidden_objective(*args: Any, **kwargs: Any) -> Any:
        """Enforce the diagnostic's zero-objective budget, including accidental helpers."""
        nonlocal objective_calls
        objective_calls += 1
        raise RuntimeError("objective/evaluator access forbidden in this diagnostic")

    def spawn(*args: Any, **kwargs: Any) -> Any:
        """Count actual unchanged subprocess launches and verify their clean environment."""
        nonlocal subprocesses
        if kwargs.get("env") != {"PATH": os.defpath, "LANG": "C.UTF-8"}:
            raise RuntimeError(
                "candidate subprocess environment differs from the contract"
            )
        child = original_spawn(*args, **kwargs)
        subprocesses += 1
        return child

    def execute(source: str, payload: dict[str, Any], timeout_s: float) -> Any:
        """Observe first/replay statuses while delegating unchanged execution semantics."""
        before = snapshot()
        value = original_execute(source, payload, timeout_s)
        after = snapshot()
        inner.append(
            {
                "source_sha256": B.source_hash(source),
                "payload_sha256": B.digest(payload),
                "result": asdict(value),
                "before": before,
                "after": after,
                "elapsed": delta(before, after),
            }
        )
        return value

    calls = []
    try:
        OP._execute_once = execute
        OP.subprocess.Popen = spawn
        B.objective = forbidden_objective
        B.evaluate = forbidden_objective
        for index, item in enumerate(protocol["order"]):
            payload = inputs[item["input"]]
            source = sources[item["policy"]]["source"]
            inner_start, child_start = len(inner), subprocesses
            before = snapshot()
            I.persist(
                ROOT / f"call_{index:02d}_started.json",
                {
                    "index": index,
                    **item,
                    "input_sha256": B.digest(payload),
                    "source_sha256": B.source_hash(source),
                    "before": before,
                },
            )
            try:
                result = asdict(OP.propose_point(source, **payload, timeout_s=2.0))
                error_type = None
            except (ValueError, TypeError, OSError, RuntimeError) as error:
                result, error_type = None, type(error).__name__
            after = snapshot()
            value = {
                "index": index,
                **item,
                "input_sha256": B.digest(payload),
                "source_sha256": B.source_hash(source),
                "timeout_s": 2.0,
                "result": result,
                "error_type": error_type,
                "before": before,
                "after": after,
                "elapsed": delta(before, after),
                "subprocesses": subprocesses - child_start,
                "executions": inner[inner_start:],
            }
            I.persist(ROOT / f"call_{index:02d}.json", value)
            calls.append(value)
    finally:
        OP._execute_once = original_execute
        OP.subprocess.Popen = original_spawn
        B.objective = original_objective
        B.evaluate = original_evaluate
    after_files = {name: digest_file(Path(name)) for name in originals}
    if (
        len(calls) != 8
        or subprocesses > 16
        or objective_calls
        or originals != after_files
    ):
        raise RuntimeError(
            "diagnostic allocation or original evidence integrity failed"
        )
    I.persist(
        ROOT / "results.json",
        {
            "schema": "EXP17-TIMEOUT-DIAGNOSTIC-01",
            "completed_ns": time.time_ns(),
            "protocol_sha256": digest_file(ROOT / "protocol.json"),
            "proposal_calls": len(calls),
            "subprocesses": subprocesses,
            "objective_calls": objective_calls,
            "model_calls": 0,
            "original_files_unchanged": True,
            "original_files_sha256": originals,
            "calls": calls,
        },
    )
    print(
        {
            "proposal_calls": len(calls),
            "subprocesses": subprocesses,
            "objective_calls": objective_calls,
            "model_calls": 0,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
