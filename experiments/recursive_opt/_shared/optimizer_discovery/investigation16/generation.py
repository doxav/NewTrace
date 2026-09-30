"""Frozen EXP-16 generation mechanism probes reusing the portable EXP-15 evaluator."""

from __future__ import annotations

import argparse
import contextlib
import gzip
import io
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.evidence import collect_metadata, environment
from experiments.recursive_opt._shared.optimizer_discovery.phase0 import _load_key, _safe
from opto.features.recursive_opt import spec as control
from opto.features.recursive_opt.measurement import is_transient_provider_error
from opto.features.recursive_opt.optimizer_program import parse_program
from opto.features.recursive_opt.runmode import make_live_llm

ROOT = Path(__file__).resolve().parent
MODEL = "deepseek/deepseek-v4-flash-0731"
BASE_SHA = "13ebda2242e1c18022591737b113030ca2ce2da2"


def fresh_tasks(stage: str, split: str, replicates: int) -> list[dict[str, Any]]:
    """Reconstruct separate balanced tasks with EXP-15 semantics in a new namespace."""
    if not stage or split not in ("train", "validation", "audit"):
        raise ValueError("a stage and a registered diagnostic split are required")
    if type(replicates) is not int or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    tasks = []
    for family in B.MANIFEST["families"]:
        for dimension in B.MANIFEST["dimensions"]:
            for index in range(replicates):
                seed = int(
                    B.digest(["EXP-16", stage, split, family, dimension, index])[:16],
                    16,
                )
                rng = random.Random(seed)
                tasks.append(
                    {
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
                )
    return tasks


def local_seed(stage: str, block: int, task: dict[str, Any]) -> int:
    """Separate diagnostic optimizer randomness from generation and instance seeds."""
    return (
        int(B.digest(["EXP-16", stage, "local", block, B.task_identity(task)])[:16], 16)
        & 0x7FFFFFFF
    )


def requests(contexts: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    """Allocate all paired cap probes in a deterministic counterbalanced order."""
    if set(contexts) != {"I", "L"}:
        raise ValueError("G1 requires both frozen prompt contexts")
    result = []
    for block_index, seed in enumerate(range(16001, 16007)):
        order = ["I", "L"] if block_index % 2 == 0 else ["L", "I"]
        for context_index, context in enumerate(order):
            caps = (
                [8000, 32000]
                if (block_index + context_index) % 2 == 0
                else [32000, 8000]
            )
            for cap in caps:
                result.append(
                    {
                        "slot_id": f"{seed}/A{context}_{cap}/slot_00",
                        "block": seed,
                        "context": context,
                        "model": MODEL,
                        "messages": contexts[context],
                        "settings": {
                            "temperature": 0.6,
                            "top_p": 1.0,
                            "max_tokens": cap,
                            "extra_body": {"reasoning": {"effort": "low"}},
                            "timeout": 300,
                            "seed": seed,
                            "num_retries": 0,
                        },
                    }
                )
    return result


def complete_slot(
    directory: Path, request: dict[str, Any], client: Any
) -> dict[str, Any]:
    """Persist exactly one response with bounded transport attempts and safe resume."""
    E.persist(directory / "request.json", request)
    target = directory / "response.json"
    if E.exists(target):
        return E.read(target)
    for started in directory.glob("started_*.json"):
        if not E.exists(directory / started.name.replace("started_", "attempt_")):
            raise RuntimeError("uncertain remote completion requires reconciliation")
    offset = len(list(directory.glob("attempt_*.json")))
    for index in range(4):
        attempt = offset + index + 1
        E.persist(
            directory / f"started_{attempt}.json",
            {"time_ns": time.time_ns(), "status": "in_flight"},
        )
        captured = io.StringIO()
        started_at = time.monotonic()
        try:
            with (
                contextlib.redirect_stdout(captured),
                contextlib.redirect_stderr(captured),
            ):
                response = client(messages=request["messages"], **request["settings"])
        except Exception as error:  # noqa: BLE001 - preserve every transport attempt
            transient = is_transient_provider_error(error)
            E.persist(
                directory / f"attempt_{attempt}.json",
                {
                    "status": "transport_failure",
                    "error_type": type(error).__name__,
                    "error": _safe(str(error)),
                    "transient": transient,
                    "possible_remote_completion_or_duplicate_billing": True,
                    "wall_s": time.monotonic() - started_at,
                    "logs": _safe(captured.getvalue())[:16000],
                },
            )
            if not transient or index == 3:
                raise RuntimeError(
                    "transport attempts exhausted; slot uncompleted"
                ) from None
            time.sleep((2, 4, 8)[index])
            continue
        content = control._optimizer_response_text(response)
        try:
            source = parse_program(content)
            parse_status = "parsed"
        except ValueError:
            source, parse_status = "", "unparsable"
        usage = getattr(response, "usage", {}) or {}
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        counters = {
            k: usage[k]
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            if usage.get(k) is not None
        }
        reasoning = (usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens"
        )
        if reasoning is not None:
            counters["reasoning_tokens"] = reasoning
        cost = usage.get("cost_usd", usage.get("cost"))
        if cost is not None:
            counters["cost_usd"] = cost
        result = {
            "completed": True,
            "completed_ns": time.time_ns(),
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "finish_reason": getattr(response.choices[0], "finish_reason", None),
            "content": content,
            "source": source,
            "source_sha256": B.source_hash(source),
            "parse_status": parse_status,
            "source_status": B.source_status(source),
            "usage": counters,
            "wall_s": time.monotonic() - started_at,
            "attempt": attempt,
        }
        E.persist(target, result)
        E.persist(
            directory / f"attempt_{attempt}.json",
            {
                "status": "completed",
                "id": result["id"],
                "wall_s": result["wall_s"],
                "logs": _safe(captured.getvalue())[:16000],
            },
        )
        return result
    raise RuntimeError("unreachable response state")


def panel(source: str, tasks: list[dict[str, Any]], block: int) -> list[dict[str, Any]]:
    """Evaluate exact source in fresh isolated subprocesses with shared local seeds."""
    return [B.evaluate(source, task, local_seed("G1", block, task)) for task in tasks]


def sparse_feedback(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct the original sparse prompt schema using fresh training results."""
    return {
        "valid": all(row["valid"] for row in rows),
        "tasks": [
            {
                "valid": row["valid"],
                "status": row["status"],
                "observations": row["observations"][:2] + row["observations"][-2:],
                "best_observed_value": min(
                    (o["value"] for o in row["observations"]), default=None
                ),
            }
            for row in rows
        ],
    }


def prepare() -> dict[str, Any]:
    """Freeze exact contexts, requests and execution provenance before live calls."""
    target = ROOT / "generation/freeze.json"
    if E.exists(target):
        return preflight()
    tasks = fresh_tasks("G1", "train", 1)
    old = {
        B.task_identity(t)
        for phase in ("pilot", "confirmation")
        for split in ("train", "validation", "holdout")
        for t in B.make_tasks(phase, split)
    }
    if old.intersection(B.task_identity(t) for t in tasks):
        raise RuntimeError("diagnostic tasks overlap previous evidence")
    selected = B.ROOT / "exp15/selected/A2_seed_41.py.gz"
    parent = gzip.decompress(selected.read_bytes()).decode()
    expected = "1684f91acdc36c0ca6aac70afeb9cc2c4eed7ab847926d5880590e059266abb7"
    if B.source_hash(parent) != expected:
        raise RuntimeError("fixed complex parent source hash mismatch")
    source_rows = {
        "parent": panel(parent, tasks, 16000),
        "seed": panel(B.SEED_SOURCE, tasks, 16000),
    }
    E.persist(ROOT / "generation/context_evaluations.json", source_rows)
    feedback = {
        "current": sparse_feedback(source_rows["parent"]),
        "previous_attempt": sparse_feedback(source_rows["seed"]),
    }
    bounded = json.dumps(feedback, sort_keys=True, allow_nan=False)[:12000]
    contexts = {"I": [{"role": "user", "content": E.INVARIANT + B.SEED_SOURCE}]}
    contexts["L"] = [
        *contexts["I"],
        {
            "role": "user",
            "content": "Improve the current optimizer using TRAINING FEEDBACK only.\nCURRENT SOURCE:\n"
            + parent
            + "\nTRAINING FEEDBACK:\n"
            + bounded,
        },
    ]
    files = [
        Path(__file__),
        ROOT / "PROTOCOL.md",
        Path(B.__file__),
        Path(E.__file__),
        Path("opto/features/recursive_opt/optimizer_program.py"),
    ]
    frozen = {
        "experiment": "EXP-16",
        "stage": "G1",
        "status": "FROZEN_DIAGNOSTIC",
        "created_ns": time.time_ns(),
        "base_sha": BASE_SHA,
        "environment": environment(),
        "files": {
            str(p.relative_to(Path.cwd()) if p.is_absolute() else p): B.source_hash(
                p.read_text()
            )
            for p in files
        },
        "tasks": tasks,
        "requests": requests(contexts),
        "parent_sha256": expected,
        "seed_sha256": B.source_hash(B.SEED_SOURCE),
        "budget": 32,
        "feedback_was_truncated": len(
            json.dumps(feedback, sort_keys=True, allow_nan=False)
        )
        > 12000,
    }
    E.persist(target, frozen)
    return frozen


def preflight() -> dict[str, Any]:
    """Refuse modified code, environment or protocol during the registered stage."""
    frozen = E.read(ROOT / "generation/freeze.json")
    for path, expected in frozen["files"].items():
        if B.source_hash(Path(path).read_text()) != expected:
            raise RuntimeError(f"G1 freeze mismatch: {path}")
    if frozen["environment"] != environment():
        raise RuntimeError("G1 environment mismatch")
    return frozen


def run() -> None:
    """Execute the frozen schedule sequentially without replacing invalid responses."""
    frozen = preflight()
    _load_key()
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        client = make_live_llm(
            "openrouter/" + MODEL,
            cache=False,
            max_retries=1,
            request_timeout_s=300,
            allow_env_overrides=False,
            empty_response_retries=0,
            budget_resource=None,
        )
    for request in frozen["requests"]:
        directory = ROOT / "generation/raw" / request["slot_id"]
        result = complete_slot(directory, request, client)
        evaluation = directory / "evaluation.json"
        if not E.exists(evaluation):
            rows = panel(result["source"], frozen["tasks"], request["block"])
            E.persist(evaluation, rows)
        rows = E.read(evaluation)
        print(
            json.dumps(
                {
                    "slot": request["slot_id"],
                    "status": result["source_status"],
                    "eligible": all(r["valid"] for r in rows),
                    "finish": result["finish_reason"],
                    "tokens": result["usage"],
                }
            ),
            flush=True,
        )
    collect_metadata(ROOT / "generation/raw")
    summarize()


def summarize() -> dict[str, Any]:
    """Include every registered response and typed evaluation in cap-level diagnostics."""
    frozen = preflight()
    rows = []
    for request in frozen["requests"]:
        directory = ROOT / "generation/raw" / request["slot_id"]
        result = E.read(directory / "response.json")
        evaluations = E.read(directory / "evaluation.json")
        if B.source_hash(result["source"]) != result["source_sha256"]:
            raise RuntimeError("generated source integrity failure")
        rows.append(
            {
                "slot_id": request["slot_id"],
                "block": request["block"],
                "context": request["context"],
                "cap": request["settings"]["max_tokens"],
                "source_status": result["source_status"],
                "finish_reason": result["finish_reason"],
                "eligible": all(r["valid"] for r in evaluations),
                "auc": (
                    B.aggregate(evaluations, "auc")
                    if all(r["valid"] for r in evaluations)
                    else None
                ),
                "usage": result["usage"],
                "wall_s": result["wall_s"],
            }
        )
    groups = {}
    for cap in (8000, 32000):
        selected = [r for r in rows if r["cap"] == cap]
        groups[str(cap)] = {
            "responses": len(selected),
            "source_valid": sum(r["source_status"] == "valid" for r in selected),
            "eligible": sum(r["eligible"] for r in selected),
            "length": sum(r["finish_reason"] == "length" for r in selected),
            "mean_wall_s": statistics.mean(r["wall_s"] for r in selected),
            "usage": {
                k: sum(r["usage"].get(k, 0) for r in selected)
                for k in (
                    "prompt_tokens",
                    "completion_tokens",
                    "reasoning_tokens",
                    "total_tokens",
                    "cost_usd",
                )
            },
        }
    result = {
        "experiment": "EXP-16",
        "stage": "G1",
        "interpretation": "exploratory mechanism probe",
        "rows": rows,
        "caps": groups,
    }
    E.persist(ROOT / "generation/results.json", result)
    print(json.dumps(groups, indent=2), flush=True)
    return result


def main() -> None:
    """Prepare, run, or inspect a distinct frozen generation-cap diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    args = parser.parse_args()
    if args.command == "prepare":
        frozen = prepare()
        print(
            json.dumps(
                {"stage": "G1", "requests": len(frozen["requests"]), "frozen": True}
            )
        )
    elif args.command == "run":
        run()
    else:
        summarize()


if __name__ == "__main__":
    main()
