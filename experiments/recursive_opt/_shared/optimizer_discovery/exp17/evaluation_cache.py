"""Shared deterministic panel cache extracted from frozen EXP-16 ownership.

Generation and selection barriers belong to the calling study. This helper keeps
the existing evaluator, cache key, immutable serialization and logical access
events, while adding authenticated reads that never evaluate or emit events.
Locks deduplicate threads in this process; they are not inter-process run locks.
"""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import search_experiment as S

_CACHE_LOCKS: dict[str, threading.Lock] = {}
_LOCK_GUARD = threading.Lock()


class CacheOwner(Protocol):
    """State and callbacks supplied by the study that owns split-access barriers."""

    config: dict[str, Any]
    frozen: dict[str, Any]
    run_root: Path
    budget: int

    def _panel_inputs(self, outer: int, split: str) -> list[tuple[dict[str, Any], int]]:
        """Return frozen task/local-seed pairs in their registered evaluation order."""
        ...

    def event(self, value: dict[str, Any]) -> None:
        """Persist one logical cache access using the calling study's event writer."""
        ...


def _validate_request(
    owner: CacheOwner, source: str, outer: int, split: str, deployment: bool
) -> None:
    """Reject invalid resource and identity inputs before any trajectory can run."""
    if (
        not isinstance(source, str)
        or type(outer) is not int
        or outer < 0
        or split not in ("train", "validation", "audit")
        or type(deployment) is not bool
        or deployment != (split == "audit")
    ):
        raise ValueError("invalid source, outer seed, split or deployment mode")
    if type(owner.budget) is not int or owner.budget < 1:
        raise ValueError("trajectory budget must be a positive integer")
    timeout = owner.config.get("timeout_s")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("proposal timeout must be finite and strictly positive")
    if any(
        not isinstance(owner.config.get(field), str) or not owner.config[field]
        for field in ("namespace", "cache_version")
    ):
        raise ValueError("cache namespace and evaluator version must be nonempty")


def cache_key(
    owner: CacheOwner,
    source: str,
    outer: int,
    split: str,
    task: dict[str, Any],
    local: int,
    deployment: bool,
) -> dict[str, Any]:
    """Construct exactly the frozen EXP-16 identity key without reading cache files."""
    _validate_request(owner, source, outer, split, deployment)
    if type(local) is not int or local < 0 or not isinstance(task, dict):
        raise ValueError("cache task and local optimizer seed are invalid")
    try:
        identity = B.task_identity(task)
        evaluator = owner.frozen["files"][str(Path(B.__file__).resolve())]
        seed = owner.frozen["seed_sha256"]
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            "cache requires a complete frozen task and evaluator identity"
        ) from None
    return {
        "namespace": owner.config["namespace"],
        "source_sha256": B.source_hash(source),
        "task_identity": identity,
        "split": split,
        "outer": outer,
        "local_seed": local,
        "budget": owner.budget,
        "deployment": deployment,
        "timeout_s": owner.config["timeout_s"],
        "seed_sha256": seed,
        "evaluator_version": owner.config["cache_version"],
        "evaluator_sha256": evaluator,
    }


def _read_row(path: Path, key: dict[str, Any]) -> dict[str, Any]:
    """Authenticate full key, exact row hash and all row-level identity fields."""
    try:
        cached = E.read(path)
        row = cached["row"]
        valid = (
            isinstance(row, dict)
            and cached["key"] == key
            and cached["row_sha256"] == B.digest(row)
            and all(
                row.get(field) == key[field]
                for field in ("source_sha256", "task_identity", "local_seed", "budget")
            )
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise RuntimeError("cached trajectory integrity failure")
    return row


def evaluate_panel(
    owner: CacheOwner,
    source: str,
    outer: int,
    split: str,
    *,
    deployment: bool = False,
    read_only: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate or authenticate an ordered panel through the shared frozen cache.

    All task/local keys are validated before workers start. In normal mode, each
    unique key is evaluated once in this process and every allocation records a
    cache-hit event. Read-only mode requires existing evidence for every key and
    produces no evaluations, writes or events, including on a cache miss. Calling
    code must enforce generation/selection barriers and overall run exclusivity.
    """
    _validate_request(owner, source, outer, split, deployment)
    workers = owner.config.get("workers")
    if type(workers) is not int or not 1 <= workers <= 8 or type(read_only) is not bool:
        raise ValueError(
            "cache requires one to eight workers and a typed read-only flag"
        )
    inputs = owner._panel_inputs(outer, split)
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("cache panel must contain a nonempty ordered input list")
    prepared = [
        (task, local, cache_key(owner, source, outer, split, task, local, deployment))
        for task, local in inputs
    ]

    def evaluate_one(
        item: tuple[dict[str, Any], int, dict[str, Any]],
    ) -> dict[str, Any]:
        """Serialize one cache key across workers and preserve its logical allocation."""
        task, local, key = item
        digest = B.digest(key)
        path = owner.run_root / "cache" / (digest + ".json")
        with _LOCK_GUARD:
            lock = _CACHE_LOCKS.setdefault(str(path.resolve()), threading.Lock())
        with lock:
            hit = E.exists(path)
            if not hit:
                if read_only:
                    raise RuntimeError(
                        "read-only cache miss; missing evidence cannot be evaluated"
                    )
                row = B.evaluate(
                    source,
                    task,
                    local,
                    budget=owner.budget,
                    deployment=deployment,
                    seed_source=B.SEED_SOURCE,
                    timeout_s=owner.config["timeout_s"],
                )
                I.persist(
                    path,
                    {
                        "key": key,
                        "row": row,
                        "row_sha256": B.digest(row),
                        "computed_clock": S.clock_snapshot(),
                    },
                )
            row = _read_row(path, key)
        if not read_only:
            owner.event(
                {
                    "event": "evaluation_cache",
                    "outer": outer,
                    "split": split,
                    "key": digest,
                    "hit": hit,
                }
            )
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(evaluate_one, prepared))
