"""Cache identity, read-only isolation and concurrency tests without live calls."""

from __future__ import annotations

import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.exp17 import evaluation_cache as C
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import search_experiment as S


class Owner:
    """Provide only the owner fields and callbacks required by the shared cache."""

    def __init__(self, root: Path) -> None:
        """Use a deterministic task without evaluating its reference or objective."""
        self.run_root = root
        self.config = {
            "namespace": "TEST-EXP17-CACHE",
            "workers": 2,
            "timeout_s": 2,
            "cache_version": "exp17-evaluation-v1",
        }
        self.frozen = {
            "seed_sha256": B.source_hash(B.SEED_SOURCE),
            "files": {str(Path(B.__file__).resolve()): "a" * 64},
        }
        self.budget = 2
        self.events: list[dict[str, Any]] = []
        self.inputs = [(B.make_tasks("pilot", "train")[0], 7)]

    def _panel_inputs(self, outer: int, split: str) -> list[tuple[dict[str, Any], int]]:
        """Return a caller-controlled panel to exercise validation before work."""
        return self.inputs

    def event(self, value: dict[str, Any]) -> None:
        """Record logical cache accesses in the same format as the production owner."""
        self.events.append(value)


def evaluated(
    source: str, task: dict[str, Any], local: int, **kwargs: Any
) -> dict[str, Any]:
    """Provide deterministic synthetic evaluator output with exact cache identity."""
    return {
        "source_sha256": B.source_hash(source),
        "task_identity": B.task_identity(task),
        "local_seed": local,
        "budget": kwargs["budget"],
        "valid": bool(source.strip()),
        "status": "valid" if source.strip() else "missing_source",
        "objective_calls": kwargs["budget"] if source.strip() else 0,
        "unused_objective_allocation": 0 if source.strip() else kwargs["budget"],
        "metrics": {"auc": 0.25} if source.strip() else None,
    }


def test_exact_key_fields_match_frozen_panel_convention(tmp_path: Path) -> None:
    owner = Owner(tmp_path)
    task, local = owner.inputs[0]
    assert C.cache_key(owner, B.SEED_SOURCE, 17011, "train", task, local, False) == {
        "namespace": "TEST-EXP17-CACHE",
        "source_sha256": B.source_hash(B.SEED_SOURCE),
        "task_identity": B.task_identity(task),
        "split": "train",
        "outer": 17011,
        "local_seed": 7,
        "budget": 2,
        "deployment": False,
        "timeout_s": 2,
        "seed_sha256": B.source_hash(B.SEED_SOURCE),
        "evaluator_version": "exp17-evaluation-v1",
        "evaluator_sha256": "a" * 64,
    }


def test_normal_mode_matches_frozen_panel_and_retains_invalid_allocations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(B, "evaluate", evaluated)
    owner, legacy = Owner(tmp_path / "new"), Owner(tmp_path / "old")
    for source in (B.SEED_SOURCE, ""):
        assert C.evaluate_panel(
            owner, source, 17011, "train"
        ) == S.SearchExperiment.panel(legacy, source, 17011, "train")
        assert C.evaluate_panel(
            owner, source, 17011, "train"
        ) == S.SearchExperiment.panel(legacy, source, 17011, "train")
    assert owner.events == legacy.events
    invalid = C.evaluate_panel(owner, "", 17011, "train")[0]
    assert invalid["metrics"] is None
    assert invalid["objective_calls"] == 0
    assert invalid["unused_objective_allocation"] == 2


def test_read_only_miss_has_no_evaluation_write_or_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = Owner(tmp_path)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        """Fail if a read-only cache miss tries to manufacture missing evidence."""
        raise AssertionError("read-only mode performed work")

    monkeypatch.setattr(B, "evaluate", forbidden)
    monkeypatch.setattr(C.I, "persist", forbidden)
    with pytest.raises(RuntimeError, match="read-only cache miss"):
        C.evaluate_panel(owner, B.SEED_SOURCE, 17011, "train", read_only=True)
    assert owner.events == []
    assert not (tmp_path / "cache").exists()


def test_read_only_hit_preserves_files_and_emits_no_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = Owner(tmp_path)
    monkeypatch.setattr(B, "evaluate", evaluated)
    expected = C.evaluate_panel(owner, B.SEED_SOURCE, 17011, "train")
    before = {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    events = copy.deepcopy(owner.events)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        """Catch unrequested evaluation or write even on an existing cache entry."""
        raise AssertionError("read-only hit performed work")

    monkeypatch.setattr(B, "evaluate", forbidden)
    monkeypatch.setattr(C.I, "persist", forbidden)
    assert (
        C.evaluate_panel(owner, B.SEED_SOURCE, 17011, "train", read_only=True)
        == expected
    )
    assert owner.events == events
    assert {
        path: (path.stat().st_mtime_ns, path.read_bytes()) for path in before
    } == before


@pytest.mark.parametrize(
    "corruption", ["key", "row_hash", "row_identity", "missing_field"]
)
def test_corrupt_cache_is_rejected_in_both_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    owner = Owner(tmp_path)
    monkeypatch.setattr(B, "evaluate", evaluated)
    C.evaluate_panel(owner, B.SEED_SOURCE, 17011, "train")
    path = next((tmp_path / "cache").glob("*.json"))
    cache = E.read(path)
    if corruption == "key":
        cache["key"]["split"] = "validation"
    elif corruption == "row_hash":
        cache["row"]["metrics"]["auc"] = 0.5
    elif corruption == "row_identity":
        cache["row"]["local_seed"] += 1
        cache["row_sha256"] = B.digest(cache["row"])
    else:
        del cache["row"]["source_sha256"]
        cache["row_sha256"] = B.digest(cache["row"])
    path.write_text(json.dumps(cache))
    for read_only in (False, True):
        with pytest.raises(RuntimeError, match="cache.*integrity"):
            C.evaluate_panel(owner, B.SEED_SOURCE, 17011, "train", read_only=read_only)


def test_concurrent_shared_key_is_evaluated_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = Owner(tmp_path)
    owner.inputs *= 2
    count = [0]
    lock = threading.Lock()

    def slow(
        source: str, task: dict[str, Any], local: int, **kwargs: Any
    ) -> dict[str, Any]:
        """Overlap workers so the per-key lock, rather than scheduling luck, deduplicates."""
        with lock:
            count[0] += 1
        time.sleep(0.02)
        return evaluated(source, task, local, **kwargs)

    monkeypatch.setattr(B, "evaluate", slow)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: C.evaluate_panel(owner, B.SEED_SOURCE, 17011, "train"),
                range(4),
            )
        )
    assert count == [1]
    assert all(result == results[0] for result in results)
    assert len(owner.events) == 8
    assert sum(not event["hit"] for event in owner.events) == 1


def test_gzip_rows_use_existing_evidence_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = Owner(tmp_path)
    monkeypatch.setattr(B, "evaluate", evaluated)
    C.evaluate_panel(owner, B.SEED_SOURCE, 17011, "train")
    path = next((tmp_path / "cache").glob("*.json"))
    cache = E.read(path)
    cache["row"]["padding"] = "x" * 500000
    cache["row_sha256"] = B.digest(cache["row"])
    path.unlink()
    C.I.persist(path, cache)
    assert not path.exists()
    assert path.with_suffix(".json.gz").exists()
    assert C.evaluate_panel(owner, B.SEED_SOURCE, 17011, "train", read_only=True) == [
        cache["row"]
    ]


def test_evaluator_receives_registered_budget_timeout_and_common_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = Owner(tmp_path)
    calls: list[dict[str, Any]] = []

    def capture(
        source: str, task: dict[str, Any], local: int, **kwargs: Any
    ) -> dict[str, Any]:
        """Observe the actual shared evaluator invocation without running a candidate."""
        calls.append(kwargs)
        return evaluated(source, task, local, **kwargs)

    monkeypatch.setattr(B, "evaluate", capture)
    C.evaluate_panel(owner, "candidate", 17011, "train")
    C.evaluate_panel(owner, "candidate", 17011, "audit", deployment=True)
    assert calls == [
        {
            "budget": 2,
            "deployment": deployment,
            "seed_source": B.SEED_SOURCE,
            "timeout_s": 2,
        }
        for deployment in (False, True)
    ]


def test_cache_identity_separates_sources_splits_and_local_seeds(
    tmp_path: Path,
) -> None:
    owner = Owner(tmp_path)
    task, local = owner.inputs[0]
    keys = [
        C.cache_key(owner, source, outer, split, task, local_seed, split == "audit")
        for source, outer, split, local_seed in [
            (B.SEED_SOURCE, 17011, "train", local),
            (B.SEED_SOURCE + "\n", 17011, "train", local),
            (B.SEED_SOURCE, 17012, "train", local),
            (B.SEED_SOURCE, 17011, "validation", local),
            (B.SEED_SOURCE, 17011, "audit", local),
            (B.SEED_SOURCE, 17011, "train", local + 1),
        ]
    ]
    assert len({B.digest(key) for key in keys}) == len(keys)


def test_parallel_panel_preserves_input_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = Owner(tmp_path)
    task = owner.inputs[0][0]
    owner.inputs = [(task, local) for local in (7, 8, 9)]

    def delayed(
        source: str, task: dict[str, Any], local: int, **kwargs: Any
    ) -> dict[str, Any]:
        """Complete the second trajectory first to test ordered thread-pool mapping."""
        if local == 7:
            time.sleep(0.02)
        return evaluated(source, task, local, **kwargs)

    monkeypatch.setattr(B, "evaluate", delayed)
    rows = C.evaluate_panel(owner, B.SEED_SOURCE, 17011, "train")
    assert [row["local_seed"] for row in rows] == [7, 8, 9]


@pytest.mark.parametrize(
    "change",
    ["workers", "budget", "deployment", "later_local", "source", "outer", "split"],
)
def test_invalid_inputs_fail_before_any_candidate_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    owner = Owner(tmp_path)
    arguments: dict[str, Any] = {
        "source": B.SEED_SOURCE,
        "outer": 17011,
        "split": "train",
    }
    if change == "workers":
        owner.config["workers"] = 9
    elif change == "budget":
        owner.budget = False
    elif change == "deployment":
        arguments["deployment"] = True
    elif change == "later_local":
        owner.inputs.append((owner.inputs[0][0], -1))
    elif change == "source":
        arguments["source"] = None
    elif change == "outer":
        arguments["outer"] = -1
    else:
        arguments["split"] = "holdout"

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        """Detect a partial panel evaluation before all input identities were validated."""
        raise AssertionError("invalid panel performed candidate work")

    monkeypatch.setattr(B, "evaluate", forbidden)
    with pytest.raises(ValueError):
        C.evaluate_panel(owner, **arguments)
    assert owner.events == []
    assert not (tmp_path / "cache").exists()
