"""Independent integrity review of the prospective F1 execution boundary."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import feedback_experiment as F
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import generation as G


def panel_row(source: str, task: dict[str, Any], block: int) -> dict[str, Any]:
    """Construct a complete deterministic evaluator row without subprocess or LLM work."""
    point = [0.0] * task["dimension"]
    value = B.objective(task, point)
    return {
        "source_sha256": B.source_hash(source),
        "task_identity": B.task_identity(task),
        "stratum": f'{task["family"]}/{task["dimension"]}',
        "local_seed": G.local_seed(F.TASK_NAMESPACE, block, task),
        "budget": 32,
        "valid": True,
        "candidate_valid": True,
        "status": "valid",
        "fallback_used": False,
        "observations": [{"x": point, "value": value}] * 32,
        "objective_calls": 32,
        "unused_objective_allocation": 0,
        "metrics": B.metrics([value] * 32, B.normalization(task), 32),
    }


@pytest.fixture
def completed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Preserve all twenty-four slots using explicit unit-only source/response metadata."""
    tasks = {
        split: G.fresh_tasks("F1_UNIT_REVIEW", split, 1)
        for split in ("train", "validation")
    }
    requests, blocks = [], []
    for block_id in range(16301, 16307):
        block = {
            "block": block_id,
            "parent": B.SEED_SOURCE,
            "parent_kind": "seed",
            "sparse": {},
            "rich": {},
        }
        blocks.append(block)
        requests.extend(F.block_requests(block, 8000))
    frozen = {"requests": requests, "blocks": blocks, "tasks": tasks, "budget": 32}
    monkeypatch.setattr(F, "ROOT", tmp_path)
    monkeypatch.setattr(F, "preflight", lambda: frozen)

    def forbidden_evaluation(*args: Any, **kwargs: Any) -> Any:
        """Never execute real evaluation in an integrity-barrier unit test."""
        raise AssertionError("unexpected evaluation before integrity rejection")

    monkeypatch.setattr(F, "evaluate_panel", forbidden_evaluation)
    for index, request in enumerate(requests):
        directory = tmp_path / "raw" / request["slot_id"]
        I.persist(directory / "request.json", request)
        I.persist(
            directory / "response.json",
            {
                "completed": True,
                "completed_ns": index + 1,
                "id": f"offline-review-{index}",
                "model": G.MODEL,
                "source": B.SEED_SOURCE,
                "source_sha256": B.source_hash(B.SEED_SOURCE),
                "content": B.SEED_SOURCE,
                "parse_status": "parsed",
                "source_status": "valid",
                "usage": {},
            },
        )
        I.persist(
            directory / "training.json",
            [
                panel_row(B.SEED_SOURCE, task, request["block"])
                for task in tasks["train"]
            ],
        )
    return frozen


def rewrite(path: Path, field: str, value: Any) -> None:
    """Deliberately corrupt a unit fixture to exercise the integrity checks."""
    data = E.read(path)
    data[field] = value
    path.write_text(json.dumps(data))


def test_barrier_requires_all_twenty_four_slots(completed_run: dict[str, Any]) -> None:
    """The twenty-fourth missing response blocks every validation call."""
    last = completed_run["requests"][-1]
    (F.ROOT / "raw" / last["slot_id"] / "response.json").unlink()
    with pytest.raises(RuntimeError, match="every response"):
        F.validate()
    assert not (F.ROOT / "validation_opened.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [("completed", False), ("source_sha256", "wrong"), ("model", "different-model")],
)
def test_corrupt_response_blocks_validation(
    completed_run: dict[str, Any],
    field: str,
    value: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source file is not sufficient proof of a completed, correctly sourced response."""
    request = completed_run["requests"][-1]
    rewrite(F.ROOT / "raw" / request["slot_id"] / "response.json", field, value)
    calls: list[Any] = []
    monkeypatch.setattr(F, "evaluate_panel", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(RuntimeError, match="response"):
        F.validate()
    assert not calls
    assert not (F.ROOT / "validation_opened.json").exists()


def test_altered_request_blocks_validation(completed_run: dict[str, Any]) -> None:
    """Actual request configuration must still match the frozen within-block design."""
    request = completed_run["requests"][0]
    path = F.ROOT / "raw" / request["slot_id"] / "request.json"
    changed = copy.deepcopy(request["settings"])
    changed["temperature"] = 1.1
    rewrite(path, "settings", changed)
    with pytest.raises(RuntimeError, match="request"):
        F.validate()


def test_no_new_call_after_validation_barrier(
    completed_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing completed response after validation cannot trigger regeneration."""
    I.persist(F.ROOT / "validation_opened.json", {"time_ns": 100, "sources": {}})
    first = completed_run["requests"][0]
    (F.ROOT / "raw" / first["slot_id"] / "response.json").unlink()
    touched: list[str] = []
    monkeypatch.setattr(F, "_load_key", lambda: touched.append("credential"))
    monkeypatch.setattr(
        F, "make_live_llm", lambda *args, **kwargs: touched.append("client")
    )
    with pytest.raises(RuntimeError, match="response|barrier|validation"):
        F.run()
    assert touched == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha256", "wrong"),
        ("task_identity", "wrong"),
        ("local_seed", -1),
        ("budget", 64),
    ],
)
def test_panel_identity_detects_wrong_source_task_seed_or_budget(
    field: str, value: Any
) -> None:
    """Metrics from a different evaluation allocation cannot enter F1 analysis."""
    tasks = G.fresh_tasks("F1_UNIT_REVIEW", "train", 1)
    rows = [panel_row(B.SEED_SOURCE, task, 16301) for task in tasks]
    rows[0][field] = value
    with pytest.raises(RuntimeError, match="panel|trajectory"):
        F.verify_panel(rows, B.SEED_SOURCE, tasks, 16301, deployment=False)


def test_cap_rule_cannot_respond_to_objective_outcomes() -> None:
    """Reversing task scores cannot change a reliability-only cap decision."""
    groups = {
        "8000": {"length": 2, "eligible": 10, "auc": 0.0},
        "32000": {"length": 1, "eligible": 11, "auc": 100.0},
    }
    assert F.choose_cap(groups) == 32000
    groups["8000"]["auc"], groups["32000"]["auc"] = 1000.0, -1000.0
    assert F.choose_cap(groups) == 32000


@pytest.fixture
def prepared_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Prepare a real manifest using entirely unit-only tasks, evaluations and G1 data."""
    original_tasks = G.fresh_tasks
    tasks = {
        split: original_tasks("F1_UNIT_REVIEW", split, 1)
        for split in ("train", "validation")
    }
    monkeypatch.setattr(F, "ROOT", tmp_path)
    monkeypatch.setattr(
        G, "fresh_tasks", lambda stage, split, replicates: copy.deepcopy(tasks[split])
    )
    monkeypatch.setattr(F, "environment", dict)
    monkeypatch.setattr(
        F,
        "_g1_result",
        lambda: {
            "caps": {
                "8000": {"length": 2, "eligible": 10},
                "32000": {"length": 0, "eligible": 12},
            }
        },
    )
    monkeypatch.setattr(
        F,
        "evaluate_panel",
        lambda source, tasks, block, **kwargs: [
            panel_row(source, task, block) for task in tasks
        ],
    )
    return F.prepare()


def test_real_preflight_accepts_valid_manifest_and_rejects_altered_checksum(
    prepared_run: dict[str, Any],
) -> None:
    """The manifest itself is sealed independently of its embedded source hashes."""
    assert F.preflight() == prepared_run
    rewrite(F.ROOT / "freeze.json", "cap", 8000)
    with pytest.raises(RuntimeError, match="checksum"):
        F.preflight()


def test_preparation_resume_reuses_contexts_and_recovers_only_unstarted_seal(
    prepared_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted freeze can resume without evaluating another context or slot."""
    (F.ROOT / "freeze_sha256.json").unlink()
    assert F.prepare() == prepared_run
    (F.ROOT / "freeze.json").unlink()
    (F.ROOT / "freeze_sha256.json").unlink()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        """Existing verified context evidence must not be silently replaced."""
        raise AssertionError("context unexpectedly reevaluated")

    monkeypatch.setattr(F, "evaluate_panel", forbidden)
    recreated = F.prepare()
    assert recreated["contexts"] == prepared_run["contexts"]
    (F.ROOT / "freeze_sha256.json").unlink()
    I.persist(F.ROOT / "raw/started_1.json", {"status": "in_flight"})
    with pytest.raises(RuntimeError, match="unsealed|checksum"):
        F.prepare()


@pytest.mark.parametrize("mutation", ["cap", "parent", "schedule", "task_namespace"])
def test_semantic_manifest_changes_fail_even_with_updated_checksum(
    prepared_run: dict[str, Any], mutation: str
) -> None:
    """Preflight reconstructs the registered design instead of trusting a self-report."""
    modified = copy.deepcopy(prepared_run)
    if mutation == "cap":
        modified["cap"] = 8000
    elif mutation == "parent":
        modified["blocks"][0]["parent"] += "# changed\n"
    elif mutation == "schedule":
        modified["requests"].pop()
    else:
        modified["task_namespace"] = "F1"
    (F.ROOT / "freeze.json").write_text(json.dumps(modified))
    (F.ROOT / "freeze_sha256.json").write_text(
        json.dumps({"sha256": B.digest(modified)})
    )
    with pytest.raises(RuntimeError):
        F.preflight()


def test_complete_unit_run_and_resume_never_generate_after_validation(
    prepared_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The entire fixed-response lifecycle preserves all slots without any live tools."""
    from types import SimpleNamespace

    calls: list[dict[str, Any]] = []

    def client(**kwargs: Any) -> Any:
        """Return unchanged source for an explicitly synthetic integration check."""
        calls.append(kwargs)
        return SimpleNamespace(
            id=f"offline-full-{len(calls)}",
            model=G.MODEL,
            usage={},
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=B.SEED_SOURCE), finish_reason="stop"
                )
            ],
        )

    monkeypatch.setattr(F, "_load_key", lambda: None)
    monkeypatch.setattr(F, "make_live_llm", lambda *args, **kwargs: client)
    monkeypatch.setattr(F, "collect_metadata", lambda *args: None)
    F.run()
    assert len(calls) == 24
    assert len(E.read(F.ROOT / "results.json")["rows"]) == 24
    barrier = E.read(F.ROOT / "validation_opened.json")
    assert len(barrier["sources"]) == len(barrier["responses"]) == 24
    F.run()
    assert len(calls) == 24
    target = F.ROOT / "raw" / prepared_run["requests"][0]["slot_id"] / "validation.json"
    rows = E.read(target)
    rows[0]["metrics"]["auc"] += 1.0
    target.write_text(json.dumps(rows))
    with pytest.raises(RuntimeError, match="integrity"):
        F.summarize()
