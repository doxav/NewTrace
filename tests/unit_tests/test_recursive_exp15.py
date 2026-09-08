"""EXP-15 accounting, isolation and real production Trace integration checks."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from artifacts.optimizer_discovery import benchmark as B
from artifacts.optimizer_discovery import exp15 as E


def response(source: str | None) -> Any:
    """Return a completed model response; unit tests never replace live evidence."""
    return SimpleNamespace(
        id="unit",
        model=B.MANIFEST["model"]["model"],
        usage={"completion_tokens": 10, "prompt_tokens": 20, "total_tokens": 30},
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=source), finish_reason="stop"
            )
        ],
    )


def test_slots_invalid_responses_and_resume(tmp_path: Path, monkeypatch: Any) -> None:
    """Completed empty responses consume slots and cannot be replaced on resume."""
    calls = []

    def client(**kwargs: Any) -> Any:
        """Capture an actual request shape and simulate a completed empty response."""
        calls.append(kwargs)
        return response(None)

    exp = E.Experiment(tmp_path, "pilot", client=client)
    first = exp.proposal(701, "A1", 0, B.SEED_SOURCE)
    assert first["source"] == "" and first["completed"]
    assert exp.proposal(701, "A1", 0, B.SEED_SOURCE) == first and len(calls) == 1
    (tmp_path / "701/A1/slot_00/attempt_1.json").unlink()
    exp.proposal(701, "A1", 0, B.SEED_SOURCE)
    assert (tmp_path / "701/A1/slot_00/attempt_1.json").exists()
    assert len(calls) == 1
    assert calls[0]["extra_body"] == {"reasoning": {"effort": "low"}}
    assert calls[0]["num_retries"] == 0
    with pytest.raises(ValueError, match="slot"):
        exp.proposal(701, "A1", 2, B.SEED_SOURCE)


def test_retry_and_uncertain_inflight_are_distinct(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Transient attempts are bounded; an interrupted remote call is never blindly replayed."""
    monkeypatch.setattr(E.time, "sleep", lambda delay: None)
    calls = []

    def client(**kwargs: Any) -> Any:
        """Simulate transient failures with no hidden retries."""
        calls.append(1)
        raise RuntimeError("connection reset")

    exp = E.Experiment(tmp_path, "pilot", client=client)
    with pytest.raises(RuntimeError, match="transport"):
        exp.proposal(701, "A1", 0, B.SEED_SOURCE)
    assert len(calls) == 4
    slot = tmp_path / "701/A1/slot_01"
    slot.mkdir(parents=True)
    (slot / "attempt_1.json").write_text(json.dumps({"status": "in_flight"}))
    with pytest.raises(RuntimeError, match="uncertain"):
        exp.proposal(701, "A1", 1, B.SEED_SOURCE)
    assert len(calls) == 4


def test_production_trace_budgets_isolation_and_selection(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Both arms allocate identical tasks; production A2 performs exactly two real updates."""
    monkeypatch.setitem(B.MANIFEST, "inner_budget", 2)
    calls = []

    def client(**kwargs: Any) -> Any:
        """Return one valid source and one invalid source per arm."""
        calls.append(kwargs)
        return response(B.SEED_SOURCE if len(calls) % 2 else "def :")

    exp = E.Experiment(tmp_path, "pilot", client=client)
    with pytest.raises(RuntimeError, match="selection"):
        exp.holdout()
    exp.generate(701, "A1")
    exp.generate(701, "A2")
    assert len(calls) == 4
    trace = json.loads((tmp_path / "701/A2/trace.json").read_text())
    assert (
        trace["result"]["portable"]
        and trace["result"]["level_results"][0]["metadata"]["trace_optimize_path"]
    )
    assert not trace["plan"]["runtime"].get("test_mode", False)
    a1 = [
        json.loads(p.read_text())
        for p in sorted((tmp_path / "701/A1").glob("slot_*/request.json"))
    ]
    a2 = [
        json.loads(p.read_text())
        for p in sorted((tmp_path / "701/A2").glob("slot_*/request.json"))
    ]
    assert a1[0]["messages"] == a1[1]["messages"]
    assert len(a2[0]["messages"]) == 2
    assert "TRAINING FEEDBACK" in a2[0]["messages"][1]["content"]
    assert all("normalization" not in p["messages"][-1]["content"] for p in a2)
    assert not list((tmp_path / "cache").glob("*holdout*"))
    exp.select(701)
    pools = [
        json.loads((tmp_path / "701" / arm / "pool.json").read_text())
        for arm in ("A1", "A2")
    ]
    panels = [
        [(r["task_identity"], r["local_seed"]) for r in pool[0]["validation"]]
        for pool in pools
    ]
    assert panels[0] == panels[1]
    assert (
        json.loads((tmp_path / "701/selection.json").read_text())["A1"]["index"] == -1
    )
    exp.freeze_selections()
    exp.holdout()
    before = len(calls)
    exp.generate(701, "A2")
    assert len(calls) == before
    result = exp.analyze()
    assert len(result["per_seed"]) == 1
    assert result["contrasts"]["A2-A1"]["mean"] == 0
    assert exp.audit()["proposal_slots"] == 4
    assert exp.audit()["ineligible_generated_candidates"] == 2
    (tmp_path / "701/selection.json").write_text("{}")
    with pytest.raises(RuntimeError, match="modified"):
        exp.freeze_selections()


def test_freeze_and_hash_mismatch_fail_closed(tmp_path: Path) -> None:
    """An unregistered confirmation configuration cannot reach the provider."""
    with pytest.raises(RuntimeError, match="freeze"):
        E.preflight(tmp_path / "absent.json")
    E.persist(tmp_path / "evidence.json", {"x": 1})
    with pytest.raises(RuntimeError, match="overwrite"):
        E.persist(tmp_path / "evidence.json", {"x": 2})


def test_paired_analysis_preserves_negative_values_and_all_seeds() -> None:
    """Uncertainty resamples paired outer seeds and does not omit unfavorable deltas."""
    result = E.paired([1, 2, 3, 4, 5])
    assert result["mean"] == 3 and result["interpretation"] == "negative signal"
    assert E.paired([0, 0, 0, 0, 0])["interpretation"] == "no detectable difference"
    with pytest.raises(ValueError):
        E.paired([])
