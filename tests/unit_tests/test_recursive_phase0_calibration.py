"""Calibration bookkeeping tests; real-provider evidence is recorded separately."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from artifacts.optimizer_discovery.phase0 import run_request


def test_generation_records_invalid_without_retry(tmp_path: Path) -> None:
    """A successful provider response with invalid code is retained, never resampled."""
    calls: list[Any] = []

    def client(**kwargs: Any) -> Any:
        """Return one syntactically broken candidate with provider metadata."""
        calls.append(kwargs)
        return SimpleNamespace(
            id="test-id",
            model="test-model",
            created=1,
            usage={"total_tokens": 3},
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="```python\ndef :\n```"),
                    finish_reason="stop",
                )
            ],
        )

    result = run_request(client, "generation_1", tmp_path / "run", seeds=[0], budget=1)
    assert len(calls) == 1
    assert result["provider_status"] == "success"
    assert result["evaluations"][0]["valid"] is False
    assert result["evaluations"][0]["error"] == "syntax_error"
    assert (tmp_path / "run" / "optimizer.py").read_text() == "def :\n"
    assert (tmp_path / "run" / "result.json").exists()


def test_provider_failure_is_recorded_without_secrets(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Permanent provider failures preserve safe evidence without retrying poor results."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "private-unit-token")

    def client(**kwargs: Any) -> Any:
        """Raise a permanent error containing an inherited credential."""
        raise ValueError("rejected private-unit-token")

    result = run_request(client, "generation_1", tmp_path / "run", seeds=[0], budget=1)
    assert result["provider_status"] == "provider_error"
    assert result["evaluations"] == []
    assert "private-unit-token" not in (tmp_path / "run" / "result.json").read_text()


def test_transient_failures_keep_all_four_attempts(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Three bounded engineering retries do not erase failed attempts."""
    from artifacts.optimizer_discovery import phase0

    delays: list[float] = []
    monkeypatch.setattr(phase0.time, "sleep", delays.append)

    def client(**kwargs: Any) -> Any:
        """Simulate a persistent transient connection failure."""
        raise RuntimeError("connection reset")

    result = run_request(client, "generation_1", tmp_path / "run", seeds=[0], budget=1)
    assert len(result["attempts"]) == 4
    assert delays == [2, 4, 8]
    assert len(list((tmp_path / "run").glob("attempt_*.json"))) == 4


def test_separate_interface_smoke_uses_its_registered_prompt(tmp_path: Path) -> None:
    """The engineering task has separate identity without rewriting original requests."""
    from artifacts.optimizer_discovery import phase0

    calls: list[Any] = []

    def client(**kwargs: Any) -> Any:
        """Capture request identity, returning an invalid source without retries."""
        calls.append(kwargs)
        return SimpleNamespace(
            id="test",
            model="test",
            usage={},
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="def :"), finish_reason="stop"
                )
            ],
        )

    run_request(client, "interface_smoke", tmp_path / "run", seeds=[0], budget=1)
    assert calls[0]["messages"][0]["content"] == phase0.ENGINEERING_SPEC["prompt"]
    assert calls[0]["max_tokens"] == 3000 and calls[0]["seed"] == 17
    assert phase0.SPEC["live"]["requests"] == [
        "generation_1",
        "generation_2",
        "clean_smoke",
    ]


def test_readiness_settings_and_numeric_reasoning_usage(tmp_path: Path) -> None:
    """New registered settings retain reasoning counters without persisting reasoning text."""
    from artifacts.optimizer_discovery import phase0

    calls: list[Any] = []

    def client(**kwargs: Any) -> Any:
        """Return a truncated response with provider usage details."""
        calls.append(kwargs)
        return SimpleNamespace(
            usage={
                "completion_tokens": 8000,
                "completion_tokens_details": {"reasoning_tokens": 8000},
                "cost": 0.001,
            },
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None, reasoning="private reasoning"
                    ),
                    finish_reason="length",
                )
            ],
        )

    result = run_request(
        client, "pilot_low_8000_17", tmp_path / "run", seeds=[0, 1, 2], budget=8
    )
    assert calls[0]["max_tokens"] == 8000
    assert calls[0]["extra_body"] == {"reasoning": {"effort": "low"}}
    assert "reasoning_effort" not in calls[0]
    assert calls[0]["timeout"] == 300
    assert calls[0]["messages"][0]["content"] == phase0.PROMPT
    assert result["response_metadata"]["usage"]["reasoning_tokens"] == 8000
    assert result["response_metadata"]["usage"]["cost_usd"] == 0.001
    assert "private reasoning" not in (tmp_path / "run" / "response.json").read_text()
    assert result["parse_status"] == "invalid" and len(calls) == 1


@pytest.mark.parametrize(
    "body,responsive,valid",
    [
        ("return [0, 0]", False, True),
        ("return [len(history) / 2, 0]", False, True),
        ("return min(history, key=lambda row: row['value'])['x']", True, True),
        ("return [99, 0]", False, False),
    ],
)
def test_history_probe_measures_values_not_length(
    body: str, responsive: bool, valid: bool
) -> None:
    """History sensitivity requires observed legal differences at fixed history length."""
    from artifacts.optimizer_discovery.phase0 import history_probe

    result = history_probe("def propose(history, bounds, seed):\n    " + body)
    assert result["valid"] is valid
    assert result["responsive"] is responsive
    assert len(result["pairs"]) == 6


def test_readiness_gate_boundaries_and_complete_batch() -> None:
    """Missing, duplicated, collapsed or insufficiently valid batches cannot turn green."""
    from artifacts.optimizer_discovery.phase0 import readiness_gate

    rows = [
        {"label": str(i), "generation_valid": i < 9, "history_responsive": i < 8}
        for i in range(10)
    ]
    menu = {"effective_menu_size": 2, "behavior_equivalence_known": True}
    expected = [str(i) for i in range(10)]
    assert readiness_gate(rows, menu, expected, phase="confirmation")["passed"]
    assert not readiness_gate(rows[:-1], menu, expected, phase="confirmation")["passed"]
    assert not readiness_gate(rows + rows[:1], menu, expected, phase="confirmation")[
        "passed"
    ]
    assert not readiness_gate(
        rows, {**menu, "effective_menu_size": 1}, expected, phase="confirmation"
    )["passed"]
    rows[0]["generation_valid"] = False
    assert not readiness_gate(rows, menu, expected, phase="confirmation")["passed"]
    with pytest.raises(ValueError, match="phase"):
        readiness_gate(rows, menu, expected, phase="unknown")


def test_readiness_request_rejects_fixture_changes(tmp_path: Path) -> None:
    """The preregistered fixture cannot be overridden at the call site."""

    def client(**kwargs: Any) -> Any:
        """Fail if invalid fixture arguments ever reach the provider."""
        raise AssertionError("provider must not be called")

    with pytest.raises(ValueError, match="fixture"):
        run_request(
            client, "pilot_default_8000_17", tmp_path / "run", seeds=[9], budget=1
        )
    assert not (tmp_path / "run").exists()


def test_summary_uses_executed_behavior_and_requires_telemetry(tmp_path: Path) -> None:
    """Complete valid executions can still fail for collapse or absent token evidence."""
    from artifacts.optimizer_discovery import phase0

    def client(**kwargs: Any) -> Any:
        """Generate a legal best-so-far proposer through the real canonical evaluator."""
        return SimpleNamespace(
            usage={"completion_tokens": 100},
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "def propose(history, bounds, seed):\n"
                            "    return min(history, key=lambda r: r['value'])['x'] if history else [0, 0]\n"
                        )
                    ),
                    finish_reason="stop",
                )
            ],
        )

    label = "pilot_default_8000_17"
    result = run_request(client, label, tmp_path / label, seeds=[0, 1, 2], budget=8)
    assert result["history_probe"]["responsive"]
    for seed in (18, 19):
        cloned_label = f"pilot_default_8000_{seed}"
        (tmp_path / cloned_label).mkdir()
        (tmp_path / cloned_label / "result.json").write_text(
            json.dumps({**result, "label": cloned_label})
        )
    summary = phase0.summarize_readiness(tmp_path, "default_8000", "pilot")
    assert summary["complete"] and summary["valid_count"] == 3
    assert summary["history_responsive_count"] == 3
    assert summary["menu_evidence"]["effective_menu_size"] == 1
    assert not summary["passed"]
    result["response_metadata"]["usage"] = {}
    (tmp_path / label / "result.json").write_text(json.dumps(result))
    summary = phase0.summarize_readiness(tmp_path, "default_8000", "pilot")
    assert summary["valid_count"] == 2 and not summary["passed"]
