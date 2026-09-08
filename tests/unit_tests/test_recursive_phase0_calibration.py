"""Calibration bookkeeping tests; real-provider evidence is recorded separately."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
