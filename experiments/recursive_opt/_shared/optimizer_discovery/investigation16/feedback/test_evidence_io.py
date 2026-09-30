"""Prospective exact-evidence requirements, distinct from frozen G1 recording."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import generation as G


def test_benign_source_and_request_remain_exact(tmp_path: Path) -> None:
    """Natural task-specific wording is data, never a credential-shaped substring."""
    source = "# task-specific heuristic\ndef propose(history, bounds, seed):\n    return [0.0]\n"
    payload = {"source": source, "source_sha256": B.source_hash(source)}
    path = tmp_path / "source.json"
    I.persist(path, payload)
    assert E.read(path) == payload
    assert json.loads(path.read_text())["source"] == source
    I.persist(path, payload)
    with pytest.raises(RuntimeError, match="overwrite"):
        I.persist(path, {"source": "changed"})


@pytest.mark.parametrize(
    "key", ["sk-or-v1-FAKE_TEST_12345678", "sk-proj-A1b2C3d4E5f6G7h8I9j0K1l2M3n4"]
)
def test_credential_shaped_text_is_rejected_without_echo(
    tmp_path: Path, key: str
) -> None:
    """Reject recognizable synthetic keys without silently rewriting scientific fields."""
    with pytest.raises(RuntimeError, match="credential") as error:
        I.persist(tmp_path / "bad.json", {"content": key})
    assert key not in str(error.value)
    assert not (tmp_path / "bad.json").exists()


def test_private_active_key_is_checked_without_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An active synthetic credential is rejected even without a recognizable prefix."""
    fake = "unit-secret-marker-90210"
    monkeypatch.setenv("OPENROUTER_API_KEY", fake)
    with pytest.raises(RuntimeError, match="credential") as error:
        I.persist(tmp_path / "bad.json", {"content": fake})
    assert fake not in str(error.value)


def test_reused_slot_recorder_preserves_raw_candidate_and_restores_module(
    tmp_path: Path,
) -> None:
    """The existing bounded transport path can use an exact recorder in this process."""
    original = G.E
    source = "# task-specific heuristic\ndef propose(history, bounds, seed):\n    return [0.0]\n"
    calls: list[dict[str, Any]] = []

    def client(**kwargs: Any) -> Any:
        """Return a unit response containing benign text formerly over-redacted."""
        calls.append(kwargs)
        return SimpleNamespace(
            id="unit",
            model=G.MODEL,
            usage={},
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=source), finish_reason="stop"
                )
            ],
        )

    request = {
        "messages": [{"role": "user", "content": "task-specific"}],
        "settings": {},
    }
    response = I.complete_slot(tmp_path / "slot", request, client)
    assert response["source"] == source
    assert E.read(tmp_path / "slot/response.json")["source"] == source
    assert E.read(tmp_path / "slot/request.json") == request
    assert G.E is original
    I.complete_slot(tmp_path / "slot", request, client)
    assert len(calls) == 1 and G.E is original


def test_lossless_compression_and_restore_after_failure(tmp_path: Path) -> None:
    """Large evidence remains exact, and failed calls do not change other module users."""
    payload = {"text": "task-specific " * 40000}
    path = tmp_path / "large.json"
    I.persist(path, payload)
    assert not path.exists() and path.with_suffix(".json.gz").exists()
    assert E.read(path) == payload
    I.persist(path, payload)
    original = G.E

    def fail(**kwargs: Any) -> Any:
        """Raise a nontransient unit failure with no provider or credential access."""
        raise ValueError("unit nontransient failure")

    with pytest.raises(RuntimeError, match="transport"):
        I.complete_slot(
            tmp_path / "failed_slot", {"messages": [], "settings": {}}, fail
        )
    assert G.E is original


def test_concurrent_slot_adapter_rejects_second_request(tmp_path: Path) -> None:
    """A second adapter cannot inherit the temporary recorder or issue another call."""
    entered, release = threading.Event(), threading.Event()

    def client(**kwargs: Any) -> Any:
        """Hold a unit call open until the concurrent rejection has been tested."""
        entered.set()
        if not release.wait(timeout=10):
            raise RuntimeError("unit synchronization timed out")
        return SimpleNamespace(
            id="unit-thread",
            model=G.MODEL,
            usage={},
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=B.SEED_SOURCE), finish_reason="stop"
                )
            ],
        )

    original = G.E
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            I.complete_slot,
            tmp_path / "first",
            {"messages": [], "settings": {}},
            client,
        )
        try:
            assert entered.wait(timeout=10)
            with pytest.raises(RuntimeError, match="concurrency"):
                I.complete_slot(
                    tmp_path / "second", {"messages": [], "settings": {}}, client
                )
            assert not (tmp_path / "second").exists()
        finally:
            release.set()
        assert future.result(timeout=10)["completed"] and G.E is original
