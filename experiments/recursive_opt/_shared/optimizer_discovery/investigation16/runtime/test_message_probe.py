"""Consecutive user-message transmission through the actual local-only provider stack."""

import copy
from typing import Any

import pytest

from experiments.recursive_opt._shared.optimizer_discovery.investigation16.runtime import message_probe as M


def test_complete_consecutive_user_messages_and_native_settings_survive_transport() -> (
    None
):
    """Retain both full messages, Unicode and code delimiters without merging or dropping content."""
    request: dict[str, Any] = {
        "model": M.P.MODEL,
        "messages": [
            {
                "role": "user",
                "content": "CONTRAT début: propose(history, bounds, seed)\n"
                + "x" * 12000
                + "\nFIN CONTRAT",
            },
            {
                "role": "user",
                "content": "CURRENT SOURCE:\n```python\ndef propose(history, bounds, seed):\n    return [0.0]*len(bounds)\n```\n"
                + "y" * 6000
                + "\nFIN SOURCE",
            },
        ],
        "settings": {
            "temperature": 0.6,
            "top_p": 1.0,
            "max_tokens": 32000,
            "extra_body": {"reasoning": {"effort": "low"}},
            "timeout": 300,
            "seed": 16301,
            "num_retries": 0,
        },
    }
    before = copy.deepcopy(request)
    result = M.run_request(request)
    assert result["request_bodies"][0]["messages"] == request["messages"]
    assert result["request_bodies"][0]["max_tokens"] == 32000
    assert result["request_bodies"][0]["reasoning"] == {"effort": "low"}
    assert result["request_bodies"][0]["seed"] == 16301
    assert result["messages_exact"] is True
    assert result["requests"] == 1
    assert result["external_connections_blocked"] == 0
    assert request == before


def test_invalid_input_is_rejected_before_any_transport(monkeypatch: Any) -> None:
    """The local helper must not accept incomplete or endpoint-changing request input."""

    def forbidden(mode: str) -> dict[str, Any]:
        """Reject accidental transport execution while testing invalid input."""
        raise AssertionError("transport must not run for invalid probe input")

    monkeypatch.setattr(M.P, "run_case", forbidden)
    with pytest.raises(ValueError, match="two-user"):
        M.run_request(
            {
                "model": M.P.MODEL,
                "messages": [{"role": "user", "content": "one"}],
                "settings": {"api_base": "https://invalid.example"},
            }
        )
