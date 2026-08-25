"""Causal tests for canonical recursive-opt transport resilience."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from opto.features.recursive_opt import runmode
from opto.features.recursive_opt import spec as S
from opto.utils import auto_retry
from opto.utils.auto_retry import TransportRetryError, retry_with_exponential_backoff


def _canonical_profile_spec() -> dict[str, Any]:
    """Build a valid canonical spec with one explicit transport profile."""
    raw = json.loads(
        Path("artifacts/control_plane_v2/golden_specs/uc4_positive.normalized.json")
        .read_text(encoding="utf-8")
    )
    raw.pop("fingerprint", None)
    raw["llm_profiles"] = {
        "forward": {
            "provider": "fake",
            "model": "fake/exact",
            "max_tokens": 8,
            "request_timeout_s": 180,
            "transport_max_attempts": 3,
            "transport_base_delay_s": 1.0,
        }
    }
    raw["levels"][0]["llm_roles"]["forward"] = "forward"
    return raw


@pytest.mark.parametrize(
    "error",
    [
        ConnectionResetError("Connection reset by peer"),
        RuntimeError("Server disconnected without sending a response."),
    ],
)
def test_transient_transport_errors_retry_then_succeed(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """The two observed provider failures receive a bounded identical retry."""
    calls = 0

    def request() -> str:
        """Fail once with the selected transport error, then return text."""
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error
        return "ok"

    monkeypatch.setattr(auto_retry.time, "sleep", lambda _delay: None)

    assert retry_with_exponential_backoff(request, max_retries=3) == "ok"
    assert calls == 2


def test_transient_exception_cause_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider wrapper cannot hide a transient cause behind its own type."""
    calls = 0

    def request() -> str:
        """Wrap one transient cause before succeeding."""
        nonlocal calls
        calls += 1
        if calls == 1:
            try:
                raise ConnectionResetError("socket reset")
            except ConnectionResetError as cause:
                raise RuntimeError("provider wrapper") from cause
        return "ok"

    monkeypatch.setattr(auto_retry.time, "sleep", lambda _delay: None)

    assert retry_with_exponential_backoff(request, max_retries=3) == "ok"
    assert calls == 2


def test_transient_transport_exhaustion_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three failed transient attempts end with one explicit transport error."""
    calls = 0

    def request() -> None:
        """Always fail with the observed reset."""
        nonlocal calls
        calls += 1
        raise ConnectionResetError("Connection reset by peer")

    monkeypatch.setattr(auto_retry.time, "sleep", lambda _delay: None)

    with pytest.raises(TransportRetryError, match="after 3 attempts"):
        retry_with_exponential_backoff(request, max_retries=3)
    assert calls == 3


def test_application_error_is_not_retried() -> None:
    """Programming and validation errors retain a single provider attempt."""
    calls = 0

    def request() -> None:
        """Raise a non-transport application error."""
        nonlocal calls
        calls += 1
        raise TypeError("malformed optimizer output")

    with pytest.raises(TypeError, match="malformed optimizer output"):
        retry_with_exponential_backoff(request, max_retries=3, base_delay=0)
    assert calls == 1


def test_request_timeout_reaches_provider_and_returns_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider honoring the canonical timeout cannot block indefinitely."""
    import litellm
    from opto.utils.llm import LiteLLM

    observed: list[float] = []

    def blocking_provider(*_args: Any, **kwargs: Any) -> None:
        """Model a provider that blocks until its declared timeout."""
        timeout_s = float(kwargs["timeout"])
        observed.append(timeout_s)
        time.sleep(timeout_s)
        raise TimeoutError("provider request timed out")

    monkeypatch.setattr(litellm, "completion", blocking_provider)
    client = runmode.CompletionTokenCompatLLM(
        LiteLLM(model="fake/exact", max_retries=1, base_delay=0),
        "fake/exact",
        request_timeout_s=0.02,
    )
    started = time.monotonic()

    with pytest.raises(TransportRetryError, match="after 1 attempts"):
        client(messages=[{"role": "user", "content": "ping"}])

    assert observed == [0.02]
    assert time.monotonic() - started < 0.5


def test_canonical_transport_policy_ignores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical policy and fingerprint ignore legacy transport environment knobs."""
    import opto.utils.llm as llm_module

    constructed: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []

    class FakeLiteLLM:
        """Capture low-level retry construction and provider kwargs."""

        def __init__(self, **kwargs: Any) -> None:
            constructed.append(copy.deepcopy(kwargs))

        def __call__(self, **kwargs: Any) -> str:
            requests.append(copy.deepcopy(kwargs))
            return "ok"

    raw = _canonical_profile_spec()
    first = S.normalize_spec(raw)
    monkeypatch.setenv("RECURSIVE_OPT_LLM_MAX_RETRIES", "19")
    monkeypatch.setenv("RECURSIVE_OPT_LLM_BASE_DELAY_S", "11")
    monkeypatch.setenv("RECURSIVE_OPT_LLM_TIMEOUT_S", "999")
    second = S.normalize_spec(raw)
    monkeypatch.setattr(llm_module, "LiteLLM", FakeLiteLLM)
    client = runmode.make_live_llm(
        "fake/exact",
        max_retries=3,
        base_delay=1.0,
        request_timeout_s=180,
        allow_env_overrides=False,
        budget_resource=None,
    )

    assert client(messages=[{"role": "user", "content": "ping"}]) == "ok"
    assert first["fingerprint"] == second["fingerprint"]
    assert constructed == [
        {
            "model": "fake/exact",
            "cache": True,
            "max_retries": 3,
            "base_delay": 1.0,
            "retry_event_callback": None,
        }
    ]
    assert requests == [
        {"messages": [{"role": "user", "content": "ping"}], "timeout": 180}
    ]


def test_canonical_profile_transport_is_normalized_validated_and_causal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First-class transport fields persist and construct the guarded provider."""
    captured: list[dict[str, Any]] = []

    def fake_make_live_llm(model: str, **kwargs: Any) -> Any:
        """Capture the exact canonical provider-construction policy."""
        captured.append({"model": model, **kwargs})
        return lambda **_request: "ok"

    normalized = S.normalize_spec(_canonical_profile_spec())
    profile = normalized["llm_profiles"]["forward"]
    usage: dict[str, dict[str, float | int]] = {}
    monkeypatch.setattr(runmode, "make_live_llm", fake_make_live_llm)
    S._make_guarded_role_client(profile, "forward", None, usage, S._BudgetGuard({}))

    assert profile["request_timeout_s"] == 180
    assert profile["transport_max_attempts"] == 3
    assert profile["transport_base_delay_s"] == 1.0
    callback = captured[0].pop("retry_event_callback")
    assert callable(callback)
    assert captured == [{
        "model": "fake/exact",
        "max_retries": 3,
        "base_delay": 1.0,
        "request_timeout_s": 180,
        "allow_env_overrides": False,
        "budget_resource": None,
    }]
    callback("transient_failure", "connection_reset")
    callback("retry", "connection_reset")
    callback("recovered", None)
    assert usage["forward"] == {
        "transport_transient_failures": 1,
        "transport_retry_attempts": 1,
        "transport_recovered_requests": 1,
        "transport_exhausted_requests": 0,
        "transport_connection_resets": 1,
        "transport_server_disconnects": 0,
    }


def test_guarded_canonical_call_recovers_without_double_logical_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two transport attempts remain one guarded role call and one response usage."""
    import litellm

    requests: list[dict[str, Any]] = []

    def provider(*_args: Any, **kwargs: Any) -> Any:
        """Reset the first identical request and return usage on the second."""
        requests.append(copy.deepcopy(kwargs))
        if len(requests) == 1:
            raise ConnectionResetError("Connection reset by peer")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        )

    monkeypatch.setattr(litellm, "completion", provider)
    monkeypatch.setattr(auto_retry.time, "sleep", lambda _delay: None)
    profile = S.normalize_spec(_canonical_profile_spec())["llm_profiles"]["forward"]
    usage: dict[str, dict[str, float | int]] = {}
    guard = S._BudgetGuard({"eval_llm_calls": 1, "total_tokens": 10})
    client = S._make_guarded_role_client(profile, "forward", None, usage, guard)

    response = client(messages=[{"role": "user", "content": "same"}])

    assert response.choices[0].message.content == "ok"
    assert requests == [
        {
            "messages": [{"role": "user", "content": "same"}],
            "max_tokens": 8,
            "timeout": 180,
        },
        {
            "messages": [{"role": "user", "content": "same"}],
            "max_tokens": 8,
            "timeout": 180,
        },
    ]
    assert usage["forward"] == {
        "transport_transient_failures": 1,
        "transport_retry_attempts": 1,
        "transport_recovered_requests": 1,
        "transport_exhausted_requests": 0,
        "transport_connection_resets": 1,
        "transport_server_disconnects": 0,
        "calls": 1,
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    assert guard.used["eval_llm_calls"] == 1
    assert guard.used["total_tokens"] == 5


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("request_timeout_s", 0, "request_timeout_s"),
        ("transport_max_attempts", 0, "transport_max_attempts"),
        ("transport_base_delay_s", -1, "transport_base_delay_s"),
    ],
)
def test_canonical_profile_rejects_invalid_transport_policy(
    field: str, value: int, message: str
) -> None:
    """Invalid timeout and retry policy values fail during normalization."""
    raw = _canonical_profile_spec()
    raw["llm_profiles"]["forward"][field] = value

    with pytest.raises(ValueError, match=message):
        S.normalize_spec(raw)


def test_canonical_profile_rejects_duplicate_request_timeout() -> None:
    """A first-class timeout cannot conflict with request_params timeout."""
    raw = _canonical_profile_spec()
    raw["llm_profiles"]["forward"]["request_params"] = {"timeout": 30}

    with pytest.raises(ValueError, match="may not duplicate request_timeout_s"):
        S.normalize_spec(raw)
