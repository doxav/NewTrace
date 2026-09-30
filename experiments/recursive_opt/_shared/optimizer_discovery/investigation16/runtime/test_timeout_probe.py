"""Local HTTP transport tests; no generative provider is contacted."""

from experiments.recursive_opt._shared.optimizer_discovery.investigation16.runtime import timeout_probe as P


def test_wrapper_default_reaches_http_transport() -> None:
    """The existing wrapper forwards its 300-second default to actual HTTPX operations."""
    result = P.run_case("fast")
    assert result["outcome"] == "completed"
    assert result["requests"] == 1
    assert result["timeout_extensions"] == [
        {"connect": 300.0, "read": 300.0, "write": 300.0, "pool": 300.0}
    ]
    assert result["request_bodies"][0]["reasoning"] == {"effort": "low"}
    assert result["request_bodies"][0]["max_tokens"] == 32000


def test_arriving_bytes_can_exceed_total_timeout() -> None:
    """Nonstreaming reads continue beyond the total configured timeout while chunks arrive."""
    result = P.run_case("drip")
    assert result["outcome"] == "completed"
    assert result["elapsed_monotonic_s"] > 1.0
    assert result["requests"] == 1
    assert result["timeout_extensions"][0]["read"] == 0.5


def test_stalled_read_times_out_without_hidden_retry() -> None:
    """The same transport rejects an idle body rather than ignoring the timeout entirely."""
    result = P.run_case("stall")
    assert result["outcome"] == "error"
    assert any("timeout" in name.lower() for name in result["error_chain"])
    assert 0.4 < result["elapsed_monotonic_s"] < 3.0
    assert result["requests"] == 1
