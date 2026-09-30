"""Test detection of suspension without mistaking normal setup overhead for a gap."""

import pytest

from experiments.recursive_opt._shared.optimizer_discovery.investigation16.throughput.timing_repair import (
    calendar_gap,
)


def test_suspension_gap_and_normal_overhead() -> None:
    """Realtime spans include suspension while monotonic execution may omit it."""
    rows = [{"started_ns": 1_000_000_000, "completed_ns": 102_000_000_000}]
    assert calendar_gap({"wall_s": 1.0}, rows) == 100.0
    assert abs(calendar_gap({"wall_s": 101.001}, rows)) < 0.002


def test_missing_timestamp_evidence_is_not_a_healthy_timing() -> None:
    """An empty timing sample cannot silently pass the suspension audit."""
    with pytest.raises(ValueError, match="timestamp"):
        calendar_gap({"wall_s": 1.0}, [])
