"""The scripted regression runner must work whenever Trace-Bench is installed."""
import pytest

from opto.features.recursive_opt.tracebench import HAVE_TB


@pytest.mark.skipif(not HAVE_TB, reason="Trace-Bench not installed (real adapter required)")
def test_review_regression_returns_rows() -> None:
    from examples.recursive_opt_review_regression import run_review_regression
    rows = run_review_regression()
    assert [r["example"] for r in rows] == ["A", "C"]
    assert all(r["memory"]["episodes"] >= 1 for r in rows)
