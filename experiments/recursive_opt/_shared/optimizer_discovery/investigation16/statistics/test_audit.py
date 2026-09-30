"""Small calculation checks for the retrospective diagnostic script."""

import pytest

from experiments.recursive_opt._shared.optimizer_discovery.investigation16.statistics.audit import (
    panel_auc,
    ranks,
    response_text_bytes,
    signflip_p,
    spearman,
)


def test_empty_response_retained_as_zero_bytes() -> None:
    """A completed response can have null content when reasoning exhausts its cap."""
    assert response_text_bytes(None) == 0
    assert response_text_bytes("é") == 2


def test_rank_ties_and_reversal() -> None:
    """Equal scores share average ranks and reversals are negative correlations."""
    assert ranks([3.0, 1.0, 1.0]) == [3.0, 1.5, 1.5]
    assert spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0
    assert spearman([1.0, 1.0], [2.0, 3.0]) is None


def test_exact_permutation_and_zero_effect() -> None:
    """Five same-direction pairs cannot reach .05 in a two-sided sign-flip test."""
    assert signflip_p([-1.0] * 5) == 2 / 32
    assert signflip_p([0.0] * 5) == 1.0
    with pytest.raises(ValueError, match="finite"):
        signflip_p([float("nan")])


def test_aggregation_keeps_strata_equal_and_rejects_invalidity() -> None:
    """Different stratum counts cannot change weights or silently drop invalidity."""
    rows = [
        {"valid": True, "stratum": "a", "metrics": {"auc": 1.0}},
        {"valid": True, "stratum": "b", "metrics": {"auc": 3.0}},
        {"valid": True, "stratum": "b", "metrics": {"auc": 5.0}},
    ]
    assert panel_auc(rows) == 2.5
    rows[0]["valid"] = False
    with pytest.raises(ValueError, match="invalid"):
        panel_auc(rows)
