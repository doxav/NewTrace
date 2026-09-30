"""Check the single first-point intervention and immutable reuse of B1 controls."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.investigation16.benchmark import first_point as P
from opto.features.recursive_opt.optimizer_program import propose_point


def test_source_changes_only_the_empty_history_branch() -> None:
    """Preserve the exact seed text except for the one declared conditional split."""
    expected = B.SEED_SOURCE.replace(
        "    if not history or rng.random() < 0.25:\n",
        "    if not history:\n"
        "        return [(low + high) / 2 for low, high in bounds]\n"
        "    if rng.random() < 0.25:\n",
    )
    assert P.variant_source() == expected
    assert B.source_status(expected) == "valid"
    original = ast.parse(B.SEED_SOURCE).body[0]
    variant = ast.parse(expected).body[0]
    assert ast.dump(original.args) == ast.dump(variant.args)
    assert P.variant_source().count("def propose(") == 1


@pytest.mark.parametrize("dimension", [2, 4])
def test_nonempty_history_equivalence_and_midpoint_through_subprocess(
    dimension: int,
) -> None:
    """Exercise both policies through the real boundary with deterministic replay."""
    bounds = [[-5.0 + index / 10, 5.0 + index / 10] for index in range(dimension)]
    variant = P.variant_source()
    first = propose_point(variant, [], bounds, 16201)
    assert first.valid
    assert first.point == [(low + high) / 2 for low, high in bounds]
    for seed in [16201, 16202]:
        for length in [1, 7, 31]:
            history = [
                {"x": [index / 10] * dimension, "value": float(length - index)}
                for index in range(length)
            ]
            original = propose_point(B.SEED_SOURCE, history, bounds, seed)
            altered = propose_point(variant, history, bounds, seed)
            assert original.valid and altered.valid
            assert original.point == altered.point


def test_freeze_reuses_every_control_and_rejects_changed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Freezing reads the 96 existing controls without executing an objective."""

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        """Reject accidental seed reevaluation or early variant execution."""
        raise AssertionError("freeze must not evaluate any policy")

    monkeypatch.setattr(B, "evaluate", forbidden)
    monkeypatch.setattr(B, "normalization", forbidden)
    frozen = P.prepare(tmp_path)
    assert len(frozen["allocations"]) == 96
    assert len({entry["id"] for entry in frozen["allocations"]}) == 96
    assert P.preflight(tmp_path) == frozen
    assert not (tmp_path / "raw").exists()
    with pytest.raises(ValueError, match="every allocated"):
        P.analyze(tmp_path)
    damaged = copy.deepcopy(frozen)
    damaged["variant_source"] += "\n"
    (tmp_path / "freeze.json").write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="frozen"):
        P.preflight(tmp_path)


def test_analysis_requires_every_row_and_retains_invalidity() -> None:
    """A missing or invalid allocation cannot turn into a numeric favorable mean."""
    row = E.read(P.ROOT / "raw/central/seed/16201_00.json")
    invalid = {
        **row,
        "valid": False,
        "status": "timeout",
        "metrics": None,
        "objective_calls": 0,
        "observations": [],
    }
    summary = P.compare([row, row], [row, invalid])
    assert summary["variant"]["n"] == 2
    assert summary["variant"]["invalid"] == 1
    assert summary["delta"] is None
    with pytest.raises(ValueError, match="paired"):
        P.compare([row], [row, invalid])


def test_completed_variant_is_reused_without_candidate_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume preserves completed row bytes and checks their source identity."""
    frozen = P.prepare(tmp_path)
    entry = frozen["allocations"][0]
    row = E.read(P.ROOT / entry["control_path"])
    fixture = {**row, "source_sha256": frozen["variant_source_sha256"]}
    path = P._path(tmp_path, entry)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(fixture))
    original = path.read_bytes()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        """Prohibit replacing any completed trajectory during resume."""
        raise AssertionError("completed trajectory was executed again")

    monkeypatch.setattr(B, "evaluate", forbidden)
    assert P._evaluate(tmp_path, frozen, entry) == fixture
    assert path.read_bytes() == original
    fixture["source_sha256"] = "wrong"
    path.write_text(json.dumps(fixture))
    with pytest.raises(ValueError, match="frozen paired"):
        P._evaluate(tmp_path, frozen, entry)
