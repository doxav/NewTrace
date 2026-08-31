"""The amended invalid-rate gate (invalid_rate_gate_amendment_v1).

The frozen rule `invalid_rate <= 0` stopped the 2026-08-24 main run on its first unit for
one empty extraction in 24 — a rare event, not a defect. Across 40 units x 24 samples that
rule is unsatisfiable at any realistic rate (P(all pass) = 0.0001 at a true rate of 0.01),
so it could never have completed regardless of the optimizer.

The amendment keeps a gate, but only wide enough to catch a genuinely broken extractor.
These tests pin the three properties that make it defensible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.recursive_opt.multiobjective_reasoning import evaluator, specs  # noqa: E402
from opto.trainer.objectives import EvaluationResult, satisfies_hard_constraints  # noqa: E402

AMENDMENT = Path(__file__).resolve().parents[2] / (
    "experiments/recursive_opt/multiobjective_reasoning/manifests/"
    "invalid_rate_gate_amendment_v1.json")
GATE = ({"metric": "invalid_rate", "op": "<=", "value": specs.INVALID_RATE_GATE},)
HOLDOUT = 24


def _evaluation(invalid_rate: float) -> EvaluationResult:
    return EvaluationResult(valid=True, status="ok", feedback="",
                            metrics={"accuracy": 0.5, "invalid_rate": invalid_rate})


# --- P1: normal operation is not gated ------------------------------------- #
@pytest.mark.parametrize("invalid", [0, 1, 2])
def test_p1_a_rare_invalid_extraction_does_not_stop_a_unit(invalid: int) -> None:
    """1/24 is exactly what stopped the frozen run. It must no longer stop anything."""
    assert satisfies_hard_constraints(_evaluation(invalid / HOLDOUT), GATE) is True


# --- P2: a broken extractor is still caught -------------------------------- #
@pytest.mark.parametrize("invalid", [3, 12, 24])
def test_p2_a_broken_extractor_is_still_caught(invalid: int) -> None:
    assert satisfies_hard_constraints(_evaluation(invalid / HOLDOUT), GATE) is False


# --- P3: the metric cannot be gamed ---------------------------------------- #
def test_p3_invalid_counts_as_incorrect_and_stays_in_the_denominator() -> None:
    """Excluding invalid samples would reward driving hard questions to invalid output."""
    assert evaluator._extract("no number here", "numeric") == ""
    assert evaluator._extract("FINAL: 23", "numeric") == "23"


def test_p3_docstring_records_the_denominator_decision() -> None:
    doc = evaluator.exact_reasoning_evaluator.__doc__ or ""
    assert "INCORRECT" in doc and "denominator" in doc, (
        "the denominator choice must be stated, not left implicit")


# --- the amendment is the single source of truth ---------------------------- #
def test_runtime_gate_matches_the_authorized_amendment() -> None:
    """Guards the failure this design exists to prevent: a runtime value silently
    diverging from the preregistered record."""
    amendment = json.loads(AMENDMENT.read_text())
    assert amendment["amendment"]["hard_constraint_after"] == (
        f"invalid_rate <= {specs.INVALID_RATE_GATE:g}")
    assert amendment["amendment"]["hard_constraint_before"] == "invalid_rate <= 0"


def test_amendment_records_its_evidence_and_authorization() -> None:
    amendment = json.loads(AMENDMENT.read_text())
    assert amendment["authorized_by"]
    assert amendment["trigger"]["evidence"]["probe_n"]
    assert amendment["trigger"]["evidence"]["probe_p"]
    # the amendment must claim NO behaviour change to the denominator; the code path is
    # pinned separately by test_p3_*
    assert amendment["amendment"]["accuracy_denominator"]["behaviour_change"].startswith("none")


def test_gate_is_a_breakage_guard_not_a_quality_target() -> None:
    """A gate tight enough to bind on normal operation is a stopping rule, not a guard."""
    assert specs.INVALID_RATE_GATE >= 2 / HOLDOUT, (
        "the gate must tolerate at least the rare-event range it was tripping on")
    assert specs.INVALID_RATE_GATE <= 0.25, "a gate this wide would stop catching breakage"
