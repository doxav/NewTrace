"""Tests for the promoted budget conversion (Item 4) and multi-seed runner (Item 3)."""
from __future__ import annotations

import pytest

from opto.features.recursive_opt.budget import (
    make_budget, budget_to_spec_dict, RecursiveOptBudget,
)
from opto.features.recursive_opt.experiments import (
    run_spec_repeated, RepeatedResult, seed_everything,
)


# ----------------------------- Item 4 -------------------------------------- #
def test_make_budget_maps_spec_keys() -> None:
    d = {"wall_time_s": 100, "optimizer_llm_calls": 8, "eval_llm_calls": 24,
         "candidates": 16, "on_exceed": "raise"}
    b = make_budget(d)
    assert b.max_wall_time_s == 100
    assert b.max_optimizer_llm_calls == 8
    assert b.max_eval_llm_calls == 24
    assert b.max_candidates == 16
    assert b.stop_policy == "raise"


def test_make_budget_passthrough_and_empty() -> None:
    b = RecursiveOptBudget(max_candidates=5)
    assert make_budget(b) is b          # instance passes through unchanged
    assert make_budget(None) is None
    assert make_budget({}) is None


def test_make_budget_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown budget keys"):
        make_budget({"walltime": 100})   # typo must surface


def test_budget_round_trips_losslessly() -> None:
    d = {"wall_time_s": 100, "candidates": 16}
    assert budget_to_spec_dict(make_budget(d)) == d
    assert make_budget(d).to_spec_dict() == d   # method form


def test_configure_budget_delegates_to_make_budget() -> None:
    # spec.py._configure_budget must use the same mapping (DRY)
    from opto.features.recursive_opt import spec as S
    b = S._configure_budget({"budget": {"wall_time_s": 42}})
    assert b.max_wall_time_s == 42
    # override beats spec["budget"] without mutating the spec
    spec = {"budget": {"wall_time_s": 42}}
    b2 = S._configure_budget(spec, override={"wall_time_s": 7})
    assert b2.max_wall_time_s == 7
    assert spec["budget"] == {"wall_time_s": 42}  # unmutated


# ----------------------------- Item 3 -------------------------------------- #
def test_repeated_result_excludes_invalid_from_mean() -> None:
    rr = RepeatedResult(level_id="x", scores=[0.9, -1.0, 0.8])
    assert rr.n_valid() == 2
    assert rr.n_invalid() == 1
    assert rr.mean() == pytest.approx(0.85)
    assert rr.best() == pytest.approx(0.9)


def test_repeated_result_all_invalid() -> None:
    rr = RepeatedResult(level_id="x", scores=[-1.0, -1.0])
    assert rr.mean() is None and rr.best() is None and rr.n_valid() == 0


def test_seed_everything_is_deterministic() -> None:
    import random
    seed_everything(123)
    a = [random.random() for _ in range(3)]
    seed_everything(123)
    b = [random.random() for _ in range(3)]
    assert a == b


def test_run_spec_repeated_aggregates_and_isolates(monkeypatch) -> None:
    """run_spec_repeated must seed, isolate memory per seed, and aggregate.

    We stub run_spec to avoid needing an LLM: it returns a score derived from the
    memory_root suffix so we can assert isolation + aggregation deterministically.
    """
    seen_roots = []

    def fake_run_spec(spec, *, optimizer=None, trainer=None, budget=None):
        seen_roots.append(spec["memory_root"])
        # score 1.0 for every seed; one level "L"
        return {"results": {"L": {"score": 1.0, "artifact": "code", "artifact_id": "a1"}},
                "levels": {}, "memory": None}

    import opto.features.recursive_opt.spec as S
    monkeypatch.setattr(S, "run_spec", fake_run_spec)

    spec = {"memory_root": "./mem_t", "levels": [{"id": "L", "surface": "config"}]}
    out = run_spec_repeated(spec, seeds=[0, 1, 2], level_id="L")
    rr = out["L"]
    assert rr.n_valid() == 3
    assert rr.mean() == pytest.approx(1.0)
    # each seed used a DISTINCT, suffixed memory root
    assert seen_roots == ["./mem_t_seed0", "./mem_t_seed1", "./mem_t_seed2"]


def test_run_spec_repeated_survives_a_bad_seed(monkeypatch) -> None:
    calls = {"n": 0}

    def flaky_run_spec(spec, *, optimizer=None, trainer=None, budget=None):
        calls["n"] += 1
        if "seed1" in spec["memory_root"]:
            raise RuntimeError("boom")
        return {"results": {"L": {"score": 0.7, "artifact": "x", "artifact_id": None}},
                "levels": {}, "memory": None}

    import opto.features.recursive_opt.spec as S
    monkeypatch.setattr(S, "run_spec", flaky_run_spec)

    spec = {"memory_root": "./mem_b", "levels": [{"id": "L", "surface": "config"}]}
    out = run_spec_repeated(spec, seeds=[0, 1, 2], level_id="L")
    rr = out["L"]
    assert calls["n"] == 3                 # all seeds attempted
    assert rr.n_valid() == 2               # the good ones counted
    assert len(rr.errors) == 1             # the bad seed recorded, not raised
    assert rr.mean() == pytest.approx(0.7)
