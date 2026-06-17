"""Item 2: numeric/Bayesian optimizers + routing policy."""
from __future__ import annotations

import pytest

from opto.features.recursive_opt.numeric_optimizers import (
    OptunaOptimizer, LeastSquaresOptimizer, route_optimizers,
    field_search_space, is_numeric_field,
)
from opto.trace.nodes import node


def _params():
    return [node("x", trainable=True, name="cfg")]


def test_field_search_space_omits_text_fields() -> None:
    space = field_search_space(["starting_artifact", "batch_design", "batch_size"])
    assert "starting_artifact" not in space          # text -> generative only
    assert space["batch_design"][0] == "cat"
    assert space["batch_size"][0] == "int"


def test_is_numeric_field() -> None:
    assert is_numeric_field("batch_size") and is_numeric_field("trace_type")
    assert not is_numeric_field("starting_artifact")


def test_optuna_learns_categorical_and_integer_optimum() -> None:
    def evaluate(a):
        s = 0.6 if a.get("batch_design") == "failure_balanced" else 0.0
        return s + 0.4 * (a.get("batch_size", 1) / 8.0)

    opt = OptunaOptimizer(_params(), evaluate=evaluate,
                          space=field_search_space(["batch_design", "batch_size"]),
                          max_trials=30)
    best = opt.step()
    assert best["batch_design"] == "failure_balanced"
    assert best["batch_size"] == 8
    assert max(s for _, s in opt.history) == pytest.approx(1.0)
    assert len(opt.history) == 30          # full progress trace available


def test_least_squares_finds_integer_optimum() -> None:
    ls = LeastSquaresOptimizer(_params(), evaluate=lambda a: a.get("batch_size", 1) / 8.0,
                               space=field_search_space(["batch_size"]), max_trials=15,
                               target=1.0)
    ls.step()
    assert ls.best_assignment["batch_size"] == 8


def test_optimizer_backward_is_noop() -> None:
    opt = OptunaOptimizer(_params(), evaluate=lambda a: 0.0,
                          space=field_search_space(["batch_size"]), max_trials=1)
    assert opt.backward() is None          # numeric: ignores graph feedback


def test_routing_splits_and_orders() -> None:
    r = route_optimizers(["starting_artifact", "batch_design", "batch_size"])
    assert r["numeric_fields"] == ["batch_design", "batch_size"]
    assert r["text_fields"] == ["starting_artifact"]
    assert r["order"] == "numeric_then_text"

    assert route_optimizers(["starting_artifact"])["order"] == "text_only"
    assert route_optimizers(["batch_size"])["order"] == "numeric_only"


def test_routing_respects_custom_policy() -> None:
    r = route_optimizers(["batch_size", "starting_artifact"],
                         policy={"order": "text_then_numeric", "numeric_optimizer": "least_squares"})
    assert r["order"] == "text_then_numeric"
    assert r["numeric_optimizer"] == "least_squares"
