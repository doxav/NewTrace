"""Item 1: batch_design + credit_horizon are now causally active in the adapter."""
from __future__ import annotations

import pytest

from opto.features.recursive_opt.tracebench import (
    TraceBenchTaskAdapter, _summarize_feedbacks,
)
from opto.features.recursive_opt.effects import effects_for, check_field_effects


def _adapter(inner_steps):
    a = TraceBenchTaskAdapter.__new__(TraceBenchTaskAdapter)
    a.inner_steps = inner_steps
    a.max_examples = 8
    return a


def test_batch_design_active_only_with_inner_steps() -> None:
    assert effects_for(_adapter(1))["batch_design"].active is True
    assert effects_for(_adapter(0))["batch_design"].active is False  # sampler needs inner training


def test_credit_horizon_now_active() -> None:
    fx = effects_for(_adapter(1))["credit_horizon"]
    assert fx.active is True
    assert set(fx.probe_values) == {"episode", "step", "truncated", "full"}


def test_batch_design_accepted_as_target_when_active() -> None:
    # check_field_effects must NOT raise for an active field at inner_steps>0
    report = check_field_effects(_adapter(1), ["batch_design"], allow_inactive=False)
    assert report.ok()


def test_batch_design_sampler_reorders() -> None:
    a = _adapter(1)
    inputs = ["short", "a very long and complex input string here", "medium one", "x"]
    infos = [0, 1, 2, 3]

    fb_in, fb_info = a._order_by_batch_design(inputs, infos, type("C", (), {"batch_design": "failure_balanced"})())
    assert fb_in[0] == max(inputs, key=len)        # hardest (longest) first

    cur_in, _ = a._order_by_batch_design(inputs, infos, type("C", (), {"batch_design": "curriculum"})())
    assert cur_in[0] == min(inputs, key=len)        # easiest (shortest) first

    rnd_in, _ = a._order_by_batch_design(inputs, infos, type("C", (), {"batch_design": "random"})())
    assert rnd_in == inputs                          # preserved


def test_credit_horizon_summarizer_varies_breadth() -> None:
    fbs = ["f0", "f1", "f2", "f3"]
    assert _summarize_feedbacks(fbs, "truncated") == "f0"
    assert _summarize_feedbacks(fbs, "full") == "f0 | f1 | f2 | f3"
    assert _summarize_feedbacks(fbs, "episode") == "f0 | f1 | f2"     # top-3
    assert "example[3]" in _summarize_feedbacks(fbs, "step")          # step-indexed, all
