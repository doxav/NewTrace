"""Guards for validity defects found in the recursive_opt assessment (D1/D3/D4/D6).

Each test pins a behaviour whose absence previously produced a *plausible-looking
number* rather than an error — the failure mode that makes meta-optimization
results untrustworthy.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

from opto.features.recursive_opt import spec as S
from opto.features.recursive_opt import tracebench as TB
from opto.features.recursive_opt.levels import (
    DEFAULT_INVALID_FLOOR,
    INVALID_CONFIG_SCORE,
    LevelConfig,
    is_invalid_score,
)
from opto.features.recursive_opt.memory import MemoryLite
from opto.trainer.objectives import EvaluationResult

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))


class _RejectingAdapter:
    """Adapter that rejects every config the way the real one rejects a trainer."""

    status = "rejecting test adapter"

    def run_task(self, cfg, task_id):
        return INVALID_CONFIG_SCORE, f"[test] {task_id}: configuration rejected"


@pytest.fixture()
def _restore_adapter():
    previous = TB.current_task_adapter()
    yield
    TB.register_task_adapter(previous)


# --------------------------------------------------------------------------- #
# D1 — the invalidity sentinel must never be reported as a score
# --------------------------------------------------------------------------- #
def test_d1_invalid_sentinel_is_floored_without_configured_clip(_restore_adapter) -> None:
    TB.register_task_adapter(_RejectingAdapter())
    score, feedback = S.make_scored_task_runner(None)(LevelConfig(), "t1")

    assert score == DEFAULT_INVALID_FLOOR
    assert score > INVALID_CONFIG_SCORE
    assert '"invalid": true' in feedback


def test_d1_invalid_sentinel_respects_the_configured_clip_floor(_restore_adapter) -> None:
    TB.register_task_adapter(_RejectingAdapter())
    score, _ = S.make_scored_task_runner({"clip": [-0.25, 1.0]})(LevelConfig(), "t1")

    assert score == -0.25


def test_d1_floored_invalidity_cannot_destroy_a_mean_or_a_promoted_prior(_restore_adapter) -> None:
    TB.register_task_adapter(_RejectingAdapter())
    score, _ = S.make_scored_task_runner(None)(LevelConfig(), "t1")

    assert (0.2 + score) / 2 > -1.0  # a single invalid task cannot dominate the mean

    memory = MemoryLite(root=tempfile.mkdtemp(), promotion_min_support=1)
    for _ in range(2):
        memory.record(level="O1", cfg={}, family="f", score=score, feedback="x")
    assert memory.family_prior("f").best_score == DEFAULT_INVALID_FLOOR


def test_d1_detector_separates_sentinels_from_real_scores() -> None:
    assert is_invalid_score(INVALID_CONFIG_SCORE)
    assert is_invalid_score(float("-inf")) and is_invalid_score(float("nan"))
    assert not is_invalid_score(-1.0) and not is_invalid_score(0.0)


# --------------------------------------------------------------------------- #
# D3 — arms scored on different task sets are not comparable
# --------------------------------------------------------------------------- #
_FAMILIES = {"gsm8k": ["internal:multiobjective_gsm8k"], "qasper": ["hf:qasper"]}


def test_d3_scored_task_ids_distinguishes_family_policy_from_prior() -> None:
    policy = S.scored_task_ids(
        {"id": "o2", "surface": "family_policy", "family": "*"}, _FAMILIES
    )
    prior = S.scored_task_ids({"id": "o3", "surface": "prior", "family": "*"}, _FAMILIES)

    assert sorted(policy) == sorted(
        ["internal:multiobjective_gsm8k", "hf:qasper"]
    ), "family_policy is scored over every family"
    assert prior == ["hf:qasper"], "prior is scored over the HELD-OUT families only"
    assert sorted(policy) != sorted(prior)


def test_d3_promotion_refuses_arms_scored_on_different_task_sets() -> None:
    from recursive_opt_three_way import ArmResult, _summarize, promotion_decision

    both = ["internal:multiobjective_gsm8k", "hf:qasper"]
    rows = []
    for seed, (std, rec) in enumerate([(0.02, 0.18), (0.01, 0.18), (0.01, 0.17)]):
        rows.append(ArmResult(arm="standard", seed=seed, score=std, scored_tasks=both))
        rows.append(ArmResult(arm="recursive", seed=seed, score=rec, scored_tasks=["hf:qasper"]))

    decision = promotion_decision({"name": "uc4-like", "summary": _summarize(rows)})

    assert decision["action"] == "invalid_comparison"
    assert decision["promote"] is False
    assert decision["comparability"]["only_in_standard"] == ["internal:multiobjective_gsm8k"]


def test_d3_promotion_still_evaluates_a_comparable_pair() -> None:
    from recursive_opt_three_way import ArmResult, _summarize, promotion_decision

    both = ["internal:multiobjective_gsm8k", "hf:qasper"]
    rows = []
    for seed, (std, rec) in enumerate([(0.10, 0.40), (0.11, 0.41), (0.09, 0.39)]):
        rows.append(ArmResult(arm="standard", seed=seed, score=std, scored_tasks=both))
        rows.append(ArmResult(arm="recursive", seed=seed, score=rec, scored_tasks=list(both)))

    decision = promotion_decision({"name": "comparable", "summary": _summarize(rows)})

    assert decision["action"] != "invalid_comparison"
    assert decision["promote"] is True


# --------------------------------------------------------------------------- #
# D4 — the prior surface must not silently train on its own holdout
# --------------------------------------------------------------------------- #
def test_d4_single_family_prior_level_is_rejected() -> None:
    memory = MemoryLite(root=tempfile.mkdtemp())
    with pytest.raises(ValueError, match="held-out transfer|HELD-OUT transfer"):
        S.compile_level({"id": "p", "surface": "prior", "family": "*"}, memory, {"only": ["t1"]})


def test_d4_degenerate_holdout_requires_an_explicit_opt_in() -> None:
    memory = MemoryLite(root=tempfile.mkdtemp())
    level = S.compile_level(
        {"id": "p", "surface": "prior", "family": "*", "allow_degenerate_holdout": True},
        memory,
        {"only": ["t1"]},
    )
    assert list(level._train) == list(level._holdout) == ["only"]


def test_d4_two_families_split_train_and_holdout() -> None:
    memory = MemoryLite(root=tempfile.mkdtemp())
    level = S.compile_level({"id": "p", "surface": "prior", "family": "*"}, memory, _FAMILIES)
    assert list(level._train) == ["gsm8k"]
    assert list(level._holdout) == ["qasper"]


# --------------------------------------------------------------------------- #
# D6 — a level with no data must not report a score
# --------------------------------------------------------------------------- #
def _empty_dataset_spec(evaluator_ref: str) -> dict:
    return {
        "schema_version": S.SCHEMA_VERSION,
        "kind": S.SPEC_KIND,
        "levels": [{
            "id": "L",
            "engine": {"name": "fixed"},
            "module": {"ref": "recursive_opt.module.reasoning_workflow@1",
                       "config": {"components": {"planner": "anything"}}},
            "objective": {"evaluator_ref": evaluator_ref,
                          "metrics": {"score": {"direction": "maximize",
                                                "source": "evaluation.metrics.score",
                                                "aggregate_examples": "mean"}},
                          "selection": {"mode": "scalar", "score_key": "score"}},
            "datasets": {"train": [], "validation": [], "holdout": []},
        }],
    }


def test_d6_level_without_any_data_reports_an_error_not_a_score() -> None:
    ref = "tests.evaluator.always_one@1"
    if ref not in S._EVALUATOR_REGISTRY:
        S.register_evaluator(
            ref,
            lambda output, example, context: EvaluationResult(
                valid=True, status="ok", metrics={"score": 1.0}, feedback=""
            ),
        )
    result = S.run_spec(_empty_dataset_spec(ref))

    assert result.status == "error"
    assert result.valid is False
    assert "at least one example" in (result.error or "")


# --------------------------------------------------------------------------- #
# D11 — the portability flag must describe what RAN, not what was declared
# --------------------------------------------------------------------------- #
def test_d11_legacy_module_cannot_declare_an_output_evaluator() -> None:
    """A legacy level scores via its own _final_eval and ignores evaluator_ref.

    Pairing it with an output-mode evaluator previously compiled and produced a
    result stamped portable=True/promotable=True whose score came from a completely
    different, undeclared code path. Portability then described the declaration
    rather than the execution, which is exactly the guarantee it is supposed to make.
    """
    raw = {
        "schema_version": S.SCHEMA_VERSION,
        "kind": S.SPEC_KIND,
        "levels": [{
            "id": "o1",
            "engine": {"name": "trace", "config": {"iterations": 1, "num_candidates": 1}},
            "module": {"ref": "recursive_opt.module.legacy_level@1",
                       "config": {"level": {"id": "o1", "surface": "config", "family": "f",
                                            "targets": ["starting_artifact"],
                                            "allow_inactive": True},
                                  "families": {"f": ["t1"]}}},
            "objective": {"evaluator_ref": "recursive_opt.evaluator.module_output@1",
                          "metrics": {"score": {"direction": "maximize",
                                                "source": "evaluation.metrics.score",
                                                "aggregate_examples": "mean"}},
                          "selection": {"mode": "scalar", "score_key": "score"}},
            "datasets": {"train": [], "validation": [], "holdout": []},
        }],
    }
    with pytest.raises(ValueError, match="legacy compatibility module"):
        S.compile_plan(raw)


def test_d11_legacy_module_with_its_own_evaluator_still_compiles() -> None:
    """The supported legacy pairing is unaffected and remains non-portable."""
    raw = {
        "schema_version": S.SCHEMA_VERSION,
        "kind": S.SPEC_KIND,
        "levels": [{
            "id": "o1",
            "engine": {"name": "trace", "config": {"iterations": 1, "num_candidates": 1}},
            "module": {"ref": "recursive_opt.module.legacy_level@1",
                       "config": {"level": {"id": "o1", "surface": "config", "family": "f",
                                            "targets": ["starting_artifact"],
                                            "allow_inactive": True},
                                  "families": {"f": ["t1"]}}},
            "objective": {"evaluator_ref": "recursive_opt.evaluator.legacy_level@1",
                          "metrics": {"score": {"direction": "maximize",
                                                "source": "evaluation.metrics.score",
                                                "aggregate_examples": "mean"}},
                          "selection": {"mode": "scalar", "score_key": "score"}},
            "datasets": {"train": [], "validation": [], "holdout": []},
        }],
    }
    plan = S.compile_plan(raw)
    assert plan.units
    assert S._evaluator_entry("recursive_opt.evaluator.legacy_level@1").mode == "legacy_module"


# --------------------------------------------------------------------------- #
# D12 — a failing level must surface its cause, not an opaque KeyError
# --------------------------------------------------------------------------- #
def test_d12_failed_legacy_level_returns_documented_keys_and_the_real_error(_restore_adapter) -> None:
    """A level that fails to compile used to return a dict WITHOUT 'results'.

    Callers (including the three-way harness) then raised `KeyError: 'results'`,
    completely masking the real cause -- in the case that found this, an
    InactiveFieldError naming exactly which trainable fields were dead.
    """
    class _Adapter:
        status = "declares only starting_artifact as active"
        PLUMBED_FIELDS = ("starting_artifact",)

        def run_task(self, cfg, task_id):
            return (0.5, "ok")

    TB.register_task_adapter(_Adapter())
    spec = {
        "families": {"f": ["t1"]},
        "memory_root": tempfile.mkdtemp(),
        "budget": {"candidates": 0, "on_exceed": "return_best"},
        # a single-family 'prior' level cannot compile (D4), so the level fails
        "levels": [{"id": "o3", "surface": "prior", "family": "*",
                    "targets": ["starting_artifact"],
                    "iterations": 1, "num_candidates": 1}],
    }
    out = S.run_spec(spec)

    assert "results" in out and "levels" in out, "documented legacy keys must always be present"
    assert out["results"] == {}
    assert out.get("errors"), "the underlying level error must be surfaced"
    assert any("HELD-OUT transfer" in message for message in out["errors"])


# --------------------------------------------------------------------------- #
# D13 — the causal-effect gate must apply on the run_spec path, not only in
#       validate_spec (which run_spec never calls)
# --------------------------------------------------------------------------- #
def test_d13_inactive_targets_are_rejected_when_compiling_a_level(_restore_adapter) -> None:
    """Optimizing a field with no active causal path must fail loudly.

    effects.check_field_effects is the package's own guard against searching a dead
    knob, but it lived only in validate_spec, which is not on the run_spec path. A
    level targeting an inactive field therefore ran to completion and returned a flat
    score surface rather than an error naming the dead field.
    """
    class _Adapter:
        status = "declares only starting_artifact as active"
        PLUMBED_FIELDS = ("starting_artifact",)

        def run_task(self, cfg, task_id):
            return (0.5, "ok")

    from opto.features.recursive_opt.effects import InactiveFieldError

    TB.register_task_adapter(_Adapter())
    memory = MemoryLite(root=tempfile.mkdtemp())
    with pytest.raises(InactiveFieldError, match="memory_policy"):
        S.compile_level(
            {"id": "o1", "surface": "config", "family": "f", "targets": ["memory_policy"]},
            memory, {"f": ["t1"]},
        )


def test_d13_active_targets_still_compile(_restore_adapter) -> None:
    class _Adapter:
        status = "declares only starting_artifact as active"
        PLUMBED_FIELDS = ("starting_artifact",)

        def run_task(self, cfg, task_id):
            return (0.5, "ok")

    TB.register_task_adapter(_Adapter())
    memory = MemoryLite(root=tempfile.mkdtemp())
    level = S.compile_level(
        {"id": "o1", "surface": "config", "family": "f", "targets": ["starting_artifact"]},
        memory, {"f": ["t1"]},
    )
    assert level is not None


def test_d13_gate_is_inert_without_a_registered_adapter(_restore_adapter) -> None:
    """Offline compilation must not require a task adapter."""
    TB.register_task_adapter(None)
    memory = MemoryLite(root=tempfile.mkdtemp())
    assert S.compile_level(
        {"id": "o1", "surface": "config", "family": "f", "targets": ["memory_policy"]},
        memory, {"f": ["t1"]},
    ) is not None


# --------------------------------------------------------------------------- #
# D14 — a per-level budget in trainer_kwargs must not crash the level
# --------------------------------------------------------------------------- #
def test_d14_search_size_in_trainer_kwargs_is_consumed_not_duplicated() -> None:
    """`allocate_levels` writes num_candidates into trainer_kwargs by design.

    Those values were then also passed positionally to optimize(), raising
    `TypeError: optimize() got multiple values for keyword argument
    'num_candidates'` — which broke every multi-level (i.e. genuinely recursive)
    run through the legacy engine.
    """
    kwargs = {"num_candidates": 2, "iterations": 3, "other": 1}
    iterations, num_candidates = S._resolve_search_size(kwargs, 4, 4)

    assert (iterations, num_candidates) == (3, 2), "trainer_kwargs values win"
    assert kwargs == {"other": 1}, "the duplicates must be consumed"


def test_d14_search_size_defaults_are_used_when_absent() -> None:
    kwargs = {"other": 1}
    assert S._resolve_search_size(kwargs, 5, 6) == (5, 6)
    assert kwargs == {"other": 1}


@pytest.mark.parametrize("bad", [{"num_candidates": 0}, {"iterations": 0}])
def test_d14_non_positive_search_size_is_rejected(bad: dict) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        S._resolve_search_size(dict(bad), 4, 4)


def test_d14_multilevel_recursive_spec_runs_end_to_end(_restore_adapter) -> None:
    """A two-level (o2 policy -> o3 prior) spec must actually execute."""
    class _Adapter:
        status = "deterministic, no LLM"
        PLUMBED_FIELDS = ("starting_artifact",)

        def run_task(self, cfg, task_id):
            return (0.5 if "a" in str(cfg.starting_artifact) else 0.25, f"[fake] {task_id}")

    TB.register_task_adapter(_Adapter())
    spec = {
        "families": {"f1": ["t1"], "f2": ["t2"]},
        "memory_root": tempfile.mkdtemp(),
        # zero candidates -> the fit is skipped, so no optimizer/LLM is needed;
        # the duplicate-kwarg crash happened before any of that mattered.
        "budget": {"candidates": 0, "on_exceed": "return_best"},
        "levels": [
            {"id": "o2_policy", "surface": "family_policy", "family": "*",
             "targets": ["starting_artifact"], "iterations": 1,
             "trainer_kwargs": {"num_candidates": 2}},
            {"id": "o3_prior", "surface": "prior", "family": "*",
             "targets": ["starting_artifact"], "iterations": 1,
             "trainer_kwargs": {"num_candidates": 2}},
        ],
    }
    out = S.run_spec(spec)

    assert not out.get("errors"), out.get("errors")
    assert set(out["results"]) == {"o2_policy", "o3_prior"}


# --------------------------------------------------------------------------- #
# D16 — the config surface silently truncated every multi-line artifact
# --------------------------------------------------------------------------- #
from opto.features.recursive_opt.levels import (  # noqa: E402
    canonicalize_cfg_text, decode_cfg, encode_cfg,
)

_FIELDS = ("starting_artifact",)


@pytest.mark.parametrize("value", [
    "def priority(item, bins):\n    return bins - item",
    "Plan step by step.\nThen verify the answer.",
    "line1\nline2\nline3",
    "Answer directly.",
    "",
])
def test_d16_config_surface_round_trips_multi_line_artifacts(value: str) -> None:
    """Any newline used to truncate the artifact at the first line, silently.

    This destroyed every code artifact and every multi-paragraph prompt the optimizer
    proposed, while the score that artifact had earned was still reported — so the
    reported artifact and the reported score described different things. It stayed
    hidden because the historical prompt menus were all single-line.
    """
    base = LevelConfig()
    encoded = encode_cfg(LevelConfig(starting_artifact=value), _FIELDS)
    assert "\n" not in encoded, "the format is line-oriented; values must stay on one line"
    assert decode_cfg(encoded, base, _FIELDS).starting_artifact == value


def test_d16_canonicalization_preserves_multi_line_code() -> None:
    base = LevelConfig()
    code = "def solve(x):\n    return x * 2"
    encoded = encode_cfg(LevelConfig(starting_artifact=code), _FIELDS)
    assert decode_cfg(canonicalize_cfg_text(encoded, base, _FIELDS),
                      base, _FIELDS).starting_artifact == code


def test_d16_single_line_encoding_is_unchanged() -> None:
    """Existing single-line configs must not gain quotes."""
    encoded = encode_cfg(LevelConfig(starting_artifact="Answer directly."), _FIELDS)
    assert encoded == "starting_artifact: Answer directly."


def test_d16_plain_quoted_single_line_values_still_decode() -> None:
    """LLMs commonly wrap a value in quotes; that must not be read as JSON escaping."""
    base = LevelConfig()
    decoded = decode_cfg('starting_artifact: "Answer directly."', base, _FIELDS)
    assert decoded.starting_artifact == "Answer directly."


# --------------------------------------------------------------------------- #
# D16b — the optimizer writes RAW multi-line text, not encoded text
# --------------------------------------------------------------------------- #
def test_d16b_raw_multi_line_proposal_survives_decoding() -> None:
    """The shape an LLM actually emits: a key line plus unprefixed continuation lines.

    JSON round-tripping only fixed encode->decode. The optimizer never writes encoded
    text, so its proposals were still truncated at the first line.
    """
    raw = "starting_artifact: def priority(item, bins):\n    return bins - item"
    decoded = decode_cfg(raw, LevelConfig(), _FIELDS).starting_artifact
    assert decoded == "def priority(item, bins):\n    return bins - item"


def test_d16b_yaml_block_style_is_understood() -> None:
    """Probe F's optimizer emitted `starting_artifact: |` and the format mangled it."""
    block = "starting_artifact: |\n  def priority(item, bins):\n      return bins - item"
    decoded = decode_cfg(block, LevelConfig(), _FIELDS).starting_artifact
    assert decoded == "def priority(item, bins):\n    return bins - item"


def test_d16b_a_later_key_ends_a_multi_line_value() -> None:
    """A continuation must not swallow the next field."""
    text = "starting_artifact: line one\nline two\nbatch_size: 8"
    cfg = decode_cfg(text, LevelConfig(), ("starting_artifact", "batch_size"))
    assert cfg.starting_artifact == "line one\nline two"
    assert cfg.batch_size == 8


def test_d16b_indented_key_like_line_is_a_continuation_not_a_key() -> None:
    """Indented text inside a code block can contain colons."""
    text = "starting_artifact: def f():\n    batch_size: not a key\n    return 1"
    cfg = decode_cfg(text, LevelConfig(), ("starting_artifact", "batch_size"))
    assert "batch_size: not a key" in cfg.starting_artifact
    assert cfg.batch_size == 4, "the default must survive; that line was code, not config"


@pytest.mark.parametrize("text,expected", [
    ("starting_artifact: Answer directly.", "Answer directly."),
    ('starting_artifact: "Answer directly."', "Answer directly."),
    ("starting_artifact: ", ""),
])
def test_d16b_single_line_decoding_is_unchanged(text: str, expected: str) -> None:
    assert decode_cfg(text, LevelConfig(), _FIELDS).starting_artifact == expected


def test_d16b_enum_fields_still_validate() -> None:
    assert decode_cfg("batch_design: curriculum", LevelConfig(),
                      ("batch_design",)).batch_design == "curriculum"
    with pytest.raises(ValueError, match="Invalid value for batch_design"):
        decode_cfg("batch_design: nonsense", LevelConfig(), ("batch_design",))
