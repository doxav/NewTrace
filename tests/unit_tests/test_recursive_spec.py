"""Tests for the declarative control-plane spec (opto/features/recursive_opt/spec.py).

Offline only: a registered fake Trace-Bench adapter (the documented extension
point) + a no-LLM optimizer drive the real Trainer without any API key.
"""
import copy
import hashlib
from pathlib import Path

import pytest

from opto.optimizers.optimizer import Optimizer
from opto.features.recursive_opt import spec as S
from opto.features.recursive_opt import tracebench as TB
from opto.features.recursive_opt import (
    MemoryLite, MetaLevel, FamilyPolicyLevel, PriorInductionLevel,
    CodeArtifactLevel, LevelConfig, best_config_from,
)


class _FakeTaskAdapter:
    """Deterministic, family/cfg-sensitive run_task test double."""
    status = "fake"

    def run_task(self, cfg, task_id):
        tid = str(task_id).lower()
        combo = tid.startswith("llm4ad:")
        batch = ({"failure_balanced": 0.10, "curriculum": 0.04} if combo
                 else {"curriculum": 0.10, "failure_balanced": 0.04})
        s = 0.5 + batch.get(cfg.batch_design, 0.0) + (0.08 if 3 <= cfg.batch_size <= 8 else -0.05)
        h = int(hashlib.md5(f"{task_id}|{cfg.to_dict()}".encode()).hexdigest(), 16) % 1000
        return max(0.0, min(1.0, s + (h / 1000 - 0.5) * 0.04)), f"[fake:{task_id}]"


@pytest.fixture(autouse=True)
def _adapter():
    TB.register_task_adapter(_FakeTaskAdapter())
    try:
        yield
    finally:
        TB.register_task_adapter(None)


class _NoLLMOptimizer(Optimizer):
    """Drives the Trainer loop with no LLM call (no-op proposals)."""
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters)
        self.steps = 0

    def step(self, *args, **kwargs):
        self.steps += 1
        return {}

    def zero_feedback(self):
        pass

    def backward(self, *args, **kwargs):
        return None


FAMILIES = {
    "combinatorial": ["llm4ad:online_bin_packing_local", "llm4ad:circle_packing"],
    "reasoning": ["hf:GSM8K", "internal:multiobjective_bbeh"],
}


def _config_level(**over):
    d = {"id": "o1", "surface": "config", "family": "combinatorial",
         "targets": ["batch_design", "batch_size"],
         "fixed": {"optimizer": "OptoPrime"}, "iterations": 2}
    d.update(over)
    return d


# --------------------------- step 1: validation --------------------------- #
def test_validate_spec_rejects_bad_specs():
    with pytest.raises(ValueError):                       # empty levels
        S.validate_spec({"families": FAMILIES, "levels": []})
    with pytest.raises(ValueError):                       # unknown surface
        S.validate_spec({"families": FAMILIES, "levels": [{"id": "x", "surface": "bogus"}]})
    with pytest.raises(ValueError):                       # duplicate id
        S.validate_spec({"families": FAMILIES, "levels": [_config_level(), _config_level()]})
    with pytest.raises(ValueError):                       # config w/o known family or task
        S.validate_spec({"families": FAMILIES,
                         "levels": [{"id": "o", "surface": "config", "family": "nope"}]})


def test_validate_spec_enforces_field_constraints():
    from opto.features.recursive_opt import levels as L
    snap = copy.deepcopy(L.CONFIG_ALLOWED_VALUES)
    try:
        spec = {"families": FAMILIES, "levels": [_config_level(
            constraints={"batch_design": ["failure_balanced", "curriculum"]},
            fixed={"batch_design": "not_a_real_value", "optimizer": "OptoPrime"})]}
        with pytest.raises(ValueError):
            S.validate_spec(spec)
    finally:
        L.CONFIG_ALLOWED_VALUES.clear()
        L.CONFIG_ALLOWED_VALUES.update(snap)


def test_validate_spec_rejects_bad_control_dicts():
    with pytest.raises(TypeError):
        S.validate_spec({"families": FAMILIES, "tracebench": "bad", "levels": [_config_level()]})
    with pytest.raises(ValueError):
        S.validate_spec({"families": FAMILIES, "scoring": {"mode": "unknown"},
                         "levels": [_config_level()]})
    with pytest.raises(ValueError):
        S.validate_spec({"families": FAMILIES, "prior_promotion": {"min_support": 0},
                         "levels": [_config_level()]})


def test_tracebench_adapter_can_be_built_from_spec_config():
    adapter = TB.TraceBenchTaskAdapter.from_config({
        "max_examples": 2,
        "inner_steps": 1,
        "inner_candidates": 3,
        "timeout_seconds": 5,
        "allowed_inner_trainers": ["MinibatchAlgorithm"],
        "eval_kwargs": {"n_train": 2},
    })
    assert adapter.max_examples == 2
    assert adapter.inner_steps == 1
    assert adapter.inner_candidates == 3
    assert adapter.eval_kwargs["timeout_seconds"] == 5
    assert adapter.allowed_inner_trainers == ("MinibatchAlgorithm",)


# --------------------------- step 1: compilation -------------------------- #
def test_compile_level_dispatches_by_surface(tmp_path: Path):
    mem = MemoryLite(root=str(tmp_path))
    cfg_l = S.compile_level(_config_level(), mem, FAMILIES)
    assert isinstance(cfg_l, MetaLevel)
    assert cfg_l._fields == ("batch_design", "batch_size")   # targets -> trainable_fields

    pol = S.compile_level({"id": "p", "surface": "family_policy", "family": "*",
                           "targets": ["batch_design", "trainer"]}, mem, FAMILIES)
    assert isinstance(pol, FamilyPolicyLevel)

    pri = S.compile_level({"id": "pr", "surface": "prior", "family": "*"}, mem, FAMILIES)
    assert isinstance(pri, PriorInductionLevel)

    sentinel = object()
    cust = S.compile_level({"id": "c", "surface": "custom",
                            "builder": lambda ls, m: sentinel}, mem, FAMILIES)
    assert cust is sentinel


# --------------------------- step 2: run + memory ------------------------- #
def test_run_spec_two_levels_populates_memory(tmp_path: Path):
    spec = {
        "families": FAMILIES, "budget": {"candidates": 50}, "memory_root": str(tmp_path),
        "levels": [
            _config_level(tools=["trace_search", "note"]),
            {"id": "o2", "surface": "family_policy", "family": "*",
             "targets": ["batch_design", "trainer"], "iterations": 2},
        ],
    }
    out = S.run_spec(spec, optimizer=_NoLLMOptimizer)
    assert set(out["results"]) == {"o1", "o2"}
    assert out["results"]["o1"]["artifact"].strip()        # non-empty config (P0.1 holds via spec)
    assert out["results"]["o2"]["surface"] == "family_policy"
    assert set(out["levels"]) == {"o1", "o2"}               # built objects returned (transparent)
    assert out["memory"].summary()["artifacts"] >= 2


def test_run_spec_uses_spec_tracebench_adapter_config(tmp_path: Path, monkeypatch):
    seen = {}

    class _ConfiguredAdapter(_FakeTaskAdapter):
        @classmethod
        def from_config(cls, config):
            seen.update(config)
            return cls()

    monkeypatch.setattr(TB, "TraceBenchTaskAdapter", _ConfiguredAdapter)
    spec = {
        "families": FAMILIES,
        "tracebench": {"max_examples": 2, "inner_steps": 0},
        "memory_root": str(tmp_path),
        "levels": [_config_level(iterations=1)],
    }
    out = S.run_spec(spec, optimizer=_NoLLMOptimizer)
    assert seen == {"max_examples": 2, "inner_steps": 0}
    assert out["results"]["o1"]["score"] > 0.0


def test_run_spec_depth_is_level_order(tmp_path: Path):
    spec = {"families": FAMILIES, "memory_root": str(tmp_path),
            "levels": [_config_level(id="a"), _config_level(id="b")]}
    out = S.run_spec(spec, optimizer=_NoLLMOptimizer)
    assert list(out["results"].keys()) == ["a", "b"]        # ordered == depth


def test_scored_task_runner_relative_delta_clips_and_reports_raw():
    def raw_runner(cfg, task_id):
        score = -100.0
        if cfg.batch_design == "failure_balanced":
            score = -80.0
        return score, f"raw {task_id}"

    run = S.make_scored_task_runner(
        {"mode": "relative_delta", "clip": [-10.0, 10.0], "report_raw": True},
        raw_runner=raw_runner,
    )
    score, feedback = run(LevelConfig(batch_design="failure_balanced"), "task")
    assert score == 10.0
    assert "SCORE_NORMALIZATION_JSON" in feedback
    assert '"baseline_score": -100.0' in feedback


# --------------------------- step 2: transfer reuse ----------------------- #
def test_transfer_reuse_warm_starts_config_from_prior(tmp_path: Path):
    mem = MemoryLite(root=str(tmp_path))
    for _ in range(3):  # >=3 episodes -> promotes a FamilyPrior
        mem.record(level="O1", cfg={"batch_design": "failure_balanced", "batch_size": 4},
                   family="combinatorial", score=0.9, feedback="good")
    assert mem.family_prior("combinatorial") is not None

    # a fresh project: weak seed config, same family, reuse turned on
    level = S.compile_level(
        _config_level(fixed={"batch_design": "random", "batch_size": 1, "optimizer": "OptoPrime"}),
        mem, FAMILIES,
    )
    info = S.reuse_priors(mem, level, _config_level())
    assert info["used_prior"] is True
    assert "failure_balanced" in best_config_from(level)    # warm-started from the prior


def test_prior_promotion_min_support_from_spec_uses_family_key(tmp_path: Path):
    spec = {
        "families": FAMILIES,
        "memory_root": str(tmp_path),
        "prior_promotion": {"min_support": 2},
        "levels": [_config_level(iterations=2)],
    }
    out = S.run_spec(spec, optimizer=_NoLLMOptimizer)
    assert out["memory"].family_prior("combinatorial") is not None


def test_prior_promotion_can_be_disabled_from_spec(tmp_path: Path):
    spec = {
        "families": FAMILIES,
        "memory_root": str(tmp_path),
        "prior_promotion": {"enabled": False, "min_support": 1},
        "levels": [_config_level(iterations=2)],
    }
    out = S.run_spec(spec, optimizer=_NoLLMOptimizer)
    assert out["memory"].family_prior("combinatorial") is None


def test_tool_reuse_by_family(tmp_path: Path):
    mem = MemoryLite(root=str(tmp_path))
    mem.record_artifact(level="config", family="combinatorial", kind="tool",
                        content="trace_search", score=0.8)
    level = S.compile_level(_config_level(), mem, FAMILIES)
    info = S.reuse_priors(mem, level, _config_level())
    assert "trace_search" in info["tools"]                  # reusable tool retrieved by family


def test_save_priors_records_tagged_artifact(tmp_path: Path):
    mem = MemoryLite(root=str(tmp_path))
    level = S.compile_level(_config_level(), mem, FAMILIES)
    rec = S.save_priors(mem, level, _config_level(tools=["note"]), score=0.7)
    assert rec.kind == "config" and rec.family == "combinatorial"
    assert mem.best_artifact(family="combinatorial", kind="tool") is not None


# ----------------------- P2/P3 regression tests --------------------------- #
def test_prior_promotion_score_gate(tmp_path: Path):
    # below the gate: never promoted, regardless of episode count
    gated = MemoryLite(root=str(tmp_path / "gated"), promotion_min_support=2,
                       promotion_min_score=0.5)
    for _ in range(4):
        gated.record(level="O1", cfg={"batch_design": "random"}, family="flat",
                     score=0.0, feedback="flat normalized run")
    assert gated.family_prior("flat") is None      # limitation #3: junk priors blocked
    # above the gate: promoted as before
    for _ in range(2):
        gated.record(level="O1", cfg={"batch_design": "curriculum"}, family="good",
                     score=0.9, feedback="real gain")
    assert gated.family_prior("good") is not None
    # default behaviour unchanged (no gate)
    legacy = MemoryLite(root=str(tmp_path / "legacy"), promotion_min_support=2)
    for _ in range(2):
        legacy.record(level="O1", cfg={}, family="flat", score=0.0, feedback="x")
    assert legacy.family_prior("flat") is not None


def test_prior_promotion_min_score_spec_validation():
    with pytest.raises(TypeError):
        S.validate_spec({"families": FAMILIES, "prior_promotion": {"min_score": "high"},
                         "levels": [_config_level()]})
    S.validate_spec({"families": FAMILIES, "prior_promotion": {"min_score": 0.2},
                     "levels": [_config_level()]})  # numeric accepted


def test_agentic_optimizer_factory_wires_reused_tools(tmp_path: Path):
    mem = MemoryLite(root=str(tmp_path))
    mem.record(level="O1", cfg={}, family="combinatorial", score=0.2,
               feedback="timeout on large bins")
    ls = _config_level(agentic=True, tools=["trace_search"])
    factory = S.agentic_optimizer_factory(ls, mem, reused_tools=["trace_search"])
    assert factory is not None
    assert "trace_search" in factory.keywords["tools"]      # learned tool re-armed
    assert S.agentic_optimizer_factory(_config_level(), mem) is None  # not agentic


def test_run_spec_agentic_level_trains_with_tools(tmp_path: Path):
    # End-to-end through the real Trainer with no LLM: AgenticOptimizer wraps the
    # no-LLM base and the run completes, proving the trainer can drive the wrapper.
    ls = _config_level(agentic={"base_optimizer_cls": _NoLLMOptimizer},
                       tools=["trace_search"], iterations=2)
    spec = {"families": FAMILIES, "memory_root": str(tmp_path), "levels": [ls]}
    out = S.run_spec(spec)          # no optimizer override: agentic factory is used
    assert out["results"][ls["id"]]["artifact"].strip()
