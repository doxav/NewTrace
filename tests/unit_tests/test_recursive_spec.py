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
    # Clean GLOBAL state per test: adapter and the global budget (a leftover
    # budget from a prior test silently turns optimize() into a no-op via
    # stop_policy=return_best — the exact mechanism behind the live P5 0.018s).
    from opto.features.recursive_opt.budget import reset_budget
    TB.register_task_adapter(_FakeTaskAdapter())
    reset_budget()
    try:
        yield
    finally:
        TB.register_task_adapter(None)
        reset_budget()


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


# ------------------- flat-surface root-cause regression tests ------------- #
class _PlumbedAdapter(_FakeTaskAdapter):
    """Adapter that declares plumbing and is sensitive to the artifact text."""
    PLUMBED_FIELDS = ("starting_artifact", "trainer", "batch_size")

    def run_task(self, cfg, task_id):
        art = str(getattr(cfg, "starting_artifact", "") or "")
        base = 0.4 + 0.1 * min(len(art), 100) / 100.0
        if "verify" in art.lower():
            base += 0.3
        return min(1.0, base), f"[plumbed:{task_id}] artifact={art[:30]!r}"


def test_validate_spec_rejects_unplumbed_targets():
    TB.register_task_adapter(_PlumbedAdapter())
    spec = {"families": FAMILIES,
            "levels": [_config_level(targets=["batch_design", "batch_size"])]}
    with pytest.raises(ValueError, match="no ACTIVE causal path"):
        S.validate_spec(spec)                      # batch_design is a no-op knob
    spec["levels"][0]["allow_unplumbed"] = True
    S.validate_spec(spec)                          # explicit override allowed
    spec2 = {"families": FAMILIES,
             "levels": [_config_level(targets=["starting_artifact", "batch_size"])]}
    S.validate_spec(spec2)                         # plumbed targets pass


def test_score_spread_detects_flat_and_nonflat_surfaces():
    TB.register_task_adapter(_PlumbedAdapter())
    live = S.score_spread("hf:GSM8K")
    assert live["spread"] > 0 and not live["flat"]  # artifact path moves the score
    assert live["failed_probes"] == 0
    assert live["valid_spread"] == live["spread"]
    assert live["invalid_probes"] == 0
    assert live["catastrophic"] is False

    class _Constant(_FakeTaskAdapter):
        def run_task(self, cfg, task_id):           # the bug we now detect
            return 0.5, "constant"
    TB.register_task_adapter(_Constant())
    flat = S.score_spread("hf:GSM8K")
    assert flat["flat"] and flat["spread"] == 0.0
    assert flat["valid_spread"] == 0.0
    assert flat["invalid_probes"] == 0
    assert flat["catastrophic"] is False

    class _RejectsPromptProbe(_FakeTaskAdapter):
        def run_task(self, cfg, task_id):
            if str(getattr(cfg, "starting_artifact", "") or ""):
                raise ValueError("prompt artifact incompatible with numeric task")
            return 0.5, "control arm ok"
    TB.register_task_adapter(_RejectsPromptProbe())
    gated = S.score_spread("internal:numeric_param")
    assert gated["flat"] and gated["spread"] == 0.0
    assert gated["failed_probes"] == 2
    assert gated["invalid_probes"] == 2
    assert gated["catastrophic"] is True
    assert any("incompatible" in row.get("error", "") for row in gated["rows"])

    class _SentinelProbe(_FakeTaskAdapter):
        def run_task(self, cfg, task_id):
            if str(getattr(cfg, "starting_artifact", "") or ""):
                return -1_000_000.0, "invalid sentinel"
            return 0.5, "control arm ok"
    TB.register_task_adapter(_SentinelProbe())
    sentinel = S.score_spread("internal:numeric_param")
    assert sentinel["valid_spread"] == 0.0
    assert sentinel["invalid_probes"] == 2
    assert sentinel["catastrophic"] is True


def test_adapter_seeds_starting_artifact_into_bundle_param():
    from opto.features.recursive_opt.tracebench import TraceBenchTaskAdapter
    from opto.features.recursive_opt.levels import LevelConfig

    class _Param:
        def __init__(self): self._data = "original"
    adapter = TraceBenchTaskAdapter.__new__(TraceBenchTaskAdapter)  # no trace_bench needed
    bundle = {"param": _Param()}
    seeded = adapter._apply_starting_artifact(
        bundle, LevelConfig(starting_artifact="Plan, then verify."))
    assert seeded and bundle["param"]._data == "Plan, then verify."
    assert not adapter._apply_starting_artifact(bundle, LevelConfig())  # empty -> no-op


def test_tracebench_agent_fn_injects_artifact_before_response(monkeypatch) -> None:
    from opto.features.recursive_opt.tracebench import TraceBenchTaskAdapter

    class _Param:
        def __init__(self) -> None:
            self._data = ""

        def __call__(self, task_input: str) -> str:
            return f"{self._data}|{task_input}"

    adapter = TraceBenchTaskAdapter.__new__(TraceBenchTaskAdapter)  # no trace_bench needed
    monkeypatch.setattr(adapter, "_load_bundle", lambda task_id, fresh=False: {"param": _Param()})

    fn = adapter.agent_fn("internal:numeric_param")

    assert fn("ALPHA", "probe") == "ALPHA|probe"
    assert fn("BETA", "probe") == "BETA|probe"


def test_compliance_objective_blocks_degenerate_terse_optimum():
    from opto.features.recursive_opt.tracebench import _text_cost
    # direct check of the compliance math used by make_multiobjective_evaluator
    terms = ("plan", "verify")
    def compliance(text):
        low = text.lower()
        return sum(1.0 for t in terms if t in low) / len(terms)
    terse, compliant = "Answer directly.", "Plan briefly, then verify the answer."
    s_terse = 1.0 - 0.5 * _text_cost(terse) + 0.5 * compliance(terse)
    s_comp  = 1.0 - 0.5 * _text_cost(compliant) + 0.5 * compliance(compliant)
    assert s_comp > s_terse        # the intended capability now wins the scalar


def test_empty_config_value_is_always_valid_control_arm():
    from opto.features.recursive_opt import levels as L
    import copy
    snap = copy.deepcopy(L.CONFIG_ALLOWED_VALUES)
    try:
        L.register_config_values("starting_artifact", ["Plan, then verify."])
        L.validate_config_field("starting_artifact", "")        # default arm: OK
        L.validate_config_field("starting_artifact", "Plan, then verify.")
        with pytest.raises(ValueError):
            L.validate_config_field("starting_artifact", "rogue value")
    finally:
        L.CONFIG_ALLOWED_VALUES.clear(); L.CONFIG_ALLOWED_VALUES.update(snap)


# ---------------- write-back regression (the "trained code lost" bug) ------- #
_GOOD_CODE = '''def batch_design(self, n, k):
    hard = [i for i in range(n) if i % 3 == 0]
    picked = hard[:k]
    for i in range(n):
        if len(picked) >= k:
            break
        if i not in picked:
            picked.append(i)
    return picked'''


class _ScriptedCodeOptimizer(Optimizer):
    """Deterministically proposes _GOOD_CODE (no LLM)."""
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters)

    def _step(self, *a, **k):
        return {self.parameters[0]: _GOOD_CODE}

    def step(self, *a, **k):
        update = self._step()
        for p, v in update.items():
            p._data = v
        return update

    def zero_feedback(self): pass
    def backward(self, *a, **k): return None


def _weak_baseline(self, n, k):
    """Deliberately weak baseline (ignores hard items)."""
    return list(range(k))


def test_optimize_writes_best_validated_code_back_to_caller_model(tmp_path: Path):
    """After optimize(), the CALLER's level must hold the best evaluated code.

    Regression for the live P1 failure: trainer logs showed the agent reaching
    score 1.0 internally while current_code() on the caller's level still held
    the baseline (PrioritySearch trains deep copies; the old restore helper
    wrote to trainer.agent and trusted exploit()'s unevaluated-None->0 ranking).
    """
    from opto.features.recursive_opt.optimize import optimize
    from opto.features.recursive_opt.tracebench import make_code_evaluator

    ev = make_code_evaluator("internal:batch_design", "batch_design")
    level = S.compile_level({"id": "wb", "surface": "code",
        "component": {"name": "batch_design", "baseline": _weak_baseline,
                      "evaluate": ev, "objective": "max"}},
        MemoryLite(root=str(tmp_path)), {})
    optimize(level, {"inputs": [None] * 3, "infos": [None] * 3}, iterations=3,
             trainer="PrioritySearch", optimizer=_ScriptedCodeOptimizer)

    code = level.current_code()
    assert "i % 3 == 0" in code, "trained code was lost on write-back"
    # and it scores 1.0 through the real validator (baseline scores 0.8)
    ns = {}
    exec(code, ns, ns)
    fn = ns["batch_design"]
    score, _ = ev(lambda *a, **k: fn(None, *a, **k), "code")
    assert score == 1.0


def test_restore_best_validated_never_clobbers_without_evaluations(tmp_path: Path):
    from opto.features.recursive_opt.optimize import restore_best_validated

    class _EmptyTrainer:
        memory = type("M", (), {"memory": []})()
        agent = None
    level = S.compile_level({"id": "nc", "surface": "code",
        "component": {"name": "batch_design", "baseline": _weak_baseline,
                      "evaluate": lambda c, f: (0.0, "x"), "objective": "max"}},
        MemoryLite(root=str(tmp_path)), {})
    before = level.current_code()
    assert restore_best_validated(_EmptyTrainer(), level) is False
    assert level.current_code() == before   # untouched


# -------- sentinel-leak root fix (the -666,666,666 confirm explosion) ------- #
def test_invalid_config_score_respects_scoring_clip_floor(tmp_path: Path):
    """Invalid candidates must score the worst LEGAL value, never -1e9."""
    spec_scoring = {"mode": "raw", "clip": [-1.0, 1.0]}
    level = S.compile_level(
        _config_level(targets=["batch_size"]), MemoryLite(root=str(tmp_path)),
        FAMILIES, scoring=spec_scoring)
    assert level._invalid_floor == -1.0          # derived from the clip floor
    out = level._run_inner("batch_size: banana", "hf:GSM8K")
    data = out.data if hasattr(out, "data") else out
    assert data["score"] == -1.0                 # was -1_000_000_000.0
    assert "invalid generated config" in data["feedback"]
    # default behaviour unchanged without a clip (backward compatible)
    bare = S.compile_level(_config_level(targets=["batch_size"]),
                           MemoryLite(root=str(tmp_path / "b")), FAMILIES)
    raw = bare._run_inner("batch_size: banana", "hf:GSM8K")
    d2 = raw.data if hasattr(raw, "data") else raw
    from opto.features.recursive_opt.levels import INVALID_CONFIG_SCORE
    assert d2["score"] == INVALID_CONFIG_SCORE


def test_run_spec_clamps_reported_scores_and_records_wall_s(tmp_path: Path):
    """Belt: even a custom level emitting a raw sentinel cannot leak it into results."""
    from opto.features.recursive_opt.levels import INVALID_CONFIG_SCORE

    class _SentinelLevel(S.MetaLevel if False else object):
        pass

    def builder(ls, memory):
        import opto.trace as trace
        from opto.trace import node
        from opto.trace import Module

        @trace.model
        class _Leaky(Module):
            def __init__(self):
                super().__init__()
                self.p = node("x", trainable=True, name="leaky")

            @trace.bundle(allow_external_dependencies=True)
            def _run(self, p):
                return {"score": INVALID_CONFIG_SCORE, "feedback": "boom"}

            def forward(self, _):
                return self._run(self.p)
        return _Leaky()

    spec = {"families": FAMILIES, "memory_root": str(tmp_path),
            "scoring": {"mode": "raw", "clip": [-1.0, 1.0]},
            "levels": [{"id": "leak", "surface": "custom", "builder": builder,
                        "iterations": 1}]}
    out = S.run_spec(spec, optimizer=_NoLLMOptimizer)
    res = out["results"]["leak"]
    assert res["score"] == -1.0                  # clamped, was -1e9 raw
    assert isinstance(res["wall_s"], float)      # fix 3: wall time recorded per level


# ------------------- wall-time objective (fix 3 root) ----------------------- #
def test_timed_guide_exposes_wall_time_objective():
    from opto.features.recursive_opt.levels import TimedGuide

    class _Inner:
        def get_feedback(self, q, r, info=None, **kw):
            import time; time.sleep(0.01)
            return 1.0, "ok"
        def get_score_dict(self, q, r, info=None, **kw):
            return {"score": 1.0}

    g = TimedGuide(_Inner())
    score, fb = g.get_feedback("q", {"score": 1.0}, None)
    d = g.get_score_dict("q", {"score": 1.0}, None)
    assert score == 1.0 and d["score"] == 1.0
    assert d["wall_time"] >= 0.01                # measured, thread-local


def test_spec_timed_guide_and_objective_config_validate():
    ls = _config_level(targets=["batch_size"])
    ls["timed_guide"] = True
    ls["objective_config"] = {"mode": "pareto", "minimize": ["wall_time"]}
    S.validate_spec({"families": FAMILIES, "levels": [ls]})


# ------------------- capability surface (promoted from example C) ----------- #
def test_capability_surface_compiles_and_runs(tmp_path: Path):
    from opto.features.recursive_opt import CapabilityArtifact

    def evaluator(capability, family):
        text = capability("probe")["answer"]
        acc = 1.0 if "verify" in text.lower() else 0.4
        return {"accuracy": acc}, f"acc={acc}", acc

    spec = {"families": FAMILIES, "memory_root": str(tmp_path),
            "levels": [{"id": "cap", "surface": "capability",
                        "seed": "Plan, then verify the answer.",
                        "evaluator": evaluator, "iterations": 1}]}
    S.validate_spec(spec)
    out = S.run_spec(spec, optimizer=_NoLLMOptimizer)
    res = out["results"]["cap"]
    assert res["score"] == 1.0
    level = out["levels"]["cap"]
    assert isinstance(level, CapabilityArtifact)
    assert "verify" in level.current_text().lower()
    with pytest.raises(TypeError):               # evaluator is mandatory
        S.validate_spec({"families": FAMILIES,
                         "levels": [{"id": "bad", "surface": "capability"}]})


# ------------------- budget no-op visibility (fix 2 root) ------------------- #
def test_optimize_warns_when_global_budget_already_exhausted(tmp_path: Path, capsys):
    from opto.features.recursive_opt.budget import RecursiveOptBudget, reset_budget
    from opto.features.recursive_opt.optimize import optimize
    level = S.compile_level(_config_level(targets=["batch_size"]),
                            MemoryLite(root=str(tmp_path)), FAMILIES)
    b = RecursiveOptBudget(max_candidates=1)
    b.used_candidates = 1                       # simulate a prior run consuming it
    reset_budget(b)
    try:
        optimize(level, {"inputs": [None], "infos": [None]}, iterations=1,
                 optimizer=_NoLLMOptimizer)
    finally:
        reset_budget()
    assert "ALREADY exhausted" in capsys.readouterr().out


# ================== causal-effect contract (effects.py) ==================== #
def test_effects_contract_conditional_activity_and_policy():
    from opto.features.recursive_opt.effects import (
        Effect, check_field_effects, effects_for, InactiveFieldError)
    from opto.features.recursive_opt.tracebench import TraceBenchTaskAdapter

    eval_only = TraceBenchTaskAdapter.__new__(TraceBenchTaskAdapter)
    eval_only.inner_steps = 0
    training = TraceBenchTaskAdapter.__new__(TraceBenchTaskAdapter)
    training.inner_steps = 2

    # trace_type is FEEDBACK/TRACE-plumbed (never claimed as score-plumbed)
    fx = effects_for(eval_only)
    assert set(fx["trace_type"].effects) == {Effect.TRACE, Effect.FEEDBACK}

    # trainer: inactive at inner_steps=0 (with the activating condition named),
    # active at inner_steps>0 — same field, mode-dependent verdict.
    with pytest.raises(InactiveFieldError, match="inner_steps > 0"):
        check_field_effects(eval_only, ["trainer"])
    rep = check_field_effects(training, ["trainer"])
    assert "trainer" in rep.active

    # configurable policy: required_effects narrows what counts as relevant
    with pytest.raises(InactiveFieldError, match="required"):
        check_field_effects(training, ["trace_type"],
                            required_effects=[Effect.MEMORY])
    rep = check_field_effects(training, ["trace_type"],
                              required_effects=[Effect.FEEDBACK])
    assert rep.active["trace_type"] == (Effect.FEEDBACK,)

    # allow_inactive: report instead of raise (diagnostic mode)
    rep = check_field_effects(eval_only, ["batch_design"], allow_inactive=True)
    assert "batch_design" in rep.inactive and "sampler" in rep.inactive["batch_design"]


def test_effects_fallback_from_legacy_plumbed_fields():
    from opto.features.recursive_opt.effects import Effect, effects_for

    class _Legacy:
        PLUMBED_FIELDS = ("starting_artifact",)
    fx = effects_for(_Legacy())
    assert fx["starting_artifact"].effects == (Effect.ARTIFACT, Effect.SCORE)
    assert effects_for(None) == {}


def test_spec_effect_policy_is_configurable(tmp_path: Path):
    """Spec-level knobs: allow_inactive + effect_policy.required_effects."""
    class _Contract(_FakeTaskAdapter):
        def field_effects(self):
            from opto.features.recursive_opt.effects import Effect, FieldEffect
            return {"starting_artifact": FieldEffect(
                        "starting_artifact", (Effect.ARTIFACT, Effect.SCORE)),
                    "memory_policy": FieldEffect(
                        "memory_policy", (Effect.MEMORY,), active=False,
                        condition="inactive until retrieval wiring exists")}
    TB.register_task_adapter(_Contract())
    base = {"families": FAMILIES, "memory_root": str(tmp_path)}
    dead = _config_level(targets=["memory_policy"])
    with pytest.raises(ValueError, match="no ACTIVE causal path"):
        S.validate_spec({**base, "levels": [dead]})
    S.validate_spec({**base, "levels": [{**dead, "allow_inactive": True}]})
    S.validate_spec({**base, "levels": [{**dead, "allow_unplumbed": True}]})  # legacy alias
    ok = _config_level(targets=["starting_artifact"],
                       effect_policy={"required_effects": ["artifact"]})
    S.validate_spec({**base, "levels": [ok]})


# ================== make_level_spec + example E regression ================= #
def test_make_level_spec_preserves_multiple_constraints():
    level = S.make_level_spec(
        id="o1", surface="config",
        constraints={"starting_artifact": ["a", "b"],
                     "batch_design": ["random", "failure_balanced"]},
        iterations=3, depends_on=["o0"])
    assert set(level["constraints"]) == {"starting_artifact", "batch_design"}
    assert level["iterations"] == 3 and level["depends_on"] == ["o0"]


def test_example_e_spec_preserves_starting_artifact_constraints():
    import importlib.util
    spec_path = Path(__file__).resolve().parents[2] / "examples" / "recursive_opt_example_E_declarative_spec.py"
    mod_spec = importlib.util.spec_from_file_location("exE_check", spec_path)
    exE = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(exE)
    constraints = exE.SPEC["levels"][0]["constraints"]
    # regression: the duplicate-"constraints" dict key silently dropped this menu
    assert "starting_artifact" in constraints and "batch_design" in constraints
    assert "" in constraints["starting_artifact"]


# ================== depends_on is enforced ================================= #
def test_depends_on_validation(tmp_path: Path):
    base = {"families": FAMILIES, "memory_root": str(tmp_path)}
    with pytest.raises(ValueError, match="EARLIER level ids"):
        S.validate_spec({**base, "levels": [
            _config_level(id="a", depends_on=["ghost"])]})
    with pytest.raises(ValueError, match="EARLIER level ids"):
        S.validate_spec({**base, "levels": [          # forward reference
            _config_level(id="a", depends_on=["b"]), _config_level(id="b")]})
    S.validate_spec({**base, "levels": [
        _config_level(id="a"), _config_level(id="b", depends_on=["a"])]})


def test_run_spec_records_artifact_lineage_and_dependencies(tmp_path: Path):
    spec = {"families": FAMILIES, "memory_root": str(tmp_path),
            "levels": [_config_level(id="o1"),
                       _config_level(id="o2", depends_on=["o1"])]}
    out = S.run_spec(spec, optimizer=_NoLLMOptimizer)
    r2 = out["results"]["o2"]
    assert r2["depends_on"] == ["o1"]
    assert r2["artifact_id"]            # regression: getattr(rec, "id") was always None


# ================== memory retrieve + reconsolidation ===================== #
def test_memorylite_retrieve_and_reconsolidate(tmp_path: Path):
    mem = MemoryLite(root=str(tmp_path), promotion_min_support=2)
    mem.record(level="O1", cfg={"trainer": "BeamsearchAlgorithm"}, family="f",
               score=0.4, feedback="a")
    mem.record(level="O1", cfg={"trainer": "UCBSearchAlgorithm"}, family="f",
               score=0.9, feedback="b")
    prior = mem.reconsolidate_family("f")
    assert prior is not None and prior.best_cfg["trainer"] == "UCBSearchAlgorithm"
    assert "episodes=2" in prior.notes

    got = mem.retrieve("f", level="O1", topk=2)
    assert [e.score for e in got["episodes"]] == [0.9, 0.4]      # best-first
    assert got["prior"].best_score == pytest.approx(0.9)
    assert mem.retrieve("*", topk=1)["prior"] is None            # priors are family-scoped
    assert mem.retrieve("f", min_score=0.5, topk=5)["episodes"][0].score == 0.9
    recent = mem.retrieve("f", sort="recent", topk=1)["episodes"][0]
    assert recent.feedback == "b"

    flat = MemoryLite(root=str(tmp_path / "flat"), promotion_min_support=1,
                      promotion_min_score=0.5)
    flat.record(level="O1", cfg={}, family="g", score=0.0, feedback="flat")
    assert flat.reconsolidate_family("g") is None                # gates still apply
