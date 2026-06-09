from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from opto import trace
from opto.features.recursive_opt import (
    best_config_from,
    canonicalize_cfg_text,
    CodeArtifactLevel,
    ComponentSpec,
    LevelConfig,
    MemoryLite,
    MetaLevel,
    RecursiveGuide,
)
from opto.features.recursive_opt.levels import INVALID_CONFIG_SCORE
from opto.features.recursive_opt.tracebench import (
    TraceBenchTaskAdapter,
    _scalarize_score_dict,
    make_code_evaluator,
    make_inner_runner,
    make_multiobjective_evaluator,
    normalize_task_id,
    validate_batch_design_indices,
)
from opto.trainer.objectives import ObjectiveConfig
from opto.trainer.guide import Guide


def _batch_design_baseline(self: Any, n: int, k: int) -> List[int]:
    """Return a simple first-k batch for code-artifact tests."""
    return list(range(k))


def _batch_design_improved(self: Any, n: int, k: int) -> List[int]:
    """Return hard examples first while preserving a fixed-size batch."""
    hard = [i for i in range(n) if i % 3 == 0]
    rest = [i for i in range(n) if i % 3 != 0]
    return (hard + rest)[:k]


def test_recursive_guide_satisfies_trainer_guide_contract() -> None:
    guide = RecursiveGuide()

    assert isinstance(guide, Guide)
    score, feedback = guide(
        "family",
        {"score": 0.75, "feedback": "typed memory helped"},
        None,
    )

    assert score == 0.75
    assert feedback == "typed memory helped"


def test_meta_level_scores_configs_and_promotes_memory_prior(tmp_path: Path) -> None:
    memory = MemoryLite(root=str(tmp_path))
    level = MetaLevel(
        cfg=LevelConfig(
            batch_size=1,
            batch_design="random",
            memory_policy="none",
            trainer="MinibatchAlgorithm",
        ),
        inner_runner=make_inner_runner("hf:GSM8K"),
        trainable_fields=(
            "batch_size",
            "batch_design",
            "memory_policy",
            "trainer",
        ),
        memory=memory,
    )
    guide = RecursiveGuide()

    candidates = [
        {
            "batch_size": 1,
            "batch_design": "random",
            "memory_policy": "none",
            "trainer": "MinibatchAlgorithm",
        },
        {
            "batch_size": 4,
            "batch_design": "failure_balanced",
            "memory_policy": "typed",
            "trainer": "BeamsearchAlgorithm",
        },
        {
            "batch_size": 8,
            "batch_design": "curriculum",
            "memory_policy": "retrieval",
            "trainer": "UCBSearchAlgorithm",
        },
    ]

    scores: List[float] = []
    for candidate in candidates:
        level.propose(**candidate)
        score, _ = guide("hf:GSM8K", level.forward("hf:GSM8K"), None)
        scores.append(score)

    summary = memory.summary()
    assert scores[1] > scores[0]
    assert scores[2] > scores[0]
    assert summary["episodes"] == 3
    assert "hf:GSM8K" in summary["priors"]

    level.propose(**candidates[0])
    repeated_score, _ = guide("hf:GSM8K", level.forward("hf:GSM8K"), None)
    assert repeated_score == scores[0]


def test_meta_level_rejects_invalid_integer_config() -> None:
    level = MetaLevel(
        cfg=LevelConfig(batch_size=1),
        inner_runner=make_inner_runner("hf:GSM8K"),
        trainable_fields=("batch_size",),
    )

    with pytest.raises(ValueError, match="Invalid value for batch_size"):
        level._decode("batch_size: failure_balanced")


def test_meta_level_rejects_invalid_enum_config() -> None:
    level = MetaLevel(
        cfg=LevelConfig(batch_design="random"),
        inner_runner=make_inner_runner("hf:GSM8K"),
        trainable_fields=("batch_design",),
    )

    with pytest.raises(ValueError, match="Invalid value for batch_design"):
        level._decode("batch_design: non_random")

    level.propose(batch_design="non_random")
    out = level.forward("hf:GSM8K")
    assert out.data["score"] == pytest.approx(INVALID_CONFIG_SCORE)
    assert "invalid generated config" in out.data["feedback"]


def test_code_artifact_level_can_improve_batch_design(tmp_path: Path) -> None:
    spec = ComponentSpec(
        name="batch_design",
        baseline=_batch_design_baseline,
        evaluate=make_code_evaluator("llm4ad:online_bin_packing_local", "batch_design"),
        objective="sample hard items while keeping batches diverse",
    )
    level = CodeArtifactLevel(spec, memory=MemoryLite(root=str(tmp_path)))
    guide = RecursiveGuide()

    baseline_score, _ = guide(
        "llm4ad:online_bin_packing_local",
        level.forward("llm4ad:online_bin_packing_local"),
        None,
    )
    level._impl = trace.bundle(trainable=True)(_batch_design_improved)
    improved_score, _ = guide(
        "llm4ad:online_bin_packing_local",
        level.forward("llm4ad:online_bin_packing_local"),
        None,
    )

    assert baseline_score == 0.8
    assert improved_score == 1.0
    assert improved_score > baseline_score
    assert level._last_node is not None


def test_batch_design_validator_names_missing_hard_items() -> None:
    baseline_score, baseline_feedback, selected = validate_batch_design_indices(
        [0, 1, 2, 3],
        n=12,
        k=4,
    )
    improved_score, improved_feedback, _ = validate_batch_design_indices(
        [0, 3, 6, 9],
        n=12,
        k=4,
    )

    assert selected == [0, 1, 2, 3]
    assert baseline_score == pytest.approx(0.8)
    assert improved_score == pytest.approx(1.0)
    assert "Missing hard indices [6, 9]" in baseline_feedback
    assert "all validation hard/failing items" in improved_feedback


def test_multiobjective_evaluator_rewards_verified_capability() -> None:
    evaluator = make_multiobjective_evaluator(
        ["internal:multiobjective_gsm8k"],
        {"accuracy": "max", "cost": "min"},
    )

    def capability(task: str) -> Dict[str, str]:
        """Return a concise verified capability answer."""
        return {
            "answer": (
                f"{task}: make a short plan, execute it, then verify/check the answer."
            )
        }

    score, feedback, scalar = evaluator(capability, "reasoning_control")

    assert score["accuracy"] > 0.9
    assert score["cost"] < 0.4
    assert scalar > 0.7
    assert "verify/check" in feedback


# --------------------------------------------------------------------------- #
# Tests added for the two-agent review fixes (traced code surface, global
# memory retrieval, family-sensitive stub, and the live-path connection).
# --------------------------------------------------------------------------- #
from opto.features.recursive_opt import AgenticOptimizer, default_optimizer_tools
from opto.features.recursive_opt import inspect_utils


class _DummyOptimizer:
    """Records backward calls without needing an LLM backend."""

    def __init__(self, parameters, **kwargs):
        self.parameters = parameters
        self.last = None

    def zero_feedback(self):
        return None

    def backward(self, target, feedback, *args, **kwargs):
        self.last = (target, feedback)
        return None

    def step(self, *args, **kwargs):
        return None


def test_code_artifact_level_supports_multiobjective_and_keeps_backward_path(tmp_path: Path) -> None:
    def baseline(self, task):
        return {"answer": f"verify {task}"}

    def evaluator(component, family):
        out = component(task="demo")
        assert "verify" in out["answer"]
        # 3-tuple multi-objective contract: (metrics, feedback, scalar)
        return {"accuracy": 0.8, "cost": 0.2}, "keep explicit verify step", 0.7

    level = CodeArtifactLevel(
        ComponentSpec("capability", baseline, evaluator),
        memory=MemoryLite(root=str(tmp_path / "mem")),
    )

    out = level("dummy-family")
    data = out.data if hasattr(out, "data") else out
    assert data["score"] == pytest.approx(0.7)
    assert data["feedback"] == "keep explicit verify step"
    assert data["metrics"]["accuracy"] == pytest.approx(0.8)
    assert data["metrics"]["cost"] == pytest.approx(0.2)

    # backward on the (now connected) output must reach the trainable code param
    opt = AgenticOptimizer(level.parameters(), tools={}, base_optimizer_cls=_DummyOptimizer)
    opt.zero_feedback()
    opt.backward(out, "improve this capability")
    assert opt.opt.last[0] is out


def test_code_artifact_level_raises_when_evaluator_never_calls_candidate() -> None:
    def baseline(self, task):
        return {"answer": task}

    def evaluator(component, family):
        return 0.5, "no candidate invocation"  # never calls component(...)

    level = CodeArtifactLevel(ComponentSpec("broken", baseline, evaluator))
    with pytest.raises(RuntimeError, match="did not invoke the candidate callable"):
        level("dummy-family")


def test_default_optimizer_tools_searches_global_failures(tmp_path: Path) -> None:
    mem = MemoryLite(root=str(tmp_path))
    mem.record(level="O1", cfg={}, family="hf:GSM8K", score=0.10, feedback="failure-a")
    mem.record(level="O1", cfg={}, family="llm4ad:online_bin_packing_local",
               score=0.20, feedback="failure-b")

    tools = default_optimizer_tools(memory=mem)  # default family=None => global
    hits = tools["trace_search"]("ignored")
    assert len(hits) == 2
    assert any("failure-a" in h for h in hits)
    assert any("failure-b" in h for h in hits)


def test_tracebench_stub_is_family_sensitive() -> None:
    llm4ad_runner = make_inner_runner("llm4ad:online_bin_packing_local")
    hf_runner = make_inner_runner("hf:GSM8K")

    llm4ad_cfg = LevelConfig(batch_size=4, batch_design="failure_balanced",
                             memory_policy="typed", trainer="BeamsearchAlgorithm",
                             trace_type="hybrid")
    hf_cfg = LevelConfig(batch_size=8, batch_design="curriculum",
                         memory_policy="retrieval", trainer="UCBSearchAlgorithm",
                         trace_type="otel")

    # each family prefers its own profile
    assert llm4ad_runner(llm4ad_cfg, "combinatorial")[0] > llm4ad_runner(hf_cfg, "combinatorial")[0]
    assert hf_runner(hf_cfg, "reasoning_control")[0] > hf_runner(llm4ad_cfg, "reasoning_control")[0]


def test_capability_artifact_live_path_keeps_trace_connection(tmp_path: Path) -> None:
    from examples.recursive_opt_example_C_learn_capability import (
        CapabilityArtifact, PROBLEMS,
    )

    evaluator = make_multiobjective_evaluator(PROBLEMS, {"accuracy": "max", "cost": "min"})
    art = CapabilityArtifact(seed_impl="Answer the task.", evaluator=evaluator,
                             memory=MemoryLite(root=str(tmp_path)))
    guide = RecursiveGuide()

    out = art.forward(PROBLEMS[0])
    score, feedback = guide(PROBLEMS[0], out, None)
    assert hasattr(out, "data")
    assert "objectives" in out.data

    opt = AgenticOptimizer(art.parameters(), tools={}, base_optimizer_cls=_DummyOptimizer)
    opt.zero_feedback()
    opt.backward(out, feedback)
    assert opt.opt.last[0] is out
    assert opt.opt.last[1] == feedback


def test_inspect_utils_code_diff_and_summary() -> None:
    before = "def f(self, n, k):\n    return list(range(k))\n"
    after = "def f(self, n, k):\n    return [i for i in range(n) if i % 3 == 0][:k]\n"
    diff = inspect_utils.code_diff(before, after, name="batch_design")
    assert "batch_design (initial)" in diff and "+    return [i" in diff
    assert inspect_utils.code_diff("x", "x") == "(no change to artifact)"
    verdict = inspect_utils.summarize(before, after, 0.8, 1.0, name="batch_design")
    assert "improved" in verdict and "changed" in verdict


# --------------------------------------------------------------------------- #
# Tests for the O2/O3 trainable levels, M2 lineage memory, the Trace-Bench
# adapter contract, and internal-trace normalization (latest review round).
# --------------------------------------------------------------------------- #
from opto.features.recursive_opt import (
    FamilyPolicyLevel,
    PriorInductionLevel,
    ArtifactRecord,
    encode_cfg,
    decode_cfg,
)
from opto.features.recursive_opt import tracebench as TB
from opto.features.recursive_opt import traces as TR


def _families():
    return {
        "combinatorial": ["llm4ad:online_bin_packing_local", "llm4ad:circle_packing"],
        "reasoning_control": ["internal:multiobjective_gsm8k", "internal:multi_param"],
    }


def test_family_policy_level_is_one_trainable_node_and_climbs() -> None:
    o2 = FamilyPolicyLevel(_families(), run_task=TB.make_task_runner())
    assert len(o2.parameters()) == 1  # ONE trainable policy node (low-dim O2)

    weak = ("combinatorial => batch_design=random, memory_policy=none, trainer=MinibatchAlgorithm\n"
            "reasoning_control => batch_design=random, memory_policy=none, trainer=MinibatchAlgorithm")
    tuned = ("combinatorial => batch_design=failure_balanced, memory_policy=typed, trainer=BeamsearchAlgorithm, trace_type=hybrid\n"
             "reasoning_control => batch_design=curriculum, memory_policy=retrieval, trainer=UCBSearchAlgorithm, trace_type=otel")
    o2.propose(weak)
    weak_score = o2.forward().data["score"]
    o2.propose(tuned)
    out = o2.forward()
    assert out.data["score"] > weak_score  # the policy node is genuinely trainable
    assert set(out.data["per_family"]) == set(_families())


def test_prior_induction_scores_on_holdout_only() -> None:
    fams = _families()
    o3 = PriorInductionLevel(
        train_families={"combinatorial": fams["combinatorial"]},
        holdout_families={"reasoning_control": fams["reasoning_control"]},
        run_task=TB.make_task_runner(),
    )
    assert len(o3.parameters()) == 1
    out = o3.forward()
    # transfer is reported only over held-out families
    assert set(out.data["per_family"]) == {"reasoning_control"}
    # a reasoning-tuned prior transfers better to the held-out family than a combo-tuned one
    o3.propose(batch_design="failure_balanced", trainer="BeamsearchAlgorithm", trace_type="hybrid")
    combo = o3.forward().data["score"]
    o3.propose(batch_design="curriculum", memory_policy="retrieval", trainer="UCBSearchAlgorithm", trace_type="otel")
    qa = o3.forward().data["score"]
    assert qa > combo


def test_memory_m2_artifact_lineage_history_and_best(tmp_path: Path) -> None:
    mem = MemoryLite(root=str(tmp_path))
    a0 = mem.record_artifact("O1", "fam", "config", "batch_design: random", 0.5)
    a1 = mem.record_artifact("O1", "fam", "config", "batch_design: failure_balanced", 0.8,
                             parent_id=a0.artifact_id)
    a2 = mem.record_artifact("O1", "fam", "config", "batch_design: curriculum", 0.7,
                             parent_id=a1.artifact_id)
    chain = mem.lineage(a2.artifact_id)
    assert [a.artifact_id for a in chain] == [a0.artifact_id, a1.artifact_id, a2.artifact_id]
    assert mem.best_artifact("fam", "config").score == pytest.approx(0.8)
    assert len(mem.artifact_history("fam", "config")) == 3
    assert mem.summary()["artifacts"] == 3
    # persists across reloads (JSONL)
    assert len(MemoryLite(root=str(tmp_path)).artifact_history("fam", "config")) == 3


def test_register_task_adapter_overrides_stub() -> None:
    class FakeAdapter:
        def run_task(self, cfg, task_id):
            return 0.99, f"real:{task_id}"

    try:
        TB.register_task_adapter(FakeAdapter())
        assert TB.using_real_tasks() is True
        score, fb = TB.make_task_runner()(LevelConfig(), "hf:GSM8K")
        assert score == pytest.approx(0.99) and "real:hf:GSM8K" in fb
    finally:
        TB.register_task_adapter(None)  # never leak global state to other tests
    assert TB.using_real_tasks() is False


def test_tracebench_task_aliases_are_normalized() -> None:
    assert normalize_task_id("hf:GSM8K") == "internal:multiobjective_gsm8k"
    assert normalize_task_id("hf:BBEH") == "hf:bbeh/boolean_expressions"


def test_tracebench_adapter_scores_real_internal_task_when_available() -> None:
    pytest.importorskip("trace_bench.registry")

    adapter = TraceBenchTaskAdapter(
        eval_kwargs={"n_train": 1, "n_val": 0, "timeout_seconds": 1},
        max_examples=1,
        inner_steps=0,
    )
    score, feedback = adapter.run_task(LevelConfig(), "internal:numeric_param")

    assert score == pytest.approx(-3.0)
    assert "[real_trace_bench:internal:numeric_param]" in feedback


def test_tracebench_multiobjective_scalarization_uses_objective_config() -> None:
    config = ObjectiveConfig(
        mode="weighted",
        weights={"error": 1.0, "tokens_in": 1e-3, "tokens_out": 1e-3},
        minimize=frozenset({"error", "tokens_in", "tokens_out"}),
    )

    concise_correct = _scalarize_score_dict(
        {"error": 0.0, "tokens_in": 10.0, "tokens_out": 20.0},
        config,
    )
    verbose_correct = _scalarize_score_dict(
        {"error": 0.0, "tokens_in": 30.0, "tokens_out": 80.0},
        config,
    )
    concise_wrong = _scalarize_score_dict(
        {"error": 1.0, "tokens_in": 10.0, "tokens_out": 20.0},
        config,
    )

    assert concise_correct == pytest.approx(-0.03)
    assert verbose_correct == pytest.approx(-0.11)
    assert concise_wrong == pytest.approx(-1.03)
    assert concise_correct > verbose_correct > concise_wrong


def test_tracebench_adapter_rejects_inner_trainer_outside_live_budget() -> None:
    adapter = TraceBenchTaskAdapter(
        max_examples=1,
        inner_steps=1,
        allowed_inner_trainers=("MinibatchAlgorithm", "PrioritySearch"),
    )
    allowed = adapter._trainer_budget_feedback(
        LevelConfig(trainer="PrioritySearch"),
        "internal:multi_param",
    )
    rejected = adapter._trainer_budget_feedback(
        LevelConfig(trainer="BeamsearchAlgorithm"),
        "internal:multi_param",
    )

    assert allowed is None
    assert rejected is not None
    score, feedback = rejected
    assert score == pytest.approx(INVALID_CONFIG_SCORE)
    assert "outside the live budget allowlist" in feedback
    assert "PrioritySearch" in feedback


def test_meta_level_allows_real_tracebench_external_dependencies() -> None:
    pytest.importorskip("trace_bench.registry")

    try:
        TB.register_task_adapter(
            TraceBenchTaskAdapter(
                eval_kwargs={"n_train": 1, "n_val": 0, "timeout_seconds": 1},
                max_examples=1,
                inner_steps=0,
            )
        )
        level = MetaLevel(
            LevelConfig(),
            inner_runner=make_inner_runner("internal:numeric_param"),
            trainable_fields=("batch_design",),
        )
        out = level.forward("internal:numeric_param")
        assert out.data["score"] == pytest.approx(-3.0)
        assert "real_trace_bench" in out.data["feedback"]
    finally:
        TB.register_task_adapter(None)


def test_multitrace_session_normalizes_internal_graph() -> None:
    cfg = LevelConfig(batch_design="failure_balanced")
    level = MetaLevel(cfg, inner_runner=make_inner_runner("hf:GSM8K"),
                      trainable_fields=("batch_design",))
    out = level.forward("hf:GSM8K")
    sess = TR.MultiTraceSession(["internal"]).__enter__()
    sess.record_internal(out)
    tgj = sess.to_tgj()
    sess.__exit__(None, None, None)
    assert "internal" in tgj["sources"]
    assert len(tgj["nodes"]) > 0  # internal trace now contributes REAL nodes
    assert all(n["source"] == "internal" for n in tgj["nodes"])


def test_encode_decode_cfg_roundtrip_is_shared_contract() -> None:
    fields = ("batch_design", "memory_policy", "trainer")
    cfg = LevelConfig(batch_design="curriculum", memory_policy="typed", trainer="UCBSearchAlgorithm")
    restored = decode_cfg(encode_cfg(cfg, fields), LevelConfig(), fields)
    assert (restored.batch_design, restored.memory_policy, restored.trainer) == (
        "curriculum", "typed", "UCBSearchAlgorithm")


def test_canonicalize_cfg_text_removes_unknown_generated_fields() -> None:
    fields = ("batch_size", "batch_design", "memory_policy", "trainer")
    raw = "\n".join(
        [
            "batch_size: 8",
            "batch_design: random",
            "memory_policy: none",
            "trainer: MinibatchAlgorithm",
            "inner_steps: 100",
            "learning_rate: 0.01",
        ]
    )
    expected = "\n".join(raw.splitlines()[:4])

    assert canonicalize_cfg_text(raw, LevelConfig(), fields) == expected

    level = MetaLevel(
        LevelConfig(),
        inner_runner=make_inner_runner("hf:GSM8K"),
        trainable_fields=fields,
    )
    level.parameters()[0]._data = raw
    assert best_config_from(level) == expected


# --------------------------------------------------------------------------- #
# Tests for the Trainer-based optimization path (examples no longer hand-roll
# the loop; they delegate to opto.trainer via optimize()).
# --------------------------------------------------------------------------- #
from opto.features.recursive_opt import (
    optimize,
    resolve_trainer,
    current_iterations,
    current_num_candidates,
    ITERATIONS,
    OPTIMIZER,
    TRAINER,
)
from opto.features.recursive_opt.tracebench import make_dataset
from opto.optimizers.optimizer import Optimizer


class _NoLLMOptimizer(Optimizer):
    """Optimizer stub: drives the Trainer loop without calling any LLM."""

    def __init__(self, parameters, **kwargs):
        super().__init__(parameters)
        self.steps = 0

    def step(self, *args, **kwargs):
        self.steps += 1
        return {}

    def zero_feedback(self):
        pass

    def backward(self, *args, **kwargs):
        pass


def test_resolve_trainer_prefers_priority_search_else_gepa_base() -> None:
    assert resolve_trainer("PrioritySearch") == "PrioritySearch"
    # unknown trainer falls back to the GEPA-style Pareto-based search
    assert resolve_trainer("DoesNotExist") == "ParetobasedPS"


def test_optimize_delegates_to_trainer_with_requested_config(monkeypatch) -> None:
    import importlib

    opt_mod = importlib.import_module("opto.features.recursive_opt.optimize")

    captured = {}

    def fake_train(**kwargs):
        captured.update(kwargs)
        return "trained"

    monkeypatch.setattr(opt_mod, "_train_returning_trainer", fake_train)
    level = MetaLevel(LevelConfig(), inner_runner=make_inner_runner("hf:GSM8K"),
                      trainable_fields=("batch_design",))
    result = opt_mod.optimize(level, make_dataset(["hf:GSM8K"], repeats=5), iterations=7)

    assert result == "trained"
    # examples delegate to a Trainer with the requested config (no hand-rolled loop)
    assert captured["algorithm"] == "PrioritySearch"   # GEPA-Base is the fallback
    assert captured["optimizer"] == "OptoPrimeV2"
    assert captured["num_steps"] == 7 and captured["num_epochs"] == 0


def test_optimize_budget_env_is_resolved_at_call_time(monkeypatch) -> None:
    import importlib

    opt_mod = importlib.import_module("opto.features.recursive_opt.optimize")
    captured = {}

    def fake_train(**kwargs):
        captured.update(kwargs)
        return "trained"

    monkeypatch.setattr(opt_mod, "_train_returning_trainer", fake_train)
    monkeypatch.setenv("RECURSIVE_OPT_ITERATIONS", "3")
    monkeypatch.setenv("RECURSIVE_OPT_NUM_CANDIDATES", "1")

    level = MetaLevel(
        LevelConfig(),
        inner_runner=make_inner_runner("hf:GSM8K"),
        trainable_fields=("batch_design",),
    )
    assert current_iterations() == 3
    assert current_num_candidates() == 1

    opt_mod.optimize(level, make_dataset(["hf:GSM8K"], repeats=5))

    assert captured["num_steps"] == 3
    assert captured["num_candidates"] == 1


def test_optimize_global_candidate_budget_clamps_outer_steps(monkeypatch) -> None:
    import importlib
    from opto.features.recursive_opt.budget import configure_budget_from_env, current_budget, reset_budget

    opt_mod = importlib.import_module("opto.features.recursive_opt.optimize")
    captured = {}

    def fake_train(**kwargs):
        captured.update(kwargs)
        return "trained"

    monkeypatch.setattr(opt_mod, "_train_returning_trainer", fake_train)
    monkeypatch.setenv("RECURSIVE_OPT_MAX_CANDIDATES", "3")
    monkeypatch.setenv("RECURSIVE_OPT_BUDGET_STOP_POLICY", "raise")
    configure_budget_from_env()
    try:
        level = MetaLevel(
            LevelConfig(),
            inner_runner=make_inner_runner("hf:GSM8K"),
            trainable_fields=("batch_design",),
        )
        result = opt_mod.optimize(
            level,
            make_dataset(["hf:GSM8K"], repeats=5),
            iterations=5,
            num_candidates=2,
        )

        assert result == "trained"
        assert captured["num_steps"] == 1
        assert captured["num_candidates"] == 2
        assert current_budget().used_candidates == 2
    finally:
        reset_budget()


def test_optimize_global_candidate_budget_zero_returns_current_state(monkeypatch) -> None:
    import importlib
    from opto.features.recursive_opt.budget import configure_budget_from_env, reset_budget

    opt_mod = importlib.import_module("opto.features.recursive_opt.optimize")
    called = False

    def fake_train(**kwargs):
        nonlocal called
        called = True
        return "trained"

    monkeypatch.setattr(opt_mod, "_train_returning_trainer", fake_train)
    monkeypatch.setenv("RECURSIVE_OPT_MAX_CANDIDATES", "0")
    monkeypatch.setenv("RECURSIVE_OPT_BUDGET_STOP_POLICY", "return_best")
    configure_budget_from_env()
    try:
        level = MetaLevel(
            LevelConfig(),
            inner_runner=make_inner_runner("hf:GSM8K"),
            trainable_fields=("batch_design",),
        )
        result = opt_mod.optimize(
            level,
            make_dataset(["hf:GSM8K"], repeats=1),
            iterations=1,
            num_candidates=1,
        )

        assert result is None
        assert called is False
    finally:
        reset_budget()


def test_optimize_runs_real_trainer_end_to_end(tmp_path: Path) -> None:
    level = MetaLevel(LevelConfig(), inner_runner=make_inner_runner("hf:GSM8K"),
                      trainable_fields=("batch_design",))
    opt = _NoLLMOptimizer(level.parameters())
    # real PrioritySearch loop, no LLM, no manual backward()/step(): must complete
    result = optimize(level, make_dataset(["hf:GSM8K"], repeats=20),
                      optimizer=opt, iterations=3, num_candidates=2)
    assert len(level.parameters()) == 1  # level intact and still trainable after training
    assert hasattr(result, "exploit")


def test_restore_best_validated_applies_candidate_module_state() -> None:
    from opto.features.recursive_opt.optimize import restore_best_validated
    from opto.trainer.algorithms.priority_search import ModuleCandidate

    level = MetaLevel(
        LevelConfig(batch_design="random"),
        inner_runner=make_inner_runner("hf:GSM8K"),
        trainable_fields=("batch_design",),
    )
    param = level.parameters()[0]
    original = param.data
    param._data = "batch_design: failure_balanced"
    candidate = ModuleCandidate(level)
    param._data = original

    class FakeTrainer:
        agent = level

        def exploit(self):
            return candidate, 1.0, {}

    assert restore_best_validated(FakeTrainer()) is True
    assert param.data == "batch_design: failure_balanced"


def test_optimize_defaults_match_requested_config() -> None:
    # the three configurable knobs default as specified in the request
    assert TRAINER == "PrioritySearch"
    assert OPTIMIZER == "OptoPrimeV2"
    assert ITERATIONS == 10


def test_live_model_preflight_redacts_provider_errors(monkeypatch) -> None:
    from opto.features.recursive_opt import runmode
    import opto.utils.llm as llm_mod

    class FailingLLM:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            raise RuntimeError("model denied for sk-test-secret in proj_testsecret")

    monkeypatch.setattr(llm_mod, "LiteLLM", FailingLLM)

    with pytest.raises(SystemExit) as exc_info:
        runmode.preflight_model("missing-model")

    message = str(exc_info.value)
    assert "missing-model" in message
    assert "sk-<redacted>" in message
    assert "proj_<redacted>" in message
    assert "sk-test-secret" not in message
    assert "proj_testsecret" not in message


def test_preflight_does_not_consume_global_optimizer_budget(monkeypatch) -> None:
    from opto.features.recursive_opt import runmode
    from opto.features.recursive_opt.budget import configure_budget_from_env, current_budget, reset_budget
    import opto.utils.llm as llm_mod

    calls = []

    class FakeLiteLLM:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            calls.append(kwargs)
            return object()

    monkeypatch.setattr(llm_mod, "LiteLLM", FakeLiteLLM)
    monkeypatch.setenv("RECURSIVE_OPT_MAX_OPTIMIZER_LLM_CALLS", "0")
    configure_budget_from_env()
    runmode._PREFLIGHTED_MODELS.discard("budget-preflight-model")
    try:
        runmode.preflight_model("budget-preflight-model")

        assert calls
        assert current_budget().used_optimizer_llm_calls == 0
    finally:
        reset_budget()
        runmode._PREFLIGHTED_MODELS.discard("budget-preflight-model")


def test_global_optimizer_llm_budget_zero_blocks_live_calls(monkeypatch) -> None:
    from opto.features.recursive_opt import runmode
    from opto.features.recursive_opt.budget import (
        BudgetExceeded,
        configure_budget_from_env,
        reset_budget,
    )
    import opto.utils.llm as llm_mod

    calls = []

    class FakeLiteLLM:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            calls.append(kwargs)
            return object()

    monkeypatch.setattr(llm_mod, "LiteLLM", FakeLiteLLM)
    monkeypatch.setenv("RECURSIVE_OPT_MAX_OPTIMIZER_LLM_CALLS", "0")
    configure_budget_from_env()
    try:
        llm = runmode.make_live_llm("gpt-5.4-nano")
        with pytest.raises(BudgetExceeded):
            llm(messages=[{"role": "user", "content": "ping"}], max_tokens=7)

        assert calls == []
    finally:
        reset_budget()


def test_gpt5_live_llm_maps_max_tokens(monkeypatch) -> None:
    from opto.features.recursive_opt import runmode
    import opto.utils.llm as llm_mod

    calls = []

    class FakeLiteLLM:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            calls.append(kwargs)
            return object()

    monkeypatch.setattr(llm_mod, "LiteLLM", FakeLiteLLM)

    llm = runmode.make_live_llm("gpt-5.4-nano")
    llm(messages=[{"role": "user", "content": "ping"}], max_tokens=7, temperature=0)

    assert calls
    assert calls[0]["max_completion_tokens"] == 7
    assert "max_tokens" not in calls[0]
