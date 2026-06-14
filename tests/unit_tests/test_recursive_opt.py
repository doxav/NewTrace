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
from opto.features.recursive_opt.levels import INVALID_CONFIG_SCORE, DEFAULT_INVALID_FLOOR
from opto.features.recursive_opt.tracebench import (
    TraceBenchTaskAdapter,
    _scalarize_score_dict,
    make_artifact_emitter_evaluator,
    make_code_evaluator,
    make_inner_runner,
    make_multiobjective_evaluator,
    make_tracebench_artifact_evaluator,
    make_tracebench_direct_answer_evaluator,
    load_tracebench_direct_answer_examples,
    normalize_task_id,
    validate_batch_design_indices,
)
from opto.trainer.objectives import ObjectiveConfig
from opto.trainer.guide import Guide

import pytest as _pytest
from opto.features.recursive_opt import tracebench as _TB


class _FakeTaskAdapter:
    """Explicit registered test double (NOT a library stub): deterministic,
    cfg/family-sensitive ``run_task`` so the recursion machinery is testable
    offline without an LLM or real Trace-Bench."""

    status = "fake test adapter"

    def run_task(self, cfg, task_id: str):
        tid = str(task_id).lower()
        combo = tid.startswith("llm4ad:") or tid.startswith("kernelbench:")
        if combo:
            batch = {"failure_balanced": 0.10, "curriculum": 0.04}
            mem = {"typed": 0.07, "retrieval": 0.04}
            trainer = {"BeamsearchAlgorithm": 0.06, "UCBSearchAlgorithm": 0.03}
        else:
            batch = {"curriculum": 0.10, "failure_balanced": 0.04}
            mem = {"retrieval": 0.07, "typed": 0.04}
            trainer = {"UCBSearchAlgorithm": 0.06, "BeamsearchAlgorithm": 0.03}
        s = 0.5
        s += batch.get(cfg.batch_design, -0.03 if cfg.batch_design == "random" else 0.0)
        s += 0.08 if 3 <= cfg.batch_size <= 8 else -0.05
        s += mem.get(cfg.memory_policy, 0.0)
        s += trainer.get(cfg.trainer, 0.0)
        import hashlib as _hl
        h = int(_hl.md5(f"{task_id}|{cfg.to_dict()}".encode()).hexdigest(), 16) % 1000
        s += (h / 1000 - 0.5) * 0.04  # deterministic per-config tie-break
        s = max(0.0, min(1.0, s))
        return s, f"[fake:{task_id}] design={cfg.batch_design}/bs={cfg.batch_size}/mem={cfg.memory_policy}"


@_pytest.fixture(autouse=True)
def _register_fake_adapter(monkeypatch):
    from opto.features.recursive_opt.budget import reset_budget

    for name in (
        "RECURSIVE_OPT_BUDGET_PRESET",
        "RECURSIVE_OPT_MAX_OPTIMIZER_LLM_CALLS",
        "RECURSIVE_OPT_MAX_EVAL_LLM_CALLS",
        "RECURSIVE_OPT_MAX_CANDIDATES",
        "RECURSIVE_OPT_MAX_WALL_TIME_SECONDS",
        "RECURSIVE_OPT_BUDGET_STOP_POLICY",
    ):
        monkeypatch.delenv(name, raising=False)
    reset_budget()
    _TB.register_task_adapter(_FakeTaskAdapter())
    try:
        yield
    finally:
        reset_budget()
        _TB.register_task_adapter(None)


def _inline_multiobjective(task_ids, objectives):
    """Inline multi-objective evaluator for tests (decoupled from Trace-Bench):
    accuracy from verify/plan keywords, cost = length proxy (matches _text_cost)."""
    from opto.features.recursive_opt.tracebench import _text_cost

    def evaluate(capability_callable, family):
        accs, costs = [], []
        for tid in task_ids:
            out = capability_callable(task=tid)
            text = str(out.get("answer", out)) if isinstance(out, dict) else str(out)
            acc = 0.45 + 0.30 * (("verify" in text.lower()) or ("check" in text.lower())) \
                + 0.20 * (("step" in text.lower()) or ("plan" in text.lower()))
            accs.append(min(acc, 1.0))
            costs.append(_text_cost(text))
        score = {"accuracy": sum(accs) / len(accs), "cost": sum(costs) / len(costs)}
        scalar = score["accuracy"] - 0.5 * score["cost"]
        fb = f"[inline-mo] accuracy={score['accuracy']:.2f} cost={score['cost']:.2f}; verify/check helps."
        return score, fb, scalar

    return evaluate


def _batch_design_baseline(self: Any, n: int, k: int) -> List[int]:
    """Return a simple first-k batch for code-artifact tests."""
    return list(range(k))


def _batch_design_improved(self: Any, n: int, k: int) -> List[int]:
    """Return hard examples first while preserving a fixed-size batch."""
    hard = [i for i in range(n) if i % 3 == 0]
    rest = [i for i in range(n) if i % 3 != 0]
    return (hard + rest)[:k]


def _artifact_emitter_good(self: Any) -> str:
    """Return the artifact text expected by the fake bundle guide."""
    return "GOOD"


def _direct_answer_good(self: Any, question: str) -> str:
    """Return the answer expected by the fake direct-answer evaluator."""
    return "GOOD"


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
    # invalid configs now score the bounded default floor (-1.0), never -1e9, so
    # a single invalid candidate cannot destroy reported means.
    assert out.data["score"] == pytest.approx(DEFAULT_INVALID_FLOOR)
    assert "invalid generated config" in out.data["feedback"]


def test_code_artifact_level_can_improve_batch_design(tmp_path: Path) -> None:
    memory = MemoryLite(root=str(tmp_path))
    spec = ComponentSpec(
        name="batch_design",
        baseline=_batch_design_baseline,
        evaluate=make_code_evaluator("llm4ad:online_bin_packing_local", "batch_design"),
        objective="sample hard items while keeping batches diverse",
    )
    level = CodeArtifactLevel(spec, memory=memory)
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
    best_code = memory.best_artifact("llm4ad:online_bin_packing_local", "code")
    assert best_code is not None
    assert best_code.score == pytest.approx(1.0)
    assert "i % 3 == 0" in best_code.content


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
    evaluator = _inline_multiobjective(
        ["internal:multiobjective_gsm8k"],
        {"accuracy": "max", "cost": "min"},
    )

    def capability(task: str) -> Dict[str, str]:
        """Return a concise verified capability answer."""
        return {"answer": f"{task}: plan, execute, then verify/check the answer."}

    score, feedback, scalar = evaluator(capability, "reasoning_control")

    assert score["accuracy"] > 0.9
    assert score["cost"] < 0.4
    assert scalar > 0.7
    assert "verify/check" in feedback


def test_text_cost_penalizes_verbosity() -> None:
    from opto.features.recursive_opt.tracebench import _text_cost

    terse = "plan, execute, verify."
    verbose = "Write an extremely detailed multi-paragraph chain-of-thought " * 4
    assert _text_cost(verbose) > _text_cost(terse)  # longer policy => higher cost
    assert 0.0 <= _text_cost(terse) <= 1.0 and _text_cost(verbose) <= 1.0


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


def test_make_task_runner_requires_registered_adapter() -> None:
    # With no adapter registered, task scoring must RAISE (no synthetic stub).
    _TB.register_task_adapter(None)
    try:
        with pytest.raises(RuntimeError, match="requires a registered Trace-Bench adapter"):
            make_inner_runner("hf:GSM8K")(LevelConfig(), "hf:GSM8K")
    finally:
        _TB.register_task_adapter(_FakeTaskAdapter())  # restore for remaining asserts
    # With an adapter, it routes through run_task.
    score, fb = make_inner_runner("hf:GSM8K")(LevelConfig(), "hf:GSM8K")
    assert 0.0 <= score <= 1.0 and "fake:hf:GSM8K" in fb


def test_capability_artifact_live_path_keeps_trace_connection(tmp_path: Path) -> None:
    from examples.recursive_opt_example_C_learn_capability import (
        CapabilityArtifact, PROBLEMS,
    )

    evaluator = _inline_multiobjective(PROBLEMS, {"accuracy": "max", "cost": "min"})
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


def test_score_bundle_prefers_trace_bench_public_evaluator(monkeypatch) -> None:
    from types import SimpleNamespace

    def fake_evaluate_bundle(
        bundle: Dict[str, Any],
        *,
        max_examples: int,
        strict_score_dict: bool,
    ) -> tuple[float, list[Any]]:
        assert max_examples == 2
        assert strict_score_dict is True
        return 0.1, [
            SimpleNamespace(reward=0.1, feedback="first", score_dict={"accuracy": 0.25}),
            SimpleNamespace(reward=0.2, feedback="second", score_dict={"accuracy": 0.75}),
        ]

    monkeypatch.setattr(TB, "_evaluate_trace_bench_bundle", fake_evaluate_bundle)
    bundle = {"train_dataset": {"inputs": ["a", "b"], "infos": [{}, {}]}}

    score, feedback = TB._score_bundle(bundle, max_examples=2)

    assert score == pytest.approx(0.5)
    assert "train_dataset: mean over 2 real example(s)" in feedback
    assert "first | second" in feedback


def test_score_bundle_rejects_missing_score_dict_from_vector_guide(monkeypatch) -> None:
    from types import SimpleNamespace

    class _Guide:
        def get_score_dict(
            self,
            task_input: str,
            response: str,
            info: Any,
        ) -> Dict[str, float]:
            return {}

    def fake_evaluate_bundle(
        bundle: Dict[str, Any],
        *,
        max_examples: int,
        strict_score_dict: bool,
    ) -> tuple[float, list[Any]]:
        return 0.1, [
            SimpleNamespace(reward=0.1, feedback="missing", score_dict=None),
        ]

    monkeypatch.setattr(TB, "_evaluate_trace_bench_bundle", fake_evaluate_bundle)
    bundle = {
        "guide": _Guide(),
        "train_dataset": {"inputs": ["a"], "infos": [{}]},
    }

    with pytest.raises(RuntimeError, match="get_score_dict returned None"):
        TB._score_bundle(bundle, max_examples=1)


def test_score_bundle_falls_back_without_trace_bench_public_evaluator(monkeypatch) -> None:
    class _Guide:
        def __call__(
            self,
            task_input: str,
            response: str,
            info: Any,
        ) -> tuple[float, str]:
            return (0.5 if response == task_input else 0.0), f"resp={response}"

    monkeypatch.setattr(TB, "_evaluate_trace_bench_bundle", None)
    bundle = {
        "param": lambda task_input: task_input,
        "guide": _Guide(),
        "train_dataset": {"inputs": ["a", "b"], "infos": [{}, {}]},
    }

    score, feedback = TB._score_bundle(bundle, max_examples=2)

    assert score == pytest.approx(0.5)
    assert "train_dataset: mean over 2 real example(s)" in feedback


def test_tracebench_artifact_evaluator_applies_artifact_text(monkeypatch) -> None:
    class _Guide:
        def __call__(
            self,
            task_input: str,
            response: str,
            info: Any,
        ) -> tuple[float, str]:
            return (1.0 if response == info else 0.0), f"response={response}"

    class _BundleAdapter(TraceBenchTaskAdapter):
        def __init__(self) -> None:
            super().__init__(max_examples=1, inner_steps=0)

        def _load_bundle(self, task_id: str, *, fresh: bool = False) -> Dict[str, Any]:
            return {
                "param": trace.node("BAD", trainable=True),
                "guide": _Guide(),
                "train_dataset": {"inputs": ["q"], "infos": ["GOOD"]},
            }

    monkeypatch.setattr(TB, "_evaluate_trace_bench_bundle", None)
    TB.register_task_adapter(_BundleAdapter())

    score, feedback = make_tracebench_artifact_evaluator("internal:fake")("GOOD")

    assert score == pytest.approx(1.0)
    assert "mode=seeded" in feedback
    assert "response=GOOD" in feedback


def test_artifact_emitter_evaluator_invokes_trainable_callable(monkeypatch, tmp_path: Path) -> None:
    class _Guide:
        def __call__(
            self,
            task_input: str,
            response: str,
            info: Any,
        ) -> tuple[float, str]:
            return (1.0 if response == info else 0.0), f"response={response}"

    class _BundleAdapter(TraceBenchTaskAdapter):
        def __init__(self) -> None:
            super().__init__(max_examples=1, inner_steps=0)

        def _load_bundle(self, task_id: str, *, fresh: bool = False) -> Dict[str, Any]:
            return {
                "param": trace.node("BAD", trainable=True),
                "guide": _Guide(),
                "train_dataset": {"inputs": ["q"], "infos": ["GOOD"]},
            }

    monkeypatch.setattr(TB, "_evaluate_trace_bench_bundle", None)
    TB.register_task_adapter(_BundleAdapter())
    memory = MemoryLite(root=str(tmp_path))
    level = CodeArtifactLevel(
        ComponentSpec(
            name="artifact_emitter",
            baseline=_artifact_emitter_good,
            evaluate=make_artifact_emitter_evaluator("internal:fake"),
        ),
        memory=memory,
    )

    score, feedback = RecursiveGuide()("internal:fake", level.forward("internal:fake"), None)

    assert score == pytest.approx(1.0)
    assert "artifact_emitter" in feedback
    assert memory.best_artifact("internal:fake", "code") is not None


def test_tracebench_direct_answer_evaluator_scores_dataset_infos(monkeypatch, tmp_path: Path) -> None:
    class _BundleAdapter(TraceBenchTaskAdapter):
        def __init__(self) -> None:
            super().__init__(max_examples=2, inner_steps=0)

        def _load_bundle(self, task_id: str, *, fresh: bool = False) -> Dict[str, Any]:
            return {
                "param": trace.node("unused", trainable=True),
                "guide": object(),
                "train_dataset": {
                    "inputs": ["q1", "q2"],
                    "infos": ["GOOD", "GOOD"],
                },
            }

    TB.register_task_adapter(_BundleAdapter())
    memory = MemoryLite(root=str(tmp_path))
    level = CodeArtifactLevel(
        ComponentSpec(
            name="direct_answer",
            baseline=_direct_answer_good,
            evaluate=make_tracebench_direct_answer_evaluator("internal:fake"),
        ),
        memory=memory,
    )

    score, feedback = RecursiveGuide()("internal:fake", level.forward("internal:fake"), None)

    assert score == pytest.approx(1.0)
    assert "accuracy=1.000" in feedback
    assert memory.best_artifact("internal:fake", "code") is not None


def test_load_tracebench_direct_answer_examples_uses_registered_bundle(monkeypatch) -> None:
    class _BundleAdapter(TraceBenchTaskAdapter):
        def __init__(self) -> None:
            super().__init__(max_examples=2, inner_steps=0)

        def _load_bundle(self, task_id: str, *, fresh: bool = False) -> Dict[str, Any]:
            return {
                "param": trace.node("unused", trainable=True),
                "guide": object(),
                "train_dataset": {
                    "inputs": ["q1", "q2", "q3"],
                    "infos": [{"answer": "A1"}, {"target": "A2"}, {"answer": "A3"}],
                },
            }

    TB.register_task_adapter(_BundleAdapter())

    examples = load_tracebench_direct_answer_examples("internal:fake")

    assert examples == [("q1", "A1"), ("q2", "A2")]


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


def test_eval_only_adapter_helper_registers_real_bounded_adapter() -> None:
    pytest.importorskip("trace_bench.registry")

    TB.register_task_adapter(None)
    try:
        assert TB.ensure_eval_only_task_adapter(require=True) is True
        assert TB.using_real_tasks() is True
        assert "inner_steps=0" in TB.real_mode_status()
    finally:
        TB.register_task_adapter(_FakeTaskAdapter())


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


def test_tracebench_adapter_declares_trace_type_as_plumbed() -> None:
    assert "trace_type" in TraceBenchTaskAdapter.PLUMBED_FIELDS


def test_tracebench_adapter_declares_credit_horizon_as_feedback_effect() -> None:
    from opto.features.recursive_opt.effects import Effect, effects_for

    adapter = TraceBenchTaskAdapter.__new__(TraceBenchTaskAdapter)
    adapter.inner_steps = 0

    effect = effects_for(adapter)["credit_horizon"]

    assert "credit_horizon" in TraceBenchTaskAdapter.PLUMBED_FIELDS
    assert effect.active
    assert Effect.FEEDBACK in effect.effects


def test_tracebench_adapter_expands_hf_family_task_ids() -> None:
    pytest.importorskip("trace_bench.registry")

    adapter = TraceBenchTaskAdapter(max_examples=1, inner_steps=0)

    task_ids = adapter._expanded_task_ids("hf:bbeh_horizon")

    assert "hf:bbeh_horizon" not in task_ids
    assert "hf:bbeh_horizon/multistep_arithmetic" in task_ids
    assert "hf:bbeh_horizon/web_of_lies" in task_ids


def test_tracebench_adapter_collects_trace_type_feedback(monkeypatch) -> None:
    from opto.features.recursive_opt import traces

    if not traces.HAVE_TRACE_IO:
        pytest.skip("optional graph/telemetry backends are not importable")

    adapter = TraceBenchTaskAdapter(max_examples=1, inner_steps=0)
    monkeypatch.setattr(adapter, "_load_bundle", lambda task_id, fresh=False: {"param": object()})
    monkeypatch.setattr(adapter, "_apply_starting_artifact", lambda bundle, cfg: False)
    monkeypatch.setattr(adapter, "_train_bundle", lambda bundle, cfg: None)
    monkeypatch.setattr(TB, "_score_bundle", lambda bundle, max_examples: (0.42, "scored"))

    score, feedback = adapter.run_task(
        LevelConfig(trace_type="hybrid"),
        "internal:numeric_param",
    )

    assert score == pytest.approx(0.42)
    assert "trace_type=hybrid" in feedback
    assert "trace_sources=otel,sysmon,internal" in feedback
    assert "task score remains the real benchmark score" in feedback


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


def test_multitrace_session_uses_available_trace_io_backends() -> None:
    if not TR.HAVE_TRACE_IO:
        pytest.skip("optional graph/telemetry backends are not importable")

    with TR.MultiTraceSession(["otel", "sysmon"]) as sess:
        sum([1, 2, 3])

    tgj = sess.to_tgj()

    assert {"internal", "otel", "sysmon"}.issubset(tgj["sources"])
    assert isinstance(tgj["documents"], list)


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
        memory = type("M", (), {"memory": []})()

        def exploit(self):
            return candidate, 1.0, {}

    # NEW contract 1: an UNEVALUATED candidate (no rollouts) must never clobber
    # the model — restore declines and the param stays at its current value.
    assert restore_best_validated(FakeTrainer(), level) is False
    assert param.data == original

    # NEW contract 2: once the candidate is validated (rollouts attached), it is
    # written back into the CALLER's model.
    candidate.add_rollouts([{"module": None, "x": None, "info": None,
                             "target": None, "score": 0.9, "feedback": "ok"}])
    FakeTrainer.memory = type("M", (), {"memory": [(-0.9, candidate)]})()
    assert restore_best_validated(FakeTrainer(), level) is True
    assert param.data == "batch_design: failure_balanced"


def test_restore_best_validated_checks_active_priority_search_candidates() -> None:
    """PrioritySearch may pop the best candidate out of heap memory for explore()."""
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
    candidate.add_rollouts([{"module": None, "x": None, "info": None,
                             "target": None, "score": 1.0, "feedback": "ok"}])

    class FakeTrainer:
        agent = level
        memory = type("M", (), {"memory": []})()
        long_term_memory = type("M", (), {"memory": []})()
        short_term_memory = type("M", (), {"memory": []})()
        _best_candidate = candidate
        _exploration_candidates: list = []

    assert restore_best_validated(FakeTrainer(), level) is True
    assert param.data == "batch_design: failure_balanced"


def test_meta_level_candidate_runtime_error_is_bounded() -> None:
    """A generated config can fail inside the task; it must not abort run_spec."""

    def exploding_runner(_cfg: LevelConfig, _family: str) -> tuple[float, str]:
        raise ValueError("bad generated artifact")

    level = MetaLevel(
        LevelConfig(),
        inner_runner=exploding_runner,
        trainable_fields=("starting_artifact",),
        invalid_floor=-1.0,
    )
    out = level.forward("family")
    data = out.data if hasattr(out, "data") else out
    assert data["score"] == -1.0
    assert "candidate runtime error" in data["feedback"]
    assert "bad generated artifact" in data["feedback"]


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


def test_budgeted_live_llm_can_be_deepcopied_for_trainer_proposals(monkeypatch) -> None:
    import copy

    from opto.features.recursive_opt import runmode
    from opto.features.recursive_opt.budget import current_budget, reset_budget
    import opto.utils.llm as llm_mod

    calls = []

    class FakeLiteLLM:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            calls.append(kwargs)
            return object()

    monkeypatch.setattr(llm_mod, "LiteLLM", FakeLiteLLM)
    reset_budget()
    try:
        llm = runmode.make_live_llm("gpt-5.4-nano")
        copied = copy.deepcopy(llm)
        copied(messages=[{"role": "user", "content": "ping"}], max_tokens=7)

        assert calls
        assert current_budget().used_optimizer_llm_calls == 1
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


# --------------------------------------------------------------------------- #
# P0/P1 regression tests: A read-back, Pareto get_score_dict, repeat helper.
# --------------------------------------------------------------------------- #
def test_best_config_from_is_non_empty_and_decodable_after_optimize() -> None:
    from opto.features.recursive_opt import best_config_from, decode_cfg, optimize
    from opto.features.recursive_opt.tracebench import make_dataset

    level = MetaLevel(LevelConfig(), inner_runner=make_inner_runner("hf:GSM8K"),
                      trainable_fields=("batch_design", "trainer"))
    opt = _NoLLMOptimizer(level.parameters())
    optimize(level, make_dataset(["hf:GSM8K"], repeats=8), optimizer=opt,
             iterations=3, num_candidates=2)
    cfg_text = best_config_from(level)
    assert cfg_text.strip()                      # never empty (was empty in live A)
    decoded = decode_cfg(cfg_text, LevelConfig(), ("batch_design", "trainer"))
    assert decoded.batch_design and decoded.trainer

    # even if the trained node is blanked, best_config_from falls back, never empty
    level.parameters()[0]._data = "   "
    assert best_config_from(level).strip()


def test_recursive_guide_get_score_dict_exposes_objectives() -> None:
    guide = RecursiveGuide()

    class _Node:
        data = {"score": 0.7, "feedback": "fb", "objectives": {"accuracy": 0.9, "cost": 0.2}}

    sd = guide.get_score_dict("task", _Node(), None)
    assert sd == {"accuracy": 0.9, "cost": 0.2}
    # scalar-only output -> {"score": ...}
    class _Plain:
        data = {"score": 0.5, "feedback": "x"}
    assert guide.get_score_dict("t", _Plain(), None) == {"score": 0.5}


def test_pareto_path_runs_with_objective_config(tmp_path: Path) -> None:
    # keep the Pareto path: passing objective_config + a dict-returning guide
    # must drive a real trainer end-to-end (no LLM) without error.
    from opto.features.recursive_opt import optimize
    from opto.features.recursive_opt.tracebench import make_dataset
    from opto.trainer.objectives import ObjectiveConfig
    from examples.recursive_opt_example_C_learn_capability import CapabilityArtifact

    ev = _inline_multiobjective(["internal:multiobjective_gsm8k"], {"accuracy": "max", "cost": "min"})
    art = CapabilityArtifact(seed_impl="plan; verify.", evaluator=ev,
                             memory=MemoryLite(root=str(tmp_path)))
    opt = _NoLLMOptimizer(art.parameters())
    optimize(art, make_dataset(["internal:multiobjective_gsm8k"], repeats=8),
             optimizer=opt, iterations=2, num_candidates=2,
             objective_config=ObjectiveConfig(mode="pareto", minimize={"cost"}))
    assert len(art.parameters()) == 1


def test_repeat_scores_reports_mean_and_std() -> None:
    from opto.features.recursive_opt import inspect_utils

    stats = inspect_utils.repeat_scores(lambda s: 0.8 + 0.1 * s, seeds=(0, 1, 2))
    assert stats["n"] == 3
    assert abs(stats["mean"] - 0.9) < 1e-9 and stats["std"] > 0
    assert "± " in inspect_utils.fmt_mean_std(stats)


def test_code_artifact_level_forward_canonicalizes_mismatched_def_name(tmp_path: Path) -> None:
    """forward() must score optimizer-emitted code even when its function name
    differs from the baseline's __name__.

    Root cause of the UC1/UC5 'write-back regression' in the use-case notebook:
    the baseline was named e.g. _weak_batch while the optimizer emitted code
    named batch_design (== spec.name). The trainable bundle resolves its callable
    by the baseline's _fun_name, so the mismatched candidate raised ExecutionError
    and forward() silently scored 0.0 — even though current_code() held the best
    (correct) candidate. forward() now canonicalizes the def-line to _fun_name.
    """
    def _weak_named_differently(self, n, k):  # name != spec.name on purpose
        return list(range(k))

    spec = ComponentSpec(
        name="batch_design",
        baseline=_weak_named_differently,
        evaluate=make_code_evaluator("llm4ad:online_bin_packing_local", "batch_design"),
        objective="sample hard items",
    )
    level = CodeArtifactLevel(spec, memory=MemoryLite(root=str(tmp_path)))
    guide = RecursiveGuide()

    # Optimizer-style update: code named after spec.name, not the baseline.
    level.parameters()[0]._data = (
        "def batch_design(self, n, k):\n"
        "    hard = [i for i in range(n) if i % 3 == 0]\n"
        "    picked = hard[:k]\n"
        "    for i in range(n):\n"
        "        if len(picked) >= k:\n"
        "            break\n"
        "        if i not in picked:\n"
        "            picked.append(i)\n"
        "    return picked\n"
    )
    score, _ = guide(
        "llm4ad:online_bin_packing_local",
        level.forward("llm4ad:online_bin_packing_local"),
        None,
    )
    assert score == 1.0           # was 0.0 before the canonicalization fix
    assert level._last_node is not None


def test_decode_cfg_canonicalizes_quoted_enum_values() -> None:
    """LLM-generated configs often quote enum values; decode must accept them.

    Root cause of UC2/UC6 '-1e9 + ExecutionError' in the use-case notebook: the
    optimizer emitted `starting_artifact: "Plan step by step..."` (with quotes),
    but the raw enum has none, so decode raised ValueError -> invalid_result.
    decode_cfg now strips a single matching surrounding quote pair.
    """
    from opto.features.recursive_opt.levels import (
        decode_cfg, LevelConfig, register_config_values,
    )

    register_config_values("starting_artifact", ["Plan step by step, then answer."])
    base = LevelConfig()
    fields = ("starting_artifact",)

    for raw in ('starting_artifact: "Plan step by step, then answer."',
                "starting_artifact: 'Plan step by step, then answer.'"):
        cfg = decode_cfg(raw, base, fields)
        assert cfg.starting_artifact == "Plan step by step, then answer."

    # empty control arm still decodes to "" (unset/default)
    assert decode_cfg("starting_artifact: ", base, fields).starting_artifact == ""
    # a genuinely unknown value is still rejected (quoting is not a bypass)
    import pytest as _pytest
    with _pytest.raises(ValueError, match="Invalid value for starting_artifact"):
        decode_cfg('starting_artifact: "totally unknown arm"', base, fields)
