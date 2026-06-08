from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from opto import trace
from opto.features.recursive_opt import (
    CodeArtifactLevel,
    ComponentSpec,
    LevelConfig,
    MemoryLite,
    MetaLevel,
    RecursiveGuide,
)
from opto.features.recursive_opt.tracebench import (
    make_code_evaluator,
    make_inner_runner,
    make_multiobjective_evaluator,
)
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


def test_multiobjective_evaluator_rewards_verified_capability() -> None:
    evaluator = make_multiobjective_evaluator(
        ["hf:GSM8K", "internal:multiobjective_bbeh"],
        {"accuracy": "max", "cost": "min"},
    )

    def capability(task: str) -> Dict[str, str]:
        """Return a concise verified capability answer."""
        return {
            "answer": (
                f"{task}: make a short plan, execute it, then verify/check the answer."
            )
        }

    score, feedback, scalar = evaluator(capability, "qa_reasoning")

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
    assert hf_runner(hf_cfg, "qa_reasoning")[0] > hf_runner(llm4ad_cfg, "qa_reasoning")[0]


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
        "qa_reasoning": ["hf:GSM8K", "internal:multiobjective_bbeh"],
    }


def test_family_policy_level_is_one_trainable_node_and_climbs() -> None:
    o2 = FamilyPolicyLevel(_families(), run_task=TB.make_task_runner())
    assert len(o2.parameters()) == 1  # ONE trainable policy node (low-dim O2)

    weak = ("combinatorial => batch_design=random, memory_policy=none, trainer=MinibatchAlgorithm\n"
            "qa_reasoning => batch_design=random, memory_policy=none, trainer=MinibatchAlgorithm")
    tuned = ("combinatorial => batch_design=failure_balanced, memory_policy=typed, trainer=BeamsearchAlgorithm, trace_type=hybrid\n"
             "qa_reasoning => batch_design=curriculum, memory_policy=retrieval, trainer=UCBSearchAlgorithm, trace_type=otel")
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
        holdout_families={"qa_reasoning": fams["qa_reasoning"]},
        run_task=TB.make_task_runner(),
    )
    assert len(o3.parameters()) == 1
    out = o3.forward()
    # transfer is reported only over held-out families
    assert set(out.data["per_family"]) == {"qa_reasoning"}
    # a qa-tuned prior transfers better to the qa held-out family than a combo-tuned one
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
