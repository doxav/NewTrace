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
