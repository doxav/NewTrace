"""Frozen spec construction shared by offline and live Experiment 0 runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from opto.features.recursive_opt import spec as control_plane

from .components import MODULE_REF
from .datasets import DATASET_REFS, DATASET_REFS_V2
from .evaluator import EVALUATOR_REF


MODEL = "deepseek/deepseek-v4-flash-0731"
RESOLVED_MODEL = "openrouter/deepseek/deepseek-v4-flash-0731"
REQUEST_TIMEOUT_S = 180
TRANSPORT_MAX_ATTEMPTS = 3
TRANSPORT_BASE_DELAY_S = 1.0
TRACE_NUM_THREADS = 4
INITIAL_ARTIFACT = {
    "analysis_instruction": (
        "Reason carefully and concisely. Identify the operations needed, compute them, "
        "and verify the result before handing analysis to the answer stage."
    ),
    "answer_instruction": (
        "Use the supplied analysis to answer exactly. For multiple choice return the "
        "option token such as (A); for numeric tasks return only the number after FINAL:."
    ),
}


def _engine_config(engine: str, proposals: int, validation_gate: bool) -> dict[str, Any]:
    if engine == "fixed":
        return {}
    if engine == "trace":
        return {
            "optimizer": "OptoPrimeV2",
            "trainer": "PrioritySearch",
            "iterations": proposals + 1,
            "num_candidates": 1,
            "optimizer_kwargs": {},
            "trainer_kwargs": {"num_threads": TRACE_NUM_THREADS},
            "validation_gate": validation_gate,
        }
    if engine == "gepa_optimize_anything":
        return {
            "engine": {
                "display_progress_bar": False,
                "parallel": False,
                "use_cloudpickle": False,
                "cache_evaluation": False,
                "max_candidate_proposals": proposals,
            },
            "reflection": {"reflection_minibatch_size": 1},
        }
    raise ValueError(f"unknown engine {engine!r}")


def build_spec(
    *,
    task: str,
    engine: str,
    seed: int,
    output_directory: str | Path | None,
    proposals: int = 1,
    validation_gate: bool = True,
    offline: bool = False,
    test_mode: bool = False,
    baseline_tokens: Mapping[str, int] | None = None,
    split_limits: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    dataset_refs = {**DATASET_REFS, **DATASET_REFS_V2}
    if task not in dataset_refs:
        raise ValueError(f"unknown task {task!r}")
    baseline = dict(baseline_tokens or {})
    datasets: dict[str, Any] = {
        split: {
            "ref": dataset_refs[task],
            "split": split,
            "config": {
                "baseline_tokens": baseline,
                **(
                    {"limit": split_limits[split]}
                    if split_limits and split in split_limits
                    else {}
                ),
            },
        }
        for split in ("train", "validation", "holdout")
    }
    raw = {
        "schema_version": control_plane.SCHEMA_VERSION,
        "kind": control_plane.SPEC_KIND,
        "runtime": {
            "offline": offline,
            "reproducible": True,
            "strict_refs": True,
            "resume": False,
            "seed": seed,
            "test_mode": test_mode,
        },
        "llm_profiles": {
            "forward_primary": {
                "provider": "openrouter",
                "model": MODEL,
                "resolved_model": RESOLVED_MODEL,
                "api_key_ref": "env:OPENROUTER_API_KEY",
                "fallbacks": [],
                "temperature": 0,
                "max_tokens": 384,
                "request_timeout_s": REQUEST_TIMEOUT_S,
                "transport_max_attempts": TRANSPORT_MAX_ATTEMPTS,
                "transport_base_delay_s": TRANSPORT_BASE_DELAY_S,
                "request_params": {"reasoning": {"enabled": False}},
            },
            "optimizer_primary": {
                "provider": "openrouter",
                "model": MODEL,
                "resolved_model": RESOLVED_MODEL,
                "api_key_ref": "env:OPENROUTER_API_KEY",
                "fallbacks": [],
                "temperature": 0,
                "max_tokens": 8192,
                "request_timeout_s": REQUEST_TIMEOUT_S,
                "transport_max_attempts": TRANSPORT_MAX_ATTEMPTS,
                "transport_base_delay_s": TRANSPORT_BASE_DELAY_S,
                "request_params": {"reasoning": {"effort": "low"}},
            },
        },
        "outputs": {
            "directory": None if output_directory is None else str(output_directory),
            "format": "json",
            "save_artifacts": True,
        },
        "budget": {
            "optimizer_llm_calls": max(1, proposals * 4),
            "eval_llm_calls": 80,
            "candidates": max(1, proposals + 1),
            "evaluator_runs": 40,
            "wall_time_s": 900,
            "total_tokens": 500000 if offline else 60000,
            "on_exceed": "fail",
        },
        "levels": [
            {
                "id": "compound-reasoning",
                "surface": {
                    "kind": "custom",
                    "targets": ["analysis_instruction", "answer_instruction"],
                },
                "module": {
                    "ref": MODULE_REF,
                    "config": dict(INITIAL_ARTIFACT),
                    "artifact": dict(INITIAL_ARTIFACT),
                    "inputs": {},
                },
                "engine": {
                    "name": engine,
                    "config": _engine_config(engine, proposals, validation_gate),
                },
                "objective": {
                    "evaluator_ref": EVALUATOR_REF,
                    "intent": (
                        "Maximize exact-answer accuracy while reducing forward tokens; "
                        "never select an invalid answer."
                    ),
                    "metrics": {
                        "accuracy": {
                            "direction": "maximize",
                            "source": "evaluation.metrics.accuracy",
                            "aggregate_examples": "mean",
                        },
                        "invalid_rate": {
                            "direction": "minimize",
                            "source": "evaluation.metrics.invalid_rate",
                            "aggregate_examples": "mean",
                        },
                        "forward_token_ratio": {
                            "direction": "minimize",
                            "source": "evaluation.metrics.forward_token_ratio",
                            "aggregate_examples": "mean",
                        },
                        "latency_s": {
                            "direction": "minimize",
                            "source": "evaluation.metrics.latency_s",
                            "aggregate_examples": "mean",
                        },
                    },
                    "selection": {
                        "mode": "weighted",
                        "weights": {"accuracy": 1.0, "forward_token_ratio": 0.10},
                    },
                    "hard_constraints": [
                        {"metric": "invalid_rate", "op": "<=", "value": 0.0}
                    ],
                    "feedback_channels": ["natural_language", "trace"],
                },
                "llm_roles": {
                    "forward": "forward_primary",
                    "optimizer": "optimizer_primary",
                    "feedback": None,
                    "judge": None,
                },
                "datasets": datasets,
            }
        ],
    }
    return raw
