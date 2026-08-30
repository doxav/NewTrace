"""Idempotent versioned registration for Experiment 0."""

from __future__ import annotations

from opto.features.recursive_opt import (
    ModuleRegistryEntry,
    register_dataset,
    register_evaluator,
    register_module,
)
from opto.features.recursive_opt import spec as control_plane

from . import components, datasets, evaluator


def register_experiment_components() -> None:
    """Register the exact module, evaluator, and datasets used by every arm."""
    register_module(
        components.MODULE_REF,
        ModuleRegistryEntry(
            build=components.build_module,
            snapshot=components.snapshot_module,
            restore=components.restore_module,
            validate_artifact=components.validate_artifact,
            capabilities=frozenset({"trace_module", "json_snapshot", "multi_component"}),
            validate_config=components.validate_config,
        ),
    )
    register_evaluator(
        evaluator.EVALUATOR_REF,
        evaluator.exact_reasoning_evaluator,
        mode=evaluator.EVALUATOR_MODE,
    )
    register_dataset(
        datasets.DATASET_REFS["object_counting"], datasets.resolve_object_counting
    )
    register_dataset(
        datasets.DATASET_REFS["boolean_expressions"], datasets.resolve_boolean_expressions
    )
    register_dataset(datasets.DATASET_REFS["gsm8k"], datasets.resolve_gsm8k)
    register_dataset(
        datasets.DATASET_REFS_V2["bbeh_object_counting"],
        datasets.resolve_bbeh_object_counting_v2,
    )
    register_dataset(
        datasets.DATASET_REFS_V2["bbeh_boolean_expressions"],
        datasets.resolve_bbeh_boolean_expressions_v2,
    )
    register_dataset(datasets.DATASET_REFS_V2["gsm8k"], datasets.resolve_gsm8k_v2)


def assert_strict_output_evaluator() -> None:
    """Fail before paid work unless the registered evaluator is output-only."""
    register_experiment_components()
    entry = control_plane._evaluator_entry(evaluator.EVALUATOR_REF)
    if entry.mode != evaluator.EVALUATOR_MODE:
        raise RuntimeError(
            f"expected evaluator mode {evaluator.EVALUATOR_MODE!r}, got {entry.mode!r}"
        )
