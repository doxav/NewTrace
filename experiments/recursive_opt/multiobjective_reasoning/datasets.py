"""Pinned, deterministic dataset adapters for Experiment 0."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from functools import lru_cache
from typing import Any, Mapping

from datasets import load_dataset


SELECTION_SEED = 1803
BBEH_REPO = "hubert233/BigBenchExtraHard"
BBEH_REVISION = "442521cdc7195d9e0fdc5ba010b6fcbd902fcc00"
GSM8K_REPO = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"

DATASET_REFS = {
    "object_counting": "recursive_experiments.dataset.bbh_object_counting@1",
    "boolean_expressions": "recursive_experiments.dataset.bbh_boolean_expressions@1",
    "gsm8k": "recursive_experiments.dataset.gsm8k@1",
}

DATASET_REFS_V2 = {
    "bbeh_object_counting": "recursive_experiments.dataset.bbeh_object_counting@2",
    "bbeh_boolean_expressions": "recursive_experiments.dataset.bbeh_boolean_expressions@2",
    "gsm8k": "recursive_experiments.dataset.gsm8k@2",
}

V2_POOL_SIZES = {"train": 16, "validation": 12, "holdout": 24}
_V2_SOURCE_COUNTS = {
    "object_counting": 200,
    "boolean_expressions": 200,
    "gsm8k:train": 7473,
    "gsm8k:test": 1319,
}

_INDICES = {
    "object_counting": {
        "train": [("object_counting", i) for i in (175, 126, 189, 110)],
        "validation": [("object_counting", i) for i in (199, 145)],
        "holdout": [("object_counting", i) for i in (55, 102)],
    },
    "boolean_expressions": {
        "train": [("boolean_expressions", i) for i in (70, 30, 170, 86)],
        "validation": [("boolean_expressions", i) for i in (130, 47)],
        "holdout": [("boolean_expressions", i) for i in (145, 166)],
    },
    "gsm8k": {
        "train": [("train", i) for i in (6512, 3890, 6955, 3147)],
        "validation": [("train", i) for i in (4100, 425)],
        "holdout": [("test", i) for i in (1211, 1132)],
    },
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _numeric_answer(raw: str) -> str:
    return str(raw).split("####")[-1].strip().replace(",", "")


@lru_cache(maxsize=None)
def _source_rows(task: str, source_split: str) -> tuple[dict[str, Any], ...]:
    if task in {"object_counting", "boolean_expressions"}:
        dataset = load_dataset(
            BBEH_REPO,
            split=source_split,
            revision=BBEH_REVISION,
        )
        return tuple(
            {
                "question": str(row["input"]),
                "expected": str(row["target"]).strip(),
            }
            for row in dataset
        )
    if task == "gsm8k":
        dataset = load_dataset(
            GSM8K_REPO,
            "main",
            split=source_split,
            revision=GSM8K_REVISION,
        )
        return tuple(
            {
                "question": str(row["question"]),
                "expected": _numeric_answer(str(row["answer"])),
            }
            for row in dataset
        )
    raise ValueError(f"unknown experiment task {task!r}")


def _resolve(task: str, split: str, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if split not in {"train", "validation", "holdout"}:
        raise ValueError(f"unsupported split {split!r}")
    unknown = set(config) - {"baseline_tokens", "limit"}
    if unknown:
        raise ValueError(f"unknown dataset config keys: {sorted(unknown)}")
    baseline_tokens = config.get("baseline_tokens", {})
    if not isinstance(baseline_tokens, Mapping):
        raise TypeError("baseline_tokens must map sample ids to positive integers")
    limit = config.get("limit")
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
    ):
        raise ValueError("dataset limit must be a positive integer")
    selected = _INDICES[task][split][:limit]
    examples: list[dict[str, Any]] = []
    for source_split, index in selected:
        row = copy.deepcopy(_source_rows(task, source_split)[index])
        sample_id = f"{task}:{source_split}:{index}"
        row.update(
            {
                "id": sample_id,
                "task": task,
                "task_kind": "choice" if task == "boolean_expressions" else "numeric",
                "source_split": source_split,
                "source_index": index,
                "split": split,
            }
        )
        if sample_id in baseline_tokens:
            value = baseline_tokens[sample_id]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"baseline token value for {sample_id!r} must be positive")
            row["baseline_forward_tokens"] = value
        examples.append(row)
    return examples


def _seeded_permutation(label: str, size: int) -> list[int]:
    """Return a stable seeded permutation without consulting model outputs."""
    seed = int.from_bytes(
        hashlib.sha256(f"{SELECTION_SEED}:{label}".encode("utf-8")).digest()[:8],
        "big",
    )
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    return indices


@lru_cache(maxsize=1)
def _v2_pool_indices() -> dict[str, dict[str, tuple[tuple[str, int], ...]]]:
    pools: dict[str, dict[str, tuple[tuple[str, int], ...]]] = {}
    for task, source_task in (
        ("bbeh_object_counting", "object_counting"),
        ("bbeh_boolean_expressions", "boolean_expressions"),
    ):
        permutation = _seeded_permutation(task, _V2_SOURCE_COUNTS[source_task])
        train_end = V2_POOL_SIZES["train"]
        validation_end = train_end + V2_POOL_SIZES["validation"]
        holdout_end = validation_end + V2_POOL_SIZES["holdout"]
        pools[task] = {
            "train": tuple((source_task, index) for index in permutation[:train_end]),
            "validation": tuple(
                (source_task, index)
                for index in permutation[train_end:validation_end]
            ),
            "holdout": tuple(
                (source_task, index)
                for index in permutation[validation_end:holdout_end]
            ),
        }
    gsm_train = _seeded_permutation("gsm8k:train", _V2_SOURCE_COUNTS["gsm8k:train"])
    gsm_test = _seeded_permutation("gsm8k:test", _V2_SOURCE_COUNTS["gsm8k:test"])
    train_end = V2_POOL_SIZES["train"]
    validation_end = train_end + V2_POOL_SIZES["validation"]
    pools["gsm8k"] = {
        "train": tuple(("train", index) for index in gsm_train[:train_end]),
        "validation": tuple(
            ("train", index) for index in gsm_train[train_end:validation_end]
        ),
        "holdout": tuple(
            ("test", index) for index in gsm_test[: V2_POOL_SIZES["holdout"]]
        ),
    }
    return pools


def v2_pool_indices() -> dict[str, dict[str, list[tuple[str, int]]]]:
    """Expose a mutable copy of the frozen deterministic v2 pool indices."""
    return {
        task: {split: list(values) for split, values in splits.items()}
        for task, splits in _v2_pool_indices().items()
    }


def _resolve_v2(
    task: str, split: str, config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if task not in DATASET_REFS_V2:
        raise ValueError(f"unknown v2 experiment task {task!r}")
    if split not in V2_POOL_SIZES:
        raise ValueError(f"unsupported split {split!r}")
    unknown = set(config) - {"baseline_tokens", "limit", "sample_ids"}
    if unknown:
        raise ValueError(f"unknown dataset config keys: {sorted(unknown)}")
    limit = config.get("limit")
    sample_ids = config.get("sample_ids")
    if limit is not None and sample_ids is not None:
        raise ValueError("dataset config cannot combine limit and sample_ids")
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
    ):
        raise ValueError("dataset limit must be a positive integer")
    if sample_ids is not None and (
        not isinstance(sample_ids, (list, tuple))
        or not all(isinstance(value, str) and value for value in sample_ids)
        or len(set(sample_ids)) != len(sample_ids)
    ):
        raise ValueError("sample_ids must be a list of unique non-empty strings")
    baseline_tokens = config.get("baseline_tokens", {})
    if not isinstance(baseline_tokens, Mapping):
        raise TypeError("baseline_tokens must map sample ids to positive integers")
    source_task = {
        "bbeh_object_counting": "object_counting",
        "bbeh_boolean_expressions": "boolean_expressions",
        "gsm8k": "gsm8k",
    }[task]
    available = _v2_pool_indices()[task][split]
    by_id = {
        f"{task}:{source_split}:{index}": (source_split, index)
        for source_split, index in available
    }
    if sample_ids is not None:
        missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"sample_ids are outside the frozen {split} pool: {missing}")
        selected = [by_id[sample_id] for sample_id in sample_ids]
    else:
        selected = list(available[:limit])
    examples: list[dict[str, Any]] = []
    for source_split, index in selected:
        row = copy.deepcopy(_source_rows(source_task, source_split)[index])
        sample_id = f"{task}:{source_split}:{index}"
        row.update(
            {
                "id": sample_id,
                "task": task,
                "task_kind": "choice"
                if task == "bbeh_boolean_expressions"
                else "numeric",
                "source_split": source_split,
                "source_index": index,
                "split": split,
            }
        )
        if sample_id in baseline_tokens:
            value = baseline_tokens[sample_id]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"baseline token value for {sample_id!r} must be positive")
            row["baseline_forward_tokens"] = value
        examples.append(row)
    return examples


def resolve_object_counting(split: str, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _resolve("object_counting", split, config)


def resolve_boolean_expressions(split: str, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _resolve("boolean_expressions", split, config)


def resolve_gsm8k(split: str, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _resolve("gsm8k", split, config)


def resolve_bbeh_object_counting_v2(
    split: str, config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return _resolve_v2("bbeh_object_counting", split, config)


def resolve_bbeh_boolean_expressions_v2(
    split: str, config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return _resolve_v2("bbeh_boolean_expressions", split, config)


def resolve_gsm8k_v2(split: str, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _resolve_v2("gsm8k", split, config)


def dataset_manifest() -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for task, splits in _INDICES.items():
        rows: list[dict[str, Any]] = []
        for split in ("train", "validation", "holdout"):
            for example in _resolve(task, split, {}):
                rows.append(
                    {
                        "id": example["id"],
                        "split": split,
                        "source_split": example["source_split"],
                        "source_index": example["source_index"],
                        "content_sha256": hashlib.sha256(
                            _canonical_json(
                                {
                                    "question": example["question"],
                                    "expected": example["expected"],
                                }
                            ).encode("utf-8")
                        ).hexdigest(),
                    }
                )
        tasks[task] = {
            "ref": DATASET_REFS[task],
            "source": BBEH_REPO if task != "gsm8k" else GSM8K_REPO,
            "source_revision": BBEH_REVISION if task != "gsm8k" else GSM8K_REVISION,
            "selection_seed": SELECTION_SEED,
            "preprocessing": "question text plus deterministic exact target normalization",
            "count": len(rows),
            "samples": rows,
            "content_sha256": hashlib.sha256(_canonical_json(rows).encode("utf-8")).hexdigest(),
        }
    manifest = {
        "schema_version": "recursive-opt-dataset-manifest/v1",
        "selection_seed": SELECTION_SEED,
        "tasks": tasks,
    }
    manifest["content_sha256"] = hashlib.sha256(
        _canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    return manifest


def dataset_manifest_v2() -> dict[str, Any]:
    """Return the frozen v2 pool manifest without evaluating any example."""
    tasks: dict[str, Any] = {}
    for task in DATASET_REFS_V2:
        rows: list[dict[str, Any]] = []
        for split in ("train", "validation", "holdout"):
            for example in _resolve_v2(task, split, {}):
                rows.append(
                    {
                        "id": example["id"],
                        "split": split,
                        "source_split": example["source_split"],
                        "source_index": example["source_index"],
                        "content_sha256": hashlib.sha256(
                            _canonical_json(
                                {
                                    "question": example["question"],
                                    "expected": example["expected"],
                                }
                            ).encode("utf-8")
                        ).hexdigest(),
                    }
                )
        is_bbeh = task.startswith("bbeh_")
        tasks[task] = {
            "ref": DATASET_REFS_V2[task],
            "classification": "high-difficulty stress candidate"
            if is_bbeh
            else "primary eligibility candidate",
            "source": BBEH_REPO if is_bbeh else GSM8K_REPO,
            "source_revision": BBEH_REVISION if is_bbeh else GSM8K_REVISION,
            "selection_seed": SELECTION_SEED,
            "selection_method": "seeded permutation of pinned source row indices",
            "preprocessing": "question text plus deterministic exact target normalization",
            "count": len(rows),
            "split_counts": {
                split: sum(row["split"] == split for row in rows)
                for split in ("train", "validation", "holdout")
            },
            "samples": rows,
            "content_sha256": hashlib.sha256(
                _canonical_json(rows).encode("utf-8")
            ).hexdigest(),
        }
    manifest = {
        "schema_version": "recursive-opt-dataset-manifest/v2",
        "selection_seed": SELECTION_SEED,
        "holdout_evaluated_during_pool_construction": False,
        "tasks": tasks,
    }
    manifest["content_sha256"] = hashlib.sha256(
        _canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    return manifest


def all_expected_answers() -> dict[str, str]:
    v1 = {
        example["question"]: example["expected"]
        for task in _INDICES
        for split in ("train", "validation", "holdout")
        for example in _resolve(task, split, {})
    }
    v2 = {
        example["question"]: example["expected"]
        for task in DATASET_REFS_V2
        for split in ("train", "validation", "holdout")
        for example in _resolve_v2(task, split, {})
    }
    return {**v1, **v2}
