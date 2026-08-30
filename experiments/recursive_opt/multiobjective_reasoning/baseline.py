"""Per-example P0 token baseline construction for the selected v2 task."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from .preflight import _canonical_json, _load_json, _run_probe_split


PACKAGE_ROOT = Path(__file__).resolve().parent


def _usage_rows(split_result: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in split_result["per_example_forward_usage"]:
        tokens = int(value["usage"].get("total_tokens", 0))
        if tokens <= 0:
            raise RuntimeError(f"baseline sample {value['sample_id']!r} has no token usage")
        rows.append(
            {
                "sample_id": str(value["sample_id"]),
                "split": split,
                "baseline_forward_tokens": tokens,
                "provider_calls": list(value["provider_calls"]),
            }
        )
    return rows


def build_baseline_token_manifest() -> dict[str, Any]:
    """Freeze P0 forward-token denominators for every selected-task pool row."""
    preregistration = _load_json(PACKAGE_ROOT / "manifests/preregistration_v2.json")
    dataset_manifest = _load_json(PACKAGE_ROOT / "manifests/dataset_manifest_v2.json")
    eligibility = _load_json(PACKAGE_ROOT / "reports/task_eligibility_v2.json")
    task = eligibility.get("selected_task")
    if not isinstance(task, str) or not task:
        raise RuntimeError("baseline manifest requires a selected eligible v2 task")
    artifact = preregistration["manual_probes"]["P0"]
    probe = eligibility["task_results"][task]["probes"]["P0"]
    rows: list[dict[str, Any]] = []
    costs: list[float | None] = []
    completed_ids: set[str] = set()
    for split in ("train", "validation"):
        split_result = probe["splits"][split]
        reused = _usage_rows(split_result, split)
        rows.extend(reused)
        completed_ids.update(row["sample_id"] for row in reused)
        costs.append(split_result["cost_usd"])
    samples = dataset_manifest["tasks"][task]["samples"]
    output_root = PACKAGE_ROOT / "reports/baseline_token_runs"
    for split in ("train", "validation", "holdout"):
        remaining = [
            str(sample["id"])
            for sample in samples
            if sample["split"] == split and sample["id"] not in completed_ids
        ]
        if not remaining:
            continue
        result = _run_probe_split(
            task=task,
            probe_name="P0-baseline",
            artifact=artifact,
            split=split,
            output_root=output_root,
            sample_ids=remaining,
        )
        new_rows = _usage_rows(result, split)
        rows.extend(new_rows)
        completed_ids.update(row["sample_id"] for row in new_rows)
        costs.append(result["cost_usd"])
    expected_ids = {str(sample["id"]) for sample in samples}
    if completed_ids != expected_ids:
        missing = sorted(expected_ids - completed_ids)
        extra = sorted(completed_ids - expected_ids)
        raise RuntimeError(f"baseline token coverage mismatch; missing={missing}, extra={extra}")
    row_by_id = {row["sample_id"]: row for row in rows}
    ordered_rows = [row_by_id[str(sample["id"])] for sample in samples]
    result: dict[str, Any] = {
        "schema_version": "recursive-opt-baseline-token-manifest/v2",
        "experiment_version": preregistration["experiment_version"],
        "selected_task": task,
        "dataset_content_sha256": dataset_manifest["tasks"][task]["content_sha256"],
        "baseline_artifact": dict(artifact),
        "baseline_artifact_fingerprint": hashlib.sha256(
            _canonical_json(artifact).encode("utf-8")
        ).hexdigest(),
        "model_profile": dict(preregistration["model_profiles"]),
        "request_params": {
            "forward": preregistration["model_profiles"]["forward"]["request_params"]
        },
        "cost_usd": sum(float(value) for value in costs)
        if costs and all(value is not None for value in costs)
        else None,
        "holdout_used_for_task_selection": False,
        "holdout_accuracy_recorded": False,
        "sample_count": len(ordered_rows),
        "split_counts": {
            split: sum(row["split"] == split for row in ordered_rows)
            for split in ("train", "validation", "holdout")
        },
        "samples": ordered_rows,
    }
    result["content_sha256"] = hashlib.sha256(
        _canonical_json(result).encode("utf-8")
    ).hexdigest()
    return result
