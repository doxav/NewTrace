"""Recompute EXP20 readiness results from immutable reader responses, without calls."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from . import pilot, prepare, task


def audit_slot(
    folder: Path, row: dict[str, Any], variant: str, model: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check identity and response hashes; reject altered scores instead of repairing them."""
    request = json.loads((folder / "request.json").read_text())
    response = json.loads((folder / "response.json").read_text())
    result = json.loads((folder / "result.json").read_text())
    if request["request_hash"] != task.digest(request["request"]):
        raise ValueError("request hash mismatch")
    if response["request_hash"] != request["request_hash"]:
        raise ValueError("response/request identity mismatch")
    raw = response["response"]
    if raw["model"] != model:
        raise ValueError("model differs from registered reader")
    answer = task.pack_answer({"content": task._response_text(raw)}, {}).data
    em, f1 = task.answer_metrics(answer["answer"], row["answer"])
    expected = {
        "id": row["id"],
        "variant": variant,
        "type": row["type"],
        "answer": answer["answer"],
        "answer_EM": em,
        "answer_F1": f1,
        "format_valid": answer["format_valid"],
        "reader_completed": True,
    }
    for key, value in expected.items():
        if result[key] != value:
            raise ValueError(f"raw/result mismatch: {key}")
    return result, response


def audit_readiness(
    root: Path, manifest: dict[str, Any], panel: list[dict[str, Any]]
) -> dict[str, Any]:
    """Retain every slot, verify raw decoding settings, and compare the original summary."""
    variants = manifest["pilot"]["variants"]
    profile = manifest["llm_profiles"]["reader"]
    records, responses = [], []
    if len(list(root.glob("*/*/result.json"))) != len(panel) * len(variants):
        raise ValueError("missing or extra pilot result slots")
    for row in panel:
        for variant in variants:
            folder = root / row["id"] / variant
            request = json.loads((folder / "request.json").read_text())["request"]
            expected_settings = {
                "temperature": profile["temperature"],
                "max_tokens": profile["max_tokens"],
                **profile["request_params"],
            }
            if {
                k: v for k, v in request.items() if k != "messages"
            } != expected_settings:
                raise ValueError("reader request settings drift")
            record, response = audit_slot(folder, row, variant, profile["model"])
            records.append(record)
            responses.append(response)
    summary = pilot.summarize(records, manifest)
    if summary != json.loads((root / "summary.json").read_text()):
        raise ValueError("raw recomputation differs from the preserved summary")
    return {
        "status": "RAW_RECOMPUTATION_IDENTICAL",
        "summary": summary,
        "F1": {
            variant: statistics.mean(
                r["answer_F1"] for r in records if r["variant"] == variant
            )
            for variant in variants
        },
        "correct_by_stratum": {
            variant: {
                kind: sum(
                    r["answer_EM"]
                    for r in records
                    if r["variant"] == variant and r["type"] == kind
                )
                for kind in manifest["dataset"]["strata"]
            }
            for variant in variants
        },
        "finish_reasons": dict(
            Counter(r["response"]["choices"][0]["finish_reason"] for r in responses)
        ),
        "upstream_providers": dict(
            Counter(r["response"].get("provider") for r in responses)
        ),
        "maximum_completion_tokens": max(
            r["usage"]["completion_tokens"] for r in responses
        ),
        "transport_events": dict(
            Counter(
                json.loads(p.read_text())["event"]
                for p in root.glob("*/*/transport_*.json")
            )
        ),
        "live_calls_by_audit": 0,
        "holdout_evaluations_by_audit": 0,
    }


def main() -> None:
    """Read the completed recovery pilot and print a deterministic, offline-only audit."""
    root = prepare.ROOT / "experiments/recursive_opt/_shared/o1_learning/exp20"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=root / "pilot_recovery_manifest.json"
    )
    parser.add_argument("--results", type=Path, default=root / "pilot_recovery")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    panel = prepare.verify_manifest(manifest)["pilot_diagnostic"]
    print(json.dumps(audit_readiness(args.results, manifest, panel), indent=2))


if __name__ == "__main__":
    main()
