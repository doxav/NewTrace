"""CLI for Experiment 0 preflight and offline contract stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baseline import build_baseline_token_manifest
from .datasets import dataset_manifest_v2
from .forecast import build_cost_forecast
from .live import run_micro_smoke, run_pilot
from .offline_contract import run_offline_contract
from .preflight import run_task_eligibility_preflight_v2
from .provenance import build_control_plane_lock_after_gepa_reflection_fix
from .registration import assert_strict_output_evaluator, register_experiment_components


PACKAGE_ROOT = Path(__file__).resolve().parent


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _offline_report(result: dict[str, object]) -> str:
    lines = [
        "# Experiment 0 offline contract report",
        "",
        f"Overall: **{'PASS' if result['passed'] else 'FAIL'}**",
        "",
        f"- runtime_tree_sha256: `{result['runtime_tree_sha256']}`",
        f"- registry_sha256: `{result['registry_sha256']}`",
        "- provider/network calls: none (deterministic local provider)",
        "",
        "## Assertions",
        "",
        "| assertion | result |",
        "|---|---|",
    ]
    assertions = result["assertions"]
    assert isinstance(assertions, dict)
    lines.extend(
        f"| {name} | {'pass' if passed else 'FAIL'} |"
        for name, passed in assertions.items()
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "dataset-manifest-v2",
            "offline-contract",
            "task-preflight-v2",
            "baseline-token-manifest",
            "control-plane-lock-after-gepa-reflection-fix",
            "micro-smoke",
            "cost-forecast",
            "pilot",
        ),
    )
    args = parser.parse_args()
    register_experiment_components()
    assert_strict_output_evaluator()
    if args.command == "dataset-manifest-v2":
        manifest = dataset_manifest_v2()
        _write_json(PACKAGE_ROOT / "manifests" / "dataset_manifest_v2.json", manifest)
        return 0
    if args.command == "task-preflight-v2":
        result = run_task_eligibility_preflight_v2()
        _write_json(PACKAGE_ROOT / "reports" / "task_eligibility_v2.json", result)
        return 0 if result["passed"] else 1
    if args.command == "baseline-token-manifest":
        result = build_baseline_token_manifest()
        _write_json(PACKAGE_ROOT / "baseline_token_manifest.json", result)
        return 0
    if args.command == "control-plane-lock-after-gepa-reflection-fix":
        result = build_control_plane_lock_after_gepa_reflection_fix()
        _write_json(
            PACKAGE_ROOT / "control_plane_lock_after_gepa_reflection_fix.json",
            result,
        )
        return 0
    if args.command == "micro-smoke":
        result = run_micro_smoke()
        _write_json(PACKAGE_ROOT / "reports" / "live_micro_smoke.json", result)
        return 0 if result["passed"] else 1
    if args.command == "cost-forecast":
        result = build_cost_forecast()
        _write_json(PACKAGE_ROOT / "reports" / "cost_forecast.json", result)
        return 0
    if args.command == "pilot":
        result = run_pilot()
        _write_json(PACKAGE_ROOT / "reports" / "pilot.json", result)
        return 0 if result["passed"] else 1
    result = run_offline_contract()
    _write_json(PACKAGE_ROOT / "reports" / "offline_contract_report.json", result)
    report = PACKAGE_ROOT / "reports" / "offline_contract_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_offline_report(result), encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
