"""Check the retrospective's local evidence, arithmetic, links, and saved fixtures."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parents[1] / "_shared"
EXP23 = HERE.parents[1] / "EXP23"
ASSESSMENT = HERE.parents[1] / "ASSESSMENT.md"


def read_json(path: Path) -> Any:
    """Read a JSON evidence file, including gzip archives."""
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def same(actual: float, expected: float, label: str) -> None:
    """Reject an arithmetic mismatch beyond floating point rounding."""
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"Arithmetic mismatch: {label}")


def main() -> None:
    """Recompute reported aggregates without rerunning models or evaluators."""
    report: dict[str, Any] = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Saved aggregate arithmetic and fixture consistency; no live calls or full raw-trajectory replay",
    }
    for name, relative in {
        "EXP15": "optimizer_discovery/exp15_results.json",
        "EXP16": "optimizer_discovery/investigation16/production_run/analysis_results.json.gz",
        "EXP18": "optimizer_discovery/exp18/run/analysis_results.json.gz",
    }.items():
        data = read_json(ARTIFACTS / relative)
        means: dict[str, float] = {}
        for arm, values in data["arms"].items():
            samples = (
                [row[arm]["auc"] for row in data["per_seed"]]
                if name == "EXP15"
                else values["auc"]["per_seed"]
            )
            expected = values["mean_auc"] if name == "EXP15" else values["auc"]["mean"]
            same(mean(samples), expected, f"{name}/{arm}")
            means[arm] = mean(samples)
        contrasts: dict[str, Any] = {}
        for label, values in data["contrasts"].items():
            same(mean(values["deltas"]), values["mean"], f"{name}/{label}")
            if "-" in label:
                first, second = label.split("-")
                if first in means and second in means:
                    same(
                        means[first] - means[second],
                        values["mean"],
                        f"{name}/{label} arms",
                    )
            contrasts[label] = {
                key: values[key]
                for key in ("mean", "paired_bootstrap_95", "interpretation")
            }
        if name == "EXP18":
            same(
                (means["M"] - means["L"] + means["PM"] - means["P"]) / 2,
                contrasts["memory"]["mean"],
                "memory",
            )
            same(
                (means["P"] - means["L"] + means["PM"] - means["M"]) / 2,
                contrasts["pareto"]["mean"],
                "pareto",
            )
            same(
                means["PM"] - means["P"] - means["M"] + means["L"],
                contrasts["interaction"]["mean"],
                "interaction",
            )
        report[name] = {"arm_means": means, "contrasts": contrasts}

    s4 = read_json(ARTIFACTS / "o1_learning/s4_results.json")
    deltas = [
        s4["arms"]["train_only_fit"][seed] - score
        for seed, score in s4["arms"]["standard_B6"].items()
    ]
    same(mean(deltas), s4["train_only_minus_standard"]["mean"], "EXP19/S4")
    report["EXP19_S4"] = s4["train_only_minus_standard"]
    exp20 = read_json(ARTIFACTS / "o1_learning/exp20/full/confirmation/results.json")
    report["EXP20"] = {}
    for arm, values in exp20["arms"].items():
        same(
            mean(values["primary_per_seed"].values()),
            values["primary_mean"],
            f"EXP20/{arm}/primary",
        )
        same(mean(values["final_per_seed"]), values["final_mean"], f"EXP20/{arm}/final")
        for seed, curve in values["curves"].items():
            same(
                mean(curve),
                values["primary_per_seed"][seed],
                f"EXP20/{arm}/{seed}/curve",
            )
        report["EXP20"][arm] = {
            key: values[key] for key in ("primary_mean", "final_mean")
        }
    for first, second in (("standard", "unchanged"), ("curriculum", "standard")):
        contrast = exp20[f"{first}_minus_{second}"]
        same(
            exp20["arms"][first]["primary_mean"]
            - exp20["arms"][second]["primary_mean"],
            contrast["mean_delta"],
            f"EXP20/{first}-{second}",
        )
        report["EXP20"][f"{first}-{second}"] = contrast

    fixtures: dict[str, Any] = {}
    roles = ("solution", "meta", "stats_insight", "problem_context", "batch_summary")
    for name in ("base", "labels", "long", "meta_failure", "rollback_early"):
        folder = EXP23 / "equivalence/out" / name
        stock, native = (
            read_json(folder / f"{kind}_trace.json") for kind in ("stock", "v2")
        )
        for key in ("events", "curve", "best_score", "meta_failures"):
            if stock[key] != native[key]:
                raise ValueError(f"Saved fixture differs: {name}/{key}")
        if any(
            stock["calls"].get(role, 0) != native["calls"].get(role, 0)
            for role in roles
        ):
            raise ValueError(f"Saved fixture call counts differ: {name}")
        fixtures[name] = {
            "events": len(stock["events"]),
            "calls": {role: stock["calls"].get(role, 0) for role in roles},
            "input_sha256": {
                kind: hashlib.sha256(
                    (folder / f"{kind}_trace.json").read_bytes()
                ).hexdigest()
                for kind in ("stock", "v2")
            },
        }
    report["EXP23_saved_fixture_comparison"] = {
        "fixtures": fixtures,
        "excluded": "stock availability pings",
    }
    report["EXP23_prism100"] = {}
    for path in sorted(
        (EXP23 / "results/prism100/20260929T164311").glob("*/result.json")
    ):
        data = read_json(path)
        report["EXP23_prism100"][path.parent.name] = {
            key: data[key]
            for key in (
                "iterations",
                "final_best_score",
                "valid_candidates",
                "optimizer_calls_n",
                "gate_failure",
                "error",
            )
        }
        report["EXP23_prism100"][path.parent.name]["policy_event_iterations"] = [
            event["iteration"] for event in data["policy_events"]
        ]

    native_checkpoint: dict[str, Any] = {}
    for name in ("trace", "llm_rewrite"):
        folder = EXP23 / "results/prism100_v2/20260929T215903" / name
        native_checkpoint[name] = {
            "path": str(folder),
            "final_result_files": sorted(
                filename
                for filename in ("result.json", "report.json", "summary.json")
                if (folder / filename).exists()
            ),
            "recorded_lines": {
                filename: len((folder / filename).read_bytes().splitlines())
                for filename in ("events.jsonl", "calls.jsonl")
            },
        }
    report["EXP23_native_live_checkpoint"] = native_checkpoint

    snapshot_path = HERE / "exp23_intermediate.json"
    snapshot = read_json(snapshot_path)
    common = min(
        arm["completed_solution_attempt_coordinate"]
        for arm in snapshot["arms"].values()
    )
    same(
        common,
        snapshot["common_completed_solution_attempt_coordinate"],
        "EXP23 common prefix",
    )
    intermediate: dict[str, Any] = {}
    for name, arm in snapshot["arms"].items():
        candidates = [
            event for event in arm["events"] if event.get("child_score") is not None
        ]
        best = max(event["child_score"] for event in candidates)
        prefix_best = max(
            event["child_score"]
            for event in candidates
            if event["iteration"] + event.get("attempts", 1) - 1 <= common
        )
        same(best, arm["best_event"]["child_score"], f"EXP23/{name}/best")
        same(
            prefix_best,
            arm["common_prefix_best_event"]["child_score"],
            f"EXP23/{name}/prefix",
        )
        intermediate[name] = {"best": best, "common_prefix_best": prefix_best}
    configs = [
        snapshot["arms"][name]["engine"]["config"] for name in ("trace", "llm_rewrite")
    ]
    differences = sorted(
        key
        for key in configs[0].keys() | configs[1].keys()
        if configs[0].get(key) != configs[1].get(key)
    )
    if differences != ["proposer"]:
        raise ValueError(
            "EXP23 intermediate engine configurations differ beyond proposer"
        )
    report["EXP23_intermediate"] = {
        "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        "captured_utc": snapshot["captured_utc"],
        "common_completed_prefix": common,
        "arms": intermediate,
        "engine_config_differences": differences,
    }

    followup_path = HERE / "exp23_followup.json"
    if followup_path.exists():
        followup = read_json(followup_path)
        common = min(
            arm["completed_solution_attempt_coordinate"]
            for arm in followup["arms"].values()
        )
        same(
            common,
            followup["common_completed_solution_attempt_coordinate"],
            "followup prefix",
        )
        for name, arm in followup["arms"].items():
            candidates = [e for e in arm["events"] if e.get("child_score") is not None]
            same(max(e["child_score"] for e in candidates), arm["best_score"], name)
            prefix = [
                e["child_score"]
                for e in candidates
                if e["iteration"] + e.get("attempts", 1) - 1 <= common
            ]
            same(max(prefix), arm["common_prefix_best_score"], f"{name}/followup")
            if arm["summary"]:
                same(
                    arm["summary"]["best_score"], arm["best_score"], f"{name}/terminal"
                )
        report["EXP23_followup"] = {
            "snapshot_sha256": hashlib.sha256(followup_path.read_bytes()).hexdigest(),
            "captured_utc": followup["captured_utc"],
            "common_completed_prefix": common,
            "scope": "Single live trajectory per proposer; comparison remains interim",
        }

    document = ASSESSMENT.read_text()
    anchors = set(re.findall(r'<a id="([^"]+)"', document))
    references = dict(
        re.findall(r"^\[([^\]]+)\]:\s+(\S+)", document, flags=re.MULTILINE)
    )
    inline = re.findall(r"\]\(([^\s)]+)\)", document)
    used = re.findall(r"\[[^\]\n]+\]\[([^\]\n]+)\]", document)
    missing = set(used) - references.keys()
    if missing:
        raise ValueError(f"Undefined document references: {sorted(missing)}")
    sources: list[dict[str, Any]] = []
    for target in sorted(set(inline + list(references.values()))):
        if target.startswith(("http:", "https:")):
            continue
        location, _, fragment = unquote(target).partition("#")
        if not location:
            if fragment not in anchors:
                raise ValueError(f"Missing local TOC anchor: {fragment}")
            continue
        path = (ASSESSMENT.parent / location).resolve()
        if not path.exists():
            raise ValueError(f"Missing document link: {target}")
        if path.is_file() and path.parent != HERE:
            sources.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    table_width: int | None = None
    tables = 0
    for line in document.splitlines():
        if line.startswith("|"):
            width = len(re.split(r"(?<!\\)\|", line))
            if table_width is None:
                table_width = width
                tables += 1
            elif width != table_width:
                raise ValueError(f"Inconsistent Markdown table width: {line[:80]}")
        else:
            table_width = None
    report["document"] = {
        "references": len(references),
        "explicit_anchors": len(anchors),
        "tables": tables,
        "words": len(document.split()),
        "bytes": ASSESSMENT.stat().st_size,
    }
    report["source_files"] = sources
    report["status"] = "PASS"
    (HERE / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "document": report["document"],
                "saved_fixtures": len(fixtures),
                "hashed_source_links": len(sources),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
