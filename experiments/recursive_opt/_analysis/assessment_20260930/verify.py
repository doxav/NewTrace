"""Audit saved experiment arithmetic and current Markdown links without live calls."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
HASHES: dict[str, str] = {}


def read(path: Path) -> Any:
    """Read and fingerprint exactly the bytes used in this audit."""
    raw = path.read_bytes()
    HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    return json.loads(raw)


def same(actual: float, expected: float) -> None:
    """Fail on a discrepancy larger than numerical rounding."""
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"Arithmetic mismatch: {actual} versus {expected}")


def check_links() -> dict[str, Any]:
    """Check navigation and report links, resolving symlinks at canonical locations."""
    files = {ROOT / "ASSESSMENT.md", ROOT / "README.md"}
    for pattern in (
        "EXP*/RESULTS.md",
        "EXP*/README.md",
        "EXP22/*/RESULTS.md",
        "EXP22/*/README.md",
    ):
        files.update(path.resolve() for path in ROOT.glob(pattern))
    count = 0
    external = []
    for path in sorted(files):
        text = re.sub(r"```.*?```", "", path.read_text(), flags=re.DOTALL)
        text = re.sub(r"`[^`\n]+`", "", text)
        definitions = dict(re.findall(r"^\[([^\]]+)\]:\s*(\S+)", text, re.MULTILINE))
        urls = re.findall(r"\]\(([^)\s]+)\)", text) + list(definitions.values())
        for ref in re.findall(r"\]\[([^\]]+)\]", text):
            if ref not in definitions:
                raise ValueError(f"Undefined reference {ref} in {path}")
        for url in urls:
            if url.startswith(("https:", "http:", "mailto:")):
                continue
            location, _, anchor = unquote(url).partition("#")
            target = (path.parent / location).resolve() if location else path
            if not target.exists():
                raise ValueError(f"Missing link in {path}: {url}")
            if not target.is_relative_to(ROOT):
                external.append(
                    {"document": str(path.relative_to(ROOT)), "target": url}
                )
            if (
                not location
                and anchor
                and path.name == "ASSESSMENT.md"
                and f'id="{anchor}"' not in text
            ):
                raise ValueError(f"Missing assessment anchor: {anchor}")
            count += 1
    return {
        "documents": len(files),
        "local_links_checked": count,
        "external_code_links": external,
    }


def campaign(folder: Path, optimum: float) -> dict[str, Any]:
    """Read an EXP24 checkpoint; exclude transport retries from solution coordinates."""
    rows = []
    for run in sorted(folder.iterdir()):
        if not (run / "events.jsonl").exists():
            continue
        events = read(run / "events.jsonl")
        iterations = [event for event in events if event["type"] == "iteration"]
        calls = read(run / "calls.jsonl")
        summary = read(run / "summary.json") if (run / "summary.json").exists() else {}
        coordinate = 0
        first = None
        for event in iterations:
            coordinate += event["attempts"]
            if first is None and (event.get("child_score") or 0) >= optimum - 1e-6:
                first = coordinate
        if summary.get("status") == "success":
            same(coordinate, summary["solution_attempts"])
        rows.append(
            {
                "run": run.name,
                "terminal_status": summary.get("status"),
                "completed_solution_coordinate": coordinate,
                "calls_to_optimum": first,
                "best_valid_score": max(
                    event.get("child_score") or 0 for event in iterations
                ),
                "wasted_attempts": sum(
                    event["attempts"] - (event["error"] is None) for event in iterations
                ),
                "deployments": sum(
                    event["type"] == "deploy" and bool(event.get("ok"))
                    for event in events
                ),
                "recorded_http_calls": len(calls),
                "reported_cost_usd": sum(call.get("cost") or 0 for call in calls),
                "calls_missing_cost": sum(call.get("cost") is None for call in calls),
            }
        )
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["run"].rsplit("_s", 1)[0], []).append(row)
    return {
        "runs": rows,
        "arms": {
            arm: {
                "first_hits": [row["calls_to_optimum"] for row in group],
                "median_first_hit": (
                    median(row["calls_to_optimum"] for row in group)
                    if all(row["calls_to_optimum"] is not None for row in group)
                    else None
                ),
            }
            for arm, group in groups.items()
        },
    }


def main() -> None:
    """Recompute selected claims and print a dated, source-fingerprinted report."""
    report: dict[str, Any] = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Saved arithmetic, status, and Markdown paths; not a full historical replay or CI rebootstrap",
    }
    for name, relative in {
        "EXP15": "EXP15/results/exp15_results.json",
        "EXP16": "EXP16/results/production_run/analysis_results.json.gz",
        "EXP18": "EXP18/results/run/analysis_results.json.gz",
    }.items():
        data = read(ROOT / relative)
        means = {}
        for arm, values in data["arms"].items():
            samples = (
                [row[arm]["auc"] for row in data["per_seed"]]
                if name == "EXP15"
                else values["auc"]["per_seed"]
            )
            means[arm] = mean(samples)
            same(
                means[arm],
                values["mean_auc"] if name == "EXP15" else values["auc"]["mean"],
            )
        for label, values in data["contrasts"].items():
            same(mean(values["deltas"]), values["mean"])
            if "-" in label:
                first, second = label.split("-")
                same(means[first] - means[second], values["mean"])
        report[name] = {"means": means, "contrasts": data["contrasts"]}
    for name, initial, learned in (
        ("EXP03", "initial", "standard"),
        ("EXP04", "baseline", "optimized"),
    ):
        data = read(
            ROOT
            / name
            / "results"
            / ("probe_f_results.json" if name == "EXP03" else "probe_k_results.json")
        )
        paired = {
            seed: {
                row["arm"]: row["score"]
                for row in data["rows"]
                if row["seed"] == seed and row["score"] is not None
            }
            for seed in data["seeds"]
        }
        deltas = [
            row[learned] - row[initial]
            for row in paired.values()
            if initial in row and learned in row
        ]
        same(mean(deltas), mean(data["paired_deltas"]))
        report[name] = {"pairs": len(deltas), "deltas": deltas, "mean": mean(deltas)}
    aa = read(ROOT / "EXP13/results/probe_aa_results.json")
    report["EXP13"] = {
        "attempts": len(aa["order"]),
        "usable": sum(row["score"] is not None for row in aa["order"]),
    }
    s4 = read(ROOT / "EXP19/results/s4_results.json")
    deltas = [
        s4["arms"]["train_only_fit"][seed] - score
        for seed, score in s4["arms"]["standard_B6"].items()
    ]
    same(mean(deltas), s4["train_only_minus_standard"]["mean"])
    report["EXP19_S4"] = s4["train_only_minus_standard"]
    qa = read(ROOT / "EXP20/results/run_store/full/confirmation/results.json")
    for arm in qa["arms"].values():
        for seed, curve in arm["curves"].items():
            same(mean(curve), arm["primary_per_seed"][seed])
        same(mean(arm["primary_per_seed"].values()), arm["primary_mean"])
        same(mean(arm["final_per_seed"]), arm["final_mean"])
    report["EXP20"] = {
        name: {key: arm[key] for key in ("primary_mean", "final_mean")}
        for name, arm in qa["arms"].items()
    }
    for name, relative, fields in [
        (
            "EXP17",
            "_shared/optimizer_discovery/exp17/USER_REQUESTED_PAUSE.json",
            ["completed_responses", "barriers"],
        ),
        (
            "EXP21",
            "EXP21/results/run_store/continuation_summary.json",
            [
                "development_completed_responses",
                "complete_curves",
                "O1_real_executed",
                "O2_real_executed",
                "confirmation_started",
            ],
        ),
        (
            "EXP22_QA",
            "EXP22/qa/results/run_store/execution_summary.json",
            ["status", "confirmation_started", "lower_responses_remaining"],
        ),
    ]:
        data = read(ROOT / relative)
        report[name] = {key: data[key] for key in fields}
    report["EXP23_native"] = {}
    for arm in ("trace", "llm_rewrite"):
        folder = ROOT / "EXP23/results/prism100_v2/20260929T215903" / arm
        data, summary = read(folder / "report.json"), read(folder / "summary.json")
        same(data["best_score"], max(data["curve"]))
        report["EXP23_native"][arm] = {
            "status": summary["status"],
            "solution_attempts": data["solution_attempts"],
            "raw_best": data["best_score"],
            "best_metrics": data["best_metrics"],
            "llm_calls": data["llm_calls"],
        }
    optimum = read(ROOT / "EXP24/results/analysis/prism_exact_optimum.json")
    same(1 / mean(optimum["per_case"]) + 1, optimum["exact_optimal_score"])
    report["EXP24_optimum"] = optimum["exact_optimal_score"]
    report["EXP24_clean"] = campaign(
        ROOT / "EXP24/results/clean_20260930T115256", optimum["exact_optimal_score"]
    )
    report["links"] = check_links()
    report["source_sha256"] = HASHES
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
