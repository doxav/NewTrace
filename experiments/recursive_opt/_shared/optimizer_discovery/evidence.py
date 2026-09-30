"""Read-only EXP-15 provenance and descriptive analysis helpers."""

from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any


def environment() -> dict[str, Any]:
    """Fingerprint Python, platform and all installed distributions without secrets."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": dict(
            sorted(
                (d.metadata["Name"].lower(), d.version)
                for d in metadata.distributions()
            )
        ),
    }


def safe_metadata(data: dict[str, Any]) -> dict[str, Any]:
    """Whitelist routing, finish, usage and billing fields from OpenRouter metadata."""
    keys = (
        "id",
        "model",
        "provider_name",
        "generation_time",
        "latency",
        "finish_reason",
        "native_finish_reason",
        "tokens_prompt",
        "tokens_completion",
        "native_tokens_prompt",
        "native_tokens_completion",
        "native_tokens_reasoning",
        "total_cost",
        "upstream_inference_cost",
    )
    return {key: data[key] for key in keys if key in data}


def describe(
    proposals: list[dict[str, Any]], trajectories: list[dict[str, Any]]
) -> dict[str, Any]:
    """Describe every allocated candidate/trajectory without treating invalidity as regret."""
    usage = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",
        "cost_usd",
    ):
        reported = [
            p["usage"][key] for p in proposals if p["usage"].get(key) is not None
        ]
        usage[key] = {
            "reported_sum": sum(reported) if reported else None,
            "reported_responses": len(reported),
            "missing_responses": len(proposals) - len(reported),
        }
    complexity = []
    for p in proposals:
        try:
            nodes = sum(1 for _ in ast.walk(ast.parse(p["source"])))
        except SyntaxError:
            nodes = None
        complexity.append(
            {"source_bytes": len(p["source"].encode()), "ast_nodes": nodes}
        )
    statuses = Counter(
        a["status"] for r in trajectories for a in r["proposal_attempts"]
    )
    attempts = sum(statuses.values())
    return {
        "responses": len(proposals),
        "source_invalid_fraction": (
            sum(p["source_status"] != "valid" for p in proposals) / len(proposals)
            if proposals
            else None
        ),
        "trajectories": len(trajectories),
        "trajectory_invalid_fraction": (
            sum(not r["valid"] for r in trajectories) / len(trajectories)
            if trajectories
            else None
        ),
        "candidate_invalid_fraction": (
            sum(not r["candidate_valid"] for r in trajectories) / len(trajectories)
            if trajectories
            else None
        ),
        "fallback_trajectories": sum(r["fallback_used"] for r in trajectories),
        "proposal_status_counts": dict(statuses),
        "valid_proposal_fraction": statuses["valid"] / attempts if attempts else None,
        "timeout_fraction": statuses["timeout"] / attempts if attempts else None,
        "exception_fraction": statuses["exception"] / attempts if attempts else None,
        "logical_execution_s": sum(r["execution_s"] for r in trajectories),
        "target_evaluations": [
            r["metrics"]["target_evaluations"] if r["metrics"] else None
            for r in trajectories
        ],
        "usage": usage,
        "source_complexity": complexity,
    }


def collect_metadata(root: Path) -> None:
    """Fetch safe billing/routing receipts without issuing generative model calls."""
    from experiments.recursive_opt._shared.optimizer_discovery.exp15 import persist, read
    from experiments.recursive_opt._shared.optimizer_discovery.phase0 import _load_key

    _load_key()
    for path in sorted(root.glob("*/A*/slot_*/response.json")):
        target = path.parent / "provider_generation.json"
        if target.exists():
            continue
        identifier = read(path)["id"]
        if not identifier:
            continue
        url = "https://openrouter.ai/api/v1/generation?" + urllib.parse.urlencode(
            {"id": identifier}
        )
        request = urllib.request.Request(
            url, headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]}
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.load(response)["data"]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            receipt = path.parent / "metadata_attempts"
            attempt = len(list(receipt.glob("*.json"))) + 1
            persist(
                receipt / f"{attempt}.json",
                {
                    "error_type": type(error).__name__,
                    "http_status": getattr(error, "code", None),
                    "id": identifier,
                },
            )
            continue
        persist(target, safe_metadata(data))


def main() -> None:
    """Collect provider receipts independently of generation and scientific selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    collect_metadata(args.root)


if __name__ == "__main__":
    main()


def verify(exp: Any) -> dict[str, Any]:
    """Audit exact slots, sources, selections and split chronology from retained evidence."""
    from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
    from experiments.recursive_opt._shared.optimizer_discovery.exp15 import INVARIANT, read

    exp.freeze_selections()
    frozen = read(exp.root / "selections_frozen.json")
    count = 0
    for outer in exp.seeds:
        selection = read(exp.root / str(outer) / "selection.json")
        generation_end = max(
            read(exp.root / str(outer) / arm / "generation_complete.json")[
                "completed_ns"
            ]
            for arm in ("A1", "A2")
        )
        for arm in ("A1", "A2"):
            directory = exp.root / str(outer) / arm
            pool = read(directory / "pool.json")
            if [c["index"] for c in pool] != list(range(-1, exp.slots)):
                raise RuntimeError("candidate pool omitted an allocated slot")
            prior = {B.MANIFEST["seed_sha256"]}
            for slot in range(exp.slots):
                response = read(directory / f"slot_{slot:02d}" / "response.json")
                request = read(directory / f"slot_{slot:02d}" / "request.json")
                settings = {
                    k: B.MANIFEST["model"][k]
                    for k in (
                        "temperature",
                        "top_p",
                        "max_tokens",
                        "extra_body",
                        "timeout",
                    )
                }
                settings.update(
                    seed=B.stable_seed("request", exp.phase, outer, slot), num_retries=0
                )
                if (
                    request["settings"] != settings
                    or request["model"] != B.MANIFEST["model"]["model"]
                ):
                    raise RuntimeError(
                        "actual request differs from registered configuration"
                    )
                if (request["outer_seed"], request["arm"], request["slot"]) != (
                    outer,
                    arm,
                    slot,
                ):
                    raise RuntimeError("proposal slot identity mismatch")
                if request["messages"][0] != {
                    "role": "user",
                    "content": INVARIANT + B.SEED_SOURCE,
                }:
                    raise RuntimeError("invariant prompt differs between arms")
                if arm == "A1" and len(request["messages"]) != 1:
                    raise RuntimeError("independent arm received feedback")
                if request["parent_sha256"] not in prior:
                    raise RuntimeError(
                        "lineage does not originate from an earlier artifact"
                    )
                if (
                    response["source_sha256"] != B.source_hash(response["source"])
                    or response["source"] != pool[slot + 1]["source"]
                ):
                    raise RuntimeError("raw candidate source hash mismatch")
                if (
                    response["completed_ns"] > generation_end
                    or generation_end >= frozen["frozen_ns"]
                ):
                    raise RuntimeError("generation occurred after selection freeze")
                if arm == "A2":
                    prior.add(response["source_sha256"])
                count += 1
            for candidate in pool:
                if candidate["source_sha256"] != B.source_hash(candidate["source"]):
                    raise RuntimeError("pool source hash mismatch")
                for split in ("train", "validation"):
                    expected = [
                        (B.task_identity(t), B.local_seed(exp.phase, outer, t))
                        for t in B.make_tasks(exp.phase, split)
                    ]
                    actual = [
                        (r["task_identity"], r["local_seed"]) for r in candidate[split]
                    ]
                    if actual != expected or any(
                        r["budget"] != exp.budget for r in candidate[split]
                    ):
                        raise RuntimeError("unequal candidate evaluation allocation")
                eligible = all(
                    r["valid"] for r in [*candidate["train"], *candidate["validation"]]
                )
                if candidate["eligible"] != eligible:
                    raise RuntimeError("candidate eligibility mismatch")
                auc = B.aggregate(candidate["validation"], "auc") if eligible else None
                if candidate["validation_auc"] != auc:
                    raise RuntimeError("validation metric mismatch")
            best = min(
                (c for c in pool if c["eligible"]),
                key=lambda c: (c["validation_auc"], c["index"]),
            )
            if selection[arm] != {
                k: best[k]
                for k in ("index", "source", "source_sha256", "validation_auc")
            }:
                raise RuntimeError("selection is not the registered validation minimum")
        heldout = read(exp.root / str(outer) / "holdout.json")
        for arm in ("A0", "A1", "A2"):
            expected_hash = (
                B.MANIFEST["seed_sha256"]
                if arm == "A0"
                else selection[arm]["source_sha256"]
            )
            expected_tasks = [
                B.task_identity(t) for t in B.make_tasks(exp.phase, "holdout")
            ]
            if [r["task_identity"] for r in heldout[arm]] != expected_tasks:
                raise RuntimeError("holdout instance mismatch")
            for row, task in zip(heldout[arm], B.make_tasks(exp.phase, "holdout")):
                if row["source_sha256"] != expected_hash or row[
                    "local_seed"
                ] != B.local_seed(exp.phase, outer, task):
                    raise RuntimeError("holdout deployment identity mismatch")
                if row["metrics"] != B.metrics(
                    [o["value"] for o in row["observations"]],
                    B.normalization(task),
                    exp.budget,
                ):
                    raise RuntimeError("holdout metric recomputation mismatch")
        events = [
            json.loads(line)
            for line in (exp.root / "events.jsonl").read_text().splitlines()
        ]
        for event in events:
            if event["event"] != "evaluation":
                continue
            split = event["key"]["split"]
            if split == "holdout" and event["time_ns"] <= frozen["frozen_ns"]:
                raise RuntimeError("holdout accessed before global selection freeze")
            if (
                split == "validation"
                and event["key"]["local_seed"]
                in {
                    B.local_seed(exp.phase, outer, t)
                    for t in B.make_tasks(exp.phase, "validation")
                }
                and event["time_ns"] <= generation_end
            ):
                raise RuntimeError("validation reached unfinished generation")
    return {
        "completed_responses": count,
        "outer_seeds": exp.seeds,
        "source_hashes_verified": True,
        "requests_match_manifest": True,
        "equal_allocations_verified": True,
        "selection_and_holdout_chronology_verified": True,
        "metrics_recomputed": True,
    }
