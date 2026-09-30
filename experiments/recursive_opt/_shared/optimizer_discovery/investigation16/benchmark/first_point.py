"""Run the single preregistered B2 first-point ablation using preserved B1 controls."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from experiments.recursive_opt._shared.optimizer_discovery.investigation16.benchmark import headroom as H
from opto.features.recursive_opt import optimizer_program as O

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "b2"
METRICS = ["auc", "final_regret", "attained", "capped_target_evaluations"]


def variant_source() -> str:
    """Alter only the empty-history branch in the exact frozen handwritten seed."""
    original = "    if not history or rng.random() < 0.25:\n"
    replacement = (
        "    if not history:\n"
        "        return [(low + high) / 2 for low, high in bounds]\n"
        "    if rng.random() < 0.25:\n"
    )
    if B.SEED_SOURCE.count(original) != 1:
        raise ValueError("the expected original seed conditional is not unique")
    source = B.SEED_SOURCE.replace(original, replacement)
    if B.source_status(source) != "valid":
        raise ValueError("first-point ablation violates the source protocol")
    return source


def _check_row(row: dict[str, Any], entry: dict[str, Any], source: str) -> None:
    """Reject reused evidence from another source, task, local seed or budget."""
    expected = {
        "source_sha256": B.source_hash(source),
        "task_identity": B.task_identity(entry["task"]),
        "local_seed": entry["local_seed"],
        "budget": 32,
        "stratum": f'{entry["task"]["family"]}/{entry["task"]["dimension"]}',
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError("trajectory differs from its frozen paired allocation")


def _payload() -> dict[str, Any]:
    """Reconstruct input identity without running objectives or candidate programs."""
    b1 = E.read(ROOT / "freeze.json")
    if b1["sources"]["seed"] != B.SEED_SOURCE or b1["budget"] != 32:
        raise ValueError("B1 control seed or budget differs from B2")
    if b1["local_seeds"] != [16201, 16202, 16203, 16204]:
        raise ValueError("B1 local seeds changed")
    entries = []
    for index, pair in enumerate(b1["pairs"]):
        for condition in ["central", "broad"]:
            for seed in b1["local_seeds"]:
                path = Path("raw") / condition / "seed" / f"{seed}_{index:02d}.json"
                if not E.exists(ROOT / path):
                    raise ValueError("required B1 control is missing")
                entry = {
                    "id": f"{condition}/{seed}_{index:02d}",
                    "condition": condition,
                    "task_index": index,
                    "task": pair[condition],
                    "local_seed": seed,
                    "control_path": str(path),
                }
                row = E.read(ROOT / path)
                _check_row(row, entry, B.SEED_SOURCE)
                if not row["valid"] or row["objective_calls"] != 32:
                    raise ValueError("B1 seed control is not a complete trajectory")
                entries.append({**entry, "control_sha256": B.digest(row)})
    if len(entries) != 96 or len({entry["id"] for entry in entries}) != 96:
        raise ValueError("B2 requires exactly the 96 distinct B1 paired controls")
    source = variant_source()
    paths = {
        "runner": Path(__file__),
        "benchmark": Path(B.__file__),
        "benchmark_manifest": B.ROOT / "exp15_manifest.json",
        "artifact_boundary": Path(O.__file__),
        "summary": Path(H.__file__),
        "writer": Path(I.__file__),
        "protocol": ROOT / "B2_PROTOCOL.md",
    }
    return {
        "stage": "EXP-16/B2 exploratory first-point ablation on public B1 fixtures",
        "b1_freeze_sha256": B.digest(b1),
        "budget": 32,
        "timeout_s": 2.0,
        "workers": 8,
        "deployment": False,
        "seed_source": B.SEED_SOURCE,
        "seed_source_sha256": B.source_hash(B.SEED_SOURCE),
        "variant_source": source,
        "variant_source_sha256": B.source_hash(source),
        "allocations": entries,
        "normalization": b1["normalization"],
        "allocated_objective_calls": 3072,
        "maximum_subprocess_executions": 6144,
        "normalization_reference_allocations_per_process": 3072,
        "source_hashes": {
            name: B.source_hash(path.read_text()) for name, path in paths.items()
        },
    }


def prepare(output: Path = OUTPUT) -> dict[str, Any]:
    """Persist the exact protocol and control identities before variant evaluation."""
    if E.exists(output / "freeze.json"):
        return preflight(output)
    frozen = {"created_ns": time.time_ns(), **_payload()}
    I.persist(output / "freeze.json", frozen)
    return frozen


def preflight(output: Path = OUTPUT) -> dict[str, Any]:
    """Rebuild and compare every frozen source, setting, input and control hash."""
    frozen = E.read(output / "freeze.json")
    expected = {"created_ns": frozen["created_ns"], **_payload()}
    if frozen != expected:
        raise ValueError("B2 frozen protocol or control evidence changed")
    return frozen


def _path(output: Path, entry: dict[str, Any]) -> Path:
    """Address one new trajectory independently from the preserved B1 controls."""
    return output / "raw" / f'{entry["id"]}.json'


def _evaluate(
    output: Path, frozen: dict[str, Any], entry: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate only a missing variant allocation, preserving completed rows exactly."""
    path = _path(output, entry)
    if E.exists(path):
        row = E.read(path)
    else:
        row = B.evaluate(
            frozen["variant_source"],
            entry["task"],
            entry["local_seed"],
            budget=frozen["budget"],
            deployment=frozen["deployment"],
            timeout_s=frozen["timeout_s"],
        )
        I.persist(path, row)
    _check_row(row, entry, frozen["variant_source"])
    return row


def compare(
    controls: list[dict[str, Any]], variants: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare complete paired panels without discarding any invalid trajectory."""
    if not controls or len(controls) != len(variants):
        raise ValueError("comparison requires nonempty equally sized paired panels")
    for control, variant in zip(controls, variants):
        if any(
            control[key] != variant[key]
            for key in ["task_identity", "local_seed", "budget", "stratum"]
        ):
            raise ValueError("comparison contains mismatched paired inputs")
    result = {"seed": H._summarize(controls), "variant": H._summarize(variants)}
    if any(not row["valid"] for row in controls + variants):
        return {**result, "delta": None}
    paired = []
    for control, variant in zip(controls, variants):
        delta = {
            metric: variant["metrics"][metric] - control["metrics"][metric]
            for metric in METRICS
        }
        first = (variant["metrics"]["curve"][0] - control["metrics"]["curve"][0]) / 32
        paired.append(
            {
                "valid": True,
                "stratum": control["stratum"],
                "metrics": {
                    **delta,
                    "first_term_auc": first,
                    "later_terms_auc": delta["auc"] - first,
                },
            }
        )
    return {
        **result,
        "delta": {
            metric: B.aggregate(paired, metric)
            for metric in METRICS + ["first_term_auc", "later_terms_auc"]
        },
        "auc_pair_counts": {
            "variant_lower": sum(row["metrics"]["auc"] < 0 for row in paired),
            "equal": sum(row["metrics"]["auc"] == 0 for row in paired),
            "variant_higher": sum(row["metrics"]["auc"] > 0 for row in paired),
        },
    }


def analyze(output: Path = OUTPUT) -> dict[str, Any]:
    """Recompute all paired summaries from every preserved allocation without execution."""
    frozen = preflight(output)
    pairs = []
    for entry in frozen["allocations"]:
        if not E.exists(_path(output, entry)):
            raise ValueError("analysis requires every allocated B2 trajectory")
        control = E.read(ROOT / entry["control_path"])
        variant = E.read(_path(output, entry))
        _check_row(variant, entry, frozen["variant_source"])
        for row in [control, variant]:
            expected = (
                B.metrics(
                    [observation["value"] for observation in row["observations"]],
                    frozen["normalization"][row["task_identity"]],
                    32,
                )
                if row["valid"]
                else None
            )
            if row["metrics"] != expected:
                raise ValueError("persisted metrics differ from the frozen values")
        pairs.append({"allocation": entry, "seed": control, "variant": variant})

    def panel(selected: list[dict[str, Any]]) -> dict[str, Any]:
        """Apply the same paired summary to a predeclared subset."""
        return compare(
            [pair["seed"] for pair in selected],
            [pair["variant"] for pair in selected],
        )

    groups = {}
    for condition in ["central", "broad"]:
        selected = [
            pair for pair in pairs if pair["allocation"]["condition"] == condition
        ]
        groups[condition] = {
            "overall": panel(selected),
            "per_local_seed": {
                str(seed): panel(
                    [pair for pair in selected if pair["seed"]["local_seed"] == seed]
                )
                for seed in [16201, 16202, 16203, 16204]
            },
            "per_stratum": {
                stratum: panel(
                    [pair for pair in selected if pair["seed"]["stratum"] == stratum]
                )
                for stratum in sorted({pair["seed"]["stratum"] for pair in selected})
            },
        }
    variants = [pair["variant"] for pair in pairs]
    return {
        "stage": frozen["stage"],
        "freeze_sha256": B.digest(frozen),
        "groups": groups,
        "new_trajectories": len(variants),
        "reused_control_trajectories": len(pairs),
        "new_objective_calls": sum(row["objective_calls"] for row in variants),
        "new_subprocess_executions": sum(
            row["subprocess_executions"] for row in variants
        ),
        "new_execution_s": sum(row["execution_s"] for row in variants),
        "unused_objective_allocations": sum(
            row["unused_objective_allocation"] for row in variants
        ),
        "paired_rows": [
            {
                "id": pair["allocation"]["id"],
                "control_sha256": B.digest(pair["seed"]),
                "variant_sha256": B.digest(pair["variant"]),
                "comparison": panel([pair]),
            }
            for pair in pairs
        ],
    }


def run(output: Path = OUTPUT) -> dict[str, Any]:
    """Run only unfinished trajectories after freezing and checking reference scales."""
    frozen = preflight(output)
    if E.exists(output / "results.json"):
        result = analyze(output)
        if result != E.read(output / "results.json"):
            raise ValueError("completed B2 result changed during recomputation")
        return result
    started = time.monotonic()
    misses_before = B._reference_scale.cache_info().misses
    tasks = {B.task_identity(row["task"]): row["task"] for row in frozen["allocations"]}
    for identity, task in tasks.items():
        if B.normalization(task) != frozen["normalization"][identity]:
            raise ValueError("B1 normalization changed before B2 evaluation")
    reference_calls = (
        B._reference_scale.cache_info().misses - misses_before
    ) * B.MANIFEST["normalization_reference_size"]
    pending = [
        entry for entry in frozen["allocations"] if not E.exists(_path(output, entry))
    ]
    attempt = len(list((output / "attempts").glob("*.json"))) + 1
    attempt_path = output / "attempts" / f"{attempt:03d}.json"
    I.persist(
        attempt_path,
        {
            "started_ns": time.time_ns(),
            "freeze_sha256": B.digest(frozen),
            "pending_ids": [entry["id"] for entry in pending],
            "normalization_reference_calls": reference_calls,
            "workers": frozen["workers"],
        },
    )
    with ThreadPoolExecutor(max_workers=frozen["workers"]) as executor:
        list(executor.map(lambda entry: _evaluate(output, frozen, entry), pending))
    result = analyze(output)
    I.persist(output / "results.json", result)
    I.persist(
        output / "timing" / f"{attempt:03d}.json",
        {"attempt": attempt, "elapsed_s": time.monotonic() - started},
    )
    return result


def main() -> None:
    """Expose separate pre-evaluation freeze, run and read-only analysis commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "analyze"])
    args = parser.parse_args()
    result = {"prepare": prepare, "run": run, "analyze": analyze}[args.action]()
    print(
        {
            "action": args.action,
            "digest": B.digest(result),
            "new_trajectories": result.get("new_trajectories"),
            "new_objective_calls": result.get("new_objective_calls"),
        }
    )


if __name__ == "__main__":
    main()
