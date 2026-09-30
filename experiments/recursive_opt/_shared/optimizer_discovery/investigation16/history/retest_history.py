"""Replay historical deterministic research fixtures without generating new code."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import socket
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from experiments.recursive_opt._history.probe_2026 import probe_r_menu_validity as historical_r
from experiments.recursive_opt._history.probe_2026 import probe_t_routing_menu as historical_t
from opto.features.recursive_opt import tracebench as tb
from opto.features.recursive_opt.levels import LevelConfig

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
PROBES = ROOT / "experiments/recursive_opt/_history/probe_2026"


def score_valid(value: Any) -> bool:
    """Recognize historical legal values, keeping the old invalid sentinel typed."""
    return isinstance(value, (int, float)) and math.isfinite(value) and abs(value) < 1e5


def normalize(row: dict[str, float]) -> dict[str, float]:
    """Reproduce historical min/max normalization only for valid varying rows."""
    if not row or not all(score_valid(value) for value in row.values()):
        raise ValueError("Normalization requires valid historical objective values")
    low, high = min(row.values()), max(row.values())
    if high <= low:
        raise ValueError("Normalization requires a nonconstant row")
    return {key: (value - low) / (high - low) for key, value in row.items()}


def exact_curves(
    row: dict[str, float], default: float, order: list[str]
) -> dict[int, dict[str, float]]:
    """Enumerate equal-size subsets and retain the common unchanged starting policy."""
    if len(order) != len(row) or set(order) != set(row):
        raise ValueError("Candidate order must be a complete permutation")
    if not row or not all(math.isfinite(value) for value in [default, *row.values()]):
        raise ValueError("Curves require a nonempty finite score table")
    result = {0: {"uniform_mean": default, "ordered": default}}
    for budget in range(1, len(row) + 1):
        bests = [
            max(default, *subset)
            for subset in itertools.combinations(row.values(), budget)
        ]
        result[budget] = {
            "uniform_mean": statistics.mean(bests),
            "ordered": max(default, *(row[name] for name in order[:budget])),
        }
    return result


def _write(path: Path, value: Any) -> None:
    """Preserve each replay artifact once, refusing to overwrite prior evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _read(name: str) -> Any:
    """Load preserved historical JSON without modifying it."""
    return json.loads((PROBES / name).read_text())


def _sha(path: Path) -> str:
    """Hash exact preserved bytes for provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(directory: Path, *arguments: str) -> str:
    """Capture local read-only Git metadata."""
    return subprocess.check_output(
        ["git", "-C", str(directory), *arguments], text=True
    ).strip()


def _blocked_socket(*args: Any, **kwargs: Any) -> None:
    """Refuse parent-process networking throughout the offline replay."""
    raise RuntimeError("Historical replay forbids network access")


def _score(adapter: Any, task: str, source: str | None) -> dict[str, Any]:
    """Evaluate exact historical source through the current real benchmark bridge."""
    bundle = adapter._load_bundle(task, fresh=True)
    node = adapter._trainable_node(bundle["param"])
    original = str(node._data)
    applied = None
    if source is not None:
        applied = adapter._apply_starting_artifact(
            bundle, LevelConfig(starting_artifact=source)
        )
    started = time.monotonic()
    try:
        score, feedback = tb._score_bundle(bundle, 2)
        value: Any = float(score)
    except (RuntimeError, ValueError, TypeError) as error:
        value = None
        feedback = f"{type(error).__name__}: {str(error)[:300]}"
    return {
        "score": value,
        "valid": score_valid(value),
        "source_sha256": hashlib.sha256((source or original).encode()).hexdigest(),
        "applied": applied,
        "parameter_preserved": str(node._data) == original,
        "feedback": feedback,
        "wall_s": time.monotonic() - started,
    }


def _w2(
    tables: dict[str, dict[str, float]], defaults: dict[str, float]
) -> dict[str, Any]:
    """Recompute leave-one-task-out ordering and an explicitly informed fixed control."""
    family = [task for task in tables if not task.endswith("ovrp_construct")]
    names = list(historical_t.MENU)
    folds = {}
    for target in family:
        sources = [task for task in family if task != target]
        normalized = {task: normalize(tables[task]) for task in sources}
        rank_score = {
            name: statistics.mean(normalized[task][name] for task in sources)
            for name in names
        }
        order = sorted(names, key=lambda name: (-rank_score[name], names.index(name)))
        row = normalize({**tables[target], "__default__": defaults[target]})
        default = row.pop("__default__")
        folds[target] = {
            "order": order,
            "curve": exact_curves(row, default, order),
            "fixed_nearest_budget1": max(default, row["nearest"]),
        }
    curve = {
        budget: {
            key: statistics.mean(folds[task]["curve"][budget][key] for task in family)
            for key in ["uniform_mean", "ordered"]
        }
        for budget in range(len(names) + 1)
    }
    c_meta = len(names) * (len(family) - 1)
    breakeven = {}
    for target in [1.0, 0.99, 0.95]:
        budgets = {
            key: next(
                budget for budget in curve if curve[budget][key] >= target - 1e-12
            )
            for key in ["uniform_mean", "ordered"]
        }
        gap = budgets["uniform_mean"] - budgets["ordered"]
        breakeven[str(target)] = {
            **budgets,
            "saved_per_task": gap,
            "K_star": c_meta / gap if gap > 0 else None,
        }
    return {
        "folds": folds,
        "curve": curve,
        "c_meta_evaluations": c_meta,
        "breakeven": breakeven,
        "fixed_nearest_budget1_mean": statistics.mean(
            fold["fixed_nearest_budget1"] for fold in folds.values()
        ),
        "fixed_nearest_meta_evaluations": 0,
    }


def run(output: Path) -> None:
    """Execute the declared deterministic replays, with incremental immutable results."""
    if output.exists():
        raise FileExistsError(
            "Replay output already exists; choose a new run directory"
        )
    output.mkdir(parents=True)
    tb.ensure_default_task_adapter(require=True)
    adapter = tb.current_task_adapter()
    provenance = {
        "label": "EXPLORATORY HISTORICAL REPLAY; not independent generalization",
        "python": platform.python_version(),
        "executable": __import__("sys").executable,
        "trace_sha": _git(ROOT, "rev-parse", "HEAD"),
        "trace_branch": _git(ROOT, "branch", "--show-current"),
        "trace_bench_sha": _git(adapter.tasks_root, "rev-parse", "HEAD"),
        "trace_bench_status": _git(adapter.tasks_root, "status", "--short"),
        "script_sha256": _sha(Path(__file__)),
        "protocol_sha256": _sha(HERE / "PROTOCOL.md"),
        "inputs": {
            name: _sha(PROBES / name)
            for name in [
                "probe_t_routing_menu.py",
                "probe_t_routing_menu.json",
                "probe_u2_default_baseline.json",
                "probe_x_llm_w2.json",
                "probe_s_knobs_results.json",
                "probe_r_menu_validity.py",
                "probe_r_results.json",
            ]
        },
    }
    _write(output / "provenance.json", provenance)
    socket.socket = _blocked_socket
    rows, defaults = {}, {}
    tables = {}
    for task in historical_t.ROUTING:
        slug = task.split("/")[-1]
        rows[task] = {}
        tables[task] = {}
        for name, source in historical_t.MENU.items():
            repeats = [_score(adapter, task, source) for _ in range(3)]
            _write(output / "routing" / f"{slug}_{name}.json", repeats)
            rows[task][name] = repeats
            if not all(row["valid"] for row in repeats):
                raise RuntimeError(
                    "A trusted historical menu candidate failed execution"
                )
            tables[task][name] = repeats[0]["score"]
        default = _score(adapter, task, None)
        _write(output / "routing" / f"{slug}_default.json", default)
        defaults[task] = default["score"]
        print(f"routing replayed: {slug}", flush=True)
    old_t = _read("probe_t_routing_menu.json")
    correlations = {
        f"{left} vs {right}": historical_t.spearman(
            list(tables[left].values()), list(tables[right].values())
        )
        for left, right in itertools.combinations(tables, 2)
    }
    summary = {
        "score_table": tables,
        "default_scores": defaults,
        "max_historical_score_difference": max(
            abs(tables[task][name] - old_t["tasks"][task]["scores"][name])
            for task in tables
            for name in tables[task]
        ),
        "max_replicate_range": max(
            max(row["score"] for row in repeats) - min(row["score"] for row in repeats)
            for task in rows.values()
            for repeats in task.values()
        ),
        "correlations": correlations,
        "w2": _w2(tables, defaults),
    }
    _write(output / "routing_summary.json", summary)
    old_x = _read("probe_x_llm_w2.json")
    source_rows, transfer_rows = [], []
    for sample in old_x["samples"]:
        if sample.get("q_norm") is None:
            continue
        result = _score(adapter, sample["task"], sample["code"])
        row = {"source": sample["task"], "index": sample["i"], **result}
        row["historical_score"] = sample["score"]
        source_rows.append(row)
        for target in old_x["transfer"]:
            if target == sample["task"]:
                continue
            transfer_rows.append(
                {
                    "source": sample["task"],
                    "index": sample["i"],
                    "target": target,
                    **_score(adapter, target, sample["code"]),
                }
            )
        _write(
            output / "transfer" / f"source_{len(source_rows):02d}.json",
            {"source_replay": row, "target_replays": transfer_rows[-2:]},
        )
    _write(
        output / "transfer_summary.json",
        {
            "source_rows": source_rows,
            "target_rows": transfer_rows,
            "source_valid": sum(row["valid"] for row in source_rows),
            "source_total": len(source_rows),
            "target_valid": sum(row["valid"] for row in transfer_rows),
            "target_total": len(transfer_rows),
        },
    )
    structures = {}
    for task in _read("probe_s_knobs_results.json")["tasks"]:
        bundle = adapter._load_bundle(task, fresh=True)
        dataset = bundle["train_dataset"]
        structures[task] = {"dataset_type": type(dataset).__name__}
        for key in ["inputs", "x", "examples"]:
            if hasattr(dataset, key):
                structures[task][f"n_{key}"] = len(getattr(dataset, key))
        if isinstance(dataset, dict):
            structures[task] = {key: len(value) for key, value in dataset.items()}
        elif isinstance(dataset, (tuple, list)):
            structures[task]["n_entries"] = len(dataset)
    _write(output / "dataset_structure.json", structures)
    menus = {}
    for task, menu in historical_r.CODE_MENUS.items():
        menus[task] = {
            "code": {
                name: _score(adapter, task, source) for name, source in menu.items()
            },
            "prose": [
                _score(adapter, task, source) for source in historical_r.PROSE_MENU
            ],
        }
        _write(output / "menu" / f"{task.split('/')[-1]}.json", menus[task])
    print(
        json.dumps(
            {
                "output": str(output),
                "routing_difference": summary["max_historical_score_difference"],
                "routing_replicate_range": summary["max_replicate_range"],
                "transfer_valid": sum(row["valid"] for row in transfer_rows),
                "transfer_total": len(transfer_rows),
            }
        ),
        flush=True,
    )


def main() -> None:
    """Read the single explicit output location and launch an offline replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    run(arguments.output)


if __name__ == "__main__":
    main()
