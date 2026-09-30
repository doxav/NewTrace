"""Plot all preserved public B2 pairs; never call objectives, candidates or a model."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I

ROOT = Path(__file__).resolve().parent
COLORS = {"gain": "#087F8C", "loss": "#CC503E", "tie": "#747474"}
MARKERS = {"gain": "o", "loss": "^", "tie": "s"}


def load_pairs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify every plotted pair against frozen allocations and the independent audit."""
    frozen = E.read(ROOT / "freeze.json")
    audit = E.read(ROOT / "independent_review.json")
    results = E.read(ROOT / "results.json")
    if (
        audit["status"] != "passed"
        or B.digest(frozen) != audit["freeze_sha256"]
        or B.digest(results) != audit["result_sha256"]
    ):
        raise ValueError(
            "B2 freeze/result differs from the completed independent audit"
        )
    identifiers = [entry["id"] for entry in frozen["allocations"]]
    if len(identifiers) != 96 or len(set(identifiers)) != 96:
        raise ValueError("plot requires every one of the 96 fixed B2 pairs")
    audited = {entry["id"]: entry for entry in audit["all_pair_identities_and_deltas"]}
    if set(audited) != set(identifiers):
        raise ValueError("independent audit pair identities differ")
    rows, input_hashes = [], {}
    for entry in frozen["allocations"]:
        paths = {
            "seed": ROOT.parent / entry["control_path"],
            "variant": ROOT / "raw" / (entry["id"] + ".json"),
        }
        pair = {
            "id": entry["id"],
            "condition": entry["condition"],
            "stratum": f"{entry['task']['family']}/{entry['task']['dimension']}",
            "local_seed": entry["local_seed"],
        }
        for policy, path in paths.items():
            row = E.read(path)
            expected_hash = audited[entry["id"]][
                "control_sha256" if policy == "seed" else "variant_sha256"
            ]
            if B.digest(row) != expected_hash:
                raise ValueError(
                    "plotted raw trajectory differs from independent audit"
                )
            expected_source = frozen[
                "seed_source_sha256" if policy == "seed" else "variant_source_sha256"
            ]
            if (
                row["source_sha256"] != expected_source
                or row["task_identity"] != B.task_identity(entry["task"])
                or row["local_seed"] != entry["local_seed"]
                or row["budget"] != 32
                or row["stratum"] != pair["stratum"]
            ):
                raise ValueError("plotted pair input identity mismatch")
            if (
                not row["valid"]
                or not row["candidate_valid"]
                or row["fallback_used"]
                or row["objective_calls"] != 32
                or len(row["observations"]) != 32
            ):
                raise ValueError(
                    "unexpected invalid or incomplete B2 trajectory; cannot drop it"
                )
            metrics = row["metrics"]
            curve = metrics["curve"]
            if (
                len(curve) != 32
                or not math.isclose(
                    statistics.mean(curve), metrics["auc"], rel_tol=1e-12
                )
                or curve[-1] != metrics["final_regret"]
            ):
                raise ValueError("preserved metric does not match its normalized curve")
            if any(
                not math.isfinite(metrics[key]) or metrics[key] <= 0
                for key in ("auc", "final_regret")
            ):
                raise ValueError(
                    "log-axis plot cannot silently drop nonpositive/nonfinite outcomes"
                )
            pair[policy] = {key: metrics[key] for key in ("auc", "final_regret")}
            physical = path if path.exists() else path.with_suffix(path.suffix + ".gz")
            input_hashes[str(physical)] = hashlib.sha256(
                physical.read_bytes()
            ).hexdigest()
        rows.append(pair)
    summaries = {}
    for condition in ("central", "broad"):
        selected = [row for row in rows if row["condition"] == condition]
        if len(selected) != 48 or set(
            Counter(row["stratum"] for row in selected).values()
        ) != {8}:
            raise ValueError("plot panel must retain 48 balanced pairs")
        summaries[condition] = {}
        for metric in ("auc", "final_regret"):
            means = {
                policy: statistics.mean(row[policy][metric] for row in selected)
                for policy in ("seed", "variant")
            }
            for policy, value in means.items():
                expected = results["groups"][condition]["overall"][policy]["metrics"][
                    metric
                ]
                if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-14):
                    raise ValueError(
                        "scatter means disagree with the saved B2 aggregate"
                    )
            counts = {
                "gain": sum(
                    row["variant"][metric] < row["seed"][metric] for row in selected
                ),
                "loss": sum(
                    row["variant"][metric] > row["seed"][metric] for row in selected
                ),
                "tie": sum(
                    row["variant"][metric] == row["seed"][metric] for row in selected
                ),
            }
            summaries[condition][metric] = {
                "pairs": len(selected),
                "means": means,
                "counts": counts,
            }
    if sum(summaries[c]["auc"]["counts"]["loss"] for c in summaries) != 26:
        raise ValueError("the 26 documented AUC losses must all remain visible")
    return rows, {
        "freeze_sha256": B.digest(frozen),
        "saved_result_sha256": B.digest(results),
        "source_sha256": frozen["variant_source_sha256"],
        "raw_input_byte_hashes": input_hashes,
        "unique_pairs": 96,
        "raw_trajectories_checked": 192,
        "panels": summaries,
        "new_objective_calls": 0,
        "candidate_executions": 0,
        "model_calls": 0,
    }


def number(value: float) -> str:
    """Format a displayed arithmetic mean without changing its stored precision."""
    return f"{value:.5f}".replace(".", ",")


def make_figure() -> dict[str, Any]:
    """Render four complete paired scatterplots with equal axes and no clipping."""
    pairs, checks = load_pairs()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 11.4))
    fig.subplots_adjust(
        left=0.075, right=0.98, top=0.835, bottom=0.19, hspace=0.53, wspace=0.12
    )
    fig.suptitle(
        "B2 : gains moyens et pertes individuelles", fontsize=20, weight="bold", y=0.978
    )
    fig.text(
        0.5,
        0.942,
        "Seul le premier point du seed est remplacé par le centre des bornes.",
        ha="center",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.917,
        "96 paires publiques · B = 32 · regret normalisé, plus bas = meilleur",
        ha="center",
        fontsize=11,
        color="#444444",
    )
    limits = {}
    for metric in ("auc", "final_regret"):
        values = [
            row[policy][metric] for row in pairs for policy in ("seed", "variant")
        ]
        limits[metric] = (min(values) / 1.5, max(values) * 1.5)
    plotted = {}
    for index, (metric, condition) in enumerate(
        (
            ("auc", "central"),
            ("auc", "broad"),
            ("final_regret", "central"),
            ("final_regret", "broad"),
        )
    ):
        ax = axes.flat[index]
        selected = [row for row in pairs if row["condition"] == condition]
        x = np.array([row["seed"][metric] for row in selected])
        y = np.array([row["variant"][metric] for row in selected])
        masks = {"gain": y < x, "loss": y > x, "tie": y == x}
        collections = []
        for category in ("gain", "tie", "loss"):
            collections.append(
                ax.scatter(
                    x[masks[category]],
                    y[masks[category]],
                    c=COLORS[category],
                    marker=MARKERS[category],
                    s=42 if category != "loss" else 53,
                    alpha=0.84,
                    edgecolors="white",
                    linewidths=0.45,
                    zorder=3 if category != "loss" else 4,
                )
            )
        if sum(len(collection.get_offsets()) for collection in collections) != 48:
            raise ValueError("scatter artist dropped a paired outcome")
        low, high = limits[metric]
        ax.set(xscale="log", yscale="log", xlim=(low, high), ylim=(low, high))
        ax.set_aspect("equal", adjustable="box")
        ax.plot([low, high], [low, high], ls="--", color="#666666", lw=1.1, zorder=1)
        ax.grid(which="major", color="#DADADA", linewidth=0.6, alpha=0.7)
        stats = checks["panels"][condition][metric]
        sx, sy = stats["means"]["seed"], stats["means"]["variant"]
        ax.scatter(
            [sx],
            [sy],
            color="#111111",
            marker="X",
            s=105,
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )
        title = ("AUC anytime" if metric == "auc" else "Regret final") + (
            " · optima centraux" if condition == "central" else " · optima élargis"
        )
        change = (
            f" · −{100 * (1 - sy / sx):.2f} %".replace(".", ",")
            if metric == "auc"
            else ""
        )
        counts = stats["counts"]
        ax.set_title(
            f"{chr(65 + index)}  {title}\nMoyennes : {number(sx)} → {number(sy)}{change}\n{counts['gain']} gains · {counts['loss']} pertes · {counts['tie']} égalités",
            fontsize=10,
            loc="left",
            pad=10,
            linespacing=1.5,
        )
        ax.set_xlabel("Seed original — échelle logarithmique", labelpad=7)
        ax.set_ylabel("B2 — échelle logarithmique", labelpad=7)
        if not np.all((x >= low) & (x <= high) & (y >= low) & (y <= high)):
            raise ValueError("figure limits exclude a recorded outcome")
        plotted[f"{condition}/{metric}"] = {
            "pair_markers": 48,
            "equality_ties_retained": counts["tie"],
            "axis_limits": [low, high],
            "scales": ["log", "log"],
            "pair_ids": [row["id"] for row in selected],
        }
    handles = [
        Line2D(
            [],
            [],
            color=COLORS[category],
            marker=MARKERS[category],
            ls="",
            markersize=7,
            label=label,
        )
        for category, label in (
            ("gain", "B2 meilleur"),
            ("loss", "B2 moins bon"),
            ("tie", "Égalité"),
        )
    ]
    handles += [
        Line2D(
            [],
            [],
            color="#111111",
            marker="X",
            ls="",
            markersize=8,
            label="Moyennes arithmétiques",
        ),
        Line2D([], [], color="#666666", ls="--", label="Diagonale d’égalité"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.069),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.5,
        0.036,
        "Tous les points sont conservés, sans écrêtage ni jitter ; certains se superposent.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.text(
        0.5,
        0.016,
        "Une paire = tâche × seed local. Ces paires ne sont pas des réplications LLM indépendantes. Aucun effet récursif n’est testé.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / f"paired_outcomes.{suffix}", dpi=220, facecolor="white")
    plt.close(fig)
    checks["plotted_panels"] = plotted
    checks["plotted_pair_markers"] = 192
    for path, digest in checks["raw_input_byte_hashes"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("plotting changed a preserved input")
    I.persist(ROOT / "paired_outcomes_checks.json", checks)
    return checks


if __name__ == "__main__":
    make_figure()
