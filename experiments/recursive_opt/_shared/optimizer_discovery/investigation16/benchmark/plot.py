"""Render the frozen B1 fixed-policy diagnostic as an exportable research figure."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E

ROOT = Path(__file__).resolve().parent
NAMES = {
    "seed": "Seed",
    "uniform": "Uniforme",
    "midpoint": "Centre fixe",
    "representative41": "Représentant EXP-15",
}


def render(results: dict[str, Any]) -> None:
    """Plot complete-group metrics without clipping poor outcomes or hiding invalidity."""
    if results["total_trajectories"] != 384:
        raise ValueError("B1 figure requires all 384 allocated trajectories")
    groups = results["groups"]
    if any(
        row["metrics"] is None
        for policies in groups.values()
        for row in policies.values()
    ):
        raise ValueError("B1 figure cannot silently omit an invalid fixed-policy group")
    plt.rcParams.update({"font.size": 10})
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    policies = list(NAMES)
    width = 0.35
    for index, (condition, label) in enumerate(
        [("central", "Optima centraux"), ("broad", "Optima élargis")]
    ):
        axes[0].bar(
            [position + (index - 0.5) * width for position in range(4)],
            [groups[condition][policy]["metrics"]["auc"] for policy in policies],
            width,
            label=label,
        )
        for policy in policies:
            axes[index + 1].plot(
                range(1, 33),
                groups[condition][policy]["metrics"]["mean_curve"],
                label=NAMES[policy],
            )
        axes[index + 1].set_title(label)
        axes[index + 1].set_xlabel("Évaluations objectif")
        axes[index + 1].set_ylabel("Regret normalisé moyen")
        axes[index + 1].set_yscale("log")
        axes[index + 1].grid(alpha=0.2)
    axes[0].set_xticks(range(4), list(NAMES.values()), rotation=25, ha="right")
    axes[0].set_ylabel("AUC normalisée (plus bas = mieux)")
    axes[0].set_title("Politiques fixées avant le diagnostic")
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    figure.suptitle(
        "B1 exploratoire : 12 nouvelles tâches × 4 seeds locaux × 2 distributions",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.005,
        "Toutes les 384 trajectoires sont retenues. Ce diagnostic ne compare pas les procédures de génération A1/A2.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.95))
    figure.savefig(ROOT / "headroom.png", dpi=160)
    figure.savefig(ROOT / "headroom.pdf")
    plt.close(figure)


if __name__ == "__main__":
    render(E.read(ROOT / "results.json"))
