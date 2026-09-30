"""Render descriptive S1 plots from completed, frozen-analysis outputs only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Plot selection quality and ranking stability without treating subsets as replications."""
    result = json.loads((ROOT / "results.json").read_text())
    if result["status"] != "COMPLETE_EXPLORATORY_FIXED_BANK":
        raise RuntimeError("plot requires completed S1 analysis")
    panels = (
        (
            "selection_mean_audit_auc",
            "Selected policy: audit AUC (lower is better)",
            "viridis_r",
        ),
        (
            "mean_rank_correlation_to_audit",
            "Train/audit rank correlation (higher is better)",
            "viridis",
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    figure.subplots_adjust(bottom=0.22, top=0.82, wspace=0.32)
    for axis, (metric, title, colors) in zip(axes, panels):
        matrix = np.array([row[metric] for row in result["summaries"]]).reshape(3, 3)
        axis.imshow(matrix, cmap=colors)
        for i in range(3):
            for j in range(3):
                axis.text(
                    j,
                    i,
                    f"{matrix[i, j]:.4f}",
                    ha="center",
                    va="center",
                    color="black",
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                )
        axis.set_xticks(range(3), [1, 2, 4])
        axis.set_yticks(range(3), [1, 2, 4])
        axis.set_xlabel("Local seeds per task")
        axis.set_ylabel("Instances per family/dimension stratum")
        axis.set_title(title, fontsize=10)
    figure.suptitle(
        "EXP-16 / S1 — fresh fixed-policy-bank measurement diagnostic", fontsize=12
    )
    figure.text(
        0.5,
        0.03,
        "200 overlapping nested subsamples of one training grid; descriptive sensitivity, not 200 independent replications.\nAll selections frozen before independent audit. No live feedback treatment is tested in this figure.",
        ha="center",
        fontsize=8,
    )
    for extension in ("png", "svg"):
        figure.savefig(
            ROOT / f"selection_quality.{extension}", dpi=170, bbox_inches="tight"
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
