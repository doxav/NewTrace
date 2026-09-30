"""Render the preserved EXP18 estimates without fitting or resampling anything."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

ANALYSIS_SHA256 = "1654ab1ff03bf1d2713bb3ec6bfe870f603cc8ae6e77b32579a074d20f94925a"
ARMS = ("L", "M", "P", "PM")
CONTRASTS = ("M-L", "PM-P", "P-L", "PM-M", "memory", "pareto", "interaction")
LABELS = (
    "M − L\nMémoire, parent scalaire",
    "PM − P\nMémoire, parent Pareto",
    "P − L\nPareto, sans mémoire",
    "PM − M\nPareto, avec mémoire",
    "Effet moyen de mémoire\n½[(M − L) + (PM − P)]",
    "Effet moyen de Pareto\n½[(P − L) + (PM − M)]",
    "Interaction\nPM − P − M + L",
)


def read_estimates(path: Path) -> dict[str, Any]:
    """Authenticate the exact analysis and retain only the plotted frozen values."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ANALYSIS_SHA256:
        raise ValueError(
            "analysis bytes differ from the registered presentation source"
        )
    result = json.loads(gzip.decompress(raw))
    seeds = result["outer_seeds"]
    if seeds != [18011, 18023, 18037, 18041, 18053, 18067]:
        raise ValueError("unexpected outer-seed order in the preserved analysis")
    values = {arm: result["arms"][arm]["auc"]["per_seed"] for arm in ARMS}
    for arm, rows in values.items():
        expected = [result["per_seed"][str(seed)][arm]["auc"] for seed in seeds]
        if rows != expected or not all(
            math.isfinite(value) and value >= 0 for value in rows
        ):
            raise ValueError("arm trajectory values or outer-seed order disagree")
    contrasts = {name: result["contrasts"][name] for name in CONTRASTS}
    for value in contrasts.values():
        lower, upper = value["paired_bootstrap_95"]
        if not (
            len(value["deltas"]) == len(seeds)
            and lower <= 0 <= upper
            and lower <= value["mean"] <= upper
            and value["interpretation"] == "inconclusive"
            and value["replication_unit"] == "outer_seed"
        ):
            raise ValueError(
                "preserved contrast does not support the figure annotation"
            )
    return {"outer_seeds": seeds, "auc_by_arm": values, "contrasts": contrasts}


def signed(value: float) -> str:
    """Format one copied estimate with a decimal comma and true minus character."""
    return f"{value:.4f}".replace("-", "−").replace(".", ",")


def axis_decimal(value: float, _position: int) -> str:
    """Format the left axis in French without changing its numeric coordinates."""
    return f"{value:.2f}".replace(".", ",")


def render(output_dir: Path) -> dict[str, Any]:
    """Save a static paired plot and forest plot with exact source provenance."""
    source = Path(__file__).resolve().parents[1] / "run" / "analysis_results.json.gz"
    estimates = read_estimates(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": "#25354a",
            "text.color": "#25354a",
            "xtick.color": "#536174",
            "ytick.color": "#536174",
            "axes.edgecolor": "#bdc7d2",
            "svg.hashsalt": "EXP18-frozen-mechanisms-v1",
            "savefig.facecolor": "white",
        }
    )
    figure = plt.figure(figsize=(16, 8.2), facecolor="white")
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 1.85),
        left=0.072,
        right=0.976,
        top=0.79,
        bottom=0.245,
        wspace=0.58,
    )
    paired = figure.add_subplot(grid[0, 0])
    right = grid[0, 1].subgridspec(1, 2, width_ratios=(1.05, 0.95), wspace=0.055)
    forest = figure.add_subplot(right[0, 0])
    numbers = figure.add_subplot(right[0, 1], sharey=forest)
    colors = ("#32688d", "#b57439", "#67917a", "#80679c", "#b96167", "#6c7a88")
    for index, (seed, color) in enumerate(zip(estimates["outer_seeds"], colors)):
        y_values = [estimates["auc_by_arm"][arm][index] for arm in ARMS]
        paired.plot(
            range(4), y_values, color=color, alpha=0.43, linewidth=1.0, zorder=1
        )
        paired.scatter(
            range(4),
            y_values,
            color=color,
            s=37,
            edgecolors="white",
            linewidths=0.55,
            zorder=2,
        )
    paired.set_xticks(range(4), ARMS, fontsize=12)
    paired.set_xlim(-0.23, 3.23)
    paired.set_ylim(0, 0.115)
    paired.yaxis.set_major_locator(MultipleLocator(0.02))
    paired.yaxis.set_major_formatter(FuncFormatter(axis_decimal))
    paired.set_ylabel("Regret normalisé (AUC)  ·  plus bas = meilleur", labelpad=12)
    paired.set_title(
        "A   Six recherches appariées",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=21,
    )
    paired.grid(axis="y", color="#e6ebf0", linewidth=0.75)
    paired.set_axisbelow(True)
    paired.spines[["top", "right"]].set_visible(False)
    paired.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color=color,
                linewidth=1,
                markersize=5,
                label=str(seed),
            )
            for seed, color in zip(estimates["outer_seeds"], colors)
        ],
        title="Graine de recherche",
        ncol=6,
        fontsize=9,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(-0.03, -0.095),
        frameon=False,
        handlelength=1.3,
        columnspacing=1.1,
    )

    row_positions = [0, 1, 2, 3, 4.35, 5.35, 6.35]
    for index, (name, position) in enumerate(zip(CONTRASTS, row_positions)):
        value = estimates["contrasts"][name]
        lower, upper = value["paired_bootstrap_95"]
        color = "#385c7a" if index < 4 else "#926235"
        forest.hlines(position, lower, upper, color=color, linewidth=2.2)
        forest.vlines(
            [lower, upper], position - 0.09, position + 0.09, color=color, linewidth=1.3
        )
        forest.scatter(
            [value["mean"]],
            [position],
            marker="o" if index < 4 else "D",
            color=color,
            s=35,
            zorder=3,
        )
        numbers.text(
            0,
            position,
            f"{signed(value['mean'])}  [{signed(lower)} ; {signed(upper)}]",
            va="center",
            fontsize=9.5,
            fontfamily="DejaVu Sans Mono",
        )
    forest.set_yticks(row_positions, LABELS, fontsize=9.3)
    forest.tick_params(axis="y", length=0, pad=11)
    forest.set_ylim(6.95, -0.75)
    forest.set_xlim(-0.038, 0.047)
    forest.set_xticks([-0.03, 0, 0.03], ["−0,03", "0", "+0,03"])
    forest.set_xlabel("Différence de regret-AUC", labelpad=9)
    forest.axvline(0, color="#687788", linewidth=1, linestyle=(0, (3, 3)), zorder=0)
    forest.axhline(3.65, color="#d6dee7", linewidth=0.8)
    forest.spines[["top", "left", "right"]].set_visible(False)
    forest.text(
        0.5,
        1.047,
        "B   Effets des mécanismes",
        transform=forest.transAxes,
        ha="center",
        fontsize=13,
        fontweight="bold",
    )
    numbers.set_xlim(0, 1)
    numbers.axis("off")
    numbers.text(
        0,
        1.047,
        "Moyenne  [IC bootstrap à 95 %]",
        transform=numbers.transAxes,
        fontsize=9.4,
        color="#536174",
    )

    figure.text(
        0.072,
        0.925,
        "EXP18  |  Mémoire des essais et choix du parent",
        fontsize=22,
        fontweight="bold",
    )
    figure.text(
        0.072,
        0.876,
        "Audit réservé · six recherches appariées · programmes choisis avant l’audit",
        fontsize=12,
        color="#536174",
    )
    figure.text(
        0.072,
        0.121,
        "Parent = programme à réécrire. L : meilleur score · M : L + mémoire · P : choix Pareto · PM : P + mémoire.",
        fontsize=10.4,
    )
    figure.text(
        0.072,
        0.081,
        "Étude exploratoire, six répétitions. Les sept IC à 95 % contiennent zéro : aucun gain établi.",
        fontsize=11,
        fontweight="bold",
    )
    figure.text(
        0.072,
        0.047,
        "Négatif : favorable au mécanisme ajouté. Pour l’interaction : écart à l’additivité, sans preuve de supériorité de PM.",
        fontsize=9.3,
        color="#536174",
    )
    figure.text(
        0.072,
        0.020,
        "Chaque IC est individuel, sans garantie simultanée. Source : EXP18/run/analysis_results.json.gz · SHA-256 1654ab1ff03b…",
        fontsize=8.5,
        color="#708092",
    )

    svg = output_dir / "exp18_mechanisms.svg"
    png = output_dir / "exp18_mechanisms.png"
    figure.savefig(
        svg, metadata={"Date": None, "Creator": "EXP18 render_mechanisms.py"}
    )
    figure.savefig(
        png,
        dpi=180,
        metadata={
            "Software": "EXP18 render_mechanisms.py",
            "SourceSHA256": ANALYSIS_SHA256,
        },
    )
    plt.close(figure)
    provenance = {
        "schema": "exp18.presentation_figure.v1",
        "source_relative_to_script": "../run/analysis_results.json.gz",
        "source_sha256": ANALYSIS_SHA256,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "matplotlib_version": matplotlib.__version__,
        "presentation_language": "fr",
        "scientific_recomputation": False,
        "contrast_order": list(CONTRASTS),
        "estimates": estimates,
        "output_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (svg, png)
        },
    }
    (output_dir / "exp18_mechanisms_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return provenance


def main() -> None:
    """Render only this presentation artifact using the available plotting environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent
    )
    args = parser.parse_args()
    print(json.dumps(render(args.output_dir)["output_sha256"], sort_keys=True))


if __name__ == "__main__":
    main()
