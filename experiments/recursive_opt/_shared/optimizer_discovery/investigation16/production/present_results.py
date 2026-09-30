"""Export complete P1 reporting tables and figures without executing research code."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "production/presentation"
INPUT_HASHES: dict[str, str] = {}


def read(relative: str) -> dict[str, Any]:
    """Read a completed evidence record and retain its exact physical identity."""
    path = ROOT / relative
    if not path.exists():
        path = Path(str(path) + ".gz")
    raw = path.read_bytes()
    INPUT_HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    """Write a flat reporting table, retaining every registered row."""
    with (OUTPUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Require completion, project all arms and export the prespecified contrasts."""
    completed = read("production_run/pipeline_complete.json")
    assert completed["stages"] == 7
    primary = read("production_run/analysis_results.json")
    control = read("production_baseline_control/results.json")
    numeric = read("production_run/numeric_verification/attempt_001.json")
    assert numeric["status"] == "PASS"
    assert primary["resources"]["completed_responses"] == 192
    assert primary["chronology"]["unique_response_ids"] == 192
    assert primary["outer_seeds"] == control["outer_seeds"]
    assert len(primary["outer_seeds"]) == 6
    assert control["deployment"]["trajectories"] == 144
    OUTPUT.mkdir(exist_ok=True)
    arms = ["A0", "I", "C", "R", "W", "B2"]
    rows: list[dict[str, Any]] = []
    searches: list[dict[str, Any]] = []
    for outer in primary["outer_seeds"]:
        for arm in arms:
            item = (
                control["per_seed"][str(outer)]["B2"]
                if arm == "B2"
                else primary["per_seed"][str(outer)][arm]
            )
            deploy = item["deployment"]
            selection = item.get("selection", {})
            rows.append(
                {
                    "outer_seed": outer,
                    "arm": arm,
                    "auc": item["auc"],
                    "final_regret": item["final_regret"],
                    "target_attainment": deploy["target_attainment_rate"],
                    "capped_target_evaluations": deploy[
                        "mean_capped_target_evaluations"
                    ],
                    "candidate_invalid_trajectories": round(
                        deploy["candidate_invalid_fraction"] * deploy["trajectories"]
                    ),
                    "fallback_trajectories": deploy["fallback_trajectories"],
                    "selection_index": selection.get("index"),
                    "source_sha256": selection.get(
                        "source_sha256", item.get("source_sha256")
                    ),
                }
            )
            if arm in ["I", "C", "R", "W"]:
                search = item["search"]
                searches.append(
                    {
                        "outer_seed": outer,
                        "arm": arm,
                        "allocated_responses": 8,
                        "eligible_generated": search["eligible_generated"],
                        "ineligible_generated": search["ineligible_generated"],
                        "selected_seed": int(search["selected_seed_source"]),
                        "train_allocations": search["train"]["trajectories"],
                        "validation_allocations": search["validation"]["trajectories"],
                    }
                )
    contrasts = []
    for scope, mapping in [
        ("primary_exploratory", primary["contrasts"]),
        ("supplementary", control["contrasts"]),
    ]:
        for name, item in mapping.items():
            contrasts.append(
                dict(
                    contrast=name,
                    scope=scope,
                    mean=item["mean"],
                    median=item["median"],
                    ci_low=item["paired_bootstrap_95"][0],
                    ci_high=item["paired_bootstrap_95"][1],
                    interpretation=item["interpretation"],
                    **{
                        f"delta_{seed}": delta
                        for seed, delta in zip(
                            primary["outer_seeds"], item["deltas"], strict=True
                        )
                    },
                )
            )
    usage = []
    for arm in ["I", "C", "R", "W"]:
        item = primary["arms"][arm]
        usage.append(
            dict(
                arm=arm,
                responses=48,
                **{
                    key: value["reported_sum"]
                    for key, value in item["known_usage"]["usage"].items()
                },
            )
        )
    assert len(rows) == 36 and len(searches) == 24 and len(contrasts) == 6
    assert sum(row["allocated_responses"] for row in searches) == 192
    assert sum(row["ineligible_generated"] for row in searches) == 18
    assert sum(row["fallback_trajectories"] for row in rows) == 0
    for filename, data in [
        ("per_seed.csv", rows),
        ("contrasts.csv", contrasts),
        ("searches.csv", searches),
        ("usage.csv", usage),
    ]:
        write_csv(filename, data)
    summary = {
        "per_seed": rows,
        "contrasts": contrasts,
        "searches": searches,
        "usage": usage,
        "input_byte_hashes": INPUT_HASHES,
        "scope": "Six paired outer seeds; exploratory. B2 is a separate fixed control; intervals imported unchanged.",
    }
    (OUTPUT / "data.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.05, 1]}
    )
    for outer in primary["outer_seeds"]:
        values = [
            next(
                row["auc"]
                for row in rows
                if row["outer_seed"] == outer and row["arm"] == arm
            )
            for arm in arms
        ]
        left.plot(
            range(6),
            values,
            "o-",
            alpha=0.7,
            linewidth=1,
            markersize=4,
            label=str(outer),
        )
    left.set(
        xticks=range(6),
        xticklabels=arms,
        ylabel="Regret-AUC normalisé (plus bas = mieux)",
        title="Toutes les réplications externes",
    )
    left.axvline(4.5, color="#777777", linestyle=":")
    left.legend(title="Seed externe", fontsize=8, ncol=2)
    for index, item in enumerate(contrasts):
        color = "#a33a32" if item["mean"] > 0 else "#216e73"
        right.errorbar(
            item["mean"],
            index,
            xerr=[[item["mean"] - item["ci_low"]], [item["ci_high"] - item["mean"]]],
            fmt="o",
            color=color,
            capsize=4,
        )
    right.axvline(0, color="#666666", linewidth=1)
    right.axhline(3.5, color="#777777", linestyle=":")
    right.set(
        yticks=range(6),
        yticklabels=[row["contrast"] for row in contrasts],
        xlabel="ΔAUC : négatif = premier bras meilleur",
        title="Moyennes et IC bootstrap appariés à 95 %",
    )
    right.invert_yaxis()
    for axis in [left, right]:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.15)
    fig.suptitle(
        "EXP-16 / P1 : le feedback détaillé ne surpasse pas l’indépendant", fontsize=14
    )
    fig.text(
        0.5,
        0.01,
        "n = 6, intervalles exploratoires fragiles. B2 : référence fixe supplémentaire. Aucun résultat omis.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    for suffix in ["png", "pdf"]:
        fig.savefig(OUTPUT / f"paired_results.{suffix}", dpi=180)
    plt.close(fig)
    for filename, digest in INPUT_HASHES.items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest
    print(
        json.dumps(
            {
                "per_seed_rows": len(rows),
                "contrasts": len(contrasts),
                "searches": len(searches),
                "new_model_calls": 0,
                "new_objective_calls": 0,
                "inputs_unchanged": True,
            }
        )
    )


if __name__ == "__main__":
    main()
