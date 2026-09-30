"""Rebuild descriptive EXP20 figures from preserved receipts; never call a provider."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path: Path) -> Any:
    """Read a preserved JSON artifact."""
    return json.loads(path.read_text())


def main() -> None:
    """Plot registered scores and separately labeled retrospective cost accounting."""
    root = Path(__file__).resolve().parent
    results = read(root / "results.json")
    frozen = read(root / "selections_frozen.json")
    seeds = frozen["seeds"]
    labels = {
        "unchanged": ("Programme inchangé", "#64748b"),
        "standard": ("Standard", "#2563eb"),
        "curriculum": ("Curriculum", "#d97706"),
    }
    curves = {
        arm: [
            100 * statistics.mean(r["curves"][str(s)][p] for s in seeds)
            for p in range(7)
        ]
        for arm, r in results["arms"].items()
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), layout="constrained")
    for arm, (label, color) in labels.items():
        axes[0].plot(range(7), curves[arm], "o-", label=label, color=color)
    axes[0].set(
        xlabel="Réponses de l’optimiseur par bras et graine, vides incluses",
        ylabel="Exactitude TEST moyenne (%)",
        ylim=(0, 70),
        title="Apprentissage sur six graines",
    )
    axes[0].legend()
    deltas = results["curriculum_minus_standard"]["deltas"]
    axes[1].bar(range(6), [100 * d for d in deltas], color="#d97706")
    mean = 100 * statistics.mean(deltas)
    axes[1].axhline(0, color="#64748b")
    axes[1].axhline(
        mean, color="#111827", ls="--", label=f"Moyenne : {mean:.2f} points"
    )
    axes[1].set_xticks(range(6), [str(s) for s in seeds], rotation=35)
    axes[1].set(
        xlabel="Graine externe",
        ylabel="Curriculum − standard (points)",
        title="Différence sur le score principal",
    )
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("EXP20 — politiques choisies sur VALIDATION, mesurées sur TEST")
    for extension in ("png", "svg"):
        fig.savefig(root / f"learning_curves.{extension}", dpi=150)
    plt.close(fig)

    # Each receipt is counted once, across both arms. This is campaign accounting,
    # not an estimate of either arm's standalone or counterfactual expense.
    costs = []
    for prefix in range(7):
        reader_keys: set[tuple[int, str]] = set()
        optimizer_receipts = []
        for seed in seeds:
            candidates: set[str] = set()
            for arm in ("standard", "curriculum"):
                chain = root / "chains" / str(seed) / arm
                for path in (chain / "reader").glob("*.json"):
                    event = read(path)
                    if event["optimizer_responses_before"] <= prefix:
                        reader_keys.add((seed, event["cache_key"]))
                for slot in range(1, prefix + 1):
                    paths = list(
                        (chain / "optimizer" / f"{slot:02d}").glob("*/response.json")
                    )
                    assert len(paths) == 1
                    optimizer_receipts.append(paths[0])
                candidates.update(
                    key
                    for key, item in frozen["selections"][str(seed)][arm][
                        "candidates"
                    ].items()
                    if item["first_slot"] <= prefix
                )
            for split in ("train_complete", "validation"):
                for key in candidates:
                    for path in (root / "evaluation" / str(seed) / split / key).glob(
                        "reader_events/*/reader/*.json"
                    ):
                        reader_keys.add((seed, read(path)["cache_key"]))
        receipts = list(optimizer_receipts)
        for seed, key in sorted(reader_keys):
            paths = list(
                (root / "reader_cache" / str(seed) / key).glob("*/response.json")
            )
            assert len(paths) == 1
            receipts.append(paths[0])
        usage = [read(path)["usage"] for path in receipts]
        costs.append(
            {
                "prefix": prefix,
                "completed_paid_calls": len(receipts),
                "reader_calls": len(reader_keys),
                "optimizer_calls": len(optimizer_receipts),
                **{
                    metric: sum(u[metric] for u in usage)
                    for metric in (
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "cost_usd",
                    )
                },
            }
        )
    test_keys = {
        (int(path.parts[-7]), read(path)["cache_key"])
        for path in root.glob("evaluation/*/holdout/*/reader_events/*/reader/*.json")
    }
    # Validate the path-derived seed before accounting; no accidental stratum pooling.
    assert {seed for seed, _ in test_keys} == set(seeds)
    test_usage = []
    for seed, key in test_keys:
        paths = list((root / "reader_cache" / str(seed) / key).glob("*/response.json"))
        assert len(paths) == 1
        test_usage.append(read(paths[0])["usage"])
    assert costs[-1]["completed_paid_calls"] + len(test_keys) == 7060
    assert (
        costs[-1]["total_tokens"] + sum(u["total_tokens"] for u in test_usage)
        == 12987439
    )
    assert all(
        a["completed_paid_calls"] <= b["completed_paid_calls"]
        for a, b in zip(costs, costs[1:])
    )
    report = {
        "status": "exploratory descriptive reconstruction after primary analysis",
        "scope": "Joint six-seed, two-arm campaign. Unique paid receipts used by learning events with response index <= prefix and full TRAIN/VALIDATION evaluation of artifacts first appearing <= prefix. No TEST or pilot costs on the curves. Shared receipts counted once; no standalone arm-cost causal comparison.",
        "timing_caveat": "This is a retrospective attribution by prefix, not chronological spending: all validation was actually deferred until generation finished.",
        "prefixes": costs,
        "test_measurement": {
            "paid_calls": len(test_keys),
            "total_tokens": sum(u["total_tokens"] for u in test_usage),
            "cost_usd": sum(u["cost_usd"] for u in test_usage),
        },
    }
    (root / "cost_curves.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for ax, metric, label in zip(
        axes,
        ("completed_paid_calls", "total_tokens", "cost_usd"),
        (
            "Appels facturés cumulés (deux bras)",
            "Tokens cumulés (deux bras)",
            "Coût rapporté cumulé (USD, deux bras)",
        ),
    ):
        for arm, (name, color) in labels.items():
            ax.plot(
                [r[metric] for r in costs], curves[arm], "o-", color=color, label=name
            )
        ax.set(xlabel=label, ylim=(0, 70))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Exactitude TEST moyenne (%)")
    axes[0].legend()
    fig.suptitle(
        "Descriptif : dépenses partagées de la campagne par préfixe 0–6\nApprentissage + sélection ; mesure TEST et pilotes exclus"
    )
    fig.savefig(root / "cost_curves.png", dpi=150)
    plt.close(fig)
    print(
        json.dumps(
            {"prefix_costs": costs, "test_measurement": report["test_measurement"]}
        )
    )


if __name__ == "__main__":
    main()
