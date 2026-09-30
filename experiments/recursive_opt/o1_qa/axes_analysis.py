"""EXP21 development choices, confirmation freeze, and complete paired reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from importlib.metadata import version
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import axes as X, campaign as C, campaign_analysis as A, meta as M, task


def dev_score(name: str) -> float:
    """Average the two complete development replications; missing seeds are errors."""
    return statistics.mean(
        X.read(X.ROOT / "development/reports" / str(s) / f"{name}.json")["primary"]
        for s in X.DEV_SEEDS
    )


def choose_development() -> dict[str, Any]:
    """Use the preregistered order to break ties; do not inspect confirmatory outcomes."""
    variants = X.variants()
    scores = {v["id"]: dev_score(v["id"]) for v in variants}
    winners = {
        axis: max(
            (v for v in variants if v["axis"] == axis), key=lambda v: scores[v["id"]]
        )
        for axis in X.AXES
    }
    combined = dict(X.BASE)
    active = []
    for axis, variant in winners.items():
        if scores[variant["id"]] > scores["standard"]:
            active.append(axis)
            combined.update({k: variant["config"][k] for k in X.AXES[axis]})
    mixtures = {"combined": combined}
    for axis in active:
        mixtures["without_" + axis] = {
            **combined,
            **{k: X.BASE[k] for k in X.AXES[axis]},
        }
    value = {
        "scores": scores,
        "winners": winners,
        "active_axes": active,
        "mixtures": mixtures,
        "interpretation": "development selection only; no claim of gain before fresh confirmation",
    }
    C.retain(X.ROOT / "development_choices.json", value)
    return value


def combine() -> None:
    """Evaluate combinations and leave-one-axis-out ablations on development only."""
    selected = choose_development()
    jobs = []
    for config in selected["mixtures"].values():
        name = M.config_name(config)
        for seed in X.DEV_SEEDS:
            if not (
                X.ROOT / "development/reports" / str(seed) / f"{name}.json"
            ).exists():
                jobs.append(
                    {
                        "root": str(X.ROOT / "development"),
                        "seed": seed,
                        "name": name,
                        "config": config,
                        "mode": "dev",
                    }
                )
    unique = {(j["seed"], j["name"]): j for j in jobs}
    X.run_jobs(list(unique.values()))
    C.retain(
        X.ROOT / "combination_results.json",
        {
            label: {
                "config": config,
                "name": M.config_name(config),
                "primary": dev_score(M.config_name(config)),
            }
            for label, config in selected["mixtures"].items()
        },
    )


def freeze_confirmation() -> dict[str, Any]:
    """Seal all chosen methods and analysis before any confirmatory learning call."""
    selected = X.read(X.ROOT / "development_choices.json")
    X.read(X.ROOT / "combination_results.json")
    auto = X.read(X.ROOT / "meta/O1_OptoPrimeV2_memory0/result.json")
    recursive = X.read(X.ROOT / "meta/O2/result.json")
    named = {
        "standard": dict(X.BASE),
        **{"axis_" + a: v["config"] for a, v in selected["winners"].items()},
        "combined": selected["mixtures"]["combined"],
        "automatic_O1": auto["selected_config"],
        "recursive_O2": recursive["selected_config"],
    }
    by_hash: dict[str, str] = {}
    configs = {}
    aliases = {}
    for name, config in named.items():
        X.validate_config(config)
        key = task.digest(config)
        if key not in by_hash:
            by_hash[key] = name
            configs[name] = config
        aliases[name] = by_hash[key]
    names = list(configs)
    order = {
        str(s): names[i % len(names) :] + names[: i % len(names)]
        for i, s in enumerate(X.TEST_SEEDS)
    }
    frozen = {
        "experiment": "EXP21-CONFIRMATION-v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "development_manifest_hash": task.digest(X.read(X.ROOT / "manifest.json")),
        "engineering_amendments": {
            p.name: task.digest(X.read(p))
            for p in sorted(X.ROOT.glob("*amendment*.json"))
        },
        "environment": {
            "python": sys.version,
            "packages": {
                name: version(name)
                for name in (
                    "litellm",
                    "numpy",
                    "pandas",
                    "pyarrow",
                    "opentelemetry-sdk",
                )
            },
        },
        "configurations": configs,
        "aliases": aliases,
        "seeds": X.TEST_SEEDS,
        "calls": X.CALLS,
        "order": order,
        "sources": {
            **X.sources(),
            str(
                Path(__file__).relative_to(task.Path(__file__).resolve().parents[3])
            ): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "primary": "mean TEST EM of validation-selected prefix policies0..6; unweighted outer-seed mean",
        "selection": "valid across every TRAIN/VALIDATION row; maximum VALIDATION EM, earliest slot ties; initial included",
        "bootstrap": "paired outer seeds, 10000 samples random.Random(20099), percentile indices249/9749; exploratory multi-axis comparisons",
        "success": "positive, null, negative or failed generated treatments all retained; no response replacement",
        "discovery_cost": "O1/O2 and screening costs separate from equal-response deployment-learning comparison; no equal-total-cost recursion claim",
    }
    C.retain(X.ROOT / "confirmation/manifest.json", frozen)
    return frozen


def verify_confirmation() -> dict[str, Any]:
    """Reject source or protocol drift before confirmation and re-analysis."""
    manifest = X.read(X.ROOT / "confirmation/manifest.json")
    if manifest["development_manifest_hash"] != task.digest(
        X.read(X.ROOT / "manifest.json")
    ):
        raise ValueError("development protocol drift")
    for name, digest in manifest["sources"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError("confirmatory source drift")
    return manifest


def confirmation(stage: str) -> None:
    """Run balanced configuration/seed jobs; final testing requires the global selection gate."""
    manifest = verify_confirmation()
    root = X.ROOT / "confirmation"
    names = list(manifest["configurations"])
    jobs = [
        {
            "root": str(root),
            "seed": seed,
            "name": manifest["order"][str(seed)][i],
            "config": manifest["configurations"][manifest["order"][str(seed)][i]],
            "mode": stage,
        }
        for i in range(len(names))
        for seed in manifest["seeds"]
    ]
    if stage == "test" and not (root / "test_gate.json").exists():
        raise ValueError(
            "TEST locked until every configuration/seed selection is frozen"
        )
    X.run_jobs(jobs)
    if stage == "select":
        selections = {
            str(seed): {
                name: X.read(root / "selection" / str(seed) / f"{name}.json")
                for name in names
            }
            for seed in manifest["seeds"]
        }
        X.freeze_test(
            root, names, manifest["seeds"], selections, calls=manifest["calls"]
        )


def summarize_curves(
    curves: dict[str, list[float]], seeds: list[int]
) -> dict[str, Any]:
    """Require all replications and all prefix points, including decreases and nonattainment."""
    if set(curves) != set(map(str, seeds)) or any(
        len(v) != X.CALLS + 1
        or not all(isinstance(x, (float, int)) and 0 <= x <= 1 for x in v)
        for v in curves.values()
    ):
        raise ValueError("complete finite seed/prefix curves required")
    values = [statistics.mean(curves[str(s)]) for s in seeds]
    return {
        "curves": curves,
        "primary_per_seed": dict(zip(map(str, seeds), values)),
        "primary_mean": statistics.mean(values),
        "primary_median": statistics.median(values),
        "mean_curve": [
            statistics.mean(curves[str(s)][p] for s in seeds)
            for p in range(X.CALLS + 1)
        ],
        "final_mean": statistics.mean(curves[str(s)][-1] for s in seeds),
        "first_target_prefix": {
            str(s): next((i for i, v in enumerate(curves[str(s)]) if v >= 0.7), None)
            for s in seeds
        },
    }


def usage(root: Path) -> dict[str, Any]:
    """Account every raw response and ambiguous request across all nested stages, without double counting."""
    groups: dict[str, dict[str, Any]] = {}
    for path in root.rglob("request.json"):
        if "reader_cache" in path.parts:
            role = "reader"
        elif "optimizer" in path.parts:
            role = "optimizer"
        else:
            continue
        stage = path.relative_to(root).parts[0]
        key = stage + "/" + role
        value = groups.setdefault(
            key,
            {
                "responses": 0,
                "pending": 0,
                "empty": 0,
                "length": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "cost_receipts": 0,
                "transport_failures": 0,
            },
        )
        response = path.with_name("response.json")
        value["transport_failures"] += sum(
            X.read(p)["event"] == "transient_failure"
            for p in path.parent.glob("transport_*.json")
        )
        if not response.exists():
            value["pending"] += 1
            continue
        receipt = X.read(response)
        value["responses"] += 1
        choice = receipt["response"]["choices"][0]
        value["empty"] += not bool(choice["message"].get("content"))
        value["length"] += choice.get("finish_reason") == "length"
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value[field] += receipt["usage"].get(field, 0) or 0
        cost = receipt["usage"].get("cost_usd")
        if cost is not None:
            value["cost_usd"] += cost
            value["cost_receipts"] += 1
    return groups


def analyze() -> dict[str, Any]:
    """Compute all registered comparisons without discarding an unfavorable seed or failed source."""
    manifest = verify_confirmation()
    root = X.ROOT / "confirmation"
    gate = X.read(root / "test_gate.json")
    seeds = manifest["seeds"]
    arms = {}
    for name in manifest["configurations"]:
        curves = {
            str(s): X.read(root / "reports" / str(s) / f"{name}.json")["curve"]
            for s in seeds
        }
        arms[name] = summarize_curves(curves, seeds)
    arms["unchanged"] = summarize_curves(
        {
            str(s): [arms["standard"]["curves"][str(s)][0]] * (X.CALLS + 1)
            for s in seeds
        },
        seeds,
    )
    contrasts = {}
    for name in arms:
        if name == "standard":
            continue
        contrasts[name + "-standard"] = A.paired(
            [arms[name]["primary_per_seed"][str(s)] for s in seeds],
            [arms["standard"]["primary_per_seed"][str(s)] for s in seeds],
        )
    auto, recursive = (
        manifest["aliases"]["automatic_O1"],
        manifest["aliases"]["recursive_O2"],
    )
    contrasts["recursive_O2-automatic_O1"] = A.paired(
        [arms[recursive]["primary_per_seed"][str(s)] for s in seeds],
        [arms[auto]["primary_per_seed"][str(s)] for s in seeds],
    )
    integrity = []
    for seed in seeds:
        for name in manifest["configurations"]:
            chain = X.read(root / "chains" / str(seed) / name / "result.json")
            selection = gate["selections"][str(seed)][name]
            measured = X.read(root / "reports" / str(seed) / f"{name}.json")
            if chain["optimizer_responses"] != manifest["calls"]:
                raise ValueError("incomplete completed-response budget")
            for h, item in gate["selections"][str(seed)][name]["candidates"].items():
                if h != task.digest(item["artifact"]):
                    raise ValueError("candidate source integrity failure")
            records = list(
                (root / "chains" / str(seed) / name / "optimizer").glob(
                    "**/response.json"
                )
            )
            if len(records) != manifest["calls"]:
                raise ValueError("missing raw response")
            integrity.append(
                {
                    "seed": seed,
                    "name": name,
                    "completed_responses": len(records),
                    "feedback_questions": chain["unique_questions_in_feedback"],
                    "curriculum_transitions": len(chain["curriculum_events"]),
                    "training_evaluations": chain["actual_train_evaluations"],
                    "candidates_in_selection": len(selection["evaluations"]),
                    "ineligible_candidates": sum(
                        not e["valid"] for e in selection["evaluations"]
                    ),
                    "selected_prefix_hashes": selection["prefixes"],
                    "final_artifact": str(
                        root
                        / "chains"
                        / str(seed)
                        / name
                        / "artifacts"
                        / f"{selection['prefixes'][-1]}.json"
                    ),
                    "final_is_initial": selection["prefixes"][-1]
                    == task.digest(task.INITIAL),
                    "final_F1": measured["evaluations"][selection["prefixes"][-1]][
                        "F1"
                    ],
                    "unique_selected_test_questions": sum(
                        e["questions"] for e in measured["evaluations"].values()
                    ),
                    "test_invalid_executions": sum(
                        e["candidate_invalid"] for e in measured["evaluations"].values()
                    ),
                    "test_fallbacks": sum(
                        e["fallback_count"] for e in measured["evaluations"].values()
                    ),
                }
            )
    result = {
        "experiment": manifest["experiment"],
        "aliases": manifest["aliases"],
        "arms": arms,
        "contrasts": contrasts,
        "integrity": integrity,
        "usage": usage(X.ROOT),
        "scope": "paired descriptive intervals; fixed question panel; n6; selected development winners; no global-optimum/novelty/amortization claim",
    }
    C.retain(root / "results.json", result)
    return result


def render(result: dict[str, Any]) -> None:
    """Publish every axis and automatic-policy result in the existing human entry point."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = X.ROOT / "confirmation"
    manifest = X.read(root / "manifest.json")
    choices = X.read(X.ROOT / "development_choices.json")
    names = {
        "batch": "Batch / curriculum",
        "trace": "Trace",
        "surface": "Surface",
        "feedback": "Feedback",
        "goal": "Goal",
        "optimizer": "Optimiseur",
        "trainer": "Trainer",
    }
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), layout="constrained", sharey=True)
    base = result["arms"]["standard"]["mean_curve"]
    initial = result["arms"]["unchanged"]["mean_curve"]
    treatments = [("axis_" + a, n) for a, n in names.items()] + [
        ("automatic_O1", "Optimisation automatique O1"),
        ("recursive_O2", "Récursion O2"),
    ]
    for ax, (label, title) in zip(axes.flat, treatments):
        actual = manifest["aliases"][label]
        curve = result["arms"][actual]["mean_curve"]
        ax.plot(
            range(7), [100 * v for v in initial], ":", color="gray", label="Inchangé"
        )
        ax.plot(
            range(7), [100 * v for v in base], "--", color="#2563eb", label="Standard"
        )
        ax.plot(range(7), [100 * v for v in curve], "o-", color="#d97706", label=label)
        ax.set(
            title=title,
            xlabel="Réponses O0",
            ylabel="Exactitude TEST (%)",
            ylim=(0, 100),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "EXP21 — choix sur développement, sélection sur VALIDATION, mesure sur TEST\nSix graines ; les coûts de découverte O1/O2 sont supplémentaires"
    )
    fig.savefig(root / "progressions.png", dpi=150)
    fig.savefig(root / "progressions.svg")
    plt.close(fig)
    lines = [
        "# EXP-21 — huit axes et optimisation automatique de l’optimiseur",
        "",
        "**Campagne terminée.** Les résultats ci-dessous incluent toutes les graines et tous les slots.",
        "Qwen `qwen/qwen-2.5-7b-instruct` résout les questions ; DeepSeek `deepseek/deepseek-v4-flash-0731` optimise les programmes et les configurations.",
        "",
        "## Résultats de confirmation",
        "",
        "Le score principal est la moyenne de l’exactitude TEST des programmes choisis sur VALIDATION aux préfixes 0…6. Une meilleure valeur signifie une meilleure qualité obtenue tôt, à nombre de réponses O0 égal. Les coûts lecteur et de découverte restent différents.",
        "",
        "| Traitement | Configuration réellement exécutée | Score principal | Exactitude finale | Delta principal vs standard [IC bootstrap 95 %] |",
        "|---|---|---:|---:|---|",
    ]
    for label in manifest["aliases"]:
        actual = manifest["aliases"][label]
        arm = result["arms"][actual]
        contrast = result["contrasts"].get(actual + "-standard")
        delta = (
            "Référence / configuration identique"
            if contrast is None
            else f"{100*contrast['mean_delta']:+.2f} [{100*contrast['bootstrap_95_percentile'][0]:+.2f} ; {100*contrast['bootstrap_95_percentile'][1]:+.2f}] points"
        )
        lines.append(
            f"| {label} | {actual} | {100*arm['primary_mean']:.2f} % | {100*arm['final_mean']:.2f} % | {delta} |"
        )
    lines.extend(
        [
            "",
            f"Programme inchangé : {100*result['arms']['unchanged']['primary_mean']:.2f} %.",
            "Les configurations identiques sont exécutées une seule fois et gardent tous leurs alias ; elles ne constituent pas des réplications indépendantes.",
            "Intervalles descriptifs à n=6, conditionnels aux questions du panel. Les multiples contrastes ne justifient pas des déclarations de significativité. Aucun optimum global, nouveauté algorithmique ou amortissement n’est établi.",
            "",
            "![Progressions par axe](exp21/confirmation/progressions.png)",
            "",
            "## Ce que l’optimisation automatique a choisi",
            "",
            "O1 modifie les paramètres natifs du Control Plane. Son évaluation lance réellement l’apprentissage O0 sur développement. O2 choisit le type d’optimiseur et la mémoire d’O1 ; les exécutions et réutilisations de résultats sont conservées.",
            "",
            "| Paramètre d’apprentissage | Standard | O1 choisi | Issu d’O2 |",
            "|---|---|---|---|",
        ]
    )
    automatic = manifest["configurations"][manifest["aliases"]["automatic_O1"]]
    recursive = manifest["configurations"][manifest["aliases"]["recursive_O2"]]
    for key in X.BASE:
        lines.append(
            f"| `{key}` | `{X.BASE[key]}` | `{automatic[key]}` | `{recursive[key]}` |"
        )
    nested = result["contrasts"]["recursive_O2-automatic_O1"]
    lines.extend(
        [
            "",
            f"O2 − O1 : {100*nested['mean_delta']:+.2f} points, intervalle [{100*nested['bootstrap_95_percentile'][0]:+.2f} ; {100*nested['bootstrap_95_percentile'][1]:+.2f}]. Les budgets de découverte diffèrent : ce n’est pas une mesure de l’avantage de profondeur à coût total égal.",
            "",
            "## Tous les contrastes de développement",
            "",
            "**Exploratoires** : ils ont servi à sélectionner les traitements ci-dessus. Les questions et graines de confirmation sont différentes.",
            "",
            "| Variante | Axe | Score principal développement | Delta vs standard |",
            "|---|---|---:|---:|",
        ]
    )
    for variant in X.variants():
        score = choices["scores"][variant["id"]]
        lines.append(
            f"| {variant['id']} | {variant['axis']} | {100*score:.2f} % | {100*(score-choices['scores']['standard']):+.2f} points |"
        )
    lines.extend(
        [
            "",
            "## Combinaison et ablations de développement",
            "",
            "| Traitement | Score principal développement |",
            "|---|---:|",
        ]
    )
    for label, value in X.read(X.ROOT / "combination_results.json").items():
        lines.append(f"| {label} | {100*value['primary']:.2f} % |")
    lines.extend(
        [
            "",
            "## Appels, coûts et validité",
            "",
            "| Étape / rôle | Réponses | Vides | Fin `length` | Tokens | Coût rapporté USD |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, value in result["usage"].items():
        lines.append(
            f"| {name} | {value['responses']} | {value['empty']} | {value['length']} | {value['total_tokens']} | {value['cost_usd']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Les coûts de développement, pilote, O1/O2 et confirmation sont séparés. Les cache hits évitent un paiement répété mais ne créent pas des propositions supplémentaires. Les non-atteintes du seuil 70 % restent censurées.",
            "[Résultats complets, par graine et invalidité](exp21/confirmation/results.json), [protocole de confirmation](exp21/confirmation/manifest.json), [sélections gelées avant TEST](exp21/confirmation/test_gate.json).",
            "",
            "## Protocole et historique conservés",
            "",
        ]
    )
    path = X.ROOT.parent / "EXP21.md"
    previous = path.read_text()
    marker = "## Question et périmètre"
    if marker in previous:
        lines.append(previous[previous.index(marker) :])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    """Keep stage transitions explicit, frozen and resumable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["combine", "freeze", "fit", "select", "test", "analyze"]
    )
    args = parser.parse_args()
    X.verify()
    if args.stage == "combine":
        combine()
    elif args.stage == "freeze":
        report = freeze_confirmation()
        print(
            json.dumps(
                {
                    "configurations": list(report["configurations"]),
                    "aliases": report["aliases"],
                }
            ),
            flush=True,
        )
    elif args.stage == "analyze":
        result = analyze()
        render(result)
        print(
            json.dumps(
                {
                    name: {"primary": row["primary_mean"], "final": row["final_mean"]}
                    for name, row in result["arms"].items()
                }
            ),
            flush=True,
        )
    else:
        confirmation(args.stage)


if __name__ == "__main__":
    main()
