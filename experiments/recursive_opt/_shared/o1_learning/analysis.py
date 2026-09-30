"""EXP-19 registered selection, ablations, paired test, and transparent cost accounting."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from experiments.recursive_opt._shared.o1_learning import study

AXIS_FIELDS = {
    "batch": ["batch_size", "curriculum"],
    "trace": ["mode", "detail", "credit_horizon"],
    "surface": ["surface"],
    "feedback": ["feedback"],
    "goal": ["goal"],
    "optimizer": ["optimizer"],
    "trainer": ["trainer"],
}
TEST_SEEDS = [19031, 19043, 19059]


def rank(row: dict[str, Any]) -> tuple[bool, int, float]:
    """Prefer valid target attainment at fewer calls; break censored ties by accuracy."""
    return (
        row["valid"],
        -(row["first_hit"] if row["first_hit"] is not None else 5),
        row["selection"]["accuracy"],
    )


def paired(
    left: dict[int, float], right: dict[int, float], *, seeds: list[int] = TEST_SEEDS
) -> dict[str, Any]:
    """Bootstrap complete paired outer-seed differences, never treating tasks as replications."""
    if set(left) != set(seeds) or set(right) != set(seeds):
        raise ValueError("paired analysis requires every registered seed exactly once")
    deltas = np.array([left[seed] - right[seed] for seed in seeds])
    rng = np.random.default_rng(19090)
    estimates = rng.choice(deltas, (10000, len(seeds)), replace=True).mean(axis=1)
    return {
        "seeds": seeds,
        "deltas": deltas.tolist(),
        "mean": float(deltas.mean()),
        "median": float(np.median(deltas)),
        "bootstrap_95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "interpretation": "exploratory; three outer seeds; fragile interval",
    }


def extra_variants() -> list[dict[str, Any]]:
    """Separate detail from horizon and use an actually distinct trainer policy."""
    base = study.variants()[0]
    return [
        {**base, "id": "detail_only", "axis": "trace", "detail": "full"},
        {**base, "id": "horizon_only", "axis": "trace", "credit_horizon": "full"},
        {**base, "id": "pareto_trainer", "axis": "trainer", "trainer": "ParetobasedPS"},
    ]


def update_progress() -> None:
    """Refresh one readable table without multiplying summary documents."""
    rows = [
        json.loads(path.read_text())
        for path in sorted((study.ROOT / "raw").glob("*/*/result.json"))
    ]
    # Nested optimizer summaries use a different schema, retained separately in
    # recursion_result.json. This table covers every task-learning run, invalid too.
    rows = [row for row in rows if "phase" in row]
    table = [
        "<!-- LIVE_RESULTS -->",
        "## Résultats disponibles — développement, pas test final",
        "",
        "| Phase | Réglage | Graine | Réponses | Exactitude sélection | Appels à 90 % | Validité |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        hit = row["first_hit"] if row["first_hit"] is not None else "non atteint"
        table.append(
            f"| {row['phase']} | {row['config']['id']} | {row['seed']} | {row['calls']} | {row['selection']['accuracy']:.3f} | {hit} | {row['valid']} |"
        )
    table.append("<!-- END_LIVE_RESULTS -->")
    path = study.ROOT / "EXP19.md"
    text = path.read_text()
    start, end = text.find("<!-- LIVE_RESULTS -->"), text.find(
        "<!-- END_LIVE_RESULTS -->"
    )
    if start >= 0:
        text = (
            text[:start]
            + "\n".join(table)
            + text[end + len("<!-- END_LIVE_RESULTS -->") :]
        )
    else:
        text += "\n\n" + "\n".join(table) + "\n"
    path.write_text(text)


def screen_reports() -> list[dict[str, Any]]:
    """Require the complete initial screen before choosing a combined design."""
    reports = []
    for config in study.variants():
        path = study.ROOT / "raw" / "screen" / f"{config['id']}_19021" / "result.json"
        if not path.exists():
            raise RuntimeError(f"screen incomplete: {config['id']}")
        reports.append(json.loads(path.read_text()))
    return reports


def combine(reports: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """Combine only improvements observed in the prespecified individual screen."""
    baseline = next(row for row in reports if row["config"]["id"] == "standard")
    config = copy.deepcopy(baseline["config"])
    changed = []
    for axis, fields in AXIS_FIELDS.items():
        contenders = [row for row in reports if row["config"]["axis"] == axis]
        # The original joint detail+horizon diagnostic is not an isolated factor.
        contenders = [row for row in contenders if row["config"]["id"] != "trace_full"]
        if not contenders:
            continue
        winner = max(contenders, key=rank)
        if rank(winner) > rank(baseline):
            for field in fields:
                config.pop(field, None)
                if field in winner["config"]:
                    config[field] = copy.deepcopy(winner["config"][field])
            changed.append(axis)
    return {**config, "id": "combined", "axis": "combined"}, changed


def remember_observations(examples: list[dict[str, Any]]) -> dict[str, str]:
    """Diagnostic reference: store supplied primitive labels without any LLM call."""
    policy = dict(study.START)
    for item in examples:
        name, left, right = item["tree"]
        if not isinstance(left, int) or not isinstance(right, int):
            raise ValueError(
                "observation-memory diagnostic accepts primitive examples only"
            )
        bits = list(policy[name])
        bits[2 * left + right] = str(item["expected"])
        policy[name] = "".join(bits)
    return policy


def coverage_diagnostic(report: dict[str, Any]) -> dict[str, Any]:
    """Separate limited training coverage from imperfect use of available observations."""
    path = (
        study.ROOT
        / "raw"
        / report["phase"]
        / f"{report['config']['id']}_{report['seed']}"
        / "evaluations.json"
    )
    records = json.loads(path.read_text())
    panels = study.panels(report["seed"])
    known = {study.digest(item["tree"]): item for item in panels["train"]}
    curve = []
    coverage = []
    for calls in range(report["calls"] + 1):
        seen = {
            row["tree_hash"]
            for row in records
            if row["calls"] < calls and row["tree_hash"] in known
        }
        policy = remember_observations([known[key] for key in sorted(seen)])
        curve.append(study.score(policy, panels["validation"]))
        coverage.append(len(seen))
    return {
        "variant": report["config"]["id"],
        "memory_reference_curve": curve,
        "unique_train_examples": coverage,
        "actual_curve": report["validation_curve"],
        "LLM_calls_reference": 0,
        "interpretation": "same already-observed primitive labels; diagnostic specialized learner, not a discovered optimizer",
    }


def token_probe(index: int) -> dict[str, Any]:
    """Retain a new 16k generation response to an archived prompt without replacing it."""
    origin = study.ROOT / "raw" / "screen" / "batch5_19021" / f"request_{index}.json"
    request = json.loads(origin.read_text())
    client = study.RecordingClient(
        study.ROOT / "raw" / "token_probe" / f"prompt_{index}"
    )
    kwargs = request["kwargs"]
    kwargs["max_tokens"] = 16000
    response = client(*request["args"], **kwargs)
    text = study.S._optimizer_response_text(response)
    row = {
        "origin": str(origin),
        "limit": 16000,
        "has_text": bool(text),
        "finish_reason": response.choices[0].finish_reason,
        "usage": client.usage,
    }
    study.persist(client.directory / "diagnostic.json", row)
    return row


def run_job(job: tuple[dict[str, Any], int, str]) -> dict[str, Any]:
    """Run one independent learning job inside its own process and RNG context."""
    config, seed, phase = job
    return study.run(config, seed, phase)


def run_jobs(jobs: list[tuple[dict[str, Any], int, str]]) -> list[dict[str, Any]]:
    """Use three isolated research workers; each live optimization remains sequential."""
    with ProcessPoolExecutor(
        max_workers=3, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        results = list(executor.map(run_job, jobs))
    update_progress()
    return results


def run_seed(job: tuple[int, int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Rotate all learned arms within one paired outer seed."""
    index, seed, arms = job
    order = arms[index:] + arms[:index]
    return [study.run(config, seed, "paired_learning") for config in order]


def freeze_and_score(
    runs: list[dict[str, Any]],
    calls_budget: int,
    filename: str,
) -> list[dict[str, Any]]:
    """Freeze validation-only prefix policies before any held-out scoring."""
    # Freeze every prefix policy before evaluating any TEST expression.
    selections = []
    for row in runs:
        if not row["valid"] and "no final textual content" not in (
            row.get("error") or ""
        ):
            raise RuntimeError(
                "incomplete or invalid paired learning run; test remains closed"
            )
        eligible = [
            candidate for candidate in row["candidate_results"] if candidate["valid"]
        ]
        prefixes = [
            max(
                (candidate for candidate in eligible if candidate["calls"] <= calls),
                key=lambda candidate: (candidate["accuracy"], -candidate["calls"]),
            )
            for calls in range(calls_budget + 1)
        ]
        selections.append(
            {"seed": row["seed"], "arm": row["config"]["id"], "prefixes": prefixes}
        )
    path = study.ROOT / filename
    if not path.exists():
        study.persist(path, selections)
    elif json.loads(path.read_text()) != selections:
        raise ValueError("prefix selections differ from frozen evidence")
    evaluated = []
    for selection in selections:
        panel = study.panels(selection["seed"])["test"]
        curve = [
            study.score(item["configuration"], panel) for item in selection["prefixes"]
        ]
        evaluated.append(
            {
                **selection,
                "test_curve": curve,
                "final_accuracy": curve[-1],
                "first_hit_test": study.first_hit(curve),
            }
        )
    return evaluated


def continue_study() -> None:
    """Execute extra activation contrasts, combined/ablation checks, then frozen three-way test."""
    from experiments.recursive_opt._shared.optimizer_discovery.phase0 import _load_key

    _load_key()
    reports = screen_reports()
    reports.extend(
        run_jobs([(config, 19021, "screen_extra") for config in extra_variants()])
    )
    with ProcessPoolExecutor(
        max_workers=3, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        probes = list(executor.map(token_probe, [0, 2, 3]))
    study.persist(study.ROOT / "token_diagnostics.json", probes)
    batch_configs = [
        {**config, "max_tokens": 16000}
        for config in study.variants()
        if config["id"]
        in {"standard", "batch5", "batch7", "curriculum3", "curriculum5", "curriculum7"}
    ]
    batch_reports = run_jobs(
        [(config, 19024, "active_curriculum") for config in batch_configs]
    )
    combined, changed = combine(
        [row for row in reports if row["config"]["axis"] != "batch"]
    )
    batch_baseline = next(
        row for row in batch_reports if row["config"]["id"] == "standard"
    )
    batch_winner = max(batch_reports, key=rank)
    if rank(batch_winner) > rank(batch_baseline):
        for field in AXIS_FIELDS["batch"]:
            combined.pop(field, None)
            if field in batch_winner["config"]:
                combined[field] = copy.deepcopy(batch_winner["config"][field])
        changed.append("batch")
    combined["max_tokens"] = 16000
    baseline = {**study.variants()[0], "max_tokens": 16000}
    configs = [combined, baseline]
    for axis in changed:
        ablation = copy.deepcopy(combined)
        ablation.update(id=f"without_{axis}", axis="ablation")
        for field in AXIS_FIELDS[axis]:
            ablation.pop(field, None)
            if field in baseline:
                ablation[field] = baseline[field]
        configs.append(ablation)
    path = study.ROOT / "combination_manifest.json"
    if not path.exists():
        study.persist(
            path,
            {
                "seed": 19022,
                "configs": configs,
                "changed_axes": changed,
                "rule": "same frozen rank; selection and ablation only, no test access",
            },
        )
    combinations = run_jobs([(config, 19022, "ablation") for config in configs])
    from experiments.recursive_opt._shared.o1_learning.recursion import execute as execute_recursion

    recursive_result = execute_recursion()
    recursive_config = {
        **recursive_result["selected_config"],
        "id": "recursive_selected",
        "max_tokens": 16000,
    }
    winner = max(combinations, key=rank)
    selected_config = {**winner["config"], "id": "O1_selected"}
    freeze = {
        "config": selected_config,
        "selected_on": 19022,
        "test_seeds": TEST_SEEDS,
        "selection_rule": "highest valid rank on development; earliest config on tie",
        "selected_development": winner["selection"],
        "recursive_config": recursive_config,
        "test_access": False,
    }
    path = study.ROOT / "test_selection_frozen.json"
    if not path.exists():
        study.persist(path, freeze)
    elif json.loads(path.read_text()) != freeze:
        raise ValueError("selection differs from frozen selection")
    arms = [baseline, selected_config, recursive_config]
    with ProcessPoolExecutor(
        max_workers=3, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        groups = list(
            executor.map(
                run_seed, [(index, seed, arms) for index, seed in enumerate(TEST_SEEDS)]
            )
        )
    runs = [row for group in groups for row in group]
    update_progress()
    evaluated = freeze_and_score(runs, 4, "all_prefix_selections_frozen.json")
    initial = {
        seed: study.score(study.START, study.panels(seed)["test"])
        for seed in TEST_SEEDS
    }
    standard = {
        row["seed"]: row["final_accuracy"]
        for row in evaluated
        if row["arm"] == "standard"
    }
    meta = {
        row["seed"]: row["final_accuracy"]
        for row in evaluated
        if row["arm"] == "O1_selected"
    }
    recursive = {
        row["seed"]: row["final_accuracy"]
        for row in evaluated
        if row["arm"] == "recursive_selected"
    }
    final = {
        "id": "EXP-19",
        "initial": initial,
        "standard": standard,
        "O1": meta,
        "recursive_selected": recursive,
        "recursive_minus_standard": paired(recursive, standard),
        "recursive_minus_O1": paired(recursive, meta),
        "recursive_preparation_calls": recursive_result["actual_calls"],
        "per_seed": evaluated,
        "O1_minus_standard": paired(meta, standard),
        "O1_minus_initial": paired(meta, initial),
        "prior_O1_completed_calls": sum(
            row["calls"] for row in reports + batch_reports + combinations
        ),
        "paired_completed_calls": sum(row["calls"] for row in runs),
        "interpretation": "exploratory selected configuration; not a general recursion-depth or amortization claim",
    }
    study.persist(study.ROOT / "results.json", final)
    print(json.dumps(final["O1_minus_standard"]), flush=True)


def run_extended_seed(
    job: tuple[int, int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Apply rotated eight-response learning arms to one independent outer seed."""
    index, seed, arms = job
    order = arms[index:] + arms[:index]
    return [study.run(config, seed, "paired_s3", calls=8) for config in order]


def repair_study() -> None:
    """Validate repaired projections and run the separately registered longer comparison."""
    from experiments.recursive_opt._shared.optimizer_discovery.phase0 import _load_key
    from experiments.recursive_opt._shared.o1_learning import recursion

    _load_key()
    configs = study.variants() + extra_variants()
    diagnostics = [
        {**next(item for item in configs if item["id"] == name), "max_tokens": 16000}
        for name in ["standard", "detail_only", "trace_step", "hybrid"]
    ]
    run_jobs([(config, 19025, "projection_s3") for config in diagnostics])
    original_root = study.ROOT
    try:
        study.ROOT = original_root / "s3"
        recursive_result = recursion.execute()
    finally:
        study.ROOT = original_root
    combinations = [
        json.loads(path.read_text())
        for path in sorted((study.ROOT / "raw" / "ablation").glob("*/result.json"))
    ]
    # Restore registered config order rather than filesystem order for tied ranks.
    order = json.loads((study.ROOT / "combination_manifest.json").read_text())[
        "configs"
    ]
    combinations.sort(
        key=lambda row: [item["id"] for item in order].index(row["config"]["id"])
    )
    winner = max(combinations, key=rank)
    arms = [
        {**study.variants()[0], "max_tokens": 16000},
        {**winner["config"], "id": "O1_selected", "max_tokens": 16000},
        {
            **recursive_result["selected_config"],
            "id": "recursive_selected",
            "max_tokens": 16000,
        },
    ]
    seeds = [19061, 19073, 19079]
    freeze = {
        "arms": arms,
        "seeds": seeds,
        "max_responses": 8,
        "test_access": False,
        "recursive_result_valid": recursive_result["O2"]["valid"],
    }
    path = study.ROOT / "s3_selection_frozen.json"
    if not path.exists():
        study.persist(path, freeze)
    elif json.loads(path.read_text()) != freeze:
        raise ValueError("S3 configuration differs from frozen selection")
    with ProcessPoolExecutor(
        max_workers=3, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        groups = list(
            executor.map(
                run_extended_seed,
                [(index, seed, arms) for index, seed in enumerate(seeds)],
            )
        )
    runs = [row for group in groups for row in group]
    update_progress()
    evaluated = freeze_and_score(runs, 8, "s3_prefix_selections_frozen.json")
    values = {
        arm["id"]: {
            row["seed"]: row["final_accuracy"]
            for row in evaluated
            if row["arm"] == arm["id"]
        }
        for arm in arms
    }
    values["initial"] = {
        seed: study.score(study.START, study.panels(seed)["test"]) for seed in seeds
    }
    result = {
        "id": "EXP19-S3",
        "seeds": seeds,
        "per_seed": evaluated,
        "arms": values,
        "O1_minus_standard": paired(
            values["O1_selected"], values["standard"], seeds=seeds
        ),
        "recursive_minus_standard": paired(
            values["recursive_selected"], values["standard"], seeds=seeds
        ),
        "recursive_minus_O1": paired(
            values["recursive_selected"], values["O1_selected"], seeds=seeds
        ),
        "paired_completed_calls": sum(row["calls"] for row in runs),
        "recursive_preparation_calls": recursive_result["actual_calls"],
        "interpretation": "exploratory three-seed comparison after engineering repair, new expressions but same concepts; O1 selection cost separate",
    }
    study.persist(study.ROOT / "s3_results.json", result)
    print(json.dumps(result["O1_minus_standard"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["progress", "continue", "repair"])
    arguments = parser.parse_args()
    {"progress": update_progress, "continue": continue_study, "repair": repair_study}[
        arguments.command
    ]()
