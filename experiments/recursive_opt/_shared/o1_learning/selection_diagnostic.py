"""EXP19-S4: isolate parent selection during fitting, reusing the common run/evaluator."""

from __future__ import annotations

import json
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from experiments.recursive_opt._shared.o1_learning import analysis, study

SEEDS = [19083, 19097, 19109]


def prompt_examples(request: dict[str, Any]) -> list[str]:
    """Extract unique primitive TRAIN identities actually present in feedback."""
    prompt = "\n".join(message["content"] for message in request["kwargs"]["messages"])
    return sorted(set(re.findall(r"for (\['[A-F]', [01], [01]\])", prompt)))


def configured_spec(config: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Alter only the canonical fit-validation split; external selection remains shared."""
    if config["id"] == "train_only_fit":
        spec["levels"][0]["datasets"]["validation"] = []
    return spec


def run_seed(job: tuple[int, int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Run a rotated pair in one isolated process, preserving the common evaluator."""
    index, seed, arms = job
    order = arms if index % 2 == 0 else list(reversed(arms))
    reports = []
    original = study.specification
    for config in order:

        def specification(*args: Any, **kwargs: Any) -> dict[str, Any]:
            """Pass the declared dataset choice through the existing canonical dict."""
            return configured_spec(config, original(*args, **kwargs))

        study.specification = specification
        try:
            reports.append(study.run(config, seed, "selection_s4", calls=4))
        finally:
            study.specification = original
    return reports


def execute() -> None:
    """Run the registered parent-selection contrast, freezing every policy before TEST."""
    from experiments.recursive_opt._shared.optimizer_discovery.phase0 import _load_key

    _load_key()
    base = {**study.variants()[0], "batch_size": 6, "max_tokens": 16000}
    arms = [{**base, "id": "standard_B6"}, {**base, "id": "train_only_fit"}]
    with ProcessPoolExecutor(
        max_workers=3, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        groups = list(
            executor.map(
                run_seed, [(index, seed, arms) for index, seed in enumerate(SEEDS)]
            )
        )
    reports = [row for group in groups for row in group]
    matched_batches = []
    for seed in SEEDS:
        for slot in range(4):
            batches = []
            for arm in arms:
                path = (
                    study.ROOT
                    / "raw"
                    / "selection_s4"
                    / f"{arm['id']}_{seed}"
                    / f"request_{slot}.json"
                )
                batches.append(prompt_examples(json.loads(path.read_text())))
            if len(batches[0]) != 6 or batches[0] != batches[1]:
                raise RuntimeError(
                    "S4 consumed TRAIN batches differ; TEST remains closed"
                )
            matched_batches.append({"seed": seed, "slot": slot, "examples": batches[0]})
    study.persist(study.ROOT / "s4_batch_match.json", matched_batches)
    evaluated = analysis.freeze_and_score(
        reports, 4, "s4_prefix_selections_frozen.json"
    )
    values = {
        arm["id"]: {
            row["seed"]: row["final_accuracy"]
            for row in evaluated
            if row["arm"] == arm["id"]
        }
        for arm in arms
    }
    values["initial"] = {
        seed: study.score(study.START, study.panels(seed)["test"]) for seed in SEEDS
    }
    result = {
        "id": "EXP19-S4",
        "seeds": SEEDS,
        "arms": values,
        "per_seed": evaluated,
        "train_only_minus_standard": analysis.paired(
            values["train_only_fit"], values["standard_B6"], seeds=SEEDS
        ),
        "completed_responses": sum(row["calls"] for row in reports),
        "interpretation": "prospectively registered mechanism diagnostic after observing a parent rollback; same concepts, new expressions; exploratory",
    }
    study.persist(study.ROOT / "s4_results.json", result)
    analysis.update_progress()
    print(json.dumps(result["train_only_minus_standard"]), flush=True)


if __name__ == "__main__":
    execute()
