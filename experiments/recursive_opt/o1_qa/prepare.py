"""Prepare a pinned dataset and a reviewable pilot protocol without live calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from . import task

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "experiments/recursive_opt/_shared/o1_learning/exp20_manifest.json"
REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
DATA_SHA = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
DATA_PATH = (
    Path.home()
    / f".cache/huggingface/hub/datasets--hotpotqa--hotpot_qa/snapshots/{REVISION}/distractor/validation-00000-of-00001.parquet"
)
COUNTS = {
    "pilot_diagnostic": 24,
    "pilot_train": 24,
    "pilot_selection": 24,
    "train": 60,
    "validation": 24,
    "holdout": 48,
}
SOURCE_PATHS = [
    "experiments/recursive_opt/o1_qa/task.py",
    "experiments/recursive_opt/o1_qa/prepare.py",
    "experiments/recursive_opt/o1_qa/pilot.py",
    "opto/trainer/algorithms/priority_search.py",
    "opto/trainer/search_template.py",
    "opto/trainer/loader.py",
    "opto/trainer/sampler.py",
    "opto/features/recursive_opt/spec.py",
    "opto/features/recursive_opt/traces.py",
    "opto/features/recursive_opt/runmode.py",
    "opto/optimizers/optoprime.py",
    "opto/optimizers/optoprime_v2.py",
]


def load_rows(path: Path = DATA_PATH) -> list[dict[str, Any]]:
    """Read only the pinned local parquet; no network, provider, or dataset substitution."""
    import pyarrow.parquet as pq

    if not path.is_file():
        raise FileNotFoundError(
            "pinned HotpotQA parquet is absent; supply --dataset pointing to that revision"
        )
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA:
        raise ValueError("HotpotQA source SHA256 mismatch")
    result = []
    for item in pq.read_table(path).to_pylist():
        context = list(zip(item["context"]["title"], item["context"]["sentences"]))
        supporting = list(
            zip(item["supporting_facts"]["title"], item["supporting_facts"]["sent_id"])
        )
        # Structural limits fixed before model outcomes. Do not truncate hidden evidence.
        if (
            len(context) != 10
            or sum(len(s) for _, sentences in context for s in sentences) > 48000
        ):
            continue
        result.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": item["answer"],
                "context": context,
                "type": item["type"],
                "supporting_facts": supporting,
            }
        )
    return result


def source_hashes() -> dict[str, str]:
    """Seal the relevant dirty-tree implementation by file content, not HEAD alone."""
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in SOURCE_PATHS
    }


def panels(path: Path = DATA_PATH) -> dict[str, list[dict[str, Any]]]:
    """Reconstruct all identities deterministically without evaluating any holdout."""
    return task.split_rows(load_rows(path), COUNTS, seed=20001)


def build_manifest(path: Path = DATA_PATH) -> dict[str, Any]:
    """Freeze pilot gates and draft the later curriculum comparison for user review."""
    data = panels(path)
    spec = task.specification(
        data["pilot_train"], seed=20003, curriculum=False, calls=2
    )
    return {
        "experiment": "EXP-20",
        "status": "DRAFT_FOR_USER_REVIEW_NO_LIVE_RUN",
        "stage": "readiness pilot only; confirmatory freeze still required",
        "dataset": {
            "repo": "hotpotqa/hotpot_qa",
            "revision": REVISION,
            "sha256": DATA_SHA,
            "source_split": "validation/distractor",
            "license": "CC-BY-SA-4.0",
            "url": f"https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/{REVISION}/distractor",
            "max_document_chars": 48000,
            "sampling_seed": 20001,
            "strata": ["bridge", "comparison"],
            "equal_stratum_weight": True,
        },
        "splits": {
            name: {
                "count": len(rows),
                "ids": [r["id"] for r in rows],
                "rows_hash": task.digest(rows),
            }
            for name, rows in data.items()
        },
        "initial_artifact": task.INITIAL,
        "initial_artifact_hash": task.digest(task.INITIAL),
        "llm_profiles": spec["llm_profiles"],
        "source_sha256": source_hashes(),
        "pilot": {
            "variants": ["seed", "repeat_seed", "all_documents", "oracle_documents"],
            "reader_slots": 96,
            "optimizer_slots": 0,
            "workers": 8,
            "deadline_s": 900,
            "oracle_is_diagnostic_only": True,
            "gates": {
                "maximum_seed_EM": 0.85,
                "minimum_oracle_EM": 0.50,
                "minimum_seed_errors_solved_by_oracle": 3,
                "maximum_format_invalid_fraction": 0.10,
                "maximum_seed_repeat_correctness_flips": 2,
            },
        },
        "fit_pilot": {
            "outer_seed": 20003,
            "proposals_per_arm": 2,
            "arms": ["standard", "curriculum"],
            "batch_size": 6,
            "history_size": 2,
            "selection_score_window": "latest_train_batch",
            "status": "canonical spec and offline integration tested; not executed",
        },
        "confirmation_draft": {
            "outer_seeds": [20011, 20023, 20037, 20041, 20053, 20071],
            "arms": ["unchanged", "standard_training", "curriculum_training"],
            "proposals_per_learning_arm": 6,
            "batch_size": 6,
            "history_size": 2,
            "primary": "equal-prefix validation-selected held-out answer_EM AUC at prefixes 0..6",
            "training_coverage_gate": "at least 20 distinct questions actually supplied to optimizer feedback per completed chain; otherwise report inadequate coverage, not curriculum inefficacy",
            "target_EM": 0.70,
            "nonattainment": "censored, never omit",
            "selection": "full validation only after fitting; earliest eligible prefix tie, including seed",
            "holdout_gate": "all outer-seed prefix selections frozen before any holdout evaluation",
            "cache": "exact reader request+model+settings within outer seed, shared across arms; immutable response",
            "budget_reporting": [
                "optimizer responses",
                "reader logical requests",
                "reader paid calls",
                "cache hits",
                "tokens",
                "cost",
                "time",
                "invalidity",
            ],
            "uncertainty": "paired outer-seed bootstrap, 10000 draws, seed 20099; exploratory at n=6",
            "secondary": "learning curve versus total paid reader+optimizer calls, not optimizer calls alone",
            "call_planning_estimate": {
                "optimizer": 72,
                "reader_uncached": 8064,
                "explanation": "12*(3*6*6+60+7*24+7*48); approximate planning count, internal terminal/replay evaluations must be audited in fit pilot before confirmation",
            },
            "one_hour": "conditional on measured end-to-end pilot forecast; incomplete is not a negative result",
        },
    }


def verify_manifest(
    manifest: dict[str, Any], path: Path = DATA_PATH
) -> dict[str, list[dict[str, Any]]]:
    """Fail before live work if data, seed policy or implementation drifted."""
    if manifest != build_manifest(path):
        raise ValueError(
            "manifest/config/source drift; revise the draft explicitly before launch"
        )
    return panels(path)


def main() -> None:
    """Write one draft, or verify it; never overwrite an existing protocol silently."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATA_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    task.register()
    if args.verify:
        data = verify_manifest(json.loads(args.manifest.read_text()), args.dataset)
    else:
        value = build_manifest(args.dataset)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("x") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        data = panels(args.dataset)
    print(
        json.dumps(
            {
                "status": "offline_preflight_passed",
                "counts": {k: len(v) for k, v in data.items()},
                "live_calls": 0,
                "manifest": str(args.manifest),
            }
        )
    )


if __name__ == "__main__":
    main()
