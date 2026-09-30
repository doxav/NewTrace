"""Frozen selection and paired analysis, separate from EXP20's production learning path."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import random
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import campaign as C, prepare, task

ROOT = prepare.ROOT / "experiments/recursive_opt/_shared/o1_learning/exp20/full"
SEEDS = [20011, 20023, 20037, 20041, 20053, 20071]
ARMS = ["standard", "curriculum"]


def select_prefixes(candidates: list[dict[str, Any]], calls: int) -> list[str]:
    """Select by validation only; earliest eligible source wins exact ties, seed first."""
    if not any(r["first_slot"] == 0 and r["valid"] for r in candidates):
        raise ValueError("valid seed candidate required")
    return [
        max(
            (r for r in candidates if r["valid"] and r["first_slot"] <= prefix),
            key=lambda r: (r["accuracy"], -r["first_slot"], r["hash"]),
        )["hash"]
        for prefix in range(calls + 1)
    ]


def freeze_selections(
    root: Path, seeds: list[int], selections: dict[str, Any], calls: int
) -> dict[str, Any]:
    """Open holdout only after every registered arm/seed/prefix has a frozen artifact."""
    if set(selections) != {str(s) for s in seeds} or any(
        set(selections[str(s)]) != set(ARMS) for s in seeds
    ):
        raise ValueError("complete seed/arm selections required")
    for seed in seeds:
        for arm in ARMS:
            item = selections[str(seed)][arm]
            if len(item["prefixes"]) != calls + 1 or any(
                h not in item["candidates"] for h in item["prefixes"]
            ):
                raise ValueError("complete prefix source mapping required")
            if any(
                task.digest(candidate["artifact"]) != key
                for key, candidate in item["candidates"].items()
            ):
                raise ValueError("selected source hash mismatch")
    representative_seed = max(
        seeds,
        key=lambda s: (
            next(
                r["accuracy"]
                for r in selections[str(s)]["curriculum"]["evaluations"]
                if r["hash"] == selections[str(s)]["curriculum"]["prefixes"][-1]
            ),
            -seeds.index(s),
        ),
    )
    frozen = {
        "seeds": seeds,
        "calls": calls,
        "selections": selections,
        "representative_curriculum": {
            "seed": representative_seed,
            "hash": selections[str(representative_seed)]["curriculum"]["prefixes"][-1],
        },
    }
    path = root / "selections_frozen.json"
    C.retain(path, frozen)
    if not (root / "selection_freeze_time.json").exists():
        C.retain(
            root / "selection_freeze_time.json",
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "selection_hash": task.digest(frozen),
            },
        )
    return frozen


def paired(a: list[float], b: list[float]) -> dict[str, Any]:
    """Describe paired outer-seed deltas with a fixed 10,000-draw percentile bootstrap."""
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("at least two complete paired outer seeds required")
    deltas = [x - y for x, y in zip(a, b)]
    rng = random.Random(20099)
    draws = sorted(
        statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(10000)
    )
    low, high = draws[249], draws[9749]
    return {
        "deltas": deltas,
        "mean_delta": statistics.mean(deltas),
        "median_delta": statistics.median(deltas),
        "bootstrap_95_percentile": [low, high],
        "interpretation": (
            "positive signal"
            if low > 0
            else "negative signal" if high < 0 else "inconclusive at this sample size"
        ),
        "caveat": "exploratory paired outer-seed bootstrap; n=6 fragile; no claim of significance",
    }


def profiles() -> dict[str, Any]:
    """Resolve exactly the same role profiles as production, without constructing a client."""
    task.register()
    return task.S._thaw(
        task.S.normalize_spec(
            task.specification([{"placeholder": True}], seed=0, curriculum=False)
        )["llm_profiles"]
    )


def evaluate_panel(
    root: Path,
    seed: int,
    artifact: dict[str, Any],
    panel: list[dict[str, Any]],
    split: str,
    *,
    deployment: bool = False,
) -> dict[str, Any]:
    """Evaluate every question with immutable per-row results and shared reader responses."""
    key = task.digest(artifact)
    if split == "holdout":
        frozen_path = root / "selections_frozen.json"
        if not frozen_path.exists():
            raise RuntimeError("holdout is locked until all selections are frozen")
        frozen = json.loads(frozen_path.read_text())
        if seed not in frozen["seeds"]:
            raise ValueError("unregistered holdout seed")
        allowed = {task.digest(task.INITIAL)}
        for arm in ARMS:
            allowed.update(frozen["selections"][str(seed)][arm]["prefixes"])
        if key not in allowed:
            raise ValueError("unselected source cannot access holdout")
    folder = root / "evaluation" / str(seed) / split / key
    profile = profiles()["reader"]
    records = []
    for row in panel:
        path = folder / f"{row['id']}.json"
        identity = {
            "artifact_hash": key,
            "row_hash": task.digest(row),
            "seed": seed,
            "split": split,
            "deployment": deployment,
        }
        if path.exists():
            value = json.loads(path.read_text())
            if value["identity"] != identity:
                raise ValueError("evaluation identity mismatch")
            records.append(value)
            continue
        journal = C.Journal(
            folder / "reader_events" / row["id"],
            root / "reader_cache" / str(seed),
            seed,
            0,
        )
        journal.phase = split
        raw_client = journal.client(profile, "forward")

        class Reader:
            deterministic_trace = True

            def __call__(self, **kwargs: Any) -> Any:
                """Match the guarded forward-role request exactly for shared caching."""
                return raw_client(
                    **{
                        **kwargs,
                        **profile["request_params"],
                        "temperature": profile["temperature"],
                        "max_tokens": profile["max_tokens"],
                    }
                )

        def assess(policy: dict[str, Any]) -> dict[str, Any]:
            """Return typed policy failure; let provider and infrastructure errors stop the run."""
            try:
                output = task.QAPolicy(policy, Reader())(row)
                evaluation = task.evaluate(output, row, {"phase": "final_evaluation"})
                return {
                    "valid": True,
                    "metrics": dict(evaluation.metrics),
                    "answer": output.data["answer"],
                    "error": None,
                }
            except (ValueError, TypeError, C.ExecutionError) as error:
                if journal.failure:
                    raise RuntimeError(
                        "reader failure, not candidate invalidity"
                    ) from error
                return {
                    "valid": False,
                    "metrics": {},
                    "answer": None,
                    "error": type(error).__name__,
                }

        value = assess(artifact)
        fallback = deployment and not value["valid"]
        candidate_valid = value["valid"]
        if fallback:
            value = assess(task.INITIAL)
            if not value["valid"]:
                raise RuntimeError("trusted deployment seed failed")
        record = {
            "identity": identity,
            **value,
            "candidate_valid": candidate_valid,
            "fallback": fallback,
        }
        C.retain(path, record)
        records.append(record)
    if len(records) != len(panel):
        raise ValueError("incomplete question coverage")
    valid = all(r["valid"] for r in records)
    return {
        "valid": valid,
        "accuracy": (
            statistics.mean(r["metrics"]["accuracy"] for r in records)
            if valid
            else None
        ),
        "F1": (
            statistics.mean(r["metrics"]["answer_f1"] for r in records)
            if valid
            else None
        ),
        "questions": len(records),
        "candidate_invalid": sum(not r["candidate_valid"] for r in records),
        "fallback_count": sum(r["fallback"] for r in records),
    }


def candidate_menu(root: Path, seed: int, arm: str) -> dict[str, Any]:
    """Keep every generated artifact, with first response index and seed at zero."""
    folder = root / "chains" / str(seed) / arm
    run = json.loads((folder / "result.json").read_text())
    first = {task.digest(task.INITIAL): 0}
    for batch in run["batch_events"]:
        for key in batch["candidate_hashes"]:
            first.setdefault(key, batch["slot_after"])
    return {
        key: {
            "artifact": json.loads((folder / "artifacts" / f"{key}.json").read_text()),
            "first_slot": slot,
        }
        for key, slot in first.items()
    }


def selection_job(args: tuple[str, int, str, str, int]) -> tuple[int, dict[str, Any]]:
    """Evaluate all TRAIN and VALIDATION questions after both learning arms finished."""
    root_s, seed, train_name, validation_name, calls = args
    root = Path(root_s)
    from experiments.recursive_opt._shared.optimizer_discovery.phase0 import _load_key

    _load_key()
    data = prepare.panels()
    selections = {}
    for arm in (ARMS if seed % 4 == 3 else list(reversed(ARMS))):
        menu = candidate_menu(root, seed, arm)
        rows = []
        for key, item in menu.items():
            train = evaluate_panel(
                root, seed, item["artifact"], data[train_name], "train_complete"
            )
            validation = evaluate_panel(
                root, seed, item["artifact"], data[validation_name], "validation"
            )
            rows.append(
                {
                    "hash": key,
                    "first_slot": item["first_slot"],
                    "valid": train["valid"] and validation["valid"],
                    "accuracy": validation["accuracy"],
                    "train": train,
                    "validation": validation,
                }
            )
        selections[arm] = {
            "candidates": menu,
            "evaluations": rows,
            "prefixes": select_prefixes(rows, calls),
        }
    C.retain(root / "selection" / f"{seed}.json", selections)
    return seed, selections


def holdout_job(args: tuple[str, int]) -> tuple[int, dict[str, Any]]:
    """Read only globally frozen selections and score every registered TEST question."""
    root_s, seed = args
    root = Path(root_s)
    from experiments.recursive_opt._shared.optimizer_discovery.phase0 import _load_key

    _load_key()
    frozen = json.loads((root / "selections_frozen.json").read_text())
    if seed not in frozen["seeds"]:
        raise ValueError("unregistered holdout seed")
    panel = prepare.panels()["holdout"]
    initial = evaluate_panel(
        root, seed, task.INITIAL, panel, "holdout", deployment=True
    )
    output = {
        "unchanged": {
            "curve": [initial["accuracy"]] * (frozen["calls"] + 1),
            "evaluations": {task.digest(task.INITIAL): initial},
        }
    }
    for arm in ARMS:
        selected = frozen["selections"][str(seed)][arm]
        evaluated = {
            key: evaluate_panel(
                root,
                seed,
                selected["candidates"][key]["artifact"],
                panel,
                "holdout",
                deployment=True,
            )
            for key in dict.fromkeys(selected["prefixes"])
        }
        output[arm] = {
            "curve": [evaluated[key]["accuracy"] for key in selected["prefixes"]],
            "evaluations": evaluated,
        }
    C.retain(root / "holdout" / f"{seed}.json", output)
    return seed, output


def fit_job(args: tuple[str, int, int, str, list[str]]) -> int:
    """Run two whole learning chains in a fresh process; no validation or TEST input."""
    root_s, seed, calls, train_name, arms = args
    from experiments.recursive_opt._shared.optimizer_discovery.phase0 import _load_key

    _load_key()
    train = prepare.panels()[train_name]
    for arm in arms:
        C.fit(Path(root_s), train, seed=seed, arm=arm, calls=calls)
    return seed


def source_hashes() -> dict[str, str]:
    """Include the adapters and analysis in addition to frozen production sources."""
    names = prepare.SOURCE_PATHS + [
        "experiments/recursive_opt/o1_qa/campaign.py",
        "experiments/recursive_opt/o1_qa/campaign_analysis.py",
    ]
    return {
        n: hashlib.sha256((prepare.ROOT / n).read_bytes()).hexdigest() for n in names
    }


def manifest(pilot_mode: bool) -> dict[str, Any]:
    """Freeze the full schedule before calls; failed readiness remains visible."""
    original = json.loads(prepare.MANIFEST.read_text())
    calls = 2 if pilot_mode else 6
    seeds = [20003] if pilot_mode else SEEDS
    return {
        "experiment": "EXP20-FIT-PILOT" if pilot_mode else "EXP20-FULL-v1",
        "readiness_amendment": "Explicit user instruction to execute full coverage despite failed readiness; not a claim that original gates passed.",
        "source_sha256": source_hashes(),
        "splits": original["splits"],
        "dataset": original["dataset"],
        "initial_artifact": task.INITIAL,
        "llm_profiles": profiles(),
        "outer_seeds": seeds,
        "optimizer_responses_per_arm": calls,
        "arms": ["unchanged", *ARMS],
        "train_split": "pilot_train" if pilot_mode else "train",
        "validation_split": "pilot_selection" if pilot_mode else "validation",
        "batch_size": 6,
        "curriculum_history_size": 2,
        "workers": 6,
        "arm_order": {
            str(seed): ARMS if i % 2 == 0 else list(reversed(ARMS))
            for i, seed in enumerate(seeds)
        },
        "evaluation_schedule": "Production (3*N+1)*6 + train_size per chain, including same discarded parent replay in control. After fitting, all unique seed/proposed sources evaluated on every TRAIN and VALIDATION question. Missing/invalid proposals retain unused allocations. Every prefix deployment evaluated on all 48 TEST questions, shared response cache avoids repeated paid calls.",
        "primary": "Per outer seed, mean TEST exact match of validation-selected prefix policies 0..6; higher better.",
        "selection": "Eligible if all complete TRAIN and VALIDATION questions valid. Maximum VALIDATION EM, earliest proposal tie; seed=0.",
        "fallback": "On selected policy execution failure on TEST, use unchanged seed for that question; report candidate-only invalidity and fallback separately. Malformed reader final line gets EM=0; no repair.",
        "uncertainty": "Paired outer-seed bootstrap: random.Random(20099), 10000 replacement samples, sorted indices249/9749. Interval strictly positive=positive signal; strictly negative=negative signal; otherwise inconclusive. n6 exploratory.",
        "target": 0.70,
        "nonattainment": "None/censored at 6 responses",
        "transport": "Existing attempts/profile, at most 4 waves only for explicit HTTP429 rejection; delays4/8/16. Unknown completion blocks. Every completed empty response consumes slot, canonical semantic retry also consumes slot. No replacement.",
        "cache": "SHA256 exact forward request + normalized profile + outer seed, shared between arms, file lock. Timing excluded from optimization trace; real timing preserved in raw responses. Optimizer XML node blocks sorted; ephemeral object addresses scrubbed; internal trace object IDs replaced by stable node labels before requests. All substantive values retained.",
        "representative": "Final selected curriculum candidate with highest validation EM across seeds; earliest registered seed breaks ties. Frozen before TEST.",
        "holdout_gate": "Every arm/seed/prefix selection frozen before first TEST evaluation.",
        "coverage": "All 60 TRAIN questions evaluated for candidate eligibility; report unique questions actually used in learning feedback separately; target>=20.",
    }


def accounting(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Audit raw request identities/settings and account for every response and retry."""
    result = {}
    for role, pattern in (
        ("reader", "reader_cache/**/request.json"),
        ("optimizer", "chains/*/*/optimizer/**/request.json"),
    ):
        profile = config["llm_profiles"][role]
        expected = {
            "temperature": profile["temperature"],
            "max_tokens": profile["max_tokens"],
            **profile["request_params"],
        }
        if role == "optimizer":
            expected["response_format"] = None  # Canonical OptoPrime default.
        events: Counter[str] = Counter()
        usage: Counter[str] = Counter()
        providers: Counter[str] = Counter()
        finishes: Counter[str] = Counter()
        completed = empty = attempts = failed = pending = cost_receipts = 0
        for path in sorted(root.glob(pattern)):
            request = json.loads(path.read_text())
            if request["request_hash"] != task.digest(request["request"]):
                raise ValueError("raw request hash mismatch")
            settings = {k: v for k, v in request["request"].items() if k != "messages"}
            if settings != expected:
                raise ValueError("raw request settings drift")
            local_events = Counter(
                json.loads(p.read_text())["event"]
                for p in path.parent.glob("transport_*.json")
            )
            events.update(local_events)
            response_path = path.with_name("response.json")
            done = response_path.exists()
            attempts += max(1, local_events["transient_failure"] + int(done))
            if not done:
                rejected = path.with_name("failure.json").exists()
                failed += rejected
                pending += not rejected
                continue
            receipt = json.loads(response_path.read_text())
            if receipt["request_hash"] != request["request_hash"]:
                raise ValueError("raw response identity mismatch")
            raw = receipt["response"]
            if raw["model"] != profile["model"]:
                raise ValueError("raw response model differs from manifest")
            completed += 1
            content = raw["choices"][0]["message"].get("content")
            empty += not bool(str(content or "").strip())
            providers[str(raw.get("provider", "unreported"))] += 1
            finishes[str(raw["choices"][0].get("finish_reason"))] += 1
            available = {
                k: v for k, v in receipt["usage"].items() if isinstance(v, (int, float))
            }
            usage.update(available)
            cost_receipts += "cost_usd" in available
        result[role] = {
            "completed_responses": completed,
            "empty_responses": empty,
            "transport_attempts": attempts,
            "transport_events": dict(events),
            "failed_request_waves": failed,
            "unreconciled_requests": pending,
            "usage": dict(usage),
            "reported_cost_usd": usage.get("cost_usd", 0.0),
            "responses_with_cost_receipt": cost_receipts,
            "failed_attempt_billing": "unknown; not imputed as free",
            "providers": dict(providers),
            "finish_reasons": dict(finishes),
        }
    return result


def analyze(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Require every outer seed and preserve unfavorable outcomes in the paired analysis."""
    seeds = config["outer_seeds"]
    records = {
        s: json.loads((root / "holdout" / f"{s}.json").read_text()) for s in seeds
    }
    arms = {}
    for arm in ["unchanged", *ARMS]:
        values = [statistics.mean(records[s][arm]["curve"]) for s in seeds]
        final = [records[s][arm]["curve"][-1] for s in seeds]
        arms[arm] = {
            "primary_per_seed": dict(zip(map(str, seeds), values)),
            "primary_mean": statistics.mean(values),
            "primary_median": statistics.median(values),
            "final_per_seed": final,
            "final_mean": statistics.mean(final),
            "curves": {str(s): records[s][arm]["curve"] for s in seeds},
            "first_target_prefix": {
                str(s): next(
                    (
                        i
                        for i, v in enumerate(records[s][arm]["curve"])
                        if v >= config["target"]
                    ),
                    None,
                )
                for s in seeds
            },
        }

    def primary(arm: str) -> list[float]:
        """Extract values in the registered outer-seed order."""
        return list(arms[arm]["primary_per_seed"].values())

    result = {
        "experiment": config["experiment"],
        "accounting": accounting(root, config),
        "arms": arms,
        "curriculum_minus_standard": paired(primary("curriculum"), primary("standard")),
        "standard_minus_unchanged": paired(primary("standard"), primary("unchanged")),
        "curriculum_minus_unchanged": paired(
            primary("curriculum"), primary("unchanged")
        ),
    }
    C.retain(root / "results.json", result)
    return result


def main() -> None:
    """Stage-aware campaign CLI; immutable freeze precedes the first live request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["freeze", "fit", "select", "holdout", "analyze"]
    )
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    root = args.root or ROOT / ("fit_pilot" if args.pilot else "confirmation")
    config = manifest(args.pilot)
    if args.stage == "freeze":
        C.retain(root / "manifest.json", config)
        C.retain(
            root / "freeze_time.json", {"utc": datetime.now(timezone.utc).isoformat()}
        )
        print(
            json.dumps(
                {
                    "status": "FROZEN",
                    "experiment": config["experiment"],
                    "manifest_hash": task.digest(config),
                }
            )
        )
        return
    if json.loads((root / "manifest.json").read_text()) != config:
        raise ValueError("frozen manifest/source drift")
    seeds, calls = config["outer_seeds"], config["optimizer_responses_per_arm"]
    with ProcessPoolExecutor(
        max_workers=min(6, len(seeds)), mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        if args.stage == "fit":
            list(
                pool.map(
                    fit_job,
                    [
                        (
                            str(root),
                            s,
                            calls,
                            config["train_split"],
                            config["arm_order"][str(s)],
                        )
                        for s in seeds
                    ],
                )
            )
        elif args.stage == "select":
            for s in seeds:
                for arm in ARMS:
                    run = json.loads(
                        (root / "chains" / str(s) / arm / "result.json").read_text()
                    )
                    if run["optimizer_responses"] != calls:
                        raise ValueError("incomplete proposal slots")
            chosen = dict(
                pool.map(
                    selection_job,
                    [
                        (
                            str(root),
                            s,
                            config["train_split"],
                            config["validation_split"],
                            calls,
                        )
                        for s in seeds
                    ],
                )
            )
            freeze_selections(
                root, seeds, {str(s): v for s, v in chosen.items()}, calls
            )
        elif args.stage == "holdout":
            if args.pilot:
                raise ValueError("pilot never reads confirmatory holdout")
            list(pool.map(holdout_job, [(str(root), s) for s in seeds]))
        elif args.stage == "analyze":
            print(json.dumps(analyze(root, config), indent=2))
    print(
        json.dumps(
            {
                "stage": args.stage,
                "status": "COMPLETE",
                "experiment": config["experiment"],
            }
        )
    )


if __name__ == "__main__":
    main()
