"""Read-only scientific audit of EXP21 confirmation; never constructs a live client."""

from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from . import axes as X, axes_analysis as Y, campaign as C, campaign_analysis as A, task


def development_inventory(root: Path) -> dict[str, Any]:
    """List every allocated slot and missing chain without selecting from partial evidence."""
    chains = []
    pending = []
    for variant in X.variants():
        for seed in X.DEV_SEEDS:
            folder = root / "chains" / str(seed) / variant["id"]
            slots = []
            for slot in range(1, X.CALLS + 1):
                directory = folder / "optimizer" / f"{slot:02d}"
                responses = list(directory.glob("**/response.json"))
                requests = list(directory.glob("**/request.json"))
                if len(responses) > 1:
                    raise ValueError(
                        "multiple completed responses in one proposal slot"
                    )
                slots.append(
                    {
                        "slot": slot,
                        "status": (
                            "complete"
                            if responses
                            else "pending" if requests else "unissued"
                        ),
                        "response": (
                            str(responses[0].relative_to(root)) if responses else None
                        ),
                    }
                )
            result = folder / "result.json"
            report = root / "reports" / str(seed) / f"{variant['id']}.json"
            chains.append(
                {
                    "name": variant["id"],
                    "axis": variant["axis"],
                    "seed": seed,
                    "slots": slots,
                    "fit_complete": result.exists(),
                    "measurement": X.read(report) if report.exists() else None,
                }
            )
    for path in root.rglob("request.json"):
        if path.with_name("response.json").exists():
            continue
        failure = path.with_name("failure.json")
        statuses = (
            X.read(failure).get("http_status_codes", []) if failure.exists() else []
        )
        request_hash = X.read(path)["request_hash"]
        later = [
            str(p.relative_to(root))
            for p in path.parent.parent.glob("*/response.json")
            if X.read(p)["request_hash"] == request_hash
        ]
        pending.append(
            {
                "path": str(path.relative_to(root)),
                "http_status_codes": statuses,
                "status": (
                    "recorded_http_rejection"
                    if statuses
                    else "remote_completion_unknown"
                ),
                "automatic_reissue_permitted": False,
                "later_recorded_response": later,
            }
        )
    slots = [s for chain in chains for s in chain["slots"]]
    complete = all(c["measurement"] is not None for c in chains)
    return {
        "complete_screen": complete,
        "winner_selection_permitted": complete,
        "allocated_slots": len(slots),
        "completed_slots": sum(s["status"] == "complete" for s in slots),
        "unissued_slots": sum(s["status"] == "unissued" for s in slots),
        "pending_slots": sum(s["status"] == "pending" for s in slots),
        "ambiguous_requests": sum(
            p["status"] == "remote_completion_unknown" for p in pending
        ),
        "locally_uncompleted_requests": sum(
            not p["later_recorded_response"] for p in pending
        ),
        "pending_requests": pending,
        "complete_fits": sum(c["fit_complete"] for c in chains),
        "complete_measurements": sum(c["measurement"] is not None for c in chains),
        "chains": chains,
    }


def check_answer_record(root: Path, path: Path, row: dict[str, Any]) -> None:
    """Recompute persisted answer metrics from the exact provider receipt, never an LLM call."""
    record = X.read(path)
    if not record["valid"]:
        assert record["answer"] is None and record["metrics"] == {}
        return
    events = sorted(
        (path.parent / "reader_events" / path.stem / "reader").glob("*.json")
    )
    event = X.read(events[-1])
    seed = record["identity"]["seed"]
    receipts = list(
        (root / "reader_cache" / str(seed) / event["cache_key"]).glob("*/response.json")
    )
    assert len(receipts) == 1
    raw = X.read(receipts[0])
    decoded = task.pack_answer(
        {"content": task._response_text(raw["response"])}, {}
    ).data
    em, f1 = task.answer_metrics(decoded["answer"], row["answer"])
    assert em == record["metrics"]["accuracy"] and f1 == record["metrics"]["answer_f1"]
    assert decoded["answer"] == record["answer"]
    assert float(decoded["format_valid"]) == record["metrics"]["format_valid"]


def audit_development() -> dict[str, Any]:
    """Validate partial evidence and retain all missing comparisons after provider interruption."""
    X.verify()
    root = X.ROOT / "development"
    value = development_inventory(root)
    panels = {
        split: {r["id"]: r for r in X.panels()[panel]}
        for split, panel in [
            ("train_complete", "dev_train"),
            ("validation", "dev_selection"),
            ("dev_probe", "dev_probe"),
        ]
    }
    hashes = set()
    for path in root.glob("chains/*/*/artifacts/*.json"):
        assert path.stem == task.digest(X.read(path))
        hashes.add(path.stem)
    counts = {split: 0 for split in panels}
    invalid = fallback = 0
    for path in root.glob("evaluation/*/*/*/*.json"):
        record = X.read(path)
        seed, split, key = path.parts[-4:-1]
        row = panels[split][path.stem]
        assert key in hashes
        assert record["identity"] == {
            "artifact_hash": key,
            "row_hash": task.digest(row),
            "seed": int(seed),
            "split": split,
            "deployment": split == "dev_probe",
        }
        check_answer_record(root, path, row)
        counts[split] += 1
        invalid += not record["candidate_valid"]
        fallback += record["fallback"]
    for chain in value["chains"]:
        if chain["measurement"] is None:
            continue
        seed, name = chain["seed"], chain["name"]
        selected = X.read(root / "selection" / str(seed) / f"{name}.json")
        assert (
            A.select_prefixes(selected["evaluations"], X.CALLS) == selected["prefixes"]
        )
        curve = [
            statistics.mean(
                X.read(
                    root / "evaluation" / str(seed) / "dev_probe" / key / f"{r}.json"
                )["metrics"]["accuracy"]
                for r in panels["dev_probe"]
            )
            for key in selected["prefixes"]
        ]
        assert curve == chain["measurement"]["curve"]
        assert statistics.mean(curve) == chain["measurement"]["primary"]
    value["audit"] = {
        "status": "PASS_FOR_PERSISTED_ROWS_ONLY",
        "raw_scores_recomputed": counts,
        "invalid_external_rows": invalid,
        "fallback_rows": fallback,
        "verified_source_hashes": len(hashes),
        "live_calls": 0,
        "confirmation_started": (X.ROOT / "confirmation/manifest.json").exists(),
        "requests": A.accounting(root, {"llm_profiles": A.profiles()}),
    }
    C.retain(X.ROOT / "development_audits" / f"{task.digest(value)}.json", value)
    return value


def main() -> None:
    """Recompute every saved score from raw reader text and every paired contrast."""
    X.verify()
    manifest = Y.verify_confirmation()
    root = X.ROOT / "confirmation"
    gate = X.read(root / "test_gate.json")
    seal = X.read(root / "test_gate_time.json")
    assert seal["hash"] == task.digest(gate)
    cutoff = datetime.fromisoformat(seal["utc"]).timestamp()
    names = list(manifest["configurations"])
    seeds = manifest["seeds"]
    assert gate["names"] == names and gate["seeds"] == seeds
    rows = {
        name: {r["id"]: r for r in X.panels()[panel]}
        for name, panel in [
            ("train_complete", "train"),
            ("validation", "validation"),
            ("test", "test"),
        ]
    }
    sources = {}
    slots = 0
    selection_records = 0
    for seed in seeds:
        for name in names:
            chain = root / "chains" / str(seed) / name
            identity = X.read(chain / "identity.json")
            assert identity == {
                "config": manifest["configurations"][name],
                "train_hash": task.digest(X.panels()["train"]),
                "seed": seed,
                "calls": X.CALLS,
            }
            # Retry waves may change receipt depth; the slot count remains fixed.
            responses = list((chain / "optimizer").glob("**/response.json"))
            assert len(responses) == X.CALLS
            slots += len(responses)
            assert all(p.stat().st_mtime < cutoff for p in responses)
            for path in (chain / "artifacts").glob("*.json"):
                value = X.read(path)
                assert path.stem == task.digest(value)
                sources[path.stem] = value
            selected = gate["selections"][str(seed)][name]
            scored = []
            for item in selected["evaluations"]:
                key = item["hash"]
                record_paths = [
                    root / "evaluation" / str(seed) / split / key / f"{row_id}.json"
                    for split in ("train_complete", "validation")
                    for row_id in rows[split]
                ]
                records = [X.read(p) for p in record_paths]
                assert all(p.stat().st_mtime < cutoff for p in record_paths)
                valid = all(r["valid"] for r in records)
                score = (
                    statistics.mean(
                        r["metrics"]["accuracy"]
                        for r in records[-len(rows["validation"]) :]
                    )
                    if valid
                    else None
                )
                assert valid == item["valid"]
                scored.append(
                    {
                        "hash": key,
                        "first_slot": item["first_slot"],
                        "valid": valid,
                        "accuracy": score,
                    }
                )
            assert A.select_prefixes(scored, X.CALLS) == selected["prefixes"]
            selection_records += 1
    typed_invalid = fallback = checked = 0
    split_counts = {k: 0 for k in rows}
    for path in root.glob("evaluation/*/*/*/*.json"):
        record = X.read(path)
        seed, split, key = path.parts[-4:-1]
        row = rows[split][path.stem]
        assert record["identity"] == {
            "artifact_hash": key,
            "row_hash": task.digest(row),
            "seed": int(seed),
            "split": split,
            "deployment": split == "test",
        }
        assert key in sources
        if split == "test":
            assert path.stat().st_mtime >= cutoff
            assert record["valid"]
            allowed = {
                h for v in gate["selections"][seed].values() for h in v["prefixes"]
            }
            assert key in allowed
        typed_invalid += not record["candidate_valid"]
        fallback += record["fallback"]
        checked += 1
        split_counts[split] += 1
        check_answer_record(root, path, row)
    expected = {k: 0 for k in rows}
    for seed in seeds:
        all_candidates = {
            h
            for item in gate["selections"][str(seed)].values()
            for h in item["candidates"]
        }
        selected = {
            h
            for item in gate["selections"][str(seed)].values()
            for h in item["prefixes"]
        }
        expected["train_complete"] += len(all_candidates) * len(rows["train_complete"])
        expected["validation"] += len(all_candidates) * len(rows["validation"])
        expected["test"] += len(selected) * len(rows["test"])
    assert split_counts == expected
    results = X.read(root / "results.json")
    for name in names:
        curves = {}
        for seed in seeds:
            selected = gate["selections"][str(seed)][name]["prefixes"]
            curve = [
                statistics.mean(
                    X.read(
                        root / "evaluation" / str(seed) / "test" / h / f"{row_id}.json"
                    )["metrics"]["accuracy"]
                    for row_id in rows["test"]
                )
                for h in selected
            ]
            assert (
                curve == X.read(root / "reports" / str(seed) / f"{name}.json")["curve"]
            )
            curves[str(seed)] = curve
        assert Y.summarize_curves(curves, seeds) == results["arms"][name]
    assert Y.analyze() == results
    config = {"llm_profiles": A.profiles()}
    requests = A.accounting(root, config)
    assert (
        requests["optimizer"]["completed_responses"]
        == len(seeds) * len(names) * X.CALLS
    )
    assert requests["optimizer"]["unreconciled_requests"] == 0
    assert requests["reader"]["unreconciled_requests"] == 0
    proof = {
        "status": "PASS",
        "seeds": seeds,
        "configurations": names,
        "completed_optimizer_responses": slots,
        "frozen_selections_verified": selection_records,
        "raw_scores_recomputed": checked,
        "by_split": split_counts,
        "typed_invalid_rows": typed_invalid,
        "deployment_fallback_rows": fallback,
        "source_hashes": "verified",
        "all_selections_preceded_test": True,
        "raw_request_configurations": "verified",
        "recomputed_aggregates": "identical",
        "live_calls_by_audit": 0,
        "accounting": requests,
    }
    C.retain(root / "raw_audit.json", proof)
    print(json.dumps({k: v for k, v in proof.items() if k != "accounting"}, indent=2))


if __name__ == "__main__":
    if sys.argv[1:] == ["--development"]:
        result = audit_development()
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in {"chains", "pending_requests"}
                },
                indent=2,
            )
        )
    else:
        main()
