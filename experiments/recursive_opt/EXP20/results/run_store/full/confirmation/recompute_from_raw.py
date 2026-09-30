"""Offline EXP20 integrity audit; no provider construction or model calls."""

from __future__ import annotations

import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from experiments.recursive_opt.o1_qa import (
    campaign as C,
    campaign_analysis as A,
    prepare,
    task,
)


def read(path: Path) -> Any:
    """Read preserved evidence without changing it."""
    return json.loads(path.read_text())


def main() -> None:
    """Recompute answer scores from raw reader receipts, then all registered contrasts."""
    root = Path(__file__).resolve().parent
    config = read(root / "manifest.json")
    assert config == A.manifest(False), "manifest or frozen source drift"
    selected = read(root / "selections_frozen.json")
    sealed = read(root / "selection_freeze_time.json")
    assert sealed["selection_hash"] == task.digest(selected)
    freeze_timestamp = datetime.fromisoformat(sealed["utc"]).timestamp()
    seeds = config["outer_seeds"]
    assert selected["seeds"] == seeds
    data = prepare.panels()
    rows_by_split = {
        "train_complete": {r["id"]: r for r in data[config["train_split"]]},
        "validation": {r["id"]: r for r in data[config["validation_split"]]},
        "holdout": {r["id"]: r for r in data["holdout"]},
    }
    artifacts = {}
    for path in root.glob("chains/*/*/artifacts/*.json"):
        artifact = read(path)
        assert task.digest(artifact) == path.stem, "artifact hash mismatch"
        artifacts[path.stem] = artifact
    expected_records = 0
    for seed in seeds:
        menus = [A.candidate_menu(root, seed, arm) for arm in A.ARMS]
        expected_records += 84 * len(set().union(*menus))
        chosen = set().union(
            *(selected["selections"][str(seed)][a]["prefixes"] for a in A.ARMS)
        )
        expected_records += 48 * len(chosen)
        for arm in A.ARMS:
            folder = root / "chains" / str(seed) / arm
            assert {p.name for p in (folder / "optimizer").iterdir()} == {
                f"{i:02d}" for i in range(1, 7)
            }
            run = read(folder / "result.json")
            assert run["optimizer_responses"] == 6
            assert run["actual_train_evaluations"] <= 174
            for index in range(1, 7):
                assert (
                    len(
                        list(
                            (folder / "optimizer" / f"{index:02d}").glob(
                                "*/response.json"
                            )
                        )
                    )
                    == 1
                )
    records = list(root.glob("evaluation/*/*/*/*.json"))
    assert len(records) == expected_records, "missing/extra evaluation rows"
    invalid = fallback = 0
    for path in records:
        record = read(path)
        seed, split, artifact_hash = path.parts[-4:-1]
        row = rows_by_split[split][path.stem]
        assert record["identity"] == {
            "artifact_hash": artifact_hash,
            "row_hash": task.digest(row),
            "seed": int(seed),
            "split": split,
            "deployment": split == "holdout",
        }
        assert artifact_hash in artifacts
        invalid += not record["candidate_valid"]
        fallback += record["fallback"]
        if split == "holdout":
            assert path.stat().st_mtime >= freeze_timestamp
            assert record["valid"], "deployment cannot omit an invalid trajectory"
        if not record["valid"]:
            assert record["metrics"] == {} and record["answer"] is None
            continue
        events = sorted(
            (path.parent / "reader_events" / path.stem / "reader").glob("*.json")
        )
        event = read(events[-1])
        receipts = list(
            (root / "reader_cache" / seed / event["cache_key"]).glob("*/response.json")
        )
        assert len(receipts) == 1
        receipt = read(receipts[0])
        assert receipt["request_hash"] == event["request_hash"]
        decoded = task.pack_answer(
            {"content": task._response_text(receipt["response"])}, {}
        ).data
        assert decoded["answer"] == record["answer"]
        em, f1 = task.answer_metrics(decoded["answer"], row["answer"])
        assert (
            em == record["metrics"]["accuracy"] and f1 == record["metrics"]["answer_f1"]
        )
        assert float(decoded["format_valid"]) == record["metrics"]["format_valid"]
    curves = {arm: [] for arm in ["unchanged", *A.ARMS]}
    for seed in seeds:
        persisted = read(root / "holdout" / f"{seed}.json")
        for arm in curves:
            hashes = (
                [task.digest(task.INITIAL)] * 7
                if arm == "unchanged"
                else selected["selections"][str(seed)][arm]["prefixes"]
            )
            curve = []
            for key in hashes:
                values = [
                    read(
                        root / "evaluation" / str(seed) / "holdout" / key / f"{r}.json"
                    )["metrics"]["accuracy"]
                    for r in rows_by_split["holdout"]
                ]
                curve.append(statistics.mean(values))
            assert curve == persisted[arm]["curve"]
            curves[arm].append(statistics.mean(curve))
    results = read(root / "results.json")
    for arm, values in curves.items():
        assert (
            dict(zip(map(str, seeds), values))
            == results["arms"][arm]["primary_per_seed"]
        )
    for a, b in [
        ("curriculum", "standard"),
        ("curriculum", "unchanged"),
        ("standard", "unchanged"),
    ]:
        assert A.paired(curves[a], curves[b]) == results[f"{a}_minus_{b}"]
    usage = A.accounting(root, config)
    assert usage == results["accounting"]
    assert usage["optimizer"]["completed_responses"] == 72
    assert not any(v["unreconciled_requests"] for v in usage.values())
    # A cache alias must never create two receipts for the same request within one outer seed.
    for seed in seeds:
        identities = [
            read(p)["request_hash"]
            for p in (root / "reader_cache" / str(seed)).glob("*/*/response.json")
        ]
        assert len(identities) == len(
            set(identities)
        ), "duplicated reader cache identity"
    report = {
        "status": "PASS",
        "raw_scores_recomputed": len(records),
        "registered_seeds": seeds,
        "completed_optimizer_slots": 72,
        "typed_invalid_rows": invalid,
        "fallback_rows": fallback,
        "primary_and_paired_recomputation": "identical",
        "source_hashes": "verified",
        "holdout_after_global_freeze": True,
        "reader_cache_identity": "unique per seed/request",
        "live_calls_by_audit": 0,
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    C.retain(root / "raw_recomputation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
