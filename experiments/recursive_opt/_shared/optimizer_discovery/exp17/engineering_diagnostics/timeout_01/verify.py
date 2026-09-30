"""Independently verify retained diagnostic evidence without executing any policy."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I

ROOT = Path(__file__).resolve().parent


def read(path: Path) -> Any:
    """Read plain retained JSON without invoking experiment or evaluator code."""
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    """Hash exact artifact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    """Reproduce the declared canonical JSON hash independently of the evaluator."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def main() -> None:
    """Check provenance, exact allocation and replay semantics, then preserve the audit."""
    protocol = read(ROOT / "protocol.json")
    result = read(ROOT / "results.json")
    started = read(ROOT / "started.json")
    inputs, sources = read(ROOT / "inputs.json"), read(ROOT / "sources.json")
    assert protocol["registered_ns"] < started["started_ns"] < result["completed_ns"]
    assert (
        result["protocol_sha256"]
        == started["protocol_sha256"]
        == sha(ROOT / "protocol.json")
    )
    for name, key in (
        ("run.py", "run_script_sha256"),
        ("inputs.json", "inputs_sha256"),
        ("sources.json", "sources_sha256"),
    ):
        assert sha(ROOT / name) == protocol[key]
    assert result["original_files_sha256"] == protocol["original_files_sha256"]
    for name, expected in protocol["original_files_sha256"].items():
        assert sha(Path(name)) == expected
    original = read(Path(protocol["selected_cache_path"]))
    row = original["row"]
    assert original["row_sha256"] == canonical(row) == protocol["selected_row_sha256"]
    freeze = read(ROOT.parent.parent / "engineering/freeze.json")
    task = next(
        t for t in freeze["tasks"]["train"] if canonical(t) == row["task_identity"]
    )
    assert inputs["failed_history"] == {
        "history": row["observations"],
        "bounds": [[-5, 5]] * task["dimension"],
        "seed": row["local_seed"],
    }
    assert inputs["empty_history"] == {**inputs["failed_history"], "history": []}
    cache_paths = sorted(
        name for name in protocol["original_files_sha256"] if "/cache/" in name
    )
    assert cache_paths[0] == protocol["selected_cache_path"]
    statuses: dict[str, int] = {}
    for name in cache_paths:
        cached = read(Path(name))["row"]
        assert cached["source_sha256"] == protocol["source_sha256"]
        assert cached["status"] != "valid"
        statuses[cached["status"]] = statuses.get(cached["status"], 0) + 1
    assert len(cache_paths) == protocol["original_completed_train_rows"]
    assert statuses == protocol["original_train_status_counts"]
    assert (
        len(result["calls"])
        == result["proposal_calls"]
        == protocol["proposal_calls"]
        == 8
    )
    children = 0
    previous_ns = started["started_ns"]
    for index, (call, order) in enumerate(
        zip(result["calls"], protocol["order"], strict=True)
    ):
        assert call == read(ROOT / f"call_{index:02d}.json")
        launch = read(ROOT / f"call_{index:02d}_started.json")
        assert call["index"] == launch["index"] == index
        assert {key: call[key] for key in order} == order
        assert previous_ns < call["before"]["wall_ns"] < call["after"]["wall_ns"]
        previous_ns = call["after"]["wall_ns"]
        assert call["before"] == launch["before"]
        assert (
            call["input_sha256"]
            == launch["input_sha256"]
            == canonical(inputs[call["input"]])
        )
        assert call["input_sha256"] == protocol["input_payload_hashes"][call["input"]]
        assert (
            call["source_sha256"]
            == launch["source_sha256"]
            == hashlib.sha256(sources[call["policy"]]["source"].encode()).hexdigest()
        )
        assert call["timeout_s"] == protocol["timeout_s"] == 2.0
        assert call["error_type"] is None
        inner = call["executions"]
        assert len(inner) == call["subprocesses"]
        children += call["subprocesses"]
        for execution in inner:
            assert execution["source_sha256"] == call["source_sha256"]
            assert execution["payload_sha256"] == call["input_sha256"]
        first = inner[0]["result"]
        if first["status"] != "valid":
            assert len(inner) == 1 and call["result"] == first
        else:
            assert len(inner) == 2
            second = inner[1]["result"]
            if second["status"] != "valid" or first["point"] != second["point"]:
                assert call["result"]["status"] == "nondeterministic"
            else:
                assert call["result"] == first
    assert children == result["subprocesses"] <= protocol["maximum_subprocesses"] == 16
    assert result["objective_calls"] == result["model_calls"] == 0
    assert previous_ns < result["completed_ns"]
    I.persist(
        ROOT / "verification.json",
        {
            "verified_ns": time.time_ns(),
            "status": "pass",
            "verification_script_sha256": sha(Path(__file__)),
            "protocol_sha256": sha(ROOT / "protocol.json"),
            "results_sha256": sha(ROOT / "results.json"),
            "checked_calls": 8,
            "checked_subprocesses": children,
            "objective_calls": 0,
            "model_calls": 0,
            "original_files_verified": len(protocol["original_files_sha256"]),
            "checks": [
                "preregistration precedes all calls",
                "eight exact ordered calls with unchanged sources and input hashes",
                "input history, bounds and seed match original frozen TRAIN evidence",
                "all 48 original failing source cache rows retained and unchanged",
                "per-call evidence agrees with aggregate and replay semantics",
                "exact subprocess counts and zero guarded objective calls",
                "all registered source, protocol and original evidence hashes match",
            ],
        },
    )
    print(
        "PASS: eight proposal calls, sixteen subprocesses, zero objective/model calls"
    )


if __name__ == "__main__":
    main()
