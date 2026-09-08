"""Descriptive EXP-15 reporting repairs; never used by generation or selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from artifacts.optimizer_discovery.exp15 import read


def attempt_timing(directory: Path) -> dict[str, Any]:
    """Associate response latency with its actual attempt and separate total slot time."""
    response = read(directory / "response.json")
    attempt_id = response["attempt"]
    matching_start = read(directory / f"started_{attempt_id}.json")["time_ns"]
    starts = [read(p)["time_ns"] for p in directory.glob("started_*.json")]
    attempts = [read(p) for p in directory.glob("attempt_*.json")]
    if (
        not starts
        or not attempts
        or sum(a["status"] == "completed" for a in attempts) != 1
    ):
        raise ValueError(
            "timing requires exactly one completed response and its attempt receipts"
        )
    slot_wall = (response["completed_ns"] - min(starts)) / 1e9
    successful_wall = (response["completed_ns"] - matching_start) / 1e9
    measured = sum(a["wall_s"] for a in attempts)
    if successful_wall < 0 or slot_wall < 0:
        raise ValueError("response precedes its registered attempt")
    return {
        "completed_attempt": attempt_id,
        "transport_attempts": len(attempts),
        "transport_failures": sum(a["status"] == "transport_failure" for a in attempts),
        "slot_wall_s": slot_wall,
        "successful_attempt_wall_s": successful_wall,
        "successful_attempt_monotonic_s": response["wall_s"],
        "measured_attempts_s": measured,
        "unattributed_wall_gap_s": slot_wall - measured,
        "gap_interpretation": "May include system suspension, backoff, scheduling and resume gaps; do not attribute it solely to provider latency.",
    }


def export_programs(root: Path) -> dict[str, Any]:
    """Export exact selected A2 bytes and attempted lineage after the global selection freeze."""
    import difflib
    import gzip
    import hashlib

    from artifacts.optimizer_discovery import benchmark as B
    from artifacts.optimizer_discovery.exp15 import persist

    frozen = read(root / "selections_frozen.json")
    selections = {}
    for outer, expected in frozen["selection_hashes"].items():
        selection = read(root / str(int(outer)) / "selection.json")
        if B.digest(selection) != expected:
            raise RuntimeError("cannot export a modified frozen selection")
        selected = selection["A2"]
        if B.source_hash(selected["source"]) != selected["source_sha256"]:
            raise RuntimeError("selected source hash mismatch")
        selections[outer] = selected
    representative = str(frozen["representative_outer_seed"])
    expected_representative = min(
        selections,
        key=lambda s: (selections[s]["validation_auc"], list(selections).index(s)),
    )
    if (
        representative != expected_representative
        or selections[representative] != frozen["representative"]
    ):
        raise RuntimeError("representative differs from the frozen validation rule")
    programs = {}
    destination = root.parent / "selected"
    destination.mkdir(parents=True, exist_ok=True)
    for outer, selected in selections.items():
        payload = gzip.compress(selected["source"].encode(), mtime=0)
        target = destination / f"A2_seed_{int(outer)}.py.gz"
        if target.exists():
            if target.read_bytes() != payload:
                raise RuntimeError("refusing to overwrite an exported source")
        else:
            temporary = target.with_suffix(".gz.pending")
            temporary.write_bytes(payload)
            temporary.replace(target)
        pool = read(root / outer / "A2/pool.json")
        sources = {B.MANIFEST["seed_sha256"]: B.SEED_SOURCE}
        lineage = []
        for path in sorted((root / outer / "A2").glob("slot_*/response.json")):
            response = read(path)
            request = read(path.parent / "request.json")
            parent_hash = request["parent_sha256"]
            if (
                parent_hash not in sources
                or B.source_hash(response["source"]) != response["source_sha256"]
            ):
                raise RuntimeError("lineage source integrity failed")
            candidate = next(c for c in pool if c["index"] == request["slot"])
            lineage.append(
                {
                    "slot": request["slot"],
                    "parent_sha256": parent_hash,
                    "source_sha256": response["source_sha256"],
                    "eligible": candidate["eligible"],
                    "selected": selected["index"] == request["slot"],
                    "train_auc": (
                        B.aggregate(candidate["train"], "auc")
                        if all(r["valid"] for r in candidate["train"])
                        else None
                    ),
                    "validation_auc": candidate["validation_auc"],
                    "diff": "".join(
                        difflib.unified_diff(
                            sources[parent_hash].splitlines(keepends=True),
                            response["source"].splitlines(keepends=True),
                            fromfile=parent_hash,
                            tofile=response["source_sha256"],
                            n=2,
                        )
                    ),
                }
            )
            sources[response["source_sha256"]] = response["source"]
        lineage_path = destination / f"A2_seed_{int(outer)}_lineage.json"
        persist(lineage_path, lineage)
        programs[outer] = {
            "source_sha256": selected["source_sha256"],
            "selected_index": selected["index"],
            "validation_auc": selected["validation_auc"],
            "gzip_path": str(target),
            "gzip_sha256": hashlib.sha256(payload).hexdigest(),
            "lineage_path": str(
                lineage_path
                if lineage_path.exists()
                else lineage_path.with_suffix(".json.gz")
            ),
        }
    result = {
        "representative_outer_seed": int(representative),
        "programs": programs,
        "export_semantics": "gzip mtime=0; decompress to exact evaluated optimizer.py bytes; no formatting or source edits",
    }
    persist(destination / "index.json", result)
    return result
