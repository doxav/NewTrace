"""Descriptive EXP-15 reporting repairs; never used by generation or selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from artifacts.optimizer_discovery.exp15 import read


def _artifact_path(path: Path) -> str:
    """Use stable repository-relative labels, or absolute paths for external exports."""
    absolute = path.resolve()
    repository = Path(__file__).resolve().parents[2]
    return str(
        absolute.relative_to(repository)
        if absolute.is_relative_to(repository)
        else absolute
    )


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
            "gzip_path": _artifact_path(target),
            "gzip_sha256": hashlib.sha256(payload).hexdigest(),
            "lineage_path": _artifact_path(
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


def read_trace(directory: Path) -> dict[str, Any]:
    """Read a completed trace, including its lossless post-run archive with hash checks."""
    import hashlib
    import json
    import lzma

    archived = directory / "trace.json.xz"
    if not archived.exists():
        return read(directory / "trace.json")
    record = read(directory / "trace_archive.json")
    try:
        packed = archived.read_bytes()
        raw = lzma.decompress(packed)
        if (
            hashlib.sha256(packed).hexdigest() != record["xz_sha256"]
            or hashlib.sha256(raw).hexdigest() != record["json_sha256"]
        ):
            raise ValueError("hash mismatch")
        return json.loads(raw)
    except (lzma.LZMAError, ValueError) as error:
        raise RuntimeError("trace archive integrity failed") from error


def archive_trace(directory: Path) -> dict[str, Any]:
    """Archive only a completed trace, preserving exact JSON bytes and original gzip provenance."""
    import gzip
    import hashlib
    import lzma

    from artifacts.optimizer_discovery.exp15 import persist

    if not (directory / "generation_complete.json").exists():
        raise RuntimeError("trace archival requires completed generation")
    original = directory / "trace.json.gz"
    archived = directory / "trace.json.xz"
    if not original.exists():
        read_trace(directory)
        return read(directory / "trace_archive.json")
    compressed = original.read_bytes()
    raw = gzip.decompress(compressed)
    packed = lzma.compress(raw, preset=9)
    if lzma.decompress(packed) != raw:
        raise RuntimeError("trace archive roundtrip failed")
    record = {
        "format": "xz preset=9; lossless post-run archive",
        "json_bytes": len(raw),
        "json_sha256": hashlib.sha256(raw).hexdigest(),
        "original_gzip_bytes": len(compressed),
        "original_gzip_sha256": hashlib.sha256(compressed).hexdigest(),
        "xz_bytes": len(packed),
        "xz_sha256": hashlib.sha256(packed).hexdigest(),
        "execution_semantics": "Frozen runner output unchanged; archiving occurs only after generation_complete. Completed-run resume does not read trace files.",
    }
    if archived.exists() and archived.read_bytes() != packed:
        raise RuntimeError("refusing to overwrite a trace archive")
    temporary = archived.with_suffix(".xz.pending")
    temporary.write_bytes(packed)
    temporary.replace(archived)
    persist(directory / "trace_archive.json", record)
    read_trace(directory)
    original.unlink()
    return record


def _event_bytes(root: Path) -> bytes:
    """Verify the compressed and original-byte hashes of the completed event stream."""
    import gzip
    import hashlib

    try:
        record = read(root / "events_archive.json")
        packed = (root / "events.jsonl.gz").read_bytes()
        raw = gzip.decompress(packed)
        if (
            hashlib.sha256(packed).hexdigest() != record["gzip_sha256"]
            or hashlib.sha256(raw).hexdigest() != record["jsonl_sha256"]
        ):
            raise ValueError("hash mismatch")
        return raw
    except (OSError, EOFError, ValueError, KeyError) as error:
        raise RuntimeError("event archive integrity failed") from error


def archive_events(root: Path) -> dict[str, Any]:
    """Archive a completed run's exact event stream without weakening size checks."""
    import gzip
    import hashlib

    from artifacts.optimizer_discovery.exp15 import exists, persist

    if not exists(root / "results.json"):
        raise RuntimeError("event archival requires completed results")
    original = root / "events.jsonl"
    if not original.exists():
        _event_bytes(root)
        return read(root / "events_archive.json")
    raw = original.read_bytes()
    packed = gzip.compress(raw, mtime=0)
    archived = root / "events.jsonl.gz"
    if archived.exists() and archived.read_bytes() != packed:
        raise RuntimeError("event log differs from its retained archive")
    temporary = archived.with_suffix(".gz.pending")
    temporary.write_bytes(packed)
    temporary.replace(archived)
    record = {
        "format": "gzip mtime=0; exact completed JSONL stream",
        "jsonl_bytes": len(raw),
        "jsonl_sha256": hashlib.sha256(raw).hexdigest(),
        "gzip_bytes": len(packed),
        "gzip_sha256": hashlib.sha256(packed).hexdigest(),
        "semantics": "Post-run storage only; restore exact bytes temporarily for the frozen analyzer.",
    }
    persist(root / "events_archive.json", record)
    if _event_bytes(root) != raw:
        raise RuntimeError("event archive roundtrip failed")
    original.unlink()
    return record


def analyze_archived(root: Path) -> dict[str, Any]:
    """Materialize exact archived events, run the frozen analyzer and verify equality."""
    from artifacts.optimizer_discovery.exp15 import Experiment

    raw = _event_bytes(root)
    path = root / "events.jsonl"
    temporary = not path.exists()
    if not temporary and path.read_bytes() != raw:
        raise RuntimeError("event log differs from its retained archive")
    if temporary:
        with path.open("xb") as stream:
            stream.write(raw)
    try:
        result = Experiment(root, "confirmation").analyze()
        if result != read(root / "results.json"):
            raise RuntimeError("recomputed analysis differs from completed results")
        return result
    finally:
        if temporary:
            path.unlink()


def main() -> None:
    """Recompute completed archived evidence without new generation or evaluation."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    result = analyze_archived(parser.parse_args().root)
    print(
        json.dumps(
            {key: result[key] for key in ("experiment", "arms", "contrasts")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
