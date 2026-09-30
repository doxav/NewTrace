"""Bounded live CLI for EXP-17 engineering and the registered C-versus-I study."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import time
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.exp17 import study as N
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import production_driver as OLD

ROOT = Path(__file__).resolve().parent
PILOT = ROOT / "engineering"
RUN = ROOT / "run"


@contextmanager
def run_lease(root: Path) -> Iterator[None]:
    """Refuse a second mutating process before it can race immutable run journals."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".process.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("run is already owned by another process") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def live_client() -> Any:
    """Use the tested client and serialize provider calls across successor processes."""
    client = OLD.live_client()

    def call(**kwargs: Any) -> Any:
        """Keep inter-study queueing separate from additional scientific proposals."""
        lock_path = Path("/tmp/trace-optimizer-successor-generation.lock")
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                return client(**kwargs)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    return call


def pilot_config() -> dict[str, Any]:
    """Return four registered engineering response slots on disjoint task instances."""
    return N.configuration(
        experiment="EXP-17",
        namespace="EXP17-E1-v1",
        arms=["I", "C"],
        outer_seeds=[17901],
        slots=2,
    )


def engineering_evidence(root: Path) -> dict[str, Any]:
    """Authenticate every pilot slot, production callback and harmless complete resume."""
    frozen = N.preflight(root)
    N.verify_chronology(root)
    if E.exists(root / "audit_results.json"):
        raise RuntimeError("engineering must not be represented as an efficacy audit")
    paths = N.barrier_paths(root, "generation_frozen.json") + N.barrier_paths(
        root, "selections_frozen.json"
    )
    before = {str(path.relative_to(root)): N.B.digest(E.read(path)) for path in paths}
    cache_before = sorted(
        (path.name, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in (root / "cache").glob("*.json*")
    )
    N.run_generation(root, client=None)
    N.select_all(root)
    after = {str(path.relative_to(root)): N.B.digest(E.read(path)) for path in paths}
    cache_after = sorted(
        (path.name, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in (root / "cache").glob("*.json*")
    )
    if before != after or cache_before != cache_after:
        raise RuntimeError("completed engineering resume changed evidence")
    eligible = 0
    for outer in frozen["config"]["outer_seeds"]:
        for arm in frozen["config"]["arms"]:
            directory = root / "raw" / str(outer) / arm
            pool = E.read(directory / "pool.json")
            if len(pool) != frozen["config"]["slots"] + 1 or not pool[0]["eligible"]:
                raise RuntimeError("engineering allocation or trusted seed failed")
            eligible += sum(row["eligible"] for row in pool[1:])
            if (
                arm != "I"
                and E.read(directory / "trace.json")["completed_update_callbacks"]
                != frozen["config"]["slots"]
            ):
                raise RuntimeError("engineering omitted a production callback")
    return {
        "stage": "EXP17-E1",
        "freeze_sha256": N.B.digest(frozen),
        "completed_responses": len(frozen["config"]["outer_seeds"])
        * len(frozen["config"]["arms"])
        * frozen["config"]["slots"],
        "eligible_generated": eligible,
        "resume_without_client_passed": True,
        "evidence_hashes": before,
        "cache_digest": N.B.digest(cache_before),
    }


def require_engineering() -> dict[str, Any]:
    """Reject missing, incomplete or changed pilot proof before confirmation starts."""
    path = PILOT / "engineering_results.json"
    if not E.exists(path):
        raise RuntimeError("completed engineering evidence is required")
    saved = E.read(path)
    if N.preflight(PILOT)["config"] != pilot_config():
        raise RuntimeError(
            "engineering configuration differs from its registered scope"
        )
    current = engineering_evidence(PILOT)
    if (
        saved.get("passed") is not True
        or current["eligible_generated"] < 1
        or any(saved.get(key) != value for key, value in current.items())
    ):
        raise RuntimeError("engineering gate or evidence integrity failed")
    return saved


def require_confirmation() -> dict[str, Any]:
    """Compare the running grid to the concrete registered 46-pair confirmation."""
    frozen = N.preflight(RUN)
    if frozen["config"] != N.confirmatory_config():
        raise RuntimeError(
            "confirmatory configuration differs from the registered grid"
        )
    return frozen


def source_archive(root: Path) -> None:
    """Authenticate frozen ZIP contents and recover only absent snapshot metadata."""
    frozen = N.preflight(root)
    path = root / "frozen_sources.zip"
    metadata = root / "source_snapshot.json"
    existed = path.exists()
    if not existed:
        if E.exists(metadata):
            raise RuntimeError("source archive metadata exists without its ZIP")
        with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, digest in frozen["files"].items():
                data = Path(name).read_bytes()
                if hashlib.sha256(data).hexdigest() != digest:
                    raise RuntimeError("source archive bytes differ from frozen text")
                archive.writestr(str(Path(name).relative_to(Path.cwd())), data)
    expected = {
        str(Path(name).relative_to(Path.cwd())): digest
        for name, digest in frozen["files"].items()
    }
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(expected) or set(names) != set(expected):
                raise RuntimeError(
                    "source archive has missing, duplicate or extra files"
                )
            if any(
                hashlib.sha256(archive.read(name)).hexdigest() != digest
                for name, digest in expected.items()
            ):
                raise RuntimeError("source archive contents differ from frozen sources")
    except (zipfile.BadZipFile, OSError, NotImplementedError) as error:
        raise RuntimeError(
            "source archive is corrupt or incomplete; bytes preserved"
        ) from error
    binding = {
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "files": len(frozen["files"]),
        "freeze_sha256": N.B.digest(frozen),
    }
    if E.exists(metadata):
        try:
            saved = E.read(metadata)
            valid = (
                isinstance(saved, dict)
                and all(saved.get(key) == value for key, value in binding.items())
                and type(saved.get("created_ns")) is int
                and saved["created_ns"] > 0
            )
        except (OSError, ValueError, TypeError):
            valid = False
        if not valid:
            raise RuntimeError(
                "source archive metadata does not authenticate this freeze"
            )
        return
    I.persist(
        metadata,
        {
            "created_ns": time.time_ns(),
            **binding,
            "recovered_missing_metadata": existed,
        },
    )


def prepare(*, engineering: bool) -> dict[str, Any]:
    """Freeze analysis and driver with separate pilot and confirmatory protocol copies."""
    from experiments.recursive_opt._shared.optimizer_discovery.exp17 import analysis, verify_numerics

    if not engineering:
        require_engineering()
    root = PILOT if engineering else RUN
    protocol = ROOT / ("PILOT_PROTOCOL.md" if engineering else "PREREG_EXP17.md")
    if engineering and not protocol.exists():
        protocol.write_bytes((ROOT / "PREREG_EXP17.md").read_bytes())
    frozen = N.prepare(
        root,
        pilot_config() if engineering else N.confirmatory_config(),
        protocol,
        extra_frozen_paths=[
            Path(__file__),
            Path(analysis.__file__),
            Path(verify_numerics.__file__),
            ROOT / "test_analysis.py",
            ROOT / "test_verify_numerics.py",
            ROOT / "test_source_archive.py",
        ],
    )
    source_archive(root)
    return frozen


def run_engineering(*, client: Any = None) -> dict[str, Any]:
    """Run four real responses, then validate complete evidence without audit access."""
    if N.preflight(PILOT)["config"] != pilot_config():
        raise RuntimeError(
            "engineering configuration differs from its registered scope"
        )
    if E.exists(PILOT / "engineering_results.json"):
        return require_engineering()
    N.run_generation(PILOT, client=live_client() if client is None else client)
    OLD.collect_receipts(PILOT / "raw")
    N.select_all(PILOT)
    value = engineering_evidence(PILOT)
    result = {"passed": value["eligible_generated"] > 0, **value}
    I.persist(PILOT / "engineering_results.json", result)
    return require_engineering()


def execute(action: str) -> None:
    """Execute an action while the CLI holds its exclusive run lease."""
    if action in {"prepare_pilot", "prepare"}:
        frozen = prepare(engineering=action == "prepare_pilot")
        print(
            json.dumps(
                {"freeze_sha256": N.B.digest(frozen), "config": frozen["config"]}
            ),
            flush=True,
        )
    elif action == "pilot":
        result = run_engineering()
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ["passed", "completed_responses", "eligible_generated"]
                }
            ),
            flush=True,
        )
    elif action == "generate":
        require_engineering()
        require_confirmation()
        N.run_generation(RUN, client=live_client())
        OLD.collect_receipts(RUN / "raw")
    elif action == "select":
        require_confirmation()
        N.select_all(RUN)
    elif action == "audit":
        require_confirmation()
        N.run_audit(RUN)
    elif action == "analyze":
        require_confirmation()
        from experiments.recursive_opt._shared.optimizer_discovery.exp17 import analysis

        I.persist(
            RUN / "analysis_results.json", analysis.summarize(analysis.read_bundle(RUN))
        )
    else:
        OLD.collect_receipts(RUN / "raw")


def main() -> None:
    """Run one frozen action with a lease covering all journal/evaluation mutations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "prepare_pilot",
            "pilot",
            "prepare",
            "generate",
            "select",
            "audit",
            "analyze",
            "receipts",
        ],
    )
    action = parser.parse_args().action
    with run_lease(PILOT if action in {"prepare_pilot", "pilot"} else RUN):
        execute(action)


if __name__ == "__main__":
    main()
