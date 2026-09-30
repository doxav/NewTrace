"""Prospectively fixed B2 reference after P1 completion; primary evidence is read only."""

from __future__ import annotations

import argparse
import fcntl
import statistics
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import generation as G
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import search_experiment as S
from experiments.recursive_opt._shared.optimizer_discovery.investigation16.production import analysis as A

ROOT = G.ROOT / "production_baseline_control"
PRIMARY_ROOT = G.ROOT / "production_run"
SOURCE_PATH = G.ROOT / "benchmark/b2/optimizer.py"
SOURCE_SHA256 = "958fbb12279a15966bf1ffa45ddffc7cd5f946ca800a3fe2b8e132a150c5c190"
PROTOCOL = Path(__file__).with_name("BASELINE_CONTROL_PROTOCOL.md")
OUTER_SEEDS = (16411, 16423, 16437, 16441, 16453, 16467)
BARRIERS = ("generation_frozen", "selections_frozen", "audit_results")


def _require(condition: bool, message: str) -> None:
    """Refuse changed, incomplete or late scientific evidence explicitly."""
    if not condition:
        raise RuntimeError(message)


def _separate(primary: Path, control: Path) -> None:
    """Prevent writing control evidence into or above the primary run tree."""
    primary, control = primary.resolve(), control.resolve()
    _require(
        not primary.is_relative_to(control) and not control.is_relative_to(primary),
        "control storage must be separate from primary evidence",
    )


def jobs(primary: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct the exact same P1 audit inputs without evaluating any objective."""
    config = primary["config"]
    expected = {
        "namespace": "P1",
        "outer_seeds": list(OUTER_SEEDS),
        "slots": 8,
        "budget": 32,
        "local_replicates": 2,
        "workers": 8,
        "timeout_s": 2,
        "arms": ["I", "C", "R", "W"],
    }
    _require(
        all(config.get(key) == value for key, value in expected.items())
        and len(primary["tasks"]["audit"]) == 12,
        "primary configuration differs from fixed baseline protocol",
    )
    return [
        {
            "id": f"{outer}/task_{index:02d}_local_{replicate}",
            "outer": outer,
            "task": task,
            "local_seed": G.local_seed(
                config["namespace"], int(B.digest([outer, replicate])[:15], 16), task
            ),
            "budget": 32,
            "source_sha256": SOURCE_SHA256,
        }
        for outer in OUTER_SEEDS
        for index, task in enumerate(primary["tasks"]["audit"])
        for replicate in range(2)
    ]


def prepare(
    primary_root: Path = PRIMARY_ROOT,
    control_root: Path = ROOT,
    *,
    source_path: Path = SOURCE_PATH,
) -> dict[str, Any]:
    """Freeze the reference before any main generation, binding the existing main freeze."""
    _separate(primary_root, control_root)
    if E.exists(control_root / "freeze.json"):
        frozen, _ = preflight(control_root)
        _require(
            frozen["primary_root"] == str(primary_root.resolve())
            and frozen["source_path"] == str(source_path.resolve()),
            "existing control registration differs",
        )
        return frozen
    started = (
        E.exists(primary_root / "generation_started.json")
        or any(primary_root.glob("raw/*/*/slot_*/started_*.json*"))
        or any(primary_root.glob("raw/*/*/slot_*/response.json*"))
    )
    _require(not started, "control must be registered before primary generation")
    primary = S.preflight(primary_root)
    source = source_path.read_text()
    _require(B.source_hash(source) == SOURCE_SHA256, "fixed B2 source hash mismatch")
    files = [
        Path(__file__),
        Path(__file__).with_name("test_baseline_control.py"),
        PROTOCOL,
        source_path,
        *[Path(module.__file__) for module in (B, E, G, I, S, A)],
        Path("opto/features/recursive_opt/optimizer_program.py"),
    ]
    frozen = {
        "schema": "investigation16.fixed_baseline_control.v1",
        "created_ns": S.clock_snapshot()["wall_ns"],
        "primary_root": str(primary_root.resolve()),
        "primary_freeze_sha256": B.digest(primary),
        "source_path": str(source_path.resolve()),
        "source": source,
        "source_sha256": SOURCE_SHA256,
        "seed_sha256": B.source_hash(B.SEED_SOURCE),
        "jobs": jobs(primary),
        "files": {
            str(path.resolve()): B.source_hash(path.read_text()) for path in files
        },
        "bootstrap": B.MANIFEST["bootstrap"],
        "workers": 8,
        "timeout_s": 2,
    }
    I.persist(control_root / "freeze.json", frozen)
    I.persist(control_root / "freeze_sha256.json", {"sha256": B.digest(frozen)})
    return frozen


def preflight(control_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check fixed source, dependencies, primary binding and schedule before any resume."""
    frozen = E.read(control_root / "freeze.json")
    primary_root = Path(frozen["primary_root"])
    _separate(primary_root, control_root)
    _require(
        B.digest(frozen) == E.read(control_root / "freeze_sha256.json")["sha256"],
        "control freeze digest mismatch",
    )
    _require(
        all(
            B.source_hash(Path(path).read_text()) == digest
            for path, digest in frozen["files"].items()
        ),
        "control frozen source mismatch",
    )
    primary = S.preflight(primary_root)
    _require(
        frozen["primary_freeze_sha256"] == B.digest(primary),
        "bound primary freeze changed",
    )
    _require(
        frozen["source_sha256"] == SOURCE_SHA256 == B.source_hash(frozen["source"])
        and frozen["seed_sha256"] == B.source_hash(B.SEED_SOURCE),
        "control or fallback source integrity mismatch",
    )
    _require(
        frozen["jobs"] == jobs(primary)
        and frozen["bootstrap"] == B.MANIFEST["bootstrap"]
        and frozen["workers"] == 8
        and frozen["timeout_s"] == 2,
        "control registered configuration mismatch",
    )
    return frozen, primary


def _primary_complete(control_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require all completion barriers and full S/A checks before reading comparators."""
    frozen, _ = preflight(control_root)
    primary_root = Path(frozen["primary_root"])
    _require(
        all(E.exists(primary_root / (name + ".json")) for name in BARRIERS)
        and E.exists(primary_root / "generation_started.json"),
        "all primary completion barriers are required",
    )
    _require(
        E.read(primary_root / "generation_started.json")["wall_ns"]
        > frozen["created_ns"],
        "control registration did not precede primary generation",
    )
    for path in A._json_paths(primary_root / "raw", "*/*/slot_*/started_*.json*"):
        _require(
            E.read(path)["time_ns"] > frozen["created_ns"],
            "control registration followed a primary request",
        )
    bundle = A.read_bundle(primary_root)
    A.summarize(bundle)
    hashes = {
        name: B.digest(E.read(primary_root / (name + ".json"))) for name in BARRIERS
    }
    verification = {
        "primary_freeze_sha256": frozen["primary_freeze_sha256"],
        "hashes": hashes,
        "audit_completed_ns": E.read(primary_root / "audit_results.json")[
            "completed_ns"
        ],
        "comparators": {
            str(outer): {
                arm: {
                    key: bundle["audit"]["per_seed"][str(outer)][arm][key]
                    for key in ("auc", "final_regret", "source_sha256")
                }
                for arm in ("A0", "R")
            }
            for outer in OUTER_SEEDS
        },
    }
    for outer in OUTER_SEEDS:
        expected = {
            (B.task_identity(job["task"]), job["local_seed"])
            for job in frozen["jobs"]
            if job["outer"] == outer
        }
        for arm in ("A0", "R"):
            rows = bundle["audit"]["per_seed"][str(outer)][arm]["rows"]
            actual = [(row["task_identity"], row["local_seed"]) for row in rows]
            _require(
                len(actual) == len(expected) and set(actual) == expected,
                "primary comparator inputs differ from control schedule",
            )
    I.persist(control_root / "primary_complete_verified.json", verification)
    return frozen, verification


def _raw_path(root: Path, job: dict[str, Any]) -> Path:
    """Map each fixed job to a separate immutable control artifact."""
    return root / "raw" / (job["id"] + ".json")


def _read_row(root: Path, job: dict[str, Any], audit_hash: str) -> dict[str, Any]:
    """Verify an already completed outcome without reexecuting an unfavorable policy."""
    saved = E.read(_raw_path(root, job))
    row = saved["row"]
    journal_path = root / "attempts" / (saved["attempt_id"] + ".started.json")
    _require(E.exists(journal_path), "completed control attempt journal is missing")
    _require(
        E.read(journal_path) == {"job_id": job["id"], "clock": saved["started"]},
        "completed control attempt journal differs",
    )
    cutoff = E.read(root / "primary_complete_verified.json")["audit_completed_ns"]
    _require(
        cutoff < saved["started"]["wall_ns"] <= saved["completed"]["wall_ns"],
        "control trajectory chronology violation",
    )
    _require(
        saved["job"] == job
        and saved["primary_audit_sha256"] == audit_hash
        and B.digest(row) == saved["row_sha256"],
        "completed control row integrity mismatch",
    )
    _require(
        row["source_sha256"] == SOURCE_SHA256
        and row["task_identity"] == B.task_identity(job["task"])
        and row["local_seed"] == job["local_seed"]
        and row["stratum"] == f"{job['task']['family']}/{job['task']['dimension']}",
        "completed control identity mismatch",
    )
    A._validate_row(row, 32, True)
    return row


def _execute(
    root: Path, frozen: dict[str, Any], job: dict[str, Any], audit_hash: str
) -> dict[str, Any]:
    """Persist one new fixed-baseline outcome; preserve errors and attempt starts separately."""
    if E.exists(_raw_path(root, job)):
        return _read_row(root, job, audit_hash)
    identifier = uuid.uuid4().hex
    started = S.clock_snapshot()
    I.persist(
        root / "attempts" / (identifier + ".started.json"),
        {"job_id": job["id"], "clock": started},
    )
    try:
        row = B.evaluate(
            frozen["source"],
            job["task"],
            job["local_seed"],
            budget=32,
            deployment=True,
            seed_source=B.SEED_SOURCE,
            timeout_s=2,
        )
        A._validate_row(row, 32, True)
    except Exception as error:
        I.persist(
            root / "attempts" / (identifier + ".error.json"),
            {
                "job_id": job["id"],
                "error_type": type(error).__name__,
                "clock": S.clock_snapshot(),
                "partial_unpersisted_work_unknown": True,
            },
        )
        raise
    I.persist(
        _raw_path(root, job),
        {
            "job": job,
            "row": row,
            "row_sha256": B.digest(row),
            "attempt_id": identifier,
            "primary_audit_sha256": audit_hash,
            "started": started,
            "completed": S.clock_snapshot(),
        },
    )
    return _read_row(root, job, audit_hash)


def _summary(
    root: Path, frozen: dict[str, Any], primary: dict[str, Any]
) -> dict[str, Any]:
    """Aggregate every fixed job and both comparisons without a winner-selection step."""
    expected = {_raw_path(root, job) for job in frozen["jobs"]}
    actual = set(A._json_paths(root / "raw", "*/*.json*"))
    _require(actual == expected, "complete control schedule required for analysis")
    rows = [
        _read_row(root, job, primary["hashes"]["audit_results"])
        for job in frozen["jobs"]
    ]
    per_seed = {}
    for outer in OUTER_SEEDS:
        panel = [row for job, row in zip(frozen["jobs"], rows) if job["outer"] == outer]
        per_seed[str(outer)] = {
            "B2": {
                "auc": B.aggregate(panel, "auc"),
                "final_regret": B.aggregate(panel, "final_regret"),
                "source_sha256": SOURCE_SHA256,
                "deployment": A._deployment(panel),
            },
            **primary["comparators"][str(outer)],
        }
    invocation_finishes = [
        E.read(path) for path in A._json_paths(root / "invocations", "*.finished.json*")
    ]
    failures = [
        E.read(path) for path in A._json_paths(root / "attempts", "*.error.json*")
    ]
    starts = A._json_paths(root / "attempts", "*.started.json*")
    completed_ids = {E.read(path)["attempt_id"] for path in actual}
    failed_ids = {
        path.name.split(".")[0]
        for path in A._json_paths(root / "attempts", "*.error.json*")
    }
    unresolved = [
        path.name.split(".")[0]
        for path in starts
        if path.name.split(".")[0] not in completed_ids | failed_ids
    ]
    values = [per_seed[str(outer)]["B2"]["auc"] for outer in OUTER_SEEDS]
    return {
        "schema": "investigation16.fixed_baseline_control_results.v1",
        "supplementary_exploratory": True,
        "freeze_sha256": B.digest(frozen),
        "primary_verification": primary,
        "source_sha256": SOURCE_SHA256,
        "outer_seeds": list(OUTER_SEEDS),
        "per_seed": per_seed,
        "B2_auc": {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
        },
        "contrasts": {
            f"{left}-{right}": E.paired(
                [
                    per_seed[str(outer)][left]["auc"]
                    - per_seed[str(outer)][right]["auc"]
                    for outer in OUTER_SEEDS
                ]
            )
            for left, right in (("B2", "A0"), ("R", "B2"))
        },
        "deployment": A._deployment(rows),
        "resources": {
            **A._allocation(rows),
            "generative_calls": 0,
            "unique_reference_design_calls_shared_with_primary": 1536,
            "recorded_invocation_reference_objective_calls": sum(
                record["reference_objective_calls"] for record in invocation_finishes
            ),
            "uncheckpointed_attempt_ids": unresolved,
            "infrastructure_errors": failures,
            "accounting_limit": "Completed rows and finished invocation counters exclude unknown work lost before persistence; no primary comparator was rerun.",
        },
        "interpretation_limits": [
            "Supplementary fixed-reference comparison; primary R-I is unchanged.",
            "Six paired outer seeds share a fixed audit panel; bootstrap uncertainty is exploratory and fragile.",
            "B2 was developed on separate public fixtures; observed transfer does not retroactively strengthen P1's seed or isolate recursive-feedback causality.",
            "Candidate-only failure and common seed fallback accompany deployment metrics.",
        ],
    }


def analyze(control_root: Path = ROOT) -> dict[str, Any]:
    """Read complete primary/control evidence and recompute supplementary metrics only."""
    frozen, primary = _primary_complete(control_root)
    return _summary(control_root, frozen, primary)


def run(control_root: Path = ROOT) -> dict[str, Any]:
    """Execute only unfinished B2 jobs after primary completion, under a single process lock."""
    frozen, primary = _primary_complete(control_root)
    with (control_root / "execution.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another control execution is active") from None
        expected = {_raw_path(control_root, job) for job in frozen["jobs"]}
        actual = set(A._json_paths(control_root / "raw", "*/*.json*"))
        _require(actual <= expected, "unexpected control checkpoint")
        for job in frozen["jobs"]:
            if _raw_path(control_root, job) in actual:
                _read_row(control_root, job, primary["hashes"]["audit_results"])
        if E.exists(control_root / "results.json"):
            result = _summary(control_root, frozen, primary)
            I.persist(control_root / "results.json", result)
            return result
        missing = [
            job for job in frozen["jobs"] if not E.exists(_raw_path(control_root, job))
        ]
        if missing:
            identifier = uuid.uuid4().hex
            started, misses = S.clock_snapshot(), B._reference_scale.cache_info().misses
            I.persist(
                control_root / "invocations" / (identifier + ".started.json"),
                {
                    "clock": started,
                    "scheduled_unfinished_jobs": [job["id"] for job in missing],
                },
            )
            try:
                with ThreadPoolExecutor(max_workers=8) as pool:
                    list(
                        pool.map(
                            lambda job: _execute(
                                control_root,
                                frozen,
                                job,
                                primary["hashes"]["audit_results"],
                            ),
                            missing,
                        )
                    )
            finally:
                ended = S.clock_snapshot()
                I.persist(
                    control_root / "invocations" / (identifier + ".finished.json"),
                    {
                        "start": started,
                        "end": ended,
                        "elapsed": S.elapsed_clocks(started, ended),
                        "reference_objective_calls": (
                            B._reference_scale.cache_info().misses - misses
                        )
                        * B.MANIFEST["normalization_reference_size"],
                    },
                )
        result = _summary(control_root, frozen, primary)
        I.persist(control_root / "results.json", result)
        return result


def main() -> None:
    """Expose explicit prepare/run/analyze stages without accessing a model client."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument("--primary-root", type=Path, default=PRIMARY_ROOT)
    parser.add_argument("--control-root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.primary_root, args.control_root)
    elif args.command == "run":
        run(args.control_root)
    else:
        I.persist(args.control_root / "results.json", analyze(args.control_root))


if __name__ == "__main__":
    main()
