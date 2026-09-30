"""Run registered successor CLIs in order, preserving a separate operational journal.

This helper does not implement scientific resume, load credentials or retry a
failed stage. Each explicit invocation delegates every stage to its frozen CLI,
which owns exact proposal identities and scientific idempotence. Child output is
inherited; only sanitized operational metadata are written here.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I

REPOSITORY = Path(__file__).resolve().parents[5]
STAGES = ("generate", "receipts", "select", "audit", "analyze", "verify_numerics")


def _driver(experiment: str) -> ModuleType:
    """Resolve an allowlisted existing driver, never an arbitrary module or path."""
    if experiment not in {"exp17", "exp18"}:
        raise ValueError("pipeline experiment must be exp17 or exp18")
    return importlib.import_module(f"experiments.recursive_opt._shared.optimizer_discovery.{experiment}.driver")


@contextmanager
def pipeline_lease(root: Path) -> Iterator[None]:
    """Exclude other pipelines without holding the child driver's process lease."""
    if not root.is_dir():
        raise RuntimeError("pipeline requires an existing prepared study directory")
    runtime = root / "runtime"
    runtime.mkdir(exist_ok=True)
    with (runtime / ".pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("pipeline already owns this registered run") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _commands(experiment: str, root: Path) -> list[list[str]]:
    """Build exact argument vectors for the five driver stages and numeric verifier."""
    return [
        [
            sys.executable,
            "-m",
            f"experiments.recursive_opt._shared.optimizer_discovery.{experiment}.driver",
            action,
        ]
        for action in STAGES[:-1]
    ] + [
        [
            sys.executable,
            "-m",
            "experiments.recursive_opt._shared.optimizer_discovery.exp17.verify_numerics",
            str(root),
        ]
    ]


def _finish(
    directory: Path, *, status: str, returncode: int, completed_steps: int
) -> dict[str, Any]:
    """Seal one launch outcome without rewriting any earlier step or launch."""
    report = {
        "schema": "optimizer_successor.pipeline_outcome.v1",
        "launch": directory.name,
        "status": status,
        "returncode": returncode,
        "completed_steps": completed_steps,
        "finished_ns": time.time_ns(),
    }
    I.persist(directory / "launch_finished.json", report)
    return report


def run_pipeline(experiment: str) -> dict[str, Any]:
    """Authenticate the registered study and run stages once, stopping on any failure."""
    if experiment not in {"exp17", "exp18"}:
        raise ValueError("pipeline experiment must be exp17 or exp18")
    driver = _driver(experiment)
    frozen = (
        driver.require_confirmation()
        if experiment == "exp17"
        else driver.require_main()
    )
    if frozen["config"]["experiment"] != experiment.replace("exp", "EXP-"):
        raise RuntimeError("pipeline and registered study identities disagree")
    engineering = driver.require_engineering()
    root = driver.RUN
    commands = _commands(experiment, root)
    with pipeline_lease(root):
        index = 1
        while (root / "runtime" / f"launch_{index:03d}").exists():
            index += 1
        directory = root / "runtime" / f"launch_{index:03d}"
        directory.mkdir()
        source = Path(__file__).read_bytes()
        source_hash = hashlib.sha256(source).hexdigest()
        I.persist(
            directory / "helper_source.json",
            {
                "source": source.decode("utf-8"),
                "source_sha256": source_hash,
                "source_bytes": len(source),
            },
        )
        I.persist(
            directory / "launch_started.json",
            {
                "schema": "optimizer_successor.pipeline_launch.v1",
                "experiment": experiment,
                "run_root": str(root),
                "launch": directory.name,
                "parent_pid": os.getpid(),
                "started_ns": time.time_ns(),
                "freeze_sha256": B.digest(frozen),
                "engineering_evidence_sha256": B.digest(engineering),
                "helper_source_sha256": source_hash,
                "commands": commands,
                "resume_policy": "Explicit invocation delegates every frozen CLI stage; no automatic retries or stage skipping.",
                "output_policy": "Inherit child output; no additional stdout/stderr capture.",
            },
        )
        for step, (stage, command) in enumerate(zip(STAGES, commands), start=1):
            started_ns = time.time_ns()
            I.persist(
                directory / f"step_{step:02d}_started.json",
                {
                    "step": step,
                    "stage": stage,
                    "command": command,
                    "started_ns": started_ns,
                },
            )
            try:
                process = subprocess.Popen(command, cwd=REPOSITORY)
                I.persist(
                    directory / f"step_{step:02d}_process.json",
                    {
                        "step": step,
                        "stage": stage,
                        "command": command,
                        "parent_pid": os.getpid(),
                        "child_pid": process.pid,
                        "recorded_ns": time.time_ns(),
                    },
                )
                returncode = process.wait()
            except BaseException as error:
                I.persist(
                    directory / f"step_{step:02d}_finished.json",
                    {
                        "step": step,
                        "stage": stage,
                        "finished_ns": time.time_ns(),
                        "returncode": None,
                        "error_type": type(error).__name__,
                    },
                )
                report = _finish(
                    directory, status="failed", returncode=1, completed_steps=step - 1
                )
                if isinstance(error, OSError):
                    return report
                raise
            I.persist(
                directory / f"step_{step:02d}_finished.json",
                {
                    "step": step,
                    "stage": stage,
                    "finished_ns": time.time_ns(),
                    "returncode": returncode,
                },
            )
            if returncode != 0:
                return _finish(
                    directory,
                    status="failed",
                    returncode=returncode,
                    completed_steps=step - 1,
                )
        return _finish(
            directory, status="complete", returncode=0, completed_steps=len(STAGES)
        )


def main() -> None:
    """Execute one explicitly selected registered pipeline and preserve its exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["exp17", "exp18"])
    report = run_pipeline(parser.parse_args().experiment)
    print(json.dumps(report), flush=True)
    code = report["returncode"]
    if code:
        raise SystemExit(code if code > 0 else 128 - code)


if __name__ == "__main__":
    main()
