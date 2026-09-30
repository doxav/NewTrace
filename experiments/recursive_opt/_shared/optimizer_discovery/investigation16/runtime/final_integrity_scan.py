"""Scan final deliverables privately and verify the preserved repository baseline."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

from experiments.recursive_opt._shared.optimizer_discovery.phase0 import _load_key

REPO = Path(__file__).resolve().parents[6]
STUDY = REPO / "experiments/recursive_opt/_shared/optimizer_discovery/investigation16"
OUTPUT = STUDY / "runtime/final_integrity_scan.json"
BASE = "13ebda2242e1c18022591737b113030ca2ce2da2"
PROBE_HASH = "ca8a08e5c2eca14c1004e2af3241ae8d9ebf2909d386eee9e1e2611f5b5e84e8"


def git(*args: str) -> str:
    """Collect repository metadata before loading the private credential."""
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def scan_stream(stream: BinaryIO, patterns: list[bytes]) -> bool:
    """Detect private bytes across chunk boundaries without returning their value."""
    overlap = max(map(len, patterns)) - 1
    tail = b""
    while chunk := stream.read(1 << 20):
        data = tail + chunk
        if any(pattern in data for pattern in patterns):
            return True
        tail = data[-overlap:] if overlap else b""
    return False


def main() -> None:
    """Inspect artifacts and archives; save only counts and public provenance."""
    if OUTPUT.exists():
        raise RuntimeError("Completed integrity evidence must not be overwritten")
    head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    changed = git("diff", "--name-only").splitlines()
    staged = git("diff", "--cached", "--name-only")
    diff_check = git("diff", "--check")
    expected_changed = {
        "experiments/recursive_opt/_history/navigation/RESEARCH_LOG.md",
        "experiments/recursive_opt/ASSESSMENT.md",
    }
    if head != BASE or set(changed) != expected_changed or staged or diff_check:
        raise RuntimeError("Repository baseline or expected change boundary differs")
    probe = REPO / "experiments/recursive_opt/_history/probe_2026/probe_aa_results.json"
    probe_hash = hashlib.sha256(probe.read_bytes()).hexdigest()
    if probe_hash != PROBE_HASH:
        raise RuntimeError("Preserved user artifact differs from its starting hash")

    # Do not create any subprocess after this point: the credential stays private.
    _load_key()
    key = os.environ.get("OPENROUTER_API_KEY", "").encode()
    if not key.startswith(b"sk-or-v1-") or len(key) < 32:
        raise RuntimeError("Credential unavailable for the private exact-byte scan")
    env_bytes = (REPO / ".env").read_bytes().strip()
    patterns = [key]
    if len(env_bytes) >= 32:
        patterns.append(env_bytes)
    for line in env_bytes.splitlines():
        name, separator, value = line.partition(b"=")
        value = value.strip().strip(b"\"'")
        if separator and name.strip().endswith(b"_SOURCE") and len(value) >= 32:
            patterns.append(value)

    roots = [STUDY, REPO / "experiments/recursive_opt/EXP16/presentation"]
    files: list[Path] = []
    for root in roots:
        for directory, subdirs, names in os.walk(root, followlinks=False):
            subdirs[:] = [
                name
                for name in subdirs
                if name not in {"__pycache__", "node_modules", ".pytest_cache"}
                and not (Path(directory) / name).is_symlink()
            ]
            files.extend(Path(directory) / name for name in names)
    files.extend(REPO / name for name in changed)
    files.extend((REPO / "tests/unit_tests").glob("test_investigation16_*.py"))
    files = sorted({path for path in files if path.is_file()})
    violations: list[str] = []
    compressed_members = 0
    physical_bytes = 0
    for path in files:
        label = str(path.relative_to(REPO))
        if path.name == ".env" or path.name.startswith(".env."):
            violations.append(label + ": forbidden environment file")
        physical_bytes += path.stat().st_size
        with path.open("rb") as stream:
            if scan_stream(stream, patterns):
                violations.append(label + ": private material detected")
        if path.suffix == ".gz":
            compressed_members += 1
            with gzip.open(path, "rb") as stream:
                if scan_stream(stream, patterns):
                    violations.append(label + ": private compressed material detected")
        elif path.suffix in {".zip", ".xlsx"}:
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    compressed_members += 1
                    if Path(member.filename).name.startswith(".env"):
                        violations.append(label + ": environment archive member")
                    with archive.open(member) as stream:
                        if scan_stream(stream, patterns):
                            violations.append(
                                label + ": private archive material detected"
                            )
    record = {
        "status": "FAIL" if violations else "PASS",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "head": head,
        "branch": branch,
        "tracked_changed_paths": changed,
        "staged_paths": [],
        "git_diff_check": "PASS",
        "user_probe_sha256": probe_hash,
        "files_scanned": len(files),
        "physical_bytes_scanned": physical_bytes,
        "compressed_members_scanned": compressed_members,
        "exact_active_credential_checked": True,
        "environment_contents_and_secret_source_value_checked": True,
        "violations": violations,
        "excluded": [
            "symlink directories",
            "node_modules",
            "__pycache__",
            ".pytest_cache",
        ],
        "model_calls": 0,
        "candidate_executions": 0,
        "objective_calls": 0,
        "scanner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    OUTPUT.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: record[key]
                for key in ["status", "files_scanned", "compressed_members_scanned"]
            }
        )
    )
    if violations:
        raise RuntimeError(
            "Final integrity scan found private material; inspect sanitized record"
        )


if __name__ == "__main__":
    main()
