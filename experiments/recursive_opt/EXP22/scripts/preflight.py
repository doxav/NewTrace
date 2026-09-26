"""Capture read-only EXP22 provenance and enforce the first scientific gate."""

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SECRET = re.compile(rb"sk-(?:or-v1-)?[A-Za-z0-9_-]{20,}")
BENCHMARKS = {
    "benchmarks/ADRS/prism/": {
        "config.yaml": "ce9257aae14f1e1f865ed08b2a47d4c30d98beb477c308b4b38f249ce86432c9",
        "initial_program.py": "db9a35fbee16d97055e07594586a9d84817fd7ac63f539a254d282b162f3d819",
        "evaluator/evaluator.py": "93a18d0f620038d9ece54ff447a788631572c5db563947605266f523c1cd9c95",
    },
    "benchmarks/math/signal_processing/": {
        "config.yaml": "294876fd92aa7486083a23b172671b917b09db1a95cc07edfc72bbd45fd5ec9d",
        "initial_program.py": "9d317dc0fccf93f53ba5db218470322c9ee140314aee465670a881f94db17378",
        "evaluator/evaluator.py": "c2bfad2a428e79e874d7a5088b7b27a65f1eae9619aa5fa06c4f394e19ff504b",
    },
}


def git(repo: Path, *args: str) -> str:
    """Run a read-only Git query without exposing raw errors or secrets."""
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=False
    )
    if result.returncode:
        raise ValueError(f"Required Git query failed: {args[0]}")
    return SECRET.sub(b"[REDACTED]", result.stdout).decode().rstrip("\n")


def branch_gate(branch: str, head: str, required_head: str | None) -> dict[str, Any]:
    """Reject missing, detached, or substituted Trace branch identities."""
    passed = branch == "recursive_opt" and required_head is not None and head == required_head
    return {
        "passed": passed,
        "required_branch": "recursive_opt",
        "actual_branch": branch,
        "actual_head": head,
        "required_head": required_head,
        "reason": None if passed else "Trace checkout does not resolve to the required recursive_opt branch",
    }


def write_json(path: Path, value: Any) -> None:
    """Write evidence only after rejecting credential-like content."""
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if SECRET.search(encoded):
        raise ValueError("Credential-like content rejected from evidence")
    path.write_bytes(encoded)


def source_hashes(repo: Path, paths: list[str]) -> dict[str, Any]:
    """Hash files on disk and at HEAD to expose working-tree deviations."""
    records = {}
    for name in git(repo, "ls-files", "--", *paths).splitlines():
        path = repo / name
        committed = subprocess.run(
            ["git", "-C", str(repo), "show", f"HEAD:{name}"],
            capture_output=True, check=False,
        )
        records[name] = {
            "working_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
            "head_sha256": hashlib.sha256(committed.stdout).hexdigest() if committed.returncode == 0 else None,
        }
    return records


def secret_scan(root: Path) -> list[str]:
    """Return offending file paths, never the credential-like matches."""
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and SECRET.search(path.read_bytes())
    )


def audit(trace: Path, sky: Path, output: Path) -> int:
    """Persist an immutable source audit; never launch an optimizer."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    evidence = output / "artifacts" / "preflight" / stamp
    evidence.mkdir(parents=True, exist_ok=False)
    repositories = {}
    hashes = {}
    paths = {
        "trace": ["opto", "pyproject.toml", "setup.py", "requirements.txt"],
        "skydiscover": ["skydiscover/optimize", *BENCHMARKS, "pyproject.toml"],
    }
    for label, repo in (("trace", trace), ("skydiscover", sky)):
        repositories[label] = {
            "path": str(repo),
            "head": git(repo, "rev-parse", "HEAD"),
            "branch": git(repo, "branch", "--show-current"),
            "status_porcelain": git(repo, "status", "--porcelain"),
        }
        hashes[label] = source_hashes(repo, paths[label])
        diff = git(repo, "diff", "HEAD", "--", *paths[label])
        (evidence / f"{label}_working_tree.patch").write_text(diff + "\n")
    try:
        required_head = git(trace, "rev-parse", "--verify", "refs/heads/recursive_opt^{commit}")
    except ValueError:
        required_head = None
    gate = branch_gate(repositories["trace"]["branch"], repositories["trace"]["head"], required_head)
    benchmarks = {}
    for prefix, expected in BENCHMARKS.items():
        for name, digest in expected.items():
            record = hashes["skydiscover"].get(prefix + name, {})
            benchmarks[prefix + name] = {
                **record,
                "supplied_sha256": digest,
                "matches_supplied_snapshot": record.get("working_sha256") == digest,
                "matches_head": record.get("working_sha256") is not None
                and record.get("working_sha256") == record.get("head_sha256"),
            }
    manifest = {
        "experiment": "EXP22 SkyDiscover/Trace PRISM and Signal comparison",
        "timestamp_utc": stamp,
        "status": "S0_BRANCH_PASSED_ONLY" if gate["passed"] else "STOPPED_PRECHECK",
        "repositories": repositories,
        "trace_branch_gate": gate,
        "benchmarks": benchmarks,
        "paid_calls": 0,
        "completed_runs": [],
        "evidence_directory": str(evidence.relative_to(output)),
        "gates_not_run": ["S0 dependency imports", "S1 evaluator parity", "S2 transport", "S3 compile", "S4 candidate", "S5 pilot"],
        "execution_authorized_by_this_script": False,
    }
    write_json(evidence / "manifest.json", manifest)
    write_json(evidence / "source_hashes.json", hashes)
    write_json(evidence / "environment.json", {
        "python": platform.python_version(), "executable": sys.executable,
        "scope": "stdlib provenance audit only; benchmark environment not created",
    })
    for name in ("manifest.json", "source_hashes.json", "environment.json"):
        target = output / name if name == "manifest.json" else output / "artifacts" / name
        target.write_bytes((evidence / name).read_bytes())
    if not gate["passed"]:
        write_json(output / "artifacts" / "STOP.json", {
            "status": "STOPPED_PRECHECK", "reason": gate["reason"], "iteration": 0,
            "last_valid_score": None, "last_active_policy": None,
            "evidence": manifest["evidence_directory"],
            "recommended_fix": "Resolve the required Trace branch identity explicitly before restarting S0; preserve dirty user work.",
        })
    leaks = secret_scan(output)
    write_json(output / "artifacts" / "secret_scan.json", {"passed": not leaks, "files_with_matches": leaks})
    print(json.dumps({"status": manifest["status"], "secret_scan_passed": not leaks, "paid_calls": 0}))
    return 0 if gate["passed"] and not leaks else 2


def main() -> int:
    """Parse repository locations for the fail-closed provenance audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=Path.home() / "code/Trace")
    parser.add_argument("--sky", type=Path, default=Path.home() / "code/evo-compare/repos/skydiscover")
    args = parser.parse_args()
    return audit(args.trace, args.sky, ROOT)


if __name__ == "__main__":
    sys.exit(main())
