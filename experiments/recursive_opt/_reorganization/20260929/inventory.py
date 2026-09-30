"""Inventory experiment files without reading raw results or following symlinks."""

from __future__ import annotations

import gzip
import json
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOTS = (Path("/home/xav/code/Trace"), Path("/home/xav/code/Trace-experiment0"))


def role(relative: Path) -> str:
    """Classify storage separately from scientific result status."""
    parts = relative.parts
    if any(
        part.startswith(".venv") or part in {"worktrees", "node_modules"}
        for part in parts
    ):
        return "runtime_or_frozen_checkout"
    if any(
        part in {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
        for part in parts
    ):
        return "regenerable_cache"
    if relative.suffix.lower() == ".md":
        return "documentation"
    if relative.suffix in {".py", ".sh", ".ipynb"}:
        return "code_or_notebook"
    if relative.suffix in {".zip", ".tar", ".tgz"}:
        return "archive"
    if relative.suffix in {".png", ".svg", ".pdf", ".html"}:
        return "presentation"
    return "configuration_or_evidence"


def group(relative: Path) -> str:
    """Return a bounded directory grouping for the human inventory."""
    parts = relative.parts
    if parts[:2] == ("experiments", "recursive_opt"):
        return "/".join(parts[:3])
    if parts[0] == "artifacts":
        return "/".join(parts[:2]) if len(parts) > 2 else "artifacts/root_documents"
    return "/".join(parts[:3])


def main() -> None:
    """Write a complete metadata stream and compact count/size/document indexes."""
    summary: dict[str, Any] = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "roots": {},
    }
    documents: list[dict[str, Any]] = []
    large: list[dict[str, Any]] = []
    with gzip.open(HERE / "inventory.jsonl.gz", "wt") as stream:
        for root in ROOTS:
            tracked = set(
                subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"])
                .decode()
                .split("\0")
            )
            aggregates: dict[str, Any] = defaultdict(
                lambda: {
                    "files": 0,
                    "bytes": 0,
                    "tracked_files": 0,
                    "roles": Counter(),
                    "extensions": Counter(),
                }
            )
            scopes = [
                root / name
                for name in (
                    "artifacts",
                    "experiments/recursive_opt",
                    "examples/notebook_outputs/recursive_opt_use_cases",
                    "outputs/recursive_opt",
                    "XP_recurse_2",
                    "notebook_outputs",
                    "trace_memory",
                    "memOLD",
                    "mem_A_multi_param",
                    "mem_A_online_bin_packing_local",
                    "logs/otlp_langgraph",
                )
            ]
            files: list[Path] = []
            for scope in scopes:
                if not scope.exists():
                    continue
                for current, directories, names in os.walk(scope, followlinks=False):
                    if Path(current) == HERE:
                        directories[:] = []
                        continue
                    directories[:] = [d for d in directories if d != ".git"]
                    for name in names:
                        if name != ".git":
                            files.append(Path(current) / name)
                    for name in directories:
                        path = Path(current) / name
                        if path.is_symlink():
                            files.append(path)
            examples = root / "examples"
            files.extend(p for p in examples.glob("*recursive*") if p.is_file())
            files.extend(p for p in examples.glob("*[Cc]urriculum*") if p.is_file())
            for path in sorted(set(files)):
                relative = path.relative_to(root)
                try:
                    stat = path.lstat()
                except FileNotFoundError:
                    continue  # A regenerable/live file disappeared during inventory.
                category = role(relative)
                extension = (
                    "".join(path.suffixes[-2:])
                    if path.suffix == ".gz"
                    else path.suffix or "(none)"
                )
                row = {
                    "root": root.name,
                    "path": relative.as_posix(),
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "tracked": relative.as_posix() in tracked,
                    "role": category,
                    "symlink": path.is_symlink(),
                }
                if row["symlink"]:
                    row["link_target"] = os.readlink(path)
                stream.write(json.dumps(row) + "\n")
                bucket = aggregates[group(relative)]
                bucket["files"] += 1
                bucket["bytes"] += stat.st_size
                bucket["tracked_files"] += int(row["tracked"])
                bucket["roles"][category] += 1
                bucket["extensions"][extension] += 1
                if category == "documentation":
                    documents.append(row)
                if (
                    stat.st_size >= 20_000_000
                    and category != "runtime_or_frozen_checkout"
                ):
                    large.append(row)
            summary["roots"][root.name] = dict(sorted(aggregates.items()))
            print(
                root.name,
                sum(v["files"] for v in aggregates.values()),
                "files inventoried",
                flush=True,
            )
    (HERE / "inventory_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (HERE / "documents.json").write_text(json.dumps(documents, indent=2) + "\n")
    (HERE / "large_files.json").write_text(
        json.dumps(sorted(large, key=lambda r: r["bytes"], reverse=True), indent=2)
        + "\n"
    )
    print("Documentation:", len(documents), "large non-runtime files:", len(large))


if __name__ == "__main__":
    main()
