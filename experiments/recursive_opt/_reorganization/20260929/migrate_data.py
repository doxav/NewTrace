"""Relocate closed evidence stores with full hash verification and legacy aliases."""

from __future__ import annotations

import ctypes
import gzip
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def relocate_with_atomic_alias(source: Path, target: Path) -> dict[str, Any]:
    """Atomically exchange a directory/file for its legacy alias on Linux.

    The original inode becomes the target. Open handles keep referring to it,
    while new lookups of the old name resolve through the alias without a gap.
    This does not freeze live content; it records filesystem identity instead.
    """
    if source.is_symlink() or not source.exists():
        raise ValueError("Atomic migration requires an existing original source")
    if target.exists() or target.is_symlink() or source in target.parents:
        raise ValueError("Atomic migration requires a vacant external target")
    target.parent.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    if before.st_dev != target.parent.stat().st_dev:
        raise ValueError("Atomic alias migration requires one filesystem")
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    target.symlink_to(
        os.path.relpath(target, source.parent), target_is_directory=source.is_dir()
    )
    result = rename(-100, os.fsencode(source), -100, os.fsencode(target), 2)
    if result != 0:
        error = ctypes.get_errno()
        target.unlink()
        raise OSError(error, os.strerror(error))
    after = target.stat()
    if (before.st_dev, before.st_ino) != (
        after.st_dev,
        after.st_ino,
    ) or source.resolve() != target.resolve():
        raise RuntimeError(
            "Atomic exchange completed but identity check failed; inspect paths before proceeding"
        )
    return {
        "source": str(source),
        "target": str(target),
        "device": before.st_dev,
        "inode": before.st_ino,
        "status": "atomic_alias_verified",
        "utc": datetime.now(timezone.utc).isoformat(),
    }


def digest_tree(root: Path) -> dict[str, str]:
    """Hash every file without following symlinks or executing saved code."""
    result: dict[str, str] = {}
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    for path in paths:
        name = "." if path == root else path.relative_to(root).as_posix()
        if path.is_symlink():
            result[name] = "symlink:" + os.readlink(path)
        elif path.is_file():
            with path.open("rb") as stream:
                result[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def relocate(source: Path, target: Path, ledger: Path) -> dict[str, Any]:
    """Move once, install a compatibility link, and roll back on verification failure."""
    if source.is_symlink() or not source.exists():
        raise ValueError("Migration requires an existing non-symlink source")
    if target.exists() or target.is_symlink() or ledger.exists():
        raise ValueError("Migration target or ledger already exists")
    if source == target or source in target.parents:
        raise ValueError("Migration target must be outside the source")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != target.parent.stat().st_dev:
        raise ValueError("This migration requires a same-filesystem atomic rename")
    before = digest_tree(source)
    record: dict[str, Any] = {
        "source": str(source),
        "target": str(target),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
        "files": before,
    }
    ledger.write_bytes(
        gzip.compress(json.dumps(record, sort_keys=True).encode(), mtime=0)
    )
    source.rename(target)
    try:
        source.symlink_to(
            os.path.relpath(target, source.parent), target_is_directory=target.is_dir()
        )
        if source.resolve() != target.resolve() or digest_tree(target) != before:
            raise RuntimeError("Post-move evidence verification failed")
    except BaseException:
        if source.is_symlink():
            source.unlink()
        if not source.exists():
            target.rename(source)
        record["status"] = "rolled_back"
        ledger.write_bytes(
            gzip.compress(json.dumps(record, sort_keys=True).encode(), mtime=0)
        )
        raise
    record["status"] = "verified"
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    ledger.write_bytes(
        gzip.compress(json.dumps(record, sort_keys=True).encode(), mtime=0)
    )
    return {k: v for k, v in record.items() if k != "files"} | {
        "file_count": len(before)
    }


def main() -> None:
    """Apply the inspected closed-store plan; never rerun completed moves silently."""
    plan = json.loads((HERE / "data_moves_plan.json").read_text())
    journal = HERE / "data_moves.jsonl"
    for index, row in enumerate(plan):
        print("Starting", row["source"], flush=True)
        result = relocate(
            Path(row["source"]),
            Path(row["target"]),
            HERE / f"data_move_{index:03d}.json.gz",
        )
        with journal.open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        print("Verified", result["file_count"], "files:", row["target"], flush=True)


if __name__ == "__main__":
    main()
