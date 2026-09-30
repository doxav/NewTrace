"""Verified storage moves used to retire the old research directories."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "20260929"))
from migrate_data import digest_tree


def relocate(source: Path, target: Path, receipt: Path) -> dict[str, Any]:
    """Move original bytes, allowing a destination containing noncolliding indexes."""
    if source.is_symlink() or not source.exists():
        raise ValueError("Expected an original existing source")
    if receipt.exists() or source == target or source in target.parents:
        raise ValueError("Receipt must be new and target outside source")
    if target.exists():
        if not source.is_dir() or not target.is_dir():
            raise ValueError("Only noncolliding directories can be merged")
        if any(
            (target / child.name).exists() or (target / child.name).is_symlink()
            for child in source.iterdir()
        ):
            raise ValueError("Destination contains a source name")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != target.parent.stat().st_dev:
        raise ValueError("Verified relocation requires the same filesystem")
    before = digest_tree(source)
    result: dict[str, Any] = {
        "source": str(source),
        "target": str(target),
        "files": before,
        "status": "prepared",
    }
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    if target.exists():
        for child in list(source.iterdir()):
            child.rename(target / child.name)
        source.rmdir()
    else:
        source.rename(target)
    after = digest_tree(target)
    if any(after.get(name) != value for name, value in before.items()):
        raise RuntimeError("Moved bytes differ; inspect receipt before continuing")
    result["status"] = "verified"
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    return {key: value for key, value in result.items() if key != "files"}


def mapped(path: Path, moves: list[tuple[Path, Path]]) -> Path:
    """Resolve the most specific lexical relocation without following aliases."""
    path = Path(os.path.abspath(path))
    for source, target in sorted(
        moves, key=lambda pair: len(pair[0].parts), reverse=True
    ):
        if path == source or source in path.parents:
            return target / path.relative_to(source)
    return path
