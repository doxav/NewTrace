"""Crash recovery and integrity of portable source snapshots; no scientific calls."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from experiments.recursive_opt._shared.optimizer_discovery.exp17 import driver as D


@pytest.fixture
def archive_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Any]]:
    """Supply two immutable local source files and a synthetic authenticated freeze."""
    monkeypatch.chdir(tmp_path)
    sources = {
        "a.py": b"def first():\n    return 1\n",
        "nested/b.py": b"# UTF-8: caf\xc3\xa9\n",
    }
    frozen: dict[str, Any] = {"schema": "unit.freeze", "files": {}}
    for name, data in sources.items():
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
        frozen["files"][str(path)] = hashlib.sha256(data).hexdigest()
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setattr(D.N, "preflight", lambda _: frozen)
    return root, frozen


def write_zip(root: Path, frozen: dict[str, Any], mode: str = "complete") -> Path:
    """Create exact or deliberately incomplete synthetic snapshots without extraction."""
    path = root / "frozen_sources.zip"
    with zipfile.ZipFile(path, "x") as archive:
        for index, name in enumerate(frozen["files"]):
            if mode == "missing" and index == 1:
                continue
            data = (
                b"altered source\n"
                if mode == "altered" and index == 0
                else Path(name).read_bytes()
            )
            relative = str(Path(name).relative_to(Path.cwd()))
            archive.writestr(relative, data)
        if mode == "extra":
            archive.writestr("unexpected.py", "unexpected")
        if mode == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate"):
                archive.writestr("a.py", Path(next(iter(frozen["files"]))).read_bytes())
    return path


def metadata(root: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    """Return metadata whose outer digest is consistent even for a bad ZIP fixture."""
    return {
        "created_ns": 123,
        "archive_sha256": hashlib.sha256(
            (root / "frozen_sources.zip").read_bytes()
        ).hexdigest(),
        "files": len(frozen["files"]),
        "freeze_sha256": D.N.B.digest(frozen),
    }


def test_complete_snapshot_replays_without_changing_bytes(
    archive_fixture: tuple[Path, dict[str, Any]],
) -> None:
    """Both initially created ZIP and authenticated metadata remain immutable."""
    root, frozen = archive_fixture
    D.source_archive(root)
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    D.source_archive(root)
    assert before == {path.name: path.read_bytes() for path in root.iterdir()}
    saved = D.E.read(root / "source_snapshot.json")
    assert saved["files"] == 2
    assert saved["freeze_sha256"] == D.N.B.digest(frozen)


def test_recovers_metadata_after_complete_zip_was_persisted(
    archive_fixture: tuple[Path, dict[str, Any]],
) -> None:
    """A crash between closing the complete ZIP and writing metadata is recoverable."""
    root, frozen = archive_fixture
    path = write_zip(root, frozen)
    before = path.read_bytes()
    D.source_archive(root)
    assert path.read_bytes() == before
    saved = D.E.read(root / "source_snapshot.json")
    assert saved["archive_sha256"] == hashlib.sha256(before).hexdigest()
    assert saved["files"] == len(frozen["files"])
    assert saved["freeze_sha256"] == D.N.B.digest(frozen)
    assert saved["recovered_missing_metadata"] is True
    D.source_archive(root)
    assert D.E.read(root / "source_snapshot.json") == saved


@pytest.mark.parametrize(
    "mode", ["missing", "extra", "duplicate", "altered", "corrupt"]
)
@pytest.mark.parametrize("has_metadata", [False, True])
def test_rejects_incomplete_corrupt_or_changed_contents_without_replacement(
    archive_fixture: tuple[Path, dict[str, Any]], mode: str, has_metadata: bool
) -> None:
    """Matching outer ZIP digests cannot authenticate wrong or missing source members."""
    root, frozen = archive_fixture
    path = root / "frozen_sources.zip"
    if mode == "corrupt":
        path.write_bytes(b"PK interrupted archive")
    else:
        write_zip(root, frozen, mode)
    if has_metadata:
        D.I.persist(root / "source_snapshot.json", metadata(root, frozen))
    before = {item.name: item.read_bytes() for item in root.iterdir()}
    with pytest.raises(RuntimeError, match="source archive"):
        D.source_archive(root)
    assert before == {item.name: item.read_bytes() for item in root.iterdir()}


@pytest.mark.parametrize(
    "field", ["archive_sha256", "freeze_sha256", "files", "created_ns"]
)
def test_rejects_metadata_not_bound_to_this_freeze(
    archive_fixture: tuple[Path, dict[str, Any]], field: str
) -> None:
    """Metadata must identify the complete current freeze, not just valid ZIP bytes."""
    root, frozen = archive_fixture
    write_zip(root, frozen)
    value = metadata(root, frozen)
    value[field] = 99 if field == "files" else -1 if field == "created_ns" else "wrong"
    D.I.persist(root / "source_snapshot.json", value)
    before = {item.name: item.read_bytes() for item in root.iterdir()}
    with pytest.raises(RuntimeError, match="source archive"):
        D.source_archive(root)
    assert before == {item.name: item.read_bytes() for item in root.iterdir()}


def test_rejects_orphan_metadata_without_creating_a_new_zip(
    archive_fixture: tuple[Path, dict[str, Any]],
) -> None:
    """An existing provenance record cannot be attached silently to a replacement ZIP."""
    root, _ = archive_fixture
    D.I.persist(root / "source_snapshot.json", {"archive_sha256": "lost"})
    with pytest.raises(RuntimeError, match="source archive"):
        D.source_archive(root)
    assert not (root / "frozen_sources.zip").exists()


def test_rejects_malformed_metadata_without_repairing_it(
    archive_fixture: tuple[Path, dict[str, Any]],
) -> None:
    """Only absent metadata is recoverable automatically; conflicting evidence stays."""
    root, frozen = archive_fixture
    write_zip(root, frozen)
    path = root / "source_snapshot.json"
    path.write_text(json.dumps({"created_ns": 123}))
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="source archive"):
        D.source_archive(root)
    assert path.read_bytes() == before
