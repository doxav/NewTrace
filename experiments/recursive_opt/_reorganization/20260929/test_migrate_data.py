"""Check migration preservation, refusal of collisions, and rollback."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "migrate_data", Path(__file__).with_name("migrate_data.py")
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load migration helper")
migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migration)


class MigrationTests(unittest.TestCase):
    """Exercise filesystem behavior without touching scientific stores."""

    def test_atomic_alias_preserves_open_writer(self) -> None:
        """An existing handle and subsequent legacy-path writes reach the same data."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / "old", root / "new"
            source.mkdir()
            with (source / "events.log").open("w") as stream:
                stream.write("before\n")
                stream.flush()
                record = migration.relocate_with_atomic_alias(source, target)
                stream.write("after\n")
            with (source / "events.log").open("a") as stream:
                stream.write("legacy\n")
            self.assertEqual(
                (target / "events.log").read_text(), "before\nafter\nlegacy\n"
            )
            self.assertEqual(target.stat().st_ino, record["inode"])
            self.assertEqual(source.resolve(), target.resolve())

    def test_atomic_alias_refuses_collision(self) -> None:
        """Never exchange an existing destination owned by another operation."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / "old", root / "new"
            source.mkdir()
            target.mkdir()
            with self.assertRaises(ValueError):
                migration.relocate_with_atomic_alias(source, target)
            self.assertFalse(source.is_symlink())

    def test_directory_preservation_and_legacy_read(self) -> None:
        """Preserve data and make the old path resolve to its new location."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / "old", root / "new/results"
            source.mkdir()
            (source / "result.json").write_bytes(b'{"score": 0.5}')
            before = migration.digest_tree(source)
            result = migration.relocate(source, target, root / "ledger.gz")
            self.assertEqual(result["status"], "verified")
            self.assertTrue(source.is_symlink())
            self.assertEqual(source.resolve(), target.resolve())
            self.assertEqual(migration.digest_tree(target), before)
            self.assertEqual((source / "result.json").read_bytes(), b'{"score": 0.5}')

    def test_refuses_existing_destination(self) -> None:
        """Do not overwrite an existing target or its contents."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / "old", root / "new"
            source.write_text("old")
            target.write_text("keep")
            with self.assertRaises(ValueError):
                migration.relocate(source, target, root / "ledger.gz")
            self.assertEqual(target.read_text(), "keep")
            self.assertEqual(source.read_text(), "old")

    def test_verification_failure_rolls_back(self) -> None:
        """Restore the original location if the post-move inventory differs."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / "old", root / "new"
            source.write_text("retained")
            with (
                patch.object(
                    migration,
                    "digest_tree",
                    side_effect=[{".": "first"}, {".": "different"}],
                ),
                self.assertRaises(RuntimeError),
            ):
                migration.relocate(source, target, root / "ledger.gz")
            self.assertFalse(source.is_symlink())
            self.assertEqual(source.read_text(), "retained")
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
