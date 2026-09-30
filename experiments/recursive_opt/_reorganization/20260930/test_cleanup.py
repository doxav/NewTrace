"""Exercise evidence preservation and refusal of destructive collisions."""

import tempfile
import unittest
from pathlib import Path

from cleanup import mapped, relocate


class CleanupTests(unittest.TestCase):
    def test_move_preserves_file_and_symlink_bytes(self) -> None:
        """Move one tree without traversing its linked evidence store."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "old"
            source.mkdir()
            (source / "data").write_bytes(b"evidence\0")
            (source / "alias").symlink_to("data")
            relocate(source, root / "new", root / "receipt.json")
            self.assertFalse(source.exists())
            self.assertEqual((root / "new/data").read_bytes(), b"evidence\0")
            self.assertEqual((root / "new/alias").readlink(), Path("data"))

    def test_merge_keeps_existing_index(self) -> None:
        """An existing navigation entry survives a noncolliding merge."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("old", "new"):
                (root / name).mkdir()
            (root / "old/data").write_text("original")
            (root / "new/README.md").write_text("index")
            relocate(root / "old", root / "new", root / "receipt.json")
            self.assertEqual((root / "new/README.md").read_text(), "index")

    def test_collision_is_refused_before_changes(self) -> None:
        """Never overwrite either version when destination names conflict."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("old", "new"):
                (root / name).mkdir()
                (root / name / "data").write_text(name)
            with self.assertRaisesRegex(ValueError, "source name"):
                relocate(root / "old", root / "new", root / "receipt.json")
            self.assertEqual((root / "old/data").read_text(), "old")
            self.assertEqual((root / "new/data").read_text(), "new")

    def test_mapping_uses_specific_prefix(self) -> None:
        """Study-level overrides take precedence over the shared directory map."""
        self.assertEqual(
            mapped(
                Path("/old/study/a"),
                [
                    (Path("/old"), Path("/new")),
                    (Path("/old/study"), Path("/experiment")),
                ],
            ),
            Path("/experiment/a"),
        )
