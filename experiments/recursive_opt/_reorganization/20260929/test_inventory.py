"""Unit checks for metadata classification and non-destructive inventory."""

from __future__ import annotations

import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "restructure_inventory", Path(__file__).with_name("inventory.py")
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load inventory helper")
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


class InventoryTests(unittest.TestCase):
    """Check role boundaries and preservation of input content."""

    def test_roles(self) -> None:
        """Recognize uppercase docs and separate runtime/cache storage."""
        self.assertEqual(inventory.role(Path("artifacts/REPORT.MD")), "documentation")
        self.assertEqual(
            inventory.role(Path("EXP22/.venv/README.md")), "runtime_or_frozen_checkout"
        )
        self.assertEqual(
            inventory.role(Path("artifacts/__pycache__/a.pyc")), "regenerable_cache"
        )
        self.assertEqual(
            inventory.role(Path("artifacts/result.json.gz")),
            "configuration_or_evidence",
        )

    def test_group(self) -> None:
        """Keep experiment boundaries and root documents explicit."""
        self.assertEqual(
            inventory.group(Path("experiments/recursive_opt/EXP22/results/a.json")),
            "experiments/recursive_opt/EXP22",
        )
        self.assertEqual(
            inventory.group(Path("artifacts/RESEARCH_LOG.md")),
            "artifacts/root_documents",
        )

    def test_inventory_does_not_read_content_or_follow_links(self) -> None:
        """Include metadata for ignored files without copying contents or linked trees."""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "repo"
            output = base / "inventory"
            output.mkdir()
            (root / "artifacts").mkdir(parents=True)
            source = root / "artifacts/raw.json"
            source.write_text("PRIVATE_PAYLOAD_NOT_FOR_METADATA")
            (root / "artifacts/README.MD").write_text("# Example")
            (root / "artifacts/loop").symlink_to(root, target_is_directory=True)
            with (
                patch.object(inventory, "HERE", output),
                patch.object(inventory, "ROOTS", (root,)),
                patch.object(
                    inventory.subprocess,
                    "check_output",
                    return_value=b"artifacts/README.MD\0",
                ),
            ):
                inventory.main()
            with gzip.open(output / "inventory.jsonl.gz", "rt") as stream:
                captured = stream.read()
            rows = [json.loads(line) for line in captured.splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertNotIn("PRIVATE_PAYLOAD_NOT_FOR_METADATA", captured)
            self.assertEqual(source.read_text(), "PRIVATE_PAYLOAD_NOT_FOR_METADATA")
            self.assertEqual(sum(row["symlink"] for row in rows), 1)
            self.assertEqual(sum(row["tracked"] for row in rows), 1)


if __name__ == "__main__":
    unittest.main()
