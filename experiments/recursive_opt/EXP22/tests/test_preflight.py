"""Verify EXP22 provenance rejection and secret-safe evidence handling."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/preflight.py"
SPEC = importlib.util.spec_from_file_location("exp22_preflight", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


class PreflightTests(unittest.TestCase):
    """Cover legitimate identity, substitutions, and unsafe persistence."""

    def test_exact_required_branch_passes(self) -> None:
        """An exact checked-out required branch passes the identity gate."""
        self.assertTrue(PREFLIGHT.branch_gate("recursive_opt", "a", "a")["passed"])

    def test_substitutions_and_missing_ref_fail(self) -> None:
        """Reject mismatched names, revisions, detached HEAD and absent refs."""
        for branch, head, required in (
            ("other", "a", "a"), ("", "a", "a"),
            ("recursive_opt", "a", "b"), ("recursive_opt", "a", None),
        ):
            with self.subTest(branch=branch, required=required):
                self.assertFalse(PREFLIGHT.branch_gate(branch, head, required)["passed"])

    def test_secret_rejected_before_write(self) -> None:
        """Evidence must not persist API credentials, even accidentally."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            with self.assertRaisesRegex(ValueError, "Credential-like"):
                PREFLIGHT.write_json(path, {"value": "".join(("sk-or-v1-", "x" * 64))})
            self.assertFalse(path.exists())

    def test_recursive_scan_and_safe_json(self) -> None:
        """Scan nested files and report paths without disclosing matches."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            PREFLIGHT.write_json(root / "safe.json", {"model": "z-ai/glm-5.3-flash"})
            self.assertEqual(PREFLIGHT.secret_scan(root), [])
            (root / "nested").mkdir()
            (root / "nested/leak.txt").write_text("".join(("sk-", "x" * 30)))
            self.assertEqual(PREFLIGHT.secret_scan(root), ["nested/leak.txt"])

    def test_git_failure_is_descriptive(self) -> None:
        """Invalid repositories fail without leaking raw subprocess output."""
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "Required Git query failed"),
        ):
            PREFLIGHT.git(Path(directory), "rev-parse", "HEAD")


if __name__ == "__main__":
    unittest.main()
