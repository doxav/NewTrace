"""Check diagnostic metering and publication redaction without network calls."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_stage import redact_diagnostics, usage_summary


class AccountingTests(unittest.TestCase):
    """Keep refused calls distinct from measured usage and preserve scientific data."""

    def test_rejected_cost_is_unknown(self) -> None:
        """Absent usage is not evidence that a rejected request was free."""
        result = usage_summary([{'role': 'solution', 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'cost': 0.01}}, {'role': 'guide', 'usage': None}])
        self.assertEqual(result['total_calls'], 2)
        self.assertEqual(result['input_tokens'], 10)
        self.assertEqual(result['output_tokens'], 5)
        self.assertEqual(result['reported_cost'], 0.01)
        self.assertEqual(result['calls_missing_cost'], 1)

    def test_account_identifier_redaction_preserves_failure(self) -> None:
        """Redaction hides account IDs without changing error codes or candidate data."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            account = 'user_' + 'a' * 30
            (directory/'stderr.log').write_text(f'429 engine_overloaded {account}')
            (directory/'candidate_history.jsonl').write_text(json.dumps({'error': f'429 {account}', 'candidate': None}) + '\n')
            redact_diagnostics(directory)
            self.assertNotIn(account, (directory/'stderr.log').read_text())
            row = json.loads((directory/'candidate_history.jsonl').read_text())
            self.assertIn('429', row['error'])
            self.assertIsNone(row['candidate'])
