"""Check diagnostic metering and publication redaction without network calls."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze, run_stage
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

    def test_analysis_separates_provider_series(self) -> None:
        """Historical runs must not affect current-route counts or outcomes."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'artifacts').mkdir()
            (root/'runs/old').mkdir(parents=True)
            (root/'runs/old/openrouter_config_sanitized.json').write_text(json.dumps({'provider': {'only': ['DeepInfra']}}))
            row = {'outbound_body': {'provider': {'only': ['novita']}}, 'passed': True, 'http_status': 200, 'usage': {'total_tokens': 4, 'cost': 0.01}}
            (root/'artifacts/openrouter_transport_validation.json').write_text(json.dumps({'direct': row, 'trace': {'primary_variant': 'CP-A'}}))
            (root/'artifacts/gates.json').write_text('{}')
            (root/'manifest.json').write_text('{}')
            with patch.object(analyze, 'ROOT', root):
                report = analyze.analyze()
                self.assertEqual(report['completion_requests'], 1)
                self.assertEqual(report['attempts'], [])
                self.assertEqual(report['status'], 'PARTIAL')
                (root/'runs/current').mkdir()
                (root/'runs/current/openrouter_config_sanitized.json').write_text(json.dumps({'provider': {'only': ['novita']}}))
                with self.assertRaisesRegex(ValueError, 'unfinished'):
                    analyze.analyze()
                current = root/'runs/current'
                (current/'config.json').write_text(json.dumps({'stage': 'one', 'task': 'prism', 'arm': 'SD-EVOX'}))
                (current/'final_result.json').write_text(json.dumps({'passed': False, 'valid_candidates': 0, 'iterations_observed': 1, 'final_best_score': 1.0}))
                (current/'http_requests.json').write_text(json.dumps([row]))
                (current/'candidate_history.jsonl').write_text(json.dumps({'error': 'LLM returned None response'}) + '\n')
                report = analyze.analyze()
                self.assertEqual(report['status'], 'STOPPED_PROVIDER')
                self.assertEqual(report['completion_requests'], 2)
                self.assertEqual(report['valid_generated_candidates'], 0)
                self.assertTrue((root/'artifacts/STOP.json').exists())

    def test_old_transport_gate_cannot_authorize_new_provider(self) -> None:
        """A previously passed DeepInfra gate cannot enable a Novita run."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'artifacts').mkdir()
            (root/'artifacts/evaluator_parity.json').write_text('{"passed": true}')
            (root/'artifacts/openrouter_transport_validation.json').write_text(json.dumps({'passed': True, 'direct': {'outbound_body': {'provider': {'only': ['DeepInfra']}}}}))
            with patch.object(run_stage, 'ROOT', root), patch('sys.argv', ['run_stage', '--stage', 'one', '--task', 'prism', '--arm', 'SD-EVOX']), self.assertRaisesRegex(ValueError, 'different routing identity'):
                run_stage.main()
            self.assertFalse((root/'runs').exists())

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
