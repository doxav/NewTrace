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

    def test_summary_role_requires_exact_stock_prompt(self) -> None:
        """Resolve stock guide calls without changing raw evidence or guessing roles."""
        request = {'role': 'trace_meta_or_preflight', 'outbound_body': {'messages': [{'role': 'system', 'content': 'Known guide prompt'}]}}
        rows, evidence = analyze.resolve_roles([request], {'stock-template.txt': 'Known guide prompt'})
        self.assertEqual(rows[0]['role'], 'guide')
        self.assertEqual(evidence[0]['source_template'], 'stock-template.txt')
        self.assertEqual(request['role'], 'trace_meta_or_preflight')
        with self.assertRaisesRegex(ValueError, 'Unclassified'):
            analyze.resolve_roles([request], {'stock-template.txt': 'Different prompt'})

    def test_strict_trajectory_metrics_and_failures(self) -> None:
        """Verify known AUC, improvement timing and rejection of corrupted evidence."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            curve = [{'iteration': i, 'best_score': 1 if i < 4 else 3} for i in range(1, 101)]
            path = root/'solution_curve.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in curve))
            result = {'passed': True, 'initial_score': 1, 'final_best_score': 3}
            requests = [{'role': 'solution', 'usage': {'cost': 0.01}} for _ in curve]
            measured = analyze.strict_metrics(root, result, requests)
            self.assertEqual(measured['auc_100'], 294)
            self.assertEqual(measured['first_improvement_iteration'], 4)
            self.assertEqual(measured['best_solution_iteration'], 4)
            self.assertEqual(measured['relative_gain'], 2)
            self.assertIsNone(measured['policy_validation_failure_rate'])
            partial = analyze.strict_metrics(root, {**result, 'passed': False}, requests)
            self.assertIsNone(partial['auc_100'])
            with self.assertRaisesRegex(ValueError, '100 observed'):
                analyze.strict_metrics(root, result, requests[:-1])
            curve[-1]['best_score'] = 2
            path.write_text(''.join(json.dumps(row)+'\n' for row in curve))
            with self.assertRaisesRegex(ValueError, 'decreases'):
                analyze.strict_metrics(root, result, requests)

    def test_advanced_requires_complete_matrix_and_deployment(self) -> None:
        """Positive pilot or incomplete strict results cannot authorize exploration."""
        attempts = []
        for task in ('prism', 'signal_processing'):
            for arm in ('SD-EVOX', 'SD-FIXED', 'TRACE-RECURSIVE', 'TRACE-FIXED'):
                score = 2 if arm == 'TRACE-RECURSIVE' else 1
                attempts.append({'config': {'stage': 'strict', 'task': task, 'arm': arm}, 'metrics': {'complete': True, 'final_best_score': score, 'relative_gain': score-1, 'auc_100': score*100, 'policy_switches': 1 if arm == 'TRACE-RECURSIVE' else 0}})
        self.assertTrue(analyze.comparisons(attempts)['advanced_numerically_eligible'])
        self.assertFalse(analyze.comparisons(attempts[:-1])['advanced_numerically_eligible'])
        for row in attempts:
            row['metrics']['policy_switches'] = 0
        self.assertFalse(analyze.comparisons(attempts)['advanced_numerically_eligible'])

    def test_rejected_cost_is_unknown(self) -> None:
        """Absent usage is not evidence that a rejected request was free."""
        result = usage_summary([{'role': 'solution', 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'cost': 0.01}}, {'role': 'guide', 'usage': None}])
        self.assertEqual(result['total_calls'], 2)
        self.assertEqual(result['input_tokens'], 10)
        self.assertEqual(result['output_tokens'], 5)
        self.assertEqual(result['reported_cost'], 0.01)
        self.assertEqual(result['calls_missing_cost'], 1)

    def test_usage_curve_does_not_include_future_retry(self) -> None:
        """A failed attempt's cumulative cost stops before the later successful retry."""
        rows = [{'role': 'guide', 'usage': {'cost': 0.01}}, {'role': 'solution', 'usage': {'cost': 0.02}}, {'role': 'solution', 'usage': {'cost': 0.03}}]
        first = usage_summary(rows, solution_limit=1)
        self.assertEqual(first['total_calls'], 2)
        self.assertAlmostEqual(first['reported_cost'], 0.03)
        self.assertAlmostEqual(usage_summary(rows, solution_limit=2)['reported_cost'], 0.06)
        for limit in (0, True, 3):
            with self.assertRaises(ValueError):
                usage_summary(rows, solution_limit=limit)

    def test_analysis_separates_provider_series(self) -> None:
        """Historical runs must not affect current-route counts or outcomes."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'artifacts').mkdir()
            (root/'runs/old').mkdir(parents=True)
            (root/'runs/old/openrouter_config_sanitized.json').write_text(json.dumps({'provider': {'only': ['DeepInfra']}}))
            (root/'runs/old_default').mkdir()
            (root/'runs/old_default/openrouter_config_sanitized.json').write_text(json.dumps({'provider': {'only': ['novita']}}))
            row = {'outbound_body': {'provider': {'only': ['novita']}, 'reasoning_effort': 'low'}, 'passed': True, 'http_status': 200, 'usage': {'total_tokens': 4, 'cost': 0.01}}
            (root/'artifacts/openrouter_transport_validation.json').write_text(json.dumps({'direct': row, 'trace': {'primary_variant': 'CP-A'}}))
            (root/'artifacts/gates.json').write_text('{}')
            (root/'manifest.json').write_text('{}')
            with patch.object(analyze, 'ROOT', root):
                report = analyze.analyze()
                self.assertEqual(report['completion_requests'], 1)
                self.assertEqual(report['attempts'], [])
                self.assertEqual(report['status'], 'PARTIAL')
                (root/'runs/current').mkdir()
                (root/'runs/current/openrouter_config_sanitized.json').write_text(json.dumps({'provider': {'only': ['novita']}, 'reasoning_effort': 'low'}))
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
            (root/'artifacts/openrouter_transport_validation.json').write_text(json.dumps({'passed': True, 'direct': {'outbound_body': {'model': run_stage.MODEL, **run_stage.EXTRA_BODY}}}))
            (root/'artifacts/gates.json').write_text(json.dumps({f'S{i}': True for i in range(6)}))
            (root/'artifacts/strict_source_hashes.json').write_text('{}')
            (root/'scripts').mkdir()
            for name in ('run_stage.py', 'framework_smoke.py', 'preflight.py'):
                (root/'scripts'/name).write_text('# Changed source\n')
            with patch.object(run_stage, 'ROOT', root), patch('sys.argv', ['run_stage', '--stage', 'strict', '--task', 'prism', '--arm', 'SD-EVOX']), self.assertRaisesRegex(ValueError, 'source differs'):
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
