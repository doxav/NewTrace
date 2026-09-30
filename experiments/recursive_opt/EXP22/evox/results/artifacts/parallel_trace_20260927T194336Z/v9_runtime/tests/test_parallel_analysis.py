"""Verify parallel reports distinguish live evidence from completed comparisons."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_parallel, plot_results


class ParallelAnalysisTests(unittest.TestCase):
    """Exercise fresh four-worker manifests without network or model calls."""

    def manifest(self, root: Path) -> Path:
        """Create the four required worker entries before run directories exist."""
        path = root/'manifest.json'
        workers = [{'task': task, 'arm': arm, 'run_directory': None, 'pid': 100 + index,
                    'returncode': None} for index, (task, arm) in enumerate(
                        (task, arm) for task in analyze_parallel.TASKS for arm in analyze_parallel.ARMS)]
        path.write_text(json.dumps({'workers': workers, 'status': 'running', 'started_at_utc': '2026-09-27T00:00:00Z'}))
        return path

    def populate(self, root: Path, worker: dict[str, Any], count: int, score: float, sealed: bool) -> None:
        """Write known trajectories and optional final HTTP accounting evidence."""
        directory = root/f"{worker['task']}_{worker['arm']}"
        directory.mkdir()
        worker['run_directory'] = str(directory)
        (directory/'config.json').write_text(json.dumps({**worker, 'stage': 'strict', 'horizon': 100}))
        curve = [{'iteration': index, 'best_score': score, 'valid_candidates': index,
                  'invalid_candidates': 0, 'llm_usage': {'roles': {'solution': index}, 'reported_cost': index*0.01,
                                                        'calls_missing_cost': 0}}
                 for index in range(1, count + 1)]
        (directory/'solution_curve.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in curve))
        if sealed:
            result = {'passed': count == 100, 'initial_score': 1.0, 'final_best_score': score,
                      'valid_candidates': count, 'invalid_attempts': 0,
                      'final_metrics': {'combined_score': score, 'success_rate': 1.0}, 'wall_time': 25.0}
            (directory/'final_result.json').write_text(json.dumps(result))
            (directory/'http_requests.json').write_text(json.dumps([{'role': 'solution', 'usage': {'cost': 0.01}} for _ in curve]))
            worker['returncode'] = 0 if count == 100 else 2

    def test_complete_pairs_have_exact_metrics_and_isolated_reports(self) -> None:
        """Compute known gains and AUC while preserving the preexisting global report."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.manifest(root)
            manifest = json.loads(path.read_text())
            for worker in manifest['workers']:
                self.populate(root, worker, 100, 3.0 if worker['arm'] == 'TRACE-RECURSIVE' else 2.0, True)
            manifest['status'] = 'complete'
            path.write_text(json.dumps(manifest))
            report = analyze_parallel.summarize(path)
            self.assertTrue(report['all_four_complete'])
            self.assertEqual(report['completed_contrasts_recursive_minus_fixed']['prism'],
                             {'final_best_score': 1.0, 'absolute_gain': 1.0, 'relative_gain': 1.0, 'auc_100': 100.0})
            (root/'RESULTS.md').write_text('Existing report stays intact')
            analyze_parallel.write_report(report, root)
            self.assertEqual((root/'RESULTS.md').read_text(), 'Existing report stays intact')
            self.assertTrue(json.loads((root/'results.json').read_text())['all_four_complete'])
            self.assertIn('One stochastic run', (root/'results.md').read_text())

    def test_live_snapshot_preserves_unknown_metrics_and_usage_boundary(self) -> None:
        """Live scores are measured but missing initial/native metrics are never invented."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.manifest(root)
            manifest = json.loads(path.read_text())
            self.populate(root, manifest['workers'][0], 4, 2.0, False)
            path.write_text(json.dumps(manifest))
            report = analyze_parallel.summarize(path)
            live = report['attempts'][0]
            self.assertEqual(live['metrics']['observed_attempts'], 4)
            self.assertEqual(live['metrics']['usage']['roles']['solution'], 4)
            self.assertEqual(live['result']['valid_candidates'], 4)
            self.assertIsNone(live['metrics']['absolute_gain'])
            self.assertIsNone(live['metrics']['auc_100'])
            self.assertEqual(live['metrics']['native_metrics'], {})
            self.assertFalse(report['completed_contrasts_recursive_minus_fixed'])
            self.assertEqual(report['attempts'][1]['state'], 'starting')

    def test_failed_or_unexited_worker_cannot_complete_pair(self) -> None:
        """Final files do not establish success before the worker exits successfully."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.manifest(root)
            manifest = json.loads(path.read_text())
            for worker in manifest['workers'][:2]:
                self.populate(root, worker, 100, 2.0, True)
            for returncode in (None, 2):
                manifest['workers'][0]['returncode'] = returncode
                path.write_text(json.dumps(manifest))
                report = analyze_parallel.summarize(path)
                self.assertFalse(report['completed_contrasts_recursive_minus_fixed'])
                self.assertFalse(report['attempts'][0]['complete'])

    def test_partial_sealed_result_has_no_full_auc(self) -> None:
        """An interrupted worker retains observed evidence but cannot enter contrasts."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.manifest(root)
            manifest = json.loads(path.read_text())
            self.populate(root, manifest['workers'][0], 21, 2.0, True)
            path.write_text(json.dumps(manifest))
            report = analyze_parallel.summarize(path)
            self.assertEqual(report['attempts'][0]['metrics']['observed_auc'], 42.0)
            self.assertIsNone(report['attempts'][0]['metrics']['auc_100'])
            self.assertFalse(report['completed_contrasts_recursive_minus_fixed'])

    def test_invalid_manifest_and_mismatched_run_fail(self) -> None:
        """Reject duplicate arms and directories belonging to another experiment."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.manifest(root)
            manifest = json.loads(path.read_text())
            manifest['workers'][-1] = dict(manifest['workers'][0])
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'four distinct'):
                analyze_parallel.summarize(path)
            path = self.manifest(root)
            manifest = json.loads(path.read_text())
            worker = manifest['workers'][0]
            self.populate(root, worker, 0, 1.0, False)
            (Path(worker['run_directory'])/'config.json').write_text(json.dumps({'task': 'wrong'}))
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'configuration'):
                analyze_parallel.summarize(path)

    def test_plot_adapter_scopes_outputs_and_preserves_partial_results(self) -> None:
        """Parallel plotting uses only its own batch and rejects unsealed snapshots."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.manifest(root)
            with self.assertRaisesRegex(ValueError, 'sealed final results'):
                plot_results.load_plot_input(path)
            manifest = json.loads(path.read_text())
            for index, worker in enumerate(manifest['workers']):
                self.populate(root, worker, 21 if index == 0 else 100, 2.0, True)
            path.write_text(json.dumps(manifest))
            report, output = plot_results.load_plot_input(path)
            self.assertEqual(output, root)
            self.assertTrue(report['parallel'])
            self.assertFalse(report['attempts'][0]['metrics']['complete'])
            self.assertEqual(len(report['attempts'][0]['metrics']['curve']), 21)
            self.assertEqual(report['attempts'][1]['config']['stage'], 'strict')
            self.assertTrue(report['attempts'][1]['metrics']['complete'])

    def test_default_plot_input_preserves_historical_paths(self) -> None:
        """Omitting the new option retains the existing report and artifact directory."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'artifacts').mkdir()
            expected = {'status': 'historical', 'attempts': []}
            (root/'artifacts/diagnostic_summary.json').write_text(json.dumps(expected))
            with patch.object(plot_results, 'ROOT', root):
                report, output = plot_results.load_plot_input(None)
            self.assertEqual(report, expected)
            self.assertEqual(output, root/'artifacts')


if __name__ == '__main__':
    unittest.main()
