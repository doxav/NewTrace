"""Offline EXP25 tests. From EXP25: ../EXP22/.venv/bin/python -I -m unittest discover -s tests -v"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

EXP25 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP25 / 'signal'))
import numpy as np  # noqa: E402
import whitebox as W  # noqa: E402


class SignalEvaluatorTests(unittest.TestCase):
    def test_reproduces_stock_scores_on_probe_programs(self) -> None:
        probe = json.loads((EXP25 / 'results' / 'analysis' / 'signal_metric_probe.json').read_text())
        ns = {}
        exec((EXP25 / 'scripts' / 'probe_signal_metric.py').read_text().split('\n\ndef score')[0], ns)
        for name, source in ns['PROGRAMS'].items():
            self.assertAlmostEqual(W.evaluate(source)[0]['combined_score'], probe[name]['combined_score'], places=4, msg=name)

    def test_valid_score_rejects_truncation_and_flags_lookahead(self) -> None:
        ns = {}
        exec((EXP25 / 'scripts' / 'probe_signal_metric.py').read_text().split('\n\ndef score')[0], ns)
        truncated = W.evaluate(ns['PROGRAMS']['truncated output (last 5 samples)'])[0]
        self.assertGreater(truncated['combined_score'], 0.79)
        self.assertLess(truncated['valid_score'], 0.45)
        self.assertEqual(W.evaluate(ns['PROGRAMS']['look-ahead centered smoother (non-causal)'])[0]['causal_fraction'], 0.0)

    def test_fallback_equals_initial_program_output(self) -> None:
        import importlib.util
        spec = importlib.util.spec_from_file_location('initial', W.INITIAL)
        initial = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(initial)
        ns = {}
        exec(W.FALLBACK_SOURCE, ns)
        x = np.random.default_rng(0).normal(size=300)
        np.testing.assert_allclose(ns['fallback'](x, window_size=20)['filtered_signal'], initial.run_signal_processing(x, window_size=20)['filtered_signal'])


class RunnerTests(unittest.TestCase):
    def test_native_arms_end_to_end_offline(self) -> None:
        for arm in ('trace_exp24', 'trace_exp23'):
            with tempfile.TemporaryDirectory() as tmp:
                done = subprocess.run([sys.executable, '-I', str(EXP25 / 'scripts' / 'run_signal.py'), '--arm', arm, '--mock', '--horizon', '3', '--out', tmp],
                                      capture_output=True, text=True, timeout=900)
                self.assertEqual(done.returncode, 0, done.stderr[-2000:])
                summary = json.loads((Path(tmp) / 'summary.json').read_text())
                self.assertEqual((summary['status'], summary['solution_attempts']), ('success', 3))
                self.assertTrue((Path(tmp) / 'evaluations.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
