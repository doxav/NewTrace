"""Offline EXP26 tests. From EXP26: TRACE_ROOT=~/code/Trace ../EXP22/.venv/bin/python -I -m unittest discover -s tests -v"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

EXP26 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP26 / 'scripts'))
os.environ.setdefault('OPENROUTER_API_KEY', 'offline-test')


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f'exp26_{name}', EXP26 / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN = _load('run_signal')
LABELS = {'diverge': 'INJ-DIVERGE use scipy.signal: savgol_filter', 'refine': 'INJ-REFINE tune scipy.signal: lfilter'}


class ArmConfigurationTests(unittest.TestCase):
    def config(self, arm: str) -> dict:
        return RUN.build_spec(arm, 42, 100, offline=True, labels=LABELS)['levels'][0]['engine']['config']

    def test_arms_differ_from_exp25_trace_exp24_only_in_the_label_source(self) -> None:
        base = RUN.E25.build_spec('trace_exp24', 42, 100, offline=True)['levels'][0]['engine']['config']
        for arm, changed in (('native', set()), ('native_pkg', {'label_packages'}), ('native_stocklabels', {'labels'})):
            config = self.config(arm)
            self.assertEqual({k for k in set(base) | set(config) if base.get(k) != config.get(k)}, changed, arm)

    def test_native_pkg_uses_stock_package_list(self) -> None:
        from skydiscover.optimize.search.evox.utils.variation_operator_generator import get_available_packages
        packages = self.config('native_pkg')['label_packages']
        self.assertEqual(packages, get_available_packages(problem_dir=str(RUN.W.SKY / 'evaluator')))
        self.assertTrue(any(p.startswith('scipy') for p in packages))

    def test_stocklabels_arm_requires_labels(self) -> None:
        with self.assertRaises(ValueError):
            RUN.build_spec('native_stocklabels', 42, 100, offline=True, labels=None)


class RunnerTests(unittest.TestCase):
    def run_arm(self, arm: str, tmp: str) -> Path:
        out, extra = Path(tmp) / arm, []
        if arm == 'native_stocklabels':
            (Path(tmp) / 'labels.json').write_text(json.dumps(LABELS))
            extra = ['--labels', str(Path(tmp) / 'labels.json')]
        done = subprocess.run([sys.executable, '-I', str(EXP26 / 'scripts' / 'run_signal.py'), '--arm', arm, '--mock', '--horizon', '3',
                               '--out', str(out), *extra], capture_output=True, text=True, timeout=900, env={**os.environ})
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        return out

    def test_every_arm_runs_and_transcribes_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for arm in RUN.ARMS:
                out = self.run_arm(arm, tmp)
                summary = json.loads((out / 'summary.json').read_text())
                self.assertEqual((summary['status'], summary['solution_attempts']), ('success', 3), arm)
                rows = [json.loads(line) for line in (out / 'transcripts.jsonl').read_text().splitlines()]
                self.assertTrue(rows and all(r['messages'] and 'response' in r for r in rows), arm)
                label_calls = [r for r in rows if 'two reusable instruction blocks' in json.dumps(r['messages'])]
                prompt = json.dumps(label_calls[0]['messages']) if label_calls else ''
                if arm == 'native_stocklabels':
                    self.assertEqual(label_calls, [])
                    self.assertEqual(json.loads((out / 'labels.json').read_text()), LABELS)
                else:
                    self.assertEqual(len(label_calls), 1, arm)
                    self.assertEqual('Available Packages in Environment' in prompt, arm == 'native_pkg', arm)


class StockAndAnalysisTests(unittest.TestCase):
    def test_stock_scripts_start_under_isolated_mode(self) -> None:
        """The campaign launches every script with `python -I`; an import that only works with the script directory
        on sys.path must fail here, not at launch."""
        for script in ('gen_stock_labels.py', 'run_evox_stock.py'):
            done = subprocess.run([sys.executable, '-I', str(EXP26 / 'scripts' / script), '--help'], capture_output=True, text=True, timeout=300,
                                  env={**os.environ})
            self.assertEqual(done.returncode, 0, f'{script}: {done.stderr[-1500:]}')

    def test_silent_fallback_is_detected(self) -> None:
        stock = _load('stock')
        from skydiscover.optimize.search.evox.utils.template import DEFAULT_DIVERGE_TEMPLATE, DEFAULT_REFINE_TEMPLATE

        class Controller:
            def __init__(self, d, r):
                self._diverge_label, self._refine_label = d, r
        self.assertFalse(stock.labels_of(Controller('d', 'r'))['fallback'])
        self.assertTrue(stock.labels_of(Controller('', 'r'))['fallback'])
        self.assertTrue(stock.labels_of(Controller(DEFAULT_DIVERGE_TEMPLATE, DEFAULT_REFINE_TEMPLATE))['fallback'])

    def test_decision_rules(self) -> None:
        analyze = _load('analyze')

        def row(p1, p2, scipy, diverge):
            return {'p1_best_valid': p1, 'p2_best_valid_causal': p2, 'scipy_share': scipy, 'label_shares': {'diverge': diverge}}
        arms = {'evox_stock': [row(0.71, 0.54, 0.7, None)] * 3, 'native': [row(0.58, 0.53, 0.0, 0.12)] * 3,
                'native_pkg': [row(0.69, 0.54, 0.4, 0.15)] * 3, 'native_stocklabels': [row(0.70, 0.53, 0.5, 0.15)] * 3}
        rules = analyze.decide(arms)['rules']
        self.assertEqual({k: rules[k] for k in rules}, {'gate_evox_reproduces_exp25': True, 'R1_label_content_explains_gap': True,
                                                         'R2_compact_port_adequate': True, 'R3_selection_implicated': False,
                                                         'R4_gap_elsewhere': False, 'R5_lead_is_lookahead_only': True})
        arms['native_stocklabels'] = [row(0.59, 0.53, 0.0, 0.08)] * 3
        rules = analyze.decide(arms)['rules']
        self.assertEqual((rules['R1_label_content_explains_gap'], rules['R3_selection_implicated'], rules['R4_gap_elsewhere']), (False, True, False))


if __name__ == '__main__':
    unittest.main()
