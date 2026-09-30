"""Offline EXP24 tests (no network). Run from EXP24 with the EXP22 venv:
    ../EXP22/.venv/bin/python -I -m unittest discover -s tests -v
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

EXP24 = Path(__file__).resolve().parents[1]
ROOT = EXP24.parent
sys.path.insert(0, str(EXP24 / 'prism'))
import whitebox as W  # noqa: E402

from opto.features.recursive_opt.coevolution import make_projection  # noqa: E402

HACK_TRACE = ROOT / 'EXP23' / 'results' / 'prism100_v2' / '20260929T215903' / 'trace' / 'best_program.py'
HACK_EVOX = ROOT / 'EXP22' / 'runs' / 'strict_prism_SD-EVOX_20260926T123925.700329Z' / 'best_solution.py'


def stock_score(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location('prism_stock', W.SKY_PRISM / 'evaluator' / 'evaluator.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with redirect_stdout(StringIO()):
        return module.evaluate(str(path))


def project(source: str) -> str:
    for item in W.projections_config():
        source, _ = make_projection(item)(source)
    return source


class WhiteboxTests(unittest.TestCase):
    def test_reproduces_stock_metric_exactly(self) -> None:
        for path in (W.INITIAL, HACK_TRACE, HACK_EVOX):
            metrics, _ = W.evaluate(path.read_text())
            self.assertAlmostEqual(metrics['combined_score'], stock_score(path)['combined_score'], places=9, msg=str(path))

    def test_valid_score_penalizes_metric_exploits(self) -> None:
        initial = W.evaluate(W.INITIAL.read_text())[0]['valid_score']
        for path in (HACK_TRACE, HACK_EVOX):
            metrics, _ = W.evaluate(path.read_text())
            self.assertGreater(metrics['combined_score'], 29.4)   # above the honest ceiling: exploit
            self.assertLess(metrics['valid_score'], initial)     # worse than the initial program on all 50 cases
            projected, artifacts = W.evaluate(project(path.read_text()))
            self.assertEqual(projected['success_rate'], 1.0)
            self.assertGreater(projected['fallback_cases'], 40)
            self.assertIn('ZeroDivisionError', artifacts['feedback'])

    def test_check_rejects_gpu_ids_outside_range(self) -> None:
        namespace = {'GPU_MEM_SIZE': 80}
        exec(W.CHECK_SOURCE, namespace)

        class Model:
            def __init__(self, size):
                self.model_size = size
        models = [Model(10), Model(10)]
        self.assertTrue(namespace['check']({0: [models[0]], 1: [models[1]]}, 2, models))
        self.assertFalse(namespace['check']({0: [models[0]], 2: [models[1]]}, 2, models))   # GPU 2 does not exist
        self.assertFalse(namespace['check']({0: [models[0]]}, 2, models))                  # a model is missing


class ExactOptimumTests(unittest.TestCase):
    def test_saved_optimum_and_spot_recomputation(self) -> None:
        saved = json.loads((EXP24 / 'results' / 'analysis' / 'prism_exact_optimum.json').read_text())
        self.assertAlmostEqual(saved['exact_optimal_score'], 26.2559717495, places=8)
        spec = importlib.util.spec_from_file_location('prism_exact', EXP24 / 'scripts' / 'prism_exact.py')
        exact = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exact)
        cases = exact.ev.generate_test_gpu_models()
        for index in (0, 22, 27):
            self.assertAlmostEqual(exact.optimum(*cases[index]), saved['per_case'][index], places=9)


class RunnerTests(unittest.TestCase):
    def test_all_arms_end_to_end_offline(self) -> None:
        for arm in ('fixed', 'llm_rewrite', 'trace'):
            with tempfile.TemporaryDirectory() as tmp:
                done = subprocess.run([sys.executable, '-I', str(EXP24 / 'scripts' / 'run_prism.py'), '--arm', arm, '--mock', '--horizon', '3', '--out', tmp],
                                      capture_output=True, text=True, timeout=900)
                self.assertEqual(done.returncode, 0, done.stderr[-2000:])
                summary = json.loads((Path(tmp) / 'summary.json').read_text())
                manifest = json.loads((Path(tmp) / 'run_manifest.json').read_text())
                self.assertEqual(summary['status'], 'success', summary)
                self.assertEqual(summary['solution_attempts'], 3)
                self.assertEqual(summary['best_metrics']['success_rate'], 1.0)
                self.assertIn('engine.py', manifest['coevolution_sha256'])


if __name__ == '__main__':
    unittest.main()
