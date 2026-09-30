"""PRISM white-box evaluation, valid score, feedback, and FallbackWrapper sources (EXP23).

Stock metric (reproduced): 1 / mean(KVPR over solved cases) + success_rate, where format or
constraint violations zero the whole score and exceptions/timeouts skip the case. That metric
rewards crashing on hard cases. Valid score: 1 / mean(KVPR over all 50 cases, a failed case
counted at the initial program's KVPR for that case) + success_rate; honest ceiling 29.40.
"""

import ast
import json
import os
import signal
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
SKY_PRISM = Path('/home/xav/code/evo-compare/repos/skydiscover/benchmarks/ADRS/prism')
INITIAL = SKY_PRISM / 'initial_program.py'
sys.path.insert(0, os.environ.get('TRACE_ROOT', str(Path.home() / 'code' / 'Trace')))
from opto.features.recursive_opt.coevolution import format_case_diagnostics  # noqa: E402

CHECK_SOURCE = '''def check(result, gpu_num, models):
    if not isinstance(result, dict):
        return False
    placed = []
    for gpu_id, assigned in result.items():
        if not isinstance(gpu_id, int) or not 0 <= gpu_id < gpu_num or not isinstance(assigned, list):
            return False
        if sum(m.model_size for m in assigned) > GPU_MEM_SIZE:
            return False
        placed.extend(assigned)
    ids = [id(m) for m in placed]
    return len(ids) == len(models) and len(set(ids)) == len(ids) and set(ids) == {id(m) for m in models}
'''


def fallback_source() -> str:
    """The initial program's compute_model_placement (a feasible baseline on all 50 cases)."""
    tree = ast.parse(INITIAL.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_model_placement')
    return ast.unparse(function)


def projections_config() -> list:
    return [{'ref': 'recursive_opt.projection.compile_check@1', 'config': {'required': ['compute_model_placement']}},
            {'ref': 'recursive_opt.projection.fallback_wrapper@1', 'config': {'entry': 'compute_model_placement', 'fallback_source': fallback_source(), 'check_source': CHECK_SOURCE}}]


def run_cases(source: str, timeout_s: float = 360.0) -> dict:
    with tempfile.TemporaryDirectory(prefix='prism-wb-') as tmp:
        program, out = Path(tmp) / 'program.py', Path(tmp) / 'out.json'
        program.write_text(source)
        env = {k: v for k, v in os.environ.items() if not any(t in k.upper() for t in ('API_KEY', 'TOKEN', 'SECRET', 'PASSWORD'))}
        process = subprocess.Popen([sys.executable, '-I', str(HERE / 'whitebox_worker.py'), str(program), str(out)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, start_new_session=True)
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return {'timeout': True, 'cases': []}
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        return json.loads(out.read_text()) if out.exists() else {'load_error': 'worker produced no result', 'cases': []}


@lru_cache(maxsize=1)
def baseline() -> tuple:
    result = run_cases(INITIAL.read_text())
    return tuple(c['kvpr'] for c in result['cases'])


@lru_cache(maxsize=1)
def bounds() -> tuple:
    """Per-case lower bound on max KVPR: aggregate ratio (mediant) and the largest single model."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('prism_stock_evaluator_lb', SKY_PRISM / 'evaluator' / 'evaluator.py')
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    out = []
    for gpu_num, models in ev.generate_test_gpu_models():
        load, size = sum(m.req_rate / m.slo for m in models), sum(m.model_size for m in models)
        out.append(max(load / (80 * gpu_num - size), max(m.req_rate / m.slo / (80 - m.model_size) for m in models)))
    return tuple(out)


def evaluate(source: str) -> tuple:
    """(metrics, artifacts): stock-equivalent combined_score, valid_score, success_rate, per-case feedback."""
    result = run_cases(source)
    if result.get('timeout') or result.get('load_error'):
        error = 'Program timed out after 360s' if result.get('timeout') else result['load_error']
        return {'validity': 0, 'combined_score': 0.0, 'valid_score': 0.0, 'success_rate': 0.0, 'error': error}, {}
    cases, base, lbs = result['cases'], baseline(), bounds()
    solved = [c for c in cases if c['ok']]
    success = len(solved) / len(cases)
    format_error = next((c for c in cases if c.get('format_error') and 'outside range' not in c['error']), None)
    stock = 0.0 if format_error or not solved else 1.0 / mean(c['kvpr'] for c in solved) + success
    counted = [c['kvpr'] if c['ok'] else base[c['case']] for c in cases]
    valid = 1.0 / mean(counted) + success
    fallbacks = [c for c in cases if c.get('fallback')]
    for c in cases:
        c['bound'] = lbs[c['case']]
        c['value'] = c.get('kvpr')
    metrics = {'combined_score': stock, 'valid_score': valid, 'success_rate': success, 'fallback_cases': len(fallbacks),
               'mean_ratio_to_bound': mean(c['kvpr'] / lbs[c['case']] for c in solved) if solved else 0.0}
    header = (f'Valid score {valid:.4f} (all 50 cases; honest ceiling 29.40), stock score {stock:.4f}, success {success:.2f}. '
              f'{len(fallbacks)} case(s) used the baseline fallback'
              + (f" (causes: {', '.join(sorted({c['fallback'] for c in fallbacks}))})" if fallbacks else '') + '.')
    body = format_case_diagnostics(cases, worst=5, lower_is_better=True, value_key='value', bound_key='bound', detail_keys=('gpus', 'models', 'total_size'))
    return metrics, {'feedback': (header + '\n' + body)[:1900]}
