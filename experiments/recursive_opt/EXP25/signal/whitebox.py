"""Signal Processing: stock metric (reproduced), valid score, per-signal feedback, fallback sources (EXP25).

Stock metric: skips failed signals and accepts any non-empty output length, so a 5-sample output scores ~0.80.
Valid score: the stock aggregation over all 5 signals, where a signal counts only if its output has the documented
length len(x) - window_size + 1 and finite values; an invalid or failed signal contributes the initial program's
per-signal metrics, and success_rate counts valid signals. Causality (look-ahead) is measured, not enforced.
"""

import json
import os
import signal as os_signal
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SKY = Path('/home/xav/code/evo-compare/repos/skydiscover/benchmarks/math/signal_processing')
INITIAL = SKY / 'initial_program.py'
sys.path.insert(0, os.environ.get('TRACE_ROOT', str(Path.home() / 'code' / 'Trace')))
from opto.features.recursive_opt.coevolution import format_case_diagnostics  # noqa: E402

FALLBACK_SOURCE = '''def fallback(noisy_signal=None, signal_length=1000, noise_level=0.3, window_size=20):
    import numpy as np
    x = np.asarray(noisy_signal, dtype=float)
    weights = np.exp(np.linspace(-2, 0, window_size))
    weights = weights / np.sum(weights)
    y = np.array([np.sum(x[i:i + window_size] * weights) for i in range(len(x) - window_size + 1)])
    return {"filtered_signal": y, "clean_signal": None, "noisy_signal": None, "correlation": 0, "noise_reduction": 0, "signal_length": len(y)}
'''
CHECK_SOURCE = '''def check(result, noisy_signal=None, signal_length=1000, noise_level=0.3, window_size=20):
    import numpy as np
    if not isinstance(result, dict) or "filtered_signal" not in result:
        return False
    y = np.asarray(result["filtered_signal"], dtype=float)
    return y.ndim == 1 and len(y) == len(noisy_signal) - window_size + 1 and bool(np.all(np.isfinite(y)))
'''
PROJECTIONS = [{'ref': 'recursive_opt.projection.compile_check@1', 'config': {'required': ['run_signal_processing']}},
               {'ref': 'recursive_opt.projection.fallback_wrapper@1', 'config': {'entry': 'run_signal_processing', 'fallback_source': FALLBACK_SOURCE, 'check_source': CHECK_SOURCE}}]


def run_signals(source: str, timeout_s: float = 360.0) -> dict:
    with tempfile.TemporaryDirectory(prefix='signal-wb-') as tmp:
        program, out = Path(tmp) / 'program.py', Path(tmp) / 'out.json'
        program.write_text(source)
        env = {k: v for k, v in os.environ.items() if not any(t in k.upper() for t in ('API_KEY', 'TOKEN', 'SECRET', 'PASSWORD'))}
        process = subprocess.Popen([sys.executable, '-I', str(HERE / 'whitebox_worker.py'), str(program), str(out)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, start_new_session=True)
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return {'timeout': True, 'signals': []}
        finally:
            try:
                os.killpg(process.pid, os_signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        return json.loads(out.read_text()) if out.exists() else {'load_error': 'worker produced no result', 'signals': []}


def aggregate(rows: list, total: int = 5) -> dict:
    """Stock aggregation (evaluator.evaluate) over the given per-signal rows; success = len(rows)/total."""
    if not rows:
        return {'combined_score': 0.0, 'success_rate': 0.0}
    mean = lambda key: float(np.mean([r[key] for r in rows]))  # noqa: E731
    success = len(rows) / total
    smooth, accuracy = 1.0 / (1.0 + mean('slope_changes') / 20.0), max(0.0, mean('correlation'))
    score = 0.4 * mean('composite_score') + 0.2 * smooth + 0.2 * accuracy + 0.1 * mean('noise_reduction') + 0.1 * success
    return {'combined_score': 0.0 if accuracy < 0.1 else score, 'success_rate': success, 'composite_score': mean('composite_score'),
            'smoothness_score': smooth, 'correlation': mean('correlation'), 'noise_reduction': mean('noise_reduction')}


@lru_cache(maxsize=1)
def baseline() -> tuple:
    return tuple(run_signals(INITIAL.read_text())['signals'])


def evaluate(source: str) -> tuple:
    """(metrics, artifacts) with stock combined_score, valid_score, success rates, causality, per-signal feedback."""
    result = run_signals(source)
    if result.get('timeout') or result.get('load_error'):
        error = 'Program timed out after 360s' if result.get('timeout') else result['load_error']
        return {'validity': 0, 'combined_score': 0.0, 'valid_score': 0.0, 'success_rate': 0.0, 'error': error}, {}
    rows, base = result['signals'], baseline()
    stock = aggregate([r for r in rows if r['stock_ok']])
    counted = [r if r['valid'] else base[r['case']] for r in rows]
    valid = aggregate(counted)
    valid_count = sum(r['valid'] for r in rows)
    valid['combined_score'] = (valid['combined_score'] - 0.1 * valid['success_rate'] + 0.1 * valid_count / 5) if valid['combined_score'] else 0.0
    probes = [r['causal'] for r in rows if 'causal' in r]
    metrics = {'combined_score': stock['combined_score'], 'success_rate': stock['success_rate'], 'valid_score': valid['combined_score'],
               'valid_success_rate': valid_count / 5, 'fallback_signals': sum(bool(r.get('fallback')) for r in rows),
               'causal_fraction': (sum(probes) / len(probes)) if probes else 0.0}
    if stock['combined_score'] == 0.0 and not any(r['stock_ok'] for r in rows):
        metrics.update(validity=0, error='All test signals failed')
    for r in rows:
        r['value'] = r.get('composite_score')
    header = (f"Valid score {metrics['valid_score']:.4f} (all 5 signals, documented output length), stock score {metrics['combined_score']:.4f}, "
              f"valid signals {valid_count}/5, fallback used on {metrics['fallback_signals']} signal(s).")
    detail = '\n'.join(f"  signal {r['case']} (length {r['length']}): " + (
        f"composite={r['composite_score']:.3f} slope_changes={r['slope_changes']:.0f} lag={r['lag_error']:.3f} avg_err={r['avg_error']:.3f} "
        f"false_reversals={r['false_reversals']:.0f} corr={r['correlation']:.3f} noise_red={r['noise_reduction']:.3f}" if r.get('stock_ok') else 'failed')
        + (f" | {r['error']}" if r.get('error') else '') for r in rows)
    body = format_case_diagnostics([{**r, 'ok': r['valid']} for r in rows], worst=0, lower_is_better=False, value_key='value')
    return metrics, {'feedback': (header + '\n' + body.split('\n')[0] + '\nPer signal:\n' + detail)[:1900]}
