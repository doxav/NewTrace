"""Evaluate one Signal Processing program signal by signal (stock generator, metrics and 10 s timeout).

Usage: python whitebox_worker.py PROGRAM.py OUT.json
Writes per-signal records: stock metric components, output length check, finiteness, causality probe, fallback use.
"""

import importlib.util
import json
import sys
import time

import numpy as np

STOCK = '/home/xav/code/evo-compare/repos/skydiscover/benchmarks/math/signal_processing/evaluator/evaluator.py'
WINDOW = 20
PREFIX_CUT = 50  # causality probe: rerun without the last 50 samples, earlier outputs must not change


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics_for(ev, filtered, noisy, clean):
    """Stock per-signal metrics, computed exactly as in evaluator.evaluate."""
    S = ev.calculate_slope_changes(filtered)
    L_recent = ev.calculate_lag_error(filtered, noisy, WINDOW)
    L_avg = ev.calculate_average_tracking_error(filtered, noisy, WINDOW)
    R = ev.calculate_false_reversal_penalty(filtered, clean, WINDOW)
    composite = ev.calculate_composite_score(S, L_recent, L_avg, R)
    correlation, noise_reduction = 0.0, 0.0
    try:
        delay = WINDOW - 1
        aligned_clean = clean[delay:delay + len(filtered)]
        n = min(len(filtered), len(aligned_clean))
        if n > 1:
            c = ev.pearsonr(filtered[:n], aligned_clean[:n])[0]
            correlation = c if not np.isnan(c) else 0.0
        aligned_noisy = noisy[delay:delay + len(filtered)][:n]
        aligned_clean = aligned_clean[:n]
        if n > 0:
            before = np.var(aligned_noisy - aligned_clean)
            after = np.var(filtered[:n] - aligned_clean)
            noise_reduction = max(0, (before - after) / before if before > 0 else 0)
    except Exception:  # noqa: BLE001 - mirrors the stock evaluator
        pass
    return {'slope_changes': ev.safe_float(S), 'lag_error': ev.safe_float(L_recent), 'avg_error': ev.safe_float(L_avg),
            'false_reversals': ev.safe_float(R), 'composite_score': ev.safe_float(composite), 'correlation': ev.safe_float(correlation),
            'noise_reduction': ev.safe_float(noise_reduction)}


def main(program_path, out_path):
    ev = load(STOCK, 'signal_stock_evaluator')
    try:
        program = load(program_path, 'program')
        entry = program.run_signal_processing
    except Exception as error:  # noqa: BLE001
        json.dump({'load_error': f'{type(error).__name__}: {error}', 'signals': []}, open(out_path, 'w'))
        return
    events = getattr(program, '_PROJECTION_EVENTS', None)
    rows = []
    for index, (noisy, clean) in enumerate(ev.generate_test_signals(5)):
        row = {'case': index, 'length': len(noisy), 'expected_output_length': len(noisy) - WINDOW + 1, 'stock_ok': False, 'valid': False}
        before = len(events) if events is not None else 0
        started = time.time()
        try:
            result = ev.run_with_timeout(entry, kwargs={'noisy_signal': noisy, 'window_size': WINDOW}, timeout_seconds=10)
            row['time_s'] = round(time.time() - started, 3)
            if not isinstance(result, dict) or 'filtered_signal' not in result or len(result['filtered_signal']) == 0:
                row['error'] = 'invalid result format (stock evaluator skips this signal)'
            else:
                filtered = np.array(result['filtered_signal'])
                row.update(stock_ok=True, output_length=int(len(filtered)), **metrics_for(ev, filtered, noisy, clean))
                finite = bool(np.all(np.isfinite(filtered.astype(float))))
                length_ok = len(filtered) == row['expected_output_length']
                row['valid'] = finite and length_ok
                if not row['valid']:
                    row['error'] = ('output length %d != documented %d' % (len(filtered), row['expected_output_length'])) if not length_ok else 'non-finite output values'
                if row['valid']:  # causality probe on a prefix
                    try:
                        short = ev.run_with_timeout(entry, kwargs={'noisy_signal': noisy[:-PREFIX_CUT], 'window_size': WINDOW}, timeout_seconds=10)
                        prefix = np.array(short['filtered_signal'], dtype=float)
                        k = min(len(prefix), len(filtered)) - 1
                        row['causal'] = bool(k > 0 and np.allclose(prefix[:k], filtered[:k].astype(float), rtol=1e-7, atol=1e-9))
                    except Exception:  # noqa: BLE001
                        row['causal'] = False
        except TimeoutError:
            row['error'] = 'TimeoutError: signal exceeded 10 s'
        except Exception as error:  # noqa: BLE001
            row['error'] = f'{type(error).__name__}: {error}'
        if events is not None and len(events) > before:
            row['fallback'] = events[before]
        rows.append(row)
    json.dump({'signals': rows}, open(out_path, 'w'))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
