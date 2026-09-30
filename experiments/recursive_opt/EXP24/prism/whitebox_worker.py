"""Evaluate one PRISM program case by case (stock generator, checks and 10 s per-case timeout).

Usage: python whitebox_worker.py PROGRAM.py OUT.json
Writes {"cases": [{case, ok, kvpr, error, time_s, fallback, format_error}...]}.
"""

import importlib.util
import json
import sys
import time

STOCK = '/home/xav/code/evo-compare/repos/skydiscover/benchmarks/ADRS/prism/evaluator/evaluator.py'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(ev, placement, gpu_num, models):
    """Stock format checks (in stock order) plus GPU ids within range; returns an error string or None."""
    if not isinstance(placement, dict):
        return f'Expected dict, got {type(placement).__name__}'
    placed = []
    for gpu_id, assigned in placement.items():
        if not isinstance(assigned, list):
            return f'GPU {gpu_id} value must be list, got {type(assigned).__name__}'
        placed.extend(assigned)
    if len(placed) != len(models):
        return f'Not all models placed: {len(placed)}/{len(models)}'
    ids = [id(m) for m in placed]
    if len(set(ids)) != len(ids):
        return 'Duplicate models detected'
    if set(ids) != {id(m) for m in models}:
        return "Placed models don't match input models (missing or foreign models)"
    if not ev.verify_gpu_mem_constraint(placement):
        return 'GPU memory constraint violated'
    if any(not isinstance(k, int) or not 0 <= k < gpu_num for k in placement):
        return 'GPU id outside range(gpu_num) (not checked by the stock evaluator)'
    return None


def main(program_path, out_path):
    ev = load(STOCK, 'prism_stock_evaluator')
    rows = []
    try:
        program = load(program_path, 'program')
        entry = program.compute_model_placement
    except Exception as error:  # noqa: BLE001
        json.dump({'load_error': f'{type(error).__name__}: {error}', 'cases': []}, open(out_path, 'w'))
        return
    events = getattr(program, '_PROJECTION_EVENTS', None)
    for index, (gpu_num, models) in enumerate(ev.generate_test_gpu_models()):
        row = {'case': index, 'gpus': gpu_num, 'models': len(models), 'total_size': sum(m.model_size for m in models), 'ok': False}
        before = len(events) if events is not None else 0
        started = time.time()
        try:
            placement = ev.run_with_timeout(entry, kwargs={'gpu_num': gpu_num, 'models': models}, timeout_seconds=10)
            error = check(ev, placement, gpu_num, models)
            if error:
                row.update(error=error, format_error=True)
            else:
                row.update(ok=True, kvpr=float(ev.calculate_kvcache_pressure(placement)))
        except TimeoutError:
            row.update(error='TimeoutError: case exceeded 10 s')
        except Exception as error:  # noqa: BLE001
            row.update(error=f'{type(error).__name__}: {error}')
        row['time_s'] = round(time.time() - started, 3)
        if events is not None and len(events) > before:
            row['fallback'] = events[before]
        rows.append(row)
    json.dump({'cases': rows}, open(out_path, 'w'))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
