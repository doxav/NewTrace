"""Two sensitivity checks behind the README conclusions (offline).

1. Many meta-steps: horizon 1000 with patience/window fixed at 10 (EvoX scales both
   with the horizon, which caps proposals at ~2-7 per run whatever the horizon).
2. Cheaper amortization: A2 PrioritySearch with 8 instead of 24 trainer iterations.
"""

import itertools
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.control_plane import SPLITS, run, stock_final
from src.meta import run_online
from src.world import World

ARMS = ('fixed:top5_reuse', 'evox_log', 'trace_exp22', 'evox_paired', 'trace_paired')


def one(job: tuple) -> tuple:
    task, arm, seed = job
    world = World(task, 'W1')
    options = {} if arm.startswith('fixed') else {'patience': 10, 'window': 10}
    if arm.endswith('paired'):
        options.update(trigger='stagnation', credit='new_best', memory_size=5)
    result = run_online(world, seed, arm, horizon=1000, **options)
    return task, arm, result['final_best'] - stock_final(task, 'W1', seed, 1000), result['proposals'], result['promotions']


def amortized(job: tuple) -> tuple:
    task, world_name, surface, iterations = job
    raw = json.loads((ROOT / 'configs' / f'A2/priority_search-{surface}' / 'raw_spec.json').read_text())
    level = raw['levels'][0]
    level['module']['config'].update(task=task, world=world_name)
    level['engine']['config']['iterations'] = iterations
    level['datasets']['holdout']['config']['count'] = 200
    (result,) = run(raw)
    return task, world_name, surface, iterations, result.evaluation.metrics['score'] * World(task, world_name).scale


def main() -> None:
    seeds = range(SPLITS['holdout'], SPLITS['holdout'] + 60)
    report: dict = {'many_meta_steps_h1000_patience10': {}, 'amortized_iterations': {}}
    with Pool(16) as pool:
        rows = pool.map(one, list(itertools.product(('prism', 'signal_processing'), ARMS, seeds)), chunksize=2)
        sds = {t: stdev(stock_final(t, 'W1', s, 1000) for s in seeds) for t in ('prism', 'signal_processing')}
        cheap = pool.map(amortized, [(t, w, 'knobs', 8) for t in ('prism', 'signal_processing') for w in ('W0', 'W1', 'W2')] + [('prism', 'W1', 'code', 8)])
    for task, arm in itertools.product(('prism', 'signal_processing'), ARMS):
        group = [r for r in rows if r[0] == task and r[1] == arm]
        gains = [r[2] / sds[task] for r in group]
        report['many_meta_steps_h1000_patience10'][f'{task}/W1/{arm}'] = {'gain_sd': round(mean(gains), 3), 'ci95': round(1.96 * stdev(gains) / math.sqrt(len(gains)), 3),
                                                                        'proposals': round(mean(r[3] for r in group), 1), 'promotions': round(mean(r[4] for r in group), 1)}
    for task, world_name, surface, iterations, diff in cheap:
        sd = stdev(stock_final(task, world_name, SPLITS['holdout'] + i, 100) for i in range(200))
        report['amortized_iterations'][f'{task}/{world_name}/{surface}/it{iterations}'] = round(diff / sd, 3)
    (ROOT / 'results' / 'sensitivity.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
