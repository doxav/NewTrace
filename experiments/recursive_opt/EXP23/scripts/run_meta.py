"""Offline comparison of meta-optimization arms (A0, A1, A2, A7) on paired holdout seeds.

Gain = (arm final best - stock uniform final best on the same seed) / SD of the stock
policy's final best across seeds (an effect size comparable across tasks and worlds).
Every online arm spends exactly the horizon in solution calls. A2 additionally spends
meta-training episodes (reported separately), which is the point of amortization.
The optimizer LLM is the feedback-blind mock: these runs test score and parent-selection
mechanics, not the quality of LLM proposals or feedback (that is A3).
"""

import argparse
import itertools
import json
import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.control_plane import SPLITS, run, stock_final
from src.meta import ONLINE_ARMS, run_online
from src.mock_llm import mutate
from src.policies import REFERENCE, STOCK, compile_policy, dump_knobs
from src.search import run_fixed
from src.world import TASKS, WORLDS, World

FIXED = tuple(f'fixed:{name}' for name in ('uniform', 'greedy', 'top5_reuse'))


def _one(job: tuple) -> dict:
    task, world_name, arm, seed, horizon = job
    world = World(task, world_name)
    options = {} if arm in FIXED else {'patience': max(10, horizon // 10), 'window': max(10, horizon // 10)}
    if arm.endswith('paired'):
        options.update(trigger='stagnation', credit='new_best', memory_size=5)
    result = run_online(world, seed, arm, surface='knobs', horizon=horizon, **options)
    base = stock_final(task, world_name, seed, horizon)
    return {'task': task, 'world': world_name, 'arm': arm, 'seed': seed, 'horizon': horizon, 'diff': result['final_best'] - base,
            'proposals': result['proposals'], 'promotions': result['promotions'], 'prompt_chars': result['mean_prompt_chars']}


def oracle_pool(task: str, world_name: str, candidates: int = 40) -> list[dict]:
    """Candidate fixed policies for the ceiling of the lever."""
    rng = random.Random(f'oracle:{task}:{world_name}')
    return [STOCK] + list(REFERENCE.values()) + [mutate(STOCK, rng, global_rate=1.0) for _ in range(candidates)]


def fixed_diff(job: tuple) -> float:
    """Mean paired raw difference of one fixed policy over a seed range."""
    task, world_name, knobs, start, count, horizon = job
    world, select = World(task, world_name), compile_policy('knobs', dump_knobs(knobs))
    return mean(run_fixed(world, select, s, horizon)[-1] - stock_final(task, world_name, s, horizon) for s in range(start, start + count))


def stock_sd(job: tuple) -> float:
    task, world_name, count, horizon = job
    return stdev(stock_final(task, world_name, SPLITS['holdout'] + i, horizon) for i in range(count))


def summarize(rows: list[dict], sd: float) -> dict:
    gains = [r['diff'] / sd for r in rows]
    half = 1.96 * stdev(gains) / math.sqrt(len(gains)) if len(gains) > 1 else float('nan')
    return {'n': len(gains), 'gain': round(mean(gains), 3), 'ci95': round(half, 3), 'p_beats_stock': round(mean(g > 0 for g in gains), 3),
            'proposals': round(mean(r['proposals'] for r in rows), 2), 'promotions': round(mean(r['promotions'] for r in rows), 2),
            'prompt_chars': round(mean(r['prompt_chars'] for r in rows))}


def amortized(job: tuple) -> dict:
    """A2 through the control plane: PrioritySearch + OptoPrimeV2 (mock) over whole episodes."""
    task, world_name, surface, holdout, sd = job
    raw = json.loads((ROOT / 'configs' / f'A2/priority_search-{surface}' / 'raw_spec.json').read_text())
    level = raw['levels'][0]
    level['module']['config'].update(task=task, world=world_name)
    level['datasets']['holdout']['config']['count'] = holdout
    (result,) = run(raw)
    if result.status != 'success':
        raise RuntimeError(result.error)
    config = level['engine']['config']
    scale = World(task, world_name).scale
    return {'key': f'{task}/{world_name}/{surface}', 'gain': round(result.evaluation.metrics['score'] * scale / sd, 3), 'policy': result.artifact['selection_policy'][:200],
            'meta_training_episodes_upper_bound': config['iterations'] * (config['num_candidates'] + 1) * config['trainer_kwargs']['batch_size']}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=200)
    parser.add_argument('--long-seeds', type=int, default=60)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--skip-amortized', action='store_true')
    parser.add_argument('--out', default=str(ROOT / 'results' / 'meta_comparison.json'))
    args = parser.parse_args()
    holdout = range(SPLITS['holdout'], SPLITS['holdout'] + args.seeds)
    jobs = [(t, w, a, s, 100) for t, w, a, s in itertools.product(TASKS, WORLDS, FIXED + ONLINE_ARMS, holdout)]
    long_holdout = range(SPLITS['holdout'], SPLITS['holdout'] + args.long_seeds)
    jobs += [(t, 'W1', a, s, h) for t, a, s, h in itertools.product(TASKS, ('fixed:uniform', 'fixed:top5_reuse', 'evox_log', 'evox_paired', 'trace_paired'), long_holdout, (300, 1000))]
    task_worlds = list(itertools.product(TASKS, WORLDS))
    with Pool(args.workers) as pool:
        sds = dict(zip([(t, w, h) for t, w in task_worlds for h in (100,)] + [(t, 'W1', h) for t in TASKS for h in (300, 1000)],
                       pool.map(stock_sd, [(t, w, args.seeds, 100) for t, w in task_worlds] + [(t, 'W1', args.long_seeds, h) for t in TASKS for h in (300, 1000)])))
        rows = pool.map(_one, jobs, chunksize=4)
        pools = {tw: oracle_pool(*tw) for tw in task_worlds}
        train = pool.map(fixed_diff, [(t, w, k, SPLITS['train'], 100, 100) for (t, w) in task_worlds for k in pools[(t, w)]])
        chosen, index = {}, 0
        for tw in task_worlds:
            scores = train[index:index + len(pools[tw])]
            index += len(pools[tw])
            chosen[tw] = pools[tw][max(range(len(scores)), key=scores.__getitem__)]
        held = pool.map(fixed_diff, [(t, w, chosen[(t, w)], SPLITS['holdout'], args.seeds, 100) for t, w in task_worlds])
        amort = [] if args.skip_amortized else pool.map(amortized, [(t, w, 'knobs', args.seeds, sds[(t, w, 100)]) for t, w in task_worlds] + [('prism', 'W1', 'code', args.seeds, sds[('prism', 'W1', 100)])])
    report: dict = {'unit': 'paired gain vs stock uniform on the same seed, in SDs of the stock final best', 'stock_sd': {'/'.join(map(str, k)): v for k, v in sds.items()}, 'online': {}, 'horizon': {}, 'oracle': {}, 'amortized': {}}
    for (task, world_name, horizon), group in itertools.groupby(sorted(rows, key=lambda r: (r['task'], r['world'], r['horizon'])), key=lambda r: (r['task'], r['world'], r['horizon'])):
        group = list(group)
        target = report['online' if horizon == 100 else 'horizon'].setdefault(f'{task}/{world_name}' + ('' if horizon == 100 else f'/h{horizon}'), {})
        for arm in sorted({r['arm'] for r in group}):
            target[arm] = summarize([r for r in group if r['arm'] == arm], sds[(task, world_name, horizon)])
    for (task, world_name), value in zip(task_worlds, held):
        report['oracle'][f'{task}/{world_name}'] = {'holdout_gain': round(value / sds[(task, world_name, 100)], 3), 'policy': chosen[(task, world_name)]}
    for item in amort:
        report['amortized'][item.pop('key')] = item
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    for section in ('online', 'horizon'):
        for key, arms in report[section].items():
            print(f'\n## {key}')
            for arm, s in arms.items():
                print(f"  {arm:18s} gain {s['gain']:+7.3f} ±{s['ci95']:.3f}  P>stock {s['p_beats_stock']:.2f}  proposals {s['proposals']:5.2f} promotions {s['promotions']:5.2f} prompt_chars {s['prompt_chars']}")
    print('\n## oracle (best fixed knob policy chosen on train seeds, holdout gain)')
    for key, value in report['oracle'].items():
        print(f"  {key}: {value['holdout_gain']:+.3f}  {value['policy']}")
    print('\n## amortized PrioritySearch (A2)')
    for key, value in report['amortized'].items():
        print(f'  {key}: {value}')


if __name__ == '__main__':
    main()
