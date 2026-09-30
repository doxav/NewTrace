"""Run every simulator alternative over the (1+2)x4 schedule, one worker process per job.

  --mode mock   offline, feedback-blind mock proposer (validates plumbing, $0)
  --mode live   z-ai/glm-5.3-flash pinned to --provider; needs OPENROUTER_API_KEY in the
                environment (never printed or written)

Twelve simulated steps cannot rank arms, so every proposed policy is also scored
offline on 200 holdout seeds x 100 steps (paired vs stock, stock-SD units): the
"policy value" measures what the LLM proposed, independent of block noise.
A4-A6 (live SkyDiscover kernels) and A7 (horizon scaling) are not run here.
"""

import argparse
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opto.utils.llm import DummyLLM

from src.control_plane import SPLITS, run, stock_final
from src.live import PROVIDERS, LiveOptimizerLLM
from src.mock_llm import MockOptimizerLLM
from src.policies import PolicyInvalid, compile_policy
from src.schedule import ARMS, run_schedule
from src.search import run_fixed
from src.world import World

ALTERNATIVE = {'evox_log': 'A0', 'trace_exp22': 'A0', 'evox_paired': 'A1', 'trace_paired': 'A1', 'llm_blind': 'A3-control', 'priority_search': 'A2'}


def make_llm(mode: str, provider: str, log: Path, seed: int):
    return LiveOptimizerLLM(provider, log) if mode == 'live' else MockOptimizerLLM(seed)


def worker(job: dict) -> dict:
    out = Path(job['out'])
    log = out / f"{job['name']}.calls.jsonl"
    llm = make_llm(job['mode'], job['provider'], log, job['seed'])
    world = World(job['task'], job['world'])
    started = time.time()
    try:
        if job['arm'] == 'priority_search':
            result = priority_search(job, llm)
        else:
            result = run_schedule(world, job['seed'], job['arm'], job['surface'], llm)
        result['error'] = None
    except Exception as error:  # noqa: BLE001 - a failed worker is reported, not hidden
        result = {'error': f'{type(error).__name__}: {str(error)[:300]}', 'trajectory': []}
    result.update(job=job['name'], task=job['task'], arm=job['arm'], alternative=ALTERNATIVE[job['arm']], wall_s=round(time.time() - started, 1),
                  llm_calls=llm.calls, mean_prompt_chars=round(mean(llm.prompt_chars)) if llm.prompt_chars else 0)
    (out / f"{job['name']}.json").write_text(json.dumps(result, indent=2) + '\n')
    return result


def priority_search(job: dict, llm) -> dict:
    """A2 via the control plane: PrioritySearch, 3 trainer steps (= 2 proposals), 4-step episodes."""
    raw = json.loads((ROOT / 'configs' / f"A2/priority_search-{job['surface']}" / 'raw_spec.json').read_text())
    raw['outputs']['directory'] = str(Path(job['out']) / f"{job['name']}.control_plane")
    raw['runtime'].update(offline=True, test_mode=True)
    level = raw['levels'][0]
    level['module']['config'].update(task=job['task'], world=job['world'], horizon=4)
    level['engine']['config'].update(iterations=3, num_candidates=1, validation_gate=False)
    level['engine']['config']['trainer_kwargs'].update(batch_size=1, num_proposals=1)
    level['datasets'] = {'train': {'ref': level['datasets']['train']['ref'], 'config': {'count': 3}}, 'holdout': {'ref': level['datasets']['holdout']['ref'], 'config': {'count': 20}}}
    raw['budget'].update(candidates=None, optimizer_llm_calls=4)
    (result,) = run(raw, llm_factory=lambda profile, role: DummyLLM(llm))
    if result.status != 'success':
        raise RuntimeError(result.error)
    return {'final_policy': result.artifact['selection_policy'], 'trajectory': [{'block': 'final', 'proposed': result.artifact['selection_policy'], 'valid': True}],
            'episodes_evaluated': result.budget.get('used', {}).get('evaluator_runs') if isinstance(result.budget, dict) else None}


def policy_value(job: tuple) -> float | None:
    """Offline value of one proposed policy: paired gain vs stock over 200 seeds x 100 steps."""
    task, world_name, surface, text = job
    try:
        select = compile_policy(surface, text)
    except PolicyInvalid:
        return None
    world = World(task, world_name)
    seeds = range(SPLITS['holdout'], SPLITS['holdout'] + 200)
    sd = stdev(stock_final(task, world_name, s, 100) for s in seeds)
    return round(mean(run_fixed(world, select, s)[-1] - stock_final(task, world_name, s, 100) for s in seeds) / sd, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('mock', 'live'), default='mock')
    parser.add_argument('--provider', choices=PROVIDERS, default='deepinfra')
    parser.add_argument('--seeds', type=int, default=3, help='independent schedule runs per arm and task')
    parser.add_argument('--world', default='W1')
    parser.add_argument('--surface', choices=('knobs', 'code'), default='code')
    args = parser.parse_args()
    out = ROOT / 'results' / 'live_schedule' / args.mode
    out.mkdir(parents=True, exist_ok=True)
    jobs = [{'name': f'{task}__{arm}__s{seed}', 'task': task, 'arm': arm, 'seed': SPLITS['holdout'] + seed, 'world': args.world, 'surface': args.surface,
             'mode': args.mode, 'provider': args.provider, 'out': str(out)}
            for task, arm, seed in itertools.product(('prism', 'signal_processing'), ARMS + ('priority_search',), range(args.seeds))]
    with Pool(len(jobs)) as pool:  # one worker per job
        results = pool.map(worker, jobs, chunksize=1)
        proposals = [(r['task'], args.world, args.surface, t['proposed']) for r in results for t in r['trajectory'] if t.get('proposed') and t.get('valid')]
        values = dict(zip([p[3] + p[0] for p in proposals], pool.map(policy_value, proposals)))
    for r in results:
        for t in r['trajectory']:
            if t.get('proposed') and t.get('valid'):
                t['policy_value_sd'] = values[t['proposed'] + r['task']]
        (out / f"{r['job']}.json").write_text(json.dumps(r, indent=2) + '\n')
    calls = [json.loads(line) for path in out.glob('*.calls.jsonl') for line in path.read_text().splitlines()]
    summary: dict = {'mode': args.mode, 'provider': args.provider, 'surface': args.surface, 'world': args.world, 'seeds': args.seeds, 'arms': {},
                     'llm': {'calls': len(calls), 'errors': sum(not c['ok'] for c in calls), 'served_by': sorted({str(c.get('served_by')) for c in calls if c.get('ok')}),
                             'cost_usd': round(sum(c.get('cost') or 0 for c in calls), 5), 'mean_latency_s': round(mean(c['latency_s'] for c in calls), 1) if calls else 0,
                             'truncated': sum(c.get('finish') == 'length' for c in calls)}}
    for task, arm in itertools.product(('prism', 'signal_processing'), ARMS + ('priority_search',)):
        group = [r for r in results if r['task'] == task and r['arm'] == arm]
        props = [t for r in group for t in r['trajectory'] if t.get('proposed')]
        vals = [t['policy_value_sd'] for t in props if t.get('policy_value_sd') is not None]
        summary['arms'][f'{task}/{arm}'] = {'alternative': ALTERNATIVE[arm], 'errors': [r['error'] for r in group if r['error']], 'proposals': len(props),
                                            'invalid': sum(not t['valid'] for t in props), 'mean_policy_value_sd': round(mean(vals), 3) if vals else None,
                                            'best_policy_value_sd': max(vals) if vals else None, 'promoted': sum(bool(t.get('promoted')) for t in props),
                                            'mean_prompt_chars': round(mean(r['mean_prompt_chars'] for r in group)), 'mean_wall_s': round(mean(r['wall_s'] for r in group), 1)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
