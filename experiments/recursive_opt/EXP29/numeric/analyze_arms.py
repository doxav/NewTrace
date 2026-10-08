"""Summarize child-run arms: holdout score, gain over the seed program, paired difference vs a reference arm.

python -m experiments.recursive_opt.EXP29.numeric.analyze_arms DIR [--reference a_priority] [--episode-seed 3]
"""
import argparse
import json
import statistics
from pathlib import Path

from . import specs, task as T


def seed_holdout(episode: str, episode_seed: int) -> float:
    items = specs.episode(episode, episode_seed)['holdout']
    return statistics.mean(-T.run_program(T.SEED_SOURCE, i['instance'], i['seed'])['log_auc'] for i in items)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--reference', default='a_priority')
    parser.add_argument('--episode-seed', type=int, default=3)
    args = parser.parse_args()
    runs = {}
    for summary in sorted(Path(args.root).glob('*/summary.json')):
        arm, episode, repeat = summary.parent.name.split('__')
        runs[(arm, episode, repeat)] = json.loads(summary.read_text())
    seeds = {e: seed_holdout(e, args.episode_seed) for e in {k[1] for k in runs}}
    arms = sorted({k[0] for k in runs})
    table = {}
    for arm in arms:
        rows = {(e, r): s for (a, e, r), s in runs.items() if a == arm}
        valid = {k: s['holdout_score'] for k, s in rows.items() if s.get('valid') and s.get('holdout_score') is not None}
        gains = [v - seeds[e] for (e, _), v in valid.items()]
        ref = {(e, r): s['holdout_score'] for (a, e, r), s in runs.items() if a == args.reference and s.get('valid')}
        paired = [valid[k] - ref[k] for k in valid if k in ref]
        table[arm] = {'n': len(rows), 'valid': len(valid), 'mean_holdout': round(statistics.mean(valid.values()), 4) if valid else None,
                      'mean_gain_vs_seed': round(statistics.mean(gains), 4) if gains else None,
                      'gain_sd': round(statistics.stdev(gains), 4) if len(gains) > 1 else None,
                      f'paired_vs_{args.reference}': round(statistics.mean(paired), 4) if paired else None,
                      'paired_wins': f'{sum(p > 0 for p in paired)}/{len(paired)}',
                      'calls': sum((s.get('optimizer_calls') or 0) for s in rows.values()),
                      'wall_s_mean': round(statistics.mean(s['wall_s'] for s in rows.values()), 0)}
    out = {'seed_holdout': {e: round(v, 4) for e, v in seeds.items()}, 'arms': table,
           'per_run': {'__'.join(k): {'holdout': s.get('holdout_score'), 'valid': s.get('valid'), 'error': s.get('error')} for k, s in runs.items()}}
    (Path(args.root) / 'analysis.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['seed_holdout']))
    for arm, row in table.items():
        print(arm, row)


if __name__ == '__main__':
    main()
