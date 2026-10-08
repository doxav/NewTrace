"""Run one O0 child spec (EXP29 numeric) and save its result, trajectory and cost. Used for P0 calibration and baselines.

python -m experiments.recursive_opt.EXP29.numeric.run_child --episode mixA --seed 1 --iterations 16 --out DIR [--trainer VariationSearch]
"""
import argparse
import json
import time
from pathlib import Path

from opto.features.recursive_opt import spec as S

from . import specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', default='mixA')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--episode-seed', type=int, default=None)
    parser.add_argument('--objective', default=None)
    parser.add_argument('--iterations', type=int, default=8)
    parser.add_argument('--trainer', default='PrioritySearch')
    parser.add_argument('--trainer-kwargs', default='{}')
    parser.add_argument('--patches', default='[]')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = specs.o0_spec(specs.episode(args.episode, args.seed if args.episode_seed is None else args.episode_seed), seed=args.seed,
                        objective_text=args.objective or specs.OBJECTIVE_TEXT, iterations=args.iterations, trainer=args.trainer,
                        trainer_kwargs=json.loads(args.trainer_kwargs), patches=json.loads(args.patches))
    (out / 'spec.json').write_text(json.dumps(raw, indent=1))
    started = time.time()
    (result,) = S.execute_plan(S.compile_plan(raw))
    data = result.to_dict()
    data['wall_s'] = round(time.time() - started, 1)
    (out / 'result.json').write_text(json.dumps(data, indent=1, default=str))
    trajectory = [c['evaluation']['score'] for c in data['metadata'].get('candidate_trajectory', [])]
    usage = data['usage'].get('optimizer', {})
    summary = {'valid': data['valid'], 'holdout_score': data['evaluation']['metrics'].get('score'), 'train_scores': trajectory,
               'optimizer_calls': usage.get('calls'), 'cost': usage.get('cost_usd') or usage.get('cost'), 'tokens': usage.get('total_tokens'),
               'wall_s': data['wall_s'], 'error': data.get('error'), 'patches': data['metadata'].get('patches')}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
