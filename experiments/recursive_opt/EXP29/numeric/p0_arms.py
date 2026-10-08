"""EXP29 P0 certificates / P1 baselines: hand-written arms on the O1 holdout episodes (mixA/B/C, episode seed 3).

python -m experiments.recursive_opt.EXP29.numeric.p0_arms --out DIR [--repeats 1 2] [--iterations 8]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from opto.features.recursive_opt import patches as P

from . import specs

HAND_MODE = '''def _next_mode(self):
    if self._variation_refine_left > 0:
        self._variation_refine_left -= 1
        return 'refine'
    if self._variation_stall >= 2:
        self._variation_stall = 0
        return self._explore_mode()
    return 'free'
'''
HAND_OBJECTIVE = specs.OBJECTIVE_TEXT + (' Strategy hints that usually help on such benchmarks: start with a few space-filling points, '
    'then exploit around the best points with a step size that shrinks on failures and grows on successes, fit a simple quadratic '
    'model of the best points to jump to its minimum on smooth functions, and restart from a new region when progress stalls '
    '(multimodal functions such as Rastrigin or Ackley have many local minima).')
ARMS = {
    'a_priority': {'trainer': 'PrioritySearch'},
    'b_variation_default': {'trainer': 'VariationSearch'},
    'c_variation_hand_schedule': {'trainer': 'VariationSearch', 'patches': [{'target': specs.MODE_TARGET, 'source': HAND_MODE}]},
    'd_priority_hand_objective': {'trainer': 'PrioritySearch', 'objective': HAND_OBJECTIVE},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--repeats', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--iterations', type=int, default=8)
    parser.add_argument('--arms', nargs='+', default=list(ARMS))
    args = parser.parse_args()
    P.validate(ARMS['c_variation_hand_schedule']['patches'])
    procs = []
    for arm in args.arms:
        for episode in specs.EPISODES:
            for repeat in args.repeats:
                out = Path(args.out) / f'{arm}__{episode}__r{repeat}'
                if (out / 'summary.json').exists():
                    continue
                config = ARMS[arm]
                cmd = [sys.executable, '-m', 'experiments.recursive_opt.EXP29.numeric.run_child', '--episode', episode, '--episode-seed', '3',
                       '--seed', str(100 + repeat), '--iterations', str(args.iterations), '--trainer', config['trainer'],
                       '--trainer-kwargs', json.dumps({'num_threads': 4, **({'variation_seed': repeat} if config['trainer'] == 'VariationSearch' else {})}),
                       '--patches', json.dumps(config.get('patches', [])), '--out', str(out)]
                if 'objective' in config:
                    cmd += ['--objective', config['objective']]
                out.mkdir(parents=True, exist_ok=True)
                procs.append(subprocess.Popen(cmd, stdout=open(out / 'stdout.log', 'w'), stderr=subprocess.STDOUT))
    for proc in procs:
        proc.wait()
    print('done', len(procs))


if __name__ == '__main__':
    main()
