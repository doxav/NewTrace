"""EXP29 P1: O1 discovers the lower trainer's mode policy, diverge text and optimizer instruction (child_spec@1).

Meta-train episodes mixA-s1 + mixB-s1 (run in parallel), meta-validation mixC-s1, meta-holdout mixA/B/C-s3 (= P0 arms).
python -m experiments.recursive_opt.EXP29.numeric.run_o1 --seed 1 --out DIR [--o1-iterations 5] [--child-iterations 8]
"""
import argparse
import json
import time
from pathlib import Path

from opto.features.recursive_opt import spec as S

from . import specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--o1-iterations', type=int, default=5)
    parser.add_argument('--child-iterations', type=int, default=8)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = specs.o1_spec([('mixA', 1), ('mixB', 1)], [('mixC', 1)], [('mixA', 3), ('mixB', 3), ('mixC', 3)], seed=args.seed,
                        child_iterations=args.child_iterations, o1_iterations=args.o1_iterations)
    (out / 'spec.json').write_text(json.dumps(raw, indent=1))
    started = time.time()
    (result,) = S.execute_plan(S.compile_plan(raw))
    data = result.to_dict()
    data['wall_s'] = round(time.time() - started, 1)
    (out / 'result.json').write_text(json.dumps(data, indent=1, default=str))
    trajectory = [{'score': c['evaluation']['score'], 'artifact': c['artifact']} for c in data['metadata'].get('candidate_trajectory', [])]
    (out / 'trajectory.json').write_text(json.dumps(trajectory, indent=1))
    summary = {'valid': data['valid'], 'holdout_score': data['evaluation']['metrics'].get('score'), 'holdout_feedback': str(data['evaluation'].get('feedback'))[:3000],
               'o1_train_scores': [t['score'] for t in trajectory], 'o1_usage': data['usage'].get('optimizer'), 'child_usage': data['usage'].get('forward'),
               'final_artifact': data['artifact'], 'wall_s': data['wall_s'], 'error': data.get('error')}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('final_artifact', 'holdout_feedback')}, default=str))


if __name__ == '__main__':
    main()
