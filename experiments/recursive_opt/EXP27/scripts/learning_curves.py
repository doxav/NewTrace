"""EXP27 Part F: learning curves on the signal task for every recorded EvoX and Trace run (CPU only, no LLM calls).

For each run, candidates in iteration order with their valid score and causal fraction (Trace: the run's own audit;
stock: re-evaluated with EXP25's white-box evaluator). Reports best valid (any) and best valid among causal
candidates after 10, 25, 42 and 100 iterations, per run and as arm medians. EXP26 runs stop producing candidates
after iteration 43-70 (key limit), so @42 is the last equal-budget point across all campaigns.
Writes results/learning_curves.json.
"""
import functools
import json
import sys
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'EXP25' / 'signal'))
import whitebox as W  # noqa: E402

import mechanisms as M  # noqa: E402
import strategy_timeline as T  # noqa: E402

W.evaluate = functools.lru_cache(maxsize=None)(W.evaluate)
CHECKPOINTS = (10, 25, 42, 100)
INITIAL = 0.499


def stock_points(run):
    T.stock_steps(run)
    points = []
    for row in M.stock_rows(run):
        if row['src']:
            metrics = W.evaluate(row['src'])[0]
            points.append((row['it'], metrics.get('valid_score') or 0.0, metrics.get('causal_fraction') == 1.0))
    return points


def native_points(run):
    audit = {}
    for line in (run / 'evaluations.jsonl').read_text().splitlines():
        e = json.loads(line)
        audit.setdefault(round(e['valid_score'] or 0.0, 12), e)
    points = []
    for line in (run / 'events.jsonl').read_text().splitlines():
        ev = json.loads(line)
        if ev.get('type') == 'iteration' and ev.get('child_score') is not None:
            e = audit.get(round(ev['child_score'], 12))
            if e:
                points.append((ev['iteration'], e['valid_score'] or 0.0, e.get('causal_fraction') == 1.0))
    return points


def curve(points):
    out = {}
    for k in CHECKPOINTS:
        upto = [p for p in points if p[0] <= k]
        out[k] = {'any': round(max([INITIAL] + [p[1] for p in upto]), 4),
                  'causal': round(max([INITIAL] + [p[1] for p in upto if p[2]]), 4)}
    return out


def arm_of(campaign, name):
    base = name.rsplit('_s', 1)[0]
    return f'EXP27 trace {base}' if campaign == 'EXP27' else ('stock EvoX' if base == 'evox_stock' else f'{campaign} {base}')


def main() -> None:
    runs = {}
    for camp in T.CAMPAIGNS:
        cname = camp.parent.parent.name
        for run in sorted(p for p in camp.glob('*_s4?') if p.is_dir()):
            stock = (run / 'candidate_history.jsonl').exists()
            runs[f'{cname}/{run.name}'] = {'arm': arm_of(cname, run.name), 'curve': curve(stock_points(run) if stock else native_points(run))}
            print(f'{cname}/{run.name}', json.dumps(runs[f'{cname}/{run.name}']['curve']), flush=True)
    arms = {}
    for r in runs.values():
        arms.setdefault(r['arm'], []).append(r['curve'])
    summary = {arm: {'runs': len(cs), **{f'{kind}@{k}': round(median(c[k][kind] for c in cs), 4) for kind in ('any', 'causal') for k in CHECKPOINTS}}
               for arm, cs in arms.items()}
    for arm, s in summary.items():
        print(f'{arm:32s}', json.dumps(s))
    (HERE.parent / 'results' / 'learning_curves.json').write_text(json.dumps({'summary': summary, 'runs': runs}, indent=1) + '\n')


if __name__ == '__main__':
    main()
