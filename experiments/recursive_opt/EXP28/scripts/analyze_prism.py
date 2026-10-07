"""EXP28 Part C analysis (CPU only): PRISM best-so-far per solution call for the VariationSearch arms and stock EvoX.

Each candidate gets (call, stock score, all-case score):
  trainer arms  evaluations.jsonl (EXP24 white-box metrics of the projected program, tagged with solution calls so far)
  EvoX          candidate_history.jsonl sources re-scored with EXP24's white-box evaluator (stock metric reproduced,
                all-case score = failed cases counted at the initial placement's KVPR)
Calls to optimum = first call whose all-case score reaches 26.2559717 (EXP24's endpoint).

Usage: python analyze_prism.py results/prism_<stamp>
"""
import functools
import importlib.util
import json
import sys
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
_spec = importlib.util.spec_from_file_location('prism_whitebox', EXP / 'EXP24' / 'prism' / 'whitebox.py')
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)
W.evaluate = functools.lru_cache(maxsize=None)(W.evaluate)
OPT = 26.2559717495


def trainer_points(run: Path):
    rows = [json.loads(line) for line in (run / 'evaluations.jsonl').read_text().splitlines()]
    return [(r['solution_calls'], r['combined_score'] or 0.0, r['valid_score'] or 0.0) for r in rows if r['solution_calls'] > 0]


def evox_points(run: Path):
    out = []
    for line in (run / 'candidate_history.jsonl').read_text().splitlines():
        row = json.loads(line)
        source = (row.get('candidate') or {}).get('solution')
        if source:
            metrics = W.evaluate(source)[0]
            out.append((row['iteration'], metrics.get('combined_score') or 0.0, metrics.get('valid_score') or 0.0))
    return out


def summarize(points):
    best_all, hit = -1.0, None
    for it, _, allcase in sorted(points):
        if hit is None and allcase >= OPT - 1e-6:
            hit = it
        best_all = max(best_all, allcase)
    return {'best_stock': round(max(s for _, s, _ in points), 4), 'best_allcase': round(best_all, 4), 'calls_to_optimum': hit,
            'last_call': max(it for it, _, _ in points), 'points': [[it, round(s, 4), round(a, 4)] for it, s, a in points]}


def main() -> None:
    campaign = Path(sys.argv[1])
    runs = {}
    for run in sorted(p for p in campaign.iterdir() if p.is_dir()):
        evox = (run / 'candidate_history.jsonl').exists()
        if not evox and not (run / 'evaluations.jsonl').exists():
            continue
        points = evox_points(run) if evox else trainer_points(run)
        if not points:
            continue
        mix = {}
        if (run / 'variation_log.json').exists():
            for r in json.loads((run / 'variation_log.json').read_text()):
                mix[r['mode']] = mix.get(r['mode'], 0) + 1
        runs[run.name] = {'arm': run.name.rsplit('_s', 1)[0], **summarize(points), 'mode_mix': mix}
        print(run.name, {k: runs[run.name][k] for k in ('best_stock', 'best_allcase', 'calls_to_optimum', 'last_call')}, flush=True)
    arms = {}
    for r in runs.values():
        arms.setdefault(r['arm'], []).append(r)
    summary = {}
    for arm, rs in arms.items():
        hits = sorted(r['calls_to_optimum'] for r in rs if r['calls_to_optimum'] is not None)
        summary[arm] = {'runs': len(rs), 'median_best_stock': round(median(r['best_stock'] for r in rs), 4),
                        'median_best_allcase': round(median(r['best_allcase'] for r in rs), 4), 'reached_optimum': f'{len(hits)}/{len(rs)}',
                        'calls_to_optimum': hits, 'median_calls_to_optimum': median(hits) if len(hits) == len(rs) else None}
        print(f'{arm:24s}', json.dumps(summary[arm]))
    (campaign / 'analysis.json').write_text(json.dumps({'optimum': OPT, 'summary': summary, 'runs': runs}, indent=1) + '\n')


if __name__ == '__main__':
    main()
