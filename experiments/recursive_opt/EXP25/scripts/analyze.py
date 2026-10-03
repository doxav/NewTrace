"""EXP25 analysis: re-score every candidate of every arm with the same evaluator (stock and valid scores).

Primary: best valid score among evaluated candidates. Secondary: valid score and causality of the returned program,
best stock score and its validity, best-valid-so-far at 25/50/100 evaluations, policy switches, provider cost.
Usage: python analyze.py results/runs_<stamp>
"""

import json
import sys
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'signal'))
import whitebox as W  # noqa: E402


def native_candidates(run: Path) -> list:
    rows = []
    for line in (run / 'evaluations.jsonl').read_text().splitlines():
        r = json.loads(line)
        rows.append({'stock': r['combined_score'] or 0.0, 'valid': r['valid_score'] or 0.0, 'valid_signals': r.get('valid_success_rate') or 0.0})
    return rows


def stock_candidates(run: Path, cache: dict) -> list:
    rows = []
    for line in (run / 'candidate_history.jsonl').read_text().splitlines():
        c = json.loads(line).get('candidate')
        if not c:
            continue
        source = c['solution']
        if source not in cache:
            metrics, _ = W.evaluate(source)
            cache[source] = {'stock': metrics.get('combined_score', 0.0), 'valid': metrics.get('valid_score', 0.0), 'valid_signals': metrics.get('valid_success_rate', 0.0)}
        rows.append({**cache[source], 'stock_recorded': (c.get('metrics') or {}).get('combined_score')})
    return rows


def run_stats(run: Path, cache: dict) -> dict:
    arm = run.name.rsplit('_s', 1)[0]
    rows = stock_candidates(run, cache) if arm == 'evox_stock' else native_candidates(run)
    returned = (run / 'best_program.py').read_text() if (run / 'best_program.py').exists() else ''
    returned_metrics = W.evaluate(returned)[0] if returned else {}
    best_stock = max(rows, key=lambda r: r['stock']) if rows else {}
    so_far, curve = 0.0, []
    for r in rows:
        so_far = max(so_far, r['valid'])
        curve.append(so_far)
    summary = json.loads((run / 'summary.json').read_text()) if (run / 'summary.json').exists() else {}
    calls = []
    if (run / 'calls.jsonl').exists():
        calls = [json.loads(line) for line in (run / 'calls.jsonl').read_text().splitlines()]
    events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()] if (run / 'events.jsonl').exists() else []
    switches = summary.get('policy_switches') if arm == 'evox_stock' else sum(e['type'] == 'deploy' and e.get('ok') for e in events)
    return {'run': run.name, 'arm': arm, 'complete': summary.get('status') == 'success', 'candidates': len(rows),
            'best_valid': round(max((r['valid'] for r in rows), default=0.0), 4),
            'returned_valid': round(returned_metrics.get('valid_score', 0.0), 4), 'returned_stock': round(returned_metrics.get('combined_score', 0.0), 4),
            'returned_valid_signals': returned_metrics.get('valid_success_rate'), 'returned_causal_fraction': returned_metrics.get('causal_fraction'),
            'best_stock': round(best_stock.get('stock', 0.0), 4), 'best_stock_all_signals_valid': best_stock.get('valid_signals') == 1.0,
            'best_valid_after': {n: round(curve[min(n, len(curve)) - 1], 4) for n in (25, 50, 100) if curve},
            'policy_switches': switches, 'cost_usd': round(sum(c.get('cost') or 0 for c in calls), 4) if calls else None}


def main() -> None:
    campaign = Path(sys.argv[1])
    cache: dict = {}
    rows = [run_stats(run, cache) for run in sorted(campaign.iterdir()) if run.is_dir() and (run / 'summary.json').exists()]
    arms = {}
    for row in rows:
        arms.setdefault(row['arm'], []).append(row)
    aggregate = {arm: {'runs': len(g), 'best_valid': [r['best_valid'] for r in g], 'median_best_valid': median(r['best_valid'] for r in g),
                       'returned_valid': [r['returned_valid'] for r in g], 'best_stock': [r['best_stock'] for r in g],
                       'best_stock_valid': [r['best_stock_all_signals_valid'] for r in g], 'returned_causal': [r['returned_causal_fraction'] for r in g]}
                 for arm, g in arms.items()}
    (campaign / 'analysis.json').write_text(json.dumps({'runs': rows, 'arms': aggregate}, indent=1) + '\n')
    for row in rows:
        print(json.dumps(row))
    for arm, a in aggregate.items():
        print(f"{arm:12s} best valid {a['best_valid']} (median {a['median_best_valid']})  returned valid {a['returned_valid']}  best stock {a['best_stock']} valid={a['best_stock_valid']}  causal {a['returned_causal']}")


if __name__ == '__main__':
    main()
