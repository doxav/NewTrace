"""EXP24 analysis: primary endpoint = solution calls until the exact PRISM optimum (26.2559717495);
secondary = best valid score, wasted attempts, children at the optimum, meta calls, cost, transport retries."""

import json
import sys
from pathlib import Path
from statistics import median

OPTIMUM = 26.2559717495


def run_stats(run: Path) -> dict:
    events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
    iterations = [e for e in events if e['type'] == 'iteration']
    attempts, first = 0, None
    for e in iterations:
        attempts += e['attempts']
        if first is None and e.get('child_score') is not None and e['child_score'] >= OPTIMUM - 1e-6:
            first = attempts
    calls = [json.loads(line) for line in (run / 'calls.jsonl').read_text().splitlines()] if (run / 'calls.jsonl').exists() else []
    summary = json.loads((run / 'summary.json').read_text()) if (run / 'summary.json').exists() else {}
    best = max([e['child_score'] for e in iterations if e.get('child_score') is not None] or [None], key=lambda v: v or 0)
    return {'run': run.name, 'complete': summary.get('status') == 'success', 'calls_to_optimum': first, 'best_valid': best,
            'wasted_attempts': sum(e['attempts'] - (e['error'] is None) for e in iterations),
            'children_at_optimum': sum((e.get('child_score') or 0) >= OPTIMUM - 1e-6 for e in iterations),
            'policy_deployments': sum(e['type'] == 'deploy' and e.get('ok') for e in events),
            'cost_usd': round(sum(c.get('cost') or 0 for c in calls), 4), 'transport_retries': sum(not c['ok'] for c in calls)}


def main() -> None:
    campaign = Path(sys.argv[1])
    rows = [run_stats(run) for run in sorted(campaign.iterdir()) if (run / 'events.jsonl').exists()]
    arms = {}
    for row in rows:
        arms.setdefault(row['run'].rsplit('_s', 1)[0], []).append(row)
    aggregate = {}
    for arm, group in arms.items():
        reached = [r['calls_to_optimum'] for r in group if r['calls_to_optimum'] is not None]
        aggregate[arm] = {'runs': len(group), 'reached_optimum': f'{len(reached)}/{len(group)}', 'calls_to_optimum': [r['calls_to_optimum'] for r in group],
                          'median_calls_to_optimum': median(reached) if len(reached) == len(group) else None,
                          'wasted_attempts': [r['wasted_attempts'] for r in group], 'children_at_optimum': [r['children_at_optimum'] for r in group],
                          'cost_usd': round(sum(r['cost_usd'] for r in group), 4)}
    report = {'optimum': OPTIMUM, 'runs': rows, 'arms': aggregate}
    (campaign / 'analysis.json').write_text(json.dumps(report, indent=1) + '\n')
    for arm, a in aggregate.items():
        print(f"{arm:12s} reached {a['reached_optimum']}  calls to optimum {a['calls_to_optimum']}  wasted {a['wasted_attempts']}  at optimum {a['children_at_optimum']}  cost ${a['cost_usd']}")


if __name__ == '__main__':
    main()
