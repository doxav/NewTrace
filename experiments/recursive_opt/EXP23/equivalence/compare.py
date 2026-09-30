"""Compare the stock EvoX trace with the v2 coevolution trace; exit 1 on any difference."""

import json
import sys
from pathlib import Path

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'out'
stock = json.loads((out / 'stock_trace.json').read_text())
v2 = json.loads((out / 'v2_trace.json').read_text())
problems = []
for index, (a, b) in enumerate(zip(stock['events'], v2['events'])):
    if a != b:
        problems.append(f'event {index}: stock={a} v2={b}')
        if len(problems) >= 10:
            break
if len(stock['events']) != len(v2['events']):
    problems.append(f"event count: stock={len(stock['events'])} v2={len(v2['events'])}")
if stock['curve'] != v2['curve']:
    problems.append('best-score curve differs')
if stock['best_score'] != v2['best_score']:
    problems.append(f"best score: stock={stock['best_score']} v2={v2['best_score']}")
for kind in ('solution', 'meta', 'stats_insight', 'problem_context', 'batch_summary'):
    if stock['calls'].get(kind, 0) != v2['calls'].get(kind, 0):
        problems.append(f"{kind} LLM calls: stock={stock['calls'].get(kind, 0)} v2={v2['calls'].get(kind, 0)}")
if stock['meta_failures'] != v2['meta_failures']:
    problems.append(f"meta failures: stock={stock['meta_failures']} v2={v2['meta_failures']}")
kinds = {}
for event in stock['events']:
    kinds[event['type']] = kinds.get(event['type'], 0) + 1
summary = {'events': len(stock['events']), 'by_type': kinds, 'best_score': stock['best_score'], 'calls_compared': {k: stock['calls'].get(k, 0) for k in ('solution', 'meta', 'stats_insight', 'problem_context', 'batch_summary')},
           'labels_used': sorted({e.get('label') for e in stock['events'] if e['type'] == 'iteration' and e.get('ok')}), 'excluded': 'availability pings (stock only: 2)'}
print(json.dumps(summary, indent=1))
print('EQUIVALENT' if not problems else 'DIFFERENCES:\n' + '\n'.join(problems))
sys.exit(1 if problems else 0)
