"""Every PRISM candidate ever scored above the valid optimum (26.256): does any of them solve all 50 cases?"""

import glob
import importlib.util
import json
import sys
from contextlib import redirect_stdout
from io import StringIO

OPT = 26.2559717495
spec = importlib.util.spec_from_file_location('ev', '/home/xav/code/evo-compare/repos/skydiscover/benchmarks/ADRS/prism/evaluator/evaluator.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
rows = []
for history in sorted(glob.glob('EXP22/runs/strict_prism_*/candidate_history.jsonl') + glob.glob('EXP22/artifacts/parallel_trace_*/v9_runtime/runs/strict_prism_*/candidate_history.jsonl')
                      + glob.glob('EXP23/results/prism100/*/*/candidate_history.jsonl')):
    for line in open(history):
        c = json.loads(line).get('candidate')
        if c and isinstance(c['metrics'].get('combined_score'), (int, float)):
            rows.append((history.split('/')[-2][:40], c['metrics']['combined_score'], c['metrics'].get('success_rate')))
for path in sorted(glob.glob('EXP23/results/prism100_v2/*/trace/evaluations/*.py')):
    with redirect_stdout(StringIO()):
        r = ev.evaluate(path)
    rows.append(('EXP23 v2 trace', r.get('combined_score', 0.0), r.get('success_rate')))
above = [r for r in rows if r[1] > OPT + 1e-6]
full = [r for r in rows if r[2] is not None and r[2] >= 1.0]
print(json.dumps({'candidates': len(rows), 'above_optimum': len(above), 'above_optimum_with_all_50_solved': sum(r[2] >= 1.0 for r in above),
                  'max_success_rate_above_optimum': max(r[2] for r in above) if above else None,
                  'best_stock_score_with_all_50_solved': max(r[1] for r in full), 'fully_solved_candidates': len(full)}, indent=1))
for r in sorted(above, key=lambda r: -r[1])[:12]:
    print(f'  {r[0]:42s} stock={r[1]:.3f} success={r[2]:.2f}')
