"""Re-score PRISM candidates: stock combined_score vs a valid score (all 50 cases solved).

The stock PRISM metric averages KVPR over *solved* cases only, so crashing on hard cases raises
the score. A candidate is valid here only if success_rate == 1 (honest ceiling: 29.40).
"""

import glob
import importlib.util
import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

EVALUATOR = '/home/xav/code/evo-compare/repos/skydiscover/benchmarks/ADRS/prism/evaluator/evaluator.py'
spec = importlib.util.spec_from_file_location('prism_evaluator', EVALUATOR)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def score(path: str) -> dict:
    with redirect_stdout(StringIO()):
        result = ev.evaluate(path)
    return {'combined': result.get('combined_score', 0.0), 'success': result.get('success_rate', 0.0)}


def summarize(name: str, rows: list) -> dict:
    valid = [r for r in rows if r['success'] >= 1.0]
    best_raw = max(rows, key=lambda r: r['combined']) if rows else None
    best_valid = max(valid, key=lambda r: r['combined']) if valid else None
    return {'run': name, 'candidates': len(rows), 'valid': len(valid), 'best_raw': best_raw and round(best_raw['combined'], 3),
            'best_raw_success': best_raw and best_raw['success'], 'best_valid': best_valid and round(best_valid['combined'], 3)}


def main() -> None:
    out = []
    for run in sys.argv[1:]:
        run = Path(run)
        if (run / 'candidate_history.jsonl').exists():  # EXP22 stock/Trace runs: metrics already recorded
            rows = []
            for line in (run / 'candidate_history.jsonl').read_text().splitlines():
                c = json.loads(line).get('candidate')
                if c and isinstance(c['metrics'].get('combined_score'), (int, float)):
                    rows.append({'combined': c['metrics']['combined_score'], 'success': c['metrics'].get('success_rate', 0.0)})
        else:  # v2 runs: re-evaluate every audited source
            rows = [score(p) for p in sorted(glob.glob(str(run / 'evaluations' / '*.py')))]
        out.append(summarize(run.name if run.name not in ('trace', 'llm_rewrite') else f'{run.parent.name}/{run.name}', rows))
        print(json.dumps(out[-1]), flush=True)


if __name__ == '__main__':
    main()
