"""EXP27 Part E: per-instruction yield in the real runs (log-only). For every iteration whose parent has no SciPy, the
share of children that introduce SciPy, by instruction (none / refine / diverge) and engine. Iterations after the run's
first SciPy candidate are excluded (lineage then carries SciPy). Writes results/instruction_yield.json.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mechanisms as M  # noqa: E402
import strategy_timeline as T  # noqa: E402


def stock(run):
    T.stock_steps(run)  # sets M.DIV / M.REF from the run's labels
    return [(r['it'], r['label'], r['src'], r['parent_src']) for r in M.stock_rows(run)]


def native(run):
    srcs = {p.stem: p.read_text() for p in (run / 'sources').glob('*.py')}
    audit = [json.loads(line) for line in (run / 'evaluations.jsonl').read_text().splitlines()]
    by_score = {}
    for e in audit:
        by_score.setdefault(round(e['valid_score'] or 0.0, 12), e['sha256'])
    src_at, rows = {0: srcs.get(by_score.get(round(audit[0]['valid_score'] or 0.0, 12)), '')}, []
    for line in (run / 'events.jsonl').read_text().splitlines():
        ev = json.loads(line)
        if ev.get('type') != 'iteration':
            continue
        sha = by_score.get(round(ev['child_score'], 12)) if ev.get('child_score') is not None else None
        src = srcs.get(sha) if sha else None
        src_at[ev['iteration']] = src
        rows.append((ev['iteration'], ev.get('label') or '', src, src_at.get(ev.get('parent_iteration'))))
    return rows


def main() -> None:
    table = {}
    for camp in T.CAMPAIGNS:
        for run in sorted(p for p in camp.glob('*_s4?') if p.is_dir()):
            is_stock = (run / 'candidate_history.jsonl').exists()
            rows = stock(run) if is_stock else native(run)
            first = min((it for it, _, src, _ in rows if src and M.SCIPY.search(src)), default=10 ** 9)
            cell = table.setdefault('stock' if is_stock else 'trace', {})
            for it, label, src, parent in rows:
                if it > first or parent is None or M.SCIPY.search(parent):
                    continue
                c = cell.setdefault(label or 'none', [0, 0])
                c[0] += 1
                c[1] += bool(src and M.SCIPY.search(src))
    out = {eng: {lab: {'calls': n, 'scipy': k, 'rate': round(k / n, 3)} for lab, (n, k) in sorted(cells.items())} for eng, cells in table.items()}
    print(json.dumps(out, indent=1))
    (HERE.parent / 'results' / 'instruction_yield.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
