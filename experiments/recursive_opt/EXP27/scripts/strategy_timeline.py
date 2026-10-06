"""EXP27 Part E: how each engine's evolved selection policy behaves over time (log-only, no LLM calls).

Per run and per 25-iteration window, over iterations that produced a candidate:
  unlabelled  share with no variation instruction (stock's prescribed default)
  refine / diverge / other  share per instruction
  best_parent share whose parent was the best candidate at that time (exploitation)
Writes results/strategy_timeline.json.

Usage: python strategy_timeline.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mechanisms as M  # noqa: E402

ROOT = HERE.parents[1]
CAMPAIGNS = [ROOT / 'EXP25' / 'results' / 'runs_20260930T235849', ROOT / 'EXP26' / 'results' / 'runs_20261006T112918',
             HERE.parent / 'results' / 'partD_20261006T195738']
WINDOWS = ((1, 25), (26, 50), (51, 75), (76, 100))


def stock_steps(run):
    if (run / 'labels.json').exists():
        labels = json.loads((run / 'labels.json').read_text())
    else:  # EXP25 stock runs keep the generated labels only in SkyDiscover's per-iteration snapshot
        import yaml
        snap = yaml.safe_load((run / 'search' / 'iteration_1' / 'labels.yaml').read_text())
        labels = {'diverge': snap['diverge_label'], 'refine': snap['refine_label']}
    M.DIV, M.REF = labels.get('diverge', '?').strip(), labels.get('refine', '?').strip()
    steps, best, best_id = [], -1.0, None
    for line in (run / 'candidate_history.jsonl').read_text().splitlines():
        d = json.loads(line)
        c = d.get('candidate')
        if not c:
            continue
        pi = c.get('parent_info') or ['', None]
        lab = (pi[0] if isinstance(pi, (list, tuple)) else '') or ''
        kind = '' if not lab else ('diverge' if lab.strip() == M.DIV else 'refine' if lab.strip() == M.REF else 'other')
        steps.append((d['iteration'], kind, d.get('parent_id') == best_id))
        score = (c.get('metrics') or {}).get('combined_score')
        if isinstance(score, (int, float)) and score > best:
            best, best_id = score, c['id']
    return steps


def native_steps(run):
    steps, best, best_it, scores = [], -1.0, None, {0: None}
    for line in (run / 'events.jsonl').read_text().splitlines():
        ev = json.loads(line)
        if ev.get('type') != 'iteration' or ev.get('child_score') is None:
            continue
        steps.append((ev['iteration'], ev.get('label') or '', ev.get('parent_iteration') == best_it))
        if ev['child_score'] > best:
            best, best_it = ev['child_score'], ev['iteration']
    return steps


def windows(steps):
    out = []
    for lo, hi in WINDOWS:
        w = [s for s in steps if lo <= s[0] <= hi]
        n = len(w) or 1
        out.append({'iters': f'{lo}-{hi}', 'n': len(w), 'unlabelled': round(sum(s[1] == '' for s in w) / n, 2),
                    'refine': round(sum(s[1] == 'refine' for s in w) / n, 2), 'diverge': round(sum(s[1] == 'diverge' for s in w) / n, 2),
                    'other': round(sum(s[1] == 'other' for s in w) / n, 2), 'best_parent': round(sum(s[2] for s in w) / n, 2)})
    return out


def main() -> None:
    result = {}
    for camp in CAMPAIGNS:
        for run in sorted(p for p in camp.glob('*_s4?') if p.is_dir()):
            stock = (run / 'candidate_history.jsonl').exists()
            result[f'{camp.parent.parent.name}/{run.name}'] = {'engine': 'stock' if stock else 'trace',
                                                               'windows': windows(stock_steps(run) if stock else native_steps(run))}
    groups = {'stock': [v for v in result.values() if v['engine'] == 'stock'], 'trace': [v for v in result.values() if v['engine'] == 'trace']}
    summary = {}
    for name, runs in groups.items():
        summary[name] = []
        for i, (lo, hi) in enumerate(WINDOWS):
            rows = [r['windows'][i] for r in runs if r['windows'][i]['n']]
            summary[name].append({'iters': f'{lo}-{hi}', 'runs': len(rows), **{k: round(sum(r[k] for r in rows) / len(rows), 2)
                                                                             for k in ('unlabelled', 'refine', 'diverge', 'other', 'best_parent')}})
    for name, rows in summary.items():
        for row in rows:
            print(name, json.dumps(row))
    (HERE.parent / 'results' / 'strategy_timeline.json').write_text(json.dumps({'summary': summary, 'runs': result}, indent=1) + '\n')


if __name__ == '__main__':
    main()
