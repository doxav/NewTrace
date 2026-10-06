"""EXP27 log-only mechanism analysis over EXP25 and EXP26 campaigns (no LLM). Writes results/mechanisms.json.

Per run: label shares (stock labels matched by exact text), duplicates, first SciPy candidate (iteration, label, score vs
best so far), share of later parents that use SciPy, best score at 42 iterations and overall, and (re-evaluated with the
EXP25 whitebox evaluator) the look-ahead share and best causal score of SciPy and non-SciPy candidates.
"""
import json, re, statistics as st, sys
from pathlib import Path

SCIPY = re.compile(r'^\s*(import scipy|from scipy)', re.M)


def stock_rows(run):
    rows, ids = [], {}
    for line in (run / 'candidate_history.jsonl').read_text().splitlines():
        d = json.loads(line); c = d.get('candidate')
        if not c:
            rows.append({'it': d['iteration'], 'src': None, 'score': None, 'label': None, 'nctx': 0, 'parent': d.get('parent_id')}); continue
        m = c.get('metrics') or {}
        pi = c.get('parent_info') or ['', None]
        lab = pi[0] if isinstance(pi, (list, tuple)) else ''
        kind = '' if not lab else ('diverge' if lab.strip() == DIV else 'refine' if lab.strip() == REF else 'other')
        ids[c['id']] = c['solution']
        rows.append({'it': d['iteration'], 'src': c['solution'], 'score': m.get('combined_score'), 'label': kind,
                     'nctx': len(c.get('other_context_ids') or []), 'parent': c.get('parent_id')})
    for r in rows:
        r['parent_src'] = ids.get(r['parent'])
    return rows


def native_rows(run):
    srcs = {p.stem: p.read_text() for p in (run / 'sources').glob('*.py')}
    by_score = {}
    for line in (run / 'evaluations.jsonl').read_text().splitlines():
        e = json.loads(line); by_score.setdefault(round(e['combined_score'] or 0, 12), e['sha256'])
    rows, it_src = [], {}
    for line in (run / 'events.jsonl').read_text().splitlines():
        e = json.loads(line)
        if e.get('type') != 'iteration':
            continue
        sc = e.get('child_score'); sha = by_score.get(round(sc, 12)) if sc is not None else None
        it_src[e['iteration']] = srcs.get(sha)
        rows.append({'it': e['iteration'], 'src': srcs.get(sha), 'score': sc, 'label': e.get('label') or '',
                     'nctx': len(e.get('context_iterations') or []), 'parent_it': e.get('parent_iteration')})
    for r in rows:
        r['parent_src'] = it_src.get(r['parent_it']) if r['parent_it'] else None
    return rows


def metrics(rows):
    ok = [r for r in rows if r['src'] is not None and r['score'] is not None]
    seen, dup, unchanged = set(), 0, 0
    for r in ok:
        dup += r['src'] in seen; seen.add(r['src']); unchanged += r['src'] == r.get('parent_src')
    sci = [bool(SCIPY.search(r['src'])) for r in ok]
    first = next((i for i, s in enumerate(sci) if s), None)
    best, bi = -1, None
    for i, r in enumerate(ok):
        if r['score'] > best: best, bi = r['score'], i
    labs = [r['label'] for r in rows if r['label'] is not None]
    s_sc = [r['score'] for r, s in zip(ok, sci) if s]; n_sc = [r['score'] for r, s in zip(ok, sci) if not s]
    return {'iters': len(rows), 'failed': len(rows) - len(ok), 'dup_frac': round(dup / max(len(ok), 1), 2),
            'unchanged_frac': round(unchanged / max(len(ok), 1), 2),
            'lab_div': round(labs.count('diverge') / max(len(labs), 1), 2), 'lab_ref': round(labs.count('refine') / max(len(labs), 1), 2),
            'ctx_mean': round(st.mean(r['nctx'] for r in rows), 1),
            'first_scipy': first, 'scipy_after': round(sum(sci[first:]) / len(sci[first:]), 2) if first is not None else None,
            'med_scipy': round(st.median(s_sc), 3) if s_sc else None, 'med_noscipy': round(st.median(n_sc), 3) if n_sc else None,
            'best': round(best, 4), 'best_at': bi, 'src_len': int(st.median(len(r['src']) for r in ok))}



def best_by(rows, n):
    xs = [r['score'] for r in rows if r['score'] is not None and int(r['it']) <= n]
    return round(max(xs), 4) if xs else None


def discovery(rows):
    ok = [r for r in rows if r['score'] is not None and r['src']]
    best, first, parents = -1.0, None, []
    for r in ok:
        sci = bool(SCIPY.search(r['src']))
        if sci and first is None:
            first = {'iteration': int(r['it']), 'label': r['label'] or '', 'score': round(r['score'], 4), 'best_before': round(best, 4)}
        if first:
            parents.append(bool(r.get('parent_src') and SCIPY.search(r['parent_src'])))
        best = max(best, r['score'])
    return {'first_scipy': first, 'later_parent_scipy': round(sum(parents) / len(parents), 2) if parents else None}


def causal_split(rows, W):
    out = {}
    srcs = list(dict.fromkeys(r['src'] for r in rows if r['src']))
    for name, group in (('scipy', [s for s in srcs if SCIPY.search(s)]), ('no_scipy', [s for s in srcs if not SCIPY.search(s)])):
        ev = [(W.evaluate(s)[0].get('valid_score') or 0.0, W.evaluate(s)[0].get('causal_fraction')) for s in group]
        causal = [v for v, c in ev if c == 1.0]
        out[name] = {'n': len(ev), 'lookahead_share': round(1 - len(causal) / len(ev), 2), 'best_valid': round(max(v for v, _ in ev), 4),
                     'best_causal': round(max(causal), 4) if causal else None} if ev else None
    return out


def main():
    import functools, importlib.util
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    sys.path.insert(0, str(root / 'EXP25' / 'signal'))
    import whitebox as W
    W.evaluate = functools.lru_cache(maxsize=None)(W.evaluate)
    campaigns = [root / 'EXP25' / 'results' / 'runs_20260930T235849', root / 'EXP26' / 'results' / 'runs_20261006T112918']
    result = {}
    for camp in campaigns:
        for run in sorted(camp.glob('*_s4?')):
            stock = (run / 'candidate_history.jsonl').exists()
            if stock:
                L = json.loads((run / 'labels.json').read_text()) if (run / 'labels.json').exists() else {'diverge': '?', 'refine': '?'}
                globals().update(DIV=L['diverge'].strip(), REF=L['refine'].strip())
            rows = stock_rows(run) if stock else native_rows(run)
            key = f"{camp.parents[1].name}/{run.name}"
            result[key] = {**metrics(rows), 'best_at_42': best_by(rows, 42), **discovery(rows), **causal_split(rows, W)}
            print(key, json.dumps(result[key]), flush=True)
    (here.parent / 'results' / 'mechanisms.json').write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
