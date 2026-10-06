"""EXP26 analysis: endpoints, mechanism variables and the pre-registered decision rules (PROTOCOL.md).

Per run: P1 best valid score; P2 best valid score among strictly causal candidates; share of distinct candidates
importing SciPy; share of iterations using DIVERGE (native arms); whether the labels name SciPy. Per arm: medians,
then rules R1-R5. Reuses EXP25's analyzer and evaluator (imported); the evaluator is memoised so stock candidates
are re-scored once for both endpoints.
Usage: python analyze.py results/<campaign>
"""

import functools
import importlib.util
import json
import re
import sys
from pathlib import Path
from statistics import median

_spec = importlib.util.spec_from_file_location('exp25_analyze', Path(__file__).resolve().parents[2] / 'EXP25' / 'scripts' / 'analyze.py')
E25A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E25A)
W = E25A.W
W.evaluate = functools.lru_cache(maxsize=None)(W.evaluate)
SCIPY = re.compile(r'^\s*(import scipy|from scipy)', re.M)
EVOX_RANGE_LOW = 0.68  # EXP25 evox_stock P1 range 0.679-0.723


def distinct_sources(run: Path, stock: bool) -> list:
    if stock:
        rows = (json.loads(line).get('candidate') for line in (run / 'candidate_history.jsonl').read_text().splitlines())
        return sorted({c['solution'] for c in rows if c})
    return [p.read_text() for p in sorted((run / 'sources').glob('*.py'))]


def best_causal(run: Path, stock: bool, sources: list):
    if stock:
        pairs = [(W.evaluate(s)[0].get('valid_score', 0.0), W.evaluate(s)[0].get('causal_fraction', 0.0)) for s in sources]
    else:
        pairs = [(r.get('valid_score') or 0.0, r.get('causal_fraction') or 0.0)
                 for r in map(json.loads, (run / 'evaluations.jsonl').read_text().splitlines())]
    causal = [v for v, c in pairs if c == 1.0]
    return (round(max(causal), 4) if causal else None), len(causal), len(pairs)


def label_shares(run: Path) -> dict:
    if not (run / 'events.jsonl').exists():
        return {}
    labels = [e.get('label') for e in map(json.loads, (run / 'events.jsonl').read_text().splitlines()) if e.get('type') == 'iteration']
    return {name: round(labels.count(name) / len(labels), 3) for name in ('diverge', 'refine', '')} if labels else {}


def run_row(run: Path, cache: dict) -> dict:
    stats = E25A.run_stats(run, cache)
    stock = stats['arm'] == 'evox_stock'
    sources = distinct_sources(run, stock)
    p2, n_causal, n_eval = best_causal(run, stock, sources)
    labels = json.loads((run / 'labels.json').read_text()) if (run / 'labels.json').exists() else {}
    return {**stats, 'p1_best_valid': stats['best_valid'], 'p2_best_valid_causal': p2, 'causal_candidates': f'{n_causal}/{n_eval}',
            'scipy_share': round(sum(bool(SCIPY.search(s)) for s in sources) / len(sources), 3) if sources else None,
            'label_shares': label_shares(run), 'labels_name_scipy': {k: 'scipy' in (labels.get(k) or '').lower() for k in ('diverge', 'refine')},
            'labels_fallback': labels.get('fallback')}


def med(rows: list, key, default=None):
    values = [key(r) for r in rows if key(r) is not None]
    return round(median(values), 4) if values else default


def decide(arms: dict) -> dict:
    p1 = {a: med(g, lambda r: r['p1_best_valid']) for a, g in arms.items()}
    p2 = {a: med(g, lambda r: r['p2_best_valid_causal']) for a, g in arms.items()}
    scipy = {a: med(g, lambda r: r['scipy_share']) for a, g in arms.items()}
    diverge = {a: med(g, lambda r: r['label_shares'].get('diverge')) for a, g in arms.items()}
    sl, pkg = p1.get('native_stocklabels'), p1.get('native_pkg')
    rules = {
        'gate_evox_reproduces_exp25': None if p1.get('evox_stock') is None else p1['evox_stock'] >= EVOX_RANGE_LOW,
        'R1_label_content_explains_gap': None if sl is None else sl >= EVOX_RANGE_LOW and (scipy.get('native_stocklabels') or 0) >= 0.30,
        'R2_compact_port_adequate': None if sl is None or pkg is None else abs(pkg - sl) <= 0.02,
        'R3_selection_implicated': None if sl is None else sl < 0.63 and (diverge.get('native_stocklabels') or 0) < 0.20,
        'R4_gap_elsewhere': None if sl is None else sl < 0.63 and (diverge.get('native_stocklabels') or 0) >= 0.20,
        'R5_lead_is_lookahead_only': None if len([v for v in p2.values() if v is not None]) < 2 else max(v for v in p2.values() if v is not None) - min(v for v in p2.values() if v is not None) <= 0.03,
    }
    return {'median_p1': p1, 'median_p2': p2, 'median_scipy_share': scipy, 'median_diverge_share': diverge, 'rules': rules}


def main() -> None:
    campaign = Path(sys.argv[1])
    cache: dict = {}
    rows = [run_row(run, cache) for run in sorted(campaign.iterdir()) if run.is_dir() and (run / 'summary.json').exists()]
    arms: dict = {}
    for row in rows:
        arms.setdefault(row['arm'], []).append(row)
    decision = decide(arms)
    (campaign / 'analysis.json').write_text(json.dumps({'runs': rows, **decision}, indent=1) + '\n')
    for row in rows:
        print(f"{row['run']:26s} P1 {row['p1_best_valid']:.4f}  P2 {row['p2_best_valid_causal']}  scipy {row['scipy_share']}  "
              f"labels {row['label_shares']}  fallback {row['labels_fallback']}")
    print(json.dumps(decision, indent=1))


if __name__ == '__main__':
    main()
