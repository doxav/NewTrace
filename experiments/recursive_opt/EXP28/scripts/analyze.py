"""EXP28 analysis (CPU only): per-call best-so-far points for the four new arms and the EXP22 Trace baselines, plus
look-ahead discovery and the SciPy yield of each variation mode. Signal references from EXP25-27 are read from
EXP27/results/learning_curves.json by the notebook.

Points are (solution call, valid score, causal) per candidate:
  coevolution arms  the run's own audit (evaluations.jsonl) mapped through events.jsonl, as EXP27 learning_curves.py
  trainer arms      evaluations.jsonl, each tagged with the number of solution calls made so far
  EXP22 baselines   candidate_history.jsonl sources re-scored with EXP25's white-box evaluator (valid score + causality)

Usage: python analyze.py results/runs_<stamp>
"""
import functools
import json
import sys
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
sys.path.insert(0, str(EXP / 'EXP27' / 'scripts'))
sys.path.insert(0, str(EXP / 'EXP25' / 'signal'))
import whitebox as W  # noqa: E402

import learning_curves as LC  # noqa: E402
from mechanisms import SCIPY  # noqa: E402

W.evaluate = functools.lru_cache(maxsize=None)(W.evaluate)
STRONG = 0.615  # look-ahead threshold of key_findings.ipynb (best causal score ever seen, 0.565, + 0.05)
EXP22_RUNS = {'EXP22 Trace fixed policy': ['strict_signal_processing_TRACE-FIXED_*'],
              'EXP22 Trace recursive': ['strict_signal_processing_TRACE-RECURSIVE_*']}


def trainer_points(run: Path):
    rows = [json.loads(line) for line in (run / 'evaluations.jsonl').read_text().splitlines()]
    return [(r['solution_calls'], r['valid_score'] or 0.0, r.get('causal_fraction') == 1.0, r['sha256']) for r in rows if r['solution_calls'] > 0]


def coevo_points(run: Path):
    return [(it, v, c, None) for it, v, c in LC.native_points(run)]


def exp22_points(history: Path):
    out = []
    for line in history.read_text().splitlines():
        row = json.loads(line)
        source = (row.get('candidate') or {}).get('solution')
        if source:
            metrics = W.evaluate(source)[0]
            out.append((row['iteration'], metrics.get('valid_score') or 0.0, metrics.get('causal_fraction') == 1.0, None))
    return out


def trainer_yield(run: Path):
    """Per mode: calls whose new candidate introduces SciPy while the incumbent best has none (before the first SciPy)."""
    modes = [r['mode'] for r in json.loads((run / 'variation_log.json').read_text())]
    rows = [json.loads(line) for line in (run / 'evaluations.jsonl').read_text().splitlines()]
    src = lambda sha: (run / 'sources' / f'{sha}.py').read_text()  # noqa: E731
    best, best_src, seen, out = -1.0, None, set(), {}
    for r in rows:
        if r['solution_calls'] == 0 or r['sha256'] in seen:
            if r['solution_calls'] == 0 and best_src is None:
                best, best_src = r['valid_score'] or 0.0, src(r['sha256'])
            seen.add(r['sha256'])
            continue
        seen.add(r['sha256'])
        mode = modes[r['solution_calls'] - 1] if r['solution_calls'] - 1 < len(modes) else 'free'
        if best_src is not None and not SCIPY.search(best_src):
            cell = out.setdefault(mode, [0, 0])
            cell[0] += 1
            cell[1] += bool(SCIPY.search(src(r['sha256'])))
        if (r['valid_score'] or 0.0) > best:
            best, best_src = r['valid_score'] or 0.0, src(r['sha256'])
    return {m: {'calls': n, 'scipy': k} for m, (n, k) in out.items()}, {m: modes.count(m) for m in set(modes)}


def coevo_yield(run: Path):
    import instruction_yield as Y
    rows = Y.native(run)
    first = min((it for it, _, s, _ in rows if s and SCIPY.search(s)), default=10 ** 9)
    out, mix = {}, {}
    for it, label, source, parent in rows:
        mix[label or 'none'] = mix.get(label or 'none', 0) + 1
        if it > first or parent is None or SCIPY.search(parent):
            continue
        cell = out.setdefault(label or 'none', [0, 0])
        cell[0] += 1
        cell[1] += bool(source and SCIPY.search(source))
    guards = sum(1 for line in (run / 'events.jsonl').read_text().splitlines() if '"diverge_guard"' in line)
    return {m: {'calls': n, 'scipy': k} for m, (n, k) in out.items()}, {**mix, 'guard_events': guards}


def summarize(points):
    pts = [(it, v, c) for it, v, c, _ in points]
    found = next((it for it, v, c in sorted(pts) if not c and v >= STRONG), None)
    return {'best_valid': round(max(v for _, v, _ in pts), 4), 'best_causal': round(max([LC.INITIAL] + [v for _, v, c in pts if c]), 4),
            'lookahead_at': found, 'last_call': max(it for it, _, _ in pts), 'points': [[it, round(v, 4), int(c)] for it, v, c in pts]}


def main() -> None:
    campaign = Path(sys.argv[1])
    runs = {}
    for run in sorted(p for p in campaign.iterdir() if p.is_dir()):
        trainer = (run / 'variation_log.json').exists()
        arm = run.name.rsplit('_s', 1)[0]
        points = trainer_points(run) if trainer else coevo_points(run)
        yields, mix = trainer_yield(run) if trainer else coevo_yield(run)
        runs[run.name] = {'arm': arm, 'level': 'trainer' if trainer else 'recursive_opt', **summarize(points), 'yield': yields, 'mode_mix': mix}
        print(run.name, {k: runs[run.name][k] for k in ('best_valid', 'best_causal', 'lookahead_at')}, yields, flush=True)
    for name, patterns in EXP22_RUNS.items():
        for pattern in patterns:
            for history in sorted((EXP / 'EXP22' / 'artifacts').glob(f'parallel_trace_*/v9_runtime/runs/{pattern}/candidate_history.jsonl')) + \
                    sorted((EXP / 'EXP22' / 'runs').glob(f'{pattern}/candidate_history.jsonl')):
                points = exp22_points(history)
                if points and max(p[0] for p in points) >= 50:
                    runs[f'baseline/{history.parent.name}'] = {'arm': name, 'level': 'baseline', **summarize(points)}
                    print(history.parent.name, {k: runs[f'baseline/{history.parent.name}'][k] for k in ('best_valid', 'best_causal', 'lookahead_at')}, flush=True)
    arms = {}
    for r in runs.values():
        arms.setdefault(r['arm'], []).append(r)
    summary = {}
    for arm, rs in arms.items():
        yields = {}
        for r in rs:
            for m, c in (r.get('yield') or {}).items():
                yields.setdefault(m, [0, 0])
                yields[m][0] += c['calls']
                yields[m][1] += c['scipy']
        summary[arm] = {'runs': len(rs), 'median_best_valid': round(median(r['best_valid'] for r in rs), 4),
                        'median_best_causal': round(median(r['best_causal'] for r in rs), 4),
                        'lookahead_runs': f"{sum(r['lookahead_at'] is not None for r in rs)}/{len(rs)}",
                        'lookahead_by_50': sum(r['lookahead_at'] is not None and r['lookahead_at'] <= 50 for r in rs),
                        'lookahead_at': sorted(r['lookahead_at'] for r in rs if r['lookahead_at'] is not None),
                        'scipy_yield': {m: f'{k}/{n}' for m, (n, k) in yields.items()}}
        print(f'{arm:28s}', json.dumps(summary[arm]))
    (campaign / 'analysis.json').write_text(json.dumps({'threshold': STRONG, 'summary': summary, 'runs': runs}, indent=1) + '\n')


if __name__ == '__main__':
    main()
