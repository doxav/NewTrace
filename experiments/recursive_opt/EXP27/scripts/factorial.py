"""EXP27 Part C: 2x2 replay, causal cue (on/off) x parent lineage (Trace / score-matched stock), per-call SciPy discovery.

Prompts: recorded EXP26 native solution prompts with distinct parents that are non-SciPy, causal and valid on all
five signals (up to --per-run per run, spread over the run). Every prompt is given its run's DIVERGE label in the
label slot, so the label is fixed within a prompt and identical across the four cells.
For each prompt, the "# Current Solution" section is re-rendered from (parent source, whitebox metrics) with the
operator's own formatting, for either the recorded Trace parent or the stock EvoX program (non-SciPy, causal, all
signals valid, never used twice) whose valid score is closest. Cue off drops the `causal_fraction` line. Everything
else in the prompt (history, context programs, label, task) is the recorded text. Sizing: scripts/power.py.
"""
import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prompt_ablation as A  # noqa: E402

ROOT = A.ROOT
EXP26 = ROOT / 'EXP26' / 'results' / 'runs_20261006T112918'
STOCK_RUNS = sorted((ROOT / 'EXP25' / 'results' / 'runs_20260930T235849').glob('evox_stock_s4?')) + sorted(EXP26.glob('evox_stock_s4?'))
CELLS = [(cue, parent) for cue in ('cue_on', 'cue_off') for parent in ('trace', 'stock')] + [('stock_like', 'trace')]  # 5th cell: powered H8 re-test


def full_metrics(source: str) -> tuple:
    metrics, artifacts = A.W.evaluate(source)
    return dict(metrics), dict(artifacts or {})


def eligible(metrics: dict) -> bool:
    return metrics.get('causal_fraction') == 1.0 and metrics.get('valid_success_rate') == 1.0 and not metrics.get('error')


def render(source: str, metrics: dict, artifacts: dict, cue: bool) -> str:
    """Program Information + code + Evaluator Feedback, as CoevolutionOperator._program renders a GuidedEvaluator parent."""
    shown = {'guided_score': metrics['valid_score'], **{k: metrics[k] for k in ('combined_score', 'success_rate', 'valid_score',
                                                                                 'valid_success_rate', 'fallback_signals', 'causal_fraction')}}
    if not cue:
        shown.pop('causal_fraction')
    lines = ['## Program Information\n', f"guided_score: {shown.pop('guided_score'):.4f}\n",
             'Score breakdown:' + ''.join(f'\n  - {k}: {v:.4f}' if isinstance(v, float) else f'\n  - {k}: {v}' for k, v in shown.items()) + '\n',
             f'\n```python\n{source}\n```\n']
    if artifacts.get('feedback'):
        lines.append(f"\n## Evaluator Feedback\n{str(artifacts['feedback'])[:2000]}\n")
    return ''.join(lines)


def rebuild(user: str, source: str, metrics: dict, artifacts: dict, cue: bool, label: str) -> str:
    head, sep, current = user.rpartition('\n# Current Solution\n')
    start = current.index('## Program Information\n')
    current = f'\n{label}\n\n' + current[start:]
    start = current.index('## Program Information\n')
    end = current.index('\n## projection\n') if '\n## projection\n' in current else current.index('\n# Task\n')
    current = current[:start] + render(source, metrics, artifacts, cue) + current[end:]
    score = f"{metrics['valid_score']:.4f}"
    head = re.sub(r'- Main Metrics: guided_score=[0-9.]+', f'- Main Metrics: guided_score={score}', head, count=1)
    head = re.sub(r'(- Focus areas: - [^\n]*-> )[0-9.]+', lambda m: m.group(1) + score, head, count=1)
    if not cue:
        head = re.sub(r'\n  - causal_fraction: [^\n]*', '', head)  # context programs too: the cue is removed everywhere
    return head + sep + current


def pick(n_prompts: int, per_run: int) -> list:
    prompts, seen = [], set()
    for run in sorted(EXP26.glob('native*_s4?')):
        label = json.loads((run / 'labels.json').read_text())['diverge']
        chosen = []
        for line in (run / 'transcripts.jsonl').read_text().splitlines():
            t = json.loads(line)
            user = t['messages'][1]['content'] if t['role'] == 'forward' else ''
            if '\n# Current Solution\n' not in user:
                continue
            source = A.parent_of(user)
            if source in seen or A.SCIPY.search(source) or not eligible(full_metrics(source)[0]):
                continue
            seen.add(source)
            chosen.append({'run': run.name, 'system': t['messages'][0]['content'], 'user': user, 'label': label})
        step = max(1, len(chosen) // per_run)
        prompts += chosen[::step][:per_run]
    random.Random(0).shuffle(prompts)
    return prompts[:n_prompts]


def stock_pool() -> list:
    pool, seen = [], set()
    for run in STOCK_RUNS:
        for line in (run / 'candidate_history.jsonl').read_text().splitlines():
            c = json.loads(line).get('candidate')
            if not c or c['solution'] in seen or A.SCIPY.search(c['solution']):
                continue
            seen.add(c['solution'])
            m, art = full_metrics(c['solution'])
            if eligible(m):
                pool.append({'run': f'{run.parents[2].name}/{run.name}', 'source': c['solution'], 'metrics': m, 'artifacts': art})
    return pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', type=int, default=20)
    parser.add_argument('--per-run', type=int, default=3)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--out', required=True)
    parser.add_argument('--deadline-s', type=float, default=300.0)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--mock', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        print(json.dumps(summarize(out), indent=1))
        return
    prompts, pool = pick(args.prompts, args.per_run), stock_pool()
    design = []
    for i, p in enumerate(prompts):
        source = A.parent_of(p['user'])
        m, art = full_metrics(source)
        match = min(pool, key=lambda s: abs(s['metrics']['valid_score'] - m['valid_score']))
        pool.remove(match)
        design.append({'id': f"{p['run']}#{i}", 'system': p['system'], 'user': p['user'], 'label': p['label'],
                       'trace': {'source': source, 'metrics': m, 'artifacts': art}, 'stock': match})
    (out / 'design.json').write_text(json.dumps([{'id': d['id'], 'trace_valid': d['trace']['metrics']['valid_score'], 'trace_chars': len(d['trace']['source']),
                                                  'stock_run': d['stock']['run'], 'stock_valid': d['stock']['metrics']['valid_score'],
                                                  'stock_chars': len(d['stock']['source'])} for d in design], indent=1) + '\n')
    llm = (lambda messages, **_: 'no change') if args.mock else A.E25.RoleClient('factorial', 'novita', out / 'calls.jsonl')
    jobs = [(d, cue, parent, s) for d in design for cue, parent in CELLS for s in range(args.samples)]
    random.Random(1).shuffle(jobs)
    done = set()
    if (out / 'completions.jsonl').exists():  # resume: the design is deterministic, keep finished completions
        done = {(r['prompt'], r['cue'], r['parent'], r['sample']) for r in map(json.loads, (out / 'completions.jsonl').read_text().splitlines())}
    jobs = [j for j in jobs if (j[0]['id'], j[1], j[2], j[3]) not in done]
    print(f'{len(done)} completions kept, {len(jobs)} to run', flush=True)
    callers = ThreadPoolExecutor(max_workers=args.workers * 4)  # abandoned (hung) requests keep their thread here

    def call(messages):
        # OpenRouter keep-alive bytes defeat the client's read timeout: enforce a wall-clock deadline and retry
        for attempt in range(args.retries):
            try:
                return callers.submit(llm, messages).result(timeout=args.deadline_s)
            except FutureTimeout:
                with (out / 'deadlines.jsonl').open('a') as stream:
                    stream.write(json.dumps({'attempt': attempt}) + '\n')
        return ''

    def one(job):
        d, cue, parent, sample = job
        par = d[parent]
        user = rebuild(d['user'], par['source'], par['metrics'], par['artifacts'], cue == 'cue_on', d['label'])
        if cue == 'stock_like':
            user = A.stock_like(user)
        reply = call([{'role': 'system', 'content': d['system']}, {'role': 'user', 'content': user}])
        child, error = A.apply_search_replace(par['source'], reply)
        if not reply:
            error = 'deadline exceeded on every attempt'
        row = {'prompt': d['id'], 'cue': cue, 'parent': parent, 'sample': sample, 'parent_valid': par['metrics']['valid_score'],
               'applied': child is not None, 'apply_error': error}
        if child is not None:
            row.update(scipy=bool(A.SCIPY.search(child)), child=A.score(child))
        return row

    with ThreadPoolExecutor(args.workers) as pool_, (out / 'completions.jsonl').open('a') as stream:
        for future in as_completed([pool_.submit(one, j) for j in jobs]):
            stream.write(json.dumps(future.result()) + '\n')
            stream.flush()
    callers.shutdown(wait=False, cancel_futures=True)
    print(json.dumps(summarize(out), indent=1), flush=True)
    os._exit(0)  # do not wait for abandoned (hung) request threads


def summarize(out: Path, boot: int = 4000) -> dict:
    """Primary: SciPy share over ALL completions (failed applies count as no). Main effects and interaction as pooled
    differences of per-prompt cell rates, with 95% prompt-bootstrap CIs."""
    rows = [json.loads(line) for line in (out / 'completions.jsonl').read_text().splitlines()]
    ids = sorted({r['prompt'] for r in rows})
    ok = lambda r: r['applied'] and r.get('child', {}).get('valid_score') is not None  # noqa: E731
    outcomes = {'scipy': lambda r: bool(r.get('scipy')),
                'scipy_lookahead': lambda r: bool(r.get('scipy')) and ok(r) and (r['child'].get('causal_fraction') or 0) < 1,
                'beats_parent': lambda r: ok(r) and r['child']['valid_score'] > r['parent_valid'] + 1e-9,
                'applied': lambda r: r['applied']}
    result = {'n': len(rows), 'prompts': len(ids), 'cells': {}, 'effects': {}}
    rate = {}
    for name, f in outcomes.items():
        grid = np.zeros((len(ids), 2, 2))
        for i, pid in enumerate(ids):
            for a, cue in enumerate(('cue_off', 'cue_on')):
                for b, parent in enumerate(('stock', 'trace')):
                    cell = [r for r in rows if r['prompt'] == pid and r['cue'] == cue and r['parent'] == parent]
                    grid[i, a, b] = np.mean([f(r) for r in cell]) if cell else np.nan
        rate[name] = grid
    for cue_i, cue in enumerate(('cue_off', 'cue_on')):
        for par_i, parent in enumerate(('stock', 'trace')):
            cell = [r for r in rows if r['cue'] == cue and r['parent'] == parent]
            result['cells'][f'{cue}/{parent}'] = {'n': len(cell), **{k: round(float(np.nanmean(rate[k][:, cue_i, par_i])), 3) for k in outcomes},
                                                  'best_valid': round(max([r['child']['valid_score'] for r in cell if ok(r)] or [0]), 4)}
    for name, f in outcomes.items():
        cell = [r for r in rows if r['cue'] == 'stock_like']
        if cell:
            result['cells']['stock_like/trace'] = result['cells'].get('stock_like/trace', {'n': len(cell)})
            result['cells']['stock_like/trace'][name] = round(float(np.mean([f(r) for r in cell])), 3)
    rng = np.random.default_rng(0)
    contrasts = {'cue_off_minus_on': lambda g: (g[:, 0, :] - g[:, 1, :]).mean(axis=1),
                 'stock_minus_trace_parent': lambda g: (g[:, :, 0] - g[:, :, 1]).mean(axis=1),
                 'interaction': lambda g: (g[:, 0, 0] - g[:, 1, 0]) - (g[:, 0, 1] - g[:, 1, 1])}
    for name, f in outcomes.items():
        for cname, c in contrasts.items():
            per = c(rate[name])
            per = per[~np.isnan(per)]
            bs = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(boot)]
            result['effects'][f'{name}:{cname}'] = {'est': round(float(per.mean()), 3), 'ci95': [round(float(np.quantile(bs, q)), 3) for q in (0.025, 0.975)],
                                                    'prompts_positive': int((per > 0).sum()), 'prompts_negative': int((per < 0).sum())}
        per = np.array([np.mean([f(r) for r in rows if r['prompt'] == pid and r['cue'] == 'stock_like']) -
                        np.mean([f(r) for r in rows if r['prompt'] == pid and r['cue'] == 'cue_off' and r['parent'] == 'trace'])
                        for pid in ids if any(r['prompt'] == pid and r['cue'] == 'stock_like' for r in rows)])
        if len(per):
            bs = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(boot)]
            result['effects'][f'{name}:stock_like_minus_cue_off(trace)'] = {'est': round(float(per.mean()), 3), 'ci95': [round(float(np.quantile(bs, q)), 3) for q in (0.025, 0.975)],
                                                                          'prompts_positive': int((per > 0).sum()), 'prompts_negative': int((per < 0).sum())}
    return result


if __name__ == '__main__':
    main()
