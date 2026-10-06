"""H6/H8 probe: replay recorded Trace solution prompts with one prompt factor removed, score completions locally.

Variants (applied to the recorded user message; system message unchanged):
  T0 recorded            the prompt exactly as EXP26 sent it
  T1 no_causal_cue       'causal_fraction' lines removed (single factor)
  T2 stock_like          T1 + valid/fallback/guided metric lines and evaluator-feedback sections removed, stock task text

Each completion's SEARCH/REPLACE diffs are applied to the parent shown in the prompt and evaluated by the EXP25
whitebox evaluator (no LLM). Outputs one JSON line per completion. --mock replaces the LLM with a scripted reply.
"""
import argparse
import importlib.util
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'EXP25' / 'signal'))
_spec = importlib.util.spec_from_file_location('exp25_run_signal', ROOT / 'EXP25' / 'scripts' / 'run_signal.py')
E25 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E25)
W = E25.W
from opto.features.recursive_opt.coevolution.operator import apply_search_replace  # noqa: E402

SCIPY = re.compile(r'^\s*(import scipy|from scipy)', re.M)
STOCK_TASK = (ROOT.parents[1].parent / 'evo-compare' / 'repos' / 'skydiscover' / 'skydiscover' / 'optimize' / 'context_builder'
              / 'default' / 'templates' / 'diff_user_message.txt')
DIAGNOSTIC = ('causal_fraction', 'valid_score', 'valid_success_rate', 'fallback_signals', 'guided_score')


def no_causal_cue(text: str) -> str:
    return re.sub(r'\n  - causal_fraction: [^\n]*', '', text)


def stock_like(text: str) -> str:
    text = re.sub(r'\n  - (%s): [^\n]*' % '|'.join(DIAGNOSTIC), '', text)
    text = text.replace('guided_score', 'combined_score')
    text = re.sub(r'\n## (Evaluator Feedback|projection|constraints)\n.*?(?=\n#{1,3} |\n```|\Z)', '', text, flags=re.S)
    head, _, _ = text.partition('# Task\n')
    task = STOCK_TASK.read_text().split('# Task\n', 1)[1].replace('{timeout_warning}', '')
    return head + '# Task\n' + task.strip()


VARIANTS = {'T0_recorded': lambda t: t, 'T1_no_causal_cue': no_causal_cue, 'T2_stock_like': stock_like}


def parent_of(user: str) -> str:
    section = user.split('\n# Current Solution\n', 1)[1]
    return re.search(r'```python\n(.*?)\n```', section, re.S).group(1)


def pick_prompts(campaign: Path, runs: list, per_run: int) -> list:
    """First `per_run` DIVERGE-labelled solution prompts of each run whose parent does not use SciPy (the discovery situation)."""
    picked = []
    for name in runs:
        n, diverge = 0, json.loads((campaign / name / 'labels.json').read_text())['diverge'].strip()[-200:]
        for line in (campaign / name / 'transcripts.jsonl').read_text().splitlines():
            t = json.loads(line)
            if t['role'] != 'forward' or '\n# Current Solution\n' not in t['messages'][1]['content']:
                continue
            user = t['messages'][1]['content']
            if diverge not in user or SCIPY.search(parent_of(user)):
                continue
            picked.append({'run': name, 'index': n, 'system': t['messages'][0]['content'], 'user': user})
            n += 1
            if n == per_run:
                break
    return picked


def score(source: str) -> dict:
    metrics, _ = W.evaluate(source)
    return {k: metrics.get(k) for k in ('combined_score', 'valid_score', 'causal_fraction', 'error')}


def summarize(out: Path) -> dict:
    """Per variant: applied rate; among applied: look-ahead share (causal_fraction < 1), SciPy share, beats-parent
    rate (valid score), best valid. Per prompt: look-ahead share, for the direction check of the decision rules."""
    rows = [json.loads(line) for line in (out / 'completions.jsonl').read_text().splitlines()]
    result = {}
    for variant in VARIANTS:
        vr = [r for r in rows if r['variant'] == variant]
        ok = [r for r in vr if r['applied'] and r['child'].get('valid_score') is not None]
        share = lambda xs, f: round(sum(map(f, xs)) / len(xs), 3) if xs else None  # noqa: E731
        lookahead = lambda r: (r['child'].get('causal_fraction') or 0) < 1  # noqa: E731
        result[variant] = {'n': len(vr), 'applied': share(vr, lambda r: r['applied']), 'lookahead': share(ok, lookahead),
                           'scipy': share(ok, lambda r: r['scipy']),
                           'beats_parent': share(ok, lambda r: (r['child']['valid_score'] or 0) > (r['parent']['valid_score'] or 0) + 1e-9),
                           'best_valid': round(max((r['child']['valid_score'] or 0) for r in ok), 4) if ok else None,
                           'per_prompt_lookahead': {f"{r['run']}#{r['prompt']}": share([x for x in ok if x['run'] == r['run']], lookahead) for r in vr}}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', default=str(ROOT / 'EXP26' / 'results' / 'runs_20261006T112918'))
    parser.add_argument('--runs', default='native_stocklabels_s42,native_stocklabels_s43,native_stocklabels_s44,native_pkg_s43')
    parser.add_argument('--per-run', type=int, default=1)
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--out', required=True)
    parser.add_argument('--mock', action='store_true')
    parser.add_argument('--summarize', action='store_true', help='only summarize an existing --out')
    args = parser.parse_args()
    out = Path(args.out)
    if args.summarize:
        print(json.dumps(summarize(out), indent=1))
        return
    out.mkdir(parents=True, exist_ok=True)
    prompts = pick_prompts(Path(args.campaign), args.runs.split(','), args.per_run)
    llm = (lambda messages, **_: 'no change') if args.mock else E25.RoleClient('ablation', 'novita', out / 'calls.jsonl')
    parents = {}
    for p in prompts:
        source = parent_of(p['user'])
        parents[source] = score(source)
    jobs = [(p, v, s) for p in prompts for v in VARIANTS for s in range(args.samples)]
    random.Random(0).shuffle(jobs)  # interleave variants so provider drift hits all equally

    def one(job):
        p, variant, sample = job
        user = VARIANTS[variant](p['user'])
        reply = llm([{'role': 'system', 'content': p['system']}, {'role': 'user', 'content': user}])
        parent = parent_of(p['user'])
        child, error = apply_search_replace(parent, reply)
        row = {'run': p['run'], 'prompt': p['index'], 'variant': variant, 'sample': sample, 'prompt_chars': len(user),
               'parent': parents[parent], 'applied': child is not None, 'apply_error': error}
        if child is not None:
            row.update(scipy=bool(SCIPY.search(child)), child=score(child))
        return row

    with ThreadPoolExecutor(args.workers) as pool, (out / 'completions.jsonl').open('a') as stream:
        for row in pool.map(one, jobs):
            stream.write(json.dumps(row) + '\n')
            stream.flush()
    print(f'{len(jobs)} completions over {len(prompts)} prompts -> {out}')
    print(json.dumps(summarize(out), indent=1))


if __name__ == '__main__':
    main()
