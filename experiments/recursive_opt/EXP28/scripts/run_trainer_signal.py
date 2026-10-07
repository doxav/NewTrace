"""EXP28 Trainer arms: VariationSearch (PrioritySearch + explicit mutation intent) on the Signal task.

  vs_stagnation   variation_schedule='stagnation', patience=5, refine_after_gain=2: DIVERGE after 5 steps without
                  improvement, REFINE for 2 steps after a DIVERGE that improved, the optimizer's own instruction otherwise
  vs_combine      variation_schedule='periodic', period=3, num_inspirations=2: every 3rd step DIVERGE + COMBINE with 2
                  random non-elite candidates from memory shown as inspirations

The program is one trainable code node (the full SkyDiscover program), optimized by OptoPrimeV2 with the task's
system message appended to its instruction. Each step = 1 solution LLM call (num_candidates=1, num_proposals=1).
Scoring matches the coevolution arms: same projections (compile check + per-call fallback), same white-box evaluator,
valid_score as the score, white-box per-signal feedback, causal_fraction hidden from the LLM (EXP27 Part D condition).
Every evaluation is logged with the number of solution calls made so far, for best-so-far curves per call.

Usage: python run_trainer_signal.py --arm vs_stagnation --seed 42 --out DIR [--steps 100] [--mock]
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
sys.path.insert(0, os.environ.get('TRACE_ROOT', str(Path.home() / 'code' / 'Trace')))
sys.path.insert(0, str(EXP / 'EXP25' / 'signal'))
import whitebox as W  # noqa: E402
import yaml  # noqa: E402

from opto import trace  # noqa: E402
from opto.features.recursive_opt.coevolution.projections import ProjectionError, make_projection  # noqa: E402
from opto.optimizers import OptoPrimeV2  # noqa: E402
from opto.trainer.algorithms import VariationSearch  # noqa: E402
from opto.trainer.guide import Guide  # noqa: E402
from opto.utils.llm import DummyLLM  # noqa: E402

_spec = importlib.util.spec_from_file_location('exp27_cue', EXP / 'EXP27' / 'scripts' / 'run_trace_cue.py')
CUE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CUE)
MODEL, HIDDEN = CUE.E25.MODEL, CUE.HIDDEN
ARMS = {'vs_stagnation': {'variation_schedule': 'stagnation', 'patience': 5, 'refine_after_gain': 2, 'num_inspirations': 0},
        'vs_combine': {'variation_schedule': 'periodic', 'period': 3, 'refine_after_gain': 2, 'num_inspirations': 2, 'inspiration_mode': 'always'},
        # ablation 2026-10-07: inspirations decoupled from the schedule (inspiration_mode / inspiration_style)
        'vs_default': {},  # VariationSearch defaults = vs_stagnation (regression check of the winner)
        'vs_stag_alt_combine': {'inspiration_mode': 'alternate', 'inspiration_style': 'combine'},
        'vs_stag_alt_context': {'inspiration_mode': 'alternate', 'inspiration_style': 'context'},
        'vs_stag_always_context': {'inspiration_mode': 'always', 'inspiration_style': 'context'},
        'vs_periodic_diverge': {'variation_schedule': 'periodic', 'period': 3}}


class SignalGuide(Guide):
    """Projects, evaluates (cached by source), logs, and returns (valid_score, feedback without the causal cue)."""

    def __init__(self, out: Path, calls: dict) -> None:
        self.out, self.calls, self.cache = out, calls, {}
        self.projections = [make_projection(p) for p in W.PROJECTIONS]
        (out / 'sources').mkdir(exist_ok=True)

    def evaluate(self, source: str):
        if source in self.cache:
            return self.cache[source]
        projected, notes = source, []
        try:
            for projection in self.projections:
                projected, note = projection(projected)
                notes.append(note)
            metrics, artifacts = W.evaluate(projected)
        except ProjectionError as error:
            metrics, artifacts = {'valid_score': 0.0, 'combined_score': 0.0, 'error': f'projection rejected the program: {error}'}, {}
        digest = hashlib.sha256(projected.encode()).hexdigest()
        (self.out / 'sources' / f'{digest}.py').write_text(projected)
        with (self.out / 'evaluations.jsonl').open('a') as stream:
            stream.write(json.dumps({'sha256': digest, 'solution_calls': self.calls['n'],
                                     **{k: metrics.get(k) for k in ('combined_score', 'valid_score', 'success_rate', 'valid_success_rate', 'causal_fraction', 'fallback_signals', 'error')}}) + '\n')
        shown = {k: v for k, v in metrics.items() if k not in HIDDEN and isinstance(v, (int, float))}
        feedback = ('Metrics: ' + ', '.join(f'{k}={v:.4f}' for k, v in sorted(shown.items())) + '\n' + str(artifacts.get('feedback', metrics.get('error', '')))
                    + ('\n' + '; '.join(notes) if notes else ''))
        self.cache[source] = (float(metrics.get('valid_score') or 0.0), feedback)
        return self.cache[source]

    def get_feedback(self, query, response, reference=None, **kwargs):
        return self.evaluate(str(response))


@trace.model
class SignalProgram:
    def __init__(self, source: str):
        self.program = trace.node(source, trainable=True, name='program',
                                  description='Complete Python program for the task; it must keep run_signal_processing(...) and its output format.')

    def forward(self, _):
        return self.program


def mock_llm(messages, **kwargs):
    user = messages[-1]['content']
    name = re.search(r'<variable name="(\w+)"', user).group(1)
    source = W.INITIAL.read_text().replace('np.linspace(-2, 0, window_size)', f'np.linspace(-{random.choice([2.5, 3, 3.5])}, 0, window_size)')
    return f'<reasoning>mock</reasoning>\n<variable>\n<name>{name}</name>\n<value>\n{source}\n</value>\n</variable>'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=sorted(ARMS), required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--provider', default='novita')
    parser.add_argument('--out', required=True)
    parser.add_argument('--mock', action='store_true')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    calls = {'n': 0}
    client = mock_llm if args.mock else CUE.DeadlineClient('forward', args.provider, out / 'calls.jsonl')
    prompts = out / 'transcripts.jsonl'

    def llm(messages=None, **kwargs):
        calls['n'] += 1
        reply = client(messages=messages, max_tokens=32000)
        with prompts.open('a') as stream:
            stream.write(json.dumps({'call': calls['n'], 'user_chars': len(messages[-1]['content']), 'variation': 'VARIATION MODE' in messages[-1]['content'],
                                     'user': messages[-1]['content'][:3000], 'response_chars': len(reply or '')}) + '\n')
        return reply

    system_message = yaml.safe_load((W.SKY / 'config.yaml').read_text())['prompt']['system_message']
    agent = SignalProgram(W.INITIAL.read_text())
    optimizer = OptoPrimeV2(agent.parameters(), llm=DummyLLM(llm), max_tokens=32000, initial_var_char_limit=100000)
    optimizer.objective = f'{optimizer.objective}\n\nTask:\n{system_message}'
    guide = SignalGuide(out, calls)
    algo = VariationSearch(agent, optimizer)
    manifest = {'experiment': 'EXP28', 'arm': args.arm, 'seed': args.seed, 'steps': args.steps, 'model': MODEL, 'provider': args.provider, 'mock': args.mock,
                'trainer': 'VariationSearch', 'trainer_kwargs': ARMS[args.arm], 'hidden_metrics': list(HIDDEN), 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    started, status, error = time.time(), 'success', None
    try:
        algo.train(guide, {'inputs': [None], 'infos': [None]}, num_steps=args.steps, num_candidates=1, num_proposals=1, batch_size=1, num_threads=1,
                   test_frequency=None, verbose=False, variation_seed=args.seed, **ARMS[args.arm])
    except Exception as exc:  # noqa: BLE001  (keep partial logs; the summary records the failure)
        status, error = 'error', f'{type(exc).__name__}: {exc}'
    evals = [json.loads(line) for line in (out / 'evaluations.jsonl').read_text().splitlines()] if (out / 'evaluations.jsonl').exists() else []
    summary = {'status': status, 'error': error, 'arm': args.arm, 'seed': args.seed, 'wall_s': round(time.time() - started, 1), 'solution_calls': calls['n'],
               'best_valid': max((e['valid_score'] or 0.0 for e in evals), default=None), 'modes': [r['mode'] for r in algo.__dict__.get('variation_log', [])]}
    (out / 'variation_log.json').write_text(json.dumps(algo.__dict__.get('variation_log', []), indent=1) + '\n')
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'modes'}))


if __name__ == '__main__':
    main()
