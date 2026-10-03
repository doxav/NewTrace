"""EXP25 Signal Processing run: native coevolution (control plane v2) with the Trace proposer.

Arms (100 solution calls each, strict budget, seeds, transport retries up to one hour):
  trace_exp24  EXP24 treatment: valid-score guide, per-signal white-box feedback, fallback projection
  trace_exp23  EXP23 configuration: stock metric as guide, no feedback artifacts, no projection
Every evaluated candidate is logged (source SHA-256, stock and valid scores) in evaluations.jsonl for re-scoring.

Usage (EXP22 venv; OPENROUTER_API_KEY unless --mock):
  python run_signal.py --arm trace_exp24 --seed 42 --out DIR [--horizon 100] [--mock]
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP25 = HERE.parent
TRACE_ROOT = Path(os.environ.get('TRACE_ROOT', str(Path.home() / 'code' / 'Trace')))
sys.path.insert(0, str(TRACE_ROOT))
sys.path.insert(0, str(EXP25 / 'signal'))
import whitebox as W  # noqa: E402
import yaml  # noqa: E402

from opto.features.recursive_opt import spec as S  # noqa: E402
from opto.features.recursive_opt.coevolution import GuidedEvaluator, evox_preset  # noqa: E402
from opto.features.recursive_opt.coevolution import control_plane as CP  # noqa: E402

MODEL = 'z-ai/glm-5.3-flash'
ARMS = ('trace_exp24', 'trace_exp23')


class RoleClient:
    """OpenRouter client pinned to one provider. Transient failures (rate limit, timeout, connection,
    5xx) are retried with capped backoff for up to ``max_outage_s``, so an outage delays the run
    instead of consuming solution budget. Every attempt is logged without content."""

    def __init__(self, role: str, provider: str, log: Path, max_outage_s: float = 3600.0) -> None:
        import openai
        self.openai, self.role, self.provider, self.log, self.max_outage_s = openai, role, provider, log, max_outage_s
        self.client = openai.OpenAI(api_key=os.environ['OPENROUTER_API_KEY'], base_url='https://openrouter.ai/api/v1', timeout=600, max_retries=0)

    def __call__(self, messages=None, **kwargs):
        transient = (self.openai.RateLimitError, self.openai.APITimeoutError, self.openai.APIConnectionError, self.openai.InternalServerError)
        first, attempt = time.time(), 0
        while True:
            started, record = time.time(), {'role': self.role, 'attempt': attempt}
            try:
                response = self.client.chat.completions.create(
                    model=MODEL, messages=messages, max_tokens=kwargs.get('max_tokens') or 32000, temperature=0.7,
                    extra_body={'provider': {'only': [self.provider], 'allow_fallbacks': False}, 'reasoning_effort': 'low', 'usage': {'include': True}})
                usage = response.usage
                record.update(ok=True, served_by=getattr(response, 'provider', None), finish=response.choices[0].finish_reason,
                              prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens, cost=getattr(usage, 'cost', None))
                return response.choices[0].message.content or ''
            except transient as error:
                record.update(ok=False, error_type=type(error).__name__)
                if time.time() - first > self.max_outage_s:
                    raise
                time.sleep(min(120, 10 * 2 ** min(attempt, 4)))
                attempt += 1
            finally:
                record['latency_s'] = round(time.time() - started, 2)
                with self.log.open('a') as stream:
                    stream.write(json.dumps(record) + '\n')


def mock_factory(profile, role):
    """Offline scripted client (tests)."""
    from opto.features.recursive_opt.coevolution import UNIFORM_POLICY_SOURCE

    def client(messages=None, **kwargs):
        if role == 'forward':
            return ('<<<<<<< SEARCH\n    weights = np.exp(np.linspace(-2, 0, window_size))\n=======\n'
                    '    weights = np.exp(np.linspace(-3, 0, window_size))\n>>>>>>> REPLACE')
        if role == 'optimizer':
            user = messages[-1]['content']
            if '<variable name="' in user:
                import re
                name = re.search(r'<variable name="(\w+)"', user).group(1)
                return f'<variable>\n<name>{name}</name>\n<value>\n{UNIFORM_POLICY_SOURCE}# revised\n</value>\n</variable>'
            return '```python\n' + UNIFORM_POLICY_SOURCE + '# revised\n```'
        return '=== DIVERGE ===\nTry a different filter family.\n=== REFINE ===\nTune the current filter.'
    return client


def sha256_tree(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*.py'))}


def audited(evaluate, out: Path, keep_artifacts: bool):
    """Log every evaluated source and both scores; the stock arm sees no feedback artifacts (as in EXP23)."""
    sources = out / 'sources'
    sources.mkdir(exist_ok=True)

    def wrapped(source: str):
        metrics, artifacts = evaluate(source)
        digest = hashlib.sha256(source.encode()).hexdigest()
        (sources / f'{digest}.py').write_text(source)
        with (out / 'evaluations.jsonl').open('a') as stream:
            stream.write(json.dumps({'sha256': digest, **{k: metrics.get(k) for k in ('combined_score', 'valid_score', 'success_rate', 'valid_success_rate', 'causal_fraction', 'fallback_signals', 'error')}}) + '\n')
        return metrics, (artifacts if keep_artifacts else {})
    return wrapped


def build_spec(arm: str, seed: int, horizon: int, offline: bool) -> dict:
    config = yaml.safe_load((W.SKY / 'config.yaml').read_text())
    system_message = config['prompt']['system_message']
    engine = {**evox_preset(horizon=horizon, summaries=True, generate_labels=True), 'proposer': 'trace', 'seed': seed, 'archive_seed': seed,
              'evaluator_timeout_s': 360, 'strict_budget': True}
    score_key = 'combined_score'
    if arm == 'trace_exp24':
        engine['projections'] = W.PROJECTIONS
        score_key = 'guided_score'
    raw = CP.coevolution_spec(level_id=f'exp25_signal_{arm}', program=W.INITIAL.read_text(), evaluator_ref=f'exp25.signal.{arm}@1',
                              system_message=system_message, engine_config=engine, problem_description=system_message,
                              evaluator_context=(W.SKY / 'evaluator' / 'evaluator.py').read_text(),
                              llm_profile={'provider': 'openrouter', 'model': MODEL, 'api_key_ref': 'env:OPENROUTER_API_KEY', 'max_tokens': 32000, 'temperature': 0.7},
                              offline=offline, seed=seed, score_key=score_key)
    raw['runtime']['test_mode'] = True
    return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--provider', default='novita')
    parser.add_argument('--out', required=True)
    parser.add_argument('--mock', action='store_true')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    CP.register()
    if args.arm == 'trace_exp24':
        evaluator = GuidedEvaluator(audited(W.evaluate, out, keep_artifacts=True), score_key='guided_score', source_key='valid_score')
    else:
        evaluator = audited(W.evaluate, out, keep_artifacts=False)
    S.register_evaluator(f'exp25.signal.{args.arm}@1', CP.program_evaluator(evaluator))
    raw = build_spec(args.arm, args.seed, args.horizon, offline=args.mock)
    plan = S.compile_plan(raw)
    manifest = {'experiment': 'EXP25', 'arm': args.arm, 'seed': args.seed, 'horizon': args.horizon, 'model': MODEL, 'provider': args.provider, 'mock': args.mock,
                'plan_fingerprint': plan.explain()['fingerprint'], 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'coevolution_sha256': sha256_tree(TRACE_ROOT / 'opto' / 'features' / 'recursive_opt' / 'coevolution'), 'signal_adapter_sha256': sha256_tree(EXP25 / 'signal')}
    (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    (out / 'raw_spec.json').write_text(json.dumps(raw, indent=1, default=str) + '\n')
    events_path = out / 'events.jsonl'

    def on_event(event):
        with events_path.open('a') as stream:
            stream.write(json.dumps(event, default=str) + '\n')
    factory = mock_factory if args.mock else (lambda profile, role: RoleClient(role, args.provider, out / 'calls.jsonl'))
    started = time.time()
    (result,) = S.execute_plan(plan, {'llm_factory': factory, 'capture': {'coevolution_on_event': on_event}})
    report = dict(result.metadata.get('report') or {}) if result.metadata else {}
    summary = {'status': result.status, 'error': result.error, 'arm': args.arm, 'seed': args.seed, 'wall_s': round(time.time() - started, 1),
               'best_metrics': report.get('best_metrics'), 'solution_attempts': report.get('solution_attempts'), 'llm_calls': report.get('llm_calls'),
               'feedback_calls': report.get('feedback_calls'), 'meta_failures': report.get('meta_failures')}
    (out / 'report.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    (out / 'best_program.py').write_text(report.get('best_source') or '')
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
