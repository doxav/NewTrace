"""EXP24 PRISM run: native coevolution (control plane v2) with the valid-score guide, per-case
white-box feedback and the fallback projection.

Arms (all share O0: diff operator, labels, guide, feedback, projection):
  fixed        stock uniform policy, no policy change (trigger 'never') - control for the meta level
  llm_rewrite  EvoX-equivalent meta level (evox_preset)
  trace        same, with the persistent OptoPrimeV2 proposer

Usage (EXP22 venv; OPENROUTER_API_KEY in the environment unless --mock):
  python run_prism.py --arm trace --seed 42 --out DIR [--horizon 100] [--mock]
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP24 = HERE.parent
TRACE_ROOT = Path(os.environ.get('TRACE_ROOT', str(Path.home() / 'code' / 'Trace')))
sys.path.insert(0, str(TRACE_ROOT))
sys.path.insert(0, str(EXP24 / 'prism'))
import whitebox as W  # noqa: E402
import yaml  # noqa: E402

from opto.features.recursive_opt import spec as S  # noqa: E402
from opto.features.recursive_opt.coevolution import GuidedEvaluator, evox_preset  # noqa: E402
from opto.features.recursive_opt.coevolution import control_plane as CP  # noqa: E402

MODEL = 'z-ai/glm-5.3-flash'
ARMS = ('fixed', 'llm_rewrite', 'trace')


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
    """Offline scripted client (tests): valid diff for O0, stock-like policy for O1, text for feedback."""
    from opto.features.recursive_opt.coevolution import UNIFORM_POLICY_SOURCE

    def client(messages=None, **kwargs):
        if role == 'forward':
            return ('<<<<<<< SEARCH\n    sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True)\n=======\n'
                    '    sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo, m.model_size), reverse=True)\n>>>>>>> REPLACE')
        if role == 'optimizer':
            user = messages[-1]['content']
            if '<variable name="' in user:  # OptoPrimeV2 format
                import re
                name = re.search(r'<variable name="(\w+)"', user).group(1)
                return f'<variable>\n<name>{name}</name>\n<value>\n{UNIFORM_POLICY_SOURCE}# revised\n</value>\n</variable>'
            return '```python\n' + UNIFORM_POLICY_SOURCE + '# revised\n```'
        return '=== DIVERGE ===\nTry a different algorithm.\n=== REFINE ===\nPolish the current one.'
    return client


def sha256_tree(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*.py'))}


def build_spec(arm: str, seed: int, horizon: int, offline: bool) -> dict:
    task = W.SKY_PRISM
    config = yaml.safe_load((task / 'config.yaml').read_text())
    system_message = config['prompt']['system_message']
    engine = {**evox_preset(horizon=horizon, summaries=True, generate_labels=True), 'seed': seed, 'archive_seed': seed,
              'evaluator_timeout_s': 360, 'projections': W.projections_config(),
              'strict_budget': True}  # cap last-iteration retries: exactly `horizon` solution calls per arm (EvoX can overshoot)
    if arm == 'fixed':
        engine['trigger'] = 'never'
    elif arm == 'trace':
        engine['proposer'] = 'trace'
    raw = CP.coevolution_spec(level_id=f'exp24_prism_{arm}', program=(task / 'initial_program.py').read_text(), evaluator_ref='exp24.prism.whitebox@1',
                              system_message=system_message, engine_config=engine, problem_description=system_message,
                              evaluator_context=(task / 'evaluator' / 'evaluator.py').read_text(),
                              llm_profile={'provider': 'openrouter', 'model': MODEL, 'api_key_ref': 'env:OPENROUTER_API_KEY', 'max_tokens': 32000, 'temperature': 0.7},
                              offline=offline, seed=seed, score_key='guided_score')
    raw['runtime']['test_mode'] = True  # llm_factory is a behavioral resource: explicit non-portable mode
    return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--provider', default='novita')
    parser.add_argument('--out', required=True)
    parser.add_argument('--mock', action='store_true', help='offline scripted LLM (tests)')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    CP.register()
    S.register_evaluator('exp24.prism.whitebox@1', CP.program_evaluator(GuidedEvaluator(W.evaluate, score_key='guided_score', source_key='valid_score')))
    raw = build_spec(args.arm, args.seed, args.horizon, offline=args.mock)
    plan = S.compile_plan(raw)
    manifest = {'experiment': 'EXP24', 'arm': args.arm, 'seed': args.seed, 'horizon': args.horizon, 'model': MODEL, 'provider': args.provider, 'mock': args.mock,
                'plan_fingerprint': plan.explain()['fingerprint'], 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'coevolution_sha256': sha256_tree(TRACE_ROOT / 'opto' / 'features' / 'recursive_opt' / 'coevolution'), 'prism_adapter_sha256': sha256_tree(EXP24 / 'prism')}
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
