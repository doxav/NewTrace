"""Live PRISM run of the recursive_opt ``coevolution`` engine through the control plane v2.

Usage (EXP22 venv, ~/code/Trace first on PYTHONPATH, OPENROUTER_API_KEY in the environment):
    python run_live_prism.py --proposer llm_rewrite|trace --out DIR [--horizon 100]
LLM: z-ai/glm-5.3-flash pinned to one OpenRouter provider (no fallback), reasoning low.
Evaluator: EXP22's process evaluator around the stock SkyDiscover PRISM evaluator.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP22 = HERE.parents[1] / 'EXP22'
sys.path.insert(0, str(EXP22))
sys.path.insert(0, os.environ.get('TRACE_ROOT', str(Path.home() / 'code' / 'Trace')))  # ~/code/Trace before the venv's frozen Trace
import openai  # noqa: E402
from skydiscover.optimize.config import load_config  # noqa: E402
from src.evaluation import SKY, TASKS, ProcessEvaluator, stock_evaluator  # noqa: E402  (EXP22)

from opto.features.recursive_opt import spec as S  # noqa: E402
from opto.features.recursive_opt.coevolution import control_plane as CP  # noqa: E402
from opto.features.recursive_opt.coevolution import evox_preset  # noqa: E402

MODEL = 'z-ai/glm-5.3-flash'


class RoleClient:
    """OpenRouter chat client pinned to one provider; bounded retry on transient errors; content-free call log."""

    def __init__(self, role: str, provider: str, log: Path) -> None:
        self.role, self.provider, self.log = role, provider, log
        self.client = openai.OpenAI(api_key=os.environ['OPENROUTER_API_KEY'], base_url='https://openrouter.ai/api/v1', timeout=600, max_retries=0)

    def __call__(self, messages=None, **kwargs):
        for attempt in range(5):
            started, record = time.time(), {'role': self.role, 'attempt': attempt}
            try:
                response = self.client.chat.completions.create(
                    model=MODEL, messages=messages, max_tokens=kwargs.get('max_tokens') or 32000, temperature=0.7,
                    extra_body={'provider': {'only': [self.provider], 'allow_fallbacks': False}, 'reasoning_effort': 'low', 'usage': {'include': True}})
                usage = response.usage
                record.update(ok=True, served_by=getattr(response, 'provider', None), finish=response.choices[0].finish_reason,
                              prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens, cost=getattr(usage, 'cost', None))
                return response.choices[0].message.content or ''
            except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as error:
                record.update(ok=False, error_type=type(error).__name__)
                if attempt == 4:
                    raise
                time.sleep(15 * 2 ** attempt)
            finally:
                record['latency_s'] = round(time.time() - started, 2)
                with self.log.open('a') as stream:
                    stream.write(json.dumps(record) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--proposer', choices=('llm_rewrite', 'trace'), required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--provider', default='novita')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    task_dir = SKY / TASKS['prism']
    evaluator = ProcessEvaluator.replace(stock_evaluator('prism'), out / 'evaluations')

    def evaluate(source: str):
        result = asyncio.run(evaluator.evaluate_program(source))
        return dict(result.metrics), dict(result.artifacts or {})

    CP.register()
    S.register_evaluator('exp23.prism.evaluator@1', CP.program_evaluator(evaluate))
    system_message = load_config(task_dir / 'config.yaml').context_builder.system_message
    config = {**evox_preset(horizon=args.horizon, summaries=True, generate_labels=True), 'proposer': args.proposer, 'evaluator_timeout_s': 360}
    raw = CP.coevolution_spec(level_id=f'prism_{args.proposer}', program=(task_dir / 'initial_program.py').read_text(), evaluator_ref='exp23.prism.evaluator@1',
                              system_message=system_message, engine_config=config, problem_description=system_message,
                              evaluator_context=(task_dir / 'evaluator' / 'evaluator.py').read_text(),
                              llm_profile={'provider': 'openrouter', 'model': MODEL, 'api_key_ref': 'env:OPENROUTER_API_KEY', 'max_tokens': 32000, 'temperature': 0.7},
                              offline=False)
    raw['runtime']['test_mode'] = True  # llm_factory is a behavioral resource: explicit non-portable mode, as EXP22 CP-A
    plan = S.compile_plan(raw)
    (out / 'raw_spec.json').write_text(json.dumps(raw, indent=1, default=str) + '\n')
    (out / 'plan.json').write_text(json.dumps(plan.explain(), indent=1, default=str) + '\n')
    events_path = out / 'events.jsonl'
    capture = {'coevolution_on_event': lambda event: events_path.open('a').write(json.dumps(event, default=str) + '\n')}
    started = time.time()
    (result,) = S.execute_plan(plan, {'llm_factory': lambda profile, role: RoleClient(role, args.provider, out / 'calls.jsonl'), 'capture': capture})
    report = dict(result.metadata.get('report') or {}) if result.metadata else {}
    summary = {'status': result.status, 'error': result.error, 'proposer': args.proposer, 'wall_s': round(time.time() - started, 1),
               'best_score': report.get('best_score'), 'solution_attempts': report.get('solution_attempts'), 'llm_calls': report.get('llm_calls'),
               'feedback_calls': report.get('feedback_calls'), 'meta_failures': report.get('meta_failures'), 'labels': list((report.get('labels') or {}).keys())}
    (out / 'report.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    (out / 'best_program.py').write_text(report.get('best_source') or '')
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    evaluator.close()
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
