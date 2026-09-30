"""Run stock SkyDiscover EvoX on the toy task with scripted LLMs; write a normalized event trace.

Requires the EXP22 venv (SkyDiscover installed). No network: every LLM call is scripted.
"""

import asyncio
import json
import sys
import uuid
from collections import Counter
from importlib import import_module
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import script  # noqa: E402
from skydiscover.optimize.config import EvoxDatabaseConfig, LLMModelConfig, load_config  # noqa: E402
from skydiscover.optimize.llm.base import LLMInterface, LLMResponse  # noqa: E402
from skydiscover.optimize.search.default_discovery_controller import DiscoveryControllerInput  # noqa: E402
from skydiscover.optimize.search.evox.controller import CoEvolutionController  # noqa: E402
from skydiscover.optimize.search.registry import create_database, get_program  # noqa: E402
from skydiscover.optimize.utils.metrics import get_score  # noqa: E402

CALLS: Counter = Counter()
EVENTS: list = []
CURVE: list = []


def classify(system: str, messages: list) -> str:
    content = str(messages[-1].get('content', '')) if messages else ''
    if not system and content == 'ping':
        return 'ping'
    if system.startswith('Summarize the population state'):
        return 'stats_insight'
    if system.startswith('Summarize the downstream problem'):
        return 'problem_context'
    if system.startswith('You are summarizing previous search algorithm attempts'):
        return 'batch_summary'
    if 'evolving a search algorithm' in system:
        return 'meta'
    return 'solution'


class MockLLM(LLMInterface):
    def __init__(self, model_cfg=None):
        self.model_cfg = model_cfg

    async def generate(self, system_message, messages, **kwargs):
        kind = classify(system_message or '', messages)
        CALLS[kind] += 1
        n = CALLS[kind]
        if kind == 'solution':
            return LLMResponse(text=script.solution_reply(n))
        if kind == 'meta':
            return LLMResponse(text='```python\n' + script.stock_policy_source(script.meta_policy(n)) + '\n```')
        return LLMResponse(text='' if kind == 'ping' else 'summary')


class Recording(CoEvolutionController):
    def __init__(self, controller_input):
        super().__init__(controller_input)
        original_run, original_post = self.search_controller.run_discovery, self.search_controller.postprocess_result

        async def run_discovery(*args, **kwargs):
            before = CALLS['meta']
            result = await original_run(*args, **kwargs)
            EVENTS.append({'type': 'proposal', 'attempts': CALLS['meta'] - before, 'ok': bool(result and not result.error)})
            return result

        async def postprocess_result(result, *args, **kwargs):
            EVENTS.append({'type': 'scored', 'score': round(float((result.child_program_dict or {}).get('metrics', {}).get('combined_score', 0.0)), 9)})
            return await original_post(result, *args, **kwargs)
        self.search_controller.run_discovery = run_discovery
        self.search_controller.postprocess_result = postprocess_result

    def _label(self, text):
        return 'refine' if text == self.database.REFINE_LABEL else 'diverge' if text == self.database.DIVERGE_LABEL else ''

    async def _run_iteration(self, iteration, retry_times=3):
        result = await super()._run_iteration(iteration, retry_times=retry_times)
        if result.error and result.prompt is None and self._fallback_database is not None:
            return result  # policy failure that SkyDiscover answers with a rollback + retry (recorded there)
        event = {'type': 'iteration', 'iteration': iteration, 'attempts': result.attempts_used, 'ok': not result.error}
        if not result.error:
            child = result.child_program_dict
            parent = self.database.get(result.parent_id)
            event.update(parent_iteration=parent.iteration_found, label=self._label((child.get('parent_info') or ('', ''))[0]),
                         context_iterations=[self.database.get(i).iteration_found for i in (result.other_context_ids or [])],
                         child_score=round(get_score(child['metrics']), 9))
        EVENTS.append(event)
        return result

    async def _evolve_search(self, solution_iter):
        EVENTS.append({'type': 'trigger', 'iteration': solution_iter, 'best': round(self._get_best_score(), 9)})
        await super()._evolve_search(solution_iter)

    def _switch_to_new_search_algorithm(self, result):
        ok = super()._switch_to_new_search_algorithm(result)
        EVENTS.append({'type': 'deploy', 'ok': ok})
        return ok

    def _restore_fallback_database(self):
        EVENTS.append({'type': 'rollback'})
        super()._restore_fallback_database()

    def _record_search_window_step(self):
        super()._record_search_window_step()
        CURVE.append(round(self._get_best_score(), 9))


async def main(horizon: int, out: Path) -> None:
    import_module('skydiscover.optimize.search.route')
    config = load_config(HERE / 'toy' / 'config.yaml')
    config.search.type = 'evox'
    config.search.database = EvoxDatabaseConfig(random_seed=42, auto_generate_variation_operators=False)
    config.search.share_llm = True
    config.search.output_dir = str(out / 'search')
    config.search.switch_interval = None
    config.max_parallel_iterations = 1
    model = LLMModelConfig(name='mock', api_base='http://mock.invalid', temperature=0.7, max_tokens=32000, timeout=60, retries=0, retry_delay=1, init_client=MockLLM)
    config.llm.models, config.llm.evaluator_models, config.llm.guide_models = [model], [model], [model]
    config.llm.api_key, config.llm.api_base = None, model.api_base
    database = create_database('evox', config.search.database)
    controller = Recording(DiscoveryControllerInput(config, str(HERE / 'toy' / 'evaluator.py'), database, output_dir=str(out)))
    source = (HERE / 'toy' / 'initial_program.py').read_text()
    metrics = (await controller.evaluator.evaluate_program(source)).metrics
    initial = get_program(config, source, str(uuid.uuid4()), metrics, 0)
    database.add(initial, iteration=0)
    database.initial_program_id, database.initial_program_score = initial.id, get_score(metrics)
    best = await controller.run_discovery(1, horizon)
    report = {'framework': 'skydiscover-evox', 'events': EVENTS, 'curve': CURVE, 'best_score': round(get_score(best.metrics), 9),
              'calls': dict(CALLS), 'switch_interval': controller._switch_interval, 'meta_failures': controller._meta_evolution_failures}
    (out / 'stock_trace.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('best_score', 'calls', 'switch_interval', 'meta_failures')}))


if __name__ == '__main__':
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'out'
    out.mkdir(parents=True, exist_ok=True)
    asyncio.run(main(horizon, out))
