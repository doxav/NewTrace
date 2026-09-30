"""EXP23 recursive_opt registrations and one raw spec per numbered alternative.

Runnable offline (simulator): A0, A1, A2, A7. A3 is the same code with a live
optimizer profile. A4-A6 need the live SkyDiscover kernel changes described in
README; their engines are registered as explicit stubs so specs compile and
plans can be inspected, while execution raises NotImplementedError.
"""

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any

from opto import trace
from opto.features.recursive_opt import spec as S
from opto.trainer.objectives import EvaluationResult

from src.meta import ONLINE_ARMS, run_online
from src.policies import REFERENCE, compile_policy, dump_knobs, stock_text
from src.search import run_fixed
from src.traced import Arena, SelectionPolicy, run_selection_window, summarize_decisions
from src.world import TASKS, WORLDS, Population, World

MODULE = 'exp23.module.selection_policy@1'
EVALUATOR = 'exp23.evaluator.episode_vs_stock@1'
DATASET = 'exp23.dataset.seeds@1'
ONLINE_ENGINE = 'exp23.engine.online_meta@1'
STUB_ENGINES = {
    'exp23.engine.live_coevolution_paired@1': 'A4: stock SkyDiscover kernel with the paired score and traced selection (see README A4).',
    'exp23.engine.live_amortized@1': 'A5: PrioritySearch over live 100-call runs, then frozen-policy holdout runs (see README A5).',
    'exp23.engine.live_task_level@1': 'A6: Trace PrioritySearch directly on the benchmark solution program (see README A6).',
}
SPLITS = {'train': 100_000, 'validation': 500_000, 'holdout': 900_000}  # disjoint seed ranges
RUN_OUTPUTS = Path(__file__).resolve().parents[1] / 'results' / 'control_plane'
LIVE_MODEL = 'z-ai/glm-5.3-flash'  # EXP22 frozen model for comparability; AGENTS.md Experiment-0 default is deepseek/deepseek-v4-flash-0731


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject unknown tasks, worlds, surfaces and undeclared fields."""
    allowed = {'task', 'world', 'surface', 'horizon', 'initial'}
    if set(config) - allowed or not {'task', 'world', 'surface', 'horizon'} <= set(config):
        raise ValueError(f'EXP23 module config requires {sorted(allowed - {"initial"})} (+ optional initial)')
    if config['task'] not in TASKS or config['world'] not in WORLDS or config['surface'] not in {'knobs', 'code'}:
        raise ValueError('unknown EXP23 task/world/surface')
    if not isinstance(config['horizon'], int) or not 4 <= config['horizon'] <= 5000:
        raise ValueError('horizon must be an integer in [4, 5000]')


class EpisodePolicy(SelectionPolicy):
    """SelectionPolicy that, given {'seed': s}, runs a whole simulated solution run inside the traced window."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config['surface'], config.get('initial') or stock_text(config['surface']))
        self.config = dict(config)

    def forward(self, example: Any):
        if isinstance(example, Arena):
            return super().forward(example)
        world = World(self.config['task'], self.config['world'])
        arena = Arena(world, Population.start(world.init), int(example['seed']), 0, self.config['horizon'], self.surface, 'solo', credit='new_best')
        return summarize_decisions(run_selection_window(self.selection_policy, arena), arena)


def build(level: Mapping[str, Any], resources: Mapping[str, Any]) -> EpisodePolicy:
    """Build the module from its validated config."""
    return EpisodePolicy(level['module']['config'])


def snapshot(module: EpisodePolicy) -> dict[str, str]:
    """Persist the exact policy text."""
    return {'selection_policy': str(module.selection_policy.data), 'surface': module.surface}


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Artifacts must be deployable policies."""
    if set(artifact) != {'selection_policy', 'surface'}:
        raise ValueError('EXP23 artifact requires selection_policy and surface')
    compile_policy(artifact['surface'], artifact['selection_policy'])


def restore(module: EpisodePolicy, artifact: Mapping[str, Any]) -> None:
    """Restore through the versioned artifact contract."""
    validate_artifact(artifact)
    module.selection_policy._data = artifact['selection_policy']


def seeds(split: str, config: Mapping[str, Any]) -> list[dict[str, int]]:
    """Disjoint seed ranges per split; config {'count': n}."""
    count = int(config.get('count', 0))
    return [{'seed': SPLITS[split] + i} for i in range(count)]


@lru_cache(maxsize=100_000)
def stock_final(task: str, world: str, seed: int, horizon: int) -> float:
    """Final best of the stock (uniform) policy on the same seed: the paired baseline."""
    return run_fixed(World(task, world), compile_policy('knobs', dump_knobs(REFERENCE['uniform'])), seed, horizon)[-1]


def evaluate_episode(output: Any, example: Any, context: Mapping[str, Any]) -> EvaluationResult:
    """Paired episode score: (policy final best - stock final best, same seed) / median |delta|."""
    data = output.data if hasattr(output, 'data') else output
    if int(example['seed']) != data['seed']:
        raise ValueError('evaluated output does not belong to this seed')
    world = World(data['task'], data['world'])
    base = stock_final(data['task'], data['world'], data['seed'], data['horizon'])
    score = (data['best_after'] - base) / world.scale
    rows = '\n'.join(f'  {k}: {v}' for k, v in data['by_tag_parent_rank_reuse'].items())
    feedback = (f"score={score:+.3f} = (final best {data['best_after']:.4f} - stock uniform policy {base:.4f} on the same seed) / median|delta| {world.scale:.4f}.\n"
                f"decisions={data['decisions']} policy_errors={data['policy_errors']}\noutcomes by parent rank and reuse:\n{rows}")
    return EvaluationResult(valid=True, status='ok', metrics={'score': score, 'final_best': data['best_after']}, feedback=feedback)


def execute_online(unit: Any, level: Any, resources: Mapping[str, Any]) -> S.RunResult:
    """Run one online meta arm on every holdout seed and report paired gains over fixed:uniform."""
    config, module_config = level.spec['engine']['config'], level.spec['module']['config']
    unknown = set(config) - {'arm', 'trigger', 'patience', 'window', 'credit', 'memory_size'}
    if unknown or config.get('arm') not in ONLINE_ARMS + tuple(f'fixed:{name}' for name in REFERENCE):
        raise ValueError(f'invalid online engine config {sorted(unknown)} / {config.get("arm")}')
    usage: dict[str, Any] = {}
    clients = S._resolve_role_clients(unit.spec, level.spec, resources, usage)
    world = World(module_config['task'], module_config['world'])
    examples = list(S.DatasetAccess(level.datasets).read('holdout', phase='final_evaluation'))
    results = []
    for example in examples:
        options = {k: config[k] for k in ('trigger', 'patience', 'window', 'credit', 'memory_size') if k in config}
        run = run_online(world, example['seed'], config['arm'], surface=module_config['surface'], horizon=module_config['horizon'], llm=clients.get('optimizer'), **options)
        base = stock_final(world.task, world.name, example['seed'], module_config['horizon'])
        results.append({**run, 'gain_vs_stock': (run['final_best'] - base) / world.scale})
    resources['_budget'].consume('candidates', len(examples) * module_config['horizon'])
    metrics = {'score': mean(r['gain_vs_stock'] for r in results), 'final_best': mean(r['final_best'] for r in results), 'p_beats_stock': mean(float(r['gain_vs_stock'] > 0) for r in results)}
    evaluation = EvaluationResult(valid=True, status='ok', metrics=metrics)
    return S.RunResult(unit_id=unit.unit_id, plan_fingerprint='', spec_fingerprint=unit.spec['fingerprint'], engine=ONLINE_ENGINE, module_ref=MODULE, status='success', valid=True, evaluation=evaluation, artifact={'selection_policy': results[-1].get('final_policy') or stock_text(module_config['surface']), 'surface': module_config['surface']}, lineage=(), usage=usage, budget=resources['_budget'].report(), metadata={'level_id': level.level_id, 'runs': [{k: v for k, v in r.items() if k != 'final_policy'} for r in results]})


def _stub(name: str):
    def execute(unit: Any, level: Any, resources: Mapping[str, Any]) -> S.RunResult:
        raise NotImplementedError(f'{name} is declared but not implemented: {STUB_ENGINES[name]}')
    return execute


_REGISTERED = False


def register() -> None:
    """Register every versioned extension before compilation (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    S.register_module(MODULE, S.ModuleRegistryEntry(build, snapshot, restore, validate_artifact, frozenset({'code', 'module'}), validate_config))
    S.register_evaluator(EVALUATOR, evaluate_episode)
    S.register_dataset(DATASET, seeds)
    S.register_engine(ONLINE_ENGINE, S.EngineRegistryEntry(execute_online, frozenset({'scalar'})))
    for name in STUB_ENGINES:
        S.register_engine(name, S.EngineRegistryEntry(_stub(name), frozenset({'scalar'})))
    _REGISTERED = True


# ---------------------------------------------------------------- spec builders

def _profile(live: bool) -> dict[str, Any]:
    if not live:
        return {'provider': 'openrouter', 'model': 'exp23/mock-random-local-proposer', 'max_tokens': 4000}
    return {'provider': 'openrouter', 'model': LIVE_MODEL, 'api_key_ref': 'env:OPENROUTER_API_KEY', 'temperature': 0.7, 'max_tokens': 8000,
            'request_timeout_s': 600, 'transport_max_attempts': 3, 'request_params': {'extra_body': {'reasoning_effort': 'low'}}}


def _level(level_id: str, task: str, world: str, surface: str, horizon: int, engine: dict, splits: dict[str, int], intent: str) -> dict[str, Any]:
    return {
        'id': level_id,
        'surface': {'kind': 'code' if surface == 'code' else 'module', 'targets': ['selection_policy']},
        'module': {'ref': MODULE, 'config': {'task': task, 'world': world, 'surface': surface, 'horizon': horizon}},
        'engine': engine,
        'objective': {'evaluator_ref': EVALUATOR, 'intent': intent, 'feedback_channels': ['natural_language'],
                      'metrics': {'score': {'direction': 'maximize', 'source': 'evaluation.metrics.score', 'aggregate_examples': 'mean'}},
                      'selection': {'mode': 'scalar', 'score_key': 'score'}},
        'llm_roles': {'optimizer': 'main'},
        'datasets': {split: {'ref': DATASET, 'config': {'count': count}} for split, count in splits.items()},
    }


def _spec(levels: list[dict], live: bool, outputs: Path, candidates: int | None, optimizer_calls: int | None, seed: int = 0) -> dict[str, Any]:
    return {'schema_version': S.SCHEMA_VERSION, 'kind': S.SPEC_KIND,
            'runtime': {'offline': not live, 'test_mode': not live, 'seed': seed},
            'llm_profiles': {'main': _profile(live)},
            'budget': {'candidates': candidates, 'optimizer_llm_calls': optimizer_calls, 'on_exceed': 'fail'},
            'outputs': {'directory': str(outputs)},
            'levels': levels}


def alternatives(directory: Path = RUN_OUTPUTS, task: str = 'prism', world: str = 'W1', seeds_holdout: int = 200) -> dict[str, dict[str, Any]]:
    """One raw spec per numbered alternative and arm (key 'A<k>/<arm>'); ``directory`` receives execution outputs."""
    specs: dict[str, dict[str, Any]] = {}
    online = lambda arm, **extra: {'name': ONLINE_ENGINE, 'config': {'arm': arm, **extra}}
    # A0 - replicate EXP22 conditions; A1 - fixed score within one run.
    for alt, arms in (('A0', ('fixed:uniform', 'fixed:greedy', 'fixed:top5_reuse', 'evox_log', 'trace_exp22')),
                      ('A1', ('evox_paired', 'trace_paired'))):
        for arm in arms:
            for trigger in (('stagnation',) if alt == 'A0' else ('stagnation', 'periodic')):
                name = f'{alt}/{arm.replace(":", "-")}' + ('' if alt == 'A0' else f'-{trigger}')
                extra = {} if alt == 'A0' else {'trigger': trigger, 'patience': 10, 'window': 10, 'credit': 'new_best', 'memory_size': 5}
                calls = seeds_holdout * 12 if arm.startswith('trace') else None
                specs[name] = _spec([_level('online_meta', task, world, 'knobs', 100, online(arm, **extra), {'holdout': seeds_holdout},
                                            'Maximize final best after 100 solution calls, paired against the stock policy on the same seeds.')],
                                    False, directory / name, seeds_holdout * 100, calls)
    # A2 - amortized meta-training with the real PrioritySearch trainer over whole episodes.
    for surface in ('knobs', 'code'):
        name = f'A2/priority_search-{surface}'
        engine = {'name': 'trace', 'config': {'optimizer': 'OptoPrimeV2', 'trainer': 'PrioritySearch', 'iterations': 24, 'num_candidates': 2, 'validation_gate': True,
                                              'optimizer_kwargs': {'memory_size': 5, 'max_tokens': 4000},
                                              'trainer_kwargs': {'batch_size': 4, 'num_batches': 1, 'num_proposals': 1, 'score_function': 'ucb', 'ucb_exploration_constant': 1.0,
                                                                 'validate_exploration_candidates': True, 'use_best_candidate_to_explore': True, 'test_frequency': None, 'log_frequency': 1, 'num_threads': 1}}}
        specs[name] = _spec([_level('amortized_policy', task, world, surface, 100, engine, {'train': 64, 'validation': 16, 'holdout': seeds_holdout},
                                    'Learn one parent-selection policy across runs; score = paired final-best gain over the stock policy.')],
                            False, directory / name, None, 200)
    # A3 - same as A1/A2 with a live optimizer LLM (only optimizer calls are paid; solutions stay simulated).
    for base in ('A1/trace_paired-stagnation', 'A2/priority_search-code'):
        live = json.loads(json.dumps(specs[base]))
        live['llm_profiles']['main'] = _profile(True)
        live['runtime'].update({'offline': False, 'test_mode': False})
        live['budget']['optimizer_llm_calls'] = 400 if base.startswith('A2') else 30 * 12
        if base.startswith('A1'):
            live['levels'][0]['datasets']['holdout']['config']['count'] = 30
            live['budget']['candidates'] = 30 * 100
        specs['A3/live_llm-' + base.split('/')[1]] = live
    # A4-A6 - live SkyDiscover experiments (stub engines until implemented).
    live_level = lambda engine, intent, surface='code': _level('live', task, world, surface, 100, {'name': engine, 'config': {}}, {'holdout': 8}, intent)
    specs['A4/live_paired_coevolution'] = _spec([live_level('exp23.engine.live_coevolution_paired@1', 'Stock EvoX kernel, LogWindowScorer replaced by the paired score; 8 seeds x {SD-FIXED, SD-EVOX, SD-EVOX-PAIRED, TRACE-PAIRED}.')], True, directory / 'A4', 8 * 4 * 100, 8 * 12)
    specs['A5/live_amortized'] = _spec([live_level('exp23.engine.live_amortized@1', 'PrioritySearch (8 iterations, 2 candidates, batch 4: <=96 live 100-call training runs), freeze the policy, then 8 paired holdout seeds vs stock.')], True, directory / 'A5', (96 + 8 * 2) * 100, 30)
    specs['A6/live_task_level'] = _spec([live_level('exp23.engine.live_task_level@1', 'Trace PrioritySearch on the solution program itself vs the EvoX solution loop at 100 calls, 8 seeds, no meta level.')], True, directory / 'A6', 8 * 2 * 100, 8 * 100)
    # A7 - horizon scaling of A0/A1/A2 in the simulator.
    for horizon in (300, 1000):
        for arm in ('evox_log', 'evox_paired', 'trace_paired'):
            name = f'A7/{arm}-h{horizon}'
            level = _level('online_meta', task, world, 'knobs', horizon, online(arm, trigger='stagnation', patience=max(10, horizon // 10), window=max(10, horizon // 10), credit='new_best', memory_size=5), {'holdout': 100},
                           f'Horizon {horizon}: does online meta-optimization pay off with more solution calls?')
            if arm == 'evox_log':
                level['engine']['config'] = {'arm': arm, 'patience': max(10, horizon // 10), 'window': max(10, horizon // 10)}
            specs[name] = _spec([level], False, directory / name, 100 * horizon, 100 * horizon // 5)
    return specs


def compile_all(directory: Path, outputs: Path = RUN_OUTPUTS, **kwargs: Any) -> dict[str, dict[str, Any]]:
    """Compile every alternative and persist raw, normalized and resolved plans under ``directory``."""
    register()
    explained = {}
    for name, raw in alternatives(outputs, **kwargs).items():
        target = directory / name
        target.mkdir(parents=True, exist_ok=True)
        plan = S.compile_plan(raw)
        for filename, value in (('raw_spec', raw), ('normalized_spec', plan.spec), ('resolved_execution_plan', plan.explain())):
            (target / f'{filename}.json').write_text(json.dumps(S._thaw(value), indent=2, sort_keys=True, default=str) + '\n')
        explained[name] = plan.explain()
    return explained


def mock_factory(profile: Mapping[str, Any], role: str):
    """llm_factory resource for offline specs: the feedback-blind mock proposer."""
    from opto.utils.llm import DummyLLM

    from src.mock_llm import MockOptimizerLLM
    return DummyLLM(MockOptimizerLLM(seed=0))


def run(raw: Mapping[str, Any], llm_factory: Any = None) -> tuple:
    """Compile and execute one spec (offline specs get the mock LLM factory unless one is given)."""
    register()
    plan = S.compile_plan(raw)
    factory = llm_factory or (mock_factory if raw['runtime']['offline'] else None)
    resources = {'llm_factory': factory} if factory else {}
    return S.execute_plan(plan, resources)


__all__ = ['alternatives', 'compile_all', 'register', 'run', 'trace']
