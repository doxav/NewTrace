"""Versioned recursive-opt registrations for the stock co-evolution kernel."""

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.evaluation import TASKS
from src.kernel import POLICY, PolicyModule, run_kernel
from src.transport import MODEL, SESSION

from opto.features.recursive_opt import spec as S
from opto.trainer.objectives import EvaluationResult

MODULE = 'exp22.module.discovery_policy@1'
ENGINE = 'exp22.engine.coevolution@1'
EVALUATOR = 'exp22.evaluator.coevolution@1'


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject unknown tasks, arms, budgets and undeclared configuration fields."""
    if set(config) != {'task', 'arm', 'horizon', 'directory'}:
        raise ValueError('EXP22 config requires exactly task, arm, horizon, directory')
    if config['task'] not in TASKS or config['arm'] not in {'TRACE-FIXED', 'TRACE-RECURSIVE'}:
        raise ValueError('Invalid EXP22 task or Trace arm')
    if not isinstance(config['horizon'], int) or isinstance(config['horizon'], bool) or not 1 <= config['horizon'] <= 100:
        raise ValueError('EXP22 horizon must be an integer in [1, 100]')
    if not isinstance(config['directory'], str) or not config['directory']:
        raise ValueError('EXP22 output directory is required')


def build(level: Mapping[str, Any], resources: Mapping[str, Any]) -> PolicyModule:
    """Initialize the same complete stock policy in either control-plane variant."""
    return PolicyModule(POLICY.read_text())


def snapshot(module: PolicyModule) -> dict[str, str]:
    """Persist the exact trainable policy source."""
    return {'policy_source': str(module.policy_source.data)}


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate serialized source shape; activation uses stock policy validation."""
    if set(artifact) != {'policy_source'} or not isinstance(artifact['policy_source'], str) or not artifact['policy_source'].strip():
        raise ValueError('EXP22 artifact requires nonempty policy_source')


def restore(module: PolicyModule, artifact: Mapping[str, Any]) -> None:
    """Restore source only through the versioned artifact contract."""
    validate_artifact(artifact)
    module.policy_source._data = artifact['policy_source']


def dataset(split: str, config: Mapping[str, Any]) -> list[dict[str, str]]:
    """Name the stock benchmark without introducing a hidden holdout panel."""
    if config.get('task') not in TASKS:
        raise ValueError('Unknown EXP22 dataset')
    return [{'task': config['task']}] if split == 'train' else []


def evaluate(output: Mapping[str, Any], target: Any, context: Mapping[str, Any]) -> EvaluationResult:
    """Project the actual stock best solution metrics into the declared objective."""
    if output.get('gate_failure'):
        raise ValueError(output['gate_failure'])
    if output['iterations_observed'] != output['horizon']:
        raise ValueError('Solution-generation accounting does not match the requested horizon')
    return EvaluationResult(valid=True, status='ok', metrics=output['final_metrics'])


def execute(unit: Any, level: Any, resources: Mapping[str, Any]) -> S.RunResult:
    """Execute the persistent hybrid kernel through the canonical plan runner."""
    config = level.spec['module']['config']
    module = S._build_level_module(level.spec, resources)
    usage: dict[str, Any] = {}
    clients = S._resolve_role_clients(unit.spec, level.spec, resources, usage)
    result = asyncio.run(run_kernel(config['task'], config['arm'], config['horizon'], Path(config['directory']), module, clients['optimizer']))
    evaluation = evaluate(result, level.datasets, {})
    guard = resources['_budget']
    guard.consume('candidates', result['iterations_observed'])
    return S.RunResult(unit_id=unit.unit_id, plan_fingerprint='', spec_fingerprint=unit.spec['fingerprint'], engine=ENGINE, module_ref=MODULE, status='success', valid=True, evaluation=evaluation, artifact=snapshot(module), lineage=(), usage=usage, budget=guard.report(), metadata={'level_id': level.level_id, 'kernel_result': result})


def register() -> None:
    """Register every explicit versioned extension before compilation."""
    S.register_module(MODULE, S.ModuleRegistryEntry(build, snapshot, restore, validate_artifact, frozenset({'code'}), validate_config))
    S.register_engine(ENGINE, S.EngineRegistryEntry(execute, frozenset({'scalar'})))
    S.register_evaluator(EVALUATOR, evaluate)
    for task in TASKS:
        S.register_dataset(f'exp22.dataset.{task}@1', dataset)


def specification(task: str, arm: str, horizon: int, directory: Path, variant: str = 'CP-B') -> dict[str, Any]:
    """Declare one complete Trace run without embedding credentials."""
    if variant not in {'CP-A', 'CP-B'}:
        raise ValueError('Unknown EXP22 control-plane variant')
    profile: dict[str, Any] = {'provider': 'openrouter', 'model': MODEL, 'api_key_ref': 'env:OPENROUTER_API_KEY', 'temperature': 0.7, 'max_tokens': 32000, 'request_timeout_s': 600, 'transport_max_attempts': 1, 'request_params': {'extra_body': {'session_id': SESSION}}}
    if variant == 'CP-B':
        profile['openrouter_routing'] = {'only': ['DeepInfra']}
    return {
        'schema_version': S.SCHEMA_VERSION, 'kind': S.SPEC_KIND,
        'runtime': {'offline': False, 'test_mode': variant == 'CP-A', 'seed': 42},
        'llm_profiles': {'main': profile}, 'budget': {'candidates': horizon, 'on_exceed': 'fail'},
        'outputs': {'directory': str(directory / 'control_plane')},
        'levels': [{
            'id': 'coevolution', 'surface': {'kind': 'code', 'targets': ['policy_source']},
            'module': {'ref': MODULE, 'config': {'task': task, 'arm': arm, 'horizon': horizon, 'directory': str(directory)}},
            'engine': {'name': ENGINE},
            'objective': {'evaluator_ref': EVALUATOR, 'intent': 'Maximize stock benchmark combined_score at equal solution generation budget.', 'metrics': {'combined_score': {'direction': 'maximize', 'source': 'evaluation.metrics.combined_score', 'aggregate_examples': 'mean'}}, 'selection': {'mode': 'scalar', 'score_key': 'combined_score'}},
            'llm_roles': {'optimizer': 'main'},
            'datasets': {'train': {'ref': f'exp22.dataset.{task}@1', 'config': {'task': task}}},
        }],
    }


def compile_all(directory: Path) -> list[dict[str, Any]]:
    """Persist raw, normalized and resolved plans for every intended Trace arm."""
    register()
    plans = []
    for task in TASKS:
        for arm in ('TRACE-FIXED', 'TRACE-RECURSIVE'):
            for variant in ('CP-A', 'CP-B'):
                target = directory / f'{task}_{arm}_{variant}'
                target.mkdir(parents=True, exist_ok=True)
                raw = specification(task, arm, 100, target, variant)
                plan = S.compile_plan(raw)
                for name, value in (('raw_spec', raw), ('normalized_spec', plan.spec), ('resolved_execution_plan', {**plan.explain(), 'units': [{'datasets': S._thaw(level.datasets), 'roles': S._thaw(level.spec['llm_roles']), 'budget': S._thaw(unit.spec['budget'])} for unit in plan.units for level in unit.levels]})):
                    (target / f'{name}.json').write_text(json.dumps(value, indent=2) + '\n')
                plans.append(plan.explain())
    return plans
