"""Control-plane v2 integration: the ``coevolution`` engine and a program-source module.

A level with ``engine.name == 'coevolution'`` runs one CoevolutionEngine. It declares two
internal levels over one shared state: O0 (the solution operator, LLM role ``forward``)
and O1 (the selection policy, LLM role ``optimizer``); ``feedback`` produces labels and
summaries. The level's module supplies the initial program; its registered evaluator
(mode ``output``) scores program sources. ``engine.config`` accepts every
``CoevolutionConfig`` field (see ``evox_preset``).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from opto.trace import node
from opto.trace.modules import Module
from opto.trainer.objectives import EvaluationResult, normalize_evaluation_result

from .. import spec as S
from .engine import CoevolutionConfig, CoevolutionEngine
from .projections import make_projection

ENGINE = 'coevolution'
MODULE = 'recursive_opt.module.program_source@1'
_FIELDS = {f.name for f in dataclasses.fields(CoevolutionConfig)}
_REGISTERED = False


class ProgramSource(Module):
    """One trainable program source (the O0 artifact)."""

    def __init__(self, source: str) -> None:
        super().__init__()
        self.program = node(source, trainable=True, name='program', description='Complete program source evaluated by the level evaluator.')

    def forward(self, _example: Any = None) -> Any:
        return self.program


def _validate_module_config(config: Mapping[str, Any]) -> None:
    if set(config) - {'program', 'language'} or not isinstance(config.get('program'), str) or not config['program'].strip():
        raise ValueError('program_source config requires a non-empty "program" string (optional "language")')


def _validate_artifact(artifact: Mapping[str, Any]) -> None:
    if set(artifact) != {'program'} or not isinstance(artifact['program'], str):
        raise ValueError('program_source artifact requires exactly {"program": str}')


def _restore(module: ProgramSource, artifact: Mapping[str, Any]) -> None:
    _validate_artifact(artifact)
    module.program._data = artifact['program']


def program_evaluator(evaluate: Callable[[str], Tuple[Dict[str, Any], Dict[str, Any]]]) -> Callable[..., EvaluationResult]:
    """Adapt ``evaluate(source) -> (metrics, artifacts)`` to a control-plane output evaluator."""
    def evaluator(output: Any, example: Any, context: Mapping[str, Any]) -> EvaluationResult:
        source = output.data if hasattr(output, 'data') else output
        metrics, artifacts = evaluate(str(source))
        valid = metrics.get('validity', 1) not in (0, -1)
        numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        return EvaluationResult(valid=valid, status='ok' if valid else 'invalid', metrics=numeric, feedback=str((artifacts or {}).get('feedback', '')),
                                artifacts={**dict(artifacts or {}), 'raw_metrics': dict(metrics)}, error=None if valid else str(metrics.get('error') or 'invalid candidate'))
    return evaluator


def _text(response: Any) -> str:
    if isinstance(response, str):
        return response
    choices = getattr(response, 'choices', None) or (response.get('choices') if isinstance(response, Mapping) else None)
    if choices:
        message = getattr(choices[0], 'message', None) or choices[0].get('message')
        content = getattr(message, 'content', None) if not isinstance(message, Mapping) else message.get('content')
        return content or ''
    return str(response or '')


def _role_llm(client: Any) -> Optional[Callable[[str, str], str]]:
    if client is None:
        return None
    return lambda system, user: _text(client(messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]))


def _capture_hook(capture: Any) -> Optional[Callable[[Dict[str, Any]], None]]:
    """Observational progress: append events to capture['coevolution_events'] and call capture['coevolution_on_event']."""
    if not isinstance(capture, dict):
        return None
    events = capture.setdefault('coevolution_events', [])
    callback = capture.get('coevolution_on_event')

    def hook(event: Dict[str, Any]) -> None:
        events.append(event)
        if callable(callback):
            callback(event)
    return hook


def _run_engine(unit: Any, level: Any, resources: Mapping[str, Any]) -> S.RunResult:
    spec = level.spec
    config = dict(spec['engine']['config'])
    unknown = set(config) - _FIELDS - {'system_message', 'problem_description', 'evaluator_context', 'projections'}
    if unknown:
        raise ValueError(f'unknown coevolution engine config keys: {sorted(unknown)}')
    system_message = config.pop('system_message', '')
    problem_description = config.pop('problem_description', None)
    evaluator_context = config.pop('evaluator_context', '')
    projections = [make_projection(item) for item in config.pop('projections', [])]
    engine_config = CoevolutionConfig(**config)
    module = S._build_level_module(spec, resources)
    evaluator = S._evaluator_entry(spec['objective']['evaluator_ref'])
    if evaluator.mode != 'output':
        raise ValueError('coevolution requires an output-mode evaluator over program sources')
    guard = resources['_budget']
    score_key = spec['objective']['selection'].get('score_key') or engine_config.score_key
    records = []

    def evaluate(source: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        guard.consume('evaluator_runs')
        result = normalize_evaluation_result(evaluator.evaluate(source, None, {'phase': 'fit', 'inputs': {}}))
        artifacts = dict(result.artifacts) if isinstance(result.artifacts, Mapping) else {}
        metrics = dict(artifacts.pop('raw_metrics', result.metrics))
        if not result.valid:
            metrics.setdefault('validity', 0)
            metrics.setdefault('error', result.error or 'invalid candidate')
        records.append({'valid': result.valid, score_key: metrics.get(score_key)})
        return metrics, artifacts

    usage: Dict[str, Any] = {}
    clients = S._resolve_role_clients(unit.spec, spec, resources, usage)
    solution_llm, meta_llm, feedback_llm = (_role_llm(clients.get(role)) for role in ('forward', 'optimizer', 'feedback'))
    if solution_llm is None or meta_llm is None:
        raise ValueError('coevolution needs llm_roles.forward (O0 operator) and llm_roles.optimizer (policy proposer)')

    def counted_solution(system: str, user: str) -> str:
        guard.consume('candidates')
        return solution_llm(system, user)
    engine = CoevolutionEngine(engine_config.__class__(**{**dataclasses.asdict(engine_config), 'score_key': score_key}), counted_solution, meta_llm, evaluate,
                               str(module.program.data), system_message, feedback_llm, problem_description, evaluator_context,
                               on_event=_capture_hook(resources.get('capture')), projections=projections)
    report = engine.run()
    module.program._data = report['best_source']
    metrics = {k: float(v) for k, v in report['best_metrics'].items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    evaluation = EvaluationResult(valid=True, status='ok', metrics=metrics)
    return S.RunResult(unit_id=f'{unit.unit_id}:{level.level_id}', plan_fingerprint='', spec_fingerprint=unit.spec['fingerprint'], engine=ENGINE,
                       module_ref=spec['module']['ref'], status='success', valid=True, evaluation=evaluation, artifact={'program': report['best_source']},
                       lineage=(), usage=usage, budget=guard.report(), metadata={'level_id': level.level_id, 'report': report})


def register() -> None:
    """Register the engine and the program-source module (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    S.register_module(MODULE, S.ModuleRegistryEntry(lambda level, resources: ProgramSource(level['module']['config']['program']),
                                                    lambda module: {'program': str(module.program.data)}, _restore, _validate_artifact,
                                                    frozenset({'code', 'module'}), _validate_module_config))
    S.register_engine(ENGINE, S.EngineRegistryEntry(_run_engine, frozenset({'scalar'})))
    _REGISTERED = True


def coevolution_spec(level_id: str, program: str, evaluator_ref: str, system_message: str, engine_config: Mapping[str, Any],
                     llm_profile: Optional[Mapping[str, Any]] = None, offline: bool = True, seed: int = 42, budget: Optional[Mapping[str, Any]] = None,
                     output_directory: Optional[str] = None, problem_description: Optional[str] = None, evaluator_context: str = '',
                     score_key: str = 'combined_score') -> Dict[str, Any]:
    """Raw v2 spec for one co-evolution level (one profile shared by the three roles unless overridden)."""
    profile = dict(llm_profile or {'provider': 'openrouter', 'model': 'offline/mock', 'max_tokens': 32000})
    config = {**dict(engine_config), 'system_message': system_message, 'evaluator_context': evaluator_context}
    if problem_description is not None:
        config['problem_description'] = problem_description
    return {
        'schema_version': S.SCHEMA_VERSION, 'kind': S.SPEC_KIND,
        'runtime': {'offline': offline, 'test_mode': offline, 'seed': seed},
        'llm_profiles': {'main': profile},
        'budget': dict(budget or {'on_exceed': 'fail'}),
        'outputs': {'directory': output_directory},
        'levels': [{
            'id': level_id,
            'surface': {'kind': 'code', 'targets': ['program']},
            'module': {'ref': MODULE, 'config': {'program': program}},
            'engine': {'name': ENGINE, 'config': config},
            'objective': {'evaluator_ref': evaluator_ref, 'intent': 'Maximize the evaluator combined_score through online co-evolution.',
                          'metrics': {score_key: {'direction': 'maximize', 'source': f'evaluation.metrics.{score_key}', 'aggregate_examples': 'mean'}},
                          'selection': {'mode': 'scalar', 'score_key': score_key}},
            'llm_roles': {'forward': 'main', 'optimizer': 'main', 'feedback': 'main'},
        }],
    }
