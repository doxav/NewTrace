"""Declarative canonical and legacy-compatible recursive optimization control plane."""
from __future__ import annotations
import json
import hashlib
import math
import os
import random
import re
import time
import copy
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple
import numpy as np
from opto.trace import bundle, node
from opto.trace.containers import Map
from opto.trace.modules import Module
from opto.trainer.objectives import EvaluationResult, apply_minimize, normalize_evaluation_result, select_evaluation_result, satisfies_hard_constraints, to_scalar_score, weighted_scalarize
from .levels import CapabilityArtifact, TimedGuide, RecursiveGuide, LevelConfig, MetaLevel, FamilyPolicyLevel, PriorInductionLevel, ComponentSpec, CodeArtifactLevel, DEFAULT_INVALID_FLOOR, best_config_from, register_config_values, validate_level_config
from .memory import ArtifactRecord, MemoryLite
from .budget import BudgetExceeded, RecursiveOptBudget, reset_budget, make_budget
from .optimize import optimize
from .progress import RecursiveOptProgressLogger
from . import tracebench as TB
SURFACES = ('config', 'code', 'family_policy', 'prior', 'capability', 'custom')
SCHEMA_VERSION = 'recursive-opt/v2alpha'
SPEC_KIND = 'recursive_optimization'
GEPA_VERSION = '0.1.4'
CANONICAL_SPEC_BLOCKS = ('runtime', 'llm_profiles', 'knowledge', 'outputs', 'budget', 'experiment', 'levels')
_LEVEL_BLOCKS = ('surface', 'module', 'engine', 'objective', 'llm_roles', 'datasets', 'bindings', 'outputs')
_FLAT_LEVEL_BLOCKS = tuple((block for block in _LEVEL_BLOCKS if block != 'outputs'))
_TOP_LEVEL_KEYS = {'schema_version', 'kind', 'fingerprint', 'extensions', *CANONICAL_SPEC_BLOCKS, *_LEVEL_BLOCKS}
_LEGACY_TOP_LEVEL_KEYS = {'families', 'budget', 'tracebench', 'scoring', 'prior_promotion', 'memory_root', 'reuse_priors', 'levels', 'trainer_kwargs', 'run_id', 'extensions'}
_BLOCK_KEYS = {'surface': {'kind', 'targets'}, 'module': {'ref', 'config', 'artifact', 'inputs'}, 'engine': {'name', 'config'}, 'runtime': {'strict_refs', 'reproducible', 'offline', 'resume', 'memory_root', 'reuse_priors', 'tracebench', 'scoring', 'prior_promotion', 'trainer_kwargs', 'run_id', 'seed', 'test_mode'}, 'objective': {'evaluator_ref', 'intent', 'metrics', 'directions', 'selection', 'hard_constraints', 'aggregation', 'feedback_channels'}, 'datasets': {'train', 'validation', 'holdout'}, 'llm_roles': {'forward', 'optimizer', 'feedback', 'judge'}, 'knowledge': {'store', 'retrieval', 'statuses', 'scope_fields', 'top_k', 'injection_codec', 'promotion_rule', 'rollback_rule'}, 'outputs': {'directory', 'format', 'save_artifacts'}, 'budget': {'optimizer_llm_calls', 'eval_llm_calls', 'candidates', 'evaluator_runs', 'wall_time_s', 'total_tokens', 'on_exceed'}, 'experiment': {'seeds', 'arms', 'matrix'}}
_LEVEL_KEYS = {'id', 'depends_on', 'ordering_only', *_LEVEL_BLOCKS}
_DATASET_REF_KEYS = {'ref', 'split', 'config'}
_METRIC_KEYS = {'direction', 'source', 'aggregate_examples'}
_PROFILE_KEYS = {'provider', 'model', 'resolved_model', 'api_key_ref', 'fallbacks', 'temperature', 'max_tokens', 'base_url'}
_ROLE_KEYS = {'forward', 'optimizer', 'feedback', 'judge'}
_ROLE_OVERRIDE_KEYS = {'profile', *_PROFILE_KEYS}
_BINDING_KEYS = {'from', 'to', 'codec', 'ordering_only'}
_VERSIONED_REF = re.compile('^[A-Za-z0-9_.-]+@[1-9][0-9]*$')
_SECRET_KEYS = {'api_key', 'apikey', 'access_token', 'token', 'secret', 'password'}

class _FrozenDict(dict):
    """JSON-serializable dictionary that rejects mutation after construction."""

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError('normalized recursive-opt specs are immutable')
    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

@dataclass(frozen=True)
class ModuleRegistryEntry:
    """Build and persist one versioned kind of generic ``trace.Module``."""
    build: Callable[[Mapping[str, Any], Mapping[str, Any]], Module]
    snapshot: Callable[[Module], Dict[str, Any]]
    restore: Callable[[Module, Mapping[str, Any]], None]
    validate_artifact: Callable[[Mapping[str, Any]], None]
    capabilities: frozenset[str]
    validate_config: Optional[Callable[[Mapping[str, Any]], None]] = None

@dataclass(frozen=True)
class EngineRegistryEntry:
    """Run one execution unit and declare supported compile-time capabilities."""
    run: Callable[['_ExecutionUnit', '_LevelPlan', Mapping[str, Any]], 'RunResult']
    capabilities: frozenset[str]

@dataclass(frozen=True)
class _LevelPlan:
    """One immutable, fully resolved level within an execution unit."""
    level_id: str
    depends_on: Tuple[str, ...]
    ordering_only: bool
    spec: Mapping[str, Any]
    datasets: Mapping[str, Any]
    fingerprint: str

@dataclass(frozen=True)
class _ExecutionUnit:
    """One internal, fully materialized arm/seed/matrix execution unit."""
    unit_id: str
    arm_id: str
    seed: Optional[int]
    matrix: Mapping[str, Any]
    spec: Mapping[str, Any]
    levels: Tuple[_LevelPlan, ...]

@dataclass(frozen=True)
class RunResult:
    """Canonical, JSON-exportable result of one compiled execution unit."""
    unit_id: str
    plan_fingerprint: str
    spec_fingerprint: str
    engine: str
    module_ref: str
    status: str
    valid: bool
    evaluation: EvaluationResult
    artifact: Mapping[str, Any]
    lineage: Tuple[Mapping[str, Any], ...]
    usage: Mapping[str, Any]
    budget: Mapping[str, Any]
    metadata: Mapping[str, Any]
    error: Optional[str] = None
    level_results: Tuple[Mapping[str, Any], ...] = ()
    portable: bool = True
    promotable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Return the canonical result as plain JSON-compatible containers."""
        return {'unit_id': self.unit_id, 'plan_fingerprint': self.plan_fingerprint, 'spec_fingerprint': self.spec_fingerprint, 'engine': self.engine, 'module_ref': self.module_ref, 'status': self.status, 'valid': self.valid, 'evaluation': {'valid': self.evaluation.valid, 'status': self.evaluation.status, 'metrics': _thaw(self.evaluation.metrics), 'feedback': _thaw(self.evaluation.feedback), 'trace': _thaw(self.evaluation.trace), 'usage': _thaw(self.evaluation.usage), 'artifacts': _thaw(self.evaluation.artifacts), 'error': self.evaluation.error}, 'artifact': _thaw(self.artifact), 'lineage': _thaw(self.lineage), 'usage': _thaw(self.usage), 'budget': _thaw(self.budget), 'metadata': _thaw(self.metadata), 'error': self.error, 'level_results': _thaw(self.level_results), 'portable': self.portable, 'promotable': self.promotable}

@dataclass(frozen=True)
class ExecutionPlan:
    """Immutable compilation product consumed by engine runners."""
    spec: Mapping[str, Any]
    units: Tuple[_ExecutionUnit, ...]
    fingerprint: str
    raw_spec: Mapping[str, Any]

    def explain(self) -> Dict[str, Any]:
        """Return a compact JSON explanation of this immutable plan."""
        return {'fingerprint': self.fingerprint, 'execution_units': len(self.units), 'engines': sorted({level.spec['engine']['name'] for unit in self.units for level in unit.levels}), 'module_refs': sorted({level.spec['module']['ref'] for unit in self.units for level in unit.levels}), 'level_ids': [level.level_id for level in self.units[0].levels], 'unit_ids': [unit.unit_id for unit in self.units]}

class DatasetAccess:
    """Capability gate that prevents holdout access during optimization phases."""
    _PHASES = {'fit', 'proposal', 'induction', 'candidate_selection', 'final_evaluation', 'promotion', 'report'}
    _HOLDOUT_PHASES = {'final_evaluation', 'promotion', 'report'}

    def __init__(self, datasets: Mapping[str, Any]) -> None:
        if not isinstance(datasets, Mapping):
            raise TypeError('datasets must be a mapping')
        unknown = set(datasets) - {'train', 'validation', 'holdout'}
        if unknown:
            raise ValueError(f'unknown dataset splits: {sorted(unknown)}')
        self._datasets = _freeze({split: _thaw(datasets.get(split, [])) for split in ('train', 'validation', 'holdout')})

    def read(self, split: str, *, phase: str) -> Any:
        """Read one split only when the named execution phase has capability."""
        if phase not in self._PHASES:
            raise ValueError(f'unknown dataset access phase {phase!r}')
        if split not in self._datasets:
            raise ValueError(f'unknown dataset split {split!r}')
        if split == 'holdout' and phase not in self._HOLDOUT_PHASES:
            raise PermissionError(f'holdout is inaccessible during {phase}')
        return _thaw(self._datasets[split])

class _ComponentModule(Module):
    """Small generic multi-component module used by registered workflows."""

    def __init__(self, components: Mapping[str, Any], inputs: Mapping[str, Any]) -> None:
        if not components:
            raise ValueError('module.config.components must be a non-empty mapping')
        self.components = Map({name: node(value, name=name, trainable=True) for name, value in components.items()})
        self.inputs = _thaw(inputs)

    def forward(self, inputs: Any) -> Dict[str, Any]:
        """Expose resolved component values alongside the module input."""
        runtime_inputs = getattr(inputs, 'data', inputs)
        resolved_inputs = {**self.inputs, **runtime_inputs} if isinstance(runtime_inputs, Mapping) else {**self.inputs, 'value': runtime_inputs}
        return {'inputs': resolved_inputs, 'components': {name: value.data for name, value in self.components.items()}}
_MODULE_REGISTRY: Dict[str, ModuleRegistryEntry] = {}
_ENGINE_REGISTRY: Dict[str, EngineRegistryEntry] = {}
_EVALUATOR_REGISTRY: Dict[str, Callable[[Module, Any, Mapping[str, Any]], EvaluationResult]] = {}
_DATASET_REGISTRY: Dict[str, Callable[[str, Mapping[str, Any]], Any]] = {}

@dataclass(frozen=True)
class _CodecEntry:
    """Typed internal codec used by causal bindings."""
    encode: Callable[[Any], Any]
    input_type: Any
    output_type: Any
    input_description: str
_CODEC_REGISTRY: Dict[str, _CodecEntry] = {}

def register_module(ref: str, entry: ModuleRegistryEntry) -> None:
    """Register one exact module reference, rejecting aliases and replacement."""
    if not isinstance(ref, str) or not _VERSIONED_REF.fullmatch(ref):
        raise ValueError('module registry keys must be exact versioned refs')
    if not isinstance(entry, ModuleRegistryEntry):
        raise TypeError('entry must be a ModuleRegistryEntry')
    if entry.validate_config is None:
        raise ValueError('module registry entries require a config validator')
    if ref in _MODULE_REGISTRY and _MODULE_REGISTRY[ref] != entry:
        raise ValueError(f'module ref {ref!r} is already registered')
    _MODULE_REGISTRY[ref] = entry

def register_engine(name: str, entry: EngineRegistryEntry) -> None:
    """Register one engine runner without importing optional dependencies."""
    if not isinstance(name, str) or not name:
        raise ValueError('engine name must be a non-empty string')
    if not isinstance(entry, EngineRegistryEntry):
        raise TypeError('entry must be an EngineRegistryEntry')
    if name in _ENGINE_REGISTRY and _ENGINE_REGISTRY[name] != entry:
        raise ValueError(f'engine {name!r} is already registered')
    _ENGINE_REGISTRY[name] = entry

def register_evaluator(ref: str, evaluator: Callable[[Module, Any, Mapping[str, Any]], EvaluationResult]) -> None:
    """Register an exact versioned evaluator without import fallback."""
    _register_callable(_EVALUATOR_REGISTRY, ref, evaluator, 'evaluator')

def register_dataset(ref: str, resolver: Callable[[str, Mapping[str, Any]], Any]) -> None:
    """Register an exact versioned dataset resolver."""
    _register_callable(_DATASET_REGISTRY, ref, resolver, 'dataset')

def _register_callable(registry: MutableMapping[str, Callable[..., Any]], ref: str, value: Callable[..., Any], kind: str) -> None:
    """Apply the shared exact-ref and replacement rules to callable registries."""
    if not isinstance(ref, str) or not _VERSIONED_REF.fullmatch(ref):
        raise ValueError(f'{kind} registry keys must be exact versioned refs')
    if not callable(value):
        raise TypeError(f'{kind} must be callable')
    if ref in registry and registry[ref] is not value:
        raise ValueError(f'{kind} ref {ref!r} is already registered')
    registry[ref] = value

def register_codec(ref: str, encode: Callable[[Any], Any], *, input_type: Any, output_type: Any, input_description: str) -> None:
    """Register an exact typed codec used by explicit causal bindings."""
    if not isinstance(ref, str) or not _VERSIONED_REF.fullmatch(ref):
        raise ValueError('codec registry keys must be exact versioned refs')
    if not callable(encode):
        raise TypeError('codec encode must be callable')
    entry = _CodecEntry(encode, input_type, output_type, input_description)
    if ref in _CODEC_REGISTRY and _CODEC_REGISTRY[ref] != entry:
        raise ValueError(f'codec ref {ref!r} is already registered')
    _CODEC_REGISTRY[ref] = entry

def build_module(spec: Mapping[str, Any], resources: Optional[Mapping[str, Any]]=None, *, level_id: Optional[str]=None) -> Module:
    """Build one declared level module, restoring its artifact and targets."""
    level = _select_level(normalize_spec(spec), level_id)
    module = _build_level_module(level, resources or {})
    return module

def _build_level_module(level: Mapping[str, Any], resources: Mapping[str, Any]) -> Module:
    """Build and initialize a module from one canonical level mapping."""
    entry = _module_entry(level['module']['ref'])
    config = level['module']['config']
    if entry.validate_config is not None:
        entry.validate_config(config)
    module = entry.build(level, resources)
    if not isinstance(module, Module):
        raise TypeError(f'module builder returned {type(module).__name__}, expected trace.Module')
    artifact = level['module'].get('artifact')
    if artifact is not None:
        entry.validate_artifact(artifact)
        entry.restore(module, artifact)
    _apply_trainable_targets(module, level['surface']['targets'])
    return module

def snapshot_module(spec: Mapping[str, Any], module: Module, *, level_id: Optional[str]=None) -> Dict[str, Any]:
    """Create and validate a JSON-serializable artifact for ``module``."""
    level = _select_level(normalize_spec(spec), level_id)
    entry = _module_entry(level['module']['ref'])
    artifact = entry.snapshot(module)
    entry.validate_artifact(artifact)
    _validate_no_callables_or_secrets(artifact, 'module artifact')
    _canonical_json(artifact)
    return _thaw(artifact)

def restore_module(spec: Mapping[str, Any], module: Module, artifact: Mapping[str, Any], *, level_id: Optional[str]=None) -> None:
    """Validate and restore a registered module artifact in place."""
    level = _select_level(normalize_spec(spec), level_id)
    entry = _module_entry(level['module']['ref'])
    entry.validate_artifact(artifact)
    entry.restore(module, artifact)

def _select_level(spec: Mapping[str, Any], level_id: Optional[str]) -> Mapping[str, Any]:
    """Select one canonical level, requiring an id for multilevel specs."""
    levels = spec['levels']
    if level_id is None:
        if len(levels) != 1:
            raise ValueError('level_id is required for a multilevel spec')
        return levels[0]
    for level in levels:
        if level['id'] == level_id:
            return level
    raise ValueError(f'unknown level id {level_id!r}')

def _apply_trainable_targets(module: Module, targets: Iterable[str]) -> None:
    """Validate target names and mark only selected parameters trainable."""
    parameters = list(module.parameters())
    names = {parameter.name.split(':', 1)[0]: parameter for parameter in parameters}
    selected = set(targets)
    if selected == {'*'}:
        selected = set(names)
    unknown = selected - set(names)
    if unknown:
        raise ValueError(f'unknown surface.targets: {sorted(unknown)}; available: {sorted(names)}')
    for name, parameter in names.items():
        parameter.trainable = name in selected

def compile_plan(raw_spec: Mapping[str, Any]) -> ExecutionPlan:
    """Normalize and compile experiment units containing immutable level plans."""
    normalized = normalize_spec(raw_spec)
    if normalized['runtime']['tracebench']:
        TB.configure_tracebench_adapter(_thaw(normalized['runtime']['tracebench']), require=True)
    _resolve_knowledge_store(normalized['knowledge']['store'])
    units = tuple(_expand_execution_units(normalized))
    if not units:
        raise ValueError('spec expansion produced no execution units')
    for unit in units:
        for level in unit.levels:
            module = _module_entry(level.spec['module']['ref'])
            if module.validate_config is not None:
                module.validate_config(level.spec['module']['config'])
            _evaluator_entry(level.spec['objective']['evaluator_ref'])
            engine = _engine_entry(level.spec['engine']['name'])
            compile_objective(level.spec['objective'], capabilities=engine.capabilities)
            for binding in level.spec['bindings']:
                if not binding['ordering_only']:
                    _codec_entry(binding['codec'])
    return ExecutionPlan(spec=normalized, units=units, fingerprint=normalized['fingerprint'], raw_spec=_freeze(_thaw(raw_spec)))

def execute_plan(plan: ExecutionPlan, resources: Optional[Mapping[str, Any]]=None) -> Tuple[RunResult, ...]:
    """Execute each unit through one ordered canonical multilevel runner."""
    if not isinstance(plan, ExecutionPlan):
        raise TypeError('plan must be an ExecutionPlan')
    runtime_resources = dict(resources or {})
    overrides = _validate_runtime_resources(plan.spec, runtime_resources)
    output_root = _prepare_output_root(plan)
    preflight_checker = runtime_resources.get('preflight_checker')
    if not plan.spec['runtime']['offline'] and (not plan.spec['runtime']['test_mode'] or preflight_checker is not None):
        preflight_llm_profiles(plan.spec, checker=preflight_checker)
    results: List[RunResult] = []
    for unit in plan.units:
        guard = _BudgetGuard(unit.spec['budget'])
        memory = runtime_resources.get('memory') or _knowledge_store(unit.spec['knowledge']['store'], unit.spec)
        upstream: Dict[str, Any] = {}
        level_results: List[RunResult] = []
        all_cached = True
        seed = unit.seed if unit.seed is not None else unit.spec['runtime']['seed']
        with _seed_scope(seed):
            for level in unit.levels:
                cached = _load_resume(plan, unit, level, output_root)
                if cached is not None:
                    result = cached
                    guard.restore(result.budget)
                else:
                    all_cached = False
                    level_resources = {**runtime_resources, '_upstream': upstream, '_budget': guard, '_memory': memory, '_overrides': overrides}
                    try:
                        result = _engine_entry(level.spec['engine']['name']).run(unit, level, level_resources)
                    except Exception as exc:
                        if _should_raise(unit.spec['budget'], exc):
                            raise
                        result = _failed_level_result(plan, unit, level, guard, exc)
                    result = RunResult(**{**result.__dict__, 'plan_fingerprint': plan.fingerprint, 'portable': not bool(overrides), 'promotable': not bool(overrides) and result.valid})
                    _persist_level_result(plan, unit, level, result, output_root)
                level_results.append(result)
                upstream[level.level_id] = {'outputs': _level_outputs(result)}
                if result.status == 'error' or (not result.valid and unit.spec['budget']['on_exceed'] == 'fail'):
                    break
        if all_cached:
            persisted = _load_final_result(plan, unit, output_root)
            if persisted is not None:
                results.append(persisted)
                continue
        final = _combine_unit_result(plan, unit, level_results, guard, overrides)
        _persist_final_result(plan, unit, final, output_root)
        results.append(final)
    return tuple(results)

def apply_bindings(spec: Mapping[str, Any], outputs: Mapping[str, Any], module_inputs: MutableMapping[str, Any], *, level_id: Optional[str]=None) -> List[Dict[str, Any]]:
    """Apply typed causal bindings and return lineage for every injected value."""
    level = _select_level(normalize_spec(spec), level_id)
    if not isinstance(outputs, Mapping):
        raise TypeError('binding outputs must be a mapping')
    if not isinstance(module_inputs, MutableMapping):
        raise TypeError('module_inputs must be a mutable mapping')
    lineage: List[Dict[str, Any]] = []
    for binding in level['bindings']:
        if binding.get('ordering_only', False):
            continue
        source = _resolve_dotted_path(outputs, binding['from'])
        entry = _codec_entry(binding['codec'])
        if not isinstance(source, entry.input_type):
            raise TypeError(f"codec {binding['codec']!r} requires {entry.input_description}; got {type(source).__name__}")
        encoded = entry.encode(source)
        if not isinstance(encoded, entry.output_type):
            raise TypeError(f"codec {binding['codec']!r} returned an invalid output type")
        destination = binding['to'].split('.')
        if destination[:2] != ['module', 'inputs'] or len(destination) < 3:
            raise ValueError('binding destinations must be below module.inputs')
        _set_nested_value(module_inputs, destination[2:], encoded)
        artifact_id = source.artifact_id if isinstance(source, ArtifactRecord) else source.get('artifact_id') if isinstance(source, Mapping) else None
        lineage.append({'from': binding['from'], 'to': binding['to'], 'codec': binding['codec'], 'artifact_id': artifact_id})
    return lineage

def compile_objective(objective: Mapping[str, Any], *, capabilities: Iterable[str]) -> Dict[str, Any]:
    """Compile a canonical objective with engine capability validation."""
    from opto.trainer.objectives import ObjectiveConfig
    if not isinstance(objective, Mapping):
        raise TypeError('objective must be a mapping')
    selection = objective.get('selection')
    metrics = objective.get('metrics')
    if not isinstance(selection, Mapping) or not isinstance(metrics, Mapping):
        raise TypeError('objective requires selection and metric descriptor mappings')
    mode = str(selection.get('mode', 'scalar'))
    supported = set(capabilities)
    if mode not in supported:
        raise ValueError(f'engine does not support objective mode {mode!r}')
    minimize = frozenset((metric for metric, descriptor in metrics.items() if descriptor['direction'] == 'minimize'))
    config = ObjectiveConfig(mode=mode, weights=dict(selection.get('weights') or {}), minimize=minimize, pareto_metrics=tuple(selection['pareto_metrics']) if selection.get('pareto_metrics') is not None else None, tie_break=str(selection.get('tie_break', 'weighted')), seed=int(selection.get('seed', 0)), scalarize_dict=str(selection.get('scalarize_dict', 'score')), score_key=str(selection.get('score_key', 'score')))
    return {'config': config, 'intent': objective.get('intent', ''), 'metrics': _freeze(_thaw(metrics)), 'hard_constraints': tuple(_thaw(objective.get('hard_constraints', ()))), 'aggregation': _freeze(_thaw(objective.get('aggregation', {}))), 'feedback_channels': tuple(objective.get('feedback_channels', ()))}

def resolve_llm_roles(spec: Mapping[str, Any], overrides: Optional[Mapping[str, Any]]=None, *, level_id: Optional[str]=None) -> Mapping[str, Any]:
    """Resolve all global or level-local LLM role overrides to exact profiles."""
    normalized = normalize_spec(spec)
    roles = _thaw(_select_level(normalized, level_id)['llm_roles'])
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise TypeError('llm role overrides must be a mapping')
        _reject_unknown_keys(overrides, _ROLE_KEYS, 'level llm_roles')
        roles.update(_thaw(overrides))
    resolved = {role: _materialize_role(value, normalized['llm_profiles'], role) for role, value in roles.items()}
    return _freeze(resolved)

def preflight_llm_profiles(spec: Mapping[str, Any], *, checker: Optional[Callable[[str], None]]=None) -> None:
    """Check every exact model used by a role once, propagating provider errors."""
    if checker is None:
        from .runmode import preflight_model
        checker = preflight_model
    normalized = normalize_spec(spec)
    checked: set[str] = set()
    for level in normalized['levels']:
        for profile in level['llm_roles'].values():
            if profile is None:
                continue
            candidates = [profile] + [normalized['llm_profiles'][name] for name in profile['fallbacks']]
            for candidate in candidates:
                model = candidate['resolved_model']
                if model not in checked:
                    checker(model)
                    checked.add(model)

def retrieve_knowledge(spec: Mapping[str, Any], memory: Optional[MemoryLite], scope: Mapping[str, str]) -> List[Any]:
    """Retrieve promoted knowledge explicitly in the runner, before module build."""
    normalized = normalize_spec(spec)
    memory = memory or _knowledge_store(normalized['knowledge']['store'], normalized)
    if not isinstance(scope, Mapping):
        raise TypeError('knowledge retrieval scope must be a mapping')
    policy = normalized['knowledge']
    scoped = {field: scope[field] for field in policy['scope_fields'] if field in scope}
    result = memory.retrieve(artifact_type='knowledge_card', statuses=policy['statuses'], scope=scoped, topk=policy['top_k'], sort=policy['retrieval'])
    return result['artifacts']

def _module_entry(ref: str) -> ModuleRegistryEntry:
    """Resolve one exact module ref without import fallback."""
    entry = _MODULE_REGISTRY.get(ref)
    if entry is None:
        raise ValueError(f'unregistered module ref {ref!r}')
    return entry

def _engine_entry(name: str) -> EngineRegistryEntry:
    """Resolve one exact engine name without fallback."""
    entry = _ENGINE_REGISTRY.get(name)
    if entry is None:
        raise ValueError(f'unregistered engine {name!r}')
    return entry

def _evaluator_entry(ref: str) -> Callable[[Module, Any, Mapping[str, Any]], EvaluationResult]:
    """Resolve one exact evaluator ref without dynamic imports."""
    evaluator = _EVALUATOR_REGISTRY.get(ref)
    if evaluator is None:
        raise ValueError(f'unregistered evaluator ref {ref!r}')
    return evaluator

def _dataset_entry(ref: str) -> Callable[[str, Mapping[str, Any]], Any]:
    """Resolve one exact dataset ref without dynamic import fallback."""
    resolver = _DATASET_REGISTRY.get(ref)
    if resolver is None:
        raise ValueError(f'unregistered dataset ref {ref!r}')
    return resolver

def _knowledge_store(ref: str, spec: Mapping[str, Any]) -> MemoryLite:
    """Resolve the minimum supported portable knowledge-store contract."""
    _resolve_knowledge_store(ref)
    promotion = spec['runtime']['prior_promotion']
    return MemoryLite(root=str(spec['runtime']['memory_root']), promotion_min_support=int(promotion.get('min_support', 3)), promote_priors=bool(promotion.get('enabled', True)), promotion_min_score=promotion.get('min_score'))

def _resolve_knowledge_store(ref: str) -> None:
    """Validate one exact knowledge-store ref without opening storage."""
    if ref != 'recursive_opt.knowledge.memory_lite@1':
        raise ValueError(f'unregistered knowledge store ref {ref!r}')

def _codec_entry(ref: str) -> _CodecEntry:
    """Resolve one exact codec ref without import fallback."""
    entry = _CODEC_REGISTRY.get(ref)
    if entry is None:
        raise ValueError(f'unregistered codec ref {ref!r}')
    return entry

def _resolve_dotted_path(value: Mapping[str, Any], path: str) -> Any:
    """Resolve a required dotted source path from execution outputs."""
    current: Any = value
    for part in path.split('.'):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f'binding source {path!r} is unavailable')
        current = current[part]
    return current

def _set_nested_value(target: MutableMapping[str, Any], parts: List[str], value: Any) -> None:
    """Set a typed binding destination below module.inputs."""
    current = target
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, MutableMapping):
            raise ValueError(f'binding destination segment {part!r} is not a mapping')
        current = child
    current[parts[-1]] = value

def _artifact_to_prior(value: Any) -> Dict[str, Any]:
    """Convert an artifact record/mapping into an injected prior with lineage."""
    if isinstance(value, ArtifactRecord):
        return {'knowledge': value.content}
    if not isinstance(value, Mapping) or 'content' not in value:
        raise TypeError('artifact_to_prior requires a mapping artifact with content')
    return {'knowledge': _thaw(value['content'])}

def _component_dict(value: Any) -> Dict[str, Any]:
    """Copy a string-keyed component mapping for module input injection."""
    if not isinstance(value, Mapping) or not all((isinstance(key, str) for key in value)):
        raise TypeError('component_dict requires a string-keyed mapping')
    return _thaw(value)

class _BudgetGuard:
    """Enforce one execution unit's common budget before consuming operations."""

    def __init__(self, limits: Mapping[str, Any]) -> None:
        self.limits = _thaw(limits)
        self.used = {'optimizer_llm_calls': 0, 'eval_llm_calls': 0, 'candidates': 0, 'evaluator_runs': 0, 'total_tokens': 0}
        self.started_at = time.monotonic()
        self.previous_wall_time_s = 0.0

    def consume(self, resource: str, amount: int=1) -> None:
        """Charge only when both wall time and the named resource have capacity."""
        self.check_wall_time()
        if resource not in self.used:
            raise ValueError(f'unknown budget resource {resource!r}')
        if not isinstance(amount, int) or amount < 0:
            raise ValueError('budget charges must be non-negative integers')
        limit = self.limits.get(resource)
        if limit is not None and self.used[resource] + amount > limit:
            raise BudgetExceeded(f'budget exhausted for {resource}: requested {amount}, used {self.used[resource]}, limit {limit}')
        self.used[resource] += amount

    def require(self, resource: str, amount: int=1) -> None:
        """Check capacity without charging it."""
        self.check_wall_time()
        limit = self.limits.get(resource)
        if limit is not None and self.used[resource] + amount > limit:
            raise BudgetExceeded(f'budget exhausted for {resource}: requested {amount}, used {self.used[resource]}, limit {limit}')

    def check_wall_time(self) -> None:
        """Raise before work when the unit wall-time limit is exhausted."""
        limit = self.limits.get('wall_time_s')
        if limit is not None and self.previous_wall_time_s + time.monotonic() - self.started_at >= limit:
            raise BudgetExceeded(f'budget exhausted for wall_time_s at limit {limit}')

    def restore(self, report: Mapping[str, Any]) -> None:
        """Restore cumulative counters from one exact resumed level result."""
        accounted = report.get('accounted', {})
        if not isinstance(accounted, Mapping):
            raise ValueError('resumed budget report is missing accounted counters')
        for resource in self.used:
            self.used[resource] = max(self.used[resource], int(accounted.get(resource, 0)))
        self.previous_wall_time_s = max(self.previous_wall_time_s, float(accounted.get('wall_time_s', 0.0)))

    def report(self) -> Mapping[str, Any]:
        """Return immutable limits and exact common counters."""
        accounted = {**self.used, 'wall_time_s': round(self.previous_wall_time_s + max(0.0, time.monotonic() - self.started_at), 6)}
        return _freeze({**self.limits, 'accounted': accounted, 'exceeded': []})

@contextmanager
def _seed_scope(seed: Optional[int]) -> Iterable[None]:
    """Apply and restore Python/NumPy RNG state for one execution unit."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % 2 ** 32)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)

def _validate_runtime_resources(spec: Mapping[str, Any], resources: Mapping[str, Any]) -> Mapping[str, str]:
    """Reject hidden behavior unless explicit non-portable test mode is enabled."""
    behavioral = {'evaluator', 'gepa_optimize', 'gepa_config', 'optimizer', 'trainer', 'llm_factory', 'memory', 'graph_executors', 'legacy_levels', 'preflight_checker'}
    observational = {'capture'}
    unknown = set(resources) - behavioral - observational
    if unknown:
        raise ValueError(f'unknown runtime resources: {sorted(unknown)}')
    active = sorted(set(resources) & behavioral)
    if active and (not spec['runtime']['test_mode']):
        raise ValueError(f'portable strict mode rejects behavioral resources: {active}; set runtime.test_mode=true only for tests')
    return _freeze({name: _override_identity(resources[name]) for name in active})

def _override_identity(value: Any) -> str:
    """Return a stable descriptive identity without serializing behavior."""
    target = value if callable(value) else type(value)
    module = getattr(target, '__module__', type(value).__module__)
    name = getattr(target, '__qualname__', getattr(target, '__name__', type(value).__qualname__))
    return f'{module}.{name}'

def _prepare_output_root(plan: ExecutionPlan) -> Optional[Path]:
    """Create the fingerprinted output root and persist immutable run inputs."""
    directory = plan.spec['outputs']['directory']
    if directory is None:
        if plan.spec['runtime']['resume']:
            raise ValueError('runtime.resume requires outputs.directory')
        return None
    root = Path(directory) / plan.fingerprint
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / 'raw_spec.json', plan.raw_spec)
    _write_json(root / 'normalized_spec.json', plan.spec)
    _write_json(root / 'resolved_execution_plan.json', {**plan.explain(), 'units': [{'unit_id': unit.unit_id, 'seed': unit.seed, 'matrix': unit.matrix, 'levels': [{'id': level.level_id, 'fingerprint': level.fingerprint, 'engine': level.spec['engine']['name'], 'module_ref': level.spec['module']['ref'], 'evaluator_ref': level.spec['objective']['evaluator_ref'], 'dataset_refs': _dataset_identities(level.spec['datasets'])} for level in unit.levels]} for unit in plan.units]})
    return root

def _write_json(path: Path, value: Any) -> None:
    """Persist JSON atomically in the destination directory."""
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def _resume_identity(plan: ExecutionPlan, unit: _ExecutionUnit, level: _LevelPlan) -> Dict[str, Any]:
    """Return every identity dimension required for safe level resume."""
    return {'spec_fingerprint': plan.fingerprint, 'unit_id': unit.unit_id, 'level_id': level.level_id, 'level_fingerprint': level.fingerprint, 'engine': level.spec['engine']['name'], 'module_ref': level.spec['module']['ref'], 'evaluator_ref': level.spec['objective']['evaluator_ref'], 'dataset_refs': _dataset_identities(level.spec['datasets'])}

def _level_result_path(root: Path, unit: _ExecutionUnit, level: _LevelPlan) -> Path:
    """Return the persisted result path for one unit/level pair."""
    return root / 'units' / unit.unit_id / 'levels' / level.level_id / 'result.json'

def _load_resume(plan: ExecutionPlan, unit: _ExecutionUnit, level: _LevelPlan, root: Optional[Path]) -> Optional[RunResult]:
    """Load only a complete result with an exact persisted identity."""
    if not unit.spec['runtime']['resume'] or root is None:
        return None
    path = _level_result_path(root, unit, level)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get('complete') is not True or payload.get('identity') != _resume_identity(plan, unit, level):
        return None
    return _run_result_from_dict(payload['result'])

def _persist_level_result(plan: ExecutionPlan, unit: _ExecutionUnit, level: _LevelPlan, result: RunResult, root: Optional[Path]) -> None:
    """Persist a complete level result and its operational side records."""
    if root is None:
        return
    path = _level_result_path(root, unit, level)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, {'complete': True, 'identity': _resume_identity(plan, unit, level), 'result': result.to_dict()})
    _write_json(path.parent / 'evaluator_records.json', result.metadata.get('evaluator_records', []))
    _write_json(path.parent / 'usage.json', result.usage)
    _write_json(path.parent / 'budget.json', result.budget)
    _write_json(path.parent / 'lineage.json', result.lineage)
    _write_json(path.parent / 'errors.json', [] if result.error is None else [result.error])
    if level.spec['outputs']['save_artifacts']:
        _write_json(path.parent / 'module_artifact.json', result.artifact)

def _persist_final_result(plan: ExecutionPlan, unit: _ExecutionUnit, result: RunResult, root: Optional[Path]) -> None:
    """Persist the final canonical unit result atomically."""
    if root is not None:
        path = root / 'units' / unit.unit_id / 'run_result.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, result.to_dict())

def _load_final_result(plan: ExecutionPlan, unit: _ExecutionUnit, root: Optional[Path]) -> Optional[RunResult]:
    """Load the exact final result after every constituent level resumed."""
    if root is None or not unit.spec['runtime']['resume']:
        return None
    path = root / 'units' / unit.unit_id / 'run_result.json'
    if not path.exists():
        return None
    try:
        result = _run_result_from_dict(json.loads(path.read_text(encoding='utf-8')))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if result.plan_fingerprint != plan.fingerprint or result.spec_fingerprint != unit.spec['fingerprint'] or result.unit_id != unit.unit_id or (len(result.level_results) != len(unit.levels)):
        return None
    return result

def _run_result_from_dict(value: Mapping[str, Any]) -> RunResult:
    """Rebuild the canonical typed result from persisted JSON."""
    evaluation = EvaluationResult(**dict(value['evaluation']))
    return RunResult(unit_id=value['unit_id'], plan_fingerprint=value['plan_fingerprint'], spec_fingerprint=value['spec_fingerprint'], engine=value['engine'], module_ref=value['module_ref'], status=value['status'], valid=value['valid'], evaluation=evaluation, artifact=_freeze(value['artifact']), lineage=tuple(_freeze(value['lineage'])), usage=_freeze(value['usage']), budget=_freeze(value['budget']), metadata=_freeze(value['metadata']), error=value.get('error'), level_results=tuple(_freeze(value.get('level_results', []))), portable=bool(value.get('portable', True)), promotable=bool(value.get('promotable', True)))

def _level_outputs(result: RunResult) -> Dict[str, Any]:
    """Expose typed outputs from an actually executed upstream level."""
    artifact_id = hashlib.sha256(_canonical_json(result.artifact).encode('utf-8')).hexdigest()
    return {'artifact': {'artifact_id': artifact_id, 'content': _thaw(result.artifact)}, 'evaluation': result.to_dict()['evaluation'], 'usage': _thaw(result.usage)}

def _failed_level_result(plan: ExecutionPlan, unit: _ExecutionUnit, level: _LevelPlan, guard: _BudgetGuard, error: Exception) -> RunResult:
    """Convert a safe non-raising level failure into a canonical result."""
    message = _safe_error(error)
    evaluation = EvaluationResult(valid=False, status='error', metrics={}, feedback='execution failed', error=message)
    return RunResult(unit_id=f'{unit.unit_id}:{level.level_id}', plan_fingerprint=plan.fingerprint, spec_fingerprint=unit.spec['fingerprint'], engine=level.spec['engine']['name'], module_ref=level.spec['module']['ref'], status='error', valid=False, evaluation=evaluation, artifact={}, lineage=(), usage=evaluation.usage, budget=guard.report(), metadata={'level_id': level.level_id}, error=message)

def _should_raise(budget: Mapping[str, Any], error: Exception) -> bool:
    """Return whether an execution exception must escape the canonical runner."""
    return isinstance(error, BudgetExceeded) and budget['on_exceed'] == 'raise'

def _combine_unit_result(plan: ExecutionPlan, unit: _ExecutionUnit, results: List[RunResult], guard: _BudgetGuard, overrides: Mapping[str, str]) -> RunResult:
    """Combine ordered level results into one canonical experiment-unit result."""
    if not results:
        raise RuntimeError('execution unit produced no level results')
    final = results[-1]
    valid = len(results) == len(unit.levels) and all((result.valid for result in results))
    metadata = {**_thaw(final.metadata), 'level_ids': [level.level_id for level in unit.levels], 'resolved_models': {result.metadata['level_id']: result.metadata.get('selected_models', {}) for result in results}, 'test_overrides': _thaw(overrides), 'arm_id': unit.arm_id, 'seed': unit.seed if unit.seed is not None else unit.spec['runtime']['seed'], 'matrix': _thaw(unit.matrix)}
    return RunResult(unit_id=unit.unit_id, plan_fingerprint=plan.fingerprint, spec_fingerprint=unit.spec['fingerprint'], engine=final.engine, module_ref=final.module_ref, status=final.status, valid=valid, evaluation=final.evaluation, artifact=final.artifact, lineage=tuple((item for result in results for item in result.lineage)), usage=_merge_usage([result.usage for result in results]), budget=guard.report(), metadata=_freeze(metadata), error=final.error, level_results=tuple((result.to_dict() for result in results)), portable=not bool(overrides), promotable=not bool(overrides) and valid)

def _merge_usage(items: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Sum canonical role usage across level results."""
    merged: Dict[str, Dict[str, float | int]] = {}
    for usage in items:
        for role, values in usage.items():
            target = merged.setdefault(role, {})
            for name, amount in values.items():
                target[name] = target.get(name, 0) + amount
    return _freeze(merged)

def _run_fixed_engine(unit: _ExecutionUnit, level: _LevelPlan, resources: Mapping[str, Any]) -> RunResult:
    """Evaluate a fixed registered module without fitting it."""
    return _run_module_engine(unit, level, resources, fit=False)

def _run_trace_engine(unit: _ExecutionUnit, level: _LevelPlan, resources: Mapping[str, Any]) -> RunResult:
    """Optimize a registered module through the existing Trace optimize path."""
    if level.spec['module']['ref'] == 'recursive_opt.module.legacy_level@1':
        return _run_legacy_trace_engine(unit, level, resources)
    return _run_module_engine(unit, level, resources, fit=True)

def _run_legacy_trace_engine(unit: _ExecutionUnit, level: _LevelPlan, resources: Mapping[str, Any]) -> RunResult:
    """Execute one migrated legacy level inside the canonical engine boundary."""
    canonical = level.spec
    configured_level = _thaw(canonical['module']['config']['level'])
    legacy = _thaw(resources.get('legacy_levels', {}).get(configured_level['id'], configured_level))
    families = _thaw(canonical['module']['config']['families'])
    memory: MemoryLite = resources['_memory']
    module = _build_level_module(canonical, {'memory': memory, **({'legacy_levels': resources['legacy_levels']} if 'legacy_levels' in resources else {})})
    reuse = reuse_priors(memory, module, legacy) if unit.spec['runtime']['reuse_priors'] else {'used_prior': False, 'tools': []}
    config = canonical['engine']['config']
    iterations = int(config['iterations'])
    num_candidates = int(config['num_candidates'])
    guard: _BudgetGuard = resources['_budget']
    should_fit = True
    try:
        guard.consume('candidates', iterations * num_candidates)
    except BudgetExceeded:
        if unit.spec['budget']['on_exceed'] != 'return_best_valid':
            raise
        should_fit = False
    objective_config = legacy.get('objective_config')
    optimizer_kwargs = {**_thaw(config['optimizer_kwargs']), **_thaw(legacy.get('optimizer_kwargs') or {})}
    trainer_kwargs = {**_thaw(unit.spec['runtime']['trainer_kwargs']), **_thaw(config['trainer_kwargs']), **_thaw(legacy.get('trainer_kwargs') or {})}
    if objective_config:
        trainer_kwargs['objective_config'] = _objective_config(objective_config)
    guide: Any = TimedGuide(RecursiveGuide()) if legacy.get('timed_guide') else RecursiveGuide()
    level_optimizer: Any = legacy.get('optimizer', config['optimizer'])
    agentic_factory = agentic_optimizer_factory(legacy, memory, reuse['tools'])
    if agentic_factory is not None and 'optimizer' not in resources:
        level_optimizer = agentic_factory
    run_id = str(unit.spec['runtime'].get('run_id') or f'recursive_opt:{unit.unit_id}')
    level_index = next((index for index, planned in enumerate(unit.levels) if planned.level_id == level.level_id))
    task_ids = _level_task_ids(legacy, families)
    capture = resources.get('capture')
    global_step = 0
    if isinstance(capture, MutableMapping):
        global_step = int(capture.setdefault('_global_step', 0))
    logger = RecursiveOptProgressLogger(memory=memory, run_id=run_id, level_id=level.level_id, level_index=level_index, task_ids=task_ids, global_step_offset=global_step, echo=True)
    memory.record_progress(run_id=run_id, level_id=level.level_id, level_index=level_index, event='level_start', level_step=0, global_step=global_step, metrics={'planned_steps': iterations, 'surface': legacy['surface'], 'objective_mode': str((objective_config or {}).get('mode', 'scalar'))}, task_ids=task_ids, budget=_thaw(guard.report()), selected_by='pareto' if (objective_config or {}).get('mode') == 'pareto' else 'objective')
    trainer_result = None
    started_at = time.monotonic()
    if should_fit:
        trainer_result = optimize(module, _dataset_for(legacy, families, iterations), guide=guide, optimizer=resources.get('optimizer', level_optimizer), trainer=resources.get('trainer', legacy.get('trainer', config['trainer'])), optimizer_kwargs=optimizer_kwargs, iterations=iterations, num_candidates=num_candidates, logger=logger, budget=RecursiveOptBudget(), **trainer_kwargs)
    wall_s = round(time.monotonic() - started_at, 6)
    selected_candidate = None
    try:
        score, data = _final_eval(module, legacy, families)
        score = _clamp(score, _clip_bounds(unit.spec['runtime']['scoring']))
        selected_candidate = _select_best_saved_candidate(memory, legacy, families, score)
    except Exception as error:
        selected_candidate = _select_best_saved_candidate(memory, legacy, families, DEFAULT_INVALID_FLOOR)
        if selected_candidate is None:
            raise
        score = _clamp(float(selected_candidate.score), _clip_bounds(unit.spec['runtime']['scoring']))
        data = {'score': score, 'feedback': 'final evaluation failed; selected best saved candidate', 'final_eval_error': _safe_error(error)}
    if selected_candidate is not None:
        _seed_from_text(module, legacy['surface'], selected_candidate.content)
        score = _clamp(float(selected_candidate.score), _clip_bounds(unit.spec['runtime']['scoring']))
        data = {**(dict(data) if isinstance(data, Mapping) else {}), 'selected_saved_candidate': selected_candidate.artifact_id, 'selected_saved_candidate_score': float(selected_candidate.score)}
    executed_steps = max(logger.executed_steps, int(getattr(trainer_result, 'n_iters', 0) or 0))
    objective_mode = str((objective_config or {}).get('mode', 'scalar'))
    selected_by = 'pareto' if objective_mode == 'pareto' else 'objective'
    progress = logger.build_summary(planned_steps=iterations, final_score=float(score), selected_by=selected_by, objective_mode=objective_mode)
    progress['executed_steps'] = executed_steps
    artifact_metrics = dict(data) if isinstance(data, Mapping) else {}
    artifact_metrics.update({'scores': dict(progress['scores']), 'progress': dict(progress)})
    record = save_priors(memory, module, legacy, score, metrics=artifact_metrics)
    progress['artifact_id'] = record.artifact_id
    memory.record_progress(run_id=run_id, level_id=level.level_id, level_index=level_index, event='level_end', level_step=max(0, executed_steps - 1) if executed_steps else None, global_step=global_step + executed_steps, artifact_id=record.artifact_id, problem_score=float(score), objective_score=float(score), metrics={'summary': progress, 'wall_s': wall_s}, task_ids=task_ids, budget=_thaw(guard.report()), selected_by=selected_by)
    artifact_text = _artifact_text(module, legacy['surface'])
    compatibility = {'surface': legacy['surface'], 'score': score, 'wall_s': wall_s, 'artifact': artifact_text, 'reused_prior': reuse['used_prior'], 'tools': reuse['tools'], 'artifact_id': record.artifact_id, 'depends_on': list(legacy.get('depends_on') or []), 'progress': progress}
    if isinstance(capture, MutableMapping):
        capture.setdefault('results', {})[level.level_id] = compatibility
        capture.setdefault('levels', {})[level.level_id] = module
        capture['memory'] = memory
        summary = capture.setdefault('progress', {'run_id': run_id, 'levels': {}})
        summary['levels'][level.level_id] = progress
        capture['_global_step'] = global_step + executed_steps
    evaluation = EvaluationResult(valid=True, status='ok', metrics={'score': float(score)}, feedback=data.get('feedback', '') if isinstance(data, Mapping) else '', trace={'legacy_data': _thaw(data)}, artifacts={'artifact_id': record.artifact_id})
    return RunResult(unit_id=f'{unit.unit_id}:{level.level_id}', plan_fingerprint='', spec_fingerprint=unit.spec['fingerprint'], engine='trace', module_ref=canonical['module']['ref'], status='success', valid=True, evaluation=evaluation, artifact=_freeze({'text': artifact_text}), lineage=(), usage=evaluation.usage, budget=guard.report(), metadata=_freeze({'level_id': level.level_id, 'legacy_compatibility': compatibility}))

def _run_gepa_engine(unit: _ExecutionUnit, level: _LevelPlan, resources: Mapping[str, Any]) -> RunResult:
    """Adapt GEPA OptimizeAnything to the canonical module/evaluator contracts."""
    spec = level.spec
    guard = resources['_budget']
    engine = _engine_entry(spec['engine']['name'])
    objective = compile_objective(spec['objective'], capabilities=engine.capabilities)
    prepared = _prepare_level(unit, level, resources, objective)
    evaluator = prepared['evaluator']
    seed_module = prepared['module']
    seed_artifact = _snapshot_level_module(spec, seed_module)
    seed_candidate = seed_artifact.get('components', seed_artifact)
    access = DatasetAccess(level.datasets)
    train = list(access.read('train', phase='fit'))
    validation = list(access.read('validation', phase='candidate_selection'))
    evaluation_info: List[Dict[str, Any]] = []

    def gepa_evaluator(candidate: Any, *, example: Any, opt_state: Any=None) -> Tuple[float, Any, Dict[str, Any]]:
        candidate_module = _build_level_module(prepared['bound_spec'], prepared['module_resources'])
        artifact = _candidate_to_artifact(seed_artifact, candidate)
        _restore_level_module(spec, candidate_module, artifact)
        evaluation = _evaluate_dataset(candidate_module, [example], prepared['fit_context'], objective, evaluator, guard, prepared['metered_roles'], prepared['records'])
        score, info = _project_for_gepa(evaluation, objective)
        evaluation_info.append(info)
        return (score, _thaw(candidate), {'evaluation': info, 'scores': info['metrics']})
    config_values = _gepa_config_values(spec['engine']['config'], unit.seed, unit.spec['budget'])
    planned_candidates = config_values.get('engine', {}).get('max_candidate_proposals')
    guard.consume('candidates', 1 if planned_candidates is None else int(planned_candidates))
    gepa_resources = {**resources, '_reflection_lm': prepared['clients'].get('optimizer'), '_budget_stopper': _GepaBudgetStopper(guard)}
    optimize_anything, config = _resolve_gepa(gepa_resources, config_values)
    gepa_result = optimize_anything(seed_candidate=seed_candidate, evaluator=gepa_evaluator, dataset=train, valset=validation, objective=spec['objective']['intent'], config=config)
    best_candidate = _gepa_best_candidate(gepa_result)
    final_module = _build_level_module(prepared['bound_spec'], prepared['module_resources'])
    _restore_level_module(spec, final_module, _candidate_to_artifact(seed_artifact, best_candidate))
    holdout = list(access.read('holdout', phase='final_evaluation'))
    final_dataset = holdout or validation or train
    evaluation = _evaluate_dataset(final_module, final_dataset, prepared['final_context'], objective, evaluator, guard, prepared['metered_roles'], prepared['records'])
    artifact = _snapshot_level_module(spec, final_module)
    return RunResult(unit_id=f'{unit.unit_id}:{level.level_id}', plan_fingerprint='', spec_fingerprint=unit.spec['fingerprint'], engine=spec['engine']['name'], module_ref=spec['module']['ref'], status='success' if evaluation.valid else 'invalid', valid=evaluation.valid, evaluation=evaluation, artifact=_freeze(artifact), lineage=prepared['lineage'], usage=_combined_runtime_usage(evaluation.usage, prepared['usage']), budget=guard.report(), metadata=_freeze({'level_id': level.level_id, 'engine_capabilities': sorted(engine.capabilities), 'gepa_version': GEPA_VERSION, 'gepa_evaluations': evaluation_info, 'evaluator_records': prepared['records'], 'objective_projection': objective['config'].mode, 'gepa_holdout_externalized': True, 'gepa_budget_mapping': {'evaluator_runs': 'engine.max_metric_calls', 'candidates': 'engine.max_candidate_proposals', 'seed': 'engine.seed', 'wall_time_s': 'stop_callbacks', 'optimizer_llm_calls': 'wrapped reflection_lm', 'eval_llm_calls': 'wrapped evaluator roles', 'total_tokens': 'wrapped role clients'}, 'selected_models': _selected_role_models(prepared['clients'])}), error=evaluation.error)

def _candidate_to_artifact(seed_artifact: Mapping[str, Any], candidate: Any) -> Dict[str, Any]:
    """Convert GEPA text/component candidates back to the registered artifact."""
    components = seed_artifact.get('components')
    if not isinstance(components, Mapping):
        if not isinstance(candidate, Mapping):
            raise TypeError('non-component GEPA candidates require a mapping artifact')
        return _thaw(candidate)
    if isinstance(candidate, str):
        if len(components) != 1:
            raise TypeError('text GEPA candidate is valid only for one-component modules')
        candidate = {next(iter(components)): candidate}
    if not isinstance(candidate, Mapping):
        raise TypeError('GEPA candidate must be text or a component mapping')
    return {'components': _thaw(candidate)}

def _project_for_gepa(evaluation: EvaluationResult, objective: Mapping[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Project canonical metrics deterministically while retaining complete info."""
    config = objective['config']
    feasible = satisfies_hard_constraints(evaluation, objective['hard_constraints'])
    if not feasible:
        score = -1000000000000.0
    elif config.mode == 'weighted':
        metrics = apply_minimize(evaluation.metrics, config.minimize)
        score = weighted_scalarize(metrics, config.weights, config.missing_value)
    elif config.mode == 'scalar':
        score = to_scalar_score(evaluation.metrics, config)
    else:
        raise ValueError("GEPA does not support objective mode 'pareto'")
    info = {'valid': evaluation.valid and feasible, 'status': evaluation.status if feasible else 'constraint_failed', 'metrics': _thaw(evaluation.metrics), 'feedback': _thaw(evaluation.feedback), 'trace': _thaw(evaluation.trace), 'usage': _thaw(evaluation.usage), 'artifacts': _thaw(evaluation.artifacts), 'error': evaluation.error}
    return (float(score), info)

def _resolve_gepa(resources: Mapping[str, Any], config_values: Mapping[str, Any]) -> Tuple[Callable[..., Any], Any]:
    """Resolve injected GEPA or import the exact pinned optional dependency."""
    injected = resources.get('gepa_optimize')
    if injected is not None:
        if not callable(injected):
            raise TypeError('gepa_optimize resource must be callable')
        return (injected, resources.get('gepa_config'))
    try:
        from importlib.metadata import version
        from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything
    except ImportError as exc:
        raise ImportError(f'GEPA engine requires optional dependency gepa=={GEPA_VERSION}; install trace-opt[gepa]') from exc
    installed = version('gepa')
    if installed != GEPA_VERSION:
        raise RuntimeError(f'GEPA version must be {GEPA_VERSION}, found {installed}')
    raw = _thaw(config_values)
    engine = EngineConfig(**raw.pop('engine', {}))
    reflection_values = raw.pop('reflection', {})
    reflection_lm = resources.get('_reflection_lm')
    if reflection_lm is not None:
        reflection_values['reflection_lm'] = reflection_lm
    reflection = ReflectionConfig(**reflection_values)
    if resources.get('_budget_stopper') is not None:
        raw['stop_callbacks'] = resources['_budget_stopper']
    config = GEPAConfig(engine=engine, reflection=reflection, **raw)
    return (optimize_anything, config)

def _gepa_config_values(config: Mapping[str, Any], seed: Optional[int], budget: Mapping[str, Any]) -> Dict[str, Any]:
    """Map common seed/evaluation/candidate limits into GEPA 0.1.4 config."""
    values = _thaw(config)
    engine = values.setdefault('engine', {})
    if not isinstance(engine, dict):
        raise TypeError('GEPA engine.config.engine must be a mapping')
    if seed is not None:
        engine.setdefault('seed', seed)
    if budget['evaluator_runs'] is not None:
        engine.setdefault('max_metric_calls', budget['evaluator_runs'])
    if budget['candidates'] is not None:
        engine.setdefault('max_candidate_proposals', budget['candidates'])
    return values

class _GepaBudgetStopper:
    """Stop GEPA between operations when the common wall-time budget expires."""

    def __init__(self, guard: _BudgetGuard) -> None:
        self.guard = guard

    def __call__(self, _state: Any) -> bool:
        try:
            self.guard.check_wall_time()
        except BudgetExceeded:
            return True
        return False

def _gepa_best_candidate(result: Any) -> Any:
    """Extract the documented best candidate from a GEPA result."""
    if isinstance(result, Mapping) and 'best_candidate' in result:
        return result['best_candidate']
    if hasattr(result, 'best_candidate'):
        return result.best_candidate
    raise TypeError('GEPA result does not expose best_candidate')

def _run_module_engine(unit: _ExecutionUnit, level: _LevelPlan, resources: Mapping[str, Any], *, fit: bool) -> RunResult:
    """Run fixed evaluation or the existing Trace optimizer over one level."""
    spec = level.spec
    guard: _BudgetGuard = resources['_budget']
    engine = _engine_entry(spec['engine']['name'])
    objective = compile_objective(spec['objective'], capabilities=engine.capabilities)
    prepared = _prepare_level(unit, level, resources, objective)
    module = prepared['module']
    evaluator = prepared['evaluator']
    access = DatasetAccess(level.datasets)
    train = list(access.read('train', phase='fit'))
    validation = list(access.read('validation', phase='candidate_selection'))
    holdout = list(access.read('holdout', phase='final_evaluation'))
    initial_artifact = _snapshot_level_module(spec, module)
    initial_validation: Optional[EvaluationResult] = None
    if fit and validation and spec['engine']['config']['validation_gate']:
        initial_validation = _evaluate_dataset(module, validation, prepared['fit_context'], objective, evaluator, guard, prepared['metered_roles'], prepared['records'])
    budget_exhausted: Optional[str] = None
    if fit:
        config = spec['engine']['config']
        try:
            guard.consume('candidates', config['iterations'] * config['num_candidates'])
        except BudgetExceeded as error:
            if unit.spec['budget']['on_exceed'] != 'return_best_valid':
                raise
            budget_exhausted = _safe_error(error)
            fit = False
    if fit:
        config = spec['engine']['config']
        optimizer_kwargs = _thaw(config['optimizer_kwargs'])
        if prepared['clients'].get('optimizer') is not None:
            optimizer_kwargs['llm'] = prepared['clients']['optimizer']
        optimizer = resources.get('optimizer', config['optimizer'])
        trainer = resources.get('trainer', config['trainer'])
        evaluated_module = _EvaluatedModule(module, evaluator, objective, prepared['fit_context'], guard, prepared['metered_roles'], prepared['records'])
        trainer_kwargs = {**_thaw(config['trainer_kwargs']), 'objective_config': objective['config']}
        if validation:
            trainer_kwargs.update({'validate_dataset': _trainer_dataset(validation), 'validate_guide': RecursiveGuide(), 'validate_exploration_candidates': bool(config['validation_gate'])})
        try:
            optimize(evaluated_module, _trainer_dataset(train), guide=RecursiveGuide(), trainer=trainer, optimizer=optimizer, optimizer_kwargs=optimizer_kwargs, iterations=config['iterations'], num_candidates=config['num_candidates'], budget=RecursiveOptBudget(), **trainer_kwargs)
        except BudgetExceeded as error:
            _restore_level_module(spec, module, initial_artifact)
            if unit.spec['budget']['on_exceed'] != 'return_best_valid':
                raise
            budget_exhausted = _safe_error(error)
        if initial_validation is not None:
            candidate_artifact = _snapshot_level_module(spec, module)
            candidate_validation = _evaluate_dataset(module, validation, prepared['fit_context'], objective, evaluator, guard, prepared['metered_roles'], prepared['records'])
            try:
                selected = select_evaluation_result([initial_validation, candidate_validation], objective['config'], objective['hard_constraints'])
            except ValueError:
                selected = initial_validation
            if selected is initial_validation:
                _restore_level_module(spec, module, initial_artifact)
            else:
                _restore_level_module(spec, module, candidate_artifact)
    dataset = holdout or validation or train
    try:
        evaluation = _evaluate_dataset(module, dataset, prepared['final_context'], objective, evaluator, guard, prepared['metered_roles'], prepared['records'])
    except BudgetExceeded as error:
        if unit.spec['budget']['on_exceed'] != 'return_best_valid' or initial_validation is None:
            raise
        _restore_level_module(spec, module, initial_artifact)
        evaluation = initial_validation
        budget_exhausted = _safe_error(error)
    artifact = _snapshot_level_module(spec, module)
    status = 'budget_exhausted' if budget_exhausted else 'success' if evaluation.valid else 'invalid'
    return RunResult(unit_id=f'{unit.unit_id}:{level.level_id}', plan_fingerprint='', spec_fingerprint=unit.spec['fingerprint'], engine=spec['engine']['name'], module_ref=spec['module']['ref'], status=status, valid=evaluation.valid, evaluation=evaluation, artifact=_freeze(artifact), lineage=prepared['lineage'], usage=_combined_runtime_usage(evaluation.usage, prepared['usage']), budget=guard.report(), metadata=_freeze({'level_id': level.level_id, 'engine_capabilities': sorted(engine.capabilities), 'module_capabilities': sorted(_module_entry(spec['module']['ref']).capabilities), 'objective_mode': objective['config'].mode, 'evaluator_records': prepared['records'], 'trace_optimize_path': fit, 'selected_models': _selected_role_models(prepared['clients']), 'budget_exhausted': budget_exhausted}), error=evaluation.error)

class _EvaluatedModule(Module):
    """Attach registered evaluator results to real Trace parameter dependencies."""

    def __init__(self, module: Module, evaluator: Callable[..., Any], objective: Mapping[str, Any], context: Mapping[str, Any], guard: _BudgetGuard, metered_roles: set[str], records: List[Dict[str, Any]]) -> None:
        self.module = module
        self.evaluator = evaluator
        self.objective = objective
        self.context = context
        self.guard = guard
        self.metered_roles = metered_roles
        self.records = records

    @bundle()
    def _attach(self, _parameters: Any, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return payload

    def forward(self, example: Any) -> Any:
        """Evaluate one example and return trainer-compatible traced objectives."""
        self.module(example)
        evaluation = _evaluate_dataset(self.module, [getattr(example, 'data', example)], self.context, self.objective, self.evaluator, self.guard, self.metered_roles)
        info = _evaluation_info(evaluation)
        self.records.append(info)
        return self._attach(list(self.module.parameters()), {'score': _objective_score(evaluation, self.objective), 'objectives': _thaw(evaluation.metrics), 'feedback': evaluation.feedback, 'trace': evaluation.trace})

    def __deepcopy__(self, memo: Dict[int, Any]) -> '_EvaluatedModule':
        """Copy trainable state while sharing guards and immutable run context."""
        copied = type(self)(copy.deepcopy(self.module, memo), self.evaluator, self.objective, self.context, self.guard, self.metered_roles, self.records)
        memo[id(self)] = copied
        return copied

def _selected_role_models(clients: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    """Return exact models selected so far by primary/fallback role clients."""
    return {role: None if client is None else str(client.selected_model) for role, client in clients.items()}

def _trainer_dataset(values: Iterable[Any]) -> Dict[str, List[Any]]:
    """Convert resolved examples into the existing Trace trainer dataset shape."""
    inputs = list(values)
    return {'inputs': inputs, 'infos': [None] * len(inputs)}

def _snapshot_level_module(level: Mapping[str, Any], module: Module) -> Dict[str, Any]:
    """Snapshot one already-normalized level without re-normalizing a whole spec."""
    entry = _module_entry(level['module']['ref'])
    artifact = entry.snapshot(module)
    entry.validate_artifact(artifact)
    _validate_no_callables_or_secrets(artifact, 'module artifact')
    return _thaw(artifact)

def _restore_level_module(level: Mapping[str, Any], module: Module, artifact: Mapping[str, Any]) -> None:
    """Restore one already-normalized level artifact exactly."""
    entry = _module_entry(level['module']['ref'])
    entry.validate_artifact(artifact)
    entry.restore(module, artifact)

def _prepare_level(unit: _ExecutionUnit, level: _LevelPlan, resources: Mapping[str, Any], objective: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve bindings, knowledge, role clients, evaluator, and phase-safe views."""
    spec = level.spec
    module_inputs = _thaw(spec['module']['inputs'])
    lineage = _apply_level_bindings(spec, resources['_upstream'], module_inputs)
    memory = resources['_memory']
    scope = {'level': level.level_id, 'kind': spec['surface']['kind'], 'family': str(spec['module']['config'].get('family', ''))}
    cards = retrieve_knowledge(unit.spec, memory, scope)
    if cards:
        codec_ref = unit.spec['knowledge']['injection_codec']
        codec = _codec_entry(codec_ref)
        encoded = []
        for card in cards:
            if not isinstance(card, codec.input_type):
                raise TypeError(f'knowledge codec {codec_ref!r} received an invalid card')
            encoded.append(codec.encode(card))
            lineage.append({'from': 'knowledge.outputs.artifacts', 'to': 'module.inputs.knowledge', 'codec': codec_ref, 'artifact_id': getattr(card, 'artifact_id', None)})
        module_inputs['knowledge'] = encoded
    bound_spec = _thaw(spec)
    bound_spec['module']['inputs'] = module_inputs
    usage: Dict[str, MutableMapping[str, float | int]] = {}
    clients = _resolve_role_clients(unit.spec, spec, resources, usage)
    module_resources = {'llm_clients': clients, 'memory': memory, **({'graph_executors': resources['graph_executors']} if 'graph_executors' in resources else {})}
    module = _build_level_module(_freeze(bound_spec), module_resources)
    evaluator = resources.get('evaluator') or _evaluator_entry(spec['objective']['evaluator_ref'])
    records: List[Dict[str, Any]] = []
    common = {'inputs': _freeze(module_inputs), 'objective': objective, 'llm_roles': _freeze(clients), 'engine': spec['engine']['name']}
    return {'module': module, 'evaluator': evaluator, 'bound_spec': _freeze(bound_spec), 'module_resources': module_resources, 'lineage': tuple(_freeze(lineage)), 'clients': clients, 'usage': usage, 'metered_roles': {role for role, client in clients.items() if client is not None}, 'fit_context': _freeze({**common, 'spec': _phase_spec(unit.spec, bound_spec, level.datasets), 'datasets': {'train': level.datasets['train'], 'validation': level.datasets['validation']}, 'phase': 'fit'}), 'final_context': _freeze({**common, 'spec': _phase_spec(unit.spec, bound_spec, level.datasets), 'datasets': {'train': level.datasets['train'], 'validation': level.datasets['validation']}, 'phase': 'final_evaluation'}), 'records': records}

def _phase_spec(global_spec: Mapping[str, Any], level: Mapping[str, Any], datasets: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a flat phase view with no holdout field or credential value."""
    return _freeze({'schema_version': SCHEMA_VERSION, 'kind': SPEC_KIND, 'runtime': {key: value for key, value in global_spec['runtime'].items() if key not in {'memory_root'}}, 'surface': _thaw(level['surface']), 'module': _thaw(level['module']), 'engine': _thaw(level['engine']), 'objective': _thaw(level['objective']), 'datasets': {'train': _thaw(datasets['train']), 'validation': _thaw(datasets['validation'])}})

def _apply_level_bindings(level: Mapping[str, Any], outputs: Mapping[str, Any], module_inputs: MutableMapping[str, Any]) -> List[Dict[str, Any]]:
    """Apply one level's typed bindings from actual prior-level outputs."""
    lineage: List[Dict[str, Any]] = []
    for binding in level['bindings']:
        if binding['ordering_only']:
            continue
        source = _resolve_dotted_path(outputs, binding['from'])
        codec = _codec_entry(binding['codec'])
        if not isinstance(source, codec.input_type):
            raise TypeError(f"codec {binding['codec']!r} requires {codec.input_description}; got {type(source).__name__}")
        encoded = codec.encode(source)
        if not isinstance(encoded, codec.output_type):
            raise TypeError(f"codec {binding['codec']!r} returned an invalid output type")
        _set_nested_value(module_inputs, binding['to'].split('.')[2:], encoded)
        lineage.append({'from': binding['from'], 'to': binding['to'], 'codec': binding['codec'], 'artifact_id': getattr(source, 'artifact_id', None) or (source.get('artifact_id') if isinstance(source, Mapping) else None)})
    return lineage

class _GuardedRoleClient:
    """Charge common call/token budgets and attribute provider usage once."""

    def __init__(self, client: Any, role: str, usage: MutableMapping[str, MutableMapping[str, float | int]], guard: _BudgetGuard, max_tokens: Optional[int], temperature: Optional[float], model: str) -> None:
        from .runmode import track_llm_usage
        self._client = track_llm_usage(client, role, usage)
        self.role = role
        self.usage = usage
        self.guard = guard
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.selected_model = model

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.max_tokens is not None:
            kwargs.setdefault('max_tokens', self.max_tokens)
        if self.temperature is not None:
            kwargs.setdefault('temperature', self.temperature)
        resource = 'optimizer_llm_calls' if self.role == 'optimizer' else 'eval_llm_calls'
        self.guard.consume(resource)
        if self.guard.limits.get('total_tokens') is not None:
            self.guard.require('total_tokens', self.max_tokens or 1)
        before = int(self.usage.get(self.role, {}).get('total_tokens', 0))
        response = self._client(*args, **kwargs)
        after = int(self.usage.get(self.role, {}).get('total_tokens', 0))
        self.guard.consume('total_tokens', max(0, after - before))
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def __deepcopy__(self, memo: Dict[int, Any]) -> '_GuardedRoleClient':
        memo[id(self)] = self
        return self

class _FallbackRoleClient:
    """Try explicitly declared provider clients in deterministic listed order."""

    def __init__(self, clients: Iterable[_GuardedRoleClient]) -> None:
        self.clients = tuple(clients)
        if not self.clients:
            raise ValueError('fallback client requires at least one provider')
        self.selected_model = self.clients[0].selected_model

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        last_error: Optional[Exception] = None
        for client in self.clients:
            try:
                response = client(*args, **kwargs)
            except Exception as error:
                last_error = error
                continue
            self.selected_model = client.selected_model
            return response
        assert last_error is not None
        raise last_error

    def __getattr__(self, name: str) -> Any:
        return getattr(self.clients[0], name)

    def __deepcopy__(self, memo: Dict[int, Any]) -> '_FallbackRoleClient':
        memo[id(self)] = self
        return self

def _resolve_role_clients(global_spec: Mapping[str, Any], level: Mapping[str, Any], resources: Mapping[str, Any], usage: MutableMapping[str, MutableMapping[str, float | int]]) -> Dict[str, Any]:
    """Construct actual role clients from exact profiles and explicit fallbacks."""
    clients: Dict[str, Any] = {}
    factory = resources.get('llm_factory')
    guard: _BudgetGuard = resources['_budget']
    for role, profile in level['llm_roles'].items():
        if profile is None or (global_spec['runtime']['offline'] and factory is None):
            clients[role] = None
            continue
        profiles = [profile] + [global_spec['llm_profiles'][name] for name in profile['fallbacks']]
        guarded = [_make_guarded_role_client(candidate, role, factory, usage, guard) for candidate in profiles]
        clients[role] = guarded[0] if len(guarded) == 1 else _FallbackRoleClient(guarded)
    return clients

def _make_guarded_role_client(profile: Mapping[str, Any], role: str, factory: Optional[Callable[..., Any]], usage: MutableMapping[str, MutableMapping[str, float | int]], guard: _BudgetGuard) -> _GuardedRoleClient:
    """Construct one exact provider attempt for a role or fallback."""
    if guard.limits.get('total_tokens') is not None and profile.get('max_tokens') is None:
        raise ValueError(f'role {role!r} requires profile.max_tokens when budget.total_tokens is finite')
    key_ref = profile.get('api_key_ref')
    if factory is None and key_ref is not None and (not os.environ.get(key_ref.split(':', 1)[1])):
        raise ValueError(f'missing credential environment reference for role {role!r}')
    if factory is not None:
        client = factory(_freeze(_thaw(profile)), role)
    else:
        from .runmode import make_live_llm
        client = make_live_llm(profile['resolved_model'], request_timeout_s=None, budget_resource=None)
    return _GuardedRoleClient(client, role, usage, guard, profile.get('max_tokens'), profile.get('temperature'), profile['resolved_model'])

def _evaluate_dataset(module: Module, dataset: Iterable[Any], context: Mapping[str, Any], objective: Mapping[str, Any], evaluator: Callable[..., Any], guard: _BudgetGuard, metered_roles: set[str], records: Optional[List[Dict[str, Any]]]=None) -> EvaluationResult:
    """Evaluate examples under the common guard and aggregate declared sources."""
    examples = list(dataset)
    batches = [[example] for example in examples] or [[]]
    results: List[EvaluationResult] = []
    for batch in batches:
        guard.consume('evaluator_runs')
        result = normalize_evaluation_result(evaluator(module, batch, context))
        _charge_reported_usage(result.usage, guard, metered_roles)
        results.append(result)
    aggregated = _aggregate_evaluations(results, objective)
    if records is not None:
        records.append(_evaluation_info(aggregated))
    return aggregated

def _charge_reported_usage(usage: Mapping[str, Any], guard: _BudgetGuard, metered_roles: set[str]) -> None:
    """Charge usage declared by evaluators only when no wrapped client did so."""
    for role, values in usage.items():
        if role in metered_roles:
            continue
        calls = int(values.get('calls', 0))
        if calls:
            guard.consume('optimizer_llm_calls' if role == 'optimizer' else 'eval_llm_calls', calls)
        tokens = int(values.get('total_tokens', 0))
        if tokens:
            guard.consume('total_tokens', tokens)

def _aggregate_evaluations(results: List[EvaluationResult], objective: Mapping[str, Any]) -> EvaluationResult:
    """Aggregate exact metric sources and retain declared feedback channels."""
    metrics: Dict[str, float] = {}
    for name, descriptor in objective['metrics'].items():
        values = [float(_evaluation_source(result, descriptor['source'])) for result in results]
        aggregation = descriptor['aggregate_examples']
        metrics[name] = {'mean': lambda: sum(values) / len(values), 'sum': lambda: sum(values), 'min': lambda: min(values), 'max': lambda: max(values)}[aggregation]()
    valid = all((result.valid for result in results))
    usage = _merge_usage((result.usage for result in results))
    channels = set(objective['feedback_channels'])
    feedback_values = [result.feedback for result in results]
    trace_values = [result.trace for result in results]
    feedback = (feedback_values[0] if len(feedback_values) == 1 else feedback_values) if 'natural_language' in channels else ''
    trace = (trace_values[0] if len(trace_values) == 1 else trace_values) if 'trace' in channels else None
    evaluation = EvaluationResult(valid=valid, status='ok' if valid else 'invalid', metrics=metrics, feedback=feedback, trace=trace, usage=_thaw(usage), artifacts=[_thaw(result.artifacts) for result in results], error=next((result.error for result in results if result.error), None))
    if satisfies_hard_constraints(evaluation, objective['hard_constraints']):
        return evaluation
    return EvaluationResult(valid=False, status='constraint_failed', metrics=metrics, feedback=feedback, trace=trace, usage=_thaw(usage), artifacts=evaluation.artifacts, error='hard constraints not satisfied')

def _evaluation_source(result: EvaluationResult, source: str) -> Any:
    """Resolve one supported objective source from a canonical evaluation."""
    root = {'evaluation': {'metrics': result.metrics}, 'usage': result.usage}
    return _resolve_dotted_path(root, source)

def _objective_score(evaluation: EvaluationResult, objective: Mapping[str, Any]) -> float:
    """Project one evaluation for trainer ranking through ObjectiveConfig."""
    if not satisfies_hard_constraints(evaluation, objective['hard_constraints']):
        return -1000000000000.0
    config = objective['config']
    if config.mode == 'scalar':
        return to_scalar_score(evaluation.metrics, config)
    metrics = apply_minimize(evaluation.metrics, config.minimize)
    return weighted_scalarize(metrics, config.weights, config.missing_value)

def _evaluation_info(evaluation: EvaluationResult) -> Dict[str, Any]:
    """Return a JSON evaluator record without losing feedback or trace."""
    return {'valid': evaluation.valid, 'status': evaluation.status, 'metrics': _thaw(evaluation.metrics), 'feedback': _thaw(evaluation.feedback), 'trace': _thaw(evaluation.trace), 'usage': _thaw(evaluation.usage), 'artifacts': _thaw(evaluation.artifacts), 'error': evaluation.error}

def _combined_runtime_usage(evaluation: Mapping[str, Any], runtime: Mapping[str, Any]) -> Mapping[str, Any]:
    """Merge evaluator and wrapped-client counters without double attribution."""
    combined: Dict[str, Dict[str, float | int]] = {}
    for role in _ROLE_KEYS:
        left = evaluation.get(role, {})
        right = runtime.get(role, {})
        names = set(left) | set(right)
        combined[role] = {name: max(left.get(name, 0), right.get(name, 0)) for name in names}
    return _freeze(combined)

def _default_module_evaluator(module: Module, dataset: Any, context: Mapping[str, Any]) -> EvaluationResult:
    """Evaluate a module whose output already follows the canonical result shape."""
    item = dataset[0] if isinstance(dataset, (list, tuple)) and dataset else context['inputs']
    return normalize_evaluation_result(module(item))

def _reasoning_evaluator(module: Module, dataset: Any, context: Mapping[str, Any]) -> EvaluationResult:
    """Score a named reasoning component against deterministic expected output."""
    item = dataset[0] if isinstance(dataset, (list, tuple)) and dataset else None
    if not isinstance(item, Mapping):
        raise TypeError('reasoning evaluator requires a mapping dataset item')
    if tuple(context['spec']['objective']['metrics']) != ('accuracy',):
        raise ValueError('reasoning evaluator supports only the accuracy metric')
    component = item.get('component')
    if not isinstance(component, str) or not component:
        raise ValueError('reasoning evaluator dataset requires a component name')
    if 'expected' not in item:
        raise ValueError('reasoning evaluator dataset requires expected output')
    output = module(item.get('input', {}))
    components = output.get('components') if isinstance(output, Mapping) else None
    if not isinstance(components, Mapping) or component not in components:
        raise ValueError(f'reasoning module output is missing component {component!r}')
    actual = components[component]
    expected = item['expected']
    score = 1.0 if actual == expected else 0.0
    return EvaluationResult(valid=True, status='ok', metrics={'accuracy': score}, feedback=f'component {component!r}: expected {expected!r}, got {actual!r}', trace={'component': component})

def _safe_error(error: Exception) -> str:
    """Return one-line, secret-redacted execution error text without traceback paths."""
    message = str(error).splitlines()[0] if str(error) else 'execution failed'
    message = re.sub('sk-[A-Za-z0-9_-]+', 'sk-<redacted>', message)
    return f'{type(error).__name__}: {message}'

def _build_component_module(spec: Mapping[str, Any], _resources: Mapping[str, Any]) -> Module:
    """Build the registered reasoning workflow from named components."""
    config = spec['module']['config']
    components = config.get('components') if isinstance(config, Mapping) else None
    if not isinstance(components, Mapping):
        raise TypeError('reasoning workflow requires module.config.components mapping')
    if not all((isinstance(name, str) and name for name in components)):
        raise ValueError('component names must be non-empty strings')
    return _ComponentModule(components, spec['module']['inputs'])

def _validate_component_config(config: Mapping[str, Any]) -> None:
    """Validate the portable named-component module configuration."""
    if not isinstance(config, Mapping):
        raise TypeError('reasoning workflow config must be a mapping')
    _reject_unknown_keys(config, {'components', 'family'}, 'module.config')
    components = config.get('components')
    if not isinstance(components, Mapping) or not components:
        raise ValueError('module.config.components must be a non-empty mapping')
    if not all((isinstance(name, str) and name for name in components)):
        raise ValueError('component names must be non-empty strings')
    if 'family' in config and (not isinstance(config['family'], str) or not config['family']):
        raise ValueError('module.config.family must be a non-empty string')

def _validate_graph_config(config: Mapping[str, Any]) -> None:
    """Validate graph adapter config without resolving an executor resource."""
    allowed = {'executor_ref', 'input_key', 'output_key', 'input_codec', 'output_codec'}
    if not isinstance(config, Mapping):
        raise TypeError('graph module config must be a mapping')
    _reject_unknown_keys(config, allowed, 'module.config')
    ref = config.get('executor_ref')
    if not isinstance(ref, str) or not _VERSIONED_REF.fullmatch(ref):
        raise ValueError('graph module requires an exact versioned executor_ref')

def _validate_legacy_config(config: Mapping[str, Any]) -> None:
    """Validate the migration-only legacy module envelope."""
    if not isinstance(config, Mapping):
        raise TypeError('legacy module config must be a mapping')
    _reject_unknown_keys(config, {'level', 'families'}, 'module.config')
    if not isinstance(config.get('level'), Mapping) or not isinstance(config.get('families'), Mapping):
        raise TypeError('legacy module config requires level and families mappings')

def _build_legacy_level_module(spec: Mapping[str, Any], resources: Mapping[str, Any]) -> Module:
    """Build the existing executable legacy level selected by migrated config."""
    config = spec['module']['config']
    overrides = resources.get('legacy_levels', {})
    configured_level = _thaw(config['level'])
    level = _thaw(overrides.get(configured_level['id'], configured_level))
    memory = resources.get('memory') or MemoryLite(root='./trace_memory')
    level_module = compile_level(level, memory, _thaw(config['families']))
    level_module._control_plane_level = level
    level_module._control_plane_families = _thaw(config['families'])
    return level_module

def _build_graph_module(spec: Mapping[str, Any], resources: Mapping[str, Any]) -> Module:
    """Build a graph module from an explicitly supplied executor resource."""
    from opto.features.graph import GraphAdapter, GraphExecutor
    config = spec['module']['config']
    allowed = {'executor_ref', 'input_key', 'output_key', 'input_codec', 'output_codec'}
    if not isinstance(config, Mapping) or set(config) - allowed:
        raise ValueError(f'graph module config keys must be a subset of {sorted(allowed)}')
    executor_ref = config.get('executor_ref')
    if not isinstance(executor_ref, str) or not _VERSIONED_REF.fullmatch(executor_ref):
        raise ValueError('graph module requires an exact versioned executor_ref')
    executors = resources.get('graph_executors')
    if not isinstance(executors, Mapping) or executor_ref not in executors:
        raise ValueError(f'graph executor resource {executor_ref!r} is unavailable')
    executor = executors[executor_ref]
    if not isinstance(executor, GraphExecutor):
        raise TypeError(f'graph executor resource {executor_ref!r} must implement GraphExecutor')
    adapter = GraphAdapter(executor, input_key=config.get('input_key', 'query'), output_key=config.get('output_key'), input_codec=config.get('input_codec', 'graph.codec.state@1'), output_codec=config.get('output_codec', 'graph.codec.output_key@1'))
    return adapter.as_module()

def _snapshot_components(module: Module) -> Dict[str, Any]:
    """Snapshot named component parameters from a component module."""
    if not isinstance(module, _ComponentModule):
        raise TypeError('component artifact requires a registered component module')
    return {'components': {name: value.data for name, value in module.components.items()}}

def _restore_components(module: Module, artifact: Mapping[str, Any]) -> None:
    """Restore named component values without changing module topology."""
    if not isinstance(module, _ComponentModule):
        raise TypeError('component artifact requires a registered component module')
    components = artifact['components']
    expected = set(module.components)
    if set(components) != expected:
        raise ValueError(f'artifact component keys {sorted(components)} do not match {sorted(expected)}')
    for name, value in components.items():
        module.components[name]._set(value)

def _validate_component_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the portable component-dict artifact contract."""
    if not isinstance(artifact, Mapping) or set(artifact) != {'components'}:
        raise ValueError("component artifact must contain only 'components'")
    components = artifact['components']
    if not isinstance(components, Mapping) or not components:
        raise ValueError('artifact.components must be a non-empty mapping')
    if not all((isinstance(name, str) and name for name in components)):
        raise ValueError('artifact component keys must be non-empty strings')

def _snapshot_graph(module: Module) -> Dict[str, Any]:
    """Snapshot a registered graph module through its explicit adapter contract."""
    from opto.features.graph import GraphModule
    if not isinstance(module, GraphModule):
        raise TypeError('graph artifact requires a registered GraphModule')
    return module.snapshot()

def _restore_graph(module: Module, artifact: Mapping[str, Any]) -> None:
    """Restore a registered graph module without replacing its executor."""
    from opto.features.graph import GraphModule
    if not isinstance(module, GraphModule):
        raise TypeError('graph artifact requires a registered GraphModule')
    module.restore(artifact)

def _validate_graph_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the portable graph artifact without importing optional LangGraph."""
    from opto.features.graph import GraphAdapter
    GraphAdapter.validate_artifact(artifact)

def _snapshot_legacy(module: Module) -> Dict[str, Any]:
    """Snapshot the executable migrated level as portable text."""
    if not hasattr(module, '_control_plane_level'):
        raise TypeError('legacy artifact requires a migrated executable level')
    level = module._control_plane_level
    return {'text': _artifact_text(module, level['surface'])}

def _restore_legacy(module: Module, artifact: Mapping[str, Any]) -> None:
    """Restore an executable migrated level's text artifact."""
    if not hasattr(module, '_control_plane_level'):
        raise TypeError('legacy artifact requires a migrated executable level')
    _seed_from_text(module, module._control_plane_level['surface'], str(artifact['text']))

def _validate_legacy_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the migrated legacy-level artifact shape."""
    if not isinstance(artifact, Mapping) or set(artifact) != {'text'}:
        raise ValueError("legacy artifact must contain only 'text'")
    if not isinstance(artifact['text'], str):
        raise TypeError('legacy artifact text must be a string')

def _legacy_level_evaluator(module: Module, _dataset: Any, _context: Mapping[str, Any]) -> EvaluationResult:
    """Evaluate a migrated level through its existing final-evaluation helper."""
    level = module._control_plane_level
    score, data = _final_eval(module, level, module._control_plane_families)
    feedback = data.get('feedback', '') if isinstance(data, Mapping) else ''
    return EvaluationResult(valid=True, status='ok', metrics={'score': score}, feedback=feedback)

def _legacy_level_dataset(split: str, config: Mapping[str, Any]) -> Any:
    """Resolve the existing legacy training dataset without hidden imports."""
    if split != 'train':
        return []
    level = _thaw(config['level'])
    iterations = int(level.get('iterations', 1))
    return _dataset_for(level, _thaw(config['families']), iterations)['inputs']

def _expand_execution_units(spec: Mapping[str, Any]) -> List[_ExecutionUnit]:
    """Expand deterministic arm/seed/matrix products from a normalized spec."""
    experiment = spec['experiment']
    seeds = list(experiment['seeds']) or [None]
    if any((seed is not None and (not isinstance(seed, int)) for seed in seeds)):
        raise TypeError('experiment.seeds must contain integers')
    arms = list(experiment['arms']) or [{'id': 'default'}]
    matrix = experiment['matrix']
    matrix_paths = sorted(matrix)
    matrix_rows = list(product(*(matrix[path] for path in matrix_paths))) if matrix_paths else [()]
    units: List[_ExecutionUnit] = []
    for arm_index, raw_arm in enumerate(arms):
        if not isinstance(raw_arm, Mapping):
            raise TypeError(f'experiment.arms[{arm_index}] must be a mapping')
        _reject_unknown_keys(raw_arm, {'id', 'engine', 'overrides'}, f'experiment.arms[{arm_index}]')
        arm_id = raw_arm.get('id', f'arm-{arm_index}')
        if not isinstance(arm_id, str) or not arm_id:
            raise ValueError(f'experiment.arms[{arm_index}].id must be non-empty')
        for seed in seeds:
            for matrix_index, values in enumerate(matrix_rows):
                effective_seed = seed if seed is not None else spec['runtime']['seed']
                unit_raw = _thaw(spec)
                unit_raw.pop('fingerprint', None)
                unit_raw['experiment'] = {'seeds': [], 'arms': [], 'matrix': {}}
                unit_raw['runtime']['seed'] = effective_seed
                _apply_arm(unit_raw, raw_arm)
                selected = dict(zip(matrix_paths, values))
                for path, value in selected.items():
                    _set_dotted_path(unit_raw, path, value)
                unit_spec = normalize_spec(unit_raw)
                seed_label = 'none' if effective_seed is None else str(effective_seed)
                unit_id = f'{arm_id}:seed-{seed_label}:matrix-{matrix_index}'
                level_plans = tuple((_compile_level_plan(level) for level in unit_spec['levels']))
                units.append(_ExecutionUnit(unit_id, arm_id, effective_seed, _freeze(selected), unit_spec, level_plans))
    return units

def _compile_level_plan(level: Mapping[str, Any]) -> _LevelPlan:
    """Resolve a canonical level's datasets and immutable identity."""
    datasets = {split: _resolve_dataset(split, value) for split, value in level['datasets'].items()}
    payload = {'level': _thaw(level), 'dataset_refs': _dataset_identities(level['datasets'])}
    fingerprint = hashlib.sha256(_canonical_json(payload).encode('utf-8')).hexdigest()
    return _LevelPlan(level_id=level['id'], depends_on=tuple(level['depends_on']), ordering_only=level['ordering_only'], spec=level, datasets=_freeze(datasets), fingerprint=fingerprint)

def _resolve_dataset(split: str, value: Any) -> Any:
    """Resolve inline data or one exact registered dataset descriptor."""
    if not isinstance(value, Mapping):
        return _thaw(value)
    resolver = _dataset_entry(value['ref'])
    resolved = resolver(str(value.get('split') or split), _freeze(_thaw(value.get('config', {}))))
    _validate_no_callables_or_secrets(resolved, f"resolved dataset {value['ref']}")
    _canonical_json(resolved)
    return _thaw(resolved)

def _dataset_identities(datasets: Mapping[str, Any]) -> Dict[str, Any]:
    """Return fingerprint-safe dataset ref or inline-content identities."""
    return {split: _thaw(value) if isinstance(value, Mapping) else {'inline_sha256': hashlib.sha256(_canonical_json(value).encode('utf-8')).hexdigest()} for split, value in datasets.items()}

def _apply_arm(spec: Dict[str, Any], arm: Mapping[str, Any]) -> None:
    """Materialize one arm's engine and dotted overrides into a unit spec."""
    engine = arm.get('engine')
    if isinstance(engine, str):
        for level in spec['levels']:
            level['engine']['name'] = engine
            level['engine']['config'] = {}
    elif isinstance(engine, Mapping):
        _reject_unknown_keys(engine, _BLOCK_KEYS['engine'], 'experiment arm engine')
        for level in spec['levels']:
            if 'name' in engine and engine['name'] != level['engine']['name'] and ('config' not in engine):
                level['engine']['config'] = {}
            level['engine'].update(_thaw(engine))
    elif engine is not None:
        raise TypeError('experiment arm engine must be a name or mapping')
    overrides = arm.get('overrides', {})
    if not isinstance(overrides, Mapping):
        raise TypeError('experiment arm overrides must be a mapping')
    for path, value in overrides.items():
        _set_dotted_path(spec, path, value)

def _set_dotted_path(spec: Dict[str, Any], path: str, value: Any) -> None:
    """Set an existing canonical dotted path, rejecting hidden new controls."""
    if not isinstance(path, str) or not path or path.startswith(('fingerprint', 'schema_version', 'kind')):
        raise ValueError(f'invalid experiment override path {path!r}')
    parts = path.split('.')
    current: Any = spec
    for part in parts[:-1]:
        if isinstance(current, list):
            matches = [item for item in current if str(item.get('id')) == part]
            if not matches and part.isdigit() and (int(part) < len(current)):
                matches = [current[int(part)]]
            if len(matches) != 1:
                raise ValueError(f'experiment override path {path!r} has unknown level {part!r}')
            current = matches[0]
            continue
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f'experiment override path {path!r} does not resolve')
        current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current:
        raise ValueError(f'experiment override path {path!r} does not name an existing field')
    current[parts[-1]] = _thaw(value)
register_module('recursive_opt.module.reasoning_workflow@1', ModuleRegistryEntry(build=_build_component_module, snapshot=_snapshot_components, restore=_restore_components, validate_artifact=_validate_component_artifact, capabilities=frozenset({'multi_component', 'json_snapshot', 'trace_module'}), validate_config=_validate_component_config))
register_module('recursive_opt.module.graph@1', ModuleRegistryEntry(build=_build_graph_module, snapshot=_snapshot_graph, restore=_restore_graph, validate_artifact=_validate_graph_artifact, capabilities=frozenset({'graph_executor', 'json_snapshot', 'trace_module', 'input_output_codecs'}), validate_config=_validate_graph_config))
register_evaluator('recursive_opt.evaluator.module_output@1', _default_module_evaluator)
register_evaluator('recursive_opt.evaluator.reasoning@1', _reasoning_evaluator)
register_evaluator('recursive_opt.evaluator.legacy_level@1', _legacy_level_evaluator)
register_dataset('recursive_opt.dataset.legacy_level@1', _legacy_level_dataset)
register_codec('recursive_opt.codec.artifact_to_prior@1', _artifact_to_prior, input_type=(ArtifactRecord, Mapping), output_type=dict, input_description='a mapping artifact with content')
register_codec('recursive_opt.codec.component_dict@1', _component_dict, input_type=Mapping, output_type=dict, input_description='a string-keyed component mapping')
register_engine('fixed', EngineRegistryEntry(run=_run_fixed_engine, capabilities=frozenset({'scalar', 'weighted', 'pareto', 'trace_module', 'rich_trace'})))
register_engine('trace', EngineRegistryEntry(run=_run_trace_engine, capabilities=frozenset({'scalar', 'weighted', 'pareto', 'trace_module', 'heterogeneous_parameters', 'rich_trace'})))
register_engine('gepa_optimize_anything', EngineRegistryEntry(run=_run_gepa_engine, capabilities=frozenset({'scalar', 'weighted', 'trace_module', 'multi_component', 'rich_trace'})))
register_module('recursive_opt.module.legacy_level@1', ModuleRegistryEntry(build=_build_legacy_level_module, snapshot=_snapshot_legacy, restore=_restore_legacy, validate_artifact=_validate_legacy_artifact, capabilities=frozenset({'legacy', 'json_snapshot', 'trace_module'}), validate_config=_validate_legacy_config))

def migrate_legacy_spec(raw_spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Migrate legacy or flat-v2 input into the canonical multilevel shape."""
    if not isinstance(raw_spec, Mapping):
        raise TypeError('spec must be a mapping')
    spec = _thaw(raw_spec)
    if 'schema_version' in spec or 'kind' in spec:
        if spec.get('schema_version') != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}, got {spec.get('schema_version')!r}")
        if spec.get('kind') != SPEC_KIND:
            raise ValueError(f"kind must be {SPEC_KIND!r}, got {spec.get('kind')!r}")
        unknown = set(spec) - _TOP_LEVEL_KEYS
        if unknown:
            raise ValueError(f'unknown spec keys: {sorted(unknown)}')
        if 'levels' in spec:
            if any((block in spec for block in _FLAT_LEVEL_BLOCKS)):
                raise ValueError('canonical levels cannot be combined with flat level blocks')
            return spec
        surface = spec.get('surface', {})
        old_levels = surface.get('levels', []) if isinstance(surface, Mapping) else []
        if old_levels:
            families = spec.get('module', {}).get('config', {}).get('families', {})
            levels = [_migrate_legacy_level(level, families) for level in old_levels]
        else:
            levels = [{'id': 'level-0', 'depends_on': [], 'ordering_only': False, **{block: spec.get(block, [] if block == 'bindings' else {}) for block in _FLAT_LEVEL_BLOCKS}}]
            if isinstance(levels[0]['surface'], Mapping):
                levels[0]['surface'].pop('levels', None)
        migrated = {key: value for key, value in spec.items() if key not in _FLAT_LEVEL_BLOCKS and key != 'fingerprint'}
        migrated['levels'] = levels
        return migrated
    unknown = set(spec) - _LEGACY_TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f'unknown legacy spec keys: {sorted(unknown)}')
    legacy_levels = spec.get('levels')
    if not isinstance(legacy_levels, list) or not legacy_levels:
        raise ValueError("legacy spec['levels'] must be a non-empty list")
    runtime = {'memory_root': spec.get('memory_root', './trace_memory'), 'reuse_priors': bool(spec.get('reuse_priors', False))}
    for key in ('tracebench', 'scoring', 'prior_promotion', 'trainer_kwargs', 'run_id'):
        if key in spec:
            runtime[key] = spec[key]
    extensions = {**spec.get('extensions', {}), 'recursive_opt.migration': {'source_schema': 'legacy'}}
    promotion = spec.get('prior_promotion')
    if promotion:
        extensions['recursive_opt.knowledge_policies'] = {'promotion_rule': promotion}
    return {'schema_version': SCHEMA_VERSION, 'kind': SPEC_KIND, 'runtime': runtime, 'budget': spec.get('budget', {}), 'levels': [_migrate_legacy_level(level, spec.get('families', {})) for level in legacy_levels], 'extensions': extensions}

def _migrate_legacy_level(level: Mapping[str, Any], families: Mapping[str, Any]) -> Dict[str, Any]:
    """Translate one executable legacy level into a canonical level block."""
    if not isinstance(level, Mapping):
        raise TypeError('legacy levels must be mappings')
    raw = _thaw(level)
    level_id = raw.get('id')
    if not isinstance(level_id, str) or not level_id:
        raise ValueError('legacy levels require non-empty ids')
    dependencies = list(raw.get('depends_on') or [])
    engine_config = {key: raw[key] for key in ('optimizer', 'trainer', 'iterations', 'num_candidates', 'optimizer_kwargs', 'trainer_kwargs') if key in raw}
    engine_config.setdefault('validation_gate', True)
    return {'id': level_id, 'depends_on': dependencies, 'ordering_only': bool(dependencies), 'surface': {'kind': str(raw.get('surface', 'custom')), 'targets': ['*']}, 'module': {'ref': 'recursive_opt.module.legacy_level@1', 'config': {'level': raw, 'families': _thaw(families)}, 'artifact': None, 'inputs': {}}, 'engine': {'name': 'trace', 'config': engine_config}, 'objective': {'evaluator_ref': 'recursive_opt.evaluator.legacy_level@1', 'metrics': {'score': {'direction': 'maximize', 'source': 'evaluation.metrics.score', 'aggregate_examples': 'mean'}}, 'selection': {'mode': 'scalar', 'score_key': 'score'}}, 'datasets': {'train': {'ref': 'recursive_opt.dataset.legacy_level@1', 'split': 'train', 'config': {'level': raw, 'families': _thaw(families)}}, 'validation': [], 'holdout': []}}

def normalize_spec(raw_spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a validated, immutable, secret-free canonical v2alpha spec."""
    migrated = migrate_legacy_spec(raw_spec)
    supplied_fingerprint = migrated.pop('fingerprint', None)
    _validate_no_callables_or_secrets(migrated)
    _validate_v2_structure(migrated)
    defaults = _canonical_defaults()
    normalized: Dict[str, Any] = {'schema_version': SCHEMA_VERSION, 'kind': SPEC_KIND}
    for block in CANONICAL_SPEC_BLOCKS[:-1]:
        value = migrated.get(block, defaults[block])
        normalized[block] = _merge_defaults(defaults[block], value, block)
    if normalized['budget']['on_exceed'] == 'return_best':
        normalized['budget']['on_exceed'] = 'return_best_valid'
    normalized['extensions'] = _thaw(migrated.get('extensions', {}))
    _normalize_llm_profiles(normalized)
    levels = migrated.get('levels')
    if not isinstance(levels, list) or not levels:
        raise ValueError('spec.levels must be a non-empty list')
    normalized['levels'] = [_normalize_level(level, normalized, index) for index, level in enumerate(levels)]
    _validate_v2_semantics(normalized)
    _validate_no_callables_or_secrets(normalized)
    fingerprint = hashlib.sha256(_canonical_json(normalized).encode('utf-8')).hexdigest()
    if supplied_fingerprint is not None and supplied_fingerprint != fingerprint:
        raise ValueError('spec fingerprint does not match normalized content')
    normalized['fingerprint'] = fingerprint
    return _freeze(normalized)

def _canonical_defaults() -> Dict[str, Any]:
    """Return mutable defaults for global canonical blocks."""
    return {'runtime': {'strict_refs': True, 'reproducible': True, 'offline': False, 'resume': False, 'memory_root': './trace_memory', 'reuse_priors': False, 'tracebench': {}, 'scoring': {}, 'prior_promotion': {}, 'trainer_kwargs': {}, 'run_id': None, 'seed': None, 'test_mode': False}, 'llm_profiles': {}, 'knowledge': {'store': 'recursive_opt.knowledge.memory_lite@1', 'retrieval': 'best', 'statuses': ['promoted'], 'scope_fields': ['family', 'level', 'kind'], 'top_k': 5, 'injection_codec': 'recursive_opt.codec.artifact_to_prior@1', 'promotion_rule': {}, 'rollback_rule': {}}, 'outputs': {'directory': None, 'format': 'json', 'save_artifacts': True}, 'budget': {'optimizer_llm_calls': None, 'eval_llm_calls': None, 'candidates': None, 'evaluator_runs': None, 'wall_time_s': None, 'total_tokens': None, 'on_exceed': 'return_best_valid'}, 'experiment': {'seeds': [], 'arms': [], 'matrix': {}}, 'levels': []}

def _normalize_level(raw_level: Mapping[str, Any], global_spec: Mapping[str, Any], index: int) -> Dict[str, Any]:
    """Materialize one fully specified canonical level."""
    if not isinstance(raw_level, Mapping):
        raise TypeError(f'levels[{index}] must be a mapping')
    _reject_unknown_keys(raw_level, _LEVEL_KEYS, f'levels[{index}]')
    level_id = raw_level.get('id', f'level-{index}')
    defaults: Dict[str, Any] = {'surface': {'kind': 'module', 'targets': ['*']}, 'module': {'ref': None, 'config': {}, 'artifact': None, 'inputs': {}}, 'engine': {'name': 'trace', 'config': {}}, 'objective': {'evaluator_ref': 'recursive_opt.evaluator.module_output@1', 'intent': 'Maximize score.', 'metrics': {'score': {'direction': 'maximize', 'source': 'evaluation.metrics.score', 'aggregate_examples': 'mean'}}, 'selection': {'mode': 'scalar', 'score_key': 'score'}, 'hard_constraints': [], 'aggregation': {'mode': 'mean'}, 'feedback_channels': ['natural_language', 'trace']}, 'llm_roles': {role: None for role in sorted(_ROLE_KEYS)}, 'datasets': {'train': [], 'validation': [], 'holdout': []}, 'bindings': [], 'outputs': _thaw(global_spec['outputs'])}
    level: Dict[str, Any] = {'id': level_id, 'depends_on': list(raw_level.get('depends_on') or []), 'ordering_only': bool(raw_level.get('ordering_only', False))}
    for block in _LEVEL_BLOCKS:
        level[block] = _merge_defaults(defaults[block], raw_level.get(block, {}), f'levels[{index}].{block}') if isinstance(defaults[block], dict) else _thaw(raw_level.get(block, defaults[block]))
    level['objective'] = _normalize_objective(level['objective'])
    level['llm_roles'] = {role: _materialize_role(value, global_spec['llm_profiles'], role) for role, value in level['llm_roles'].items()}
    level['bindings'] = [{'ordering_only': False, **_thaw(binding)} for binding in level['bindings']]
    if level['engine']['name'] == 'trace':
        trace_defaults = {'optimizer': 'OptoPrimeV2', 'trainer': 'PrioritySearch', 'iterations': 4, 'num_candidates': 4, 'optimizer_kwargs': {}, 'trainer_kwargs': {}, 'validation_gate': True}
        level['engine']['config'] = _merge_defaults(trace_defaults, level['engine']['config'], f'levels[{index}].engine.config')
    return level

def _normalize_objective(objective: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert list/directions objectives into canonical metric descriptors."""
    result = _thaw(objective)
    metrics = result.get('metrics')
    if isinstance(metrics, list):
        directions = result.pop('directions', {})
        aggregate = result.get('aggregation', {}).get('mode', 'mean')
        result['metrics'] = {name: {'direction': directions.get(name, 'maximize'), 'source': f'evaluation.metrics.{name}', 'aggregate_examples': aggregate} for name in metrics}
    else:
        if result.get('directions'):
            raise ValueError('objective.directions is supported only with the metrics-list shorthand')
        result.pop('directions', None)
        aggregate = result.get('aggregation', {}).get('mode', 'mean')
        result['metrics'] = {name: {'aggregate_examples': aggregate, **_thaw(descriptor)} for name, descriptor in metrics.items()}
    return result

def explain_spec(raw_spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a JSON-serializable explanation of defaults and execution intent."""
    normalized = normalize_spec(raw_spec)
    experiment = normalized['experiment']
    arm_count = max(1, len(experiment['arms']))
    seed_count = max(1, len(experiment['seeds']))
    matrix_count = 1
    for values in experiment['matrix'].values():
        matrix_count *= len(values)
    return {'schema_version': normalized['schema_version'], 'kind': normalized['kind'], 'fingerprint': normalized['fingerprint'], 'engines': [level['engine']['name'] for level in normalized['levels']], 'module_refs': [level['module']['ref'] for level in normalized['levels']], 'objective_modes': [level['objective']['selection']['mode'] for level in normalized['levels']], 'levels': [level['id'] for level in normalized['levels']], 'execution_units': arm_count * seed_count * matrix_count, 'portable': not normalized['runtime']['test_mode']}

def _merge_defaults(defaults: Dict[str, Any], value: Any, block: str) -> Dict[str, Any]:
    """Merge one validated control-plane block over its materialized defaults."""
    if not isinstance(value, Mapping):
        raise TypeError(f'spec[{block!r}] must be a mapping')
    return {**_thaw(defaults), **_thaw(value)}

def _thaw(value: Any) -> Any:
    """Copy mappings and sequences into mutable JSON-compatible containers."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value

def _freeze(value: Any) -> Any:
    """Recursively freeze a JSON-compatible value without breaking json.dumps."""
    if isinstance(value, dict):
        return _FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple((_freeze(item) for item in value))
    return value

def _canonical_json(value: Any) -> str:
    """Serialize canonical spec content for stable SHA-256 fingerprints."""
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(',', ':'), sort_keys=True)

def _reject_unknown_keys(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    """Reject structural typos at one control-plane path."""
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f'unknown {path} keys: {sorted(unknown)}')

def _validate_v2_structure(spec: Dict[str, Any]) -> None:
    """Validate v2alpha container types and reject unknown structural keys."""
    _reject_unknown_keys(spec, _TOP_LEVEL_KEYS, 'spec')
    if spec.get('schema_version') != SCHEMA_VERSION or spec.get('kind') != SPEC_KIND:
        raise ValueError(f'spec must declare {SCHEMA_VERSION!r} and kind {SPEC_KIND!r}')
    for block in ('runtime', 'knowledge', 'outputs', 'budget', 'experiment'):
        value = spec.get(block, {})
        if not isinstance(value, Mapping):
            raise TypeError(f'spec[{block!r}] must be a mapping')
        _reject_unknown_keys(value, _BLOCK_KEYS[block], block)
    profiles = spec.get('llm_profiles', {})
    if not isinstance(profiles, Mapping):
        raise TypeError("spec['llm_profiles'] must be a mapping")
    for name, profile in profiles.items():
        if not isinstance(name, str) or not name:
            raise ValueError('llm profile names must be non-empty strings')
        if not isinstance(profile, Mapping):
            raise TypeError(f'llm profile {name!r} must be a mapping')
        _reject_unknown_keys(profile, _PROFILE_KEYS, f'llm_profiles.{name}')
    levels = spec.get('levels')
    if not isinstance(levels, list):
        raise TypeError('spec.levels must be a list')
    for index, level in enumerate(levels):
        if not isinstance(level, Mapping):
            raise TypeError(f'levels[{index}] must be a mapping')
        _reject_unknown_keys(level, _LEVEL_KEYS, f'levels[{index}]')
        for block in _LEVEL_BLOCKS:
            value = level.get(block, [] if block == 'bindings' else {})
            if block == 'bindings':
                if not isinstance(value, list):
                    raise TypeError(f'levels[{index}].bindings must be a list')
                for binding_index, binding in enumerate(value):
                    if not isinstance(binding, Mapping):
                        raise TypeError(f'levels[{index}].bindings[{binding_index}] must be a mapping')
                    _reject_unknown_keys(binding, _BINDING_KEYS, f'levels[{index}].bindings[{binding_index}]')
                continue
            if not isinstance(value, Mapping):
                raise TypeError(f'levels[{index}].{block} must be a mapping')
            _reject_unknown_keys(value, _BLOCK_KEYS[block], f'levels[{index}].{block}')
    extensions = spec.get('extensions', {})
    if not isinstance(extensions, Mapping):
        raise TypeError("spec['extensions'] must be a mapping")
    for namespace in extensions:
        if not isinstance(namespace, str) or '.' not in namespace:
            raise ValueError("extension keys must contain a namespace, for example 'acme.audit'")

def _normalize_llm_profiles(spec: Dict[str, Any]) -> None:
    """Materialize exact provider models, secret refs, and fallbacks."""
    profiles: Dict[str, Dict[str, Any]] = {}
    for name, raw_profile in spec['llm_profiles'].items():
        profile = {'provider': None, 'model': None, 'resolved_model': None, 'api_key_ref': None, 'fallbacks': [], 'temperature': None, 'max_tokens': None, 'base_url': None, **_thaw(raw_profile)}
        if profile['provider'] == 'openrouter':
            profile['model'] = profile['model'] or 'deepseek/deepseek-v4-flash-0731'
            profile['resolved_model'] = f"openrouter/{profile['model']}"
            profile['api_key_ref'] = profile['api_key_ref'] or 'env:OPENROUTER_API_KEY'
        elif profile['model'] is not None:
            profile['resolved_model'] = profile['resolved_model'] or profile['model']
        profiles[name] = profile
    spec['llm_profiles'] = profiles

def _materialize_role(value: Any, profiles: Mapping[str, Mapping[str, Any]], role: str) -> Optional[Dict[str, Any]]:
    """Resolve one role name/override into a fully explicit profile mapping."""
    if value is None:
        return None
    if isinstance(value, str):
        profile_name = value
        overrides: Dict[str, Any] = {}
    elif isinstance(value, Mapping):
        profile_name = value.get('profile')
        overrides = {key: _thaw(item) for key, item in value.items() if key != 'profile'}
    else:
        raise TypeError(f'llm role {role!r} must be a profile name, mapping, or null')
    if not isinstance(profile_name, str) or profile_name not in profiles:
        raise ValueError(f'llm role {role!r} references unknown profile {profile_name!r}')
    profile = {**_thaw(profiles[profile_name]), **overrides, 'profile': profile_name}
    if profile['provider'] == 'openrouter':
        profile['resolved_model'] = f"openrouter/{profile['model']}"
    elif profile.get('model') is not None:
        profile['resolved_model'] = profile.get('resolved_model') or profile['model']
    return profile

def _validate_v2_semantics(spec: Dict[str, Any]) -> None:
    """Validate every canonical field as active, designed validation, or rejected."""
    for path, ref in (('knowledge.store', spec['knowledge']['store']), ('knowledge.injection_codec', spec['knowledge']['injection_codec'])):
        if not isinstance(ref, str) or not _VERSIONED_REF.fullmatch(ref):
            raise ValueError(f'{path} must be a versioned registry ref')
    knowledge = spec['knowledge']
    if knowledge['retrieval'] not in {'best', 'recent'}:
        raise ValueError("knowledge.retrieval must be 'best' or 'recent'")
    if not isinstance(knowledge['statuses'], list) or not knowledge['statuses']:
        raise ValueError('knowledge.statuses must be a non-empty list')
    if not set(knowledge['statuses']) <= {'candidate', 'promoted', 'rejected', 'rolled_back', 'superseded'}:
        raise ValueError('knowledge.statuses contains an unknown status')
    if not isinstance(knowledge['scope_fields'], list) or not all((isinstance(field, str) and field for field in knowledge['scope_fields'])):
        raise ValueError('knowledge.scope_fields must contain field names')
    if not isinstance(knowledge['top_k'], int) or knowledge['top_k'] <= 0:
        raise ValueError('knowledge.top_k must be a positive integer')
    if knowledge['promotion_rule'] or knowledge['rollback_rule']:
        raise ValueError('knowledge promotion_rule/rollback_rule are not operational; use a namespaced extension')
    outputs = spec['outputs']
    if outputs['format'] != 'json':
        raise ValueError("outputs.format currently supports only 'json'")
    if outputs['directory'] is not None and (not isinstance(outputs['directory'], str)):
        raise TypeError('outputs.directory must be a string or null')
    if not isinstance(outputs['save_artifacts'], bool):
        raise TypeError('outputs.save_artifacts must be boolean')
    budget = spec['budget']
    for name in ('optimizer_llm_calls', 'eval_llm_calls', 'candidates', 'evaluator_runs', 'total_tokens'):
        value = budget[name]
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise ValueError(f'budget.{name} must be a non-negative integer or null')
    if budget['wall_time_s'] is not None and (not isinstance(budget['wall_time_s'], (int, float)) or budget['wall_time_s'] < 0):
        raise ValueError('budget.wall_time_s must be a non-negative number or null')
    if budget['on_exceed'] not in {'fail', 'raise', 'return_best_valid'}:
        raise ValueError('budget.on_exceed must be fail, raise, or return_best_valid')
    runtime = spec['runtime']
    if runtime['strict_refs'] is not True:
        raise ValueError('runtime.strict_refs=false is unsupported')
    for name in ('reproducible', 'offline', 'resume', 'reuse_priors', 'test_mode'):
        if not isinstance(runtime[name], bool):
            raise TypeError(f'runtime.{name} must be boolean')
    if runtime['seed'] is not None and (not isinstance(runtime['seed'], int)):
        raise TypeError('runtime.seed must be an integer or null')
    levels = spec['levels']
    seen_levels: set[str] = set()
    for index, level in enumerate(levels):
        level_id = level['id']
        if not isinstance(level_id, str) or not level_id or level_id in seen_levels:
            raise ValueError('levels require unique non-empty ids')
        dependencies = level['depends_on']
        if not isinstance(dependencies, list) or any((dep not in seen_levels for dep in dependencies)):
            raise ValueError(f'level {level_id!r} dependencies must reference earlier levels')
        if dependencies and (not level['ordering_only']):
            for dependency in dependencies:
                if not any((binding.get('from', '').startswith(f'{dependency}.outputs.') for binding in level['bindings'] if not binding['ordering_only'])):
                    raise ValueError(f'decorative dependency {dependency!r} -> {level_id!r} requires a binding')
        _validate_level_semantics(level, spec['llm_profiles'], spec['outputs'], runtime['reproducible'], index)
        seen_levels.add(level_id)
    for name, profile in spec['llm_profiles'].items():
        _validate_profile(profile, f'llm profile {name!r}', spec['llm_profiles'], runtime['reproducible'], name)
    for key in ('seeds', 'arms'):
        if not isinstance(spec['experiment'][key], list):
            raise TypeError(f'experiment.{key} must be a list')
    matrix = spec['experiment']['matrix']
    if not isinstance(matrix, Mapping) or any((not isinstance(values, list) or not values for values in matrix.values())):
        raise ValueError('experiment.matrix must map paths to non-empty value lists')

def _validate_level_semantics(level: Mapping[str, Any], profiles: Mapping[str, Any], global_outputs: Mapping[str, Any], reproducible: bool, index: int) -> None:
    """Validate references and causal controls for one canonical level."""
    module_ref = level['module']['ref']
    if not isinstance(module_ref, str) or not _VERSIONED_REF.fullmatch(module_ref):
        raise ValueError(f'levels[{index}].module.ref must be a versioned registry ref')
    targets = level['surface']['targets']
    if not isinstance(targets, list) or any((not isinstance(target, str) or not target for target in targets)):
        raise ValueError(f'levels[{index}].surface.targets must contain non-empty strings')
    if len(set(targets)) != len(targets):
        raise ValueError(f'levels[{index}].surface.targets must be unique')
    engine = level['engine']
    if not isinstance(engine['name'], str) or not engine['name']:
        raise ValueError(f'levels[{index}].engine.name must be non-empty')
    if engine['name'] == 'trace':
        config = engine['config']
        _reject_unknown_keys(config, {'optimizer', 'trainer', 'iterations', 'num_candidates', 'optimizer_kwargs', 'trainer_kwargs', 'validation_gate'}, f'levels[{index}].engine.config')
        for name in ('iterations', 'num_candidates'):
            if not isinstance(config[name], int) or config[name] <= 0:
                raise ValueError(f'levels[{index}].engine.config.{name} must be positive')
        if not isinstance(config['validation_gate'], bool):
            raise TypeError(f'levels[{index}].engine.config.validation_gate must be boolean')
        for name in ('optimizer_kwargs', 'trainer_kwargs'):
            if not isinstance(config[name], Mapping):
                raise TypeError(f'levels[{index}].engine.config.{name} must be a mapping')
    elif engine['name'] == 'fixed':
        if engine['config']:
            raise ValueError(f'levels[{index}].engine.config must be empty for fixed')
    elif engine['name'] == 'gepa_optimize_anything':
        _validate_gepa_engine_config(engine['config'], index)
    _validate_objective_semantics(level['objective'], index)
    for role, value in level['llm_roles'].items():
        if role not in _ROLE_KEYS:
            raise ValueError(f'levels[{index}] has unknown LLM role {role!r}')
        profile_name = value.get('profile') if isinstance(value, Mapping) else value
        if profile_name is not None and profile_name not in profiles:
            raise ValueError(f'levels[{index}] role {role!r} references unknown profile')
        if value is not None:
            _validate_profile(value, f'levels[{index}] role {role!r}', profiles, reproducible, profile_name)
    for split, value in level['datasets'].items():
        if isinstance(value, Mapping):
            _reject_unknown_keys(value, _DATASET_REF_KEYS, f'levels[{index}].datasets.{split}')
            ref = value.get('ref')
            if not isinstance(ref, str) or not _VERSIONED_REF.fullmatch(ref):
                raise ValueError(f'levels[{index}].datasets.{split}.ref must be versioned')
            if not isinstance(value.get('config', {}), Mapping):
                raise TypeError(f'levels[{index}].datasets.{split}.config must be a mapping')
        elif not isinstance(value, (list, tuple)):
            raise TypeError(f'levels[{index}].datasets.{split} must be inline data or a dataset ref')
    outputs = level['outputs']
    if outputs['directory'] != global_outputs['directory'] or outputs['format'] != global_outputs['format']:
        raise ValueError(f'levels[{index}].outputs may override only save_artifacts')
    if not isinstance(outputs['save_artifacts'], bool):
        raise TypeError(f'levels[{index}].outputs.save_artifacts must be boolean')
    for binding_index, binding in enumerate(level['bindings']):
        if binding['ordering_only']:
            raise ValueError('binding.ordering_only is unsupported; use level.ordering_only')
        for field in ('from', 'to', 'codec'):
            if not isinstance(binding.get(field), str) or not binding[field]:
                raise ValueError(f'levels[{index}].bindings[{binding_index}].{field} is required')
        if not _VERSIONED_REF.fullmatch(binding['codec']):
            raise ValueError(f'levels[{index}].bindings[{binding_index}].codec must be versioned')
        if not binding['to'].startswith('module.inputs.'):
            raise ValueError(f'levels[{index}].bindings[{binding_index}].to must be below module.inputs')
        source_level = binding['from'].split('.', 1)[0]
        if source_level not in level['depends_on']:
            raise ValueError(f'levels[{index}].bindings[{binding_index}] must source a declared dependency')

def _validate_profile(profile: Mapping[str, Any], path: str, profiles: Mapping[str, Any], reproducible: bool, profile_name: Optional[str]) -> None:
    """Validate one global or materialized role-specific LLM profile."""
    if not profile['provider'] or not profile['model']:
        raise ValueError(f'{path} requires provider and exact model')
    if str(profile['model']).endswith('latest') and reproducible:
        raise ValueError(f'{path} cannot use a latest alias in reproducible mode')
    key_ref = profile['api_key_ref']
    if key_ref is not None and (not isinstance(key_ref, str) or not key_ref.startswith('env:')):
        raise ValueError(f'{path} api_key_ref must use env:NAME')
    fallbacks = profile['fallbacks']
    if not isinstance(fallbacks, list):
        raise TypeError(f'{path} fallbacks must be a list')
    if any((not isinstance(fallback, str) or fallback not in profiles or fallback == profile_name for fallback in fallbacks)):
        raise ValueError(f'{path} fallbacks must name other declared profiles')
    if len(set(fallbacks)) != len(fallbacks):
        raise ValueError(f'{path} fallbacks must be unique')
    if profile['temperature'] is not None and not isinstance(profile['temperature'], (int, float)):
        raise TypeError(f'{path} temperature must be numeric or null')
    max_tokens = profile['max_tokens']
    if max_tokens is not None and (not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0):
        raise ValueError(f'{path} max_tokens must be positive or null')
    if profile['base_url'] is not None:
        raise ValueError(f'{path} base_url is unsupported; configure the provider externally')

def _validate_gepa_engine_config(config: Mapping[str, Any], index: int) -> None:
    """Validate the JSON-safe GEPA 0.1.4 configuration subset we construct."""
    if not isinstance(config, Mapping):
        raise TypeError(f'levels[{index}].engine.config must be a mapping')
    _reject_unknown_keys(config, {'engine', 'reflection'}, f'levels[{index}].engine.config')
    engine = config.get('engine', {})
    reflection = config.get('reflection', {})
    if not isinstance(engine, Mapping) or not isinstance(reflection, Mapping):
        raise TypeError('GEPA engine and reflection config values must be mappings')
    _reject_unknown_keys(engine, {'run_dir', 'seed', 'display_progress_bar', 'raise_on_exception', 'use_cloudpickle', 'track_best_outputs', 'max_metric_calls', 'max_candidate_proposals', 'max_reflection_cost', 'val_evaluation_policy', 'candidate_selection_strategy', 'frontier_type', 'acceptance_criterion', 'parallel', 'max_workers', 'cache_evaluation', 'cache_evaluation_storage', 'best_example_evals_k', 'capture_stdio'}, f'levels[{index}].engine.config.engine')
    _reject_unknown_keys(reflection, {'skip_perfect_score', 'perfect_score', 'batch_sampler', 'reflection_minibatch_size', 'module_selector', 'reflection_lm', 'reflection_lm_kwargs', 'reflection_prompt_template'}, f'levels[{index}].engine.config.reflection')

def _validate_objective_semantics(objective: Mapping[str, Any], index: int) -> None:
    """Validate canonical metric descriptors and selection controls."""
    evaluator_ref = objective['evaluator_ref']
    if not isinstance(evaluator_ref, str) or not _VERSIONED_REF.fullmatch(evaluator_ref):
        raise ValueError(f'levels[{index}].objective.evaluator_ref must be versioned')
    metrics = objective['metrics']
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError(f'levels[{index}].objective.metrics must be a non-empty mapping')
    for name, descriptor in metrics.items():
        if not isinstance(name, str) or not name or (not isinstance(descriptor, Mapping)):
            raise TypeError(f'levels[{index}].objective metric descriptors must be mappings')
        _reject_unknown_keys(descriptor, _METRIC_KEYS, f'levels[{index}].objective.metrics.{name}')
        if descriptor.get('direction') not in {'maximize', 'minimize'}:
            raise ValueError(f'metric {name!r} direction must be maximize or minimize')
        if not isinstance(descriptor.get('source'), str) or not descriptor['source']:
            raise ValueError(f'metric {name!r} source must be non-empty')
        if descriptor.get('aggregate_examples') not in {'mean', 'sum', 'min', 'max'}:
            raise ValueError(f'metric {name!r} aggregate_examples is unsupported')
    selection = objective['selection']
    if not isinstance(selection, Mapping):
        raise TypeError('objective.selection must be a mapping')
    _reject_unknown_keys(selection, {'mode', 'weights', 'score_key', 'tie_break', 'pareto_metrics', 'seed', 'scalarize_dict'}, 'objective.selection')
    if selection.get('mode') not in {'scalar', 'weighted', 'pareto'}:
        raise ValueError('objective.selection.mode must be scalar, weighted, or pareto')
    weights = selection.get('weights', {})
    if not isinstance(weights, Mapping) or any((metric not in metrics or not isinstance(weight, (int, float)) or weight < 0 for metric, weight in weights.items())):
        raise ValueError('objective.selection.weights must be non-negative declared metrics')
    pareto_metrics = selection.get('pareto_metrics')
    if pareto_metrics is not None and (not isinstance(pareto_metrics, list) or not pareto_metrics or any((metric not in metrics for metric in pareto_metrics))):
        raise ValueError('objective.selection.pareto_metrics must name declared metrics')
    for key in ('hard_constraints', 'feedback_channels'):
        if not isinstance(objective[key], list):
            raise TypeError(f'objective.{key} must be a list')
    for index, constraint in enumerate(objective['hard_constraints']):
        if not isinstance(constraint, Mapping):
            raise TypeError(f'objective.hard_constraints[{index}] must be a mapping')
        _reject_unknown_keys(constraint, {'metric', 'op', 'value'}, f'objective.hard_constraints[{index}]')
        if constraint.get('metric') not in metrics:
            raise ValueError(f'hard constraint {index} must name a declared metric')
        if constraint.get('op') not in {'<', '<=', '==', '!=', '>=', '>'}:
            raise ValueError(f'hard constraint {index} has unsupported operator')
        if not isinstance(constraint.get('value'), (int, float)):
            raise TypeError(f'hard constraint {index} value must be numeric')
    if not isinstance(objective['aggregation'], Mapping):
        raise TypeError('objective.aggregation must be a mapping')
    _reject_unknown_keys(objective['aggregation'], {'mode', 'weights'}, 'objective.aggregation')
    channels = objective['feedback_channels']
    if any((channel not in {'natural_language', 'trace'} for channel in channels)):
        raise ValueError('objective.feedback_channels supports natural_language and trace')
    aggregation = objective['aggregation']
    _reject_unknown_keys(aggregation, {'mode', 'weights'}, 'objective.aggregation')
    if aggregation.get('mode') not in {'mean', 'sum', 'min', 'max'}:
        raise ValueError('objective.aggregation.mode is unsupported')
    if aggregation.get('weights'):
        raise ValueError('objective.aggregation.weights is unsupported; use objective.selection.weights')

def _validate_no_callables_or_secrets(value: Any, path: str='spec') -> None:
    """Reject callables, non-string mapping keys, secrets, and non-JSON values."""
    if callable(value):
        raise TypeError(f'{path} contains a callable; use a versioned registry ref')
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f'{path} contains a non-string key')
            if key.lower() in _SECRET_KEYS and item is not None:
                raise ValueError(f'{path}.{key} contains a secret value; use an env reference')
            _validate_no_callables_or_secrets(item, f'{path}.{key}')
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_no_callables_or_secrets(item, f'{path}[{index}]')
        return
    if value is not None and (not isinstance(value, (str, int, float, bool))):
        raise TypeError(f'{path} contains non-JSON value {type(value).__name__}')

def make_level_spec(*, id: str, surface: str, targets: Optional[List[str]]=None, fixed: Optional[Dict[str, Any]]=None, constraints: Optional[Dict[str, List[str]]]=None, **kwargs: Any) -> Dict[str, Any]:
    """Safe level-spec builder (DRY for examples/notebooks)."""
    level: Dict[str, Any] = {'id': id, 'surface': surface}
    if targets:
        level['targets'] = list(targets)
    if fixed:
        level['fixed'] = dict(fixed)
    if constraints:
        level['constraints'] = {k: list(v) for k, v in constraints.items()}
    level.update(kwargs)
    return level
_DEFAULT_POLICY_FIELDS = ('starting_artifact', 'trace_type', 'batch_design')

def validate_spec(spec: dict) -> dict:
    """Validate a RecursiveSpec dict (structure, ids, surfaces, field values)."""
    if not isinstance(spec, dict):
        raise TypeError('spec must be a dict')
    if 'schema_version' in spec or 'kind' in spec:
        return normalize_spec(spec)
    families = spec.get('families', {})
    if not isinstance(families, dict):
        raise TypeError("spec['families'] must be {name: [task_ids]}")
    levels = spec.get('levels')
    if not levels or not isinstance(levels, list):
        raise ValueError("spec['levels'] must be a non-empty list (its order is the depth)")
    if not isinstance(spec.get('budget', {}), dict):
        raise TypeError("spec['budget'] must be a dict")
    _validate_tracebench_config(spec.get('tracebench', {}))
    _validate_scoring_config(spec.get('scoring', {}))
    _validate_prior_promotion_config(spec.get('prior_promotion', {}))
    seen: set = set()
    for ls in levels:
        if not isinstance(ls, dict):
            raise TypeError('each level must be a dict')
        lid = ls.get('id')
        if not lid or lid in seen:
            raise ValueError(f"each level needs a unique 'id' (got {lid!r})")
        seen.add(lid)
        surface = ls.get('surface')
        if surface not in SURFACES:
            raise ValueError(f'level {lid}: surface must be one of {SURFACES}, got {surface!r}')
        for field, allowed in (ls.get('constraints') or {}).items():
            register_config_values(field, allowed)
        if surface in ('config', 'code', 'family_policy', 'prior'):
            fixed = ls.get('fixed') or {}
            cfg = LevelConfig(**fixed)
            fields = tuple(ls.get('targets') or ()) + tuple(fixed.keys())
            validate_level_config(cfg, fields)
        if surface in ('config', 'family_policy', 'prior'):
            _check_plumbing(ls)
        deps = ls.get('depends_on') or []
        earlier = seen - {lid}
        unknown = [d for d in deps if d not in earlier]
        if unknown:
            raise ValueError(f'level {lid}: depends_on {unknown} must reference EARLIER level ids (seen so far: {sorted(earlier)}). depends_on is enforced, not decorative.')
        tasks = ls.get('tasks')
        if tasks is not None:
            if not isinstance(tasks, list) or not tasks or (not all((str(t).strip() for t in tasks))):
                raise ValueError(f'level {lid}: tasks must be a non-empty list of task ids')
        if surface == 'config' and ls.get('family') not in families and (not ls.get('task')) and (not tasks):
            raise ValueError(f"level {lid}: config needs a known 'family', explicit 'task', or non-empty 'tasks'")
        if surface == 'code' and (not isinstance(ls.get('component'), dict)):
            raise ValueError(f"level {lid}: code surface needs a 'component' dict")
        if surface == 'capability' and (not callable(ls.get('evaluator'))):
            raise TypeError(f"level {ls.get('id')}: capability surface requires a callable 'evaluator'")
        if surface == 'custom' and (not callable(ls.get('builder'))):
            raise ValueError(f"level {lid}: custom surface needs a callable 'builder'")
        if surface in ('family_policy', 'prior'):
            if not _resolve_families(ls, families):
                raise ValueError(f'level {lid}: {surface} needs at least one family')
    return spec

def compile_level(level_spec: dict, memory: MemoryLite, families: Dict[str, List[str]], scoring: Optional[dict]=None):
    """Compile one level dict into the matching existing level object."""
    surface = level_spec['surface']
    score_config = level_spec.get('scoring', scoring)
    clip = _clip_bounds(score_config)
    floor = clip[0] if clip else DEFAULT_INVALID_FLOOR
    if surface == 'custom':
        return level_spec['builder'](level_spec, memory)
    if surface == 'config':
        task_ids = _config_task_ids(level_spec, families)
        cfg = LevelConfig(**level_spec.get('fixed') or {})
        kwargs: Dict[str, Any] = {'memory': memory}
        if level_spec.get('targets'):
            kwargs['trainable_fields'] = tuple(level_spec['targets'])
        inner_runner = _make_inner_runner(task_ids[0], score_config) if len(task_ids) == 1 else _make_task_set_inner_runner(task_ids, score_config)
        return MetaLevel(cfg=cfg, inner_runner=inner_runner, invalid_floor=floor, **kwargs)
    if surface == 'code':
        c = level_spec['component']
        comp = ComponentSpec(name=c['name'], baseline=c['baseline'], evaluate=c['evaluate'], objective=c.get('objective', ''))
        return CodeArtifactLevel(comp, memory=memory)
    if surface == 'capability':
        return CapabilityArtifact(level_spec.get('seed', ''), evaluator=level_spec['evaluator'], memory=memory)
    if surface == 'family_policy':
        fams = _resolve_families(level_spec, families)
        return FamilyPolicyLevel(fams, run_task=make_scored_task_runner(score_config), invalid_floor=floor, policy_fields=tuple(level_spec.get('targets') or _DEFAULT_POLICY_FIELDS), memory=memory)
    if surface == 'prior':
        fams = _resolve_families(level_spec, families)
        names = list(fams)
        train = {names[0]: fams[names[0]]}
        holdout = {n: fams[n] for n in names[1:]} or {names[0]: fams[names[0]]}
        return PriorInductionLevel(train, holdout, run_task=make_scored_task_runner(score_config), invalid_floor=floor, fields=tuple(level_spec.get('targets') or _DEFAULT_POLICY_FIELDS), memory=memory)
    raise ValueError(f'unknown surface {surface!r}')

def reuse_priors(memory: MemoryLite, level, level_spec: dict) -> dict:
    """Warm-start a level from (family, level) memory and load reusable tools."""
    surface = level_spec['surface']
    family = str(level_spec.get('family') or '*')
    used_prior = False
    if surface == 'config' and hasattr(level, 'warm_start_from_memory'):
        before = best_config_from(level)
        level.warm_start_from_memory(family)
        used_prior = best_config_from(level) != before or memory.family_prior(family) is not None
    else:
        previous = memory.retrieve(family, kind=surface, topk=1)['artifacts']
        prev = previous[0] if previous else None
        if prev is not None and hasattr(level, 'propose'):
            try:
                _seed_from_text(level, surface, prev.content)
                used_prior = True
            except Exception:
                used_prior = False
    tools = [a.content for a in memory.retrieve(family, kind='tool')['artifacts']]
    return {'used_prior': used_prior, 'tools': tools}

def save_priors(memory: MemoryLite, level, level_spec: dict, score: float, metrics: Optional[dict]=None):
    """Persist the learned artifact (+ declared tools) tagged by family and level."""
    surface = level_spec['surface']
    family = str(level_spec.get('family') or '*')
    rec = memory.record_artifact(level=surface, family=family, kind=surface, content=_artifact_text(level, surface), score=float(score), metrics=metrics)
    for tool in level_spec.get('tools') or []:
        memory.record_artifact(level=surface, family=family, kind='tool', content=str(tool), score=float(score))
    return rec

def run_spec(spec: dict, *, optimizer: Any=None, trainer: Optional[str]=None, budget: 'RecursiveOptBudget | dict | None'=None, seeds: Optional[Iterable[int]]=None, resources: Optional[Mapping[str, Any]]=None) -> Any:
    """Migrate, normalize, compile, and execute a recursive optimization spec."""
    if not isinstance(spec, dict):
        raise TypeError('spec must be a dict')
    legacy_input = 'schema_version' not in spec and 'kind' not in spec
    portable_input, legacy_levels = _portable_legacy_input(spec) if legacy_input else (spec, {})
    raw = migrate_legacy_spec(portable_input)
    raw.pop('fingerprint', None)
    if budget is not None:
        from .budget import budget_to_spec_dict
        raw['budget'] = _thaw(budget) if isinstance(budget, dict) else budget_to_spec_dict(budget)
    if seeds is not None:
        raw.setdefault('experiment', {})['seeds'] = list(seeds)
    runtime_resources = dict(resources or {})
    if optimizer is not None:
        runtime_resources['optimizer'] = optimizer
    if trainer is not None:
        runtime_resources['trainer'] = trainer
    if legacy_levels:
        runtime_resources['legacy_levels'] = legacy_levels
    if legacy_input and (optimizer is not None or trainer is not None or legacy_levels):
        raw.setdefault('runtime', {})['test_mode'] = True
    capture: Optional[Dict[str, Any]] = None
    if legacy_input:
        capture = {}
        runtime_resources['capture'] = capture
    results = execute_plan(compile_plan(raw), runtime_resources)
    if not legacy_input:
        return results[0] if len(results) == 1 else results
    if len(results) != 1:
        return results
    assert capture is not None
    capture.pop('_global_step', None)
    memory = capture.get('memory')
    progress = capture.get('progress')
    if isinstance(memory, MemoryLite) and isinstance(progress, Mapping):
        memory.write_run_summary(dict(progress))
    return capture

def _portable_legacy_input(spec: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract callable legacy fixtures into an explicit non-portable override."""
    local_levels: Dict[str, Any] = {}
    for level in spec.get('levels', []):
        if isinstance(level, Mapping) and _contains_callable(level):
            level_id = level.get('id')
            if not isinstance(level_id, str) or not level_id:
                raise ValueError('legacy callable levels require a non-empty id')
            local_levels[level_id] = _thaw(level)
    return (_strip_callables(spec), local_levels)

def _contains_callable(value: Any) -> bool:
    """Return whether a nested legacy value contains executable behavior."""
    if callable(value):
        return True
    if isinstance(value, Mapping):
        return any((_contains_callable(item) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return any((_contains_callable(item) for item in value))
    return False

def _strip_callables(value: Any) -> Any:
    """Copy legacy data while omitting behavior carried by callable values."""
    if isinstance(value, Mapping):
        return {key: _strip_callables(item) for key, item in value.items() if not callable(item)}
    if isinstance(value, (list, tuple)):
        return [_strip_callables(item) for item in value if not callable(item)]
    return value

def _resolve_families(level_spec: dict, families: Dict[str, List[str]]) -> Dict[str, List[str]]:
    sel = level_spec.get('families')
    fam = level_spec.get('family')
    if sel in (None, '*') and fam in (None, '*'):
        return dict(families)
    if isinstance(sel, list):
        return {k: families[k] for k in sel}
    if isinstance(fam, list):
        return {k: families[k] for k in fam}
    if isinstance(fam, str) and fam in families:
        return {fam: families[fam]}
    return dict(families)

def _config_task_ids(level_spec: dict, families: Dict[str, List[str]]) -> List[str]:
    """Return concrete task ids for a config level."""
    tasks = level_spec.get('tasks')
    if tasks is not None:
        return [str(task) for task in tasks]
    task = level_spec.get('task')
    if task:
        return [str(task)]
    return [str(families[level_spec['family']][0])]

def _level_task_ids(level_spec: dict, families: Dict[str, List[str]]) -> List[str]:
    """Return task ids associated with any level for progress records."""
    if level_spec.get('surface') == 'config' and (level_spec.get('tasks') or level_spec.get('task') or level_spec.get('family') in families):
        return _config_task_ids(level_spec, families)
    selected = _resolve_families(level_spec, families)
    out: List[str] = []
    for tasks in selected.values():
        out.extend((str(task) for task in tasks))
    return out

def _dataset_for(level_spec: dict, families: Dict[str, List[str]], iterations: int) -> dict:
    fam = level_spec.get('family')
    task = level_spec.get('task')
    tasks = level_spec.get('tasks')
    if level_spec['surface'] in ('family_policy', 'prior') or not (fam or task or tasks):
        return {'inputs': [None] * iterations, 'infos': [None] * iterations}
    if tasks:
        return TB.make_dataset([f"task_set:{level_spec.get('id', 'config')}"], repeats=iterations)
    return TB.make_dataset([fam or task], repeats=iterations)

def _artifact_text(level, surface: str) -> str:
    if surface == 'config':
        return best_config_from(level)
    if surface == 'code':
        return level.current_code()
    if surface == 'capability':
        return str(level.impl.data)
    if surface == 'family_policy':
        return str(getattr(level, '_policy_node').data)
    if surface == 'prior':
        return str(getattr(level, '_prior_node').data)
    out = level.forward(None)
    data = out.data if hasattr(out, 'data') else out
    return str(data)

def _candidate_artifact_kind(surface: str) -> Optional[str]:
    """Return the kind used for validated candidate artifacts, if any."""
    return {'config': 'config_candidate', 'family_policy': 'policy', 'prior': 'prior'}.get(surface)

def _candidate_artifact_families(level_spec: dict, families: Dict[str, List[str]]) -> List[str]:
    """Return candidate artifact family labels for one level spec."""
    surface = level_spec['surface']
    if surface == 'config':
        task_ids = _config_task_ids(level_spec, families)
        labels = []
        if level_spec.get('tasks'):
            labels.append(f"task_set:{level_spec.get('id', 'config')}")
        for value in (level_spec.get('family'), level_spec.get('task')):
            if value:
                labels.append(str(value))
        labels.extend(task_ids)
        return list(dict.fromkeys(labels))
    if surface == 'family_policy':
        return ['<multi>']
    if surface == 'prior':
        return ['<holdout>']
    return []

def _select_best_saved_candidate(memory: MemoryLite, level_spec: dict, families: Dict[str, List[str]], final_score: float):
    """Return the best validated candidate when it beats the final state."""
    kind = _candidate_artifact_kind(level_spec['surface'])
    if not kind:
        return None
    candidates = []
    for family in _candidate_artifact_families(level_spec, families):
        candidates.extend(memory.artifact_history(family=family, kind=kind))
    if not candidates:
        return None
    best = max(candidates, key=lambda artifact: float(artifact.score))
    return best if float(best.score) > float(final_score) else None

def _seed_from_text(level, surface: str, text: str) -> None:
    if surface == 'config':
        getattr(level, '_cfg_node')._data = text
    elif surface == 'capability':
        level.impl._data = text
    elif surface == 'family_policy':
        level.propose(text)
    elif surface == 'prior':
        getattr(level, '_prior_node')._data = text
    elif surface == 'code':
        getattr(level, '_impl')._data = text

def _final_eval(level, level_spec: dict, families: Dict[str, List[str]]):
    surface = level_spec['surface']
    if surface == 'config':
        label = level_spec.get('family') or level_spec.get('task')
        if label is None and level_spec.get('tasks'):
            label = f"task_set:{level_spec.get('id', 'config')}"
        if label is None:
            label = families[level_spec['family']][0]
        out = level.forward(label)
    elif surface == 'code':
        fam = level_spec.get('family')
        task = level_spec.get('task') or (families.get(fam, [None])[0] if fam else None)
        out = level.forward(task)
    else:
        out = level.forward(None)
    data = out.data if hasattr(out, 'data') else out
    score = float(data.get('score', 0.0)) if isinstance(data, dict) else 0.0
    return (score, data)

def _check_plumbing(level_spec: dict) -> None:
    """Validate targets through the adapter's CAUSAL-EFFECT contract."""
    from .effects import check_field_effects
    adapter = TB._TASK_ADAPTER
    if adapter is None:
        return
    policy = level_spec.get('effect_policy') or {}
    check_field_effects(adapter, level_spec.get('targets') or [], required_effects=policy.get('required_effects'), allow_inactive=bool(level_spec.get('allow_inactive') or level_spec.get('allow_unplumbed') or policy.get('allow_inactive')))

def score_spread(task_id: str, probes: Optional[List[dict]]=None, scoring: Optional[dict]=None) -> dict:
    """Pre-flight diagnostic: prove the config->score surface is non-flat."""
    probes = probes or [{}, {'starting_artifact': 'Answer directly.'}, {'starting_artifact': 'Plan step by step, then verify the answer before replying.'}]
    runner = make_scored_task_runner(scoring)
    rows = []
    for p in probes:
        try:
            score, _ = runner(LevelConfig(**p), task_id)
            value = float(score)
            if not math.isfinite(value):
                raise ValueError(f'non-finite score {value!r}')
            rows.append({'probe': p, 'score': value})
        except Exception as exc:
            rows.append({'probe': p, 'score': None, 'error': f'{type(exc).__name__}: {str(exc).splitlines()[0]}'})

    def is_invalid_probe(row: dict) -> bool:
        """Return whether a probe produced no usable score signal."""
        if row.get('score') is None:
            return True
        score = float(row['score'])
        return not math.isfinite(score) or score <= -999999.0
    valid_scores = [float(row['score']) for row in rows if not is_invalid_probe(row)]
    invalid_probes = sum((1 for row in rows if is_invalid_probe(row)))
    valid_spread = max(valid_scores) - min(valid_scores) if valid_scores else 0.0
    return {'task': task_id, 'rows': rows, 'spread': valid_spread, 'valid_spread': valid_spread, 'flat': valid_spread < 1e-09, 'failed_probes': invalid_probes, 'invalid_probes': invalid_probes, 'catastrophic': invalid_probes > 0}

def agentic_optimizer_factory(level_spec: dict, memory: MemoryLite, reused_tools: Optional[List[str]]=None):
    """Build an AgenticOptimizer factory wiring (declared + reused) tools."""
    agentic = level_spec.get('agentic')
    if not agentic:
        return None
    from .capabilities import AgenticOptimizer, default_optimizer_tools, select_optimizer_tools
    cfg = agentic if isinstance(agentic, dict) else {}
    family = level_spec.get('family')
    available = default_optimizer_tools(memory=memory, family=family if isinstance(family, str) and family != '*' else None)
    default_names = list(dict.fromkeys((level_spec.get('tools') or []) + list(reused_tools or [])))
    policy = cfg.get('tool_policy', level_spec.get('tool_policy'))
    if policy is None:
        tools = {n: available[n] for n in default_names if n in available} or available
    else:
        tools = select_optimizer_tools(available, policy, default_tools=default_names, max_tools=int(cfg.get('tool_budget', 3))) or ({n: available[n] for n in default_names if n in available} or available)
    configured_kwargs = {'tools': tools, 'tool_budget': int(cfg.get('tool_budget', 3))}
    if cfg.get('base_optimizer_cls') is not None:
        configured_kwargs['base_optimizer_cls'] = cfg['base_optimizer_cls']

    class ConfiguredAgenticOptimizer(AgenticOptimizer):
        """Agentic optimizer class configured from a declarative level spec."""
        keywords = configured_kwargs

        def __init__(self, parameters: list, **optimizer_kwargs: Any) -> None:
            super().__init__(parameters, **{**configured_kwargs, **optimizer_kwargs})
    return ConfiguredAgenticOptimizer

def _memory_from_spec(spec: dict) -> MemoryLite:
    """Create MemoryLite from spec-level prior-promotion controls."""
    promotion = spec.get('prior_promotion') or {}
    return MemoryLite(root=spec.get('memory_root', './trace_memory'), promotion_min_support=int(promotion.get('min_support', 3)), promote_priors=bool(promotion.get('enabled', True)), promotion_min_score=promotion.get('min_score'))

def make_scored_task_runner(scoring: Optional[dict]=None, *, raw_runner: Optional[Callable[[LevelConfig, str], Tuple[float, str]]]=None) -> Callable[[LevelConfig, str], Tuple[float, str]]:
    """Wrap a task runner with optional spec-level score normalization."""
    cfg = scoring or {}
    _validate_scoring_config(cfg)
    runner = raw_runner or TB.make_task_runner()
    mode = cfg.get('mode', 'raw')
    clip = _clip_bounds(cfg)
    report_raw = bool(cfg.get('report_raw', mode != 'raw'))
    baseline_cache: Dict[str, float] = {}
    baseline_cfg = _baseline_config(cfg)

    def run(level_cfg: LevelConfig, task_id: str) -> Tuple[float, str]:
        raw_score, feedback = runner(level_cfg, task_id)
        score = float(raw_score)
        meta: Dict[str, Any] = {'mode': mode, 'raw_score': score}
        if mode == 'relative_delta':
            key = str(task_id)
            if key not in baseline_cache:
                baseline_cache[key] = float(runner(baseline_cfg, task_id)[0])
            meta['baseline_score'] = baseline_cache[key]
            score = score - baseline_cache[key]
        if clip is not None:
            lo, hi = clip
            score = min(max(score, lo), hi)
            meta['clip'] = [lo, hi]
        meta['score'] = score
        if report_raw:
            feedback = f'{feedback} SCORE_NORMALIZATION_JSON={json.dumps(meta, sort_keys=True)}'
        return (float(score), feedback)
    return run

def _make_inner_runner(task_id: str, scoring: Optional[dict]) -> Callable[[LevelConfig, Any], Tuple[float, str]]:
    """Bind a possibly normalized task runner to one Trace-Bench task id."""
    run_task = make_scored_task_runner(scoring)

    def inner_runner(cfg: LevelConfig, _family: Any) -> Tuple[float, str]:
        return run_task(cfg, task_id)
    return inner_runner

def _make_task_set_inner_runner(task_ids: List[str], scoring: Optional[dict]) -> Callable[[LevelConfig, Any], Tuple[float, str]]:
    """Bind a normalized task runner to a fixed multi-task evaluation set."""
    if not task_ids:
        raise ValueError('task_ids must be non-empty')
    run_task = make_scored_task_runner(scoring)

    def inner_runner(cfg: LevelConfig, _family: Any) -> Tuple[float, str]:
        scores: List[float] = []
        feedbacks: List[str] = []
        for task_id in task_ids:
            score, feedback = run_task(cfg, task_id)
            scores.append(float(score))
            feedbacks.append(f'{task_id}: {feedback}')
        mean_score = sum(scores) / len(scores)
        return (mean_score, f'[task_set] mean={mean_score:.3f} over {len(task_ids)} task(s). ' + ' || '.join(feedbacks))
    return inner_runner

def _baseline_config(scoring: dict) -> LevelConfig:
    """Return the baseline config used by relative score normalization."""
    baseline = scoring.get('baseline', 'default_config')
    if baseline in (None, 'default_config'):
        return LevelConfig()
    if isinstance(baseline, dict):
        return LevelConfig(**baseline)
    raise ValueError("scoring.baseline must be 'default_config' or a LevelConfig dict")

def _clamp(value: float, clip: Optional[Tuple[float, float]]) -> float:
    """Bound a reported score to the configured clip range (no-op without one)."""
    if clip is None:
        return float(value)
    lo, hi = clip
    return float(min(hi, max(lo, value)))

def _clip_bounds(scoring: Optional[dict]) -> Optional[Tuple[float, float]]:
    """Return optional score clipping bounds from a scoring config."""
    if not scoring:
        return None
    raw = scoring.get('clip')
    if raw is None and scoring.get('mode') == 'clip':
        raw = [scoring.get('min', float('-inf')), scoring.get('max', float('inf'))]
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError('scoring.clip must be [min, max]')
    lo, hi = (float(raw[0]), float(raw[1]))
    if lo > hi:
        raise ValueError('scoring.clip minimum cannot exceed maximum')
    return (lo, hi)

def _validate_tracebench_config(config: dict) -> None:
    """Validate optional top-level Trace-Bench adapter config."""
    if not config:
        return
    if not isinstance(config, dict):
        raise TypeError("spec['tracebench'] must be a dict")
    TB.TraceBenchTaskAdapter.from_config(config)

def _validate_scoring_config(config: dict) -> None:
    """Validate optional score-normalization config."""
    if not config:
        return
    if not isinstance(config, dict):
        raise TypeError("spec['scoring'] must be a dict")
    mode = config.get('mode', 'raw')
    if mode not in {'raw', 'clip', 'relative_delta'}:
        raise ValueError('scoring.mode must be one of raw, clip, relative_delta')
    _clip_bounds(config)
    if mode == 'relative_delta':
        _baseline_config(config)

def _validate_prior_promotion_config(config: dict) -> None:
    """Validate optional prior-promotion config."""
    if not config:
        return
    if not isinstance(config, dict):
        raise TypeError("spec['prior_promotion'] must be a dict")
    min_support = int(config.get('min_support', 3))
    if min_support <= 0:
        raise ValueError('prior_promotion.min_support must be positive')
    min_score = config.get('min_score')
    if min_score is not None and (not isinstance(min_score, (int, float))):
        raise TypeError('prior_promotion.min_score must be a number')

def _configure_budget(spec: dict, override: 'RecursiveOptBudget | dict | None'=None):
    """Install the global budget from the spec dict (or an explicit override)."""
    source = override if override is not None else spec.get('budget')
    budget = make_budget(source)
    if budget is None:
        return None
    reset_budget(budget)
    return budget

def _objective_config(oc):
    if isinstance(oc, dict):
        from opto.trainer.objectives import ObjectiveConfig
        return ObjectiveConfig(mode=oc.get('mode', 'pareto'), minimize=set(oc.get('minimize', [])))
    return oc
