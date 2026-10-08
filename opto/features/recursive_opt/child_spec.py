"""Recursion as a declared level: ``recursive_opt.module.child_spec@1`` + ``recursive_opt.evaluator.child_spec@1``.

An upper level's trainable parameters are **slots**: values at dotted paths of a complete child spec (any v2 spec, e.g.
an O0 run, or itself an O1 spec for O2). Evaluating a candidate writes the slot values and the episode's overrides into
the child, validates it with the normal strict compiler, runs it in its own forked worker process (so concurrent
candidates may carry different code patches) and returns the child's final score and feedback::

    "module": {"ref": "recursive_opt.module.child_spec@1",
               "config": {"child": {...v2 spec...},
                          "slots": {"next_mode": "levels.O0.engine.config.patches.0.source"},
                          "example_paths": ["levels.O0.datasets"],
                          "score": "evaluation.metrics.score"}},
    "objective": {"evaluator_ref": "recursive_opt.evaluator.child_spec@1"},
    "datasets": {"train": [{"levels.O0.datasets": {...episode...}}, ...], ...}

Child LLM calls are reported as the parent's evaluation usage (role ``forward``), so equal-budget accounting includes
the cost of recursion. Identical child specs are run once per process (cache by fingerprint).
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any, Dict, Mapping

from opto.trainer.objectives import EvaluationResult

from . import patches as P
from . import spec as S

CHILD_RESOURCES: Dict[str, Any] = {}  # test-only runtime resources for children (e.g. a scripted llm_factory)
_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()
_CONFIG_KEYS = {'child', 'slots', 'example_paths', 'score', 'workers'}


def _get(value: Any, path: str) -> Any:
    for part in path.split('.'):
        if isinstance(value, list):
            matches = [item for item in value if isinstance(item, Mapping) and str(item.get('id')) == part]
            value = matches[0] if len(matches) == 1 else value[int(part)]
        else:
            value = value[part]
    return value


def _set(value: Any, path: str, new: Any) -> None:
    """Write an existing field (mapping key or list index) of a raw spec; unknown paths fail like arm overrides."""
    head, _, last = path.rpartition('.')
    owner = _get(value, head) if head else value
    if isinstance(owner, list) and last.isdigit() and int(last) < len(owner):
        owner[int(last)] = S._thaw(new)
    elif isinstance(owner, dict) and last in owner:
        owner[last] = S._thaw(new)
    else:
        raise ValueError(f'path {path!r} does not name an existing field of the child spec')


def _validate_config(config: Mapping[str, Any]) -> None:
    if not isinstance(config, Mapping):
        raise TypeError('child_spec config must be a mapping')
    S._reject_unknown_keys(config, _CONFIG_KEYS, 'module.config')
    if not isinstance(config.get('child'), Mapping) or not isinstance(config.get('slots'), Mapping) or not config['slots']:
        raise ValueError('child_spec requires a child spec mapping and a non-empty slots mapping')
    child = S._thaw(config['child'])
    for name, path in config['slots'].items():
        try:
            _get(child, path)  # the slot must name an existing field of the child template
        except (KeyError, IndexError, ValueError, TypeError) as error:
            raise ValueError(f'slot {name!r}: path {path!r} does not name a field of the child spec') from error
    S.normalize_spec(S._thaw(config['child']))  # the template itself must be a valid spec


def _build(spec: Mapping[str, Any], _resources: Mapping[str, Any]):
    config = spec['module']['config']
    child = S._thaw(config['child'])
    return S._ComponentModule({name: _get(child, path) for name, path in config['slots'].items()}, spec['module']['inputs'])


def _usage(result: Mapping[str, Any]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for values in (result.get('usage') or {}).values():
        for key, name in (('calls', 'calls'), ('total_tokens', 'total_tokens'), ('cost_usd', 'cost'), ('cost', 'cost')):
            totals[name] = totals.get(name, 0) + float(values.get(key, 0) or 0)
    return totals


def _run_child(child: Dict[str, Any], score_path: str) -> Dict[str, Any]:
    """Executed inside a worker process: compile, run, summarize (JSON only)."""
    try:
        results = S.execute_plan(S.compile_plan(child), dict(CHILD_RESOURCES))
    except Exception as error:  # noqa: BLE001 - an invalid child is an invalid candidate, not a crash
        return {'valid': False, 'error': S._safe_error(error)}
    scores, feedback, usage, fallbacks = [], [], {}, {}
    for result in results:
        data = result.to_dict()
        for level in data.get('level_results') or [data]:
            for name, count in ((level.get('metadata') or {}).get('patches') or {}).get('fallbacks', {}).items():
                fallbacks[name] = fallbacks.get(name, 0) + count
        if not data['valid']:
            return {'valid': False, 'error': data.get('error') or data['status'], 'patch_fallbacks': fallbacks}
        scores.append(float(_get(data, score_path)))
        feedback.append(str(data['evaluation'].get('feedback') or '')[:1500])
        for key, value in _usage(data).items():
            usage[key] = usage.get(key, 0) + value
    return {'valid': True, 'score': sum(scores) / len(scores), 'scores': scores, 'feedback': feedback, 'usage': usage,
            'patch_fallbacks': fallbacks}


def _evaluate(output: Any, example: Any, context: Mapping[str, Any]) -> EvaluationResult:
    config = context['spec']['module']['config']
    data = getattr(output, 'data', output)
    child = S._thaw(config['child'])
    for name, path in config['slots'].items():
        _set(child, path, data['components'][name])
    example = getattr(example, 'data', example) or {}
    allowed = tuple(config.get('example_paths') or ())
    for path, value in dict(example).items():
        if not any(path == p or path.startswith(p + '.') for p in allowed):
            raise ValueError(f'episode override {path!r} is outside example_paths {allowed}')
        _set(child, path, value)
    key = hashlib.sha256(json.dumps(child, sort_keys=True, default=str).encode()).hexdigest()
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
    if cached is None:
        cached = P.run_isolated([(_run_child, (child, config.get('score', 'evaluation.metrics.score')), {})])[0]
        with _CACHE_LOCK:
            _CACHE[key] = cached
    if not cached['valid']:
        return EvaluationResult(valid=False, status='invalid', feedback=f'child run invalid: {cached["error"]}',
                                error=cached['error'], artifacts={'child_key': key, 'patch_fallbacks': cached.get('patch_fallbacks', {})})
    usage = cached['usage']
    fallback_note = f'; patched-code fallbacks {cached["patch_fallbacks"]}' if cached['patch_fallbacks'] else ''
    feedback = f'child score {cached["score"]:.6g} (per unit {[round(s, 6) for s in cached["scores"]]}){fallback_note}\n' + '\n'.join(cached['feedback'])
    return EvaluationResult(valid=True, status='ok', metrics={'score': cached['score'], 'child_calls': usage.get('calls', 0), 'child_cost': usage.get('cost', 0.0)},
                            feedback=feedback, usage={'forward': {'calls': int(usage.get('calls', 0)), 'total_tokens': int(usage.get('total_tokens', 0))}},
                            artifacts={'child_key': key, 'patch_fallbacks': cached['patch_fallbacks']})


S.register_module('recursive_opt.module.child_spec@1', S.ModuleRegistryEntry(
    build=_build, snapshot=S._snapshot_components, restore=S._restore_components, validate_artifact=S._validate_component_artifact,
    capabilities=frozenset({'recursive', 'multi_component', 'json_snapshot', 'trace_module'}), validate_config=_validate_config))
S.register_evaluator('recursive_opt.evaluator.child_spec@1', _evaluate)
