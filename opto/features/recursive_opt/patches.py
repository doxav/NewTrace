"""Declarative code patches: replace any Trace function, method or module constant during one run.

A patch is JSON, so it can live in a control-plane spec and be optimized as text by an upper level::

    {"target": "opto.trainer.algorithms.variation_search:VariationSearch._next_mode",
     "source": "def _next_mode(self):\\n    return 'diverge'"}
    {"target": "opto.trainer.algorithms.variation_search:VARIATION_INSTRUCTIONS", "value": {...}}

No core-library change is needed: targets are resolved by import path and patched with ``setattr`` for the duration of
:func:`applied`, then restored. Rules:

* targets live under ``opto.`` but never under ``opto.features.recursive_opt``: the runner and the scorer
  (evaluators, budget, selection) cannot be patched, so a patch can make search worse but cannot fake a score;
* a function patch is exactly one ``def`` named like the target, with the original parameter names; a short static
  denylist rejects process, file and import escapes (this is a guard against accidents, **not a security sandbox**);
* a failing patched function falls back to the original and is counted in :data:`FAILURES`;
* one process holds one patch set at a time. Different versions run in parallel in **separate worker processes**
  (see :func:`run_isolated`); a second, different set requested concurrently in the same process raises.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
import importlib
import inspect
import json
import multiprocessing
import textwrap
import threading
from typing import Any, Callable, Dict, Iterator, List, Mapping, Sequence, Tuple

FORBIDDEN_PREFIX = 'opto.features.recursive_opt'
DENY_NAMES = {'open', 'exec', 'eval', 'compile', '__import__', 'globals', 'breakpoint', 'input'}
DENY_MODULES = {'os', 'sys', 'subprocess', 'socket', 'shutil', 'importlib', 'ctypes', 'pathlib', 'multiprocessing', 'signal'}
FAILURES: Dict[str, int] = {}
_LOCK = threading.Lock()
_ACTIVE: Dict[str, Any] = {'key': None, 'depth': 0}


def _resolve(target: str) -> Tuple[Any, str, Any]:
    """Return (owner, attribute, current static value) for ``module:attr.path``."""
    if not isinstance(target, str) or target.count(':') != 1:
        raise ValueError(f'patch target {target!r} must be "module:attribute.path"')
    module_name, path = target.split(':')
    if not module_name.startswith('opto.') or module_name.startswith(FORBIDDEN_PREFIX):
        raise ValueError(f'patch target {target!r} must be under opto. and outside {FORBIDDEN_PREFIX}')
    owner: Any = importlib.import_module(module_name)
    parts = path.split('.')
    for part in parts[:-1]:
        owner = getattr(owner, part)
    if not hasattr(owner, parts[-1]):
        raise ValueError(f'patch target {target!r} does not exist')
    return owner, parts[-1], inspect.getattr_static(owner, parts[-1])


def _function(static: Any) -> Any:
    return static.__func__ if isinstance(static, (staticmethod, classmethod)) else static


def default_source(target: str) -> str:
    """Current source of a function target: the natural starting value of a code slot."""
    return textwrap.dedent(inspect.getsource(_function(_resolve(target)[2])))


def _compile(target: str, source: str, original: Callable) -> Callable:
    name = target.rsplit('.', 1)[-1].rsplit(':', 1)[-1]
    tree = ast.parse(textwrap.dedent(source))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != name:
        raise ValueError(f'patch {target!r}: source must be exactly one function named {name}')
    wanted = [p.name for p in inspect.signature(original).parameters.values()]
    args = tree.body[0].args
    got = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs] + [a.arg for a in (args.vararg, args.kwarg) if a]
    if got != wanted:
        raise ValueError(f'patch {target!r}: parameters {got} must equal {wanted}')
    for node in ast.walk(tree):
        modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
        if any(m.split('.')[0] in DENY_MODULES for m in modules):
            raise ValueError(f'patch {target!r}: import of {modules} is not allowed')
        if isinstance(node, ast.Name) and node.id in DENY_NAMES:
            raise ValueError(f'patch {target!r}: name {node.id!r} is not allowed')
        if isinstance(node, ast.Attribute) and node.attr.startswith('__') and node.attr != '__name__':
            raise ValueError(f'patch {target!r}: dunder attribute {node.attr!r} is not allowed')
    namespace = dict(vars(inspect.getmodule(original)))  # the original's globals, copied
    exec(compile(tree, f'<patch {target}>', 'exec'), namespace)  # noqa: S102 - validated patch source
    return namespace[name]


def _guarded(target: str, patched: Callable, original: Callable) -> Callable:
    @functools.wraps(original)
    def call(*args, **kwargs):
        try:
            return patched(*args, **kwargs)
        except Exception:  # noqa: BLE001 - a bad patch costs one fallback, never the run
            FAILURES[target] = FAILURES.get(target, 0) + 1
            return original(*args, **kwargs)
    return call


def validate(patches: Sequence[Mapping[str, Any]]) -> None:
    """Fail fast (at spec validation) on malformed, forbidden or non-compiling patches."""
    if not isinstance(patches, (list, tuple)):
        raise TypeError('patches must be a list')
    seen = set()
    for patch in patches:
        if not isinstance(patch, Mapping) or set(patch) not in ({'target', 'source'}, {'target', 'value'}):
            raise ValueError('each patch needs "target" and exactly one of "source" or "value"')
        if patch['target'] in seen:
            raise ValueError(f'duplicate patch target {patch["target"]!r}')
        seen.add(patch['target'])
        _, _, static = _resolve(patch['target'])
        if 'source' in patch:
            if patch['source'] is None:
                continue
            if not callable(_function(static)):
                raise ValueError(f'patch {patch["target"]!r}: "source" requires a function target')
            _compile(patch['target'], patch['source'], _function(static))
        else:
            json.dumps(patch['value'])


def fingerprint(patches: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(list(patches), sort_keys=True, default=str).encode()).hexdigest()[:16]


@contextlib.contextmanager
def applied(patches: Sequence[Mapping[str, Any]]) -> Iterator[None]:
    """Apply patches for the duration of the block, restoring originals afterwards (re-entrant for the same set)."""
    patches = [p for p in (patches or []) if not ('source' in p and p['source'] is None)]
    if not patches:
        yield
        return
    validate(patches)
    key = fingerprint(patches)
    with _LOCK:
        if _ACTIVE['key'] not in (None, key):
            raise RuntimeError('another patch set is active in this process; run different versions in separate '
                               'worker processes (patches.run_isolated)')
        first = _ACTIVE['key'] is None
        _ACTIVE.update(key=key, depth=_ACTIVE['depth'] + 1)
    saved: List[Tuple[Any, str, Any]] = []
    try:
        if first:
            for patch in patches:
                owner, attr, static = _resolve(patch['target'])
                if 'value' in patch:
                    new: Any = json.loads(json.dumps(patch['value']))
                else:
                    original = _function(static)
                    new = _guarded(patch['target'], _compile(patch['target'], patch['source'], original), original)
                    if isinstance(static, (staticmethod, classmethod)):
                        new = type(static)(new)
                saved.append((owner, attr, static))
                setattr(owner, attr, new)
        yield
    finally:
        for owner, attr, static in reversed(saved):
            setattr(owner, attr, static)
        with _LOCK:
            _ACTIVE['depth'] -= 1
            if _ACTIVE['depth'] == 0:
                _ACTIVE['key'] = None


def _child(connection, function: Callable, args: tuple, kwargs: dict) -> None:
    try:
        connection.send((True, function(*args, **kwargs)))
    except BaseException as error:  # noqa: BLE001 - surfaced to the parent
        connection.send((False, f'{type(error).__name__}: {error}'))
    connection.close()


def run_isolated(calls: Sequence[Tuple[Callable, tuple, dict]], workers: int = 1) -> List[Any]:
    """Run ``function(*args, **kwargs)`` calls, each in its own fresh forked process, at most ``workers`` at a time.

    Forking keeps registries (evaluators, datasets, modules) registered by the parent; each process can apply its own
    patch set without affecting the parent or its siblings, and may fork again (nested recursion). Results must be
    picklable; an exception in a call is re-raised in the parent as RuntimeError. Without ``fork`` the calls run
    sequentially in-process (one patch set at a time).
    """
    if 'fork' not in multiprocessing.get_all_start_methods():
        return [function(*args, **kwargs) for function, args, kwargs in calls]
    context = multiprocessing.get_context('fork')
    results: List[Any] = [None] * len(calls)
    pending = list(enumerate(calls))
    running: List[Tuple[int, Any, Any]] = []
    while pending or running:
        while pending and len(running) < max(1, workers):
            index, (function, args, kwargs) = pending.pop(0)
            parent_end, child_end = context.Pipe(duplex=False)
            process = context.Process(target=_child, args=(child_end, function, args, kwargs))
            process.start()
            child_end.close()
            running.append((index, process, parent_end))
        index, process, connection = running.pop(0)
        try:
            ok, value = connection.recv()
        except EOFError:
            ok, value = False, f'worker exited with code {process.exitcode}'
        process.join()
        if not ok:
            raise RuntimeError(f'isolated call failed: {value}')
        results[index] = value
    return results
