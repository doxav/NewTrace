"""Projections for co-evolution candidates (the Trace projection idea applied to program sources).

A projection maps a generated source to the source that is evaluated and deployed:
``projection(source) -> (projected_source, note)``, or raises ``ProjectionError`` to reject it
before any evaluation. The candidate keeps its editable source; the deployable program is the
projected one. Projections are registered by reference so control-plane specs can declare them.
"""

from __future__ import annotations

import ast
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

Projection = Callable[[str], Tuple[str, str]]


class ProjectionError(ValueError):
    """The candidate cannot be projected onto the feasible set (rejected before evaluation)."""


class CompileCheck:
    """Reject sources that do not parse or do not define the required top-level names."""

    def __init__(self, required: Sequence[str] = ()) -> None:
        self.required = tuple(required)

    def __call__(self, source: str) -> Tuple[str, str]:
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            raise ProjectionError(f'SyntaxError: {error.msg} (line {error.lineno})') from None
        defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        defined |= {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
        missing = [name for name in self.required if name not in defined]
        if missing:
            raise ProjectionError(f'missing required definition(s): {", ".join(missing)}')
        return source, ''


class FallbackWrapper:
    """Per-call projection onto a feasible baseline.

    The deployed ``entry`` calls the candidate; if it raises or ``check(result, *args, **kwargs)``
    is false, it returns the baseline ``fallback(*args, **kwargs)`` instead and records the cause
    in ``_PROJECTION_EVENTS``. ``fallback_source`` and ``check_source`` each define one function
    (any names); they are renamed internally.
    """

    def __init__(self, entry: str, fallback_source: str, check_source: str) -> None:
        self.entry = entry
        self.fallback_source = self._rename(fallback_source, f'_fallback_{entry}')
        self.check_source = self._rename(check_source, f'_check_{entry}')

    @staticmethod
    def _rename(source: str, name: str) -> str:
        tree = ast.parse(source)
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if len(functions) != 1:
            raise ValueError('fallback/check source must define exactly one top-level function')
        functions[-1].name = name
        return ast.unparse(tree)

    def __call__(self, source: str) -> Tuple[str, str]:
        entry = self.entry
        wrapped = (f'{source.rstrip()}\n\n\n# --- projection: per-call fallback onto a feasible baseline ---\n'
                   f'_PROJECTION_EVENTS = []\n_candidate_{entry} = {entry}\n\n\n{self.fallback_source}\n\n\n{self.check_source}\n\n\n'
                   f'def {entry}(*args, **kwargs):\n'
                   f'    try:\n'
                   f'        result = _candidate_{entry}(*args, **kwargs)\n'
                   f'    except Exception as error:\n'
                   f'        _PROJECTION_EVENTS.append(type(error).__name__)\n'
                   f'        return _fallback_{entry}(*args, **kwargs)\n'
                   f'    try:\n'
                   f'        valid = _check_{entry}(result, *args, **kwargs)\n'
                   f'    except Exception:\n'
                   f'        valid = False\n'
                   f'    if valid:\n'
                   f'        return result\n'
                   f"    _PROJECTION_EVENTS.append('invalid')\n"
                   f'    return _fallback_{entry}(*args, **kwargs)\n')
        return wrapped, f'{entry} wrapped with a per-call fallback onto the baseline'


_REGISTRY: Dict[str, Callable[[Mapping[str, Any]], Projection]] = {
    'recursive_opt.projection.compile_check@1': lambda config: CompileCheck(config.get('required', ())),
    'recursive_opt.projection.fallback_wrapper@1': lambda config: FallbackWrapper(config['entry'], config['fallback_source'], config['check_source']),
}


def register_projection(ref: str, factory: Callable[[Mapping[str, Any]], Projection]) -> None:
    """Register a projection factory ``factory(config) -> projection`` under a versioned ref."""
    if '@' not in ref:
        raise ValueError('projection refs must be versioned, e.g. "my.projection@1"')
    _REGISTRY[ref] = factory


def make_projection(spec: Mapping[str, Any]) -> Projection:
    """Build a projection from ``{'ref': ..., 'config': {...}}``."""
    ref = spec.get('ref')
    if ref not in _REGISTRY:
        raise ValueError(f'unknown projection ref {ref!r}; register it with register_projection()')
    return _REGISTRY[ref](dict(spec.get('config') or {}))
