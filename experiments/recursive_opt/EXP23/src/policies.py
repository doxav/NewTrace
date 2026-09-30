"""Selection-policy surfaces: bounded knobs (JSON) or a small select_parent function (code).

Both are strings so the same Trace node, optimizer and mock proposer handle them.
The policy only chooses the parent, which is the only lever the simulator models;
EvoX's full EvolvedProgramDatabase surface also chooses context programs (live only).
"""

import json
import math
import random
from collections.abc import Callable

KNOBS = {
    'temperature': (0.02, 5.0),   # softmax temperature over rank_pct; low = greedy
    'epsilon': (0.0, 1.0),        # probability of a uniform random parent
    'reuse_penalty': (0.0, 3.0),  # logit penalty per prior reuse of the parent
    'elite_k': (1, 64),           # restrict softmax to the top-k members
}
STOCK = {'temperature': 5.0, 'epsilon': 1.0, 'reuse_penalty': 0.0, 'elite_k': 64}  # EvoX initial strategy: uniform parent
REFERENCE = {
    'uniform': STOCK,
    'greedy': {'temperature': 0.02, 'epsilon': 0.0, 'reuse_penalty': 0.0, 'elite_k': 1},
    'soft_0.3': {'temperature': 0.3, 'epsilon': 0.0, 'reuse_penalty': 0.0, 'elite_k': 64},
    'soft_0.1': {'temperature': 0.1, 'epsilon': 0.1, 'reuse_penalty': 0.0, 'elite_k': 64},
    'top5_reuse': {'temperature': 0.2, 'epsilon': 0.0, 'reuse_penalty': 1.0, 'elite_k': 5},
    'greedy_eps0.3': {'temperature': 0.02, 'epsilon': 0.3, 'reuse_penalty': 0.0, 'elite_k': 1},
}
CODE_STOCK = '''def select_parent(members, rng):
    """Return the index of the parent to mutate.

    members: list of dicts with keys score, rank (0 = best), rank_pct (1 = best),
    uses (times already used as parent), age (iterations since creation).
    rng: random.Random; use it for every random choice.
    """
    return rng.randrange(len(members))
'''
SAFE_BUILTINS = {name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name) for name in ('abs', 'min', 'max', 'sum', 'len', 'range', 'sorted', 'enumerate', 'zip', 'float', 'int', 'round', 'list', 'dict', 'any', 'all', 'isinstance', 'ValueError')}


class PolicyInvalid(ValueError):
    """A proposed policy failed the validator (analogue of EvoX's strategy validator)."""


def dump_knobs(knobs: dict) -> str:
    """Canonical text form of a knob policy."""
    return json.dumps({k: knobs[k] for k in KNOBS}, sort_keys=True)


def parse_knobs(text: str) -> dict:
    """Parse and bounds-check a knob policy; out-of-range values are rejected, not clipped."""
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        raise PolicyInvalid(f'knob policy is not JSON: {error}') from None
    if not isinstance(value, dict) or set(value) != set(KNOBS):
        raise PolicyInvalid(f'knob policy must have exactly {sorted(KNOBS)}')
    for key, (low, high) in KNOBS.items():
        if not isinstance(value[key], (int, float)) or isinstance(value[key], bool) or not low <= value[key] <= high:
            raise PolicyInvalid(f'{key} must be a number in [{low}, {high}]')
    value['elite_k'] = int(value['elite_k'])
    return value


def knob_selector(knobs: dict) -> Callable[[list[dict], random.Random], int]:
    """Compile knobs into a parent selector."""
    def select(members: list[dict], rng: random.Random) -> int:
        if rng.random() < knobs['epsilon']:
            return rng.randrange(len(members))
        elite = sorted(range(len(members)), key=lambda i: members[i]['rank'])[:knobs['elite_k']]
        logits = [members[i]['rank_pct'] / knobs['temperature'] - knobs['reuse_penalty'] * members[i]['uses'] for i in elite]
        top = max(logits)
        weights = [math.exp(v - top) for v in logits]
        return rng.choices(elite, weights=weights)[0]
    return select


def code_selector(source: str) -> Callable[[list[dict], random.Random], int]:
    """Compile a select_parent function with restricted builtins (not a security sandbox)."""
    def limited_import(name, *args, **kwargs):
        if name not in ('math', 'random'):
            raise ImportError(f'only math and random may be imported, not {name!r}')
        return __import__(name, *args, **kwargs)
    namespace: dict = {'__builtins__': {**SAFE_BUILTINS, '__import__': limited_import}, 'math': math}
    try:
        exec(compile(source, '<selection_policy>', 'exec'), namespace)  # noqa: S102 - policies are LLM-written code, as in stock EvoX
    except Exception as error:  # noqa: BLE001
        raise PolicyInvalid(f'policy code does not compile: {type(error).__name__}: {error}') from None
    function = namespace.get('select_parent')
    if not callable(function):
        raise PolicyInvalid('policy code must define select_parent(members, rng)')
    return function


def compile_policy(surface: str, text: str) -> Callable[[list[dict], random.Random], int]:
    """Compile and validate a policy on synthetic populations before deployment."""
    select = knob_selector(parse_knobs(text)) if surface == 'knobs' else code_selector(text) if surface == 'code' else None
    if select is None:
        raise ValueError(f'unknown surface {surface!r}')
    probe = random.Random(0)
    for size in (1, 2, 7, 40):
        members = [{'score': probe.random(), 'rank': r, 'rank_pct': 1 - r / max(1, size - 1), 'uses': probe.randrange(4), 'age': probe.randrange(50)} for r in range(size)]
        try:
            index = select(members, random.Random(size))
        except Exception as error:  # noqa: BLE001
            raise PolicyInvalid(f'select_parent raised {type(error).__name__}: {error}') from None
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < size:
            raise PolicyInvalid(f'select_parent must return an index in [0, {size})')
    return select


def stock_text(surface: str) -> str:
    """Stock EvoX-equivalent policy (uniform parent) on either surface."""
    return dump_knobs(STOCK) if surface == 'knobs' else CODE_STOCK
