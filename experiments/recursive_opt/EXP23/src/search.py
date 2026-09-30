"""One solution-search step and the policy scores computed from its decision records."""

import math
import random
from collections.abc import Callable
from statistics import mean

from src.policies import STOCK, knob_selector
from src.world import Population, World

UNIFORM = knob_selector(STOCK)
CREDITS = ('gain', 'pct', 'beat_parent', 'new_best')


def step(world: World, population: Population, select: Callable, seed: int, iteration: int, tag: str = 'active') -> dict:
    """Consume one solution call: choose a parent, generate, insert, and record the decision."""
    view = population.view(iteration)
    rng = random.Random(f'policy:{seed}:{iteration}')
    error = None
    try:
        index = select(view, rng)
        if not isinstance(index, int) or not 0 <= index < len(view):
            raise ValueError('index out of range')
    except Exception as exc:  # runtime policy failure: stock fallback, like EvoX restoring its previous database  # noqa: BLE001
        error, index = type(exc).__name__, UNIFORM(view, rng)
    parent = population.members[index]
    best_before = population.best
    child = world.generate(parent, seed, iteration)
    record = {'iteration': iteration, 'tag': tag, 'parent_score': parent.score, 'parent_rank': view[index]['rank'], 'parent_rank_pct': view[index]['rank_pct'], 'parent_uses': parent.uses, 'best_before': best_before, 'child': child, 'child_pct': None, 'policy_error': error}
    parent.uses += 1
    if child is not None:
        record['child_pct'] = population.percentile(child)
        population.add(child, iteration)
    return record


def credit(record: dict, kind: str, scale: float) -> float:
    """Per-decision credit; invalid children earn zero."""
    child = record['child']
    if child is None:
        return 0.0
    if kind == 'gain':
        return max(0.0, child - record['best_before']) / scale
    if kind == 'pct':
        return record['child_pct']
    if kind == 'beat_parent':
        return float(child > record['parent_score'])
    if kind == 'new_best':
        return float(child > record['best_before'])
    raise ValueError(kind)


def log_window(start: float, end: float, horizon: int = 10) -> float:
    """Stock EvoX LogWindowScorer combined_score (the biased baseline)."""
    return (end - start) * (1.0 + math.log(1.0 + max(0.0, start))) / math.sqrt(horizon)


def paired_score(records: list[dict], kind: str, scale: float) -> dict:
    """Challenger minus incumbent mean credit over one interleaved window.

    Both policies act on the same live population during the same iterations,
    so search stage cancels out. Positive favours the challenger.
    """
    groups = {tag: [credit(r, kind, scale) for r in records if r['tag'] == tag] for tag in ('challenger', 'incumbent')}
    if not groups['challenger'] or not groups['incumbent']:
        return {'score': 0.0, 'n_challenger': len(groups['challenger']), 'n_incumbent': len(groups['incumbent'])}
    return {'score': mean(groups['challenger']) - mean(groups['incumbent']), 'n_challenger': len(groups['challenger']), 'n_incumbent': len(groups['incumbent'])}


def interleave(window: int, seed: int, start: int) -> list[str]:
    """Balanced random assignment of iterations to challenger/incumbent."""
    tags = ['challenger', 'incumbent'] * (window // 2) + (['challenger'] if window % 2 else [])
    random.Random(f'assign:{seed}:{start}').shuffle(tags)
    return tags


def run_fixed(world: World, select: Callable, seed: int, horizon: int = 100) -> list[float]:
    """Best-so-far curve of one fixed policy."""
    population = Population.start(world.init)
    curve = []
    for iteration in range(1, horizon + 1):
        step(world, population, select, seed, iteration)
        curve.append(population.best)
    return curve


def summarize(records: list[dict], scale: float) -> dict:
    """Compact, decision-level evidence for the optimizer (bounded size, no source code)."""
    buckets: dict[str, list[dict]] = {}
    for r in records:
        rank = 'rank0(best)' if r['parent_rank'] == 0 else 'top25%' if r['parent_rank_pct'] >= 0.75 else 'mid' if r['parent_rank_pct'] >= 0.25 else 'bottom25%'
        buckets.setdefault(f"{r['tag']}|{rank}|{'reused' if r['parent_uses'] else 'fresh'}", []).append(r)
    table = {}
    for key, rows in sorted(buckets.items()):
        valid = [r for r in rows if r['child'] is not None]
        table[key] = {'n': len(rows), 'invalid': len(rows) - len(valid), 'child_beats_parent': sum(r['child'] > r['parent_score'] for r in valid), 'new_best': sum(r['child'] > r['best_before'] for r in valid), 'mean_child_pct': round(mean([r['child_pct'] for r in valid]), 3) if valid else None, 'gain_in_median_deltas': round(sum(credit(r, 'gain', scale) for r in rows), 3)}
    return {'decisions': len(records), 'policy_errors': sum(bool(r['policy_error']) for r in records), 'by_tag_parent_rank_reuse': table}
