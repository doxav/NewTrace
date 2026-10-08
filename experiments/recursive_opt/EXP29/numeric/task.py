"""EXP29 numeric task family: black-box optimizer programs ``propose(history, bounds, seed)`` (OptimizerProgramV0).

Extends EXP15-18 (sphere / rosenbrock) with multimodal families (rastrigin, ackley) and wider shifts, so that
centre-starting and pure local search do not saturate. The objective stays in the host; the candidate program sees only
past points and values (``optimizer_program.propose_point``: fresh process per proposal, determinism replay).

Score = -mean over the budget of log10(normalized best-so-far regret, floored at 1e-6): higher is better, 6 is perfect.
The log scale rewards precision across orders of magnitude; the plain AUC is dominated by the first random draws.
Registered for the control plane as ``exp29.evaluator.bbo@1`` (module ``recursive_opt.module.reasoning_workflow@1``,
component ``optimizer``); dataset items are ``{"instance": {...}, "seed": int}``.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from functools import lru_cache
from itertools import pairwise
from typing import Any

from opto.features.recursive_opt import spec as S
from opto.features.recursive_opt.optimizer_program import propose_point
from opto.trainer.objectives import EvaluationResult

FAMILIES = ('sphere', 'rosenbrock', 'rastrigin', 'ackley')
BOUNDS = (-5.0, 5.0)
BUDGET = 64
FLOOR = 1e-6  # log-regret floor
TARGET = 0.01
SEED_SOURCE = '''def propose(history, bounds, seed):
    """Mix uniform exploration with decreasing incumbent-centered Gaussian steps."""
    import random
    rng = random.Random(seed + len(history))
    if not history or rng.random() < 0.25:
        return [rng.uniform(low, high) for low, high in bounds]
    best = min(history, key=lambda row: row["value"])["x"]
    scale = 0.2 / (1.0 + len(history) / (4.0 * len(bounds))) ** 0.5
    return [max(low, min(high, x + rng.gauss(0.0, scale * (high - low))))
            for x, (low, high) in zip(best, bounds)]
'''


def _seed(*parts: Any) -> int:
    return int(hashlib.sha256(json.dumps(['EXP29', *parts]).encode()).hexdigest()[:12], 16)


def instance(family: str, dimension: int, split: str, index: int) -> dict:
    rng = random.Random(_seed('instance', family, dimension, split, index))
    return {'family': family, 'dimension': dimension, 'shift': [round(rng.uniform(-4, 4), 6) for _ in range(dimension)],
            'scales': [round(rng.uniform(0.75, 1.5), 6) for _ in range(dimension)], 'amplitude': round(10 ** rng.uniform(-1, 1), 6),
            'id': f'{family}-{dimension}d-{split}{index}'}


def objective(task: dict, point: list) -> float:
    z = [(x - s) / c for x, s, c in zip(point, task['shift'], task['scales'])]
    family, n = task['family'], len(z)
    if family == 'sphere':
        value = sum(x * x for x in z)
    elif family == 'rosenbrock':
        y = [1 + x for x in z]
        value = sum(100 * (b - a * a) ** 2 + (1 - a) ** 2 for a, b in pairwise(y))
    elif family == 'rastrigin':
        value = 10 * n + sum(x * x - 10 * math.cos(2 * math.pi * x) for x in z)
    elif family == 'ackley':
        value = (-20 * math.exp(-0.2 * math.sqrt(sum(x * x for x in z) / n)) - math.exp(sum(math.cos(2 * math.pi * x) for x in z) / n)
                 + 20 + math.e)
    else:
        raise ValueError(family)
    return float(task['amplitude'] * max(0.0, value))


@lru_cache(maxsize=None)
def _scale(serialized: str) -> float:
    task = json.loads(serialized)
    rng = random.Random(_seed('normalization', task['id']))
    return statistics.mean(objective(task, [rng.uniform(*BOUNDS) for _ in range(task['dimension'])]) for _ in range(128))


def run_program(source: str, task: dict, seed: int, budget: int = BUDGET) -> dict:
    """One trajectory; invalid proposals stop it (typed status, no imputed score)."""
    bounds = [list(BOUNDS)] * task['dimension']
    history: list = []
    for _ in range(budget):
        proposal = propose_point(source, history, bounds, seed, timeout_s=2.0)
        if not proposal.valid:
            return {'valid': False, 'status': proposal.status, 'evaluations': len(history), 'stderr': proposal.stderr[-400:]}
        history.append({'x': proposal.point, 'value': objective(task, proposal.point)})
    scale, best, curve = _scale(json.dumps(task, sort_keys=True)), math.inf, []
    for row in history:
        best = min(best, row['value'] / scale)
        curve.append(best)
    hit = next((i + 1 for i, v in enumerate(curve) if v <= TARGET), None)
    return {'valid': True, 'status': 'valid', 'auc': statistics.mean(curve), 'log_auc': statistics.mean(math.log10(max(v, FLOOR)) for v in curve), 'final_regret': curve[-1], 'target_evaluations': hit,
            'curve_head': [round(v, 4) for v in curve[:: max(1, budget // 8)]]}


def evaluator(output: Any, example: Any, context: Any) -> EvaluationResult:
    data = getattr(output, 'data', output)
    source = data['components']['optimizer']
    item = getattr(example, 'data', example)
    task, seed = item['instance'], int(item['seed'])
    result = run_program(source, task, seed)
    if not result['valid']:
        return EvaluationResult(valid=False, status='invalid', error=result['status'],
                                feedback=f'{task["id"]}: invalid program ({result["status"]}) after {result["evaluations"]} evaluations. {result["stderr"]}')
    feedback = (f'{task["id"]} (bounds {list(BOUNDS)}, budget {BUDGET}): mean log10 regret {result["log_auc"]:.3f} (score {-result["log_auc"]:.3f}), regret AUC {result["auc"]:.4f}, final normalized regret '
                f'{result["final_regret"]:.4g}, target {TARGET} reached at evaluation {result["target_evaluations"]}; '
                f'best-so-far regret every {BUDGET // 8} evaluations {result["curve_head"]}')
    return EvaluationResult(valid=True, status='ok', metrics={'score': -result['log_auc'], 'final_regret': result['final_regret']}, feedback=feedback)


S.register_evaluator('exp29.evaluator.bbo@1', evaluator)


def items(families, dimensions, split: str, count: int, seed: int = 0) -> list:
    """Dataset items for a split: ``count`` instances per (family, dimension), one local seed each."""
    return [{'instance': instance(f, d, split, i), 'seed': _seed('local', f, d, split, i, seed) % 100000}
            for f in families for d in dimensions for i in range(count)]
