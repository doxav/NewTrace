import random
import math

def propose(history, bounds, seed):
    """
    Propose a point for black-box minimization.

    Args:
        history: list of dicts with 'x' (list of floats) and 'value' (float).
        bounds: list of [low, high] pairs.
        seed: integer for deterministic randomness.

    Returns:
        list of floats, one per dimension, within bounds.
    """
    rng = random.Random(seed + len(history))
    dim = len(bounds)
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    ranges = [highs[i] - lows[i] for i in range(dim)]

    if not history:
        # Random initial point
        return [rng.uniform(lows[i], highs[i]) for i in range(dim)]

    # Find best point so far
    best = min(history, key=lambda obs: obs['value'])
    best_x = best['x']

    # With probability 0.7, do local search around best; else random
    if rng.random() < 0.7:
        # Gaussian perturbation scaled to 10% of range
        sigma = [ranges[i] / 10.0 for i in range(dim)]
        candidate = [
            best_x[i] + rng.gauss(0, sigma[i])
            for i in range(dim)
        ]
        # Clip to bounds
        candidate = [
            min(max(candidate[i], lows[i]), highs[i])
            for i in range(dim)
        ]
        return candidate
    else:
        # Random point
        return [rng.uniform(lows[i], highs[i]) for i in range(dim)]
