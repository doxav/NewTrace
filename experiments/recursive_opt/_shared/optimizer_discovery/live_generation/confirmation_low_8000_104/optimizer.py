import random

def propose(history, bounds, seed):
    """
    Propose a single point for black-box minimization.

    Args:
        history: list of dicts with keys 'x' (list of coordinates) and 'value' (float).
        bounds: list of [low, high] pairs, one per dimension.
        seed: integer seed for reproducibility.

    Returns:
        list of floats, one coordinate per dimension, within bounds.
    """
    rng = random.Random(seed + len(history))

    # No history: uniform random sampling
    if not history:
        return [rng.uniform(low, high) for low, high in bounds]

    # Find the best observed point (lowest value)
    best = min(history, key=lambda h: h['value'])
    best_x = best['x']

    # With probability 0.5, perturb the best point; otherwise sample uniformly
    if rng.random() < 0.5:
        proposed = []
        for i, (low, high) in enumerate(bounds):
            # Gaussian perturbation with std = 10% of the range
            std = (high - low) / 10.0
            val = best_x[i] + rng.gauss(0, std)
            # Clip to bounds
            val = max(low, min(high, val))
            proposed.append(val)
        return proposed
    else:
        return [rng.uniform(low, high) for low, high in bounds]
