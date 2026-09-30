import random

def propose(history, bounds, seed):
    """
    Propose a point for black-box minimization.

    Args:
        history: list of dicts with 'x' (list of coordinates) and 'value' (float).
        bounds: list of [low, high] pairs.
        seed: integer for deterministic randomness.

    Returns:
        list of coordinates within bounds.
    """
    rng = random.Random(seed + len(history))
    dim = len(bounds)
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    ranges = [high - low for low, high in bounds]

    if not history:
        # Uniform random search when no prior observations
        return [rng.uniform(low, high) for low, high in bounds]

    # Find the best point so far
    best = min(history, key=lambda h: h['value'])
    best_x = best['x']

    # Exploration vs exploitation
    if rng.random() < 0.3:
        # Explore: uniform random
        return [rng.uniform(low, high) for low, high in bounds]
    else:
        # Exploit: perturb the best point with Gaussian noise
        new_x = []
        for i in range(dim):
            std = ranges[i] * 0.1
            perturb = rng.gauss(0, std)
            val = best_x[i] + perturb
            # Clip to bounds
            val = max(lows[i], min(highs[i], val))
            new_x.append(val)
        return new_x
