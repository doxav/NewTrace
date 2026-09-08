import random

def propose(history, bounds, seed):
    """
    Propose a single point for black-box minimization.
    history: list of dicts with keys 'x' (list of floats) and 'value' (float).
    bounds: list of [low, high] pairs.
    seed: integer.
    Returns a list of floats within bounds.
    """
    rng = random.Random(seed + len(history))
    dim = len(bounds)

    if not history:
        # No prior observations: uniform random search
        return [rng.uniform(low, high) for low, high in bounds]

    # Find best observed point
    best = min(history, key=lambda obs: obs['value'])
    best_x = best['x']

    # Compute ranges for scaling perturbations
    ranges = [high - low for low, high in bounds]

    # Decide whether to explore randomly or exploit near best point
    if rng.random() < 0.2:
        # Random exploration
        return [rng.uniform(low, high) for low, high in bounds]

    # Perturb best point with small noise
    proposed = []
    for i, (low, high) in enumerate(bounds):
        if ranges[i] > 0:
            noise = rng.uniform(-0.1 * ranges[i], 0.1 * ranges[i])
        else:
            noise = 0.0
        val = best_x[i] + noise
        # Clamp to bounds
        val = max(low, min(high, val))
        proposed.append(val)
    return proposed
