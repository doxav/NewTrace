import random

def propose(history, bounds, seed):
    """
    Propose a point for black-box minimization.
    Uses history to guide search with a simple local perturbation strategy.
    """
    rng = random.Random(seed + len(history))
    n = len(bounds)
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    ranges = [highs[i] - lows[i] for i in range(n)]

    # No prior observations: uniform random search
    if not history:
        return [rng.uniform(lows[i], highs[i]) for i in range(n)]

    # Find best point seen so far
    best = min(history, key=lambda h: h['value'])
    best_x = best['x']

    # Exploration vs exploitation
    if rng.random() < 0.3:
        # Explore: random point
        return [rng.uniform(lows[i], highs[i]) for i in range(n)]

    # Exploit: perturb best point with Gaussian noise
    # Step size decreases slightly with more history to refine
    scale = 0.1 / (1 + 0.05 * len(history))
    new_x = []
    for i in range(n):
        step = ranges[i] * scale
        noise = rng.gauss(0, step)
        val = best_x[i] + noise
        val = max(lows[i], min(highs[i], val))  # clip
        new_x.append(val)
    return new_x
