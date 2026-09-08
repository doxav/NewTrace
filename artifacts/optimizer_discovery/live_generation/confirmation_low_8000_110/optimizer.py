import random

def propose(history, bounds, seed):
    """
    Propose a point for black-box minimization.
    Uses random search with local perturbation around the best observed point.
    """
    rng = random.Random(seed + len(history))

    # No history: uniform random point
    if not history:
        return [rng.uniform(low, high) for low, high in bounds]

    # Find best observed point
    best = min(history, key=lambda obs: obs['value'])
    best_x = best['x']

    # 20% chance to explore uniformly, otherwise exploit around best
    if rng.random() < 0.2:
        return [rng.uniform(low, high) for low, high in bounds]

    # Perturb best point with Gaussian noise, clipped to bounds
    new_x = []
    for i, (low, high) in enumerate(bounds):
        sigma = (high - low) * 0.1
        val = best_x[i] + rng.gauss(0, sigma)
        val = max(low, min(high, val))
        new_x.append(val)
    return new_x
