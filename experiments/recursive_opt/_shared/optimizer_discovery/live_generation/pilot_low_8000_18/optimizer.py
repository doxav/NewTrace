import random

def propose(history, bounds, seed):
    """
    Propose a point for black-box minimization.
    """
    rng = random.Random(seed + len(history))

    # No history: random point in bounds
    if not history:
        return [rng.uniform(low, high) for low, high in bounds]

    # Find best observed point (lowest value)
    best = min(history, key=lambda obs: obs['value'])
    best_x = best['x']

    # Exploration vs exploitation
    if rng.random() < 0.5:
        # Pure exploration: uniform random point
        return [rng.uniform(low, high) for low, high in bounds]
    else:
        # Local perturbation of the best point
        point = []
        for i, (low, high) in enumerate(bounds):
            width = high - low
            step = rng.uniform(-0.2 * width, 0.2 * width)
            val = best_x[i] + step
            val = max(low, min(high, val))  # clip to bounds
            point.append(val)
        return point
