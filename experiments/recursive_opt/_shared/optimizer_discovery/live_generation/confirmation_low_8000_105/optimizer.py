import random

def propose(history, bounds, seed):
    """
    Propose a point for black-box minimization.
    history: list of dicts with 'x' (list of floats) and 'value' (float)
    bounds: list of [low, high] pairs
    seed: integer
    Returns a list of floats within bounds.
    """
    rng = random.Random(seed + len(history))
    dim = len(bounds)

    if not history:
        # No observations: uniform random
        return [rng.uniform(low, high) for low, high in bounds]

    # Find best observed point (lowest value)
    best = min(history, key=lambda h: h['value'])
    best_x = best['x']

    # With probability 0.5, sample near best; otherwise uniform
    if rng.random() < 0.5:
        # Local perturbation step size shrinks with more data
        step = [(high - low) / (len(history) + 1) for low, high in bounds]
        perturbed = [
            best_x[i] + rng.uniform(-step[i], step[i])
            for i in range(dim)
        ]
        # Clip to bounds
        return [
            min(max(perturbed[i], bounds[i][0]), bounds[i][1])
            for i in range(dim)
        ]
    else:
        # Uniform random exploration
        return [rng.uniform(low, high) for low, high in bounds]
