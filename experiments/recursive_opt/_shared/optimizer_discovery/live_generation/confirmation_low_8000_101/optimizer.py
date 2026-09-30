import random

def propose(history, bounds, seed):
    """
    Propose one point for a black-box minimization problem.
    history: list of observations, each with 'x' (list) and 'value' (float)
    bounds: list of [low, high] pairs
    seed: integer
    Returns a list of coordinates within bounds.
    """
    rng = random.Random(seed + len(history))
    ndim = len(bounds)
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    ranges = [h - l for l, h in bounds]

    # No history: uniform random search
    if not history:
        return [rng.uniform(lows[i], highs[i]) for i in range(ndim)]

    # Parse history (handle dict or tuple)
    xs = []
    vals = []
    for obs in history:
        if isinstance(obs, dict):
            x = obs['x']
            val = obs['value']
        else:
            x, val = obs[0], obs[1]
        xs.append(x)
        vals.append(val)

    # Find best point so far
    best_idx = min(range(len(vals)), key=vals.__getitem__)
    best_x = xs[best_idx]

    # Exploration vs exploitation
    # 20% chance of pure random exploration
    if rng.random() < 0.2:
        return [rng.uniform(lows[i], highs[i]) for i in range(ndim)]

    # Otherwise, local search around the best point
    # Step size shrinks with more observations (sqrt decay)
    sigma = [ranges[i] * (0.5 / (1 + len(history)) ** 0.5) for i in range(ndim)]
    proposal = [best_x[i] + rng.gauss(0, sigma[i]) for i in range(ndim)]

    # Clip to bounds
    proposal = [min(max(proposal[i], lows[i]), highs[i]) for i in range(ndim)]
    return proposal
