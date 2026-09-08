import random

def propose(history, bounds, seed):
    rng = random.Random(seed + len(history))
    if not history:
        return [rng.uniform(low, high) for low, high in bounds]
    best = min(history, key=lambda h: h['value'])
    best_x = best['x']
    if rng.random() < 0.2:
        return [rng.uniform(low, high) for low, high in bounds]
    result = []
    for i, (low, high) in enumerate(bounds):
        sigma = 0.1 * (high - low)
        delta = rng.gauss(0, sigma)
        val = best_x[i] + delta
        val = max(low, min(high, val))
        result.append(val)
    return result
