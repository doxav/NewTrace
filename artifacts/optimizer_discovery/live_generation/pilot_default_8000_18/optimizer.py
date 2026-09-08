import random

def propose(history, bounds, seed):
    rng = random.Random(seed + len(history))
    dim = len(bounds)

    def random_point():
        return [rng.uniform(low, high) for low, high in bounds]

    if not history:
        return random_point()

    best = min(history, key=lambda obs: obs['value'])
    best_x = best['x']

    # 50% chance to explore, 50% chance to exploit around best
    if rng.random() < 0.5:
        return random_point()

    # Perturb the best point with a Gaussian
    new_x = []
    for i, (low, high) in enumerate(bounds):
        std = (high - low) / 10.0
        val = best_x[i] + rng.gauss(0, std)
        val = max(low, min(high, val))  # clamp to bounds
        new_x.append(val)
    return new_x
