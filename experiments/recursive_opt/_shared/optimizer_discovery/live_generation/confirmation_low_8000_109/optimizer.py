import random

def propose(history, bounds, seed):
    rng = random.Random(seed + len(history))
    if not history:
        return [rng.uniform(low, high) for low, high in bounds]

    best = min(history, key=lambda h: h['value'])
    best_x = best['x']

    # Exploit with probability 0.7, else explore
    if rng.random() < 0.7:
        point = []
        for i, (low, high) in enumerate(bounds):
            center = best_x[i]
            std = (high - low) / 10.0
            val = rng.gauss(center, std)
            val = max(low, min(high, val))
            point.append(val)
        return point
    else:
        return [rng.uniform(low, high) for low, high in bounds]
