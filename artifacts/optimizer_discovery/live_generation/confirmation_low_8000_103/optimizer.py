import random
import math

def propose(history, bounds, seed):
    """
    Propose a single point for black-box minimization.
    Uses a simple random search with local exploitation around the best observed point.
    """
    rng = random.Random(seed + len(history))
    dim = len(bounds)

    # If no history, sample uniformly
    if not history:
        return [rng.uniform(low, high) for low, high in bounds]

    # Find best point (lowest value)
    best_x = min(history, key=lambda obs: obs["value"])["x"]

    # Decide whether to explore or exploit
    if rng.random() < 0.2:  # 20% exploration
        return [rng.uniform(low, high) for low, high in bounds]

    # Exploit: sample around best point with small std
    proposal = []
    for i, (low, high) in enumerate(bounds):
        std = 0.1 * (high - low)
        # Sample from normal, clip to bounds
        val = rng.gauss(best_x[i], std)
        val = max(low, min(high, val))
        proposal.append(val)
    return proposal
