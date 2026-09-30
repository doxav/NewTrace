def propose(history, bounds, seed):
    import random
    rng = random.Random(seed + len(history))
    
    def get_x_val(obs):
        if isinstance(obs, dict):
            return obs['x'], obs['value']
        else:
            return obs[0], obs[1]
    
    if not history:
        return [rng.uniform(low, high) for low, high in bounds]
    
    best_x = None
    best_val = float('inf')
    for obs in history:
        x, val = get_x_val(obs)
        if val < best_val:
            best_val = val
            best_x = x
    
    if rng.random() < 0.2:
        return [rng.uniform(low, high) for low, high in bounds]
    else:
        new_x = []
        for i, (low, high) in enumerate(bounds):
            step = (high - low) * 0.1
            noise = rng.gauss(0, step)
            val = best_x[i] + noise
            val = max(low, min(high, val))
            new_x.append(val)
        return new_x
