def propose(history, bounds, seed):
    return [(low + high) / 2 for low, high in bounds]
