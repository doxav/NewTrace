GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Multi-restart greedy placement minimizing max KVPR.

    Runs a greedy assignment under several model orderings (deterministic
    sort keys plus random shuffles) and returns the placement with the
    lowest maximum KVPR across GPUs.
    """

    def greedy(order):
        placement = {g: [] for g in range(gpu_num)}
        free = [GPU_MEM_SIZE] * gpu_num          # remaining memory per GPU
        load = [0.0] * gpu_num                   # sum of req_rate/slo per GPU
        for m in order:
            best, best_kvpr = None, float("inf")
            for g in range(gpu_num):
                if m.model_size <= free[g]:
                    kvpr = (load[g] + m.req_rate / m.slo) / (free[g] - m.model_size)
                    if kvpr < best_kvpr:
                        best, best_kvpr = g, kvpr
            if best is None:
                return None, float("inf")        # infeasible ordering
            placement[best].append(m)
            load[best] += m.req_rate / m.slo
            free[best] -= m.model_size
        max_kvpr = max((l / f) if f > 0 else float("inf") for l, f in zip(load, free))
        return placement, max_kvpr

    # Candidate orderings: deterministic sorts + random restarts
    orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.model_size),
    ]
    import random
    rng = random.Random(42)
    orders += [list(models) for _ in range(0)]
    for _ in range(200):
        shuffled = list(models)
        rng.shuffle(shuffled)
        orders.append(shuffled)

    best_place, best_kvpr = None, float("inf")
    for order in orders:
        place, kvpr = greedy(order)
        if place is not None and kvpr < best_kvpr:
            best_place, best_kvpr = place, kvpr

    if best_place is None:
        raise ValueError("Unable to place all models on the available GPUs.")
    return best_place


# EVOLVE-BLOCK-END


if __name__ == "__main__":
    # Test the algorithm

    import numpy as np
    from evaluator import calculate_kvcache_pressure, generate_test_gpu_models, safe_float

    test_cases = generate_test_gpu_models()
    all_kvpr = []
    for i, (gpu_num, gpu_models) in enumerate(test_cases):

        results = compute_model_placement(gpu_num, gpu_models)
        max_kvpr = calculate_kvcache_pressure(results)
        all_kvpr.append(safe_float(max_kvpr))

    avg_kvpr = np.mean(all_kvpr)
    if avg_kvpr != 0:
        avg_kvpr = 1.0 / avg_kvpr

    print(f"Max KVPR: {avg_kvpr:.3f}")
