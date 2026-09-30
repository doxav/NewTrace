GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def _greedy(gpu_num, models, order_key):
    """Greedy pass: assign each model (in given order) to the GPU minimizing the resulting KVPR."""
    placement = {g: [] for g in range(gpu_num)}
    free = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num  # sum of req_rate/slo per GPU

    for m in sorted(models, key=order_key):
        best, best_kvpr = None, float("inf")
        for g in range(gpu_num):
            if m.model_size <= free[g]:
                kvpr = (load[g] + m.req_rate / m.slo) / (free[g] - m.model_size)
                if kvpr < best_kvpr:
                    best_kvpr, best = kvpr, g
        if best is None:
            return None  # infeasible for this ordering
        placement[best].append(m)
        load[best] += m.req_rate / m.slo
        free[best] -= m.model_size
    return placement


def compute_model_placement(gpu_num, models):
    """
    Minimize max KVPR by running greedy placements under several orderings
    (by load/size ratio desc/asc, by size desc) and returning the best result.
    """
    orders = [
        lambda m: -(m.req_rate / m.slo),
        lambda m: (m.req_rate / m.slo),
        lambda m: -m.model_size,
        lambda m: -(m.req_rate / m.slo / max(m.model_size, 1e-9)),
    ]
    best, best_kvpr = None, float("inf")
    for key in orders:
        p = _greedy(gpu_num, models, key)
        if p is None:
            continue
        kvpr = max(
            (sum(m.req_rate / m.slo for m in ms) / (GPU_MEM_SIZE - sum(m.model_size for m in ms)))
            if ms else 0.0
            for ms in p.values()
        )
        if kvpr < best_kvpr:
            best_kvpr, best = kvpr, p
    if best is None:
        raise ValueError("Unable to place all models on the GPUs.")
    return best


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
