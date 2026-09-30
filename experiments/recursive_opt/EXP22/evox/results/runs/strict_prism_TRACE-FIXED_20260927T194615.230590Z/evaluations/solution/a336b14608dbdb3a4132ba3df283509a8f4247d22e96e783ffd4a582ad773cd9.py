GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """Minimize max KVPR: run the resulting-KVPR greedy heuristic under
    several sort keys and return the placement with the smallest maximum
    KVPR. Each model is assigned to the feasible GPU minimizing KVPR after
    placement ((load+r)/(free-size)); require strictly positive remaining
    free memory to avoid division by zero. If nothing fits, fall back to
    the GPU with the most free memory to guarantee feasibility."""
    best, best_kvpr = None, float("inf")
    for key in (
        lambda m: m.req_rate / m.slo,
        lambda m: m.model_size,
        lambda m: (m.req_rate / m.slo) / m.model_size,
    ):
        placement = {g: [] for g in range(gpu_num)}
        free = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for m in sorted(models, key=key, reverse=True):
            r = m.req_rate / m.slo
            best_g, best_ratio = None, float("inf")
            for g in range(gpu_num):
                if 0 < free[g] - m.model_size:
                    ratio = (load[g] + r) / (free[g] - m.model_size)
                    if ratio < best_ratio:
                        best_ratio, best_g = ratio, g
            if best_g is None:
                best_g = max(range(gpu_num), key=lambda g: free[g])
            placement[best_g].append(m)
            load[best_g] += r
            free[best_g] -= m.model_size
        mk = max(
            (sum(m.req_rate / m.slo for m in ms)
             / (GPU_MEM_SIZE - sum(m.model_size for m in ms))
             for ms in placement.values() if ms),
            default=0.0,
        )
        if mk < best_kvpr:
            best_kvpr, best = mk, placement
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
