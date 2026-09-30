GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """

    """Greedy placement + local search to minimize max KVPR."""
    w = [m.req_rate / m.slo for m in models]

    # Greedy: sort by weight descending, place on GPU minimizing resulting KVPR
    order = sorted(range(len(models)), key=lambda i: w[i], reverse=True)
    placement = {g: [] for g in range(gpu_num)}
    load = [0.0] * gpu_num
    used = [0.0] * gpu_num

    for i in order:
        m = models[i]
        best_g, best_kvpr = None, float("inf")
        for g in range(gpu_num):
            if used[g] + m.model_size <= GPU_MEM_SIZE:
                kvpr = (load[g] + w[i]) / (GPU_MEM_SIZE - used[g] - m.model_size)
                if kvpr < best_kvpr:
                    best_kvpr, best_g = kvpr, g
        if best_g is None:
            raise ValueError(f"Cannot place model of size {m.model_size} GB")
        placement[best_g].append(m)
        load[best_g] += w[i]
        used[best_g] += m.model_size

    def max_kvpr():
        return max((load[g] / (GPU_MEM_SIZE - used[g]) if used[g] < GPU_MEM_SIZE else float("inf"))
                   for g in range(gpu_num))

    # Local search: move single models while it reduces max KVPR
    cur = max_kvpr()
    improved = True
    while improved:
        improved = False
        for g in range(gpu_num):
            for m in list(placement[g]):
                wi = m.req_rate / m.slo
                for h in range(gpu_num):
                    if h == g or used[h] + m.model_size > GPU_MEM_SIZE:
                        continue
                    # simulate move g -> h
                    new_loads = load[:]
                    new_used = used[:]
                    new_loads[g] -= wi; new_used[g] -= m.model_size
                    new_loads[h] += wi; new_used[h] += m.model_size
                    new_max = max(
                        (new_loads[x] / (GPU_MEM_SIZE - new_used[x])
                         if new_used[x] < GPU_MEM_SIZE else float("inf"))
                        for x in range(gpu_num))
                    if new_max < cur - 1e-12:
                        placement[g].remove(m)
                        placement[h].append(m)
                        load, used, cur = new_loads, new_used, new_max
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return placement


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
