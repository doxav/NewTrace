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

    """Greedy KVPR-minimizing placement + local search (single-model moves).

    1) Sort models by weight (req_rate/slo) descending.
    2) Greedily assign each model to the GPU minimizing resulting KVPR
       (fallback: GPU with most free memory, to guarantee feasibility).
    3) Local search: repeatedly move a model to another GPU if it reduces
       the maximum KVPR, until no improvement is possible.
    """
    w = {id(m): m.req_rate / m.slo for m in models}
    sorted_models = sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True)

    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    free = [float(GPU_MEM_SIZE) for _ in range(gpu_num)]
    load = [0.0 for _ in range(gpu_num)]

    def kvpr(g, extra_w=0.0, extra_s=0.0):
        denom = free[g] - extra_s
        if denom <= 0:
            return float("inf")
        return (load[g] + extra_w) / denom

    # Greedy placement
    for model in sorted_models:
        best_idx, best_ratio = None, float("inf")
        for gpu_id in range(gpu_num):
            if model.model_size <= free[gpu_id]:
                r = kvpr(gpu_id, w[id(model)], model.model_size)
                if r < best_ratio:
                    best_ratio, best_idx = r, gpu_id
        if best_idx is None:  # fallback: GPU with most free memory
            best_idx = max(range(gpu_num), key=lambda g: free[g])
        placement[best_idx].append(model)
        load[best_idx] += w[id(model)]
        free[best_idx] -= model.model_size

    # Local search: single-model moves
    def max_kvpr():
        return max(kvpr(g) for g in range(gpu_num))

    cur = max_kvpr()
    improved = True
    while improved:
        improved = False
        for g in range(gpu_num):
            for m in list(placement[g]):
                wi = w[id(m)]
                for h in range(gpu_num):
                    if h == g or m.model_size > free[h]:
                        continue
                    load[g] -= wi; free[g] += m.model_size
                    load[h] += wi; free[h] -= m.model_size
                    nm = max_kvpr()
                    if nm < cur - 1e-12:
                        placement[g].remove(m)
                        placement[h].append(m)
                        cur = nm
                        improved = True
                        break
                    # revert
                    load[g] += wi; free[g] -= m.model_size
                    load[h] -= wi; free[h] += m.model_size
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
