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

    """Multi-order greedy + move-based local search.

    For each of several sort orders, greedily assign each model to the GPU
    minimizing the RESULTING KVPR (w + r) / (mem - s). Then refine the best
    greedy result by repeatedly moving a single model from the GPU with the
    highest KVPR to another GPU whenever it strictly reduces the max KVPR.
    """

    def greedy(order):
        placement = {g: [] for g in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE] * gpu_num
        weighted = [0.0] * gpu_num
        for model in order:
            best_idx, best_ratio = None, float("inf")
            for g in range(gpu_num):
                if model.model_size <= shared_kv[g]:
                    ratio = (weighted[g] + model.req_rate / model.slo) / (
                        shared_kv[g] - model.model_size
                    )
                    if ratio < best_ratio:
                        best_ratio, best_idx = ratio, g
            if best_idx is None:
                return None
            placement[best_idx].append(model)
            weighted[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        return placement, weighted, shared_kv

    def max_kvpr(weighted, shared_kv):
        return max(
            w / mem if mem > 0 else float("inf")
            for w, mem in zip(weighted, shared_kv)
        )

    orders = [
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True),
        sorted(models, key=lambda m: m.model_size),
    ]

    best, best_state, best_kvpr = None, None, float("inf")
    for order in orders:
        res = greedy(order)
        if res is None:
            continue
        p, weighted, shared_kv = res
        v = max_kvpr(weighted, shared_kv)
        if v < best_kvpr:
            best_kvpr, best, best_state = v, p, (weighted, shared_kv)

    # Fallback: first-fit by decreasing size, to guarantee success whenever
    # a feasible placement exists at all.
    if best is None:
        best = {g: [] for g in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE] * gpu_num
        for model in sorted(models, key=lambda m: m.model_size, reverse=True):
            for g in range(gpu_num):
                if model.model_size <= shared_kv[g]:
                    best[g].append(model)
                    shared_kv[g] -= model.model_size
                    break
            else:
                raise ValueError("Unable to place all models on the available GPUs.")
        return best

    # Move-based local search on the best placement: repeatedly move one
    # model out of the GPU with the highest KVPR whenever it strictly
    # reduces the overall maximum KVPR.
    weighted, shared_kv = best_state
    improved = True
    while improved:
        improved = False
        cur = max_kvpr(weighted, shared_kv)
        src = max(
            range(gpu_num),
            key=lambda g: weighted[g] / shared_kv[g] if shared_kv[g] > 0 else float("inf"),
        )
        for m in list(best[src]):
            w = m.req_rate / m.slo
            for dst in range(gpu_num):
                if dst == src or m.model_size > shared_kv[dst]:
                    continue
                weighted[src] -= w; shared_kv[src] += m.model_size
                weighted[dst] += w; shared_kv[dst] -= m.model_size
                new = max_kvpr(weighted, shared_kv)
                if new < cur - 1e-12:
                    best[src].remove(m); best[dst].append(m)
                    improved = True
                    break
                weighted[src] += w; shared_kv[src] -= m.model_size
                weighted[dst] -= w; shared_kv[dst] += m.model_size
            if improved:
                break

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
