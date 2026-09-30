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

    """Multi-order greedy: for each sort order (by size, by weight, by ratio),
    assign each model to the GPU minimizing the RESULTING KVPR
    (w + r) / (mem - s). Keep the placement with the lowest max KVPR.
    A best-fit (tightest remaining memory) fallback guarantees feasibility."""

    def greedy(order, bestfit=False):
        placement = {g: [] for g in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE] * gpu_num
        weighted = [0.0] * gpu_num
        for model in order:
            best_idx, best_key = None, float("inf")
            for g in range(gpu_num):
                if model.model_size <= shared_kv[g]:
                    if bestfit:
                        key = shared_kv[g] - model.model_size
                    else:
                        key = (weighted[g] + model.req_rate / model.slo) / (
                            shared_kv[g] - model.model_size
                        )
                    if key < best_key:
                        best_key, best_idx = key, g
            if best_idx is None:
                return None
            placement[best_idx].append(model)
            weighted[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        return placement, weighted, shared_kv

    def max_kvpr(weighted, shared_kv):
        return max(
            (weighted[g] / shared_kv[g]) if shared_kv[g] > 0 else float("inf")
            for g in range(gpu_num)
        )

    orders = [
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True),
        sorted(models, key=lambda m: m.model_size),
    ]

    best, best_v = None, float("inf")
    for order in orders:
        res = greedy(order)
        if res is None:
            res = greedy(order, bestfit=True)  # feasibility fallback
        if res is None:
            continue
        p, weighted, shared_kv = res
        v = max_kvpr(weighted, shared_kv)
        if v < best_v:
            best_v, best = v, p

    if best is None:
        raise ValueError("Unable to place all models on the available GPUs.")
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
