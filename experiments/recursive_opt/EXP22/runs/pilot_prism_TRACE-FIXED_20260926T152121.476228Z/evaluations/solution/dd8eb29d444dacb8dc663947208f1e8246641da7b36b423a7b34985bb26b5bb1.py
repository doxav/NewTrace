GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize max KVPR via greedy placement over multiple sort orders.
    For each order, assign each model to the GPU minimizing the resulting
    KVPR ((w + r/s) / (mem - size)) while fitting in memory; keep best result.
    """

    def try_order(sorted_models):
        placement = {g: [] for g in range(gpu_num)}
        mem = [GPU_MEM_SIZE] * gpu_num
        w = [0.0] * gpu_num
        for model in sorted_models:
            best_idx, best_kvpr = None, float("inf")
            for g in range(gpu_num):
                if model.model_size <= mem[g]:
                    kvpr = (w[g] + model.req_rate / model.slo) / (mem[g] - model.model_size)
                    if kvpr < best_kvpr:
                        best_kvpr, best_idx = kvpr, g
            if best_idx is None:
                return None  # infeasible for this order
            placement[best_idx].append(model)
            w[best_idx] += model.req_rate / model.slo
            mem[best_idx] -= model.model_size
        return placement

    best = None
    best_kvpr = float("inf")
    keys = [
        lambda m: m.req_rate / m.slo,                      # by load descending
        lambda m: m.req_rate / (m.slo * m.model_size),     # by load/size descending
        lambda m: m.model_size,                            # by size descending
    ]
    for key in keys:
        result = try_order(sorted(models, key=key, reverse=True))
        if result is None:
            continue
        mx = max(
            (sum(m.req_rate / m.slo for m in result[g]) / (GPU_MEM_SIZE - sum(m.model_size for m in result[g])))
            if result[g] and sum(m.model_size for m in result[g]) < GPU_MEM_SIZE else 0.0
            for g in range(gpu_num)
        )
        if mx < best_kvpr:
            best_kvpr, best = mx, result

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
