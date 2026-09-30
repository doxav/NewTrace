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

    """Greedy placement minimizing maximum KVPR.

    Sort models by req_rate/slo descending; assign each model to the feasible
    GPU with the lowest current KVPR (w/mem). If no GPU fits, fall back to
    the GPU with the most free memory. Simple, fast, and robust.
    """
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    mem = [float(GPU_MEM_SIZE)] * gpu_num
    w = [0.0] * gpu_num

    for model in sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True):
        best, best_ratio = None, float("inf")
        for g in range(gpu_num):
            if model.model_size <= mem[g] and mem[g] > 0:
                ratio = w[g] / mem[g]
                if ratio < best_ratio:
                    best_ratio, best = ratio, g
        if best is None:
            best = max(range(gpu_num), key=lambda g: mem[g])
        placement[best].append(model)
        w[best] += model.req_rate / model.slo
        mem[best] -= model.model_size

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
