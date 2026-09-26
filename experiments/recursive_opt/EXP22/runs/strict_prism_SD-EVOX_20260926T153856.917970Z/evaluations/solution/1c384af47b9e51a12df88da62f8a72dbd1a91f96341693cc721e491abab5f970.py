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

    """Greedy placement minimizing maximum KVPR across GPUs.

    Sorts models by req_rate/slo descending, then assigns each model to the
    GPU (with sufficient memory) whose resulting KVPR is lowest. If no GPU
    can fit the model, falls back to the GPU with the most remaining memory.
    """
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [float(GPU_MEM_SIZE) for _ in range(gpu_num)]
    weighted_req = [0.0 for _ in range(gpu_num)]

    for model in sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True):
        best_idx = None
        best_ratio = float("inf")
        for gpu_id in range(gpu_num):
            if model.model_size <= shared_kv[gpu_id] and shared_kv[gpu_id] > 0:
                ratio = weighted_req[gpu_id] / shared_kv[gpu_id]
                if ratio < best_ratio:
                    best_ratio, best_idx = ratio, gpu_id
        if best_idx is None:
            best_idx = max(range(gpu_num), key=lambda g: shared_kv[g])
        placement[best_idx].append(model)
        weighted_req[best_idx] += model.req_rate / model.slo
        shared_kv[best_idx] -= model.model_size

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
