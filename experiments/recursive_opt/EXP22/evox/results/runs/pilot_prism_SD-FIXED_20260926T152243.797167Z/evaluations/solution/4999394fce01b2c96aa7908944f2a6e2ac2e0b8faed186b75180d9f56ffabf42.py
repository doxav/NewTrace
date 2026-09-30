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

    # Greedy placement: sort by req_rate/slo descending, then assign each model
    # to the GPU that minimizes the POST-assignment KVPR (w + r/s) / (mem - s),
    # which directly targets minimizing the max KVPR.
    sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True)

    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]  # remaining memory per GPU
    weighted_req_rate = [0.0 for _ in range(gpu_num)]  # sum of r_j / s_j per GPU

    for model in sorted_models:
        best_idx = None
        best_ratio = float("inf")

        for gpu_id in range(gpu_num):
            if model.model_size <= shared_kv[gpu_id]:
                new_ratio = (weighted_req_rate[gpu_id] + model.req_rate / model.slo) / (
                    shared_kv[gpu_id] - model.model_size
                )
                if new_ratio < best_ratio:
                    best_ratio = new_ratio
                    best_idx = gpu_id

        if best_idx is None:
            raise ValueError(f"Cannot place model of size {model.model_size} GB")

        placement[best_idx].append(model)
        weighted_req_rate[best_idx] += model.req_rate / model.slo
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
