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

    # Greedy KVPR-minimizing placement (post-assignment KVPR heuristic)
    # 1) Sort models by r_j / s_j in descending order
    sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True)

    # 2) Initialize per-GPU states
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]  # remaining memory per GPU
    weighted_req_rate = [0.0 for _ in range(gpu_num)]  # sum of r_j / s_j per GPU

    # 3) Assign each model to the GPU minimizing the POST-assignment KVPR
    #    (w + r) / (mem - s), which directly targets the final objective.
    for model in sorted_models:
        best_idx = None
        best_ratio = float("inf")

        for gpu_id in range(gpu_num):
            if model.model_size <= shared_kv[gpu_id] and shared_kv[gpu_id] > 0:
                ratio = (weighted_req_rate[gpu_id] + model.req_rate / model.slo) / (
                    shared_kv[gpu_id] - model.model_size
                )
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_idx = gpu_id

        # Fallback: place on GPU with most remaining memory (keeps feasibility)
        if best_idx is None:
            best_idx = max(range(gpu_num), key=lambda g: shared_kv[g])
            if model.model_size > shared_kv[best_idx]:
                raise ValueError(
                    f"Unable to place model of size {model.model_size} GB on any GPU. "
                    f"Remaining per-GPU memory: {shared_kv}"
                )

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
