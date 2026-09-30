GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def _greedy(gpu_num, models, key):
    """
    Minimize max KVPR: run the greedy resulting-KVPR heuristic under several
    sort keys and return the placement with the smallest maximum KVPR.
    """
    best, best_kvpr = None, float("inf")
    for key in (
        lambda m: m.req_rate / m.slo,
        lambda m: m.model_size,
        lambda m: (m.req_rate / m.slo) / m.model_size,
    ):
        p = _greedy(gpu_num, models, key)
        mk = _max_kvpr(p)
        if mk < best_kvpr:
            best_kvpr, best = mk, p
    return best

    # Greedy: sort models by `key` descending, assign each model to the GPU
    # that minimizes the *resulting* KVPR after placement. If no GPU fits,
    # fall back to the GPU with the most free memory (avoids hard failures).
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]
    weighted_req_rate = [0.0 for _ in range(gpu_num)]

    for model in sorted(models, key=key, reverse=True):
        r = model.req_rate / model.slo
        best_idx = None
        best_ratio = float("inf")
        for gpu_id in range(gpu_num):
            if model.model_size <= shared_kv[gpu_id]:
                ratio = (weighted_req_rate[gpu_id] + r) / (
                    shared_kv[gpu_id] - model.model_size
                )
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_idx = gpu_id
        if best_idx is None:
            best_idx = max(range(gpu_num), key=lambda g: shared_kv[g])
        placement[best_idx].append(model)
        weighted_req_rate[best_idx] += r
        shared_kv[best_idx] -= model.model_size

    return placement


def _max_kvpr(placement):
    """Maximum KVPR across all GPUs of a placement."""
    return max(
        sum(m.req_rate / m.slo for m in ms)
        / (GPU_MEM_SIZE - sum(m.model_size for m in ms))
        for ms in placement.values()
        if ms
    )


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
