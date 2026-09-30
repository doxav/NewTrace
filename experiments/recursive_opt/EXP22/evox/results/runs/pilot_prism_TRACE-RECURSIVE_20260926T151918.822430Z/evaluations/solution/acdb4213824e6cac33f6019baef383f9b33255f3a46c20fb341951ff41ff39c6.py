GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def _greedy(gpu_num, models, sort_key):
    """Greedy placement: sort models by sort_key, assign each to the GPU
    minimizing the post-placement KVPR while fitting in memory."""
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]
    weighted = [0.0 for _ in range(gpu_num)]
    for model in sorted(models, key=sort_key):
        best_idx, best_ratio = None, float("inf")
        for gpu_id in range(gpu_num):
            if model.model_size <= shared_kv[gpu_id]:
                ratio = (weighted[gpu_id] + model.req_rate / model.slo) / (
                    shared_kv[gpu_id] - model.model_size
                )
                if ratio < best_ratio:
                    best_ratio, best_idx = ratio, gpu_id
        if best_idx is None:
            return None
        placement[best_idx].append(model)
        weighted[best_idx] += model.req_rate / model.slo
        shared_kv[best_idx] -= model.model_size
    return placement


def _max_kvpr(placement):
    """Compute the maximum KVPR across GPUs for a placement."""
    worst = 0.0
    for models in placement.values():
        load = sum(m.req_rate / m.slo for m in models)
        free = GPU_MEM_SIZE - sum(m.model_size for m in models)
        worst = max(worst, load / free)
    return worst


def compute_model_placement(gpu_num, models):
    """
    Compute a placement minimizing the maximum KVPR across GPUs.

    Approach: run the greedy KVPR-minimizing heuristic under several sort
    orders (size desc, load desc, load/size desc, load/size asc) and return
    the placement with the lowest maximum KVPR.
    """
    candidates = [
        _greedy(gpu_num, models, lambda m: m.model_size),
        _greedy(gpu_num, models, lambda m: -(m.req_rate / m.slo)),
        _greedy(gpu_num, models, lambda m: -(m.req_rate / m.slo / m.model_size)),
        _greedy(gpu_num, models, lambda m: m.req_rate / m.slo / m.model_size),
    ]
    valid = [c for c in candidates if c is not None]
    if not valid:
        raise ValueError("Unable to place all models on the given GPUs.")
    return min(valid, key=_max_kvpr)


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
