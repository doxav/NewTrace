GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def _greedy(gpu_num, models, key):
    """Greedy: assign models (sorted by key desc) to the GPU minimizing the
    POST-assignment KVPR (load + r/s) / (free - size); on infeasibility,
    fall back to the GPU with most free memory to guarantee success."""
    placement = {g: [] for g in range(gpu_num)}
    free = [GPU_MEM_SIZE for _ in range(gpu_num)]
    load = [0.0 for _ in range(gpu_num)]
    for model in sorted(models, key=key, reverse=True):
        best_idx, best_ratio = None, float("inf")
        for g in range(gpu_num):
            if model.model_size <= free[g]:
                r = (load[g] + model.req_rate / model.slo) / (free[g] - model.model_size)
                if r < best_ratio:
                    best_ratio, best_idx = r, g
        if best_idx is None:
            best_idx = max(range(gpu_num), key=lambda g: free[g])
        placement[best_idx].append(model)
        load[best_idx] += model.req_rate / model.slo
        free[best_idx] -= model.model_size
    return placement


def _max_kvpr(placement):
    """Max KVPR across GPUs of a placement."""
    worst = 0.0
    for ms in placement.values():
        load = sum(m.req_rate / m.slo for m in ms)
        used = sum(m.model_size for m in ms)
        if GPU_MEM_SIZE - used > 0:
            worst = max(worst, load / (GPU_MEM_SIZE - used))
        elif load > 0:
            worst = max(worst, float("inf"))
    return worst


def compute_model_placement(gpu_num, models):
    """
    Compute a placement minimizing the maximum KVPR across all GPUs.

    Runs the post-assignment-KVPR greedy under several model orderings
    (by req_rate/slo, by model_size, combined, and input order), then
    returns the placement with the lowest max KVPR.
    """
    candidates = [
        _greedy(gpu_num, models, lambda m: m.req_rate / m.slo),
        _greedy(gpu_num, models, lambda m: m.model_size),
        _greedy(gpu_num, models, lambda m: (m.req_rate / m.slo, m.model_size)),
        _greedy(gpu_num, models, lambda m: 0),
    ]
    return min(candidates, key=_max_kvpr)


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
