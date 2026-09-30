GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def _greedy(gpu_num, models, key, post=True):
    """Greedy: assign models (sorted by key desc) to the GPU minimizing KVPR.
    If post=True, minimize POST-assignment KVPR (load+r/s)/(free-size);
    otherwise minimize current KVPR load/free. Strict memory fit: if no GPU
    can fit the model, return None (infeasible)."""
    placement = {g: [] for g in range(gpu_num)}
    free = [GPU_MEM_SIZE for _ in range(gpu_num)]
    load = [0.0 for _ in range(gpu_num)]
    for model in sorted(models, key=key, reverse=True):
        best_idx, best_ratio = None, float("inf")
        for g in range(gpu_num):
            if model.model_size <= free[g]:
                if post:
                    r = (load[g] + model.req_rate / model.slo) / (free[g] - model.model_size)
                else:
                    r = load[g] / free[g]
                if r < best_ratio:
                    best_ratio, best_idx = r, g
        if best_idx is None:
            return None
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
    (by req_rate/slo, by model_size, combined, and input order), keeping
    only strictly feasible placements, then returns the one with the
    lowest max KVPR. If all fail, falls back to the pre-assignment greedy.
    """
    keys = [
        lambda m: m.req_rate / m.slo,
        lambda m: m.model_size,
        lambda m: (m.req_rate / m.slo, m.model_size),
        lambda m: 0,
    ]
    candidates = [p for p in (_greedy(gpu_num, models, k) for k in keys) if p is not None]
    if not candidates:
        candidates = [p for p in (_greedy(gpu_num, models, k, post=False) for k in keys)
                      if p is not None]
    if not candidates:
        raise ValueError("Unable to place all models on the GPUs")
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
