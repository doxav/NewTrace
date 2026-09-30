GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def _greedy(gpu_num, models, key, resulting):
    """Strict greedy placement: sort models by `key` descending; assign each
    model to the feasible GPU minimizing the resulting KVPR
    ((load+r)/(free-size)) if `resulting`, else the current KVPR (load/free).
    Raises ValueError if a model fits no GPU (never overcommits memory)."""
    placement = {g: [] for g in range(gpu_num)}
    free = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num
    for m in sorted(models, key=key, reverse=True):
        r = m.req_rate / m.slo
        best_idx, best_ratio = None, float("inf")
        for g in range(gpu_num):
            if 0 < m.model_size <= free[g]:
                ratio = (load[g] + r) / (free[g] - m.model_size) if resulting \
                    else load[g] / free[g]
                if ratio < best_ratio:
                    best_ratio, best_idx = ratio, g
        if best_idx is None:
            raise ValueError(f"Model of size {m.model_size} GB fits no GPU")
        placement[best_idx].append(m)
        load[best_idx] += r
        free[best_idx] -= m.model_size
    return placement


def _max_kvpr(placement):
    """Return the maximum KVPR across all GPUs of a placement."""
    return max(
        (sum(m.req_rate / m.slo for m in ms) / (GPU_MEM_SIZE - sum(m.model_size for m in ms))
         for ms in placement.values() if ms),
        default=0.0,
    )


def compute_model_placement(gpu_num, models):
    """Minimize max KVPR: run the strict greedy heuristic under both the
    current-KVPR and resulting-KVPR criteria, each with three sort orders
    (req_rate/slo, model size, ratio), returning the placement with the
    smallest max KVPR. Raises if no strict placement is feasible."""
    best, best_kvpr = None, float("inf")
    for resulting in (False, True):
        for key in (lambda m: m.req_rate / m.slo, lambda m: m.model_size,
                    lambda m: (m.req_rate / m.slo) / m.model_size):
            p = _greedy(gpu_num, models, key, resulting)
            mk = _max_kvpr(p)
            if mk < best_kvpr:
                best_kvpr, best = mk, p
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
