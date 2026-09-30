GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def _greedy(gpu_num, models, key, resulting):
    """Greedy placement: sort models by `key` descending; assign each model to
    a feasible GPU minimizing the resulting KVPR (if `resulting`) or the
    current KVPR (otherwise). Raises ValueError if a model fits nowhere, so
    placements are always memory-feasible (no overcommitment)."""
    placement = {g: [] for g in range(gpu_num)}
    free = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num
    for m in sorted(models, key=key, reverse=True):
        r = m.req_rate / m.slo
        best, best_ratio = None, float("inf")
        for g in range(gpu_num):
            if m.model_size <= free[g]:
                ratio = (load[g] + r) / (free[g] - m.model_size) if resulting else load[g] / free[g]
                if ratio < best_ratio:
                    best_ratio, best = ratio, g
        if best is None:
            raise ValueError("Model does not fit on any GPU.")
        placement[best].append(m)
        load[best] += r
        free[best] -= m.model_size
    return placement


def _max_kvpr(placement):
    """Maximum KVPR across GPUs of a placement."""
    return max(
        sum(m.req_rate / m.slo for m in ms) / (GPU_MEM_SIZE - sum(m.model_size for m in ms))
        for ms in placement.values()
    ) if any(placement.values()) else 0.0


def compute_model_placement(gpu_num, models):
    """Minimize max KVPR: run two greedy heuristics (resulting-KVPR and
    current-KVPR) under several sort keys, keep only memory-feasible
    placements, and return the one with the smallest maximum KVPR."""
    r = lambda m: m.req_rate / m.slo
    s = lambda m: m.model_size
    keys = (r, s, lambda m: r(m) / s(m), lambda m: (r(m), s(m)),
            lambda m: s(m) / (r(m) + 1e-9), lambda m: -r(m))
    best, best_kvpr = None, float("inf")
    for resulting in (True, False):
        for key in keys:
            try:
                p = _greedy(gpu_num, models, key, resulting)
            except ValueError:
                continue
            mk = _max_kvpr(p)
            if mk < best_kvpr:
                best_kvpr, best = mk, p
    if best is None:
        raise ValueError("No feasible placement found for the given models.")
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
