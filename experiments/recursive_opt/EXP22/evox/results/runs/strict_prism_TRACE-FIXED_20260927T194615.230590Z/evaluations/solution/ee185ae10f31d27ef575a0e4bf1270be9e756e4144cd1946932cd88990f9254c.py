GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def _greedy(gpu_num, models, key, resulting):
    """Greedy placement: sort models by `key` descending; assign each model
    to the feasible GPU minimizing the current KVPR (load/free) if `resulting`
    is False, else the resulting KVPR ((load+r)/(free-size)). Falls back to
    the GPU with the most free memory if nothing fits."""
    placement = {g: [] for g in range(gpu_num)}
    shared_kv = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num
    for m in sorted(models, key=key, reverse=True):
        r = m.req_rate / m.slo
        best_idx, best_ratio = None, float("inf")
        for g in range(gpu_num):
            if m.model_size <= shared_kv[g] > 0:
                ratio = (load[g] + r) / (shared_kv[g] - m.model_size) if resulting \
                    else load[g] / shared_kv[g]
                if ratio < best_ratio:
                    best_ratio, best_idx = ratio, g
        if best_idx is None:
            best_idx = max(range(gpu_num), key=lambda g: shared_kv[g])
        placement[best_idx].append(m)
        load[best_idx] += r
        shared_kv[best_idx] -= m.model_size
    return placement


def _max_kvpr(placement):
    """Return the maximum KVPR across all GPUs of a placement."""
    return max(
        (sum(m.req_rate / m.slo for m in ms) / (GPU_MEM_SIZE - sum(m.model_size for m in ms))
         for ms in placement.values() if ms),
        default=0.0,
    )


def _local_search(placement, gpu_num):
    """Move-only local search: repeatedly try moving one model off the
    hottest (max-KVPR) GPU to another feasible GPU; accept only strict
    improvements. Free memory is recomputed each round, so state stays valid."""
    for _ in range(80):
        cur = _max_kvpr(placement)
        free = [GPU_MEM_SIZE - sum(m.model_size for m in placement[g]) for g in range(gpu_num)]
        load = [sum(m.req_rate / m.slo for m in placement[g]) for g in range(gpu_num)]
        hot = max(range(gpu_num), key=lambda g: load[g] / max(free[g], 1e-9))
        improved = False
        for m in list(placement[hot]):
            for g in range(gpu_num):
                if g != hot and m.model_size <= free[g]:
                    placement[hot].remove(m)
                    placement[g].append(m)
                    if _max_kvpr(placement) < cur - 1e-12:
                        improved = True
                        break
                    placement[g].remove(m)
                    placement[hot].append(m)
            if improved:
                break
        if not improved:
            break
    return placement


def compute_model_placement(gpu_num, models):
    """Minimize max KVPR: run both greedy variants (current-KVPR and
    resulting-KVPR) under three sort keys, pick the best placement, then
    refine it with a cheap move-only local search."""
    best, best_kvpr = None, float("inf")
    for resulting in (False, True):
        for key in (lambda m: m.req_rate / m.slo, lambda m: m.model_size,
                    lambda m: (m.req_rate / m.slo) / m.model_size):
            p = _greedy(gpu_num, models, key, resulting)
            mk = _max_kvpr(p)
            if mk < best_kvpr:
                best_kvpr, best = mk, p
    if gpu_num > 1 and best is not None:
        p = _local_search(best, gpu_num)
        if _max_kvpr(p) < best_kvpr:
            best = p
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
