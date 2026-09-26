GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Approach: binary search on a KVPR threshold T. For each T, run a greedy
    feasibility check (models sorted by req_rate/slo desc, placed on the GPU
    whose resulting KVPR is smallest while staying <= T). Binary search
    converges to the minimal feasible T. Falls back to pure memory-fit
    greedy if even a huge T is infeasible (memory-bound instances).
    """

    def try_place(T):
        """Greedy placement under threshold T; returns placement or None."""
        placement = {g: [] for g in range(gpu_num)}
        mem = [float(GPU_MEM_SIZE)] * gpu_num
        w = [0.0] * gpu_num
        for m in sorted(models, key=lambda x: x.req_rate / x.slo, reverse=True):
            r = m.req_rate / m.slo
            best, best_kv = None, float("inf")
            for g in range(gpu_num):
                if m.model_size <= mem[g]:
                    kv = (w[g] + r) / (mem[g] - m.model_size)
                    if kv <= T and kv < best_kv:
                        best_kv, best = kv, g
            if best is None:
                return None
            placement[best].append(m)
            w[best] += r
            mem[best] -= m.model_size
        return placement

    def fallback():
        """Memory-only greedy placement (guaranteed if total memory suffices)."""
        placement = {g: [] for g in range(gpu_num)}
        mem = [float(GPU_MEM_SIZE)] * gpu_num
        for m in sorted(models, key=lambda x: -x.model_size):
            best = max(range(gpu_num), key=lambda g: mem[g])
            placement[best].append(m)
            mem[best] -= m.model_size
        return placement

    lo, hi = 0.0, 1e9
    best = try_place(hi)
    if best is None:
        return fallback()
    for _ in range(60):
        mid = (lo + hi) / 2
        p = try_place(mid)
        if p is not None:
            hi, best = mid, p
        else:
            lo = mid
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
