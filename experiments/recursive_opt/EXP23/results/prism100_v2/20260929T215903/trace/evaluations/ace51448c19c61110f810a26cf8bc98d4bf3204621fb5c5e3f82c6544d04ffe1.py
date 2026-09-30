GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Approach: binary search on the answer (target KVPR threshold T) with a
    greedy feasibility check. For a target T, a GPU can host a set S iff
    sum(req/slo) <= T * (80 - sum(size)). We try to pack all models under
    that constraint using several orderings; binary search shrinks T.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """
    n = len(models)
    if n == 0:
        return {g: [] for g in range(gpu_num)}

    req = [m.req_rate / m.slo for m in models]
    size = [m.model_size for m in models]

    def try_pack(order, T):
        """Greedily pack models so every GPU's KVPR stays <= T.
        Returns placement dict or None if it fails."""
        placement = {g: [] for g in range(gpu_num)}
        used = [0.0] * gpu_num
        load = [0.0] * gpu_num
        for i in order:
            best_g = None
            best_kv = float("inf")
            for g in range(gpu_num):
                if used[g] + size[i] <= GPU_MEM_SIZE and \
                        load[g] + req[i] <= T * (GPU_MEM_SIZE - used[g] - size[i]) + 1e-12:
                    kv = (load[g] + req[i]) / (GPU_MEM_SIZE - used[g] - size[i])
                    if kv < best_kv:
                        best_kv = kv
                        best_g = g
            if best_g is None:
                return None
            placement[best_g].append(models[i])
            used[best_g] += size[i]
            load[best_g] += req[i]
        return placement

    # Orderings for the feasibility check (most constrained first).
    orders = [
        sorted(range(n), key=lambda i: size[i], reverse=True),
        sorted(range(n), key=lambda i: req[i] / max(size[i], 1e-9), reverse=True),
        sorted(range(n), key=lambda i: req[i], reverse=True),
        list(range(n)),
    ]

    def feasible(T):
        for order in orders:
            p = try_pack(order, T)
            if p is not None:
                return p
        return None

    # Initial bounds: lower bound from total load spread evenly,
    # upper bound from the worst single-model KVPR (always feasible).
    total_req = sum(req)
    lo = total_req / (gpu_num * GPU_MEM_SIZE)
    hi = max((req[i] + 1e-9) / max(GPU_MEM_SIZE - size[i], 1e-9) for i in range(n))
    hi = max(hi * 2.0, lo * 4.0, 1e-6)

    best = None
    for _ in range(40):
        mid = (lo + hi) / 2.0
        p = feasible(mid)
        if p is not None:
            best = p
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-9:
            break

    if best is None:
        # Fallback: greedy KVPR minimization without threshold (always places).
        placement = {g: [] for g in range(gpu_num)}
        used = [0.0] * gpu_num
        for i in sorted(range(n), key=lambda i: size[i], reverse=True):
            best_g = max(range(gpu_num), key=lambda g: GPU_MEM_SIZE - used[g])
            placement[best_g].append(models[i])
            used[best_g] += size[i]
        best = placement
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
