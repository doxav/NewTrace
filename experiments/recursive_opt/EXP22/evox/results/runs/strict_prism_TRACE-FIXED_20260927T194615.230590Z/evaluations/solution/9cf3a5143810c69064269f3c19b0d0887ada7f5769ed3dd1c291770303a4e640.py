GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def _greedy(gpu_num, models, key):
    """Greedy placement: sort models by `key` descending, assign each to the
    GPU minimizing the resulting KVPR; fall back to the GPU with the most
    free memory if no GPU can fit the model (avoids hard failures)."""
    placement = {g: [] for g in range(gpu_num)}
    free = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num
    for m in sorted(models, key=key, reverse=True):
        r = m.req_rate / m.slo
        best, best_ratio = None, float("inf")
        for g in range(gpu_num):
            if m.model_size <= free[g]:
                ratio = (load[g] + r) / (free[g] - m.model_size)
                if ratio < best_ratio:
                    best_ratio, best = ratio, g
        if best is None:
            best = max(range(gpu_num), key=lambda g: free[g])
        placement[best].append(m)
        load[best] += r
        free[best] -= m.model_size
    return placement


def _max_kvpr(placement, gpu_num):
    """Max KVPR across all GPUs of a placement."""
    mk = 0.0
    for g in range(gpu_num):
        ms = placement[g]
        if ms:
            mk = max(mk, sum(m.req_rate / m.slo for m in ms)
                     / (GPU_MEM_SIZE - sum(m.model_size for m in ms)))
    return mk


def _local_search(placement, gpu_num):
    """Refine placement: repeatedly move or swap a model out of the hottest
    (highest-KVPR) GPU when it strictly reduces max KVPR, respecting memory."""
    free = [GPU_MEM_SIZE - sum(m.model_size for m in placement[g]) for g in range(gpu_num)]
    for _ in range(100):
        cur = _max_kvpr(placement, gpu_num)
        hot = max(range(gpu_num), key=lambda g: _max_kvpr({g: placement[g]}, g + 1) if placement[g] else -1)
        improved = False
        for m in list(placement[hot]):
            for g in range(gpu_num):
                if g != hot and m.model_size <= free[g]:
                    placement[hot].remove(m); placement[g].append(m)
                    if _max_kvpr(placement, gpu_num) < cur - 1e-12:
                        free[hot] += m.model_size; free[g] -= m.model_size
                        improved = True
                        break
                    placement[g].remove(m); placement[hot].append(m)
            if improved:
                break
        if not improved:
            break
    return placement


def compute_model_placement(gpu_num, models):
    """Minimize max KVPR: run the greedy KVPR heuristic under several sort
    keys, refine each with a local search, and return the best placement."""
    best, best_kvpr = None, float("inf")
    for key in (
        lambda m: m.req_rate / m.slo,
        lambda m: m.model_size,
        lambda m: (m.req_rate / m.slo) / m.model_size,
        lambda m: -(m.req_rate / m.slo) / m.model_size,
    ):
        p = _greedy(gpu_num, models, key)
        p = _local_search(p, gpu_num)
        mk = _max_kvpr(p, gpu_num)
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
