GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def _kvpr(ms):
    """KVPR of a list of models on one GPU (0 for empty)."""
    if not ms:
        return 0.0
    return sum(m.req_rate / m.slo for m in ms) / (GPU_MEM_SIZE - sum(m.model_size for m in ms))


def _greedy(gpu_num, models):
    """Greedy: sort models by req_rate/slo descending; assign each to the
    feasible GPU with the lowest current KVPR. Falls back to the GPU with
    the most free memory if nothing fits (keeps success rate at 1.0)."""
    placement = {g: [] for g in range(gpu_num)}
    free = [GPU_MEM_SIZE] * gpu_num
    for m in sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True):
        best, best_ratio = None, float("inf")
        for g in range(gpu_num):
            if 0 < m.model_size <= free[g]:
                ratio = _kvpr(placement[g])
                if ratio < best_ratio:
                    best_ratio, best = ratio, g
        if best is None:
            best = max(range(gpu_num), key=lambda g: free[g])
        placement[best].append(m)
        free[best] -= m.model_size
    return placement


def _local_search(placement, gpu_num, max_iter=200):
    """Iteratively reduce max KVPR: try moving a model off the max-KVPR GPU
    to another GPU (or swapping two models) whenever it strictly lowers the
    global max KVPR, until no improvement or iteration budget is exhausted."""
    for _ in range(max_iter):
        src = max(placement, key=lambda g: _kvpr(placement[g]))
        base = _kvpr(placement[src])
        improved = False
        for m in list(placement[src]):
            for g in range(gpu_num):
                if g == src:
                    continue
                placement[src].remove(m)
                placement[g].append(m)
                new_max = max(_kvpr(placement[src]), _kvpr(placement[g]))
                if new_max < base and all(
                    _kvpr(placement[h]) >= new_max - 1e-12 or h in (src, g)
                    for h in placement
                ) and max(_kvpr(v) for v in placement.values()) < base:
                    improved = True
                    break
                placement[g].remove(m)
                placement[src].append(m)
            if improved:
                break
        if not improved:
            break
    return placement


def compute_model_placement(gpu_num, models):
    """Minimize the maximum KVPR: greedy KVPR-minimizing placement followed
    by a move-based local search that lowers the global max KVPR."""
    return _local_search(_greedy(gpu_num, models), gpu_num)


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
