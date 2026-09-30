GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def _greedy(gpu_num, models, key):
    """Greedy placement: process models in `key` order (descending), assign each
    to the GPU minimizing the *resulting* KVPR; fall back to the GPU with most
    free memory if nothing fits."""
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


def _max_kvpr(placement):
    """Maximum KVPR across all GPUs of a placement (0 if all empty)."""
    return max(
        (sum(m.req_rate / m.slo for m in ms)
         / (GPU_MEM_SIZE - sum(m.model_size for m in ms)))
        for ms in placement.values() if ms
    ) if any(placement.values()) else 0.0


def _local_search(placement, gpu_num):
    """Move-based local search on the hottest (max-KVPR) GPU: try moving one
    model to another GPU; accept only strict improvements. State is recomputed
    from the placement each round, so it cannot corrupt or fail."""
    for _ in range(100):
        cur = _max_kvpr(placement)
        free = {g: GPU_MEM_SIZE - sum(m.model_size for m in ms)
                for g, ms in placement.items()}
        hot = max(placement, key=lambda g: (
            sum(m.req_rate / m.slo for m in placement[g]) / max(free[g], 1e-9)
            if placement[g] else -1.0))
        improved = False
        for m in list(placement[hot]):
            for g in placement:
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
    """Minimize max KVPR: run the greedy KVPR heuristic under several sort keys,
    refine the best placement with a move-based local search, and return the
    placement with the smallest maximum KVPR."""
    best, best_kvpr = None, float("inf")
    for key in (
        lambda m: m.req_rate / m.slo,
        lambda m: m.model_size,
        lambda m: (m.req_rate / m.slo) / m.model_size,
        lambda m: -m.model_size,
    ):
        p = _greedy(gpu_num, models, key)
        mk = _max_kvpr(p)
        if mk < best_kvpr:
            best_kvpr, best = mk, p
    p = _local_search(best, gpu_num)
    mk = _max_kvpr(p)
    if mk < best_kvpr:
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
