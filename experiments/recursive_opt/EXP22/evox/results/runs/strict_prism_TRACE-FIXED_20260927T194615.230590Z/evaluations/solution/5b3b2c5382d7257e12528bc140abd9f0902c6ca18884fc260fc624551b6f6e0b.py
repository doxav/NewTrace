GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def _greedy(gpu_num, models, key, resulting=True):
    """Greedy placement: sort models by `key` descending; assign each model to
    a feasible GPU minimizing the resulting KVPR (if `resulting`) or the
    current KVPR (otherwise). Strictly requires the model to fit; raises
    ValueError if it fits nowhere, so memory is never overcommitted."""
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
            raise ValueError(f"Model of size {m.model_size} GB fits no GPU")
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


def _local_search(placement, gpu_num):
    """Local search on the hottest (max-KVPR) GPU: try moving one model out or
    swapping it with a model on another GPU; accept only strict improvements.
    State (free memory) is recomputed each round, so it cannot corrupt."""
    for _ in range(60):
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
            done = False
            for g in range(gpu_num):
                if g == hot:
                    continue
                for b in list(placement[g]):
                    if m.model_size - b.model_size <= free[g] and b.model_size - m.model_size <= free[hot]:
                        placement[hot].remove(m)
                        placement[g].remove(b)
                        placement[hot].append(b)
                        placement[g].append(m)
                        if _max_kvpr(placement) < cur - 1e-12:
                            done = improved = True
                            break
                        placement[g].remove(m)
                        placement[hot].remove(b)
                        placement[hot].append(m)
                        placement[g].append(b)
                if done:
                    break
            if done:
                break
        if not improved:
            break
    return placement


def compute_model_placement(gpu_num, models):
    """Minimize max KVPR: run the strict resulting-KVPR greedy under a few
    sort keys, refine the best candidate with a move/swap local search, and
    return the best placement. If every strict attempt fails (a model fits
    nowhere), fall back to a most-free-memory-first assignment."""
    r = lambda m: m.req_rate / m.slo
    s = lambda m: m.model_size
    keys = (r, s, lambda m: r(m) / s(m))
    best, best_kvpr = None, float("inf")
    for key in keys:
        try:
            p = _greedy(gpu_num, models, key)
        except ValueError:
            continue
        mk = _max_kvpr(p)
        if mk < best_kvpr:
            best_kvpr, best = mk, p
    if best is None:  # fallback: place largest models on freest GPUs
        placement = {g: [] for g in range(gpu_num)}
        free = [GPU_MEM_SIZE] * gpu_num
        for m in sorted(models, key=s, reverse=True):
            g = max(range(gpu_num), key=lambda g: free[g])
            placement[g].append(m)
            free[g] -= m.model_size
        return placement
    if gpu_num > 1:
        best = _local_search(best, gpu_num)
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
