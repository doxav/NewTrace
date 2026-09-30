GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def _greedy(gpu_num, models, key, resulting):
    """Greedy placement: sort models by `key` descending; assign each model to
    a feasible GPU minimizing the resulting KVPR (if `resulting`) or the
    current KVPR (otherwise). If no GPU can fit the model, fall back to the
    GPU with the most free memory so a placement is always produced."""
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
            best = max(range(gpu_num), key=lambda g: free[g])
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
    """Local search on the hottest (max-KVPR) GPU: try moving one model to
    another GPU, or swapping it with a model there; accept strict improvements
    of the global max KVPR only."""
    for _ in range(40):
        cur = _max_kvpr(placement)
        free = [GPU_MEM_SIZE - sum(m.model_size for m in placement[g]) for g in range(gpu_num)]
        load = [sum(m.req_rate / m.slo for m in placement[g]) for g in range(gpu_num)]
        hot = max(range(gpu_num), key=lambda g: load[g] / max(free[g], 1e-9))
        improved = False
        for m in list(placement[hot]):
            for g in range(gpu_num):
                if g == hot or m.model_size > free[g]:
                    continue
                placement[hot].remove(m)
                placement[g].append(m)
                if _max_kvpr(placement) < cur - 1e-12:
                    improved = True
                    break
                placement[g].remove(m)
                placement[hot].append(m)
                for b in list(placement[g]):
                    if m.model_size - b.model_size > free[g] or b.model_size - m.model_size > free[hot]:
                        continue
                    placement[g].remove(b)
                    placement[hot].append(b)
                    if _max_kvpr(placement) < cur - 1e-12:
                        improved = True
                        break
                    placement[hot].remove(b)
                    placement[g].append(b)
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return placement


def compute_model_placement(gpu_num, models):
    """Minimize max KVPR: run two greedy heuristics (resulting-KVPR and
    current-KVPR) under several sort keys, refine each with a move/swap local
    search, and return the placement with the smallest maximum KVPR."""
    r = lambda m: m.req_rate / m.slo
    s = lambda m: m.model_size
    keys = (r, s, lambda m: r(m) / s(m), lambda m: (r(m), s(m)),
            lambda m: s(m) / (r(m) + 1e-9), lambda m: -r(m))
    best, best_kvpr = None, float("inf")
    for resulting in (True, False):
        for key in keys:
            p = _local_search(_greedy(gpu_num, models, key, resulting), gpu_num)
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
