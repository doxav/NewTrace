GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def _greedy(gpu_num, models, key):
    """Greedy placement: process models in `key` order (descending), assign each
    to the feasible GPU minimizing the *resulting* KVPR. Strictly memory-
    feasible: raises ValueError if a model fits nowhere (never overcommits)."""
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
            raise ValueError("Model does not fit on any GPU.")
        placement[best].append(m)
        load[best] += r
        free[best] -= m.model_size
    return placement


def _max_kvpr(placement, gpu_num):
    """Compute max KVPR over all GPUs of a placement (0 if all empty)."""
    mk = 0.0
    for g in range(gpu_num):
        ms = placement[g]
        if ms:
            mk = max(mk, sum(m.req_rate / m.slo for m in ms)
                     / max(GPU_MEM_SIZE - sum(m.model_size for m in ms), 1e-9))
    return mk


def _local_search(placement, gpu_num):
    """Local search on the hottest (max-KVPR) GPU: try moving one model out or
    swapping it with a model on another GPU; accept only strict improvements.
    All state (free memory) is recomputed each round, so it cannot corrupt."""
    for _ in range(150):
        cur = _max_kvpr(placement, gpu_num)
        free = [GPU_MEM_SIZE - sum(m.model_size for m in placement[g]) for g in range(gpu_num)]
        load = [sum(m.req_rate / m.slo for m in placement[g]) for g in range(gpu_num)]
        hot = max(range(gpu_num), key=lambda g: load[g] / max(free[g], 1e-9))
        improved = False
        for m in list(placement[hot]):
            # try a simple move
            for g in range(gpu_num):
                if g != hot and m.model_size <= free[g]:
                    placement[hot].remove(m)
                    placement[g].append(m)
                    if _max_kvpr(placement, gpu_num) < cur - 1e-12:
                        improved = True
                        break
                    placement[g].remove(m)
                    placement[hot].append(m)
            if improved:
                break
            # try a swap
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
                        if _max_kvpr(placement, gpu_num) < cur - 1e-12:
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
    """Minimize max KVPR: run the greedy resulting-KVPR heuristic under several
    sort keys, refine each with a move/swap local search, then run one final
    local-search pass on the overall best placement; return the placement with
    the smallest maximum KVPR."""
    best, best_kvpr = None, float("inf")
    for key in (
        lambda m: m.req_rate / m.slo,
        lambda m: m.model_size,
        lambda m: (m.req_rate / m.slo) / m.model_size,
        lambda m: m.model_size / (m.req_rate / m.slo + 1e-9),
    ):
        try:
            g = _greedy(gpu_num, models, key)
        except ValueError:
            continue
        p = _local_search(g, gpu_num)
        mk = _max_kvpr(p, gpu_num)
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
