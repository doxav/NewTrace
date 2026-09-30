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


def _max_kvpr(placement, gpu_num):
    """Compute max KVPR over all GPUs of a placement."""
    mk = 0.0
    for g in range(gpu_num):
        ms = placement[g]
        if ms:
            mk = max(mk, sum(m.req_rate / m.slo for m in ms)
                     / (GPU_MEM_SIZE - sum(m.model_size for m in ms)))
    return mk


def _local_search(placement, gpu_num, models):
    """Local search: repeatedly move or swap models between GPUs when it
    strictly reduces the maximum KVPR, respecting memory limits."""
    free = {g: GPU_MEM_SIZE - sum(m.model_size for m in placement[g]) for g in range(gpu_num)}
    for _ in range(200):
        loads = {g: (sum(m.req_rate / m.slo for m in placement[g]),
                     sum(m.model_size for m in placement[g])) for g in range(gpu_num)}
        cur = _max_kvpr(placement, gpu_num)
        hot = max(range(gpu_num), key=lambda g: loads[g][0] / max(loads[g][1], 1e-9) if loads[g][1] else -1)
        improved = False
        # try moving one model out of the hottest GPU
        for m in list(placement[hot]):
            for g in range(gpu_num):
                if g != hot and m.model_size <= free[g]:
                    placement[hot].remove(m)
                    placement[g].append(m)
                    if _max_kvpr(placement, gpu_num) < cur - 1e-12:
                        free[hot] += m.model_size
                        free[g] -= m.model_size
                        improved = True
                        break
                    placement[g].remove(m)
                    placement[hot].append(m)
            if improved:
                break
        if not improved:
            # try swapping pairs between hottest GPU and others
            done = False
            for a in list(placement[hot]):
                for g in range(gpu_num):
                    if g == hot:
                        continue
                    for b in list(placement[g]):
                        if a.model_size - b.model_size <= free[g] and b.model_size - a.model_size <= free[hot]:
                            placement[hot].remove(a); placement[g].remove(b)
                            placement[hot].append(b); placement[g].append(a)
                            if _max_kvpr(placement, gpu_num) < cur - 1e-12:
                                free[hot] += a.model_size - b.model_size
                                free[g] += b.model_size - a.model_size
                                done = True
                                break
                            placement[hot].remove(b); placement[g].remove(a)
                            placement[hot].append(a); placement[g].append(b)
                    if done:
                        break
                if done:
                    break
            if not done:
                break
    return placement


def compute_model_placement(gpu_num, models):
    """Minimize max KVPR: run the greedy KVPR heuristic under several sort keys,
    then refine each with a move/swap local search; return the best placement."""
    best, best_kvpr = None, float("inf")
    for key in (
        lambda m: m.req_rate / m.slo,
        lambda m: m.model_size,
        lambda m: (m.req_rate / m.slo) / m.model_size,
        lambda m: -m.model_size,
    ):
        p = _greedy(gpu_num, models, key)
        p = _local_search(p, gpu_num, models)
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
