GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize the maximum KVPR via binary search on the threshold T.

    A GPU hosting set S is T-feasible iff
        sum(req/slo over S) <= T * (80 - sum(size over S)) and sum(size) <= 80.
    Binary search T; for each T run greedy feasibility with several orderings,
    then polish the best placement with move/swap local search.
    """
    import time

    start = time.time()
    n = len(models)
    if n == 0:
        return {g: [] for g in range(gpu_num)}

    req = [m.req_rate / m.slo for m in models]
    size = [m.model_size for m in models]

    def try_pack(order, T):
        """Greedy feasibility check for threshold T; returns placement or None."""
        placement = {g: [] for g in range(gpu_num)}
        load = [0.0] * gpu_num
        used = [0.0] * gpu_num
        for i in order:
            best_g, best_slack = None, None
            for g in range(gpu_num):
                new_used = used[g] + size[i]
                if new_used > GPU_MEM_SIZE:
                    continue
                cap = T * (GPU_MEM_SIZE - new_used)
                new_load = load[g] + req[i]
                if new_load <= cap + 1e-12:
                    slack = cap - new_load
                    if best_slack is None or slack > best_slack:
                        best_slack, best_g = slack, g
            if best_g is None:
                return None
            placement[best_g].append(models[i])
            load[best_g] += req[i]
            used[best_g] += size[i]
        return placement

    orders = [
        sorted(range(n), key=lambda i: req[i] / max(size[i], 1e-9), reverse=True),
        sorted(range(n), key=lambda i: size[i], reverse=True),
        sorted(range(n), key=lambda i: req[i], reverse=True),
        sorted(range(n), key=lambda i: req[i] / max(size[i], 1e-9)),
    ]

    def feasible_for_T(T):
        for order in orders:
            p = try_pack(order, T)
            if p is not None:
                return p
        return None

    def max_kvpr(placement):
        best = 0.0
        for g in range(gpu_num):
            denom = GPU_MEM_SIZE - sum(m.model_size for m in placement[g])
            if denom <= 0:
                return float("inf")
            kvpr = sum(m.req_rate / m.slo for m in placement[g]) / denom
            if kvpr > best:
                best = kvpr
        return best

    def ffd_placement():
        """Memory-feasibility baseline: first-fit-decreasing by size."""
        placement = {g: [] for g in range(gpu_num)}
        free = [GPU_MEM_SIZE] * gpu_num
        for i in sorted(range(n), key=lambda i: size[i], reverse=True):
            for g in range(gpu_num):
                if size[i] <= free[g]:
                    placement[g].append(models[i])
                    free[g] -= size[i]
                    break
            else:
                return None
        return placement

    # Binary search on threshold T.
    lo = 0.0
    hi = max(req) / max(1e-9, GPU_MEM_SIZE - max(size)) * (gpu_num + 1) + 1.0
    best_placement = None
    for _ in range(50):
        mid = (lo + hi) / 2.0
        p = feasible_for_T(mid)
        if p is not None:
            best_placement = p
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-9 or time.time() - start > 4.0:
            break

    # Fallbacks to guarantee a valid placement.
    if best_placement is None:
        best_placement = ffd_placement()
    if best_placement is None:
        best_placement = {g: [] for g in range(gpu_num)}
        for i in sorted(range(n), key=lambda i: size[i], reverse=True):
            g = min(range(gpu_num), key=lambda x: sum(m.model_size for m in best_placement[x]))
            best_placement[g].append(models[i])

    # Local-search polish: best-improvement moves and swaps reducing max KVPR.
    load = [sum(m.req_rate / m.slo for m in best_placement[g]) for g in range(gpu_num)]
    used = [sum(m.model_size for m in best_placement[g]) for g in range(gpu_num)]

    def kvpr_from():
        best = 0.0
        for g in range(gpu_num):
            d = GPU_MEM_SIZE - used[g]
            v = load[g] / d if d > 1e-12 else float("inf")
            if v > best:
                best = v
        return best

    best = kvpr_from()
    improved = True
    while improved and time.time() - start < 7.0:
        improved = False
        best_action = None
        for src in range(gpu_num):
            for mi in range(len(best_placement[src])):
                m = best_placement[src][mi]
                i = models.index(m)
                for dst in range(gpu_num):
                    if dst == src or used[dst] + size[i] > GPU_MEM_SIZE:
                        continue
                    # try move
                    load[src] -= req[i]; used[src] -= size[i]
                    load[dst] += req[i]; used[dst] += size[i]
                    s = kvpr_from()
                    if s < best - 1e-12 and (best_action is None or s < best_action[0]):
                        best_action = (s, src, mi, dst)
                    load[src] += req[i]; used[src] += size[i]
                    load[dst] -= req[i]; used[dst] -= size[i]
                    # try swaps
                    for dj in range(len(best_placement[dst])):
                        m2 = best_placement[dst][dj]
                        j = models.index(m2)
                        if used[dst] - size[j] + size[i] > GPU_MEM_SIZE:
                            continue
                        if used[src] - size[i] + size[j] > GPU_MEM_SIZE:
                            continue
                        load[src] += req[j] - req[i]; used[src] += size[j] - size[i]
                        load[dst] += req[i] - req[j]; used[dst] += size[i] - size[j]
                        s = kvpr_from()
                        if s < best - 1e-12 and (best_action is None or s < best_action[0]):
                            best_action = (s, src, mi, dst, dj)
                        load[src] -= req[j] - req[i]; used[src] -= size[j] - size[i]
                        load[dst] -= req[i] - req[j]; used[dst] -= size[i] - size[j]
        if best_action:
            if len(best_action) == 4:
                s, src, mi, dst = best_action
                m = best_placement[src].pop(mi)
                best_placement[dst].append(m)
                i = models.index(m)
                load[src] -= req[i]; used[src] -= size[i]
                load[dst] += req[i]; used[dst] += size[i]
            else:
                s, src, mi, dst, dj = best_action
                m1, m2 = best_placement[src][mi], best_placement[dst][dj]
                i1, i2 = models.index(m1), models.index(m2)
                best_placement[src][mi], best_placement[dst][dj] = m2, m1
                load[src] += req[i2] - req[i1]; used[src] += size[i2] - size[i1]
                load[dst] += req[i1] - req[i2]; used[dst] += size[i1] - size[i2]
            best = s
            improved = True

    return best_placement


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
