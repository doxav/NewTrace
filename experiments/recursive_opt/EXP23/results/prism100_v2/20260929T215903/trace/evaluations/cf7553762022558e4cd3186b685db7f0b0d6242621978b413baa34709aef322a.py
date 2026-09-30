GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize the maximum KVPR across GPUs via binary search on the answer:
    for a threshold T, check whether models can be packed so every GPU has
    KVPR <= T, using best-fit-decreasing under both memory and load caps.
    The best feasible packing is then refined with move/swap local search.
    """
    import random

    if not models:
        return {g: [] for g in range(gpu_num)}

    req = [m.req_rate / m.slo for m in models]
    size = [m.model_size for m in models]
    n = len(models)

    def pack(T, order_key):
        """Greedy packing under threshold T. Returns placement or None."""
        placement = {g: [] for g in range(gpu_num)}
        used = [0.0] * gpu_num
        load = [0.0] * gpu_num
        order = sorted(range(n), key=order_key, reverse=True)
        for i in order:
            best_g, best_metric = None, None
            for g in range(gpu_num):
                if used[g] + size[i] > GPU_MEM_SIZE:
                    continue
                if load[g] + req[i] > T * (GPU_MEM_SIZE - used[g] - size[i]) + 1e-9:
                    continue
                slack = T * (GPU_MEM_SIZE - used[g] - size[i]) - (load[g] + req[i])
                # best-fit: leave the least slack
                if best_metric is None or slack < best_metric:
                    best_metric, best_g = slack, g
            if best_g is None:
                return None
            placement[best_g].append(models[i])
            used[best_g] += size[i]
            load[best_g] += req[i]
        return placement

    def max_kvpr(placement):
        best = 0.0
        for ms in placement.values():
            denom = GPU_MEM_SIZE - sum(m.model_size for m in ms)
            if denom <= 0:
                return float("inf")
            best = max(best, sum(m.req_rate / m.slo for m in ms) / denom)
        return best

    def refine(placement, rng, rounds=40):
        """First-improvement moves and swaps to lower max KVPR."""
        placement = {g: list(ms) for g, ms in placement.items()}
        best = max_kvpr(placement)
        for _ in range(rounds):
            improved = False
            for src in range(gpu_num):
                for mi in range(len(placement[src])):
                    m1 = placement[src][mi]
                    i1 = models.index(m1)
                    for dst in range(gpu_num):
                        if dst == src:
                            continue
                        d_used = sum(m.model_size for m in placement[dst])
                        # try move
                        if d_used + size[i1] <= GPU_MEM_SIZE:
                            placement[src].pop(mi)
                            placement[dst].append(m1)
                            s = max_kvpr(placement)
                            if s < best - 1e-12:
                                best, improved = s, True
                                break
                            placement[dst].pop()
                            placement[src].insert(mi, m1)
                        # try swap
                        for mj in range(len(placement[dst])):
                            m2 = placement[dst][mj]
                            i2 = models.index(m2)
                            s_used = sum(m.model_size for m in placement[src])
                            if s_used - size[i1] + size[i2] > GPU_MEM_SIZE:
                                continue
                            if d_used - size[i2] + size[i1] > GPU_MEM_SIZE:
                                continue
                            placement[src][mi], placement[dst][mj] = m2, m1
                            s = max_kvpr(placement)
                            if s < best - 1e-12:
                                best, improved = s, True
                                break
                            placement[src][mi], placement[dst][mj] = m1, m2
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return placement

    # Lower bound: total load spread over all free memory
    total_load = sum(req)
    total_size = sum(size)
    lo = total_load / max(GPU_MEM_SIZE * gpu_num - total_size, 1e-9)
    hi = max_kvpr({g: models for g in range(gpu_num)} if gpu_num == 1
                  else {0: models, **{g: [] for g in range(1, gpu_num)}})
    hi = max(hi, lo) * 2 + 1.0

    order_keys = [
        lambda i: size[i],
        lambda i: req[i],
        lambda i: req[i] / max(size[i], 1e-9),
    ]

    best_placement = None
    best_score = float("inf")
    rng = random.Random(42)

    for _ in range(30):
        mid = (lo + hi) / 2.0
        found = None
        for key in order_keys:
            found = pack(mid, key)
            if found is not None:
                break
        if found is not None:
            best_placement, best_score = found, max_kvpr(found)
            hi = mid
        else:
            if lo >= mid:
                break
            lo = mid
        if hi - lo < 1e-4:
            break

    # Fallback: simple FFD to guarantee a valid placement
    if best_placement is None:
        placement = {g: [] for g in range(gpu_num)}
        used = [0.0] * gpu_num
        for i in sorted(range(n), key=lambda i: size[i], reverse=True):
            g = min(range(gpu_num), key=lambda x: used[x])
            placement[g].append(models[i])
            used[g] += size[i]
        best_placement = placement
        best_score = max_kvpr(placement)

    # Local search refinement, keeping the best result
    refined = refine(best_placement, rng)
    s = max_kvpr(refined)
    if s < best_score:
        best_placement, best_score = refined, s

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
