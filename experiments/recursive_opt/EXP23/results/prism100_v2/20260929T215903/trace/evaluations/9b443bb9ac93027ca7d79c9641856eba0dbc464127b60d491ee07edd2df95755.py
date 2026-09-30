GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize the maximum KVPR across GPUs via binary search on the answer:
    for a threshold T, check if models can be packed so every GPU satisfies
    sum(load) <= T * (80 - sum(size)) with sum(size) <= 80.
    """
    if not models:
        return {g: [] for g in range(gpu_num)}

    req = [m.req_rate / m.slo for m in models]
    size = [m.model_size for m in models]
    n = len(models)

    def pack(order, T):
        """Greedy packing under threshold T; returns placement or None."""
        placement = {g: [] for g in range(gpu_num)}
        load = [0.0] * gpu_num
        used = [0.0] * gpu_num
        for i in order:
            best_g, best_key = None, None
            for g in range(gpu_num):
                free = GPU_MEM_SIZE - used[g] - size[i]
                if free < 0:
                    continue
                if load[g] + req[i] <= T * free + 1e-12:
                    # prefer the tightest fit (least remaining capacity)
                    key = T * free - (load[g] + req[i])
                    if best_key is None or key < best_key:
                        best_key, best_g = key, g
            if best_g is None:
                return None
            placement[best_g].append(models[i])
            load[best_g] += req[i]
            used[best_g] += size[i]
        return placement

    def max_kvpr(placement):
        best = 0.0
        for ms in placement.values():
            denom = GPU_MEM_SIZE - sum(m.model_size for m in ms)
            if denom <= 0:
                return float("inf")
            best = max(best, sum(m.req_rate / m.slo for m in ms) / denom)
        return best

    def polish(placement):
        """First-improvement moves targeting the max-KVPR GPU."""
        placement = {g: list(ms) for g, ms in placement.items()}
        loads = [sum(m.req_rate / m.slo for m in placement[g]) for g in range(gpu_num)]
        used = [sum(m.model_size for m in placement[g]) for g in range(gpu_num)]
        idx = {id(m): i for i, m in enumerate(models)}

        def score():
            return max(loads[g] / (GPU_MEM_SIZE - used[g]) for g in range(gpu_num))

        best = score()
        for _ in range(200):
            src = max(range(gpu_num), key=lambda g: loads[g] / (GPU_MEM_SIZE - used[g]))
            improved = False
            for mi in range(len(placement[src])):
                i = idx[id(placement[src][mi])]
                for dst in range(gpu_num):
                    if dst == src or used[dst] + size[i] > GPU_MEM_SIZE:
                        continue
                    loads[src] -= req[i]; used[src] -= size[i]
                    loads[dst] += req[i]; used[dst] += size[i]
                    s = score()
                    if s < best - 1e-12:
                        placement[dst].append(placement[src].pop(mi))
                        best = s
                        improved = True
                        break
                    loads[src] += req[i]; used[src] += size[i]
                    loads[dst] -= req[i]; used[dst] -= size[i]
                if improved:
                    break
            if not improved:
                break
        return placement

    orders = [
        sorted(range(n), key=lambda i: size[i], reverse=True),
        sorted(range(n), key=lambda i: req[i], reverse=True),
        sorted(range(n), key=lambda i: req[i] / size[i], reverse=True),
        list(range(n)),
    ]

    def feasible(T):
        for order in orders:
            p = pack(order, T)
            if p is not None:
                return p
        return None

    # Binary search on threshold T
    total_req = sum(req)
    total_size = sum(size)
    lo = total_req / max(gpu_num * GPU_MEM_SIZE - total_size, 1e-9)
    hi = max(lo * 4.0, 1.0)
    best_p = feasible(hi)
    if best_p is None:
        # Last resort: place each model somewhere it fits (or most-free GPU)
        placement = {g: [] for g in range(gpu_num)}
        free = [float(GPU_MEM_SIZE)] * gpu_num
        for m in sorted(models, key=lambda m: m.model_size, reverse=True):
            g = max(range(gpu_num), key=lambda x: free[x])
            placement[g].append(m)
            free[g] -= m.model_size
        return placement

    for _ in range(40):
        mid = (lo + hi) / 2.0
        p = feasible(mid)
        if p is not None:
            best_p = p
            hi = mid
        else:
            lo = mid

    best_p = polish(best_p)
    return best_p


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
