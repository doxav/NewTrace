GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize the maximum KVPR across GPUs via binary search on the answer.

    For a target threshold T, a placement is feasible if every GPU g satisfies
        sum(req_rate/slo) <= T * (GPU_MEM_SIZE - sum(model_size))
    We binary-search T and use a randomized greedy packing as the feasibility
    check, keeping the best feasible placement found.
    """
    import random
    import time

    start = time.time()
    rng = random.Random(42)
    n = len(models)
    if n == 0:
        return {g: [] for g in range(gpu_num)}

    req = [m.req_rate / m.slo for m in models]
    size = [m.model_size for m in models]

    def try_pack(order, T, mode):
        """Greedy packing under threshold T. Returns placement or None."""
        placement = {g: [] for g in range(gpu_num)}
        loads = [0.0] * gpu_num
        used = [0.0] * gpu_num
        for i in order:
            s, r = size[i], req[i]
            best_g, best_key = None, None
            for g in range(gpu_num):
                free = GPU_MEM_SIZE - used[g] - s
                if free < 0:
                    continue
                new_load = loads[g] + r
                if new_load > T * free + 1e-12:
                    continue
                # prefer GPU with most remaining slack under T
                key = T * free - new_load
                if mode == 1:
                    key = -free  # prefer fullest-fitting GPU (tight packing)
                if best_key is None or key > best_key:
                    best_key, best_g = key, g
            if best_g is None:
                return None
            placement[best_g].append(models[i])
            loads[best_g] += r
            used[best_g] += s
        return placement

    def kvpr_of(placement):
        best = 0.0
        for g in range(gpu_num):
            denom = GPU_MEM_SIZE - sum(m.model_size for m in placement[g])
            if denom <= 0:
                return float("inf")
            kvpr = sum(m.req_rate / m.slo for m in placement[g]) / denom
            if kvpr > best:
                best = kvpr
        return best

    # Upper bound: single-GPU-style bound (all load / min free memory)
    total_req = sum(req)
    hi = max(kvpr_of({g: [models[i]] for g in range(min(n, gpu_num))}), total_req / GPU_MEM_SIZE)
    hi = max(hi, 1e-6) * 4.0

    # Always-feasible fallback: place each model greedily minimizing KVPR
    def fallback():
        placement = {g: [] for g in range(gpu_num)}
        loads = [0.0] * gpu_num
        used = [0.0] * gpu_num
        for i in sorted(range(n), key=lambda x: size[x], reverse=True):
            best_g, best_v = None, None
            for g in range(gpu_num):
                if used[g] + size[i] > GPU_MEM_SIZE:
                    continue
                v = (loads[g] + req[i]) / (GPU_MEM_SIZE - used[g] - size[i])
                if best_v is None or v < best_v:
                    best_v, best_g = v, g
            if best_g is None:
                best_g = min(range(gpu_num), key=lambda x: used[x])
            placement[best_g].append(models[i])
            loads[best_g] += req[i]
            used[best_g] += size[i]
        return placement

    best_placement = fallback()
    best_score = kvpr_of(best_placement)

    base_orders = [
        sorted(range(n), key=lambda i: size[i], reverse=True),
        sorted(range(n), key=lambda i: req[i], reverse=True),
        sorted(range(n), key=lambda i: req[i] / max(size[i], 1e-9), reverse=True),
        list(range(n)),
    ]

    for attempt in range(200):
        if time.time() - start > 1.5:
            break
        if attempt < len(base_orders):
            order = base_orders[attempt]
            mode = attempt % 2
        else:
            order = list(range(n))
            rng.shuffle(order)
            mode = rng.randrange(2)

        # binary search on T for this order
        lo, hi_t = 0.0, hi
        feasible_T, feasible_P = None, None
        for _ in range(25):
            mid = (lo + hi_t) / 2.0
            p = try_pack(order, mid, mode)
            if p is not None:
                hi_t = mid
                feasible_T, feasible_P = mid, p
            else:
                lo = mid
        if feasible_P is not None:
            score = kvpr_of(feasible_P)
            if score < best_score - 1e-12:
                best_score = score
                best_placement = feasible_P

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
