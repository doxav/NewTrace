GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize the maximum KVPR across GPUs via binary search on the answer.

    For a target threshold T, a GPU hosting a set S is feasible iff
    sum(req_rate/slo over S) <= T * (GPU_MEM_SIZE - sum(model_size over S))
    and sum(model_size over S) <= GPU_MEM_SIZE. We binary search T and use
    a greedy feasibility check with several orderings; keep the best
    achieved (actual) max-KVPR placement.
    """
    import time

    start = time.time()
    n = len(models)
    if n == 0:
        return {g: [] for g in range(gpu_num)}

    req = [m.req_rate / m.slo for m in models]
    size = [m.model_size for m in models]

    def actual_max_kvpr(assign):
        best = 0.0
        for g in range(gpu_num):
            used = 0.0
            load = 0.0
            for i in assign[g]:
                used += size[i]
                load += req[i]
            denom = GPU_MEM_SIZE - used
            if denom <= 0:
                return float("inf")
            kvpr = load / denom
            if kvpr > best:
                best = kvpr
        return best

    def check(T, order):
        """Greedy feasibility for threshold T with given model order.
        Returns assignment (list of index-lists per GPU) or None."""
        assign = [[] for _ in range(gpu_num)]
        used = [0.0] * gpu_num
        load = [0.0] * gpu_num
        for i in order:
            placed = False
            best_g = -1
            best_key = None
            for g in range(gpu_num):
                new_used = used[g] + size[i]
                if new_used > GPU_MEM_SIZE:
                    continue
                new_load = load[g] + req[i]
                # capacity constraint for threshold T
                if new_load > T * (GPU_MEM_SIZE - new_used) + 1e-12:
                    continue
                # prefer tightest memory fit to preserve flexibility
                key = GPU_MEM_SIZE - new_used
                if best_key is None or key < best_key:
                    best_key = key
                    best_g = g
            if best_g < 0:
                return None
            assign[best_g].append(i)
            used[best_g] += size[i]
            load[best_g] += req[i]
        return assign

    # Orderings for the feasibility check (largest / heaviest first)
    orders = [
        sorted(range(n), key=lambda i: size[i], reverse=True),
        sorted(range(n), key=lambda i: req[i], reverse=True),
        sorted(range(n), key=lambda i: req[i] / max(size[i], 1e-9), reverse=True),
        sorted(range(n), key=lambda i: req[i] / max(GPU_MEM_SIZE - size[i], 1e-9), reverse=True),
        list(range(n)),
    ]

    def check_any(T):
        best_assign = None
        best_score = float("inf")
        for order in orders:
            a = check(T, order)
            if a is not None:
                s = actual_max_kvpr(a)
                if s < best_score:
                    best_score = s
                    best_assign = a
                if time.time() - start > 6.0:
                    break
        return best_assign, best_score

    # Upper bound: place everything greedily ignoring T (most free memory)
    assign = [[] for _ in range(gpu_num)]
    used = [0.0] * gpu_num
    load = [0.0] * gpu_num
    for i in sorted(range(n), key=lambda i: size[i], reverse=True):
        g = min(range(gpu_num), key=lambda x: used[x])
        assign[g].append(i)
        used[g] += size[i]
        load[g] += req[i]
    hi = max(actual_max_kvpr(assign), 1e-9)
    lo = 0.0
    best_assign = assign
    best_score = actual_max_kvpr(assign)

    # Binary search on threshold T
    for _ in range(40):
        if time.time() - start > 6.0:
            break
        mid = (lo + hi) / 2.0
        a, s = check_any(mid)
        if a is not None:
            hi = mid
            if s < best_score:
                best_score = s
                best_assign = a
        else:
            lo = mid
        if hi - lo < 1e-9:
            break

    # Convert to model objects
    placement = {g: [models[i] for i in best_assign[g]] for g in range(gpu_num)}
    return placement


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
