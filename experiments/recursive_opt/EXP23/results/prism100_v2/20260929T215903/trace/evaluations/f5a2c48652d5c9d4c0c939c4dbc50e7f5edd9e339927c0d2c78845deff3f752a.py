GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize the maximum KVPR via binary search on the answer.

    For a threshold T, a placement is feasible if every GPU satisfies:
        sum(req_rate/slo) <= T * (80 - sum(model_size))
    We binary search T and use a best-fit-decreasing style greedy feasibility
    check (trying several model orderings). Falls back to a guaranteed
    packing (FFD) if the threshold search cannot find a feasible packing.
    """
    n = len(models)
    if n == 0:
        return {g: [] for g in range(gpu_num)}

    req = [m.req_rate / m.slo for m in models]
    size = [m.model_size for m in models]

    # Edge case: a single model larger than GPU memory -> place it alone.
    if max(size) > GPU_MEM_SIZE:
        placement = {g: [] for g in range(gpu_num)}
        placement[0] = list(models)
        return placement

    orderings = [
        sorted(range(n), key=lambda i: size[i], reverse=True),
        sorted(range(n), key=lambda i: req[i], reverse=True),
        sorted(range(n), key=lambda i: req[i] / max(size[i], 1e-9), reverse=True),
        sorted(range(n), key=lambda i: req[i] / max(GPU_MEM_SIZE - size[i], 1e-9), reverse=True),
    ]

    def feasible_with(order, T, assign):
        used = [0.0] * gpu_num
        load = [0.0] * gpu_num
        for i in order:
            s, r = size[i], req[i]
            best_g, best_slack = -1, -1.0
            for g in range(gpu_num):
                if used[g] + s > GPU_MEM_SIZE:
                    continue
                new_load = load[g] + r
                if new_load > T * (GPU_MEM_SIZE - used[g] - s) + 1e-12:
                    continue
                # slack = remaining load capacity after placing
                slack = T * (GPU_MEM_SIZE - used[g] - s) - new_load
                if slack > best_slack:
                    best_slack = slack
                    best_g = g
            if best_g < 0:
                return False
            assign[i] = best_g
            used[best_g] += s
            load[best_g] += r
        return True

    def check(T):
        for order in orderings:
            assign = [-1] * n
            if feasible_with(order, T, assign):
                return assign
        return None

    # Binary search on T
    lo = 1e-9
    hi = sum(req) / max(GPU_MEM_SIZE - max(0, sum(size) - (gpu_num - 1) * GPU_MEM_SIZE), 1e-9) + 1.0
    hi = max(hi, max(req) / max(GPU_MEM_SIZE - size[i] if False else 1.0, 1e-9))
    # simpler safe upper bound
    hi = max(sum(req) / 1e-6, 1.0) if gpu_num == 0 else (sum(req) + max(req)) / max(GPU_MEM_SIZE - min(size), 1e-6)
    best_assign = None
    for _ in range(40):
        mid = (lo + hi) / 2.0
        assign = check(mid)
        if assign is not None:
            best_assign = assign
            hi = mid
        else:
            lo = mid

    if best_assign is not None:
        placement = {g: [] for g in range(gpu_num)}
        for i in range(n):
            placement[best_assign[i]].append(models[i])
        return placement

    # Fallback: guaranteed packing via First-Fit-Decreasing on size
    placement = {g: [] for g in range(gpu_num)}
    free = [GPU_MEM_SIZE] * gpu_num
    for i in sorted(range(n), key=lambda i: size[i], reverse=True):
        for g in range(gpu_num):
            if size[i] <= free[g]:
                placement[g].append(models[i])
                free[g] -= size[i]
                break
        else:
            # place on emptiest GPU (should not happen given edge case above)
            g = max(range(gpu_num), key=lambda x: free[x])
            placement[g].append(models[i])
            free[g] -= size[i]
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
