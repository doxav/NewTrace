GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize the maximum KVPR via binary search on the answer.

    Key insight: a placement has max KVPR <= T iff for every GPU,
        sum(req_i) <= T * (80 - sum(size_i))
    which rearranges to
        sum(req_i + T * size_i) <= 80 * T.
    So feasibility at threshold T is a bin-packing problem where each model
    has transformed cost (req + T*size) and each GPU has capacity 80*T.
    We binary search T and check feasibility with FFD under several orderings.
    """
    import time

    n = len(models)
    if n == 0:
        return {g: [] for g in range(gpu_num)}

    req = [m.req_rate / m.slo for m in models]
    size = [m.model_size for m in models]

    def try_pack(T, order):
        """FFD bin packing with transformed costs; returns placement or None."""
        cap = GPU_MEM_SIZE * T
        # A single model exceeding the whole capacity makes T infeasible.
        placement = {g: [] for g in range(gpu_num)}
        used = [0.0] * gpu_num
        for i in order:
            c = req[i] + T * size[i]
            if c > cap:
                return None
            placed = False
            for g in range(gpu_num):
                if used[g] + c <= cap + 1e-12:
                    placement[g].append(models[i])
                    used[g] += c
                    placed = True
                    break
            if not placed:
                return None
        return placement

    def feasible(T):
        """Try FFD under several orderings; return best placement or None."""
        orders = [
            sorted(range(n), key=lambda i: req[i] + T * size[i], reverse=True),
            sorted(range(n), key=lambda i: size[i], reverse=True),
            sorted(range(n), key=lambda i: req[i], reverse=True),
            sorted(range(n), key=lambda i: (req[i] + T * size[i]) / max(size[i], 1e-9), reverse=True),
        ]
        best = None
        for order in orders:
            p = try_pack(T, order)
            if p is not None:
                return p
        return best

    start = time.time()

    # Lower bound: T must satisfy total transformed cost <= gpu_num * cap,
    # i.e. T >= total_req / (80*gpu_num - total_size). Also each single model
    # needs req_i + T*size_i <= 80*T  =>  T >= req_i / (80 - size_i).
    total_req = sum(req)
    total_size = sum(size)
    lo = total_req / max(80.0 * gpu_num - total_size, 1e-9)
    for i in range(n):
        if size[i] < GPU_MEM_SIZE:
            lo = max(lo, req[i] / (GPU_MEM_SIZE - size[i]))
    lo = max(lo, 1e-9)
    hi = max(lo * 4.0, 1.0)

    # Ensure hi is feasible (or fall back).
    result = feasible(hi)
    expand = 0
    while result is None and expand < 20:
        hi *= 2.0
        result = feasible(hi)
        expand += 1
    if result is None:
        # Truly infeasible packing: fall back to best-effort greedy on sizes.
        placement = {g: [] for g in range(gpu_num)}
        used = [0.0] * gpu_num
        for i in sorted(range(n), key=lambda i: size[i], reverse=True):
            g = min(range(gpu_num), key=lambda x: used[x])
            if used[g] + size[i] <= GPU_MEM_SIZE:
                placement[g].append(models[i])
                used[g] += size[i]
            else:
                placement[0].append(models[i])
        return placement

    best_placement = result
    for _ in range(60):
        if time.time() - start > 5.0:
            break
        mid = (lo + hi) / 2.0
        p = feasible(mid)
        if p is not None:
            hi = mid
            best_placement = p
        else:
            lo = mid
        if hi - lo < 1e-9 * max(hi, 1.0):
            break

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
