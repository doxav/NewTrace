GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize the maximum KVPR across GPUs via binary search on the answer.

    For a threshold T, a GPU hosting a set S is feasible iff
        sum(req/slo over S) <= T * (80 - sum(size over S))  and  sum(size) <= 80.
    We binary search T and, for each T, run a greedy feasibility check with
    several orderings (models sorted by load density descending, etc.).
    """

    import random

    n = len(models)
    if n == 0:
        return {g: [] for g in range(gpu_num)}

    req = [m.req_rate / m.slo for m in models]
    size = [m.model_size for m in models]

    def try_pack(order, T):
        """Greedy feasibility check for threshold T. Returns placement or None."""
        placement = {g: [] for g in range(gpu_num)}
        load = [0.0] * gpu_num
        used = [0.0] * gpu_num
        for i in order:
            placed = False
            # Prefer the GPU that keeps the most slack under T.
            best_g, best_slack = None, None
            for g in range(gpu_num):
                new_load = load[g] + req[i]
                new_used = used[g] + size[i]
                if new_used > GPU_MEM_SIZE:
                    continue
                cap = T * (GPU_MEM_SIZE - new_used)
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

    def feasible_for_T(T):
        """Try several model orderings for threshold T."""
        orders = [
            sorted(range(n), key=lambda i: req[i] / max(size[i], 1e-9), reverse=True),
            sorted(range(n), key=lambda i: size[i], reverse=True),
            sorted(range(n), key=lambda i: req[i], reverse=True),
            sorted(range(n), key=lambda i: req[i] / max(size[i], 1e-9)),
        ]
        for order in orders:
            p = try_pack(order, T)
            if p is not None:
                return p
        return None

    def max_kvpr(placement):
        best = 0.0
        for gpu_models in placement.values():
            denom = GPU_MEM_SIZE - sum(m.model_size for m in gpu_models)
            if denom <= 0:
                return float("inf")
            kvpr = sum(m.req_rate / m.slo for m in gpu_models) / denom
            best = max(best, kvpr)
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

    def backtrack_placement():
        """Exact memory-feasibility via backtracking (small model counts)."""
        order = sorted(range(n), key=lambda i: size[i], reverse=True)
        placement = {g: [] for g in range(gpu_num)}
        free = [GPU_MEM_SIZE] * gpu_num

        def rec(k):
            if k == n:
                return True
            i = order[k]
            tried = set()
            for g in range(gpu_num):
                if size[i] <= free[g] and free[g] not in tried:
                    tried.add(free[g])
                    placement[g].append(models[i])
                    free[g] -= size[i]
                    if rec(k + 1):
                        return True
                    free[g] += size[i]
                    placement[g].pop()
            return False

        return placement if rec(0) else None

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
        if hi - lo < 1e-9:
            break

    # Fallbacks to guarantee a valid placement.
    if best_placement is None:
        best_placement = backtrack_placement()
    if best_placement is None:
        best_placement = ffd_placement()
    if best_placement is None:
        # Single oversized model or infeasible: best-effort spread.
        best_placement = {g: [] for g in range(gpu_num)}
        for i in sorted(range(n), key=lambda i: size[i], reverse=True):
            g = min(range(gpu_num), key=lambda x: sum(m.model_size for m in best_placement[x]))
            best_placement[g].append(models[i])

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
