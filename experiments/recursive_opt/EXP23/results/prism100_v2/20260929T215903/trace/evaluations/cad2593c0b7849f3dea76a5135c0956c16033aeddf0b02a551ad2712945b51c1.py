GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize the maximum KVPR via binary search on the answer.

    For a threshold T, a GPU hosting a set S is feasible iff
    sum(req/slo for S) <= T * (80 - sum(size for S)).  We binary search T and
    use a greedy feasibility check with several orderings.
    """
    if not models:
        return {g: [] for g in range(gpu_num)}

    req = [m.req_rate / m.slo for m in models]
    size = [m.model_size for m in models]
    n = len(models)

    # Edge case: a single model bigger than GPU memory -> place it alone.
    if max(size) > GPU_MEM_SIZE:
        placement = {g: [] for g in range(gpu_num)}
        idx = size.index(max(size))
        placement[0].append(models[idx])
        for i in range(n):
            if i != idx:
                placement[i % gpu_num].append(models[i])
        return placement

    def try_pack(order, T):
        """Greedy packing under threshold T. Returns placement or None."""
        placement = {g: [] for g in range(gpu_num)}
        loads = [0.0] * gpu_num
        used = [0.0] * gpu_num
        for i in order:
            placed = False
            # Prefer GPUs whose resulting KVPR stays lowest while <= T.
            cands = []
            for g in range(gpu_num):
                if used[g] + size[i] <= GPU_MEM_SIZE:
                    new_load = loads[g] + req[i]
                    if new_load <= T * (GPU_MEM_SIZE - used[g] - size[i]) + 1e-12:
                        cands.append((new_load / (GPU_MEM_SIZE - used[g] - size[i]), g))
            if cands:
                cands.sort()
                _, g = cands[0]
                placement[g].append(models[i])
                loads[g] += req[i]
                used[g] += size[i]
                placed = True
            if not placed:
                return None
        return placement

    def feasible(T):
        # Try multiple orderings for robustness.
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

    lo = 0.0
    hi = sum(req) / max(GPU_MEM_SIZE - min(size), 1e-9) + 1.0
    best = None
    for _ in range(40):
        mid = (lo + hi) / 2.0
        p = feasible(mid)
        if p is not None:
            best = p
            hi = mid
        else:
            lo = mid

    if best is None:
        # Fallback: greedy minimizing resulting KVPR, always places.
        placement = {g: [] for g in range(gpu_num)}
        loads = [0.0] * gpu_num
        used = [0.0] * gpu_num
        for i in sorted(range(n), key=lambda i: size[i], reverse=True):
            best_g, best_v = 0, float("inf")
            for g in range(gpu_num):
                if used[g] + size[i] <= GPU_MEM_SIZE:
                    v = (loads[g] + req[i]) / (GPU_MEM_SIZE - used[g] - size[i])
                    if v < best_v:
                        best_v, best_g = v, g
            placement[best_g].append(models[i])
            loads[best_g] += req[i]
            used[best_g] += size[i]
        best = placement
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
