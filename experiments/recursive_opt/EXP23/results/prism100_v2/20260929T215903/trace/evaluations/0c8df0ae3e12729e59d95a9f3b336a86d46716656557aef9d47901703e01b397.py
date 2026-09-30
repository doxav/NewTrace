GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize the maximum KVPR via binary search on the answer T.

    KVPR <= T for a GPU holding set S means:
        sum(req_i) <= T * (80 - sum(size_i))
      <=> sum(req_i + T * size_i) <= T * 80
    So for a target T, each model has weight w_i = req_i + T*size_i and we
    must pack them into gpu_num bins of capacity T*80. Feasibility is
    checked with first-fit-decreasing over several orderings; T is
    binary-searched, then a light local search polishes the result.
    """
    if not models:
        return {g: [] for g in range(gpu_num)}

    req = [m.req_rate / m.slo for m in models]
    size = [m.model_size for m in models]
    n = len(models)

    def pack(order, T):
        """FFD packing with weights req + T*size, cap T*80, plus memory limit."""
        cap = T * GPU_MEM_SIZE
        if cap <= 0:
            return None
        bins_load = [0.0] * gpu_num
        bins_used = [0.0] * gpu_num
        placement = {g: [] for g in range(gpu_num)}
        for i in order:
            w = req[i] + T * size[i]
            if w > cap:
                return None
            placed = False
            for g in range(gpu_num):
                if bins_load[g] + w <= cap and bins_used[g] + size[i] <= GPU_MEM_SIZE:
                    placement[g].append(models[i])
                    bins_load[g] += w
                    bins_used[g] += size[i]
                    placed = True
                    break
            if not placed:
                return None
        return placement

    def max_kvpr(placement):
        best = 0.0
        for g, ms in placement.items():
            denom = GPU_MEM_SIZE - sum(m.model_size for m in ms)
            if denom <= 0:
                return float("inf")
            kvpr = sum(m.req_rate / m.slo for m in ms) / denom
            if kvpr > best:
                best = kvpr
        return best

    # A model too big for a single GPU: best-effort place everything on GPU 0.
    if any(s > GPU_MEM_SIZE for s in size):
        return {g: [] for g in range(gpu_num)} if gpu_num == 0 else \
            {0: list(models), **{g: [] for g in range(1, gpu_num)}}

    # Upper bound on T: put everything on one GPU (always feasible).
    total_req = sum(req)
    hi = total_req / max(GPU_MEM_SIZE - sum(size), 1e-9) + 1e-6
    lo = 0.0

    orders = [
        sorted(range(n), key=lambda i: size[i], reverse=True),
        sorted(range(n), key=lambda i: req[i] + 1e-9 * size[i], reverse=True),
        sorted(range(n), key=lambda i: req[i] / max(size[i], 1e-9), reverse=True),
        list(range(n)),
    ]

    best_placement = None

    for _ in range(60):
        mid = (lo + hi) / 2.0
        found = None
        for order in orders:
            p = pack(order, mid)
            if p is not None:
                found = p
                break
        if found is not None:
            best_placement = found
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-9:
            break

    if best_placement is None:
        # Fallback: FFD by size (feasibility-first).
        placement = {g: [] for g in range(gpu_num)}
        free = [GPU_MEM_SIZE] * gpu_num
        for i in sorted(range(n), key=lambda i: size[i], reverse=True):
            g = next((g for g in range(gpu_num) if size[i] <= free[g]),
                     max(range(gpu_num), key=lambda x: free[x]))
            placement[g].append(models[i])
            free[g] -= size[i]
        best_placement = placement

    # Light local-search polish: single moves that reduce max KVPR.
    best_score = max_kvpr(best_placement)
    improved = True
    while improved:
        improved = False
        for src in range(gpu_num):
            for mi in range(len(best_placement[src])):
                m = best_placement[src][mi]
                for dst in range(gpu_num):
                    if dst == src:
                        continue
                    used = sum(x.model_size for x in best_placement[dst])
                    if m.model_size > GPU_MEM_SIZE - used:
                        continue
                    best_placement[src].pop(mi)
                    best_placement[dst].append(m)
                    s = max_kvpr(best_placement)
                    if s < best_score - 1e-12:
                        best_score = s
                        improved = True
                        break
                    best_placement[dst].pop()
                    best_placement[src].insert(mi, m)
                if improved:
                    break
            if improved:
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
