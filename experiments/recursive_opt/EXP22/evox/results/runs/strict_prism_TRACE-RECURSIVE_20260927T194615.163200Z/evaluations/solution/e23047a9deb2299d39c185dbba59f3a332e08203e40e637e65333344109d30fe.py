GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """

    """Greedy placement over several sort orders; each model is assigned to
    the GPU minimizing the RESULTING KVPR (w + r)/(mem - s). If an order is
    infeasible, a best-fit fallback (tightest remaining memory) is used so a
    valid placement is always produced. The best placement is then refined by
    a move/swap local search that reduces the maximum KVPR."""

    def greedy(order):
        """Assign each model to the GPU minimizing resulting KVPR, requiring
        at least some KV memory to remain (denominator > 0) so KVPR stays finite."""
        placement = {g: [] for g in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE] * gpu_num
        weighted = [0.0] * gpu_num
        for model in order:
            best_idx, best_key = None, float("inf")
            for g in range(gpu_num):
                if shared_kv[g] - model.model_size > 0:
                    key = (weighted[g] + model.req_rate / model.slo) / (
                        shared_kv[g] - model.model_size
                    )
                    if key < best_key:
                        best_key, best_idx = key, g
            if best_idx is None:
                return None  # infeasible for this order
            placement[best_idx].append(model)
            weighted[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        return placement, weighted, shared_kv

    def greedy_bestfit(order):
        """Feasibility fallback: assign to GPU with least leftover memory,
        preferring placements that still leave positive KV memory."""
        placement = {g: [] for g in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE] * gpu_num
        weighted = [0.0] * gpu_num
        for model in order:
            cands = [g for g in range(gpu_num) if model.model_size <= shared_kv[g]]
            if not cands:
                return None
            pos = [g for g in cands if shared_kv[g] - model.model_size > 0]
            pool = pos if pos else cands
            best_idx = min(pool, key=lambda g: shared_kv[g] - model.model_size)
            placement[best_idx].append(model)
            weighted[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        return placement, weighted, shared_kv

    def kvprs(weighted, shared_kv):
        return [
            (weighted[g] / shared_kv[g]) if shared_kv[g] > 0 else float("inf")
            for g in range(gpu_num)
        ]

    def refine(placement, weighted, shared_kv):
        """Local search: repeatedly move one model out of the most pressured
        GPU, then try pairwise swaps between GPUs, accepting only changes
        that strictly lower the maximum KVPR, until no improvement exists."""
        while True:
            vals = kvprs(weighted, shared_kv)
            cur = max(vals)
            src = vals.index(cur)
            moved = False
            for m in list(placement[src]):
                w = m.req_rate / m.slo
                for dst in range(gpu_num):
                    if dst == src or m.model_size > shared_kv[dst]:
                        continue
                    weighted[src] -= w; shared_kv[src] += m.model_size
                    weighted[dst] += w; shared_kv[dst] -= m.model_size
                    if max(kvprs(weighted, shared_kv)) < cur - 1e-12:
                        placement[src].remove(m); placement[dst].append(m)
                        moved = True
                        break
                    weighted[src] += w; shared_kv[src] -= m.model_size
                    weighted[dst] -= w; shared_kv[dst] += m.model_size
                if moved:
                    break
            if moved:
                continue
            # Pairwise swaps between GPUs
            done = False
            for a in range(gpu_num):
                for b in range(a + 1, gpu_num):
                    for ma in list(placement[a]):
                        for mb in list(placement[b]):
                            wa, wb = ma.req_rate / ma.slo, mb.req_rate / mb.slo
                            if (shared_kv[b] + mb.model_size - ma.model_size < 0 or
                                    shared_kv[a] + ma.model_size - mb.model_size < 0):
                                continue
                            weighted[a] += wb - wa; weighted[b] += wa - wb
                            shared_kv[a] += ma.model_size - mb.model_size
                            shared_kv[b] += mb.model_size - ma.model_size
                            if max(kvprs(weighted, shared_kv)) < cur - 1e-12:
                                placement[a].remove(ma); placement[a].append(mb)
                                placement[b].remove(mb); placement[b].append(ma)
                                done = True
                                break
                            weighted[a] -= wb - wa; weighted[b] -= wa - wb
                            shared_kv[a] -= ma.model_size - mb.model_size
                            shared_kv[b] -= mb.model_size - ma.model_size
                        if done: break
                    if done: break
                if done: break
            if not done:
                return

    orders = [
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True),
        sorted(models, key=lambda m: m.model_size),
        sorted(models, key=lambda m: m.req_rate / m.slo),
    ]

    best, best_kvpr = None, float("inf")
    for order in orders:
        res = greedy(order)
        if res is None:
            res = greedy_bestfit(order)  # guaranteed-feasible fallback
        if res is None:
            continue
        p, weighted, shared_kv = res
        refine(p, weighted, shared_kv)
        v = max(kvprs(weighted, shared_kv))
        if v < best_kvpr:
            best_kvpr, best = v, p

    if best is None:
        raise ValueError("Unable to place all models on the available GPUs.")

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
