GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def _kvpr(load, free):
    """KVPR of one GPU: weighted load divided by free memory."""
    return load / free if free > 0 else float("inf")


def _greedy(gpu_num, models, order_key):
    """Greedy pass: place models (in given order) on the GPU minimizing
    the KVPR after placement. Returns None if infeasible."""
    placement = {g: [] for g in range(gpu_num)}
    free = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num
    for m in sorted(models, key=order_key):
        best, best_kvpr = None, float("inf")
        for g in range(gpu_num):
            if m.model_size <= free[g]:
                kvpr = (load[g] + m.req_rate / m.slo) / (free[g] - m.model_size)
                if kvpr < best_kvpr:
                    best_kvpr, best = kvpr, g
        if best is None:
            return None
        placement[best].append(m)
        load[best] += m.req_rate / m.slo
        free[best] -= m.model_size
    return placement


def _safe_greedy(gpu_num, models):
    """Fallback: place heaviest-load models on GPUs with best load/free ratio;
    succeeds whenever a feasible packing exists."""
    placement = {g: [] for g in range(gpu_num)}
    free = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num
    for m in sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True):
        best, best_ratio = None, float("inf")
        for g in range(gpu_num):
            if m.model_size <= free[g]:
                ratio = load[g] / free[g]
                if ratio < best_ratio:
                    best_ratio, best = ratio, g
        if best is None:
            raise ValueError("Unable to place all models on the GPUs.")
        placement[best].append(m)
        load[best] += m.req_rate / m.slo
        free[best] -= m.model_size
    return placement


def _refine(placement, gpu_num, rounds=50):
    """Local search: repeatedly move a model off the max-KVPR GPU to another
    GPU if it lowers the global max KVPR and fits in memory."""
    for _ in range(rounds):
        loads = {g: sum(m.req_rate / m.slo for m in ms) for g, ms in placement.items()}
        frees = {g: GPU_MEM_SIZE - sum(m.model_size for m in ms) for g, ms in placement.items()}
        src = max(placement, key=lambda g: _kvpr(loads[g], frees[g]))
        src_kvpr = _kvpr(loads[src], frees[src])
        best_move = None
        for m in placement[src]:
            for g in range(gpu_num):
                if g == src or m.model_size > frees[g]:
                    continue
                new_max = max(
                    _kvpr(loads[g] + m.req_rate / m.slo, frees[g] - m.model_size),
                    max(_kvpr(loads[h], frees[h]) for h in range(gpu_num)
                        if h not in (src, g)),
                    _kvpr(loads[src] - m.req_rate / m.slo, frees[src] + m.model_size),
                )
                if new_max < src_kvpr - 1e-12 and (best_move is None or new_max < best_move[0]):
                    best_move = (new_max, m, g)
        if best_move is None:
            break
        _, m, g = best_move
        placement[src].remove(m)
        placement[g].append(m)
    return placement


def compute_model_placement(gpu_num, models):
    """Minimize max KVPR: run resulting-KVPR greedy under several model
    orderings, refine the best with local search, fall back to a safe
    greedy if all orderings are infeasible."""
    orders = [
        lambda m: -(m.req_rate / m.slo),
        lambda m: (m.req_rate / m.slo),
        lambda m: -m.model_size,
        lambda m: -(m.req_rate / m.slo / max(m.model_size, 1e-9)),
    ]
    best, best_kvpr = None, float("inf")
    for key in orders:
        p = _greedy(gpu_num, models, key)
        if p is None:
            continue
        p = _refine(p, gpu_num)
        kvpr = max(_kvpr(sum(m.req_rate / m.slo for m in ms),
                         GPU_MEM_SIZE - sum(m.model_size for m in ms))
                   for ms in p.values())
        if kvpr < best_kvpr:
            best_kvpr, best = kvpr, p
    if best is None:
        best = _safe_greedy(gpu_num, models)
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
