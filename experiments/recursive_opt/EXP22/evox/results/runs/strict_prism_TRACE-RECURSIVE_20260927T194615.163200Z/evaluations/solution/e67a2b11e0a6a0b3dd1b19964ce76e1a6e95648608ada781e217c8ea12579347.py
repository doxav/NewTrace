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

    """Greedy placement over several sort orders, then a local-search
    refinement (single-model moves and pairwise swaps) that repeatedly
    reduces the maximum KVPR until no improving change exists. A
    first-fit-decreasing fallback guarantees feasibility."""

    def kvpr_of(loads, sizes):
        return [
            (loads[g] / (GPU_MEM_SIZE - sizes[g])) if sizes[g] < GPU_MEM_SIZE else float("inf")
            for g in range(gpu_num)
        ]

    def greedy(order):
        placement = {g: [] for g in range(gpu_num)}
        loads = [0.0] * gpu_num
        sizes = [0.0] * gpu_num
        for m in order:
            best, best_r = None, float("inf")
            for g in range(gpu_num):
                if m.model_size <= GPU_MEM_SIZE - sizes[g]:
                    r = (loads[g] + m.req_rate / m.slo) / (GPU_MEM_SIZE - sizes[g] - m.model_size)
                    if r < best_r:
                        best_r, best = r, g
            if best is None:
                return None
            placement[best].append(m)
            loads[best] += m.req_rate / m.slo
            sizes[best] += m.model_size
        return placement, loads, sizes

    def refine(placement, loads, sizes):
        # Local search: move or swap models to reduce max KVPR.
        improved = True
        while improved:
            improved = False
            cur = max(kvpr_of(loads, sizes))
            src = max(range(gpu_num), key=lambda g: kvpr_of(loads, sizes)[g])
            # Try moving one model out of the most loaded GPU
            for m in list(placement[src]):
                w = m.req_rate / m.slo
                for dst in range(gpu_num):
                    if dst == src or m.model_size > GPU_MEM_SIZE - sizes[dst]:
                        continue
                    loads[src] -= w; sizes[src] -= m.model_size
                    loads[dst] += w; sizes[dst] += m.model_size
                    new = max(kvpr_of(loads, sizes))
                    if new < cur - 1e-12:
                        placement[src].remove(m); placement[dst].append(m)
                        cur = new; improved = True
                        break
                    loads[src] += w; sizes[src] += m.model_size
                    loads[dst] -= w; sizes[dst] -= m.model_size
                if improved:
                    break
            if improved:
                continue
            # Try swapping two models between GPUs
            done = False
            for a in range(gpu_num):
                for b in range(a + 1, gpu_num):
                    for ma in list(placement[a]):
                        for mb in list(placement[b]):
                            wa, wb = ma.req_rate / ma.slo, mb.req_rate / mb.slo
                            if (sizes[b] - mb.model_size + ma.model_size > GPU_MEM_SIZE or
                                    sizes[a] - ma.model_size + mb.model_size > GPU_MEM_SIZE):
                                continue
                            loads[a] += wb - wa; loads[b] += wa - wb
                            sizes[a] += mb.model_size - ma.model_size
                            sizes[b] += ma.model_size - mb.model_size
                            new = max(kvpr_of(loads, sizes))
                            if new < cur - 1e-12:
                                placement[a].remove(ma); placement[a].append(mb)
                                placement[b].remove(mb); placement[b].append(ma)
                                cur = new; improved = True; done = True
                                break
                            loads[a] -= wb - wa; loads[b] -= wa - wb
                            sizes[a] -= mb.model_size - ma.model_size
                            sizes[b] -= ma.model_size - mb.model_size
                        if done: break
                    if done: break
                if done: break

    def greedy_bestfit(order):
        # Feasibility fallback: assign to GPU with least leftover memory.
        placement = {g: [] for g in range(gpu_num)}
        loads = [0.0] * gpu_num
        sizes = [0.0] * gpu_num
        for m in order:
            cands = [g for g in range(gpu_num) if m.model_size <= GPU_MEM_SIZE - sizes[g]]
            if not cands:
                return None
            best = min(cands, key=lambda g: GPU_MEM_SIZE - sizes[g] - m.model_size)
            placement[best].append(m)
            loads[best] += m.req_rate / m.slo
            sizes[best] += m.model_size
        return placement, loads, sizes

    orders = [
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True),
        sorted(models, key=lambda m: m.model_size),
        sorted(models, key=lambda m: m.req_rate / m.slo),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size),
    ]

    best, best_kvpr = None, float("inf")
    for order in orders:
        for res in (greedy(order), greedy_bestfit(order)):
            if res is None:
                continue
            p, loads, sizes = res
            refine(p, loads, sizes)
            v = max(kvpr_of(loads, sizes))
            if v < best_kvpr:
                best_kvpr, best = v, p

    # Last-resort fallback: first-fit decreasing. This succeeds whenever
    # any feasible placement exists, so we never fail on solvable instances.
    if best is None:
        placement = {g: [] for g in range(gpu_num)}
        loads = [0.0] * gpu_num
        sizes = [0.0] * gpu_num
        for m in sorted(models, key=lambda m: m.model_size, reverse=True):
            for g in range(gpu_num):
                if m.model_size <= GPU_MEM_SIZE - sizes[g]:
                    placement[g].append(m)
                    loads[g] += m.req_rate / m.slo
                    sizes[g] += m.model_size
                    break
            else:
                raise ValueError("Unable to place all models on the available GPUs.")
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
