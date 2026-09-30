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

    """Multi-candidate placement: greedy passes over several sort orders
    (each model to the GPU minimizing its resulting KVPR), a guaranteed
    feasible best-fit fallback, then a move-based local search that
    strictly reduces the maximum KVPR. The best placement wins."""

    def kvprs(loads, sizes):
        return [
            (loads[g] / (GPU_MEM_SIZE - sizes[g])) if sizes[g] < GPU_MEM_SIZE else float("inf")
            for g in range(gpu_num)
        ]

    def greedy(order, bestfit=False):
        placement = {g: [] for g in range(gpu_num)}
        loads = [0.0] * gpu_num
        sizes = [0.0] * gpu_num
        for m in order:
            best, best_key = None, float("inf")
            for g in range(gpu_num):
                free = GPU_MEM_SIZE - sizes[g]
                if m.model_size > free:
                    continue
                if bestfit:
                    key = free - m.model_size  # tightest fit
                else:
                    key = (loads[g] + m.req_rate / m.slo) / (free - m.model_size)
                if key < best_key:
                    best_key, best = key, g
            if best is None:
                return None
            placement[best].append(m)
            loads[best] += m.req_rate / m.slo
            sizes[best] += m.model_size
        return placement, loads, sizes

    def refine(placement, loads, sizes):
        """Local search: move models off the highest-KVPR GPU (and try
        pairwise swaps) while the max KVPR strictly decreases."""
        while True:
            vals = kvprs(loads, sizes)
            cur = max(vals)
            src = vals.index(cur)
            moved = False
            for m in list(placement[src]):
                w = m.req_rate / m.slo
                for dst in range(gpu_num):
                    if dst == src or m.model_size > GPU_MEM_SIZE - sizes[dst]:
                        continue
                    loads[src] -= w; sizes[src] -= m.model_size
                    loads[dst] += w; sizes[dst] -= m.model_size
                    if max(kvprs(loads, sizes)) < cur - 1e-12:
                        placement[src].remove(m); placement[dst].append(m)
                        moved = True
                        break
                    loads[src] += w; sizes[src] += m.model_size
                    loads[dst] -= w; sizes[dst] += m.model_size
                if moved:
                    break
            if not moved:
                return

    orders = [
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True),
        sorted(models, key=lambda m: m.model_size),
    ]

    best, best_kvpr = None, float("inf")
    for order in orders:
        for bf in (False, True):
            res = greedy(order, bestfit=bf)
            if res is None:
                continue
            p, loads, sizes = res
            refine(p, loads, sizes)
            v = max(kvprs(loads, sizes))
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
