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

    """Multi-order greedy (KVPR-minimizing + best-fit fallback) with
    move-based local search on every candidate placement.

    For each sort order, greedily assign each model to the GPU minimizing
    the RESULTING KVPR (w + r) / (mem - s); if that fails, retry with a
    best-fit rule (least leftover memory) for feasibility. Then refine
    each result by moving models from the highest-KVPR GPU whenever it
    strictly reduces the max KVPR.
    """

    def greedy(order, bestfit=False):
        placement = {g: [] for g in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE] * gpu_num
        weighted = [0.0] * gpu_num
        for model in order:
            best_idx, best_key = None, float("inf")
            for g in range(gpu_num):
                if model.model_size <= shared_kv[g]:
                    if bestfit:
                        key = shared_kv[g] - model.model_size  # least leftover
                    else:
                        key = (weighted[g] + model.req_rate / model.slo) / (
                            shared_kv[g] - model.model_size
                        )
                    if key < best_key:
                        best_key, best_idx = key, g
            if best_idx is None:
                return None
            placement[best_idx].append(model)
            weighted[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        return placement, weighted, shared_kv

    def max_kvpr(weighted, shared_kv):
        return max(
            w / mem if mem > 0 else float("inf")
            for w, mem in zip(weighted, shared_kv)
        )

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
            p, weighted, shared_kv = res
            # Move-based local search on each candidate placement
            improved = True
            while improved:
                improved = False
                cur = max_kvpr(weighted, shared_kv)
                src = max(range(gpu_num), key=lambda g: weighted[g] / shared_kv[g] if shared_kv[g] > 0 else float("inf"))
                for m in list(p[src]):
                    w = m.req_rate / m.slo
                    for dst in range(gpu_num):
                        if dst == src or m.model_size > shared_kv[dst]:
                            continue
                        weighted[src] -= w; shared_kv[src] += m.model_size
                        weighted[dst] += w; shared_kv[dst] -= m.model_size
                        new = max_kvpr(weighted, shared_kv)
                        if new < cur - 1e-12:
                            p[src].remove(m); p[dst].append(m)
                            cur = new; improved = True
                            break
                        weighted[src] += w; shared_kv[src] -= m.model_size
                        weighted[dst] -= w; shared_kv[dst] += m.model_size
                    if improved:
                        break
            v = max_kvpr(weighted, shared_kv)
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
