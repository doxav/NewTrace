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

    """Greedy placement over multiple sort orders; keep the placement whose
    maximum KVPR is lowest. Each greedy pass assigns each model to the GPU
    with the smallest current KVPR among GPUs where it fits."""

    def greedy(order):
        placement = {g: [] for g in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE] * gpu_num
        weighted = [0.0] * gpu_num
        for model in order:
            best_idx, best_ratio = None, float("inf")
            for g in range(gpu_num):
                if model.model_size <= shared_kv[g] and shared_kv[g] > 0:
                    ratio = weighted[g] / shared_kv[g]
                    if ratio < best_ratio:
                        best_ratio, best_idx = ratio, g
            if best_idx is None:
                return None  # infeasible for this order
            placement[best_idx].append(model)
            weighted[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        return placement

    def max_kvpr(placement):
        vals = []
        for g in range(gpu_num):
            w = sum(m.req_rate / m.slo for m in placement[g])
            s = sum(m.model_size for m in placement[g])
            vals.append(w / (GPU_MEM_SIZE - s) if s < GPU_MEM_SIZE else float("inf"))
        return max(vals)

    # Greedy pass that assigns each model to the GPU minimizing the
    # RESULTING KVPR (weighted load / remaining memory after placement).
    def greedy_resulting(order):
        placement = {g: [] for g in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE] * gpu_num
        weighted = [0.0] * gpu_num
        for model in order:
            best_idx, best_ratio = None, float("inf")
            for g in range(gpu_num):
                if model.model_size <= shared_kv[g]:
                    ratio = (weighted[g] + model.req_rate / model.slo) / (
                        shared_kv[g] - model.model_size
                    )
                    if ratio < best_ratio:
                        best_ratio, best_idx = ratio, g
            if best_idx is None:
                return None
            placement[best_idx].append(model)
            weighted[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        return placement

    orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo),
    ]

    best = None
    best_kvpr = float("inf")
    for order in orders:
        for fn in (greedy, greedy_resulting):
            p = fn(order)
            if p is not None:
                v = max_kvpr(p)
                if v < best_kvpr:
                    best_kvpr, best = v, p

    if best is None:
        # Best-fit-decreasing fallback: classic bin-packing heuristic that
        # maximizes feasibility when ratio-greedy passes all fail.
        best = {g: [] for g in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE] * gpu_num
        for model in sorted(models, key=lambda m: m.model_size, reverse=True):
            best_idx, best_left = None, float("inf")
            for g in range(gpu_num):
                left = shared_kv[g] - model.model_size
                if left >= 0 and left < best_left:
                    best_left, best_idx = left, g
            if best_idx is None:
                raise ValueError("Unable to place all models on the available GPUs.")
            best[best_idx].append(model)
            shared_kv[best_idx] -= model.model_size

    # Lightweight refinement: move a model off the highest-KVPR GPU when it
    # strictly reduces the max KVPR, to squeeze out extra quality cheaply.
    weighted = [sum(m.req_rate / m.slo for m in best[g]) for g in range(gpu_num)]
    while True:
        vals = [weighted[g] / shared_kv[g] if shared_kv[g] > 0 else float("inf")
                for g in range(gpu_num)]
        cur = max(vals)
        src = vals.index(cur)
        moved = False
        for m in list(best[src]):
            w = m.req_rate / m.slo
            for dst in range(gpu_num):
                if dst == src or m.model_size > shared_kv[dst]:
                    continue
                weighted[src] -= w; shared_kv[src] += m.model_size
                weighted[dst] += w; shared_kv[dst] -= m.model_size
                if max(weighted[g] / shared_kv[g] if shared_kv[g] > 0 else float("inf")
                       for g in range(gpu_num)) < cur - 1e-12:
                    best[src].remove(m); best[dst].append(m)
                    moved = True
                    break
                weighted[src] += w; shared_kv[src] -= m.model_size
                weighted[dst] -= w; shared_kv[dst] += m.model_size
            if moved:
                break
        if not moved:
            break

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
