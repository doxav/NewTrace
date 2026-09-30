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

    def greedy(order):
        """Greedy: assign each model to feasible GPU with lowest load/free ratio."""
        placement = {g: [] for g in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for model in order:
            best_idx, best_ratio = None, float("inf")
            for g in range(gpu_num):
                if model.model_size <= shared_kv[g] and shared_kv[g] > 0:
                    ratio = load[g] / shared_kv[g]
                    if ratio < best_ratio:
                        best_ratio, best_idx = ratio, g
            if best_idx is None:
                return None
            placement[best_idx].append(model)
            load[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        return placement

    def first_fit():
        """Fallback: first-fit-decreasing by size for guaranteed feasibility."""
        placement = {g: [] for g in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE] * gpu_num
        for model in sorted(models, key=lambda m: m.model_size, reverse=True):
            for g in range(gpu_num):
                if model.model_size <= shared_kv[g]:
                    placement[g].append(model)
                    shared_kv[g] -= model.model_size
                    break
            else:
                return None
        return placement

    def max_kvpr(placement):
        if placement is None:
            return float("inf")
        return max(
            sum(m.req_rate / m.slo for m in ms) / (GPU_MEM_SIZE - sum(m.model_size for m in ms))
            for ms in placement.values()
        )

    # Greedy variant 2: assign each model to GPU with min current load/free ratio
    def greedy2(order):
        placement = {g: [] for g in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for model in order:
            best_idx, best_ratio = None, float("inf")
            for g in range(gpu_num):
                if model.model_size <= shared_kv[g]:
                    ratio = load[g] / shared_kv[g]
                    if ratio < best_ratio:
                        best_ratio, best_idx = ratio, g
            if best_idx is None:
                return None
            placement[best_idx].append(model)
            load[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        return placement

    # Local search: repeatedly move/swap models between GPUs if it lowers max KVPR
    def improve(p):
        if p is None:
            return None
        for _ in range(50):
            cur = max_kvpr(p)
            improved = False
            for g1 in range(gpu_num):
                for m in list(p[g1]):
                    # try moving m to another GPU
                    for g2 in range(gpu_num):
                        if g1 == g2:
                            continue
                        used2 = sum(x.model_size for x in p[g2])
                        if m.model_size > GPU_MEM_SIZE - used2:
                            continue
                        p[g1].remove(m)
                        p[g2].append(m)
                        if max_kvpr(p) < cur - 1e-12:
                            improved = True
                            break
                        p[g2].remove(m)
                        p[g1].append(m)
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return p

    # Try several orderings; keep the feasible one minimizing max KVPR
    by_rate = sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True)
    by_size = sorted(models, key=lambda m: m.model_size, reverse=True)
    by_ratio = sorted(models, key=lambda m: m.req_rate / m.slo / m.model_size, reverse=True)
    best = None
    best_kvpr = float("inf")
    for order in (by_rate, by_size, by_ratio):
        p = improve(greedy(order))
        kvpr = max_kvpr(p)
        if kvpr < best_kvpr:
            best, best_kvpr = p, kvpr
    if best is None:
        best = first_fit()
    if best is None:
        raise ValueError("Unable to place all models on GPUs")
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
