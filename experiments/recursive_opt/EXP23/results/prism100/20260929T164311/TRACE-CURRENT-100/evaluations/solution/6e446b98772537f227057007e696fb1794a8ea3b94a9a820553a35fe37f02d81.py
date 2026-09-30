GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize max KVPR with a simple, reliable heuristic.

    1. Ratio-greedy: sort by req_rate/slo descending, assign each model to
       the feasible GPU with the lowest load/remaining-memory ratio.
    2. Fallback: first-fit-decreasing by size for feasibility.
    3. Move-only local search: relocate a model to another GPU whenever it
       strictly lowers max KVPR and memory fits (never breaks feasibility).
    """

    def max_kvpr(p):
        return max(
            sum(m.req_rate / m.slo for m in ms)
            / max(GPU_MEM_SIZE - sum(m.model_size for m in ms), 1e-9)
            for ms in p.values()
        )

    def greedy(order):
        """Assign each model to the feasible GPU with lowest load/memory ratio."""
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
        """Fallback: first-fit-decreasing by size for feasibility."""
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

    def improve(p):
        """Local search: move models between GPUs if max KVPR strictly drops."""
        for _ in range(30):
            cur = max_kvpr(p)
            improved = False
            for g1 in range(gpu_num):
                for m in list(p[g1]):
                    for g2 in range(gpu_num):
                        if g1 == g2:
                            continue
                        if m.model_size > GPU_MEM_SIZE - sum(x.model_size for x in p[g2]):
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

    p = greedy(sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True))
    if p is None:
        p = first_fit()
    if p is None:
        raise ValueError("Unable to place all models on GPUs")
    return improve(p)


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
