GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Simple, robust greedy placement minimizing max KVPR.

    Try several model orderings; each greedy pass assigns every model to
    the feasible GPU minimizing the resulting KVPR. Keep the best feasible
    result. Fall back to first-fit-decreasing by size for guaranteed
    feasibility, since the KVPR-greedy can fail on tight memory cases.
    """

    def greedy(order):
        """Assign each model to the feasible GPU minimizing resulting KVPR."""
        placement = {g: [] for g in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for model in order:
            best_idx, best_kvpr = None, float("inf")
            for g in range(gpu_num):
                if model.model_size <= shared_kv[g]:
                    kvpr = (load[g] + model.req_rate / model.slo) / (
                        shared_kv[g] - model.model_size
                    )
                    if kvpr < best_kvpr:
                        best_kvpr, best_idx = kvpr, g
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

    def max_kvpr(p):
        if p is None:
            return float("inf")
        return max(
            sum(m.req_rate / m.slo for m in ms) / (GPU_MEM_SIZE - sum(m.model_size for m in ms))
            for ms in p.values()
        )

    candidates = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo / m.model_size, reverse=True),
    ]
    best, best_kvpr = None, float("inf")
    for order in candidates:
        p = greedy(order)
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
