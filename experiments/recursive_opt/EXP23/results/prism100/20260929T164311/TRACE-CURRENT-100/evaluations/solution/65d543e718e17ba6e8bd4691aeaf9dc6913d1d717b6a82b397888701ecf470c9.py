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
        """Greedy: assign each model to the feasible GPU with the lowest
        current load/free-memory ratio (a proxy for resulting KVPR)."""
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

    def max_kvpr(placement):
        if placement is None:
            return float("inf")
        return max(
            sum(m.req_rate / m.slo for m in ms)
            / max(GPU_MEM_SIZE - sum(m.model_size for m in ms), 1e-9)
            for ms in placement.values()
        )

    # Single proven ordering: load descending. Local search added no gain
    # in max KVPR and only cost execution time, so it is removed.
    p = greedy(sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True))
    if p is None:
        p = first_fit()
    if p is None:
        raise ValueError("Unable to place all models on GPUs")
    return p


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
