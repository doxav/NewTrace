GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Greedy placement minimizing max KVPR.

    Primary ordering: heaviest load (req_rate/slo) first; assign each model
    to the feasible GPU with the lowest load/free-memory ratio. Fallback:
    largest model first (best-fit for tight memory cases).
    """

    def greedy(order):
        """Assign each model to the GPU minimizing load/free-memory ratio."""
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

    # Primary: heaviest load first; fallback: largest model first (fit-friendly)
    p = greedy(sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True))
    if p is None:
        p = greedy(sorted(models, key=lambda m: m.model_size, reverse=True))
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
