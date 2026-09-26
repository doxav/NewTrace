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

    """Greedy placement trying multiple orderings; each model goes to the
    GPU minimizing the resulting KVPR after placement. Returns best result."""

    def run(order):
        placement = {g: [] for g in range(gpu_num)}
        shared = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for m in order:
            best, best_kvpr = None, float("inf")
            for g in range(gpu_num):
                rem = shared[g] - m.model_size
                if rem < 0:
                    continue
                kvpr = (load[g] + m.req_rate / m.slo) / rem
                if kvpr < best_kvpr:
                    best_kvpr, best = kvpr, g
            if best is None:
                return None
            placement[best].append(m)
            load[best] += m.req_rate / m.slo
            shared[best] -= m.model_size
        return placement, max(load[g] / shared[g] for g in range(gpu_num))

    orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo / m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo),
    ]

    best_result = None
    best_kvpr = float("inf")
    for order in orders:
        res = run(order)
        if res is not None and res[1] < best_kvpr:
            best_result, best_kvpr = res[0], res[1]

    if best_result is None:
        raise ValueError("Unable to place all models on any GPU.")
    return best_result


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
