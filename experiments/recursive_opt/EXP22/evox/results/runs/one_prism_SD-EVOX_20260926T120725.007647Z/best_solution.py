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

    """Greedy placement minimizing the *resulting* KVPR at each step,
    tried over several model orderings; keeps the best placement found."""

    def greedy(order):
        placement = {g: [] for g in range(gpu_num)}
        mem = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num  # sum of req_rate/slo per GPU
        for m in order:
            best_g, best_kvpr = None, float("inf")
            for g in range(gpu_num):
                if m.model_size <= mem[g]:
                    new_kvpr = (load[g] + m.req_rate / m.slo) / (mem[g] - m.model_size)
                    if new_kvpr < best_kvpr:
                        best_kvpr, best_g = new_kvpr, g
            if best_g is None:
                return None  # infeasible for this ordering
            placement[best_g].append(m)
            load[best_g] += m.req_rate / m.slo
            mem[best_g] -= m.model_size
        return max(load[g] / mem[g] for g in range(gpu_num)), placement

    keys = [
        lambda m: m.req_rate / m.slo,          # weighted rate desc
        lambda m: m.model_size,                # size desc (big first)
        lambda m: m.req_rate / m.slo / m.model_size,  # ratio desc
        lambda m: m.req_rate,                  # rate desc
    ]
    best = None
    for key in keys:
        res = greedy(sorted(models, key=key, reverse=True))
        if res and (best is None or res[0] < best[0]):
            best = res
    if best is None:
        raise ValueError("Unable to place all models on any GPU.")
    return best[1]


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
