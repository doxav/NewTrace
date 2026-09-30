GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Greedy placement minimizing max KVPR: sort models by req_rate/slo
    descending, then assign each model to the feasible GPU with the
    lowest current load/free-memory ratio. Fall back to other orderings
    if infeasible, keeping the placement with lowest max KVPR.
    """

    def greedy(order):
        placement = {g: [] for g in range(gpu_num)}
        mem = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num  # sum of req_rate/slo per GPU
        for m in order:
            best_idx, best_score = None, float("inf")
            for g in range(gpu_num):
                if m.model_size <= mem[g] and load[g] / mem[g] < best_score:
                    best_score, best_idx = load[g] / mem[g], g
            if best_idx is None:
                return None  # infeasible ordering
            placement[best_idx].append(m)
            load[best_idx] += m.req_rate / m.slo
            mem[best_idx] -= m.model_size
        return placement, max(
            (load[g] / mem[g] if mem[g] > 0 else float("inf") for g in range(gpu_num)),
            default=0.0,
        )

    best, best_kvpr = None, float("inf")
    for order in (
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        list(models),
    ):
        result = greedy(order)
        if result and result[1] < best_kvpr:
            best_kvpr, best = result[1], result[0]

    if best is None:
        raise ValueError("Unable to place all models on the GPUs.")
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

    print(f"Max KVPR: {np.mean(all_kvpr):.3f}")
