GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize max KVPR via greedy placement: assign each model to the GPU
    that minimizes the *resulting* KVPR after placement. Try several model
    orderings (by req_rate/slo desc, by size desc, original) and keep the
    placement with the lowest maximum KVPR.
    """

    def greedy(order):
        placement = {g: [] for g in range(gpu_num)}
        mem = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num  # sum of req_rate/slo per GPU
        for m in order:
            best_idx, best_kvpr = None, float("inf")
            for g in range(gpu_num):
                if m.model_size <= mem[g]:
                    # KVPR after placing this model on GPU g
                    kvpr = (load[g] + m.req_rate / m.slo) / (mem[g] - m.model_size)
                    if kvpr < best_kvpr:
                        best_kvpr, best_idx = kvpr, g
            if best_idx is None:
                return None  # infeasible ordering
            placement[best_idx].append(m)
            load[best_idx] += m.req_rate / m.slo
            mem[best_idx] -= m.model_size
        return placement, max(
            (load[g] / mem[g] if mem[g] > 0 else float("inf") for g in range(gpu_num)),
            default=0.0,
        )

    best = None
    best_kvpr = float("inf")
    orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.model_size),
        list(models),
    ]
    for order in orders:
        result = greedy(order)
        if result is None:
            continue
        placement, max_kvpr = result
        if max_kvpr < best_kvpr:
            best_kvpr, best = max_kvpr, placement

    if best is None:
        raise ValueError("Unable to place all models on the GPUs.")
    return best


# EVOLVE-BLOCK-END


if __name__ == "__main__":
    import numpy as np
    from evaluator import calculate_kvcache_pressure, generate_test_gpu_models, safe_float

    test_cases = generate_test_gpu_models()
    all_kvpr = []
    for i, (gpu_num, gpu_models) in enumerate(test_cases):

        results = compute_model_placement(gpu_num, gpu_models)
        max_kvpr = calculate_kvcache_pressure(results)
        all_kvpr.append(safe_float(max_kvpr))

    print(f"Max KVPR: {np.mean(all_kvpr):.3f}")
