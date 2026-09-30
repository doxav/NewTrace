GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize max KVPR via greedy placement over multiple model orderings and
    two assignment criteria: (a) minimize resulting KVPR after placement,
    (b) minimize current load/free-memory ratio. Keep the feasible placement
    with the lowest maximum KVPR.
    """

    def greedy(order, criterion):
        placement = {g: [] for g in range(gpu_num)}
        mem = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num  # sum of req_rate/slo per GPU
        for m in order:
            best_idx, best_score = None, float("inf")
            for g in range(gpu_num):
                if m.model_size <= mem[g]:
                    score = (load[g] + m.req_rate / m.slo) / (mem[g] - m.model_size) \
                        if criterion == "kvpr" else load[g] / mem[g]
                    if score < best_score:
                        best_score, best_idx = score, g
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
    orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.model_size),
        list(models),
    ]
    for order in orders:
        for criterion in ("kvpr", "ratio"):
            result = greedy(order, criterion)
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
