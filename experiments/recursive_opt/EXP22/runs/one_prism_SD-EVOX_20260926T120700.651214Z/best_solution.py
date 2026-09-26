GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize max KVPR via best-fit greedy: place each model on the GPU that
    minimizes the resulting KVPR after placement. Try multiple sort orders
    and return the placement with the lowest max KVPR.
    """

    def run(order):
        placement = {g: [] for g in range(gpu_num)}
        free = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num  # sum of req_rate/slo per GPU
        for m in order:
            best, best_kvpr = None, float("inf")
            for g in range(gpu_num):
                if m.model_size <= free[g]:
                    kvpr = (load[g] + m.req_rate / m.slo) / (free[g] - m.model_size)
                    if kvpr < best_kvpr:
                        best_kvpr, best = kvpr, g
            if best is None:
                return None
            placement[best].append(m)
            load[best] += m.req_rate / m.slo
            free[best] -= m.model_size
        return placement

    def max_kvpr(placement):
        if placement is None:
            return float("inf")
        return max(
            sum(m.req_rate / m.slo for m in ms) / (GPU_MEM_SIZE - sum(m.model_size for m in ms))
            for ms in placement.values()
        )

    best_placement, best_score = None, float("inf")
    for key in (
        lambda m: m.req_rate / m.slo,                      # descending load/size
        lambda m: m.model_size,                            # ascending size
        lambda m: -(m.req_rate / m.slo) / m.model_size,    # descending load per GB
    ):
        p = run(sorted(models, key=key, reverse=True))
        if p is not None:
            s = max_kvpr(p)
            if s < best_score:
                best_score, best_placement = s, p

    if best_placement is None:
        raise ValueError("Unable to place all models on the given GPUs.")
    return best_placement


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
