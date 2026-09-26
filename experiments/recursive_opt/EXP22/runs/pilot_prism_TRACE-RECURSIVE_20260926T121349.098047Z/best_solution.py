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

    """Greedy placement trying multiple sort heuristics; returns the placement
    with the smallest maximum KVPR. Each greedy pass assigns models (largest
    first so they fit) to the feasible GPU minimizing the POST-placement KVPR."""

    def greedy(order):
        placement = {g: [] for g in range(gpu_num)}
        mem = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for model in order:
            best_idx, best_ratio = None, float("inf")
            for g in range(gpu_num):
                if model.model_size <= mem[g]:
                    ratio = (load[g] + model.req_rate / model.slo) / (mem[g] - model.model_size)
                    if ratio < best_ratio:
                        best_ratio, best_idx = ratio, g
            if best_idx is None:
                return None  # infeasible ordering
            placement[best_idx].append(model)
            load[best_idx] += model.req_rate / model.slo
            mem[best_idx] -= model.model_size
        return placement

    def max_kvpr(placement):
        return max(
            (sum(m.req_rate / m.slo for m in ms) / (GPU_MEM_SIZE - sum(m.model_size for m in ms)))
            if ms else 0.0
            for ms in placement.values()
        )

    best_placement, best_score = None, float("inf")
    keys = [
        lambda m: (m.model_size, m.req_rate / m.slo),
        lambda m: m.model_size,
        lambda m: (m.req_rate / m.slo, m.model_size),
        lambda m: m.req_rate / m.slo,
        lambda m: (m.req_rate / m.slo / m.model_size if m.model_size else 0),
        lambda m: (m.model_size, -m.req_rate / m.slo),
    ]
    for key in keys:
        order = sorted(models, key=key, reverse=True)
        p = greedy(order)
        if p is not None:
            score = max_kvpr(p)
            if score < best_score:
                best_score, best_placement = score, p

    if best_placement is None:
        raise ValueError("Unable to place all models on the available GPUs.")
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
