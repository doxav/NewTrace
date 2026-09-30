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

    """Multi-start greedy: try several sort orders; for each, place each model
    on the GPU minimizing the RESULTING KVPR (w + r)/(mem - s). Return the
    placement with the lowest maximum KVPR."""

    def run(order):
        placement = {g: [] for g in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE] * gpu_num
        weighted = [0.0] * gpu_num
        for model in order:
            best_idx, best_ratio = None, float("inf")
            for g in range(gpu_num):
                if model.model_size <= shared_kv[g]:
                    ratio = (weighted[g] + model.req_rate / model.slo) / (
                        shared_kv[g] - model.model_size
                    )
                    if ratio < best_ratio:
                        best_ratio, best_idx = ratio, g
            if best_idx is None:
                raise ValueError(
                    f"Unable to place model of size {model.model_size} GB: {shared_kv}"
                )
            placement[best_idx].append(model)
            weighted[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        max_kvpr = max(
            (w / m if m > 0 else 0.0) for w, m in zip(weighted, shared_kv)
        )
        return max_kvpr, placement

    best = None
    seen = set()
    for key in (
        lambda m: m.model_size,
        lambda m: m.req_rate / m.slo,
        lambda m: m.req_rate / (m.slo * m.model_size),
    ):
        order = tuple(sorted(models, key=key, reverse=True))
        if order in seen:
            continue
        seen.add(order)
        kvpr, placement = run(order)
        if best is None or kvpr < best[0]:
            best = (kvpr, placement)
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
