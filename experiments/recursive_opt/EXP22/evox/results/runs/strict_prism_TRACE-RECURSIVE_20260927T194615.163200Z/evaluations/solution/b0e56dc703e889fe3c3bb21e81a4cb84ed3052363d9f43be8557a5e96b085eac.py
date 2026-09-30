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

    """Greedy placement over several sort orders; each model goes to the GPU
    minimizing the RESULTING KVPR (w + r)/(mem - s). The placement with the
    lowest maximum KVPR is returned. A best-fit fallback guarantees feasibility
    on tight instances."""

    def run_greedy(order, bestfit=False):
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]
        weighted_req_rate = [0.0 for _ in range(gpu_num)]
        for model in order:
            best_idx, best_ratio = None, float("inf")
            for gpu_id in range(gpu_num):
                if model.model_size <= shared_kv[gpu_id]:
                    if bestfit:
                        new_ratio = shared_kv[gpu_id] - model.model_size
                    else:
                        new_ratio = (weighted_req_rate[gpu_id] + model.req_rate / model.slo) / (
                            shared_kv[gpu_id] - model.model_size
                        )
                    if new_ratio < best_ratio:
                        best_ratio, best_idx = new_ratio, gpu_id
            if best_idx is None:
                return None
            placement[best_idx].append(model)
            weighted_req_rate[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        return placement

    def max_kvpr(placement):
        vals = []
        for g in range(gpu_num):
            s = sum(m.model_size for m in placement[g])
            w = sum(m.req_rate / m.slo for m in placement[g])
            vals.append(w / (GPU_MEM_SIZE - s) if s < GPU_MEM_SIZE else float("inf"))
        return max(vals)

    orders = [
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
    ]

    best, best_kvpr = None, float("inf")
    for order in orders:
        p = run_greedy(order)
        if p is None:
            p = run_greedy(order, bestfit=True)  # feasibility fallback
        if p is not None:
            v = max_kvpr(p)
            if v < best_kvpr:
                best_kvpr, best = v, p

    if best is None:
        raise ValueError("Unable to place all models on the available GPUs.")
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
