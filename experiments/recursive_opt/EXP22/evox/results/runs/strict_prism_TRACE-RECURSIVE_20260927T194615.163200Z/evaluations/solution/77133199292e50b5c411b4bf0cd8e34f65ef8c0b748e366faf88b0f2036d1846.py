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

    """Multi-order greedy placement; keep the placement whose maximum KVPR
    is lowest. Each greedy pass assigns each model to the feasible GPU
    minimizing the RESULTING KVPR (w + r) / (mem - s), which directly
    targets the objective. A best-fit fallback by descending size (least
    leftover memory) guarantees feasibility whenever a placement exists."""

    def greedy(order, bestfit=False):
        placement = {g: [] for g in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE] * gpu_num
        weighted = [0.0] * gpu_num
        for model in order:
            best_idx, best_key = None, float("inf")
            for g in range(gpu_num):
                if model.model_size <= shared_kv[g]:
                    if bestfit:
                        key = shared_kv[g] - model.model_size
                    else:
                        key = (weighted[g] + model.req_rate / model.slo) / (
                            shared_kv[g] - model.model_size
                        )
                    if key < best_key:
                        best_key, best_idx = key, g
            if best_idx is None:
                return None  # infeasible for this order
            placement[best_idx].append(model)
            weighted[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        return placement

    def max_kvpr(placement):
        vals = []
        for g in range(gpu_num):
            w = sum(m.req_rate / m.slo for m in placement[g])
            s = sum(m.model_size for m in placement[g])
            vals.append(w / (GPU_MEM_SIZE - s) if s < GPU_MEM_SIZE else float("inf"))
        return max(vals)

    orders = [
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True),
    ]

    best = None
    best_kvpr = float("inf")
    for order in orders:
        for bf in (False, True):
            p = greedy(order, bestfit=bf)
            if p is not None:
                v = max_kvpr(p)
                if v < best_kvpr:
                    best_kvpr, best = v, p

    if best is None:
        # Last resort: best-fit by descending size; succeeds whenever any
        # feasible placement exists.
        p = {g: [] for g in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE] * gpu_num
        for model in sorted(models, key=lambda m: m.model_size, reverse=True):
            cands = [g for g in range(gpu_num) if model.model_size <= shared_kv[g]]
            if not cands:
                raise ValueError("Unable to place all models on the available GPUs.")
            g = min(cands, key=lambda g: shared_kv[g])
            p[g].append(model)
            shared_kv[g] -= model.model_size
        return p

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
