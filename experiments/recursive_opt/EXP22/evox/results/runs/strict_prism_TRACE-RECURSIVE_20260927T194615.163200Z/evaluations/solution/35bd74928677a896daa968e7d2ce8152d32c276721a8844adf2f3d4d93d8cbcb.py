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

    """Greedy placement over two sort orders (size desc, load desc); each model
    is assigned to the GPU minimizing the RESULTING KVPR. The best placement is
    then refined by a local search that moves one model at a time off the
    most-pressured GPU whenever the move strictly lowers the maximum KVPR."""

    def greedy(order):
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
                return None
            placement[best_idx].append(model)
            weighted[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        return placement, weighted, shared_kv

    def kvprs(weighted, shared_kv):
        return [
            (weighted[g] / shared_kv[g]) if shared_kv[g] > 0 else float("inf")
            for g in range(gpu_num)
        ]

    def refine(placement, weighted, shared_kv):
        """Local search: repeatedly move one model out of the GPU with the
        highest KVPR, accepting only moves that strictly reduce max KVPR."""
        while True:
            vals = kvprs(weighted, shared_kv)
            cur = max(vals)
            src = vals.index(cur)
            improved = False
            for m in list(placement[src]):
                w = m.req_rate / m.slo
                for dst in range(gpu_num):
                    if dst == src or m.model_size > shared_kv[dst]:
                        continue
                    weighted[src] -= w; shared_kv[src] += m.model_size
                    weighted[dst] += w; shared_kv[dst] -= m.model_size
                    if max(kvprs(weighted, shared_kv)) < cur - 1e-12:
                        placement[src].remove(m); placement[dst].append(m)
                        improved = True
                        break
                    weighted[src] += w; shared_kv[src] -= m.model_size
                    weighted[dst] -= w; shared_kv[dst] += m.model_size
                if improved:
                    break
            if not improved:
                return

    best, best_kvpr = None, float("inf")
    for order in (
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
    ):
        res = greedy(order)
        if res is None:
            continue
        p, weighted, shared_kv = res
        refine(p, weighted, shared_kv)
        v = max(kvprs(weighted, shared_kv))
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
