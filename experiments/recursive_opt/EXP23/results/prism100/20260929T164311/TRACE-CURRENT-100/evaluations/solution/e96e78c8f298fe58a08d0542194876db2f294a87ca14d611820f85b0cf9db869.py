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

    def max_kvpr(placement):
        """Max KVPR across GPUs; epsilon guards against zero free memory."""
        if placement is None:
            return float("inf")
        return max(
            sum(m.req_rate / m.slo for m in ms)
            / max(GPU_MEM_SIZE - sum(m.model_size for m in ms), 1e-9)
            for ms in placement.values()
        )

    def greedy(order):
        """Greedy: assign each model to GPU minimizing resulting KVPR."""
        placement = {g: [] for g in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for model in order:
            best_idx, best_kvpr = None, float("inf")
            for g in range(gpu_num):
                if shared_kv[g] - model.model_size > 1e-9:
                    kvpr = (load[g] + model.req_rate / model.slo) / (
                        shared_kv[g] - model.model_size
                    )
                    if kvpr < best_kvpr:
                        best_kvpr, best_idx = kvpr, g
            if best_idx is None:
                return None
            placement[best_idx].append(model)
            load[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        return placement

    def improve(p):
        """Local search: move models off the max-KVPR GPU if it lowers max KVPR."""
        for _ in range(50):
            cur = max_kvpr(p)
            src = max(p, key=lambda g: max_kvpr({g: p[g]}))
            for m in list(p[src]):
                for g in range(gpu_num):
                    if g == src or m.model_size > GPU_MEM_SIZE - sum(x.model_size for x in p[g]):
                        continue
                    p[src].remove(m)
                    p[g].append(m)
                    if max_kvpr(p) < cur - 1e-12:
                        break
                    p[g].remove(m)
                    p[src].append(m)
                else:
                    continue
                break
            else:
                break
        return p

    def first_fit():
        """Fallback: first-fit-decreasing by size for guaranteed feasibility."""
        placement = {g: [] for g in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE] * gpu_num
        for model in sorted(models, key=lambda m: m.model_size, reverse=True):
            for g in range(gpu_num):
                if model.model_size <= shared_kv[g]:
                    placement[g].append(model)
                    shared_kv[g] -= model.model_size
                    break
            else:
                return None
        return placement

    # Try several orderings; keep the one minimizing max KVPR
    candidates = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo / m.model_size, reverse=True),
    ]
    best = None
    best_kvpr = float("inf")
    for order in candidates:
        p = greedy(order)
        kvpr = max_kvpr(p)
        if kvpr < best_kvpr:
            best, best_kvpr = p, kvpr
    if best is None:
        # All greedy orderings failed; recover feasibility via FFD by size.
        best = improve(first_fit())
    if best is None:
        raise ValueError("Unable to place all models on GPUs")
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
