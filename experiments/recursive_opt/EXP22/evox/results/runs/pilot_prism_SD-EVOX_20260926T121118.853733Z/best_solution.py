GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def _greedy(gpu_num, models, order):
    """Greedy pass: place each model (in given order) on the GPU minimizing
    the resulting KVPR (wrr + r/s) / (mem - size). Returns placement or None."""
    placement = {g: [] for g in range(gpu_num)}
    mem = [GPU_MEM_SIZE] * gpu_num
    wrr = [0.0] * gpu_num
    for model in order:
        best_idx, best_kvpr = None, float("inf")
        for g in range(gpu_num):
            if model.model_size <= mem[g]:
                kvpr = (wrr[g] + model.req_rate / model.slo) / (mem[g] - model.model_size)
                if kvpr < best_kvpr:
                    best_kvpr, best_idx = kvpr, g
        if best_idx is None:
            return None
        placement[best_idx].append(model)
        wrr[best_idx] += model.req_rate / model.slo
        mem[best_idx] -= model.model_size
    return placement


def _max_kvpr(placement, gpu_num):
    """Max KVPR across all GPUs of a placement."""
    mx = 0.0
    for g in range(gpu_num):
        m = GPU_MEM_SIZE - sum(x.model_size for x in placement[g])
        load = sum(x.req_rate / x.slo for x in placement[g])
        if m > 0:
            mx = max(mx, load / m)
    return mx


def compute_model_placement(gpu_num, models):
    """
    Minimize max KVPR via multi-start greedy placement. Tries several model
    orderings (by req_rate/slo desc, model_size desc, and combined keys) and
    keeps the placement with the lowest max KVPR.
    """
    orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo * m.model_size, reverse=True),
        list(models),
    ]
    best, best_score = None, float("inf")
    for order in orders:
        p = _greedy(gpu_num, models, order)
        if p is not None:
            s = _max_kvpr(p, gpu_num)
            if s < best_score:
                best_score, best = s, p
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
