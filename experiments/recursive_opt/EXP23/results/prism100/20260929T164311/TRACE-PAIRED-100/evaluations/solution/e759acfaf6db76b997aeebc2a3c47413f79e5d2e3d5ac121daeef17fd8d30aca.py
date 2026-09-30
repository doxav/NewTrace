GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def _greedy(gpu_num, models, key, reverse):
    """Greedy pass: sort by key, assign each model to the GPU minimizing
    the resulting KVPR (w + r)/(mem - s). Returns placement or None."""
    placement = {g: [] for g in range(gpu_num)}
    mem = [GPU_MEM_SIZE] * gpu_num
    w = [0.0] * gpu_num
    for m in sorted(models, key=key, reverse=reverse):
        r = m.req_rate / m.slo
        best, best_ratio = None, float("inf")
        for g in range(gpu_num):
            if m.model_size <= mem[g]:
                ratio = (w[g] + r) / (mem[g] - m.model_size)
                if ratio < best_ratio:
                    best, best_ratio = g, ratio
        if best is None:
            return None
        placement[best].append(m)
        w[best] += r
        mem[best] -= m.model_size
    return placement


def _max_kvpr(placement, gpu_num):
    """Max KVPR across GPUs for a placement."""
    vals = [
        sum(m.req_rate / m.slo for m in placement[g])
        / (GPU_MEM_SIZE - sum(m.model_size for m in placement[g]))
        for g in range(gpu_num)
        if placement[g]
    ]
    return max(vals) if vals else 0.0


def compute_model_placement(gpu_num, models):
    """
    Minimize max KVPR: run greedy placement (post-assignment KVPR scoring,
    multiple sort orders), keep the best valid result. Fall back to
    size-descending first-fit if no greedy variant fits.
    """
    best, best_kvpr = None, float("inf")
    for key, reverse in [
        (lambda m: m.req_rate / m.slo, True),
        (lambda m: m.req_rate / m.slo, False),
        (lambda m: m.model_size, True),
        (lambda m: m.model_size, False),
        (lambda m: m.req_rate, True),
    ]:
        p = _greedy(gpu_num, models, key, reverse)
        if p is None:
            continue
        kvpr = _max_kvpr(p, gpu_num)
        if kvpr < best_kvpr:
            best_kvpr, best = kvpr, p
    if best is None:
        # Feasibility fallback: size-descending first-fit
        best, mem = {g: [] for g in range(gpu_num)}, [GPU_MEM_SIZE] * gpu_num
        for m in sorted(models, key=lambda m: m.model_size, reverse=True):
            g = next((g for g in range(gpu_num) if m.model_size <= mem[g]), None)
            if g is None:
                raise ValueError("Unable to place all models on GPUs")
            best[g].append(m)
            mem[g] -= m.model_size
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
