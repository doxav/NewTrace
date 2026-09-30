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

    """Bisection on KVPR threshold T: for each T, greedily check feasibility by
    assigning models (sorted by req_rate/slo desc) to the GPU with most free
    memory such that resulting KVPR <= T and memory fits. Minimize feasible T."""

    def try_threshold(T):
        placement = {g: [] for g in range(gpu_num)}
        mem = [float(GPU_MEM_SIZE)] * gpu_num
        w = [0.0] * gpu_num
        for m in sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True):
            r, s = m.req_rate / m.slo, m.model_size
            cands = [g for g in range(gpu_num) if s <= mem[g] and (w[g] + r) / (mem[g] - s) <= T]
            if not cands:
                return None
            g = max(cands, key=lambda g: mem[g])
            placement[g].append(m)
            w[g] += r
            mem[g] -= s
        return placement

    # Upper bound: greedy best-fit ignoring threshold
    placement = {g: [] for g in range(gpu_num)}
    mem = [float(GPU_MEM_SIZE)] * gpu_num
    w = [0.0] * gpu_num
    for m in sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True):
        r, s = m.req_rate / m.slo, m.model_size
        cands = [g for g in range(gpu_num) if s <= mem[g]]
        g = max(cands, key=lambda g: mem[g]) if cands else 0
        placement[g].append(m)
        w[g] += r
        mem[g] -= s
    hi = max(w[i] / mem[i] if mem[i] > 0 else float("inf") for i in range(gpu_num))
    best = placement
    lo = 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        res = try_threshold(mid)
        if res is not None:
            best, hi = res, mid
        else:
            lo = mid
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
