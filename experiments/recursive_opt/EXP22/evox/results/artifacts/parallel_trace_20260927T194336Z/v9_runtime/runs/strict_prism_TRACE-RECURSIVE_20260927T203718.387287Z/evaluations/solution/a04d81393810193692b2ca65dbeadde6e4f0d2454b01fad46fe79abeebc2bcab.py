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

    """Greedy + local search placement minimizing max KVPR.

    Greedy: sort models by size descending, assign each to the feasible
    GPU minimizing the resulting KVPR ((w + r/s) / (mem - size)); if no
    GPU fits, fall back to the GPU with the most free memory (guarantees
    a valid placement). Then local search: repeatedly move a model off
    the most loaded GPU when the move lowers the max KVPR of the two
    affected GPUs.
    """
    placement = {g: [] for g in range(gpu_num)}
    w = [0.0] * gpu_num   # sum of req_rate/slo per GPU
    mem = [float(GPU_MEM_SIZE)] * gpu_num  # free memory per GPU

    def kvpr(g):
        return w[g] / mem[g] if mem[g] > 0 else float("inf")

    # Greedy: larger models first, choose GPU minimizing resulting KVPR
    for m in sorted(models, key=lambda x: -x.model_size):
        cands = [g for g in range(gpu_num) if m.model_size <= mem[g]]
        if not cands:
            best = max(range(gpu_num), key=lambda g: mem[g])
        else:
            best = min(cands, key=lambda g: (w[g] + m.req_rate / m.slo) / (mem[g] - m.model_size))
        placement[best].append(m)
        w[best] += m.req_rate / m.slo
        mem[best] -= m.model_size

    # Local search: single moves off the most loaded GPU
    for _ in range(300):
        src = max(range(gpu_num), key=kvpr)
        cur = max(kvpr(x) for x in range(gpu_num))
        improved = False
        for m in list(placement[src]):
            r = m.req_rate / m.slo
            for g in range(gpu_num):
                if g == src or m.model_size > mem[g]:
                    continue
                new_max = max((w[src] - r) / (mem[src] + m.model_size),
                              (w[g] + r) / (mem[g] - m.model_size))
                if new_max < cur - 1e-12:
                    placement[src].remove(m); placement[g].append(m)
                    w[src] -= r; mem[src] += m.model_size
                    w[g] += r; mem[g] -= m.model_size
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return placement


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
