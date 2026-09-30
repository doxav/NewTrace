GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Greedy KVPR-minimizing placement: sort models by req_rate/slo descending,
    then assign each model to the GPU minimizing the resulting KVPR
    ((w + r) / (mem - s)) among GPUs where it fits. Falls back to
    placing on the GPU with most remaining memory if no greedy choice
    fits, ensuring a valid placement is always returned when feasible.
    """
    placement = {g: [] for g in range(gpu_num)}
    mem = [GPU_MEM_SIZE] * gpu_num
    w = [0.0] * gpu_num
    for m in sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True):
        r = m.req_rate / m.slo
        best, best_ratio = None, float("inf")
        for g in range(gpu_num):
            if m.model_size <= mem[g]:
                ratio = (w[g] + r) / (mem[g] - m.model_size)
                if ratio < best_ratio:
                    best, best_ratio = g, ratio
        if best is None:
            # Fallback: place on GPU with most remaining memory
            best = max(range(gpu_num), key=lambda g: mem[g])
            if m.model_size > mem[best]:
                raise ValueError(
                    f"Unable to place model of size {m.model_size} GB on any GPU. "
                    f"Remaining per-GPU memory: {mem}"
                )
        placement[best].append(m)
        w[best] += r
        mem[best] -= m.model_size
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
