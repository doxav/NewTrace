GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def _max_kvpr(placement, gpu_num):
    """Max KVPR across GPUs for a placement."""
    vals = [
        sum(m.req_rate / m.slo for m in placement[g])
        / (GPU_MEM_SIZE - sum(m.model_size for m in placement[g]))
        for g in range(gpu_num)
        if placement[g]
    ]
    return max(vals) if vals else 0.0


def _local_search(placement, gpu_num, iters=30):
    """Move-only hill-climb: relocate a single model to another GPU when it
    lowers max KVPR and memory stays feasible. Few iterations keep it fast."""
    mem = [GPU_MEM_SIZE - sum(m.model_size for m in placement[g]) for g in range(gpu_num)]
    for _ in range(iters):
        cur = _max_kvpr(placement, gpu_num)
        improved = False
        for g1 in range(gpu_num):
            for m1 in list(placement[g1]):
                for g2 in range(gpu_num):
                    if g2 == g1 or m1.model_size > mem[g2]:
                        continue
                    placement[g1].remove(m1)
                    placement[g2].append(m1)
                    mem[g1] += m1.model_size
                    mem[g2] -= m1.model_size
                    if _max_kvpr(placement, gpu_num) < cur - 1e-12:
                        improved = True
                        break
                    placement[g2].remove(m1)
                    placement[g1].append(m1)
                    mem[g2] += m1.model_size
                    mem[g1] -= m1.model_size
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return placement


def compute_model_placement(gpu_num, models):
    """
    Greedy KVPR-minimizing placement: sort models by req_rate/slo descending,
    assign each model to the GPU with the lowest current KVPR (w / remaining
    mem) among GPUs where it fits; force-place on the GPU with most remaining
    memory if none fits (raise only if truly impossible). Then refine with a
    light move-only local search to further reduce max KVPR.
    """
    placement = {g: [] for g in range(gpu_num)}
    mem = [GPU_MEM_SIZE] * gpu_num
    w = [0.0] * gpu_num
    for m in sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True):
        best, best_ratio = None, float("inf")
        for g in range(gpu_num):
            if m.model_size <= mem[g]:
                ratio = w[g] / mem[g]
                if ratio < best_ratio:
                    best, best_ratio = g, ratio
        if best is None:
            best = max(range(gpu_num), key=lambda g: mem[g])
            if m.model_size > mem[best]:
                raise ValueError(
                    f"Unable to place model of size {m.model_size} GB on any GPU. "
                    f"Remaining per-GPU memory: {mem}"
                )
        placement[best].append(m)
        w[best] += m.req_rate / m.slo
        mem[best] -= m.model_size
    return _local_search(placement, gpu_num)


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
