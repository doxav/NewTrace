GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def _max_kvpr(placement, gpu_num):
    """Compute max KVPR across GPUs for a placement."""
    vals = [
        sum(m.req_rate / m.slo for m in placement[g])
        / (GPU_MEM_SIZE - sum(m.model_size for m in placement[g]))
        for g in range(gpu_num)
        if placement[g]
    ]
    return max(vals) if vals else 0.0


def _greedy(gpu_num, models, key, reverse):
    """Greedy pass: sort by key, assign each model to the GPU minimizing the
    post-assignment KVPR (w + r) / (mem - s) among GPUs where it fits.
    Returns placement or None if infeasible."""
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


def _local_search(placement, gpu_num, iters=40):
    """Hill-climb: repeatedly move one model to another GPU, or swap two
    models across GPUs, if it lowers max KVPR and keeps memory feasible."""
    mem = {g: GPU_MEM_SIZE - sum(m.model_size for m in placement[g]) for g in range(gpu_num)}
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
                for g2 in range(gpu_num):
                    if g2 <= g1:
                        continue
                    for m2 in list(placement[g2]):
                        d = m1.model_size - m2.model_size
                        if mem[g1] + d < 0 or mem[g2] - d < 0:
                            continue
                        placement[g1].remove(m1); placement[g2].remove(m2)
                        placement[g1].append(m2); placement[g2].append(m1)
                        mem[g1] += d; mem[g2] -= d
                        if _max_kvpr(placement, gpu_num) < cur - 1e-12:
                            improved = True
                            break
                        placement[g2].remove(m1); placement[g1].remove(m2)
                        placement[g1].append(m1); placement[g2].append(m2)
                        mem[g1] -= d; mem[g2] += d
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return placement


def compute_model_placement(gpu_num, models):
    """
    Minimize max KVPR: run greedy placement using post-assignment KVPR
    scoring with multiple sort orders, refine the best with a move-only
    local search. Fall back to size-descending first-fit for feasibility.
    """
    best, best_kvpr = None, float("inf")
    for key, reverse in [
        (lambda m: m.req_rate / m.slo, True),
        (lambda m: m.model_size, True),
        (lambda m: m.req_rate / m.slo, False),
    ]:
        p = _greedy(gpu_num, models, key, reverse)
        if p is not None:
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
    return _local_search(best, gpu_num)


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
