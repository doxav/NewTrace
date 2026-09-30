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

    """Greedy placement (models sorted by req_rate/slo descending; each model
    goes to the feasible GPU with the lowest CURRENT KVPR), followed by a
    local search that repeatedly applies single-model moves and pairwise
    swaps whenever they strictly reduce the maximum KVPR."""

    def max_kvpr(weighted, shared_kv):
        return max(
            w / mem if mem > 0 else float("inf")
            for w, mem in zip(weighted, shared_kv)
        )

    # Greedy: assign each model to the feasible GPU with minimum current KVPR.
    placement = {g: [] for g in range(gpu_num)}
    shared_kv = [GPU_MEM_SIZE] * gpu_num
    weighted = [0.0] * gpu_num
    for model in sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True):
        best_idx, best_ratio = None, float("inf")
        for g in range(gpu_num):
            if model.model_size <= shared_kv[g] and shared_kv[g] > 0:
                ratio = weighted[g] / shared_kv[g]
                if ratio < best_ratio:
                    best_ratio, best_idx = ratio, g
        if best_idx is None:
            raise ValueError("Unable to place all models on the available GPUs.")
        placement[best_idx].append(model)
        weighted[best_idx] += model.req_rate / model.slo
        shared_kv[best_idx] -= model.model_size

    # Local search: moves out of the most pressured GPU, then pairwise swaps.
    improved = True
    while improved:
        improved = False
        cur = max_kvpr(weighted, shared_kv)
        src = max(range(gpu_num), key=lambda g: weighted[g] / shared_kv[g])
        for m in list(placement[src]):
            w = m.req_rate / m.slo
            for dst in range(gpu_num):
                if dst == src or m.model_size > shared_kv[dst]:
                    continue
                weighted[src] -= w; shared_kv[src] += m.model_size
                weighted[dst] += w; shared_kv[dst] -= m.model_size
                if max_kvpr(weighted, shared_kv) < cur - 1e-12:
                    placement[src].remove(m); placement[dst].append(m)
                    improved = True
                    break
                weighted[src] += w; shared_kv[src] -= m.model_size
                weighted[dst] -= w; shared_kv[dst] += m.model_size
            if improved:
                break
        if improved:
            continue
        for a in range(gpu_num):
            for b in range(a + 1, gpu_num):
                done = False
                for ma in list(placement[a]):
                    for mb in list(placement[b]):
                        wa, wb = ma.req_rate / ma.slo, mb.req_rate / mb.slo
                        if (shared_kv[b] + mb.model_size - ma.model_size < 0 or
                                shared_kv[a] + ma.model_size - mb.model_size < 0):
                            continue
                        weighted[a] += wb - wa; weighted[b] += wa - wb
                        shared_kv[a] += ma.model_size - mb.model_size
                        shared_kv[b] += mb.model_size - ma.model_size
                        if max_kvpr(weighted, shared_kv) < cur - 1e-12:
                            placement[a].remove(ma); placement[a].append(mb)
                            placement[b].remove(mb); placement[b].append(ma)
                            improved = done = True
                            break
                        weighted[a] -= wb - wa; weighted[b] -= wa - wb
                        shared_kv[a] -= ma.model_size - mb.model_size
                        shared_kv[b] -= mb.model_size - ma.model_size
                    if done: break
                if done: break
            if improved: break

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
