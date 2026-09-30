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

    """Greedy placement (size-desc, minimize resulting KVPR), then a local-search
    refinement that repeatedly moves/swaps models to reduce the maximum KVPR.
    Refinement starts from a feasible placement and only accepts memory-safe
    improving changes, so success rate is preserved."""

    def greedy():
        sorted_models = sorted(models, key=lambda m: m.model_size, reverse=True)
        placement = {g: [] for g in range(gpu_num)}
        loads = [0.0] * gpu_num
        sizes = [0.0] * gpu_num
        for m in sorted_models:
            best, best_r = None, float("inf")
            for g in range(gpu_num):
                if m.model_size <= GPU_MEM_SIZE - sizes[g]:
                    r = (loads[g] + m.req_rate / m.slo) / (
                        GPU_MEM_SIZE - sizes[g] - m.model_size
                    )
                    if r < best_r:
                        best_r, best = r, g
            if best is None:
                raise ValueError("Unable to place all models on the available GPUs.")
            placement[best].append(m)
            loads[best] += m.req_rate / m.slo
            sizes[best] += m.model_size
        return placement, loads, sizes

    def max_kvpr(loads, sizes):
        return max(
            (loads[g] / (GPU_MEM_SIZE - sizes[g])) if sizes[g] < GPU_MEM_SIZE else float("inf")
            for g in range(gpu_num)
        )

    placement, loads, sizes = greedy()
    cur = max_kvpr(loads, sizes)

    improved = True
    while improved:
        improved = False
        src = max(range(gpu_num), key=lambda g: loads[g] / max(GPU_MEM_SIZE - sizes[g], 1e-9))
        # Try moving one model out of the most pressured GPU
        for m in list(placement[src]):
            w = m.req_rate / m.slo
            for dst in range(gpu_num):
                if dst == src or m.model_size > GPU_MEM_SIZE - sizes[dst]:
                    continue
                loads[src] -= w; sizes[src] -= m.model_size
                loads[dst] += w; sizes[dst] += m.model_size
                new = max_kvpr(loads, sizes)
                if new < cur - 1e-12:
                    placement[src].remove(m); placement[dst].append(m)
                    cur = new; improved = True
                    break
                loads[src] += w; sizes[src] += m.model_size
                loads[dst] -= w; sizes[dst] -= m.model_size
            if improved:
                break
        if improved:
            continue
        # Try pairwise swaps between GPUs
        done = False
        for a in range(gpu_num):
            for b in range(a + 1, gpu_num):
                for ma in list(placement[a]):
                    for mb in list(placement[b]):
                        wa, wb = ma.req_rate / ma.slo, mb.req_rate / mb.slo
                        if (sizes[b] - mb.model_size + ma.model_size > GPU_MEM_SIZE or
                                sizes[a] - ma.model_size + mb.model_size > GPU_MEM_SIZE):
                            continue
                        loads[a] += wb - wa; loads[b] += wa - wb
                        sizes[a] += mb.model_size - ma.model_size
                        sizes[b] += ma.model_size - mb.model_size
                        new = max_kvpr(loads, sizes)
                        if new < cur - 1e-12:
                            placement[a].remove(ma); placement[a].append(mb)
                            placement[b].remove(mb); placement[b].append(ma)
                            cur = new; improved = True; done = True
                            break
                        loads[a] -= wb - wa; loads[b] -= wa - wb
                        sizes[a] -= mb.model_size - ma.model_size
                        sizes[b] -= ma.model_size - mb.model_size
                    if done: break
                if done: break
            if done: break

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
