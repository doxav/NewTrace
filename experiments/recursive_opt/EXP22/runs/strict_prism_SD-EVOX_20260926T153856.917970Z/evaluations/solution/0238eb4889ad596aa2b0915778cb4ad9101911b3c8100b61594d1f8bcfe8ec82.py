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

    """Multi-order greedy minimizing max KVPR with fallback + local search.

    Runs the KVPR-greedy heuristic under several model orderings, keeping the
    best result. If a model fits nowhere, falls back to the GPU with the most
    remaining memory (never raises). Then refines via local search: moves a
    model off the max-KVPR GPU whenever it lowers the max KVPR.
    """

    def kvprs(w, mem):
        return [w[i] / mem[i] if mem[i] > 0 else float("inf") for i in range(len(mem))]

    def run(order):
        placement = {g: [] for g in range(gpu_num)}
        mem = [float(GPU_MEM_SIZE)] * gpu_num
        w = [0.0] * gpu_num
        for model in order:
            r = model.req_rate / model.slo
            best, best_kv = None, float("inf")
            for g in range(gpu_num):
                if model.model_size <= mem[g]:
                    kv = (w[g] + r) / (mem[g] - model.model_size)
                    if kv < best_kv:
                        best_kv, best = kv, g
            if best is None:
                best = max(range(gpu_num), key=lambda g: mem[g])
            placement[best].append(model)
            w[best] += r
            mem[best] -= model.model_size
        return placement, w, mem

    # Try several orderings, keep the best by max KVPR
    by_rate = sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True)
    by_ratio = sorted(models, key=lambda m: m.req_rate / m.slo / m.model_size, reverse=True)
    by_size = sorted(models, key=lambda m: m.model_size, reverse=True)

    best_place = best_w = best_mem = None
    best_max = float("inf")
    for order in (by_rate, by_ratio, by_size, models):
        p, w, mem = run(order)
        mx = max(kvprs(w, mem))
        if mx < best_max:
            best_max, best_place, best_w, best_mem = mx, p, w, mem

    # Local search: move a model to another GPU if it lowers max KVPR
    improved = True
    while improved:
        improved = False
        kvs = kvprs(best_w, best_mem)
        src = max(range(gpu_num), key=lambda g: kvs[g])
        for model in list(best_place[src]):
            r = model.req_rate / model.slo
            for dst in range(gpu_num):
                if dst == src or model.model_size > best_mem[dst]:
                    continue
                # simulate move
                new_w = list(best_w)
                new_mem = list(best_mem)
                new_w[src] -= r
                new_mem[src] += model.model_size
                new_w[dst] += r
                new_mem[dst] -= model.model_size
                new_max = max(kvprs(new_w, new_mem))
                if new_max < best_max - 1e-12:
                    best_place[src].remove(model)
                    best_place[dst].append(model)
                    best_w, best_mem = new_w, new_mem
                    best_max = new_max
                    improved = True
                    break
            if improved:
                break

    return best_place


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
