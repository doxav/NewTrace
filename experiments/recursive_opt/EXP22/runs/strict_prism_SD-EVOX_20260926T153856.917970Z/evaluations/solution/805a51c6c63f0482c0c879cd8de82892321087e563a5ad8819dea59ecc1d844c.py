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

    """Multi-start greedy placement over several orderings (choosing the GPU
    whose resulting KVPR is lowest, with a most-free-memory fallback), keeping
    the best result, then local-search refinement using moves and swaps to
    minimize max KVPR."""

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

    orders = (
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo / m.model_size, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        list(models),
    )
    best_place = best_w = best_mem = None
    best_max = float("inf")
    for order in orders:
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
        if improved:
            continue
        # Try swaps between src and other GPUs
        for m1 in list(best_place[src]):
            r1, s1 = m1.req_rate / m1.slo, m1.model_size
            for dst in range(gpu_num):
                if dst == src:
                    continue
                for m2 in list(best_place[dst]):
                    s2 = m2.model_size
                    if s2 - s1 > best_mem[src] or s1 - s2 > best_mem[dst]:
                        continue
                    r2 = m2.req_rate / m2.slo
                    new_w = list(best_w)
                    new_mem = list(best_mem)
                    new_w[src] += r2 - r1
                    new_mem[src] += s1 - s2
                    new_w[dst] += r1 - r2
                    new_mem[dst] += s2 - s1
                    new_max = max(kvprs(new_w, new_mem))
                    if new_max < best_max - 1e-12:
                        best_place[src].remove(m1)
                        best_place[dst].remove(m2)
                        best_place[src].append(m2)
                        best_place[dst].append(m1)
                        best_w, best_mem, best_max = new_w, new_mem, new_max
                        improved = True
                        break
                if improved:
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
