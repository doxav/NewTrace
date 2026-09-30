GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Multi-start randomized greedy: each restart places models in a randomized
    order (biased toward high req_rate/slo), each on the GPU minimizing the
    resulting max KVPR; keeps the best placement, then refines with a
    relocation local search.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """
    import random

    def kvprs(w, mem):
        return [w[i] / mem[i] if mem[i] > 0 else float("inf") for i in range(len(mem))]

    def greedy(order):
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

    def refine(place, w, mem, cur_max):
        improved = True
        while improved:
            improved = False
            kvs = kvprs(w, mem)
            src = max(range(gpu_num), key=lambda g: kvs[g])
            for model in list(place[src]):
                r = model.req_rate / model.slo
                for dst in range(gpu_num):
                    if dst == src or model.model_size > mem[dst]:
                        continue
                    nw = list(w); nm = list(mem)
                    nw[src] -= r; nm[src] += model.model_size
                    nw[dst] += r; nm[dst] -= model.model_size
                    nmax = max(kvprs(nw, nm))
                    if nmax < cur_max - 1e-12:
                        place[src].remove(model); place[dst].append(model)
                        w, mem, cur_max = nw, nm, nmax
                        improved = True
                        break
                if improved:
                    break
        return place, w, mem, cur_max

    heavy = sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True)
    best_place, best_w, best_mem = greedy(heavy)
    best_max = max(kvprs(best_w, best_mem))

    rng = random.Random(42)
    for _ in range(30):
        order = list(models)
        # randomized perturbation: shuffle small-weight models, keep heavy first
        k = max(1, len(order) // 2)
        order[:k] = heavy[:k]
        tail = order[k:]
        rng.shuffle(tail)
        order[k:] = tail
        place, w, mem = greedy(order)
        mx = max(kvprs(w, mem))
        if mx < best_max - 1e-12:
            place, w, mem, mx = refine(place, w, mem, mx)
            if mx < best_max - 1e-12:
                best_place, best_w, best_mem, best_max = place, w, mem, mx

    best_place, best_w, best_mem, best_max = refine(
        best_place, best_w, best_mem, best_max)
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
