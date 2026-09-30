GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize maximum KVPR via multi-order greedy placement + local search.

    1) Run the KVPR-greedy heuristic under several model orderings and keep
       the best (lowest max KVPR) feasible result.
    2) Refine with a local search: repeatedly move a model off the GPU with
       the highest KVPR to another GPU if doing so lowers the max KVPR.
    3) If a model cannot fit anywhere during greedy assignment, place it on
       the GPU with the most remaining memory (safe fallback, no exception).
    """

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

    def max_kvpr(w, mem):
        return max(wi / mi if mi > 0 else float("inf") for wi, mi in zip(w, mem))

    orders = (
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo / m.model_size, reverse=True),
        list(models),
    )

    best_place = best_w = best_mem = None
    best_max = float("inf")
    for order in orders:
        p, w, mem = run(order)
        mx = max_kvpr(w, mem)
        if mx < best_max:
            best_max, best_place, best_w, best_mem = mx, p, w, mem

    # Local search: move models from the max-KVPR GPU if it lowers max KVPR
    improved = True
    while improved:
        improved = False
        kvs = [best_w[g] / best_mem[g] if best_mem[g] > 0 else float("inf")
               for g in range(gpu_num)]
        src = max(range(gpu_num), key=lambda g: kvs[g])
        for model in list(best_place[src]):
            r = model.req_rate / model.slo
            for dst in range(gpu_num):
                if dst == src or model.model_size > best_mem[dst]:
                    continue
                new_w = list(best_w)
                new_mem = list(best_mem)
                new_w[src] -= r
                new_mem[src] += model.model_size
                new_w[dst] += r
                new_mem[dst] -= model.model_size
                new_max = max_kvpr(new_w, new_mem)
                if new_max < best_max - 1e-12:
                    best_place[src].remove(model)
                    best_place[dst].append(model)
                    best_w, best_mem, best_max = new_w, new_mem, new_max
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
