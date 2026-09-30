GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Two-phase placement minimizing max KVPR:
    1) Greedy: assign models (sorted by req_rate/slo desc) to the GPU that
       minimizes the resulting KVPR while fitting in memory.
    2) Local search: repeatedly try moving/swapping a model from the most
       loaded GPU to reduce the maximum KVPR, until no improvement exists.
    """

    def kvpr(load, free):
        return load / free if free > 0 else float("inf")

    # --- Phase 1: greedy ---
    sorted_models = sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True)
    placement = {g: [] for g in range(gpu_num)}
    load = [0.0] * gpu_num
    free = [float(GPU_MEM_SIZE)] * gpu_num

    for model in sorted_models:
        best_idx, best_kv = None, float("inf")
        for g in range(gpu_num):
            if model.model_size <= free[g]:
                k = kvpr(load[g] + model.req_rate / model.slo, free[g] - model.model_size)
                if k < best_kv:
                    best_kv, best_idx = k, g
        if best_idx is None:
            raise ValueError(f"Cannot place model of size {model.model_size} GB")
        placement[best_idx].append(model)
        load[best_idx] += model.req_rate / model.slo
        free[best_idx] -= model.model_size

    # --- Phase 2: local search ---
    def max_kvpr():
        return max(kvpr(load[g], free[g]) for g in range(gpu_num))

    improved = True
    while improved:
        improved = False
        cur_max = max_kvpr()
        # GPU with the current max KVPR
        src = max(range(gpu_num), key=lambda g: kvpr(load[g], free[g]))
        # Try moving a model out of src
        for m in list(placement[src]):
            r = m.req_rate / m.slo
            for dst in range(gpu_num):
                if dst == src or m.model_size > free[dst]:
                    continue
                new_src = kvpr(load[src] - r, free[src] + m.model_size)
                new_dst = kvpr(load[dst] + r, free[dst] - m.model_size)
                others = max(kvpr(load[g], free[g]) for g in range(gpu_num) if g not in (src, dst)) if gpu_num > 2 else 0
                if max(new_src, new_dst, others) < cur_max - 1e-12:
                    placement[src].remove(m)
                    placement[dst].append(m)
                    load[src] -= r; free[src] += m.model_size
                    load[dst] += r; free[dst] -= m.model_size
                    improved = True
                    break
            if improved:
                break
        # Try swapping a model from src with one from another GPU
        if not improved:
            for m1 in list(placement[src]):
                r1 = m1.req_rate / m1.slo
                for dst in range(gpu_num):
                    if dst == src:
                        continue
                    for m2 in list(placement[dst]):
                        r2 = m2.req_rate / m2.slo
                        if m2.model_size - m1.model_size > free[src] or m1.model_size - m2.model_size > free[dst]:
                            continue
                        new_src = kvpr(load[src] - r1 + r2, free[src] + m1.model_size - m2.model_size)
                        new_dst = kvpr(load[dst] - r2 + r1, free[dst] + m2.model_size - m1.model_size)
                        others = max(kvpr(load[g], free[g]) for g in range(gpu_num) if g not in (src, dst)) if gpu_num > 2 else 0
                        if max(new_src, new_dst, others) < cur_max - 1e-12:
                            placement[src].remove(m1); placement[src].append(m2)
                            placement[dst].remove(m2); placement[dst].append(m1)
                            load[src] += r2 - r1; free[src] += m1.model_size - m2.model_size
                            load[dst] += r1 - r2; free[dst] += m2.model_size - m1.model_size
                            improved = True
                            break
                    if improved:
                        break
                if improved:
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
