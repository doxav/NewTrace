GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize the maximum KVPR across GPUs using binary search on the answer
    (a KVPR threshold T) with a greedy feasibility check, plus a light
    move/swap local search on the best found placement.
    """
    import random

    n = len(models)
    if n == 0:
        return {g: [] for g in range(gpu_num)}

    req = [m.req_rate / m.slo for m in models]
    size = [m.model_size for m in models]
    order_by_size = sorted(range(n), key=lambda i: size[i], reverse=True)

    def feasible(T):
        """Can we pack all models so every GPU's KVPR <= T?
        Each GPU g has a memory budget 80 and a load budget T * (80 - used).
        Greedy: place each model (largest first) on a GPU that can hold it
        while keeping load[g] + req <= T * (80 - used[g] - size)."""
        used = [0.0] * gpu_num
        load = [0.0] * gpu_num
        for i in order_by_size:
            placed = False
            # prefer GPU with most remaining "effective capacity" for this model
            best_g, best_key = None, None
            for g in range(gpu_num):
                free = GPU_MEM_SIZE - used[g] - size[i]
                if free < 0:
                    continue
                if load[g] + req[i] <= T * free + 1e-9:
                    # choose the GPU where the model is "least tight"
                    key = T * free - (load[g] + req[i])
                    if best_key is None or key > best_key:
                        best_key, best_g = key, g
                        placed = True
            if not placed:
                return None
            load[best_g] += req[i]
            used[best_g] += size[i]
        return used[:], load[:]

    def max_kvpr(loads, used):
        best = 0.0
        for g in range(gpu_num):
            d = GPU_MEM_SIZE - used[g]
            if d <= 0:
                return float("inf")
            best = max(best, loads[g] / d)
        return best

    # --- Binary search on T ---
    total_req = sum(req)
    lo = total_req / (gpu_num * GPU_MEM_SIZE)  # lower bound
    hi = max_kvpr([total_req], [0.0])  # all on one GPU: an upper bound
    best_pack = None
    for _ in range(40):
        mid = (lo + hi) / 2.0
        res = feasible(mid)
        if res is not None:
            best_pack = res
            hi = mid
        else:
            lo = mid

    # Build placement from best_pack (or fallback)
    def build(used, load):
        placement = {g: [] for g in range(gpu_num)}
        # reconstruct by re-running feasible at threshold hi (guaranteed feasible)
        u = [0.0] * gpu_num
        l = [0.0] * gpu_num
        T = hi * (1 + 1e-9)
        for i in order_by_size:
            best_g, best_key = None, None
            for g in range(gpu_num):
                free = GPU_MEM_SIZE - u[g] - size[i]
                if free < 0:
                    continue
                if l[g] + req[i] <= T * free + 1e-9:
                    key = T * free - (l[g] + req[i])
                    if best_key is None or key > best_key:
                        best_key, best_g = key, g
            if best_g is None:
                best_g = max(range(gpu_num), key=lambda g: GPU_MEM_SIZE - u[g])
            placement[best_g].append(models[i])
            l[best_g] += req[i]
            u[best_g] += size[i]
        return placement, l, u

    placement, loads, used = build(best_pack, None)

    # --- Light local search: moves and swaps, first-improvement ---
    rng = random.Random(42)
    best = max_kvpr(loads, used)
    for _ in range(300):
        improved = False
        for src in range(gpu_num):
            for mi in range(len(placement[src])):
                i = models.index(placement[src][mi])
                for dst in range(gpu_num):
                    if dst == src:
                        continue
                    # try move
                    if used[dst] + size[i] <= GPU_MEM_SIZE:
                        loads[src] -= req[i]; used[src] -= size[i]
                        loads[dst] += req[i]; used[dst] += size[i]
                        s = max_kvpr(loads, used)
                        if s < best - 1e-12:
                            m = placement[src].pop(mi)
                            placement[dst].append(m)
                            best = s
                            improved = True
                            break
                        loads[src] += req[i]; used[src] += size[i]
                        loads[dst] -= req[i]; used[dst] -= size[i]
                    # try swap
                    done = False
                    for nj in range(len(placement[dst])):
                        j = models.index(placement[dst][nj])
                        if used[dst] - size[j] + size[i] > GPU_MEM_SIZE:
                            continue
                        if used[src] - size[i] + size[j] > GPU_MEM_SIZE:
                            continue
                        loads[src] += req[j] - req[i]
                        loads[dst] += req[i] - req[j]
                        used[src] += size[j] - size[i]
                        used[dst] += size[i] - size[j]
                        s = max_kvpr(loads, used)
                        if s < best - 1e-12:
                            placement[src][mi], placement[dst][nj] = placement[dst][nj], placement[src][mi]
                            best = s
                            improved = True
                            done = True
                            break
                        loads[src] -= req[j] - req[i]
                        loads[dst] -= req[i] - req[j]
                        used[src] -= size[j] - size[i]
                        used[dst] -= size[i] - size[j]
                    if done:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
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
