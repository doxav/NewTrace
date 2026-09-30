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

    import random
    import time

    start = time.time()
    n = len(models)
    if n == 0:
        return {g: [] for g in range(gpu_num)}

    req = [m.req_rate / m.slo for m in models]
    size = [m.model_size for m in models]
    idx_of = {id(m): i for i, m in enumerate(models)}

    def kvpr_of(loads, used):
        best = 0.0
        for g in range(gpu_num):
            denom = GPU_MEM_SIZE - used[g]
            if denom <= 0:
                return float("inf")
            k = loads[g] / denom
            if k > best:
                best = k
        return best

    def build(order, randomized):
        # Greedy: place on GPU minimizing resulting per-GPU KVPR.
        placement = {g: [] for g in range(gpu_num)}
        loads = [0.0] * gpu_num
        used = [0.0] * gpu_num
        for m in order:
            i = idx_of[id(m)]
            cands = []
            for g in range(gpu_num):
                if used[g] + size[i] <= GPU_MEM_SIZE:
                    cands.append(((loads[g] + req[i]) / (GPU_MEM_SIZE - used[g] - size[i]), g))
            if not cands:
                # forced: most free memory
                g = max(range(gpu_num), key=lambda x: GPU_MEM_SIZE - used[x])
            elif randomized:
                cands.sort()
                k = max(1, min(len(cands), int(0.25 * len(cands)) + 1))
                _, g = cands[rng.randrange(k)]
            else:
                g = min(cands)[1]
            placement[g].append(m)
            loads[g] += req[i]
            used[g] += size[i]
        return placement, loads, used

    def local_search(placement, loads, used, best):
        # First-improvement moves/swaps focused on the max-KVPR GPU,
        # with incremental load/memory updates (fast).
        placement = {g: list(ms) for g, ms in placement.items()}
        for _ in range(100):
            src = max(range(gpu_num), key=lambda g: loads[g] / max(GPU_MEM_SIZE - used[g], 1e-12))
            improved = False
            for mi in range(len(placement[src])):
                i = idx_of[id(placement[src][mi])]
                for dst in range(gpu_num):
                    if dst == src:
                        continue
                    # move
                    if used[dst] + size[i] <= GPU_MEM_SIZE:
                        loads[src] -= req[i]; used[src] -= size[i]
                        loads[dst] += req[i]; used[dst] += size[i]
                        s = kvpr_of(loads, used)
                        if s < best - 1e-12:
                            placement[dst].append(placement[src].pop(mi))
                            best = s
                            improved = True
                            break
                        loads[src] += req[i]; used[src] += size[i]
                        loads[dst] -= req[i]; used[dst] -= size[i]
                    # swap
                    done = False
                    for nj in range(len(placement[dst])):
                        j = idx_of[id(placement[dst][nj])]
                        if used[dst] - size[j] + size[i] > GPU_MEM_SIZE:
                            continue
                        if used[src] - size[i] + size[j] > GPU_MEM_SIZE:
                            continue
                        loads[src] += req[j] - req[i]
                        loads[dst] += req[i] - req[j]
                        used[src] += size[j] - size[i]
                        used[dst] += size[i] - size[j]
                        s = kvpr_of(loads, used)
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
            if not improved:
                break
        return placement, loads, used, best

    rng = random.Random(42)
    det_orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / (GPU_MEM_SIZE - m.model_size), reverse=True),
        list(models),
    ]

    best_placement = None
    best_score = float("inf")
    attempts = 0
    while attempts < 40 and time.time() - start < 2.0:
        if attempts < len(det_orders):
            order = det_orders[attempts]
            randomized = False
        else:
            order = list(models)
            rng.shuffle(order)
            randomized = True
        attempts += 1
        placement, loads, used = build(order, randomized)
        placement, loads, used, score = local_search(placement, loads, used, kvpr_of(loads, used))
        if score < best_score:
            best_score = score
            best_placement = placement
        if best_score <= 0:
            break

    if best_placement is None:
        best_placement, _, _ = build(det_orders[1], False)
    return best_placement


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
