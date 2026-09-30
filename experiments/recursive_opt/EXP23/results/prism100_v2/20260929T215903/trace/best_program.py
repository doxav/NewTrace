GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize the maximum KVPR across GPUs.

    Approach: randomized greedy constructions (GRASP) followed by a
    best-improvement local search over moves and swaps, keeping the best
    feasible placement found.
    """
    import random
    import time

    start = time.time()
    rng = random.Random(42)
    n = len(models)
    if n == 0:
        return {g: [] for g in range(gpu_num)}

    req = [m.req_rate / m.slo for m in models]
    size = [m.model_size for m in models]

    def kvpr_of(loads, used):
        best = 0.0
        for g in range(gpu_num):
            denom = GPU_MEM_SIZE - used[g]
            if denom <= 0:
                return float("inf")
            kvpr = loads[g] / denom
            if kvpr > best:
                best = kvpr
        return best

    def construct(alpha):
        # Randomized greedy: each model goes to one of the GPUs with the
        # smallest resulting KVPR, with randomness among near-best options.
        placement = {g: [] for g in range(gpu_num)}
        loads = [0.0] * gpu_num
        used = [0.0] * gpu_num
        order = sorted(range(n), key=lambda i: req[i] / max(size[i], 1e-9), reverse=True)
        for i in order:
            cands = []
            for g in range(gpu_num):
                if used[g] + size[i] <= GPU_MEM_SIZE:
                    denom = GPU_MEM_SIZE - used[g] - size[i]
                    cands.append(((loads[g] + req[i]) / denom, g))
            if not cands:
                # infeasible for this construction; put on emptiest GPU
                g = min(range(gpu_num), key=lambda x: used[x])
                placement[g].append(models[i])
                loads[g] += req[i]
                used[g] += size[i]
                continue
            cands.sort()
            k = max(1, min(len(cands), int(alpha * len(cands)) + 1))
            _, g = cands[rng.randrange(k)]
            placement[g].append(models[i])
            loads[g] += req[i]
            used[g] += size[i]
        return placement, loads, used

    def local_search(placement, loads, used):
        placement = {g: list(ms) for g, ms in placement.items()}
        best = kvpr_of(loads, used)
        improved = True
        while improved and time.time() - start < 1.5:
            improved = False
            # best-improvement move
            best_move = None
            for src in range(gpu_num):
                for mi in range(len(placement[src])):
                    m = placement[src][mi]
                    i = models.index(m)
                    for dst in range(gpu_num):
                        if dst == src:
                            continue
                        if used[dst] + size[i] > GPU_MEM_SIZE:
                            continue
                        loads[src] -= req[i]
                        loads[dst] += req[i]
                        used[src] -= size[i]
                        used[dst] += size[i]
                        score = kvpr_of(loads, used)
                        if score < best - 1e-12 and (best_move is None or score < best_move[0]):
                            best_move = (score, src, mi, dst)
                        loads[src] += req[i]
                        loads[dst] -= req[i]
                        used[src] += size[i]
                        used[dst] -= size[i]
            if best_move:
                score, src, mi, dst = best_move
                m = placement[src][mi]
                i = models.index(m)
                placement[src].pop(mi)
                placement[dst].append(m)
                loads[src] -= req[i]
                loads[dst] += req[i]
                used[src] -= size[i]
                used[dst] += size[i]
                best = score
                improved = True
                continue
            # best-improvement swap
            best_swap = None
            for src in range(gpu_num):
                for mi in range(len(placement[src])):
                    m1 = placement[src][mi]
                    i1 = models.index(m1)
                    for dst in range(src + 1, gpu_num):
                        for mj in range(len(placement[dst])):
                            m2 = placement[dst][mj]
                            i2 = models.index(m2)
                            if used[dst] - size[i2] + size[i1] > GPU_MEM_SIZE:
                                continue
                            if used[src] - size[i1] + size[i2] > GPU_MEM_SIZE:
                                continue
                            loads[src] += req[i2] - req[i1]
                            loads[dst] += req[i1] - req[i2]
                            used[src] += size[i2] - size[i1]
                            used[dst] += size[i1] - size[i2]
                            score = kvpr_of(loads, used)
                            if score < best - 1e-12 and (best_swap is None or score < best_swap[0]):
                                best_swap = (score, src, mi, dst, mj)
                            loads[src] -= req[i2] - req[i1]
                            loads[dst] -= req[i1] - req[i2]
                            used[src] -= size[i2] - size[i1]
                            used[dst] -= size[i1] - size[i2]
            if best_swap:
                score, src, mi, dst, mj = best_swap
                m1, m2 = placement[src][mi], placement[dst][mj]
                i1, i2 = models.index(m1), models.index(m2)
                placement[src][mi], placement[dst][mj] = m2, m1
                loads[src] += req[i2] - req[i1]
                loads[dst] += req[i1] - req[i2]
                used[src] += size[i2] - size[i1]
                used[dst] += size[i1] - size[i2]
                best = score
                improved = True
        return placement, loads, used, best

    best_placement = None
    best_score = float("inf")
    restarts = 0
    while restarts < 60 and time.time() - start < 1.8:
        alpha = 0.0 if restarts < 5 else rng.random()
        placement, loads, used = construct(alpha)
        placement, loads, used, score = local_search(placement, loads, used)
        if score < best_score:
            best_score = score
            best_placement = placement
        restarts += 1

    if best_placement is None:
        best_placement = {g: [] for g in range(gpu_num)}
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
