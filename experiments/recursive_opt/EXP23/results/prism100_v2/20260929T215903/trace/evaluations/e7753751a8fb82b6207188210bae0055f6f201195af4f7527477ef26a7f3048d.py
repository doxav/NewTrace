GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize the maximum KVPR across GPUs.

    Approach: multiple greedy constructions with different orderings and
    randomized restarts (GRASP), each followed by a best-improvement local
    search over single-model moves and pairwise swaps. Keeps the best
    feasible placement found. Works on model indices to be robust to
    duplicate model objects.
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
    # Map object identity -> index (robust to duplicate model objects)
    idx_of = {}
    for i, m in enumerate(models):
        idx_of.setdefault(id(m), i)

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

    def construct(order, alpha):
        # Randomized greedy: place each model on a GPU with small resulting
        # KVPR; randomness among near-best options when alpha > 0.
        placement = {g: [] for g in range(gpu_num)}
        loads = [0.0] * gpu_num
        used = [0.0] * gpu_num
        for i in order:
            cands = []
            for g in range(gpu_num):
                if used[g] + size[i] <= GPU_MEM_SIZE:
                    denom = GPU_MEM_SIZE - used[g] - size[i]
                    cands.append(((loads[g] + req[i]) / denom, g))
            if not cands:
                # infeasible for this construction; put on emptiest GPU
                g = min(range(gpu_num), key=lambda x: used[x])
            else:
                cands.sort()
                if alpha <= 0:
                    g = cands[0][1]
                else:
                    k = max(1, min(len(cands), int(alpha * len(cands)) + 1))
                    g = cands[rng.randrange(k)][1]
            placement[g].append(models[i])
            loads[g] += req[i]
            used[g] += size[i]
        return placement, loads, used

    def local_search(placement, loads, used):
        best = kvpr_of(loads, used)
        improved = True
        while improved and time.time() - start < 2.0:
            improved = False
            # best-improvement move
            best_move = None
            for src in range(gpu_num):
                for mi in range(len(placement[src])):
                    i = idx_of[id(placement[src][mi])]
                    for dst in range(gpu_num):
                        if dst == src or used[dst] + size[i] > GPU_MEM_SIZE:
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
                i = idx_of[id(placement[src][mi])]
                placement[dst].append(placement[src].pop(mi))
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
                    i1 = idx_of[id(placement[src][mi])]
                    for dst in range(src + 1, gpu_num):
                        for mj in range(len(placement[dst])):
                            i2 = idx_of[id(placement[dst][mj])]
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
                i1 = idx_of[id(placement[src][mi])]
                i2 = idx_of[id(placement[dst][mj])]
                placement[src][mi], placement[dst][mj] = placement[dst][mj], placement[src][mi]
                loads[src] += req[i2] - req[i1]
                loads[dst] += req[i1] - req[i2]
                used[src] += size[i2] - size[i1]
                used[dst] += size[i1] - size[i2]
                best = score
                improved = True
        return placement, loads, used, best

    def ffd_feasible():
        free = [GPU_MEM_SIZE] * gpu_num
        for i in sorted(range(n), key=lambda x: size[x], reverse=True):
            for g in range(gpu_num):
                if size[i] <= free[g]:
                    free[g] -= size[i]
                    break
            else:
                return False
        return True

    candidate_orders = [
        sorted(range(n), key=lambda i: req[i] / max(size[i], 1e-9), reverse=True),
        sorted(range(n), key=lambda i: req[i], reverse=True),
        sorted(range(n), key=lambda i: size[i], reverse=True),
        sorted(range(n), key=lambda i: req[i] / max(GPU_MEM_SIZE - size[i], 1e-9), reverse=True),
    ]

    best_placement = None
    best_score = float("inf")
    restarts = 0
    max_restarts = 60
    while restarts < max_restarts and time.time() - start < 2.0:
        if restarts < len(candidate_orders):
            order = candidate_orders[restarts]
            alpha = 0.0
        else:
            order = list(range(n))
            rng.shuffle(order)
            alpha = rng.random()
        placement, loads, used = construct(order, alpha)
        placement, loads, used, score = local_search(placement, loads, used)
        if score < best_score:
            best_score = score
            best_placement = placement
        restarts += 1

    if best_placement is None:
        # Fallback: FFD by size (feasible when possible), else best-effort
        placement = {g: [] for g in range(gpu_num)}
        free = [GPU_MEM_SIZE] * gpu_num
        for i in sorted(range(n), key=lambda x: size[x], reverse=True):
            for g in range(gpu_num):
                if size[i] <= free[g]:
                    placement[g].append(models[i])
                    free[g] -= size[i]
                    break
            else:
                g = max(range(gpu_num), key=lambda x: free[x])
                placement[g].append(models[i])
                free[g] -= size[i]
        best_placement = placement
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
