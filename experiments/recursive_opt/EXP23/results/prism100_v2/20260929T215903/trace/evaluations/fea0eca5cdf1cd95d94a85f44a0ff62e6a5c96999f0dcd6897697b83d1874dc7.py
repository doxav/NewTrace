GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize the maximum KVPR across GPUs using GRASP-style randomized
    greedy construction followed by best-improvement move/swap local search.
    """
    import random
    import time

    start = time.time()
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

    def construct(order, alpha):
        placement = {g: [] for g in range(gpu_num)}
        loads = [0.0] * gpu_num
        used = [0.0] * gpu_num
        for i in order:
            cands = []
            for g in range(gpu_num):
                if used[g] + size[i] <= GPU_MEM_SIZE:
                    cands.append(((loads[g] + req[i]) / (GPU_MEM_SIZE - used[g] - size[i]), g))
            if not cands:
                g = min(range(gpu_num), key=lambda x: used[x])
            else:
                cands.sort()
                k = max(1, min(len(cands), int(alpha * len(cands)) + 1))
                _, g = cands[rng.randrange(k)]
            placement[g].append(i)
            loads[g] += req[i]
            used[g] += size[i]
        return placement, loads, used

    def local_search(placement, loads, used, deadline):
        placement = {g: list(idxs) for g, idxs in placement.items()}
        best = kvpr_of(loads, used)
        while time.time() < deadline:
            improved = False
            # best-improvement single move
            best_move = None
            for src in range(gpu_num):
                for mi in range(len(placement[src])):
                    i = placement[src][mi]
                    for dst in range(gpu_num):
                        if dst == src or used[dst] + size[i] > GPU_MEM_SIZE:
                            continue
                        loads[src] -= req[i]; loads[dst] += req[i]
                        used[src] -= size[i]; used[dst] += size[i]
                        score = kvpr_of(loads, used)
                        loads[src] += req[i]; loads[dst] -= req[i]
                        used[src] += size[i]; used[dst] -= size[i]
                        if score < best - 1e-12 and (best_move is None or score < best_move[0]):
                            best_move = (score, src, mi, dst)
            if best_move:
                score, src, mi, dst = best_move
                i = placement[src].pop(mi)
                placement[dst].append(i)
                loads[src] -= req[i]; loads[dst] += req[i]
                used[src] -= size[i]; used[dst] += size[i]
                best = score
                improved = True
                continue
            # best-improvement swap
            best_swap = None
            for src in range(gpu_num):
                for mi in range(len(placement[src])):
                    i1 = placement[src][mi]
                    for dst in range(src + 1, gpu_num):
                        for mj in range(len(placement[dst])):
                            i2 = placement[dst][mj]
                            if used[dst] - size[i2] + size[i1] > GPU_MEM_SIZE:
                                continue
                            if used[src] - size[i1] + size[i2] > GPU_MEM_SIZE:
                                continue
                            loads[src] += req[i2] - req[i1]
                            loads[dst] += req[i1] - req[i2]
                            used[src] += size[i2] - size[i1]
                            used[dst] += size[i1] - size[i2]
                            score = kvpr_of(loads, used)
                            loads[src] -= req[i2] - req[i1]
                            loads[dst] -= req[i1] - req[i2]
                            used[src] -= size[i2] - size[i1]
                            used[dst] -= size[i1] - size[i2]
                            if score < best - 1e-12 and (best_swap is None or score < best_swap[0]):
                                best_swap = (score, src, mi, dst, mj)
            if best_swap:
                score, src, mi, dst, mj = best_swap
                placement[src][mi], placement[dst][mj] = placement[dst][mj], placement[src][mi]
                i1, i2 = placement[src][mi], placement[dst][mj]
                loads[src] += req[i2] - req[i1]
                loads[dst] += req[i1] - req[i2]
                used[src] += size[i2] - size[i1]
                used[dst] += size[i1] - size[i2]
                best = score
                improved = True
                continue
            if not improved:
                break
        return placement, loads, used, best

    rng = random.Random(42)
    orders = [
        sorted(range(n), key=lambda i: req[i] / max(size[i], 1e-9), reverse=True),
        sorted(range(n), key=lambda i: size[i], reverse=True),
        sorted(range(n), key=lambda i: req[i], reverse=True),
        list(range(n)),
    ]

    best_placement = None
    best_score = float("inf")
    attempt = 0
    while attempt < 40 and time.time() - start < 2.0:
        alpha = 0.0 if attempt < len(orders) else rng.random()
        order = orders[attempt] if attempt < len(orders) else rng.sample(range(n), n)
        placement, loads, used = construct(order, alpha)
        placement, loads, used, score = local_search(
            placement, loads, used, start + 0.1 + attempt * 0.05)
        if score < best_score:
            best_score = score
            best_placement = placement
        attempt += 1

    if best_placement is None or best_score == float("inf"):
        # FFD fallback for feasibility
        placement = {g: [] for g in range(gpu_num)}
        free = [GPU_MEM_SIZE] * gpu_num
        for i in sorted(range(n), key=lambda i: size[i], reverse=True):
            g = next((g for g in range(gpu_num) if size[i] <= free[g]),
                     max(range(gpu_num), key=lambda x: free[x]))
            placement[g].append(i)
            free[g] -= size[i]
        best_placement = placement

    return {g: [models[i] for i in idxs] for g, idxs in best_placement.items()}


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
