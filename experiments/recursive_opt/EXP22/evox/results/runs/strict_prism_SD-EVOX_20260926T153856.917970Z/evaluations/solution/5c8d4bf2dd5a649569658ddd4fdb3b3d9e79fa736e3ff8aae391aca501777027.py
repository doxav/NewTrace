GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START

import random

def compute_model_placement(gpu_num, models):
    """Two-phase approach: (1) guaranteed-feasible best-fit-decreasing packing
    (largest models first, only into GPUs with enough memory), diversified by
    randomized restarts; (2) local search minimizing max KVPR via single-model
    moves and pairwise swaps, always respecting memory limits."""

    def maxk(w, mem):
        return max(wi / mi if mi > 0 else float("inf") for wi, mi in zip(w, mem))

    def pack(order):
        """Best-fit-decreasing packing that strictly respects memory."""
        place = {g: [] for g in range(gpu_num)}
        w = [0.0] * gpu_num
        mem = [float(GPU_MEM_SIZE)] * gpu_num
        for m in order:
            r, s = m.req_rate / m.slo, m.model_size
            fits = [g for g in range(gpu_num) if s <= mem[g]]
            if fits:
                g = min(fits, key=lambda g: (w[g] + r) / (mem[g] - s))
            else:
                g = max(range(gpu_num), key=lambda g: mem[g])
            place[g].append(m)
            w[g] += r
            mem[g] -= s
        return place, w, mem

    def improve(place, w, mem, val):
        """Local search with moves and swaps from the max-KVPR GPU."""
        improved = True
        while improved:
            improved = False
            kvs = [w[g] / mem[g] if mem[g] > 0 else float("inf") for g in range(gpu_num)]
            src = max(range(gpu_num), key=lambda g: kvs[g])
            # Try single-model moves
            for m in list(place[src]):
                r, s = m.req_rate / m.slo, m.model_size
                for dst in range(gpu_num):
                    if dst == src or s > mem[dst]:
                        continue
                    w[src] -= r; mem[src] += s
                    w[dst] += r; mem[dst] -= s
                    nv = maxk(w, mem)
                    if nv < val - 1e-12:
                        val = nv
                        place[src].remove(m); place[dst].append(m)
                        improved = True
                        break
                    w[src] += r; mem[src] -= s
                    w[dst] -= r; mem[dst] += s
                if improved:
                    break
            if improved:
                continue
            # Try pairwise swaps between src and other GPUs
            for m1 in list(place[src]):
                r1, s1 = m1.req_rate / m1.slo, m1.model_size
                for dst in range(gpu_num):
                    if dst == src:
                        continue
                    for m2 in list(place[dst]):
                        r2, s2 = m2.req_rate / m2.slo, m2.model_size
                        if s2 - s1 > mem[src] or s1 - s2 > mem[dst]:
                            continue
                        w[src] += r2 - r1; mem[src] += s1 - s2
                        w[dst] += r1 - r2; mem[dst] += s2 - s1
                        nv = maxk(w, mem)
                        if nv < val - 1e-12:
                            val = nv
                            place[src].remove(m1); place[dst].remove(m2)
                            place[src].append(m2); place[dst].append(m1)
                            improved = True
                            break
                        w[src] -= r2 - r1; mem[src] -= s1 - s2
                        w[dst] -= r1 - r2; mem[dst] -= s2 - s1
                    if improved:
                        break
                if improved:
                    break
        return val

    by_size = sorted(models, key=lambda m: m.model_size, reverse=True)
    by_weight = sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True)
    rng = random.Random(12345)

    best = None
    best_val = float("inf")
    orders = [by_size, by_weight, list(models)]
    for _ in range(20):
        o = list(models)
        rng.shuffle(o)
        orders.append(o)
    for order in orders:
        place, w, mem = pack(order)
        val = improve(place, w, mem, maxk(w, mem))
        if val < best_val:
            best_val, best = val, place
    return best

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
