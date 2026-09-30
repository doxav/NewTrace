GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START

import random

def compute_model_placement(gpu_num, models):
    """Multi-start randomized greedy placement: sample model orderings biased
    by weight (req_rate/slo), place each model on the GPU minimizing resulting
    KVPR, and keep the best placement over many restarts."""

    def run(seed):
        rng = random.Random(seed)
        order = sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True)
        for i in range(len(order)):
            j = rng.randrange(i, len(order))
            order[i], order[j] = order[j], order[i]
        place = {g: [] for g in range(gpu_num)}
        w = [0.0] * gpu_num
        mem = [float(GPU_MEM_SIZE)] * gpu_num
        for m in order:
            r, s = m.req_rate / m.slo, m.model_size
            cands = [(w[g] + r) / (mem[g] - s) if s <= mem[g] else float("inf")
                     for g in range(gpu_num)]
            g = min(range(gpu_num), key=lambda g: cands[g])
            if cands[g] == float("inf"):
                g = max(range(gpu_num), key=lambda g: mem[g])
            place[g].append(m)
            w[g] += r
            mem[g] -= s
        return place, w, mem

    def maxk(w, mem):
        return max(wi / mi if mi > 0 else float("inf") for wi, mi in zip(w, mem))

    best = None
    best_val = float("inf")
    n = max(30, 10 * len(models))
    for seed in range(n):
        place, w, mem = run(seed)
        improved = True
        val = maxk(w, mem)
        while improved:
            improved = False
            src = max(range(gpu_num), key=lambda g: w[g] / mem[g] if mem[g] > 0 else float("inf"))
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
