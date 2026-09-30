GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Bisection on KVPR threshold T with randomized bin-packing feasibility check.
    Minimizes max KVPR by binary searching T: for each T, try (with random
    restarts) to place all models such that every GPU's KVPR stays <= T.
    Returns the best feasible placement found.
    """
    import random

    ms = list(models)
    weights = [m.req_rate / m.slo for m in ms]
    total_mem = sum(m.model_size for m in ms)
    lo = sum(weights) / float(gpu_num * GPU_MEM_SIZE)  # lower bound on max KVPR
    hi = sum(weights) / max(GPU_MEM_SIZE - total_mem / gpu_num, 1e-6) + 1.0

    def feasible(T, order, rng):
        place = {g: [] for g in range(gpu_num)}
        w = [0.0] * gpu_num
        mem = [float(GPU_MEM_SIZE)] * gpu_num
        for idx in order:
            m, r = ms[idx], weights[idx]
            ok = [g for g in range(gpu_num)
                  if m.model_size <= mem[g] and (w[g] + r) / (mem[g] - m.model_size) <= T]
            if not ok:
                return None
            # choose GPU minimizing resulting KVPR, with slight randomization
            ok.sort(key=lambda g: (w[g] + r) / (mem[g] - m.model_size) + rng.random() * T * 1e-3)
            g = ok[0]
            place[g].append(m)
            w[g] += r
            mem[g] -= m.model_size
        return place

    rng = random.Random(12345)
    best_place, best_T = None, hi + 1
    # Initial feasible placement via very high T
    order0 = sorted(range(len(ms)), key=lambda i: -weights[i])
    cur = feasible(hi + 10, order0, rng)
    if cur is None:
        # fallback: spread models round-robin (memory may be tight but best effort)
        best_place = {g: [] for g in range(gpu_num)}
        for i, m in enumerate(ms):
            best_place[i % gpu_num].append(m)
        return best_place
    best_place, best_T = cur, hi

    for _ in range(60):  # bisection iterations
        mid = (lo + best_T) / 2
        found = None
        for trial in range(8):  # randomized restarts for feasibility
            order = sorted(range(len(ms)), key=lambda i: (-weights[i], rng.random())) \
                if trial == 0 else rng.sample(range(len(ms)), len(ms))
            found = feasible(mid, order, rng)
            if found is not None:
                break
        if found is not None:
            best_place, best_T = found, mid
        else:
            lo = mid
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
