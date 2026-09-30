GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize the maximum KVPR across GPUs.

    Approach: greedy constructions under several orderings (plus randomized
    restarts), each followed by a first-improvement local search over moves
    and swaps. Only feasible placements are kept; a first-fit-decreasing
    seed guarantees a valid starting point when one exists.
    """
    import random

    rng = random.Random(42)
    if not models:
        return {g: [] for g in range(gpu_num)}

    def max_kvpr(placement):
        best = 0.0
        for gpu_models in placement.values():
            denom = GPU_MEM_SIZE - sum(m.model_size for m in gpu_models)
            if denom <= 0:
                return float("inf")
            best = max(best, sum(m.req_rate / m.slo for m in gpu_models) / denom)
        return best

    def feasible(placement):
        return all(sum(m.model_size for m in ms) <= GPU_MEM_SIZE
                   for ms in placement.values())

    def greedy(order):
        placement = {g: [] for g in range(gpu_num)}
        free = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for model in order:
            req = model.req_rate / model.slo
            best_g, best_kv = None, float("inf")
            for g in range(gpu_num):
                if model.model_size <= free[g]:
                    kv = (load[g] + req) / (free[g] - model.model_size)
                    if kv < best_kv:
                        best_kv, best_g = kv, g
            if best_g is None:
                return None
            placement[best_g].append(model)
            load[best_g] += req
            free[best_g] -= model.model_size
        return placement

    def ffd():
        # Best-fit-decreasing: place largest models first, choosing among
        # feasible GPUs the one minimizing the resulting KVPR.
        placement = {g: [] for g in range(gpu_num)}
        free = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for model in sorted(models, key=lambda m: m.model_size, reverse=True):
            req = model.req_rate / model.slo
            best_g, best_kv = None, float("inf")
            for g in range(gpu_num):
                if model.model_size <= free[g]:
                    kv = (load[g] + req) / (free[g] - model.model_size)
                    if kv < best_kv:
                        best_kv, best_g = kv, g
            if best_g is None:
                return None
            placement[best_g].append(model)
            load[best_g] += req
            free[best_g] -= model.model_size
        return placement

    def local_search(placement):
        # Best-improvement local search with incremental load/free tracking.
        placement = {g: list(ms) for g, ms in placement.items()}
        load = [sum(m.req_rate / m.slo for m in placement[g]) for g in range(gpu_num)]
        used = [sum(m.model_size for m in placement[g]) for g in range(gpu_num)]

        def kvpr():
            best = 0.0
            for g in range(gpu_num):
                d = GPU_MEM_SIZE - used[g]
                if d <= 0:
                    return float("inf")
                v = load[g] / d
                if v > best:
                    best = v
            return best

        best = kvpr()
        for _ in range(100):
            improved = False
            # best-improvement move
            best_mv = None
            for src in range(gpu_num):
                for mi in range(len(placement[src])):
                    m = placement[src][mi]
                    req = m.req_rate / m.slo
                    for dst in range(gpu_num):
                        if dst == src:
                            continue
                        if used[dst] + m.model_size > GPU_MEM_SIZE:
                            continue
                        load[src] -= req; load[dst] += req
                        used[src] -= m.model_size; used[dst] += m.model_size
                        s = kvpr()
                        if s < best - 1e-12 and (best_mv is None or s < best_mv[0]):
                            best_mv = (s, src, mi, dst)
                        load[src] += req; load[dst] -= req
                        used[src] += m.model_size; used[dst] -= m.model_size
            if best_mv:
                s, src, mi, dst = best_mv
                m = placement[src].pop(mi)
                placement[dst].append(m)
                load[src] -= m.req_rate / m.slo; load[dst] += m.req_rate / m.slo
                used[src] -= m.model_size; used[dst] += m.model_size
                best = s
                improved = True
                continue
            # best-improvement swap
            best_sw = None
            for src in range(gpu_num):
                for mi in range(len(placement[src])):
                    m1 = placement[src][mi]
                    r1, z1 = m1.req_rate / m1.slo, m1.model_size
                    for dst in range(src + 1, gpu_num):
                        for mj in range(len(placement[dst])):
                            m2 = placement[dst][mj]
                            r2, z2 = m2.req_rate / m2.slo, m2.model_size
                            if used[dst] - z2 + z1 > GPU_MEM_SIZE:
                                continue
                            if used[src] - z1 + z2 > GPU_MEM_SIZE:
                                continue
                            load[src] += r2 - r1; load[dst] += r1 - r2
                            used[src] += z2 - z1; used[dst] += z1 - z2
                            s = kvpr()
                            if s < best - 1e-12 and (best_sw is None or s < best_sw[0]):
                                best_sw = (s, src, mi, dst, mj)
                            load[src] -= r2 - r1; load[dst] -= r1 - r2
                            used[src] -= z2 - z1; used[dst] -= z1 - z2
            if best_sw:
                s, src, mi, dst, mj = best_sw
                placement[src][mi], placement[dst][mj] = placement[dst][mj], placement[src][mi]
                m1, m2 = placement[src][mi], placement[dst][mj]
                load[src] += m2.req_rate / m2.slo - m1.req_rate / m1.slo
                load[dst] += m1.req_rate / m1.slo - m2.req_rate / m2.slo
                used[src] += m2.model_size - m1.model_size
                used[dst] += m1.model_size - m2.model_size
                best = s
                improved = True
            if not improved:
                break
        return placement

    candidate_orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / max(m.model_size, 1e-9), reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo),
        list(models),
    ]

    best_placement = ffd()
    if best_placement is not None:
        best_placement = local_search(best_placement)
        best_score = max_kvpr(best_placement)
    else:
        best_score = float("inf")

    attempts = 0
    while attempts < 120:
        if attempts < len(candidate_orders):
            order = candidate_orders[attempts]
        else:
            order = list(models)
            rng.shuffle(order)
        attempts += 1
        placement = greedy(order)
        if placement is None:
            continue
        placement = local_search(placement)
        score = max_kvpr(placement)
        if score < best_score:
            best_score = score
            best_placement = placement

    if best_placement is None:
        # Last resort: place largest models on GPU with most free memory
        placement = {g: [] for g in range(gpu_num)}
        free = [GPU_MEM_SIZE] * gpu_num
        for model in sorted(models, key=lambda m: m.model_size, reverse=True):
            g = max(range(gpu_num), key=lambda x: free[x])
            placement[g].append(model)
            free[g] -= model.model_size
        return placement
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
