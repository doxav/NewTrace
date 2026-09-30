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
        placement = {g: [] for g in range(gpu_num)}
        free = [GPU_MEM_SIZE] * gpu_num
        for model in sorted(models, key=lambda m: m.model_size, reverse=True):
            for g in range(gpu_num):
                if model.model_size <= free[g]:
                    placement[g].append(model)
                    free[g] -= model.model_size
                    break
            else:
                return None
        return placement

    def local_search(placement):
        placement = {g: list(ms) for g, ms in placement.items()}
        best = max_kvpr(placement)
        for _ in range(100):
            improved = False
            # moves
            for src in range(gpu_num):
                if not placement[src]:
                    continue
                for mi in range(len(placement[src])):
                    model = placement[src][mi]
                    src_used = sum(m.model_size for m in placement[src])
                    for dst in range(gpu_num):
                        if dst == src:
                            continue
                        if src_used - model.model_size + sum(m.model_size for m in placement[dst]) + model.model_size > GPU_MEM_SIZE:
                            continue
                        placement[src].pop(mi)
                        placement[dst].append(model)
                        score = max_kvpr(placement)
                        if score < best - 1e-12:
                            best = score
                            improved = True
                            break
                        placement[dst].pop()
                        placement[src].insert(mi, model)
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # swaps
            for src in range(gpu_num):
                if not placement[src]:
                    continue
                for mi in range(len(placement[src])):
                    m1 = placement[src][mi]
                    for dst in range(src + 1, gpu_num):
                        for mj in range(len(placement[dst])):
                            m2 = placement[dst][mj]
                            if m1 is m2:
                                continue
                            s_used = sum(m.model_size for m in placement[src])
                            d_used = sum(m.model_size for m in placement[dst])
                            if s_used - m1.model_size + m2.model_size > GPU_MEM_SIZE:
                                continue
                            if d_used - m2.model_size + m1.model_size > GPU_MEM_SIZE:
                                continue
                            placement[src][mi], placement[dst][mj] = m2, m1
                            score = max_kvpr(placement)
                            if score < best - 1e-12:
                                best = score
                                improved = True
                                break
                            placement[src][mi], placement[dst][mj] = m1, m2
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
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
    best_score = max_kvpr(best_placement) if best_placement else float("inf")

    attempts = 0
    while attempts < 40:
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
