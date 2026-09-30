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

    def greedy(order):
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]
        load = [0.0 for _ in range(gpu_num)]

        for model in order:
            best_idx = None
            best_kvpr = float("inf")
            for gpu_id in range(gpu_num):
                if model.model_size <= shared_kv[gpu_id]:
                    new_kvpr = (load[gpu_id] + model.req_rate / model.slo) / (
                        shared_kv[gpu_id] - model.model_size
                    )
                    if new_kvpr < best_kvpr:
                        best_kvpr = new_kvpr
                        best_idx = gpu_id
            if best_idx is None:
                return None
            placement[best_idx].append(model)
            load[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        return placement

    def max_kvpr(placement):
        best = 0.0
        for gpu_models in placement.values():
            denom = GPU_MEM_SIZE - sum(m.model_size for m in gpu_models)
            if denom <= 0:
                return float("inf")
            kvpr = sum(m.req_rate / m.slo for m in gpu_models) / denom
            best = max(best, kvpr)
        return best

    # Try several deterministic orderings plus randomized restarts
    candidate_orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True),
        list(models),
    ]

    def place_anywhere(order):
        # Always places: pick GPU minimizing resulting KVPR; if none fits,
        # place on GPU with most free memory.
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]
        load = [0.0 for _ in range(gpu_num)]
        for model in order:
            req = model.req_rate / model.slo
            best_idx = None
            best_kvpr = float("inf")
            for gpu_id in range(gpu_num):
                if model.model_size <= shared_kv[gpu_id]:
                    new_kvpr = (load[gpu_id] + req) / (shared_kv[gpu_id] - model.model_size)
                    if new_kvpr < best_kvpr:
                        best_kvpr = new_kvpr
                        best_idx = gpu_id
            if best_idx is None:
                best_idx = max(range(gpu_num), key=lambda g: shared_kv[g])
            placement[best_idx].append(model)
            load[best_idx] += req
            shared_kv[best_idx] -= model.model_size
        return placement

    def fits(gpu_models, extra_size):
        return sum(m.model_size for m in gpu_models) + extra_size <= GPU_MEM_SIZE

    def local_search(placement, rng):
        # Incremental local search focused on the max-KVPR GPU.
        placement = {g: list(ms) for g, ms in placement.items()}
        used = [sum(m.model_size for m in placement[g]) for g in range(gpu_num)]
        load = [sum(m.req_rate / m.slo for m in placement[g]) for g in range(gpu_num)]

        def kvpr(g):
            denom = GPU_MEM_SIZE - used[g]
            return load[g] / denom if denom > 0 else float("inf")

        def global_max():
            return max(kvpr(g) for g in range(gpu_num))

        for _ in range(200):
            cur = global_max()
            worst = max(range(gpu_num), key=kvpr)
            improved = False
            # Try moving one model out of the worst GPU
            for mi in range(len(placement[worst]) - 1, -1, -1):
                model = placement[worst][mi]
                req = model.req_rate / model.slo
                for dst in range(gpu_num):
                    if dst == worst or used[dst] + model.model_size > GPU_MEM_SIZE:
                        continue
                    placement[worst].pop(mi)
                    placement[dst].append(model)
                    used[worst] -= model.model_size
                    used[dst] += model.model_size
                    load[worst] -= req
                    load[dst] += req
                    if global_max() < cur - 1e-12:
                        improved = True
                        break
                    placement[dst].pop()
                    placement[worst].insert(mi, model)
                    used[worst] += model.model_size
                    used[dst] -= model.model_size
                    load[worst] += req
                    load[dst] -= req
                if improved:
                    break
            if improved:
                continue
            # Try swapping a model between the worst GPU and others
            done = False
            for mi in range(len(placement[worst])):
                m1 = placement[worst][mi]
                r1 = m1.req_rate / m1.slo
                for dst in range(gpu_num):
                    if dst == worst:
                        continue
                    for dj in range(len(placement[dst])):
                        m2 = placement[dst][dj]
                        if m1 is m2:
                            continue
                        r2 = m2.req_rate / m2.slo
                        if used[dst] - m2.model_size + m1.model_size > GPU_MEM_SIZE:
                            continue
                        if used[worst] - m1.model_size + m2.model_size > GPU_MEM_SIZE:
                            continue
                        placement[worst][mi] = m2
                        placement[dst][dj] = m1
                        load[worst] += r2 - r1
                        load[dst] += r1 - r2
                        if global_max() < cur - 1e-12:
                            improved = True
                            done = True
                            break
                        placement[worst][mi] = m1
                        placement[dst][dj] = m2
                        load[worst] += r1 - r2
                        load[dst] += r2 - r1
                    if done:
                        break
                if done:
                    break
            if not improved:
                break
        return placement

    def feasible(placement):
        for gpu_models in placement.values():
            if sum(m.model_size for m in gpu_models) > GPU_MEM_SIZE:
                return False
        return True

    def backtrack_placement():
        # Exact feasibility via backtracking (largest models first).
        order = sorted(models, key=lambda m: m.model_size, reverse=True)
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        free = [GPU_MEM_SIZE for _ in range(gpu_num)]

        nodes = [0]

        def rec(i):
            nodes[0] += 1
            if nodes[0] > 20000:
                raise RuntimeError
            if i == len(order):
                return True
            model = order[i]
            tried = set()
            for gpu_id in range(gpu_num):
                if model.model_size <= free[gpu_id] and free[gpu_id] not in tried:
                    tried.add(free[gpu_id])
                    placement[gpu_id].append(model)
                    free[gpu_id] -= model.model_size
                    if rec(i + 1):
                        return True
                    free[gpu_id] += model.model_size
                    placement[gpu_id].pop()
            return False

        try:
            return placement if rec(0) else None
        except RuntimeError:
            return None

    rng = random.Random(42)
    best_placement = None
    best_score = float("inf")

    def consider(placement):
        nonlocal best_placement, best_score
        if placement is None or not feasible(placement):
            return
        placement = local_search(placement, rng)
        score = max_kvpr(placement)
        if score < best_score:
            best_score = score
            best_placement = placement

    # Feasibility-first seed (guaranteed valid if one exists)
    consider(backtrack_placement())

    attempts = 0
    while attempts < 40 and best_score > 0:
        if attempts < len(candidate_orders):
            order = candidate_orders[attempts]
        else:
            order = list(models)
            rng.shuffle(order)
        attempts += 1
        placement = greedy(order)
        if placement is None:
            continue
        consider(placement)

    if best_placement is None:
        best_placement = place_anywhere(sorted(models, key=lambda m: m.model_size, reverse=True))
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
