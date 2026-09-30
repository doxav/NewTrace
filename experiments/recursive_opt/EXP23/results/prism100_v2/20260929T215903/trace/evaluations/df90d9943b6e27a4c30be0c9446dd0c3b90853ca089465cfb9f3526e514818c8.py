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

    def max_kvpr(placement):
        best = 0.0
        for gpu_models in placement.values():
            denom = GPU_MEM_SIZE - sum(m.model_size for m in gpu_models)
            if denom <= 0:
                return float("inf")
            kvpr = sum(m.req_rate / m.slo for m in gpu_models) / denom
            best = max(best, kvpr)
        return best

    def greedy(order):
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]
        load = [0.0 for _ in range(gpu_num)]
        for model in order:
            best_idx = None
            best_kvpr = float("inf")
            for gpu_id in range(gpu_num):
                if model.model_size <= shared_kv[gpu_id] and shared_kv[gpu_id] > 0:
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

    def ffd_placement():
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        free = [GPU_MEM_SIZE for _ in range(gpu_num)]
        for model in sorted(models, key=lambda m: m.model_size, reverse=True):
            placed = False
            for g in range(gpu_num):
                if model.model_size <= free[g]:
                    placement[g].append(model)
                    free[g] -= model.model_size
                    placed = True
                    break
            if not placed:
                return None
        return placement

    def local_search(placement):
        # Move and swap models between GPUs to reduce max KVPR,
        # using incremental load/used tracking for speed.
        placement = {g: list(ms) for g, ms in placement.items()}
        loads = [0.0] * gpu_num
        used = [0.0] * gpu_num
        for g in range(gpu_num):
            for m in placement[g]:
                loads[g] += m.req_rate / m.slo
                used[g] += m.model_size

        def score():
            best = 0.0
            for g in range(gpu_num):
                denom = GPU_MEM_SIZE - used[g]
                if denom <= 1e-12:
                    return float("inf")
                s = loads[g] / denom
                if s > best:
                    best = s
            return best

        best_score = score()
        for _ in range(200):
            improved = False
            # GPUs sorted by KVPR descending: focus on the most crowded first.
            order = sorted(range(gpu_num),
                           key=lambda g: loads[g] / max(GPU_MEM_SIZE - used[g], 1e-12),
                           reverse=True)
            # Single-model moves
            for src in order:
                for mi in range(len(placement[src])):
                    model = placement[src][mi]
                    req = model.req_rate / model.slo
                    sz = model.model_size
                    for dst in order:
                        if dst == src:
                            continue
                        if sz > GPU_MEM_SIZE - used[dst]:
                            continue
                        loads[src] -= req; loads[dst] += req
                        used[src] -= sz; used[dst] += sz
                        s = score()
                        if s < best_score - 1e-12:
                            placement[src].pop(mi)
                            placement[dst].append(model)
                            best_score = s
                            improved = True
                            break
                        loads[src] += req; loads[dst] -= req
                        used[src] += sz; used[dst] -= sz
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Pairwise swaps
            for src in order:
                for mi in range(len(placement[src])):
                    m1 = placement[src][mi]
                    r1 = m1.req_rate / m1.slo
                    s1 = m1.model_size
                    for dst in order:
                        if dst <= src:
                            continue
                        for dj in range(len(placement[dst])):
                            m2 = placement[dst][dj]
                            r2 = m2.req_rate / m2.slo
                            s2 = m2.model_size
                            if s2 - s1 > GPU_MEM_SIZE - used[src] or s1 - s2 > GPU_MEM_SIZE - used[dst]:
                                continue
                            loads[src] += r2 - r1; loads[dst] += r1 - r2
                            used[src] += s2 - s1; used[dst] += s1 - s2
                            s = score()
                            if s < best_score - 1e-12:
                                placement[src][mi] = m2
                                placement[dst][dj] = m1
                                best_score = s
                                improved = True
                                break
                            loads[src] -= r2 - r1; loads[dst] -= r1 - r2
                            used[src] -= s2 - s1; used[dst] -= s1 - s2
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
                if model.model_size <= shared_kv[gpu_id] and shared_kv[gpu_id] > 0:
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

    rng = random.Random(42)
    best_placement = ffd_placement()
    best_score = max_kvpr(best_placement) if best_placement is not None else float("inf")

    import time
    start = time.time()
    attempts = 0
    max_attempts = 200
    while attempts < max_attempts and time.time() - start < 2.0:
        if attempts < len(candidate_orders):
            order = candidate_orders[attempts]
        else:
            order = list(models)
            rng.shuffle(order)
        attempts += 1
        placement = greedy(order)
        if placement is None:
            placement = place_anywhere(order)
        placement = local_search(placement)
        score = max_kvpr(placement)
        if score < best_score and all(
            sum(m.model_size for m in ms) <= GPU_MEM_SIZE for ms in placement.values()
        ):
            best_score = score
            best_placement = placement

    if best_placement is None:
        # Last resort: place largest models on GPU with most free memory
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        free = [GPU_MEM_SIZE for _ in range(gpu_num)]
        for model in sorted(models, key=lambda m: m.model_size, reverse=True):
            idx = max(range(gpu_num), key=lambda g: free[g])
            placement[idx].append(model)
            free[idx] -= model.model_size
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
