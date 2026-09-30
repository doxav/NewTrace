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

    def local_search(placement, rng, iters=300):
        # Move and swap models between GPUs to reduce max KVPR.
        placement = {g: list(ms) for g, ms in placement.items()}
        best_score = max_kvpr(placement)
        for _ in range(iters):
            improved = False
            # Try single-model moves first
            for src in range(gpu_num):
                if not placement[src]:
                    continue
                for mi in range(len(placement[src])):
                    model = placement[src][mi]
                    for dst in range(gpu_num):
                        if dst == src:
                            continue
                        if not fits(placement[dst], model.model_size):
                            continue
                        placement[src].pop(mi)
                        placement[dst].append(model)
                        score = max_kvpr(placement)
                        if score < best_score - 1e-12:
                            best_score = score
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
            # Try pairwise swaps (one model from src for one from dst)
            for src in range(gpu_num):
                if not placement[src]:
                    continue
                for mi in range(len(placement[src])):
                    m1 = placement[src][mi]
                    for dst in range(src + 1, gpu_num):
                        for dj in range(len(placement[dst])):
                            m2 = placement[dst][dj]
                            if m1 is m2:
                                continue
                            if not fits(placement[dst], m1.model_size - m2.model_size):
                                continue
                            if not fits(placement[src], m2.model_size - m1.model_size):
                                continue
                            placement[src][mi] = m2
                            placement[dst][dj] = m1
                            score = max_kvpr(placement)
                            if score < best_score - 1e-12:
                                best_score = score
                                improved = True
                                break
                            placement[src][mi] = m1
                            placement[dst][dj] = m2
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return placement

    rng = random.Random(42)
    best_placement = None
    best_score = float("inf")
    attempts = 0
    max_attempts = 150
    while attempts < max_attempts:
        if attempts < len(candidate_orders):
            order = candidate_orders[attempts]
        else:
            order = list(models)
            rng.shuffle(order)
        attempts += 1
        placement = greedy(order)
        if placement is None:
            placement = place_anywhere(order)
        placement = local_search(placement, rng)
        score = max_kvpr(placement)
        if score < best_score:
            best_score = score
            best_placement = placement

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
