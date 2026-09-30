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
        # Always places feasibly when possible: pick GPU minimizing resulting
        # KVPR; if none fits, place on GPU with most free memory (best effort).
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]
        load = [0.0 for _ in range(gpu_num)]
        # First pass: place large models first to keep packing feasible.
        for model in sorted(order, key=lambda m: m.model_size, reverse=True):
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

    def local_search(placement, rng, iters=300):
        # Try moving or swapping models between GPUs to reduce max KVPR.
        placement = {g: list(ms) for g, ms in placement.items()}
        best_score = max_kvpr(placement)
        for _ in range(iters):
            gpu_ids = list(range(gpu_num))
            rng.shuffle(gpu_ids)
            improved = False
            for src in gpu_ids:
                if not placement[src]:
                    continue
                for mi in range(len(placement[src])):
                    model = placement[src][mi]
                    for dst in gpu_ids:
                        if dst == src:
                            continue
                        # Try move
                        if model.model_size <= GPU_MEM_SIZE - sum(m.model_size for m in placement[dst]):
                            placement[src].pop(mi)
                            placement[dst].append(model)
                            score = max_kvpr(placement)
                            if score < best_score - 1e-12:
                                best_score = score
                                improved = True
                                break
                            placement[dst].pop()
                            placement[src].insert(mi, model)
                        # Try swap
                        for dj in range(len(placement[dst])):
                            other = placement[dst][dj]
                            freed_src = GPU_MEM_SIZE - sum(m.model_size for m in placement[src]) + model.model_size
                            freed_dst = GPU_MEM_SIZE - sum(m.model_size for m in placement[dst]) + other.model_size
                            if other.model_size <= freed_src and model.model_size <= freed_dst:
                                placement[src][mi] = other
                                placement[dst][dj] = model
                                score = max_kvpr(placement)
                                if score < best_score - 1e-12:
                                    best_score = score
                                    improved = True
                                    break
                                placement[src][mi] = model
                                placement[dst][dj] = other
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
    max_attempts = 100
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
