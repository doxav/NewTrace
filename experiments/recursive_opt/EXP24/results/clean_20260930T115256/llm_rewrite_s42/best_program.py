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

    def greedy(order):
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]
        weighted = [0.0 for _ in range(gpu_num)]
        for model in order:
            best_idx = None
            best_score = float("inf")
            for gpu_id in range(gpu_num):
                if model.model_size <= shared_kv[gpu_id]:
                    rem = shared_kv[gpu_id] - model.model_size
                    new_kv = (weighted[gpu_id] + model.req_rate / model.slo) / rem if rem > 0 else float("inf")
                    if new_kv < best_score:
                        best_score = new_kv
                        best_idx = gpu_id
            if best_idx is None:
                return None, None, None
            placement[best_idx].append(model)
            weighted[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        return placement, shared_kv, weighted

    def local_search(placement, shared_kv, weighted):
        def kvpr(g):
            return weighted[g] / shared_kv[g] if shared_kv[g] > 0 else float("inf")

        improved = True
        rounds = 500
        while improved and rounds > 0:
            improved = False
            rounds -= 1
            current_max = max(kvpr(g) for g in range(gpu_num))
            # single-model moves
            for g1 in range(gpu_num):
                for model in list(placement[g1]):
                    w = model.req_rate / model.slo
                    for g2 in range(gpu_num):
                        if g2 == g1 or model.model_size > shared_kv[g2]:
                            continue
                        new_s2 = shared_kv[g2] - model.model_size
                        new_kv1 = (weighted[g1] - w) / (shared_kv[g1] + model.model_size)
                        new_kv2 = (weighted[g2] + w) / new_s2 if new_s2 > 0 else float("inf")
                        new_max = max(new_kv1, new_kv2)
                        if new_max < max(kvpr(g1), kvpr(g2)) - 1e-12 and new_max < current_max + 1e-12:
                            placement[g1].remove(model)
                            placement[g2].append(model)
                            weighted[g1] -= w
                            shared_kv[g1] += model.model_size
                            weighted[g2] += w
                            shared_kv[g2] = new_s2
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # pairwise swaps
            for g1 in range(gpu_num):
                for m1 in list(placement[g1]):
                    for g2 in range(gpu_num):
                        if g2 <= g1:
                            continue
                        for m2 in list(placement[g2]):
                            w1 = m1.req_rate / m1.slo
                            w2 = m2.req_rate / m2.slo
                            if m2.model_size > shared_kv[g1] + m1.model_size:
                                continue
                            if m1.model_size > shared_kv[g2] + m2.model_size:
                                continue
                            new_s1 = shared_kv[g1] + m1.model_size - m2.model_size
                            new_s2 = shared_kv[g2] + m2.model_size - m1.model_size
                            if new_s1 <= 0 or new_s2 <= 0:
                                continue
                            new_max = max(
                                (weighted[g1] - w1 + w2) / new_s1,
                                (weighted[g2] - w2 + w1) / new_s2,
                            )
                            if new_max < max(kvpr(g1), kvpr(g2)) - 1e-12:
                                placement[g1].remove(m1)
                                placement[g2].remove(m2)
                                placement[g1].append(m2)
                                placement[g2].append(m1)
                                weighted[g1] += w2 - w1
                                weighted[g2] += w1 - w2
                                shared_kv[g1] = new_s1
                                shared_kv[g2] = new_s2
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return placement, shared_kv, weighted

    def evaluate(weighted, shared_kv):
        return max(
            weighted[g] / shared_kv[g] if shared_kv[g] > 0 else float("inf")
            for g in range(gpu_num)
        )

    # Deterministic multi-start: several orderings, each followed by local search,
    # plus seeded randomized restarts for extra diversity.
    import random

    rng = random.Random(12345)
    base_orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.model_size),
        sorted(models, key=lambda m: m.req_rate / m.slo),
        sorted(models, key=lambda m: (m.model_size / (m.req_rate / m.slo)), reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) * m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.model_size / (m.req_rate / m.slo))),
    ]
    # Seeded randomized orders (weighted random shuffle biased by load/size)
    for _ in range(20):
        order = list(models)
        keys = [m.req_rate / m.slo for m in order]
        # weighted shuffle: pick proportional to key + small noise
        shuffled = []
        pool = list(zip(order, keys))
        while pool:
            total = sum(k for _, k in pool) + 1e-12
            r = rng.random() * total
            acc = 0.0
            for i, (m, k) in enumerate(pool):
                acc += k
                if acc >= r:
                    shuffled.append(m)
                    pool.pop(i)
                    break
        base_orders.append(shuffled)

    best_placement = None
    best_val = float("inf")
    for order in base_orders:
        placement, shared_kv, weighted = greedy(order)
        if placement is None:
            continue
        placement, shared_kv, weighted = local_search(placement, shared_kv, weighted)
        val = evaluate(weighted, shared_kv)
        if val < best_val:
            best_val = val
            best_placement = placement

    if best_placement is None:
        raise ValueError("Unable to place all models on the available GPUs.")

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


# --- projection: per-call fallback onto a feasible baseline ---
_PROJECTION_EVENTS = []
_candidate_compute_model_placement = compute_model_placement


def _fallback_compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """
    sorted_models = sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True)
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]
    weighted_req_rate = [0.0 for _ in range(gpu_num)]
    for model in sorted_models:
        best_idx = None
        best_ratio = float('inf')
        for gpu_id in range(gpu_num):
            if model.model_size <= shared_kv[gpu_id] and shared_kv[gpu_id] > 0:
                current_ratio = weighted_req_rate[gpu_id] / shared_kv[gpu_id]
                if current_ratio < best_ratio:
                    best_ratio = current_ratio
                    best_idx = gpu_id
        if best_idx is None:
            raise ValueError(f'Unable to place model of size {model.model_size} GB on any GPU. Remaining per-GPU memory: {shared_kv}')
        placement[best_idx].append(model)
        weighted_req_rate[best_idx] += model.req_rate / model.slo
        shared_kv[best_idx] -= model.model_size
    return placement


def _check_compute_model_placement(result, gpu_num, models):
    if not isinstance(result, dict):
        return False
    placed = []
    for gpu_id, assigned in result.items():
        if not isinstance(gpu_id, int) or not 0 <= gpu_id < gpu_num or (not isinstance(assigned, list)):
            return False
        if sum((m.model_size for m in assigned)) > GPU_MEM_SIZE:
            return False
        placed.extend(assigned)
    ids = [id(m) for m in placed]
    return len(ids) == len(models) and len(set(ids)) == len(ids) and (set(ids) == {id(m) for m in models})


def compute_model_placement(*args, **kwargs):
    try:
        result = _candidate_compute_model_placement(*args, **kwargs)
    except Exception as error:
        _PROJECTION_EVENTS.append(type(error).__name__)
        return _fallback_compute_model_placement(*args, **kwargs)
    try:
        valid = _check_compute_model_placement(result, *args, **kwargs)
    except Exception:
        valid = False
    if valid:
        return result
    _PROJECTION_EVENTS.append('invalid')
    return _fallback_compute_model_placement(*args, **kwargs)
