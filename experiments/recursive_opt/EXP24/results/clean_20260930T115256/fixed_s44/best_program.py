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

    n = len(models)
    if n == 0:
        return {g: [] for g in range(gpu_num)}

    import random

    weights = [m.req_rate / m.slo for m in models]
    sizes = [m.model_size for m in models]
    EPS = 1e-9
    # never fill a GPU completely: a zero denominator makes KVPR invalid
    CAP = GPU_MEM_SIZE - 1e-6

    def kvpr_of(used, load, g):
        denom = GPU_MEM_SIZE - used[g]
        if denom <= EPS:
            return float("inf")
        return load[g] / denom

    def greedy(order):
        assign = [-1] * n
        used = [0.0] * gpu_num
        load = [0.0] * gpu_num
        for i in order:
            best_g, best_score = None, float("inf")
            for g in range(gpu_num):
                if used[g] + sizes[i] > CAP:
                    continue
                score = (load[g] + weights[i]) / (GPU_MEM_SIZE - used[g] - sizes[i])
                if score < best_score:
                    best_score, best_g = score, g
            if best_g is None:
                return None
            assign[i] = best_g
            used[best_g] += sizes[i]
            load[best_g] += weights[i]
        return assign, used, load

    def local_search(assign, used, load):
        cur = max(kvpr_of(used, load, g) for g in range(gpu_num))
        improved = True
        while improved:
            improved = False
            for i in range(n):
                g_from = assign[i]
                # try moving model i
                for g in range(gpu_num):
                    if g == g_from or used[g] + sizes[i] > CAP:
                        continue
                    nlf, nlt = load[g_from] - weights[i], load[g] + weights[i]
                    nuf, nut = used[g_from] - sizes[i], used[g] + sizes[i]
                    cand = max(
                        max(kvpr_of(used, load, k) for k in range(gpu_num)
                            if k != g_from and k != g),
                        nlf / (GPU_MEM_SIZE - nuf),
                        nlt / (GPU_MEM_SIZE - nut),
                    )
                    if cand < cur - 1e-12:
                        load[g_from], load[g] = nlf, nlt
                        used[g_from], used[g] = nuf, nut
                        assign[i] = g
                        g_from = g
                        cur = cand
                        improved = True
                g_from = assign[i]
                # try swapping model i with model j
                for j in range(n):
                    g_to = assign[j]
                    if g_to == g_from:
                        continue
                    nuf = used[g_from] - sizes[i] + sizes[j]
                    nut = used[g_to] - sizes[j] + sizes[i]
                    if nuf > CAP or nut > CAP:
                        continue
                    nlf = load[g_from] - weights[i] + weights[j]
                    nlt = load[g_to] - weights[j] + weights[i]
                    cand = max(
                        max(kvpr_of(used, load, k) for k in range(gpu_num)
                            if k != g_from and k != g_to),
                        nlf / (GPU_MEM_SIZE - nuf),
                        nlt / (GPU_MEM_SIZE - nut),
                    )
                    if cand < cur - 1e-12:
                        load[g_from], load[g_to] = nlf, nlt
                        used[g_from], used[g_to] = nuf, nut
                        assign[i], assign[j] = g_to, g_from
                        g_from = assign[i]
                        cur = cand
                        improved = True
        return cur

    # Multiple greedy orders (deterministic + randomized), keep the best
    rng = random.Random(12345)
    orders = [
        sorted(range(n), key=lambda i: -weights[i]),
        sorted(range(n), key=lambda i: -(weights[i] / max(sizes[i], 1e-9))),
        sorted(range(n), key=lambda i: -sizes[i]),
        sorted(range(n), key=lambda i: weights[i] / max(sizes[i], 1e-9)),
        list(range(n)),
    ]
    for _ in range(15):
        o = list(range(n))
        rng.shuffle(o)
        orders.append(o)

    best_assign, best_val = None, float("inf")
    for order in orders:
        res = greedy(order)
        if res is None:
            continue
        assign, used, load = res
        val = local_search(assign, used, load)
        if val < best_val:
            best_val = val
            best_assign = assign

    if best_assign is None:
        # Feasible fallback: best-fit by size, never filling a GPU completely
        best_assign = [-1] * n
        used = [0.0] * gpu_num
        for i in sorted(range(n), key=lambda k: -sizes[k]):
            cands = [g for g in range(gpu_num)
                     if used[g] + sizes[i] < GPU_MEM_SIZE - EPS]
            if cands:
                g = max(cands, key=lambda g: GPU_MEM_SIZE - used[g] - sizes[i])
            else:
                g = min(range(gpu_num), key=lambda g: used[g])
            best_assign[i] = g
            used[g] += sizes[i]

    placement = {g: [] for g in range(gpu_num)}
    for i, g in enumerate(best_assign):
        placement[g].append(models[i])
    return placement


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
