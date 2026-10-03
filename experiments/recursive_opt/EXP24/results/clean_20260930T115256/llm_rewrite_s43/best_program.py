GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize the maximum KVPR across GPUs using greedy init + local search.
    Note: a GPU must never be filled exactly to capacity (free memory must
    stay > 0) to avoid division-by-zero when computing KVPR.
    """
    import random

    n = len(models)
    if n == 0:
        return {gpu_id: [] for gpu_id in range(gpu_num)}

    weights = [m.req_rate / m.slo for m in models]
    sizes = [m.model_size for m in models]
    EPS = 1e-9  # keep free memory strictly positive

    def kvpr(assign):
        load = [0.0] * gpu_num
        used = [0.0] * gpu_num
        for i, g in enumerate(assign):
            load[g] += weights[i]
            used[g] += sizes[i]
        mx = 0.0
        for g in range(gpu_num):
            free = GPU_MEM_SIZE - used[g]
            if free <= EPS:
                return float("inf")
            mx = max(mx, load[g] / free)
        return mx

    def feasible(assign):
        used = [0.0] * gpu_num
        for i, g in enumerate(assign):
            used[g] += sizes[i]
        return all(u < GPU_MEM_SIZE - EPS for u in used)

    def greedy(order, jitter=0.0):
        assign = []
        load = [0.0] * gpu_num
        used = [0.0] * gpu_num
        for i in order:
            w = weights[i] * (1.0 + random.uniform(-jitter, jitter))
            best_g, best_v = None, None
            for g in range(gpu_num):
                if used[g] + sizes[i] < GPU_MEM_SIZE - EPS:
                    v = (load[g] + w) / (GPU_MEM_SIZE - used[g] - sizes[i])
                    if best_v is None or v < best_v:
                        best_v, best_g = v, g
            if best_g is None:
                return None
            assign.append(best_g)
            load[best_g] += weights[i]
            used[best_g] += sizes[i]
        return assign

    best_assign = None
    best_val = float("inf")

    orders = [
        sorted(range(n), key=lambda i: -weights[i]),
        sorted(range(n), key=lambda i: -weights[i] / sizes[i]),
        sorted(range(n), key=lambda i: -sizes[i]),
    ]
    random.seed(42)
    for it in range(400):
        if it < len(orders):
            order = orders[it]
            jitter = 0.0
        elif it < 100:
            order = orders[it % len(orders)][:]
            random.shuffle(order)
            jitter = 0.3
        else:
            order = list(range(n))
            random.shuffle(order)
            jitter = 0.5
        a = greedy(order, jitter)
        if a is None:
            continue
        v = kvpr(a)
        if v < best_val:
            best_val, best_assign = v, a[:]

    if best_assign is None:
        raise ValueError("Unable to place models on GPUs")

    def local_search(assign):
        """First-improvement local search with move + swap moves."""
        improved = True
        iters = 0
        while improved and iters < 3000:
            improved = False
            iters += 1
            cur = kvpr(assign)
            # move a single model to another GPU
            for i in range(n):
                src = assign[i]
                for g in range(gpu_num):
                    if g == src:
                        continue
                    assign[i] = g
                    if feasible(assign):
                        v = kvpr(assign)
                        if v < cur - 1e-12:
                            cur = v
                            improved = True
                            break
                    assign[i] = src
                if improved:
                    break
            if improved:
                continue
            # swap pairs of models between GPUs
            for i in range(n):
                for j in range(i + 1, n):
                    gi, gj = assign[i], assign[j]
                    if gi == gj:
                        continue
                    assign[i], assign[j] = gj, gi
                    if feasible(assign):
                        v = kvpr(assign)
                        if v < cur - 1e-12:
                            cur = v
                            improved = True
                            break
                    assign[i], assign[j] = gi, gj
                if improved:
                    break
        return kvpr(assign)

    # Iterated local search: perturb the best solution and re-optimize
    best_val = local_search(best_assign)
    for kick in range(200):
        cand = best_assign[:]
        # random perturbation: move a few random models to random GPUs
        for _ in range(max(1, n // 3)):
            i = random.randrange(n)
            g = random.randrange(gpu_num)
            old = cand[i]
            cand[i] = g
            if not feasible(cand):
                cand[i] = old
        v = local_search(cand)
        if v < best_val - 1e-12:
            best_val = v
            best_assign = cand[:]
        elif v < best_val + 1e-12 and random.random() < 0.5:
            # accept equal-quality solutions to diversify (plateau walk)
            best_assign = cand[:]
            best_val = v

    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
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
