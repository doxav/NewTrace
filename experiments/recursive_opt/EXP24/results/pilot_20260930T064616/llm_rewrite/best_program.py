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

    req = [m.req_rate / m.slo for m in models]
    sizes = [m.model_size for m in models]
    n = len(models)

    def greedy(order):
        assign = [-1] * n
        load = [0.0] * gpu_num
        mem = [float(GPU_MEM_SIZE)] * gpu_num
        for i in order:
            best_idx = None
            best_ratio = float("inf")
            for g in range(gpu_num):
                if sizes[i] <= mem[g]:
                    ratio = load[g] / mem[g]
                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_idx = g
            if best_idx is None:
                return None
            assign[i] = best_idx
            load[best_idx] += req[i]
            mem[best_idx] -= sizes[i]
        return assign

    def max_kvpr(assign, load, mem):
        return max((load[g] / mem[g] if mem[g] > 0 else float("inf"))
                   for g in range(gpu_num))

    base_order = sorted(range(n), key=lambda i: req[i] / sizes[i], reverse=True)
    best_assign = None
    best_val = float("inf")

    random.seed(12345)

    def try_local_search(assign):
        load = [0.0] * gpu_num
        mem = [float(GPU_MEM_SIZE)] * gpu_num
        for i in range(n):
            load[assign[i]] += req[i]
            mem[assign[i]] -= sizes[i]

        for _ in range(3000):
            cur = max_kvpr(assign, load, mem)
            improved = False
            # Deterministic best-improvement moves: try moving each model
            # to the GPU that minimizes resulting max KVPR.
            for i in range(n):
                old = assign[i]
                best_g, best_new = old, cur
                for g in range(gpu_num):
                    if g == old or sizes[i] > mem[g]:
                        continue
                    load[old] -= req[i]; mem[old] += sizes[i]
                    load[g] += req[i]; mem[g] -= sizes[i]
                    new = max_kvpr(assign, load, mem)
                    load[old] += req[i]; mem[old] -= sizes[i]
                    load[g] -= req[i]; mem[g] += sizes[i]
                    if new < best_new - 1e-12:
                        best_new = new
                        best_g = g
                if best_g != old:
                    load[old] -= req[i]; mem[old] += sizes[i]
                    load[best_g] += req[i]; mem[best_g] -= sizes[i]
                    assign[i] = best_g
                    improved = True
            # Swap pairs between GPUs
            for i in range(n):
                for j in range(i + 1, n):
                    gi, gj = assign[i], assign[j]
                    if gi == gj:
                        continue
                    if sizes[j] > mem[gi] + sizes[i] or sizes[i] > mem[gj] + sizes[j]:
                        continue
                    # check feasibility of swap
                    if sizes[j] - sizes[i] > mem[gi] or sizes[i] - sizes[j] > mem[gj]:
                        continue
                    cur = max_kvpr(assign, load, mem)
                    load[gi] += req[j] - req[i]; mem[gi] += sizes[i] - sizes[j]
                    load[gj] += req[i] - req[j]; mem[gj] += sizes[j] - sizes[i]
                    assign[i], assign[j] = gj, gi
                    new = max_kvpr(assign, load, mem)
                    if new < cur - 1e-12:
                        improved = True
                    else:
                        load[gi] += req[i] - req[j]; mem[gi] += sizes[j] - sizes[i]
                        load[gj] += req[j] - req[i]; mem[gj] += sizes[i] - sizes[j]
                        assign[i], assign[j] = gi, gj
            if not improved:
                break
        return max_kvpr(assign, load, mem)

    orders = [base_order]
    for _ in range(20):
        o = list(range(n))
        random.shuffle(o)
        orders.append(o)
    # biased orders: sorted by size desc, req desc
    orders.append(sorted(range(n), key=lambda i: sizes[i], reverse=True))
    orders.append(sorted(range(n), key=lambda i: req[i], reverse=True))

    for order in orders:
        assign = greedy(order)
        if assign is None:
            continue
        val = try_local_search(assign)
        if val < best_val - 1e-12:
            best_val = val
            best_assign = list(assign)

    if best_assign is None:
        raise ValueError("Unable to place models on any GPU")

    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    for i in range(n):
        placement[best_assign[i]].append(models[i])
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
