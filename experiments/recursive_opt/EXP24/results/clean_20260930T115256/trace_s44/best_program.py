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

    n = len(models)
    if n == 0 or gpu_num <= 0:
        return {gpu_id: [] for gpu_id in range(gpu_num)}

    def evaluate(order):
        """Greedy: assign each model to GPU minimizing resulting KVPR."""
        placement = {g: [] for g in range(gpu_num)}
        free = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for m in order:
            best_idx = None
            best_kvpr = float("inf")
            for g in range(gpu_num):
                rem = free[g] - m.model_size
                if rem > 0:
                    new_kvpr = (load[g] + m.req_rate / m.slo) / rem
                    if new_kvpr < best_kvpr:
                        best_kvpr = new_kvpr
                        best_idx = g
            if best_idx is None:
                return None
            placement[best_idx].append(m)
            load[best_idx] += m.req_rate / m.slo
            free[best_idx] -= m.model_size
        return placement, max(load[g] / free[g] for g in range(gpu_num))

    def max_kvpr(placement):
        loads = [sum(m.req_rate / m.slo for m in placement[g]) for g in range(gpu_num)]
        frees = [GPU_MEM_SIZE - sum(m.model_size for m in placement[g]) for g in range(gpu_num)]
        return max(l / f if f > 0 else float("inf") for l, f in zip(loads, frees))

    rng = random.Random(12345)
    best = None
    best_kvpr = float("inf")

    # Deterministic orderings first
    orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo / m.model_size, reverse=True),
        sorted(models, key=lambda m: m.model_size / (m.req_rate / m.slo), reverse=True),
    ]
    for order in orders:
        res = evaluate(order)
        if res and res[1] < best_kvpr:
            best, best_kvpr = res[0], res[1]

    # Randomized restarts
    for _ in range(600):
        order = models[:]
        rng.shuffle(order)
        res = evaluate(order)
        if res and res[1] < best_kvpr:
            best, best_kvpr = res[0], res[1]

    if best is None:
        best = {g: [] for g in range(gpu_num)}
        for i, m in enumerate(models):
            best[i % gpu_num].append(m)
        return best

    # Local search: move / swap models between GPUs to reduce max KVPR
    def local_search(sol):
        improved = True
        while improved:
            improved = False
            cur = max_kvpr(sol)
            for src in range(gpu_num):
                for m in list(sol[src]):
                    for dst in range(gpu_num):
                        if dst == src:
                            continue
                        used_dst = sum(x.model_size for x in sol[dst])
                        if used_dst + m.model_size >= GPU_MEM_SIZE:
                            continue
                        sol[src].remove(m)
                        sol[dst].append(m)
                        new_kvpr = max_kvpr(sol)
                        if new_kvpr < cur - 1e-12:
                            improved = True
                            break
                        sol[dst].remove(m)
                        sol[src].append(m)
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            for a in range(gpu_num):
                for b in range(a + 1, gpu_num):
                    for ma in list(sol[a]):
                        for mb in list(sol[b]):
                            sol[a].remove(ma)
                            sol[b].remove(mb)
                            sol[a].append(mb)
                            sol[b].append(ma)
                            if max_kvpr(sol) < cur - 1e-12:
                                improved = True
                                break
                            sol[a].remove(mb)
                            sol[b].remove(ma)
                            sol[a].append(ma)
                            sol[b].append(mb)
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return sol

    best = local_search(best)
    best_kvpr = max_kvpr(best)

    # Iterated local search: perturb and re-optimize
    for _ in range(60):
        cand = {g: list(v) for g, v in best.items()}
        all_models = [m for g in range(gpu_num) for m in cand[g]]
        for _ in range(max(1, n // 4)):
            m = rng.choice(all_models)
            src = next(g for g in range(gpu_num) if m in cand[g])
            dst = rng.randrange(gpu_num)
            if dst == src:
                continue
            if sum(x.model_size for x in cand[dst]) + m.model_size < GPU_MEM_SIZE:
                cand[src].remove(m)
                cand[dst].append(m)
        try:
            cand = local_search(cand)
            kv = max_kvpr(cand)
            if kv < best_kvpr - 1e-12:
                best, best_kvpr = cand, kv
        except Exception:
            pass
    return best


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
