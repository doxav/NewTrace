GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def _greedy(gpu_num, sorted_models):
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [float(GPU_MEM_SIZE) for _ in range(gpu_num)]
    weighted_req_rate = [0.0 for _ in range(gpu_num)]

    for model in sorted_models:
        best_idx = None
        best_ratio = float("inf")
        for gpu_id in range(gpu_num):
            if model.model_size <= shared_kv[gpu_id] and shared_kv[gpu_id] > 0:
                current_ratio = weighted_req_rate[gpu_id] / shared_kv[gpu_id]
                if current_ratio < best_ratio:
                    best_ratio = current_ratio
                    best_idx = gpu_id
        if best_idx is None:
            raise ValueError("no fit")
        placement[best_idx].append(model)
        weighted_req_rate[best_idx] += model.req_rate / model.slo
        shared_kv[best_idx] -= model.model_size
    return placement, weighted_req_rate, shared_kv


def _local_search(gpu_num, placement, weighted_req_rate, shared_kv):
    def kvpr(gid):
        return weighted_req_rate[gid] / shared_kv[gid] if shared_kv[gid] > 0 else float("inf")

    def max_kvpr():
        return max(kvpr(g) for g in range(gpu_num)) if gpu_num else 0.0

    improved = True
    while improved:
        improved = False
        cur_max = max_kvpr()
        # single-model moves
        for src in range(gpu_num):
            for mi, model in enumerate(placement[src]):
                for dst in range(gpu_num):
                    if dst == src or model.model_size > shared_kv[dst]:
                        continue
                    others = [kvpr(g) for g in range(gpu_num) if g not in (src, dst)] or [0.0]
                    src_after = (weighted_req_rate[src] - model.req_rate / model.slo) / (shared_kv[src] + model.model_size)
                    dst_den = shared_kv[dst] - model.model_size
                    dst_after = (weighted_req_rate[dst] + model.req_rate / model.slo) / dst_den if dst_den > 0 else float("inf")
                    new_max = max(others + [src_after, dst_after])
                    if new_max < cur_max - 1e-12:
                        placement[src].pop(mi)
                        placement[dst].append(model)
                        weighted_req_rate[src] -= model.req_rate / model.slo
                        weighted_req_rate[dst] += model.req_rate / model.slo
                        shared_kv[src] += model.model_size
                        shared_kv[dst] -= model.model_size
                        improved = True
                        cur_max = new_max
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # pairwise swaps
        for a in range(gpu_num):
            for b in range(a + 1, gpu_num):
                for ma in placement[a]:
                    for mb in placement[b]:
                        delta_a = mb.model_size - ma.model_size
                        if delta_a >= shared_kv[a] or -delta_a >= shared_kv[b]:
                            continue
                        ra = weighted_req_rate[a] - ma.req_rate / ma.slo + mb.req_rate / mb.slo
                        rb = weighted_req_rate[b] - mb.req_rate / mb.slo + ma.req_rate / ma.slo
                        others = [kvpr(g) for g in range(gpu_num) if g not in (a, b)]
                        new_max = max(others + [ra / (shared_kv[a] - delta_a), rb / (shared_kv[b] + delta_a)])
                        if new_max < cur_max - 1e-12:
                            ia = placement[a].index(ma)
                            ib = placement[b].index(mb)
                            placement[a][ia], placement[b][ib] = mb, ma
                            weighted_req_rate[a], weighted_req_rate[b] = ra, rb
                            shared_kv[a] -= delta_a
                            shared_kv[b] += delta_a
                            improved = True
                            cur_max = new_max
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break


def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """
    orderings = [
        sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo / m.model_size), reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo * m.model_size), reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=False),
        sorted(models, key=lambda m: m.model_size, reverse=False),
        sorted(models, key=lambda m: (m.req_rate / m.slo / m.model_size), reverse=False),
    ]

    best_placement = None
    best_max = float("inf")

    def evaluate(wrr, skv):
        return max(
            (wrr[g] / skv[g] for g in range(len(skv)) if skv[g] > 0),
            default=0.0,
        )

    def try_ordering(sorted_models, rng=None):
        """Greedy placement, optionally with random tie-breaking/perturbation."""
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        shared_kv = [float(GPU_MEM_SIZE) for _ in range(gpu_num)]
        weighted_req_rate = [0.0 for _ in range(gpu_num)]
        for model in sorted_models:
            candidates = []
            for gpu_id in range(gpu_num):
                if model.model_size <= shared_kv[gpu_id] and shared_kv[gpu_id] > 0:
                    candidates.append((weighted_req_rate[gpu_id] / shared_kv[gpu_id], gpu_id))
            if not candidates:
                return None, None, None
            candidates.sort()
            if rng is not None and len(candidates) > 1:
                # pick among the near-best candidates randomly
                k = min(len(candidates), 3)
                best_idx = candidates[rng.randrange(k)][1]
            else:
                best_idx = candidates[0][1]
            placement[best_idx].append(model)
            weighted_req_rate[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        _local_search(gpu_num, placement, weighted_req_rate, shared_kv)
        return placement, weighted_req_rate, shared_kv

    import random
    rng = random.Random(12345)

    for sorted_models in orderings:
        for attempt in range(6):
            use_rng = rng if attempt > 0 else None
            if use_rng is not None:
                shuffled = list(sorted_models)
                rng.shuffle(shuffled)
                # keep a light bias: sort back by descending pressure, stable-ish
                shuffled.sort(key=lambda m: (m.req_rate / m.slo), reverse=True)
                # apply small random perturbation by rotating
                cut = rng.randrange(len(shuffled) + 1)
                shuffled = shuffled[cut:] + shuffled[:cut]
                placement, wrr, skv = try_ordering(shuffled, use_rng)
            else:
                placement, wrr, skv = try_ordering(sorted_models)
            if placement is None:
                continue
            cur = evaluate(wrr, skv)
            if cur < best_max - 1e-15:
                best_max = cur
                best_placement = placement
    if best_placement is None:
        raise ValueError("Unable to place all models on GPUs")
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
