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

    def gpu_kvpr(wrr, skv, idx):
        return wrr[idx] / skv[idx] if skv[idx] > 0 else float("inf")

    def greedy(order):
        """Greedy KVPR-minimizing placement. Returns (placement, skv, wrr) or None."""
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]
        weighted_req_rate = [0.0 for _ in range(gpu_num)]

        for model in order:
            best_idx = None
            best_ratio = float("inf")
            for gpu_id in range(gpu_num):
                if 0 < model.model_size <= shared_kv[gpu_id]:
                    current_ratio = weighted_req_rate[gpu_id] / shared_kv[gpu_id]
                    if current_ratio < best_ratio:
                        best_ratio = current_ratio
                        best_idx = gpu_id
            if best_idx is None:
                return None
            placement[best_idx].append(model)
            weighted_req_rate[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        return placement, shared_kv, weighted_req_rate

    def local_search(placement, shared_kv, weighted_req_rate):
        """Reduce max KVPR via moves and swaps. Returns final max KVPR."""
        for _ in range(200):
            current_max = max(gpu_kvpr(weighted_req_rate, shared_kv, i) for i in range(gpu_num))
            improved = False

            # Try moving a single model from the most-loaded GPU
            src = max(range(gpu_num), key=lambda i: gpu_kvpr(weighted_req_rate, shared_kv, i))
            for m in list(placement[src]):
                w = m.req_rate / m.slo
                for dst in range(gpu_num):
                    if dst == src or m.model_size >= shared_kv[dst]:
                        continue
                    new_src = (weighted_req_rate[src] - w) / (shared_kv[src] + m.model_size)
                    new_dst = (weighted_req_rate[dst] + w) / (shared_kv[dst] - m.model_size)
                    others_max = max(
                        (gpu_kvpr(weighted_req_rate, shared_kv, i) for i in range(gpu_num) if i not in (src, dst)),
                        default=0.0,
                    )
                    if max(new_src, new_dst, others_max) < current_max - 1e-12:
                        placement[src].remove(m)
                        placement[dst].append(m)
                        weighted_req_rate[src] -= w
                        weighted_req_rate[dst] += w
                        shared_kv[src] += m.model_size
                        shared_kv[dst] -= m.model_size
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue

            # Try swapping two models between GPUs
            done = False
            for a in range(gpu_num):
                for b in range(a + 1, gpu_num):
                    for ma in placement[a]:
                        wa = ma.req_rate / ma.slo
                        for mb in placement[b]:
                            wb = mb.req_rate / mb.slo
                            free_a = shared_kv[a] + ma.model_size - mb.model_size
                            free_b = shared_kv[b] + mb.model_size - ma.model_size
                            if free_a <= 0 or free_b <= 0:
                                continue
                            na = (weighted_req_rate[a] - wa + wb) / free_a
                            nb = (weighted_req_rate[b] - wb + wa) / free_b
                            others_max = max(
                                (gpu_kvpr(weighted_req_rate, shared_kv, i) for i in range(gpu_num) if i not in (a, b)),
                                default=0.0,
                            )
                            if max(na, nb, others_max) < current_max - 1e-12:
                                placement[a].remove(ma)
                                placement[b].remove(mb)
                                placement[a].append(mb)
                                placement[b].append(ma)
                                weighted_req_rate[a] += wb - wa
                                weighted_req_rate[b] += wa - wb
                                shared_kv[a] += ma.model_size - mb.model_size
                                shared_kv[b] += mb.model_size - ma.model_size
                                improved = True
                                done = True
                                break
                        if done:
                            break
                    if done:
                        break
                if done:
                    break
            if not improved:
                break

        return max(gpu_kvpr(weighted_req_rate, shared_kv, i) for i in range(gpu_num))

    # Multi-start: several deterministic orderings + randomized restarts, keep the best
    base_orders = [
        sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True),
        sorted(models, key=lambda m: m.model_size),
    ]
    rng = random.Random(12345)

    best_placement = None
    best_score = float("inf")

    for it in range(60):
        if it < len(base_orders):
            order = base_orders[it]
        else:
            order = list(base_orders[0])
            for _ in range(rng.randint(1, max(1, len(order) // 2))):
                i, j = rng.randrange(len(order)), rng.randrange(len(order))
                order[i], order[j] = order[j], order[i]

        result = greedy(order)
        if result is None:
            continue
        p, skv, wrr = result
        score = local_search(p, skv, wrr)
        if score < best_score - 1e-15:
            best_score = score
            best_placement = p

    if best_placement is None:
        raise ValueError("Unable to place models on any GPU")

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
