GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize the maximum KVPR across GPUs.
    Greedy placement from multiple orderings, then local search (moves + swaps).
    """

    def kvpr(wrr, kv, g):
        return wrr[g] / kv[g] if kv[g] > 1e-12 else float("inf")

    def max_kvpr(wrr, kv):
        return max(kvpr(wrr, kv, g) for g in range(gpu_num))

    def run_greedy(order):
        placement = {g: [] for g in range(gpu_num)}
        shared_kv = [float(GPU_MEM_SIZE) for _ in range(gpu_num)]
        wrr = [0.0 for _ in range(gpu_num)]
        for model in order:
            best_idx = None
            best_val = float("inf")
            # First pass: keep at least a little free memory (avoid zero denominator)
            for strict in (True, False):
                for g in range(gpu_num):
                    if strict:
                        if model.model_size >= shared_kv[g]:
                            continue
                    elif model.model_size > shared_kv[g]:
                        continue
                    val = (wrr[g] + model.req_rate / model.slo) / (
                        shared_kv[g] - model.model_size
                    )
                    if val < best_val:
                        best_val = val
                        best_idx = g
                if best_idx is not None:
                    break
            if best_idx is None:
                return None, None, None
            placement[best_idx].append(model)
            wrr[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        return placement, shared_kv, wrr

    base = list(models)
    orders = [
        sorted(base, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(base, key=lambda m: m.req_rate / m.slo),
        sorted(base, key=lambda m: m.model_size, reverse=True),
        sorted(base, key=lambda m: m.model_size),
        sorted(base, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True),
        sorted(base, key=lambda m: (m.req_rate / m.slo) * m.model_size, reverse=True),
        base,
    ]
    # Randomized restarts for extra diversity
    import random

    rng = random.Random(12345)
    for _ in range(30):
        o = base[:]
        rng.shuffle(o)
        orders.append(o)

    def general_swaps(placement, shared, wrr):
        """Try swapping any pair of models between any two GPUs to reduce max KVPR."""
        cur_max = max_kvpr(wrr, shared)
        best = None
        best_new_max = cur_max
        for a in range(gpu_num):
            for b in range(a + 1, gpu_num):
                for m1 in placement[a]:
                    for m2 in placement[b]:
                        if m1 is m2:
                            continue
                        r1 = m1.req_rate / m1.slo
                        r2 = m2.req_rate / m2.slo
                        free_a = shared[a] + m1.model_size
                        free_b = shared[b] + m2.model_size
                        # strict checks: denominators must stay > 0
                        if m2.model_size >= free_a or m1.model_size >= free_b:
                            continue
                        new_a = (wrr[a] - r1 + r2) / (free_a - m2.model_size)
                        new_b = (wrr[b] - r2 + r1) / (free_b - m1.model_size)
                        new_max = max(
                            max(
                                kvpr(wrr, shared, x)
                                for x in range(gpu_num)
                                if x not in (a, b)
                            ),
                            new_a,
                            new_b,
                        )
                        if new_max < best_new_max - 1e-12:
                            best_new_max = new_max
                            best = (a, m1, b, m2)
        if best is None:
            return False
        a, m1, b, m2 = best
        r1 = m1.req_rate / m1.slo
        r2 = m2.req_rate / m2.slo
        placement[a].remove(m1)
        placement[b].remove(m2)
        placement[a].append(m2)
        placement[b].append(m1)
        wrr[a] += r2 - r1
        shared[a] += m1.model_size - m2.model_size
        wrr[b] += r1 - r2
        shared[b] += m2.model_size - m1.model_size
        return True

    def full_local_search(placement, shared, wrr):
        placement, shared, wrr = local_search(placement, shared, wrr)
        for _ in range(50):
            if not general_swaps(placement, shared, wrr):
                break
            placement, shared, wrr = local_search(placement, shared, wrr)
        return placement, shared, wrr

    def local_search(placement, shared, wrr):
        """Move / swap models off the hottest GPU until no improvement."""
        placement = {g: list(v) for g, v in placement.items()}
        shared = list(shared)
        wrr = list(wrr)
        improved = True
        iterations = 0
        while improved and iterations < 300:
            improved = False
            iterations += 1
            cur_max = max_kvpr(wrr, shared)
            hot = max(range(gpu_num), key=lambda g: kvpr(wrr, shared, g))

            best_move = None
            best_new_max = cur_max
            for mi, model in enumerate(placement[hot]):
                rate = model.req_rate / model.slo
                for g in range(gpu_num):
                    if g == hot or model.model_size >= shared[g]:
                        continue
                    new_hot = (wrr[hot] - rate) / (shared[hot] + model.model_size)
                    new_g = (wrr[g] + rate) / (shared[g] - model.model_size)
                    new_max = max(
                        max(
                            kvpr(wrr, shared, x)
                            for x in range(gpu_num)
                            if x not in (hot, g)
                        ),
                        new_hot,
                        new_g,
                    )
                    if new_max < best_new_max - 1e-12:
                        best_new_max = new_max
                        best_move = (mi, g)
            if best_move is not None:
                mi, g = best_move
                model = placement[hot].pop(mi)
                rate = model.req_rate / model.slo
                wrr[hot] -= rate
                shared[hot] += model.model_size
                placement[g].append(model)
                wrr[g] += rate
                shared[g] -= model.model_size
                improved = True
                continue

            best_swap = None
            best_new_max = cur_max
            for mi, m1 in enumerate(placement[hot]):
                r1 = m1.req_rate / m1.slo
                for g in range(gpu_num):
                    if g == hot:
                        continue
                    for m2 in placement[g]:
                        r2 = m2.req_rate / m2.slo
                        if r2 >= r1:
                            continue
                        hot_free = shared[hot] + m1.model_size
                        g_free = shared[g] + m2.model_size
                        if m2.model_size >= hot_free or m1.model_size >= g_free:
                            continue
                        new_hot = (wrr[hot] - r1 + r2) / (hot_free - m2.model_size)
                        new_g = (wrr[g] - r2 + r1) / (g_free - m1.model_size)
                        new_max = max(
                            max(
                                kvpr(wrr, shared, x)
                                for x in range(gpu_num)
                                if x not in (hot, g)
                            ),
                            new_hot,
                            new_g,
                        )
                        if new_max < best_new_max - 1e-12:
                            best_new_max = new_max
                            best_swap = (mi, g, m2)
            if best_swap is not None:
                mi, g, m2 = best_swap
                m1 = placement[hot][mi]
                r1 = m1.req_rate / m1.slo
                r2 = m2.req_rate / m2.slo
                placement[hot].remove(m1)
                placement[g].remove(m2)
                placement[hot].append(m2)
                placement[g].append(m1)
                wrr[hot] += r2 - r1
                shared[hot] += m1.model_size - m2.model_size
                wrr[g] += r1 - r2
                shared[g] += m2.model_size - m1.model_size
                improved = True
        return placement, shared, wrr

    # Run greedy from multiple orderings, then local-search the best few starts
    results = []
    for order in orders:
        p, kv, w = run_greedy(order)
        if p is not None:
            results.append((max_kvpr(w, kv), p, kv, w))

    if not results:
        raise ValueError("Unable to place models on any GPU")

    results.sort(key=lambda t: t[0])
    best_placement, best_shared, best_wrr = None, None, None
    best_max = float("inf")
    for _, p, kv, w in results[:10]:
        p2, kv2, w2 = full_local_search(p, kv, w)
        m = max_kvpr(w2, kv2)
        if m < best_max:
            best_max, best_placement, best_shared, best_wrr = m, p2, kv2, w2

    # Iterated local search: perturb best solution, re-optimize, keep if improved
    import random as _random

    rng2 = _random.Random(999)
    all_models = list(models)
    cur_p, cur_s, cur_w = best_placement, best_shared, best_wrr
    for _ in range(60):
        # Perturb: randomly move a few models between GPUs (feasible moves only)
        p = {g: list(v) for g, v in cur_p.items()}
        s = list(cur_s)
        w = list(cur_w)
        for _try in range(3):
            src = rng2.randrange(gpu_num)
            if not p[src]:
                continue
            mi = rng2.randrange(len(p[src]))
            model = p[src][mi]
            cands = [
                g
                for g in range(gpu_num)
                if g != src and model.model_size < s[g] and s[g] - model.model_size > 1e-12
            ]
            if not cands:
                continue
            g = rng2.choice(cands)
            rate = model.req_rate / model.slo
            p[src].pop(mi)
            w[src] -= rate
            s[src] += model.model_size
            p[g].append(model)
            w[g] += rate
            s[g] -= model.model_size
        p2, s2, w2 = full_local_search(p, s, w)
        m2 = max_kvpr(w2, s2)
        if m2 < best_max - 1e-12:
            best_max, best_placement, best_shared, best_wrr = m2, p2, s2, w2
            cur_p, cur_s, cur_w = p2, s2, w2
        elif m2 < max_kvpr(cur_w, cur_s) + 1e-12:
            # accept equal/slightly worse to diversify
            cur_p, cur_s, cur_w = p2, s2, w2

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
