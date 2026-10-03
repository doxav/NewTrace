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
        # Greedy KVPR-minimizing placement for a given model order
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]
        weighted_req_rate = [0.0 for _ in range(gpu_num)]

        for model in order:
            best_idx = None
            best_ratio = float("inf")

            for gpu_id in range(gpu_num):
                if model.model_size <= shared_kv[gpu_id] and shared_kv[gpu_id] > 0:
                    current_ratio = weighted_req_rate[gpu_id] / shared_kv[gpu_id]
                    if current_ratio < best_ratio:
                        best_ratio = current_ratio
                        best_idx = gpu_id

            if best_idx is None:
                raise ValueError(
                    f"Unable to place model of size {model.model_size} GB on any GPU. "
                    f"Remaining per-GPU memory: {shared_kv}"
                )

            placement[best_idx].append(model)
            weighted_req_rate[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size

        return placement, shared_kv, weighted_req_rate

    # Try several greedy orders, keep the best initial placement by max KVPR
    orders = [
        sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo)),
        sorted(models, key=lambda m: (m.req_rate / m.slo / m.model_size), reverse=True),
        sorted(models, key=lambda m: (m.model_size), reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo / m.model_size)),
        sorted(models, key=lambda m: (m.model_size, m.req_rate / m.slo), reverse=True),
        sorted(models, key=lambda m: (m.slo / m.req_rate)),
        sorted(models, key=lambda m: (m.slo / m.req_rate), reverse=True),
    ]

    # Add deterministic random restarts for extra diversity
    import random
    rng = random.Random(12345)
    base = list(models)
    for _ in range(40):
        shuffled = base[:]
        rng.shuffle(shuffled)
        orders.append(shuffled)
    # Perturbed heuristic orders
    for _ in range(20):
        shuffled = base[:]
        rng.shuffle(shuffled)
        shuffled.sort(key=lambda m: m.req_rate / m.slo / m.model_size, reverse=True)
        # swap a few random pairs to perturb
        for _ in range(max(1, len(shuffled) // 5)):
            i, j = rng.randrange(len(shuffled)), rng.randrange(len(shuffled))
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        orders.append(shuffled)

    def local_search(placement, shared_kv, weighted_req_rate):
        def kvpr(gpu_id):
            free = shared_kv[gpu_id]
            return weighted_req_rate[gpu_id] / free if free > 1e-9 else float("inf")

        def max_kvpr():
            return max(kvpr(g) for g in range(gpu_num))

        improved = True
        max_iters = 200
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            cur_max = max_kvpr()

            # Try single-model moves
            for src in range(gpu_num):
                if not placement[src]:
                    continue
                for mi, model in enumerate(placement[src]):
                    w = model.req_rate / model.slo
                    for dst in range(gpu_num):
                        if dst == src or model.model_size > shared_kv[dst]:
                            continue
                        old_max = max(kvpr(g) for g in (src, dst))
                        den_s = shared_kv[src] + model.model_size
                        den_d = shared_kv[dst] - model.model_size
                        if den_d <= 1e-9:
                            continue
                        new_src = (weighted_req_rate[src] - w) / den_s if den_s > 1e-9 else float("inf")
                        new_dst = (weighted_req_rate[dst] + w) / den_d
                        if max(new_src, new_dst) < old_max - 1e-12 and max(new_src, new_dst) < cur_max:
                            placement[src].pop(mi)
                            placement[dst].append(model)
                            weighted_req_rate[src] -= w
                            weighted_req_rate[dst] += w
                            shared_kv[src] += model.model_size
                            shared_kv[dst] -= model.model_size
                            cur_max = max_kvpr()
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

            if improved:
                continue

            # Try pairwise swaps
            for a in range(gpu_num):
                for b in range(a + 1, gpu_num):
                    done = False
                    for ma in placement[a]:
                        wa = ma.req_rate / ma.slo
                        for mb in placement[b]:
                            wb = mb.req_rate / mb.slo
                            if ma.model_size - mb.model_size > shared_kv[b]:
                                continue
                            if mb.model_size - ma.model_size > shared_kv[a]:
                                continue
                            den_a = shared_kv[a] + ma.model_size - mb.model_size
                            den_b = shared_kv[b] + mb.model_size - ma.model_size
                            if den_a <= 1e-9 or den_b <= 1e-9:
                                continue
                            new_a = (weighted_req_rate[a] - wa + wb) / den_a
                            new_b = (weighted_req_rate[b] - wb + wa) / den_b
                            old_max = max(kvpr(a), kvpr(b))
                            if max(new_a, new_b) < old_max - 1e-12 and max(new_a, new_b) < cur_max:
                                placement[a].remove(ma)
                                placement[b].remove(mb)
                                placement[a].append(mb)
                                placement[b].append(ma)
                                weighted_req_rate[a] += wb - wa
                                weighted_req_rate[b] += wa - wb
                                shared_kv[a] += ma.model_size - mb.model_size
                                shared_kv[b] += mb.model_size - ma.model_size
                                cur_max = max_kvpr()
                                improved = True
                                done = True
                                break
                        if done:
                            break
                    if done:
                        break
                if done:
                    break

        return max_kvpr()

    # Run local search from every greedy order, keep the overall best result.
    # Early-exit if we reach a very good placement.
    best_state = None
    best_final_max = float("inf")
    for order in orders:
        try:
            p, sk, wr = greedy(order)
        except ValueError:
            continue
        final_max = local_search(p, sk, wr)
        if final_max < best_final_max:
            best_final_max = final_max
            best_state = (p, sk, wr)
        if best_final_max <= 1e-9:
            break

    if best_state is None:
        raise ValueError("Unable to place models on any GPU.")

    # Iterated local search: ruin-and-recreate perturbations around the best.
    def copy_state(state):
        p, sk, wr = state
        return ({g: list(ms) for g, ms in p.items()}, list(sk), list(wr))

    rng2 = random.Random(98765)
    cur_state = copy_state(best_state)
    cur_max = best_final_max
    for _ in range(150):
        p, sk, wr = copy_state(cur_state)
        # Perturb: move a few random models to random feasible GPUs
        all_models = [m for ms in p.values() for m in ms]
        n_perturb = min(len(all_models), rng2.randint(2, 4))
        for _ in range(n_perturb):
            m = rng2.choice(all_models)
            srcs = [g for g in range(gpu_num) if m in p[g]]
            if not srcs:
                continue
            src = srcs[0]
            feasible = [
                g for g in range(gpu_num)
                if g != src and m.model_size <= sk[g] and sk[g] - m.model_size > 1e-9
            ]
            if not feasible:
                continue
            dst = rng2.choice(feasible)
            w = m.req_rate / m.slo
            p[src].remove(m)
            p[dst].append(m)
            wr[src] -= w
            wr[dst] += w
            sk[src] += m.model_size
            sk[dst] -= m.model_size
        pert_max = local_search(p, sk, wr)
        if pert_max < cur_max - 1e-12:
            cur_max = pert_max
            cur_state = (p, sk, wr)
        if pert_max < best_final_max - 1e-12:
            best_final_max = pert_max
            best_state = (p, sk, wr)
        if best_final_max <= 1e-9:
            break

    return best_state[0]


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
