GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize the maximum KVPR: greedy initial solutions + binary search on
    threshold T with a MILP feasibility check (scipy HiGHS), then a
    local-search polish (moves + swaps).
    """
    n = len(models)
    if n == 0:
        return {g: [] for g in range(gpu_num)}

    sizes = [float(m.model_size) for m in models]
    weights = [float(m.req_rate) / float(m.slo) for m in models]

    # ---------- greedy initial solutions ----------
    def greedy(order):
        placement = {g: [] for g in range(gpu_num)}
        shared_kv = [float(GPU_MEM_SIZE)] * gpu_num
        load = [0.0] * gpu_num
        for idx in order:
            best_g, best_key = None, None
            for g in range(gpu_num):
                if sizes[idx] <= shared_kv[g] and shared_kv[g] > 1e-9:
                    key = (load[g] / shared_kv[g], -shared_kv[g])
                    if best_key is None or key < best_key:
                        best_key, best_g = key, g
            if best_g is None:
                return None
            placement[best_g].append(models[idx])
            load[best_g] += weights[idx]
            shared_kv[best_g] -= sizes[idx]
        return placement

    def max_kvpr(placement):
        vals = []
        for g in range(gpu_num):
            used = sum(m.model_size for m in placement[g])
            load = sum(m.req_rate / m.slo for m in placement[g])
            free = GPU_MEM_SIZE - used
            vals.append(load / free if free > 1e-9 else float("inf"))
        return max(vals)

    orders = [
        sorted(range(n), key=lambda i: weights[i], reverse=True),
        sorted(range(n), key=lambda i: sizes[i], reverse=True),
        sorted(range(n), key=lambda i: weights[i] / sizes[i], reverse=True),
        list(range(n)),
    ]
    best_placement = None
    best_val = float("inf")
    for order in orders:
        p = greedy(order)
        if p is not None:
            v = max_kvpr(p)
            if v < best_val:
                best_val, best_placement = v, p

    # ---------- binary search on T with MILP feasibility ----------
    try:
        import time
        import numpy as np
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import lil_matrix

        # variables: x[i,g] (n*gpu_num binaries) then T (continuous)
        nv = n * gpu_num + 1
        T_idx = n * gpu_num
        c = np.zeros(nv)
        c[T_idx] = 1.0

        integrality = np.zeros(nv)
        integrality[:T_idx] = 1

        lb = np.zeros(nv)
        ub = np.ones(nv)
        lb[T_idx], ub[T_idx] = 0.0, 1e9

        # each model assigned exactly once
        A_assign = lil_matrix((n, nv))
        for i in range(n):
            for g in range(gpu_num):
                A_assign[i, i * gpu_num + g] = 1.0
        assign_con = LinearConstraint(A_assign.tocsr(), 1.0, 1.0)

        def build_constraints(T):
            # memory: sum_i size_i x[i,g] <= 80
            A_mem = lil_matrix((gpu_num, nv))
            for g in range(gpu_num):
                for i in range(n):
                    A_mem[g, i * gpu_num + g] = sizes[i]
            mem_con = LinearConstraint(A_mem.tocsr(), -np.inf, float(GPU_MEM_SIZE))
            # KVPR: sum_i (w_i + T*size_i) x[i,g] <= T*80
            A_kv = lil_matrix((gpu_num, nv))
            for g in range(gpu_num):
                for i in range(n):
                    A_kv[g, i * gpu_num + g] = weights[i] + T * sizes[i]
                A_kv[g, T_idx] = -80.0 * T
            kv_con = LinearConstraint(A_kv.tocsr(), -np.inf, 80.0 * T)
            return [assign_con, mem_con, kv_con]

        t_start = time.time()
        lo, hi = 1e-9, max(best_val if best_val < float("inf") else 1.0, 1e-3) * 2.0
        best_x = None
        for _ in range(12):
            if time.time() - t_start > 4.0:
                break
            T = 0.5 * (lo + hi)
            res = milp(
                c=c,
                integrality=integrality,
                bounds=Bounds(lb, ub),
                constraints=build_constraints(T),
                options={"time_limit": 0.5, "mip_rel_gap": 0.01},
            )
            if res.success and res.x is not None:
                hi = T
                best_x = res.x
            else:
                lo = T
            if hi - lo < 1e-5 or hi - lo < 1e-4 * hi:
                break

        if best_x is not None:
            placement = {g: [] for g in range(gpu_num)}
            for i in range(n):
                for g in range(gpu_num):
                    if best_x[i * gpu_num + g] > 0.5:
                        placement[g].append(models[i])
                        break
            v = max_kvpr(placement)
            if v < best_val:
                best_val, best_placement = v, placement
    except Exception:
        pass

    if best_placement is None:
        best_placement = greedy(sorted(range(n), key=lambda i: sizes[i], reverse=True))
        if best_placement is None:
            best_placement = {g: [] for g in range(gpu_num)}
            for i in range(n):
                best_placement[i % gpu_num].append(models[i])

    # ---------- local-search polish: moves, swaps, pair-moves + ILS ----------
    import random
    import time as _time

    def state_from(placement):
        load = [0.0] * gpu_num
        mem = [0.0] * gpu_num
        for g in range(gpu_num):
            for m in placement[g]:
                load[g] += m.req_rate / m.slo
                mem[g] += m.model_size
        return load, mem

    def kvpr_of(load, mem):
        best = 0.0
        for g in range(gpu_num):
            free = GPU_MEM_SIZE - mem[g]
            if free > 1e-9:
                r = load[g] / free
                if r > best:
                    best = r
        return best

    def kvpr_key(load, mem):
        # lexicographic (max, second-max) to break plateaus
        vals = sorted(
            (load[g] / (GPU_MEM_SIZE - mem[g])
             if GPU_MEM_SIZE - mem[g] > 1e-9 else float("inf"))
            for g in range(gpu_num)
        )
        return (vals[-1], vals[-2] if gpu_num >= 2 else 0.0)

    def apply_move(m, src, dst):
        best_placement[src].remove(m)
        best_placement[dst].append(m)

    def local_search(load, mem, cur_key, deadline):
        improved = True
        while improved and _time.time() < deadline:
            improved = False
            # single-model moves
            for src in range(gpu_num):
                for m in list(best_placement[src]):
                    w = m.req_rate / m.slo
                    s = m.model_size
                    for dst in range(gpu_num):
                        if dst == src or mem[dst] + s > GPU_MEM_SIZE - 1e-9:
                            continue
                        nl, nm = load[:], mem[:]
                        nl[src] -= w; nl[dst] += w
                        nm[src] -= s; nm[dst] += s
                        if kvpr_key(nl, nm) < cur_key:
                            apply_move(m, src, dst)
                            load, mem, cur_key = nl, nm, kvpr_key(nl, nm)
                            improved = True
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
                    for ma in list(best_placement[a]):
                        wa, sa = ma.req_rate / ma.slo, ma.model_size
                        for mb in list(best_placement[b]):
                            wb, sb = mb.req_rate / mb.slo, mb.model_size
                            if mem[a] - sa + sb > GPU_MEM_SIZE - 1e-9:
                                continue
                            if mem[b] - sb + sa > GPU_MEM_SIZE - 1e-9:
                                continue
                            nl, nm = load[:], mem[:]
                            nl[a] += wb - wa; nl[b] += wa - wb
                            nm[a] += sb - sa; nm[b] += sa - sb
                            if kvpr_key(nl, nm) < cur_key:
                                best_placement[a].remove(ma)
                                best_placement[b].remove(mb)
                                best_placement[a].append(mb)
                                best_placement[b].append(ma)
                                load, mem, cur_key = nl, nm, kvpr_key(nl, nm)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # pair-moves from the bottleneck GPU (two models out at once)
            b_g = max(range(gpu_num), key=lambda g: kvpr_key(load, mem)[0])
            items = list(best_placement[b_g])
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    m1, m2 = items[i], items[j]
                    w1, s1 = m1.req_rate / m1.slo, m1.model_size
                    w2, s2 = m2.req_rate / m2.slo, m2.model_size
                    for d1 in range(gpu_num):
                        if d1 == b_g or mem[d1] + s1 > GPU_MEM_SIZE - 1e-9:
                            continue
                        for d2 in range(gpu_num):
                            if d2 == b_g or d2 == d1:
                                continue
                            if mem[d2] + s2 + (s1 if d2 == d1 else 0) > GPU_MEM_SIZE - 1e-9:
                                continue
                            nl, nm = load[:], mem[:]
                            nl[b_g] -= w1 + w2
                            nm[b_g] -= s1 + s2
                            nl[d1] += w1; nm[d1] += s1
                            nl[d2] += w2; nm[d2] += s2
                            if kvpr_key(nl, nm) < cur_key:
                                apply_move(m1, b_g, d1)
                                apply_move(m2, b_g, d2)
                                load, mem, cur_key = nl, nm, kvpr_key(nl, nm)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return load, mem, cur_key

    try:
        t0 = _time.time()
        deadline = t0 + 4.0

        best_copy = {g: list(v) for g, v in best_placement.items()}
        best_key_val = float("inf")

        load, mem = state_from(best_placement)
        cur_key = kvpr_key(load, mem)
        load, mem, cur_key = local_search(load, mem, cur_key, deadline)
        best_key_val = cur_key
        best_copy = {g: list(best_placement[g]) for g in range(gpu_num)}

        # iterated local search with random kicks
        rng = random.Random(12345)
        while _time.time() < deadline:
            # kick: random feasible moves
            for _ in range(rng.randint(2, 4)):
                src = rng.randrange(gpu_num)
                if not best_placement[src]:
                    continue
                m = rng.choice(best_placement[src])
                s = m.model_size
                dsts = [g for g in range(gpu_num)
                        if g != src and mem[g] + s <= GPU_MEM_SIZE - 1e-9]
                if dsts:
                    dst = rng.choice(dsts)
                    w = m.req_rate / m.slo
                    apply_move(m, src, dst)
                    load[src] -= w; load[dst] += w
                    mem[src] -= s; mem[dst] += s
            cur_key = kvpr_key(load, mem)
            load, mem, cur_key = local_search(load, mem, cur_key, deadline)
            if cur_key < best_key_val:
                best_key_val = cur_key
                best_copy = {g: list(best_placement[g]) for g in range(gpu_num)}
            else:
                # revert to best
                for g in range(gpu_num):
                    best_placement[g][:] = best_copy[g]
                load, mem = state_from(best_placement)
                cur_key = kvpr_key(load, mem)
    except Exception:
        pass

    return best_copy


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
