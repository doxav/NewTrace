GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


import random
import time

_EPS = 1e-9


def _kvpr(weighted_req_rate, shared_kv):
    # Division-safe: a fully-packed GPU (mem == 0) yields huge pressure
    return max(w / max(m, _EPS) for w, m in zip(weighted_req_rate, shared_kv))


def _argmax(wrr, mem):
    best_i, best_v = 0, -1.0
    for i in range(len(wrr)):
        v = wrr[i] / max(mem[i], _EPS)
        if v > best_v:
            best_v, best_i = v, i
    return best_i


def _greedy(gpu_num, models, order):
    placement = {g: [] for g in range(gpu_num)}
    shared_kv = [float(GPU_MEM_SIZE) for _ in range(gpu_num)]
    wrr = [0.0 for _ in range(gpu_num)]
    for model in order:
        best_idx = None
        best_score = float("inf")
        for g in range(gpu_num):
            if model.model_size <= shared_kv[g] - _EPS:
                # prefer lower resulting KVPR, tie-break on free memory
                score = (wrr[g] + model.req_rate / model.slo) / (shared_kv[g] - model.model_size)
                if score < best_score:
                    best_score = score
                    best_idx = g
        if best_idx is None:
            return None, None, None
        placement[best_idx].append(model)
        wrr[best_idx] += model.req_rate / model.slo
        shared_kv[best_idx] -= model.model_size
    return placement, wrr, shared_kv


def _try_move(placement, wrr, mem, mdl, g_src, g_dst):
    placement[g_src].remove(mdl)
    placement[g_dst].append(mdl)
    wrr[g_src] -= mdl.req_rate / mdl.slo
    wrr[g_dst] += mdl.req_rate / mdl.slo
    mem[g_src] += mdl.model_size
    mem[g_dst] -= mdl.model_size


def _undo_move(placement, wrr, mem, mdl, g_src, g_dst):
    placement[g_dst].remove(mdl)
    placement[g_src].append(mdl)
    wrr[g_src] += mdl.req_rate / mdl.slo
    wrr[g_dst] -= mdl.req_rate / mdl.slo
    mem[g_src] -= mdl.model_size
    mem[g_dst] += mdl.model_size


def _local_search(best_placement, best_wrr, best_mem, best_val, gpu_num, deadline):
    """Moves, double-moves off the max GPU, and swaps improving max-KVPR."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # targeted double-moves: relocate two models from the max-KVPR GPU
        g_max = _argmax(best_wrr, best_mem)
        src_models = list(best_placement[g_max])
        for a in range(len(src_models)):
            if time.time() >= deadline:
                break
            for b in range(a + 1, len(src_models)):
                m1, m2 = src_models[a], src_models[b]
                for g_dst in range(gpu_num):
                    if g_dst == g_max:
                        continue
                    if m1.model_size + m2.model_size > best_mem[g_dst]:
                        continue
                    _try_move(best_placement, best_wrr, best_mem, m1, g_max, g_dst)
                    _try_move(best_placement, best_wrr, best_mem, m2, g_max, g_dst)
                    v = _kvpr(best_wrr, best_mem)
                    if v < best_val - 1e-12:
                        best_val = v
                        improved = True
                        src_models = list(best_placement[g_max])
                        break
                    _undo_move(best_placement, best_wrr, best_mem, m2, g_max, g_dst)
                    _undo_move(best_placement, best_wrr, best_mem, m1, g_max, g_dst)
                if improved or time.time() >= deadline:
                    break
            if improved or time.time() >= deadline:
                break
        if improved or time.time() >= deadline:
            continue
        # try moves
        for g_src in range(gpu_num):
            for mdl in list(best_placement[g_src]):
                for g_dst in range(gpu_num):
                    if g_dst == g_src or mdl.model_size > best_mem[g_dst]:
                        continue
                    best_placement[g_src].remove(mdl)
                    best_placement[g_dst].append(mdl)
                    best_wrr[g_src] -= mdl.req_rate / mdl.slo
                    best_wrr[g_dst] += mdl.req_rate / mdl.slo
                    best_mem[g_src] += mdl.model_size
                    best_mem[g_dst] -= mdl.model_size
                    v = _kvpr(best_wrr, best_mem)
                    if v < best_val - 1e-12:
                        best_val = v
                        improved = True
                        break
                    best_placement[g_dst].remove(mdl)
                    best_placement[g_src].append(mdl)
                    best_wrr[g_src] += mdl.req_rate / mdl.slo
                    best_wrr[g_dst] -= mdl.req_rate / mdl.slo
                    best_mem[g_src] -= mdl.model_size
                    best_mem[g_dst] += mdl.model_size
                if time.time() >= deadline:
                    break
            if time.time() >= deadline:
                break
        if improved or time.time() >= deadline:
            continue
        # try swaps
        items = [(g, mdl) for g in range(gpu_num) for mdl in best_placement[g]]
        for i in range(len(items)):
            g1, m1 = items[i]
            for j in range(i + 1, len(items)):
                g2, m2 = items[j]
                if g1 == g2:
                    continue
                if m1.model_size - m2.model_size > best_mem[g2] or \
                   m2.model_size - m1.model_size > best_mem[g1]:
                    continue
                r1, r2 = m1.req_rate / m1.slo, m2.req_rate / m2.slo
                best_placement[g1].remove(m1)
                best_placement[g2].remove(m2)
                best_placement[g1].append(m2)
                best_placement[g2].append(m1)
                best_wrr[g1] += r2 - r1
                best_wrr[g2] += r1 - r2
                best_mem[g1] += m1.model_size - m2.model_size
                best_mem[g2] += m2.model_size - m1.model_size
                v = _kvpr(best_wrr, best_mem)
                if v < best_val - 1e-12:
                    best_val = v
                    improved = True
                    break
                best_placement[g1].remove(m2)
                best_placement[g2].remove(m1)
                best_placement[g1].append(m1)
                best_placement[g2].append(m2)
                best_wrr[g1] -= r2 - r1
                best_wrr[g2] -= r1 - r2
                best_mem[g1] -= m1.model_size - m2.model_size
                best_mem[g2] -= m2.model_size - m1.model_size
            if improved or time.time() >= deadline:
                break
    return best_val


def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Uses multi-start randomized greedy placement followed by local search
    (moves and swaps) on the maximum-KVPR objective.
    """
    if not models:
        return {g: [] for g in range(gpu_num)}

    base = sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True)
    rng = random.Random(12345)
    deadline = time.time() + 180  # time budget

    best_placement, best_wrr, best_mem = _greedy(gpu_num, models, base)
    if best_placement is None:
        raise ValueError("Unable to place all models on GPUs")
    best_val = _kvpr(best_wrr, best_mem)

    # Multi-start randomized greedy
    starts = 0
    while time.time() < deadline and starts < 300:
        starts += 1
        order = list(base)
        # random perturbation: shuffle a fraction of the order
        for _ in range(rng.randint(1, 4)):
            i, j = rng.randrange(len(order)), rng.randrange(len(order))
            order[i], order[j] = order[j], order[i]
        p, w, m = _greedy(gpu_num, models, order)
        if p is not None:
            v = _kvpr(w, m)
            if v < best_val:
                best_val, best_placement, best_wrr, best_mem = v, p, w, m

    best_val = _local_search(
        best_placement, best_wrr, best_mem, best_val, gpu_num, deadline
    )

    # Iterated local search: perturb the incumbent with random "kick" moves
    # (accepting temporary worsening), re-optimize, keep if better.
    all_models = list(models)
    cur_placement = {g: list(v) for g, v in best_placement.items()}
    cur_wrr = list(best_wrr)
    cur_mem = list(best_mem)
    cur_val = best_val
    iters = 0
    while time.time() < deadline and iters < 400:
        iters += 1
        # kick: apply a few random valid moves to the current solution
        for _ in range(rng.randint(2, 5)):
            src = rng.randrange(gpu_num)
            if not cur_placement[src]:
                continue
            mdl = cur_placement[src][rng.randrange(len(cur_placement[src]))]
            dst = rng.randrange(gpu_num)
            if dst == src or mdl.model_size > cur_mem[dst]:
                continue
            _try_move(cur_placement, cur_wrr, cur_mem, mdl, src, dst)
        # re-optimize the perturbed solution
        v = _local_search(
            cur_placement, cur_wrr, cur_mem, _kvpr(cur_wrr, cur_mem),
            gpu_num, min(time.time() + 5, deadline)
        )
        if v < best_val - 1e-12:
            best_val = v
            best_placement = {g: list(vs) for g, vs in cur_placement.items()}
            best_wrr, best_mem = list(cur_wrr), list(cur_mem)
            cur_val = v
        elif v < cur_val + 1e-12:
            cur_val = v  # accept equal/slightly better incumbents
        else:
            # revert to best incumbent
            cur_placement = {g: list(vs) for g, vs in best_placement.items()}
            cur_wrr, cur_mem, cur_val = list(best_wrr), list(best_mem), best_val

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
