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

    def greedy(order):
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]
        load = [0.0 for _ in range(gpu_num)]

        for model in order:
            best_idx = None
            best_kvpr = float("inf")
            for gpu_id in range(gpu_num):
                if model.model_size <= shared_kv[gpu_id]:
                    new_kvpr = (load[gpu_id] + model.req_rate / model.slo) / (
                        shared_kv[gpu_id] - model.model_size
                    )
                    if new_kvpr < best_kvpr:
                        best_kvpr = new_kvpr
                        best_idx = gpu_id
            if best_idx is None:
                return None
            placement[best_idx].append(model)
            load[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        return placement

    def max_kvpr(placement):
        best = 0.0
        for gpu_models in placement.values():
            denom = GPU_MEM_SIZE - sum(m.model_size for m in gpu_models)
            if denom <= 0:
                return float("inf")
            kvpr = sum(m.req_rate / m.slo for m in gpu_models) / denom
            best = max(best, kvpr)
        return best

    # Try several deterministic orderings plus randomized restarts
    candidate_orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True),
        list(models),
    ]

    def place_anywhere(order):
        # Always places feasibly when possible: pick GPU minimizing resulting
        # KVPR; if none fits, place on GPU with most free memory (best effort).
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]
        load = [0.0 for _ in range(gpu_num)]
        # First pass: place large models first to keep packing feasible.
        for model in sorted(order, key=lambda m: m.model_size, reverse=True):
            req = model.req_rate / model.slo
            best_idx = None
            best_kvpr = float("inf")
            for gpu_id in range(gpu_num):
                if model.model_size <= shared_kv[gpu_id]:
                    new_kvpr = (load[gpu_id] + req) / (shared_kv[gpu_id] - model.model_size)
                    if new_kvpr < best_kvpr:
                        best_kvpr = new_kvpr
                        best_idx = gpu_id
            if best_idx is None:
                best_idx = max(range(gpu_num), key=lambda g: shared_kv[g])
            placement[best_idx].append(model)
            load[best_idx] += req
            shared_kv[best_idx] -= model.model_size
        return placement

    def local_search(placement, loads, used, iters=200):
        # Incremental first-improvement local search over moves and swaps,
        # always attacking the current max-KVPR GPU.
        placement = {g: list(ms) for g, ms in placement.items()}
        best = 0.0
        for g in range(gpu_num):
            denom = GPU_MEM_SIZE - used[g]
            k = loads[g] / denom if denom > 1e-12 else float("inf")
            if k > best:
                best = k
        for _ in range(iters):
            src = max(range(gpu_num),
                      key=lambda g: loads[g] / max(GPU_MEM_SIZE - used[g], 1e-12))
            improved = False
            for mi in range(len(placement[src])):
                model = placement[src][mi]
                req = model.req_rate / model.slo
                sz = model.model_size
                for dst in range(gpu_num):
                    if dst == src:
                        continue
                    # Try move
                    if sz <= GPU_MEM_SIZE - used[dst]:
                        loads[src] -= req; used[src] -= sz
                        loads[dst] += req; used[dst] += sz
                        score = max(
                            (loads[g] / (GPU_MEM_SIZE - used[g])
                             if GPU_MEM_SIZE - used[g] > 1e-12 else float("inf"))
                            for g in range(gpu_num))
                        if score < best - 1e-12:
                            m = placement[src].pop(mi)
                            placement[dst].append(m)
                            best = score
                            improved = True
                            break
                        loads[src] += req; used[src] += sz
                        loads[dst] -= req; used[dst] -= sz
                    # Try swap
                    done = False
                    for dj in range(len(placement[dst])):
                        other = placement[dst][dj]
                        oreq = other.req_rate / other.slo
                        osz = other.model_size
                        if used[dst] - osz + sz > GPU_MEM_SIZE:
                            continue
                        if used[src] - sz + osz > GPU_MEM_SIZE:
                            continue
                        loads[src] += oreq - req; used[src] += osz - sz
                        loads[dst] += req - oreq; used[dst] += sz - osz
                        score = max(
                            (loads[g] / (GPU_MEM_SIZE - used[g])
                             if GPU_MEM_SIZE - used[g] > 1e-12 else float("inf"))
                            for g in range(gpu_num))
                        if score < best - 1e-12:
                            placement[src][mi], placement[dst][dj] = other, model
                            best = score
                            improved = True
                            done = True
                            break
                        loads[src] -= oreq - req; used[src] -= osz - sz
                        loads[dst] -= req - oreq; used[dst] -= sz - osz
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return placement, loads, used, best

    def state_of(placement):
        loads = [0.0] * gpu_num
        used = [0.0] * gpu_num
        for g, ms in placement.items():
            for m in ms:
                loads[g] += m.req_rate / m.slo
                used[g] += m.model_size
        return loads, used

    # Binary search on threshold T: max KVPR <= T iff every GPU satisfies
    # sum(model_size + req/T) <= 80, i.e. bin-packing with effective sizes.
    reqs = [m.req_rate / m.slo for m in models]
    sizes = [m.model_size for m in models]
    total_load = sum(reqs)
    total_size = sum(sizes)
    lo = total_load / max(GPU_MEM_SIZE * gpu_num - total_size, 1e-9)
    hi = max(total_load / max(GPU_MEM_SIZE - max(sizes) if sizes else 1, 1e-9), lo) * 2 + 1.0
    order_keys = [
        lambda i: -(sizes[i] + reqs[i]),
        lambda i: -(sizes[i]),
        lambda i: -(reqs[i]),
        lambda i: -(reqs[i] / max(sizes[i], 1e-9)),
    ]

    def pack(T, key):
        placement = {g: [] for g in range(gpu_num)}
        used = [0.0] * gpu_num
        for i in sorted(range(len(models)), key=key):
            eff = sizes[i] + reqs[i] / T
            if eff > GPU_MEM_SIZE:
                return None
            placed = False
            for g in range(gpu_num):
                if used[g] + eff <= GPU_MEM_SIZE + 1e-12:
                    placement[g].append(models[i])
                    used[g] += eff
                    placed = True
                    break
            if not placed:
                return None
        return placement

    binary_placements = []
    for _ in range(60):
        mid = (lo + hi) / 2.0
        found = None
        for key in order_keys:
            p = pack(mid, key)
            if p is not None:
                found = p
                break
        if found is not None:
            binary_placements.append(found)
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-4:
            break

    rng = random.Random(42)
    best_placement = None
    best_score = float("inf")
    attempts = 0
    max_attempts = 40
    while attempts < max_attempts:
        if attempts < len(candidate_orders):
            order = candidate_orders[attempts]
        else:
            order = list(models)
            rng.shuffle(order)
        attempts += 1
        placement = greedy(order)
        if placement is None:
            placement = place_anywhere(order)
        loads, used = state_of(placement)
        placement, loads, used, score = local_search(placement, loads, used)
        if score < best_score:
            best_score = score
            best_placement = placement

    # Also refine binary-search placements with local search.
    for bp in binary_placements:
        loads, used = state_of(bp)
        bp, loads, used, score = local_search(bp, loads, used)
        if score < best_score:
            best_score = score
            best_placement = bp

    if best_placement is None:
        if models:
            best_placement = place_anywhere(sorted(models, key=lambda m: m.model_size, reverse=True))
        else:
            best_placement = {g: [] for g in range(gpu_num)}
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
