GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize the maximum KVPR across GPUs via binary search on the answer,
    followed by a local-search refinement.

    For a threshold T, a GPU hosting set S is feasible iff
    sum(req_rate/slo) <= T * (GPU_MEM_SIZE - sum(model_size)) and
    sum(model_size) <= GPU_MEM_SIZE. Binary search T; each check greedily
    packs models under multiple orderings. Then refine the best placement
    with move/swap local search.
    """
    import time

    start = time.time()
    n = len(models)
    if n == 0:
        return {g: [] for g in range(gpu_num)}

    req = [m.req_rate / m.slo for m in models]
    size = [m.model_size for m in models]

    def actual_max_kvpr(assign):
        best = 0.0
        for g in range(gpu_num):
            used = 0.0
            load = 0.0
            for i in assign[g]:
                used += size[i]
                load += req[i]
            denom = GPU_MEM_SIZE - used
            if denom <= 0:
                return float("inf")
            kvpr = load / denom
            if kvpr > best:
                best = kvpr
        return best

    def check(T, order):
        """Greedy feasibility for threshold T with given model order."""
        assign = [[] for _ in range(gpu_num)]
        used = [0.0] * gpu_num
        load = [0.0] * gpu_num
        for i in order:
            best_g = -1
            best_key = None
            for g in range(gpu_num):
                new_used = used[g] + size[i]
                if new_used > GPU_MEM_SIZE:
                    continue
                new_load = load[g] + req[i]
                if new_load > T * (GPU_MEM_SIZE - new_used) + 1e-12:
                    continue
                key = GPU_MEM_SIZE - new_used
                if best_key is None or key < best_key:
                    best_key = key
                    best_g = g
            if best_g < 0:
                return None
            assign[best_g].append(i)
            used[best_g] += size[i]
            load[best_g] += req[i]
        return assign

    orders = [
        sorted(range(n), key=lambda i: size[i], reverse=True),
        sorted(range(n), key=lambda i: req[i], reverse=True),
        sorted(range(n), key=lambda i: req[i] / max(size[i], 1e-9), reverse=True),
        sorted(range(n), key=lambda i: req[i] / max(GPU_MEM_SIZE - size[i], 1e-9), reverse=True),
        list(range(n)),
    ]

    def check_any(T):
        best_assign = None
        best_score = float("inf")
        for order in orders:
            a = check(T, order)
            if a is not None:
                s = actual_max_kvpr(a)
                if s < best_score:
                    best_score = s
                    best_assign = a
                if time.time() - start > 5.0:
                    break
        return best_assign, best_score

    # Upper bound: greedy packing by most free memory
    assign = [[] for _ in range(gpu_num)]
    used = [0.0] * gpu_num
    load = [0.0] * gpu_num
    for i in sorted(range(n), key=lambda i: size[i], reverse=True):
        g = min(range(gpu_num), key=lambda x: used[x])
        assign[g].append(i)
        used[g] += size[i]
        load[g] += req[i]
    hi = max(actual_max_kvpr(assign), 1e-9)
    lo = 0.0
    best_assign = assign
    best_score = actual_max_kvpr(assign)

    for _ in range(40):
        if time.time() - start > 5.0:
            break
        mid = (lo + hi) / 2.0
        a, s = check_any(mid)
        if a is not None:
            hi = mid
            if s < best_score:
                best_score = s
                best_assign = a
        else:
            lo = mid
        if hi - lo < 1e-9:
            break

    # Local-search refinement: moves and swaps on index assignment
    loads = [0.0] * gpu_num
    used = [0.0] * gpu_num
    for g in range(gpu_num):
        for i in best_assign[g]:
            loads[g] += req[i]
            used[g] += size[i]

    def kvpr_from(loads, used):
        best = 0.0
        for g in range(gpu_num):
            denom = GPU_MEM_SIZE - used[g]
            k = loads[g] / denom if denom > 1e-12 else float("inf")
            if k > best:
                best = k
        return best

    best = kvpr_from(loads, used)
    for _ in range(300):
        if time.time() - start > 7.0:
            break
        improved = False
        for src in range(gpu_num):
            for mi in range(len(best_assign[src])):
                i = best_assign[src][mi]
                for dst in range(gpu_num):
                    if dst == src:
                        continue
                    # move
                    if used[dst] + size[i] <= GPU_MEM_SIZE:
                        loads[src] -= req[i]; used[src] -= size[i]
                        loads[dst] += req[i]; used[dst] += size[i]
                        s = kvpr_from(loads, used)
                        if s < best - 1e-12:
                            best_assign[src].pop(mi)
                            best_assign[dst].append(i)
                            best = s
                            improved = True
                            break
                        loads[src] += req[i]; used[src] += size[i]
                        loads[dst] -= req[i]; used[dst] -= size[i]
                    # swap
                    done = False
                    for mj in range(len(best_assign[dst])):
                        j = best_assign[dst][mj]
                        if used[dst] - size[j] + size[i] > GPU_MEM_SIZE:
                            continue
                        if used[src] - size[i] + size[j] > GPU_MEM_SIZE:
                            continue
                        loads[src] += req[j] - req[i]
                        loads[dst] += req[i] - req[j]
                        used[src] += size[j] - size[i]
                        used[dst] += size[i] - size[j]
                        s = kvpr_from(loads, used)
                        if s < best - 1e-12:
                            best_assign[src][mi], best_assign[dst][mj] = j, i
                            best = s
                            improved = True
                            done = True
                            break
                        loads[src] -= req[j] - req[i]
                        loads[dst] -= req[i] - req[j]
                        used[src] -= size[j] - size[i]
                        used[dst] -= size[i] - size[j]
                    if done:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    placement = {g: [models[i] for i in best_assign[g]] for g in range(gpu_num)}
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
