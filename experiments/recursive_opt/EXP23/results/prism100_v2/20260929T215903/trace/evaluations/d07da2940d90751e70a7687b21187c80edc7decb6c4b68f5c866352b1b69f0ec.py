GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize the maximum KVPR via binary search on the threshold T.

    A placement has max KVPR <= T iff for every GPU:
        sum(req_rate/slo) <= T * (80 - sum(model_size))
    which rearranges to: sum(model_size + (req_rate/slo)/T) <= 80.
    So feasibility for a given T is a bin-packing check with effective
    item sizes size + load/T, solved with First-Fit-Decreasing.
    """

    if not models:
        return {gpu_id: [] for gpu_id in range(gpu_num)}

    req = [m.req_rate / m.slo for m in models]
    sizes = [m.model_size for m in models]

    def pack(T, order_key):
        """FFD with effective sizes size + req/T. Returns placement or None."""
        placement = {g: [] for g in range(gpu_num)}
        used = [0.0] * gpu_num
        idx = sorted(range(len(models)), key=order_key)
        for i in idx:
            eff = sizes[i] + req[i] / T
            if eff > GPU_MEM_SIZE:
                return None  # single model cannot fit under this T
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

    def kvpr_of(placement):
        best = 0.0
        for gpu_models in placement.values():
            denom = GPU_MEM_SIZE - sum(m.model_size for m in gpu_models)
            if denom <= 0:
                return float("inf")
            best = max(best, sum(m.req_rate / m.slo for m in gpu_models) / denom)
        return best

    # Lower bound: total load spread over all free memory.
    total_load = sum(req)
    total_size = sum(sizes)
    lo = total_load / max(GPU_MEM_SIZE * gpu_num - total_size, 1e-9)
    hi = max(total_load / max(GPU_MEM_SIZE - max(sizes), 1e-9), lo) * 2 + 1.0

    best = None
    best_score = float("inf")
    order_keys = [
        lambda i: -(sizes[i] + req[i]),          # big effective size first
        lambda i: -(sizes[i]),                   # big physical size first
        lambda i: -(req[i]),                     # big load first
        lambda i: -(req[i] / max(sizes[i], 1e-9)),  # dense first
    ]

    # Binary search on T; keep the best feasible placement found.
    for _ in range(60):
        mid = (lo + hi) / 2.0
        found = None
        for key in order_keys:
            p = pack(mid, key)
            if p is not None:
                found = p
                break
        if found is not None:
            score = kvpr_of(found)
            if score < best_score:
                best_score = score
                best = found
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-4:
            break

    # Polish: local search (moves + swaps) to reduce max KVPR.
    if best is not None:
        for _ in range(100):
            loads = [0.0] * gpu_num
            used = [0.0] * gpu_num
            for g in range(gpu_num):
                for m in best[g]:
                    loads[g] += m.req_rate / m.slo
                    used[g] += m.model_size
            cur = kvpr_of(best)
            gmax = max(range(gpu_num), key=lambda x: loads[x] / max(GPU_MEM_SIZE - used[x], 1e-9))
            improved = False
            best_gain = (None, cur)
            # try moves out of the max-KVPR GPU
            for m in list(best[gmax]):
                i_req = m.req_rate / m.slo
                i_size = m.model_size
                for dst in range(gpu_num):
                    if dst == gmax or used[dst] + i_size > GPU_MEM_SIZE:
                        continue
                    new_max = cur
                    for g in range(gpu_num):
                        if g == gmax:
                            v = (loads[g] - i_req) / (GPU_MEM_SIZE - used[g] + i_size)
                        elif g == dst:
                            v = (loads[g] + i_req) / (GPU_MEM_SIZE - used[dst] - i_size)
                        else:
                            v = loads[g] / max(GPU_MEM_SIZE - used[g], 1e-9)
                        if v > new_max:
                            new_max = v
                    if new_max < best_gain[1] - 1e-12:
                        best_gain = ((gmax, m, dst, None), new_max)
            # try swaps between gmax and others
            for m1 in list(best[gmax]):
                for dst in range(gpu_num):
                    if dst == gmax:
                        continue
                    for m2 in list(best[dst]):
                        if used[dst] - m2.model_size + m1.model_size > GPU_MEM_SIZE:
                            continue
                        if used[gmax] - m1.model_size + m2.model_size > GPU_MEM_SIZE:
                            continue
                        new_max = cur
                        for g in range(gpu_num):
                            if g == gmax:
                                v = (loads[g] - m1.req_rate / m1.slo + m2.req_rate / m2.slo) / (
                                    GPU_MEM_SIZE - used[g] + m1.model_size - m2.model_size)
                            elif g == dst:
                                v = (loads[g] + m1.req_rate / m1.slo - m2.req_rate / m2.slo) / (
                                    GPU_MEM_SIZE - used[dst] - m1.model_size + m2.model_size)
                            else:
                                v = loads[g] / max(GPU_MEM_SIZE - used[g], 1e-9)
                            if v > new_max:
                                new_max = v
                        if new_max < best_gain[1] - 1e-12:
                            best_gain = ((gmax, m1, dst, m2), new_max)
            if best_gain[0] is None:
                break
            gmax, m1, dst, m2 = best_gain[0]
            best[gmax].remove(m1)
            best[dst].append(m1)
            if m2 is not None:
                best[dst].remove(m2)
                best[gmax].append(m2)

    if best is None:
        # Fallback: pack by physical size, place any unplaceable model
        # on the GPU with the most free memory.
        placement = {g: [] for g in range(gpu_num)}
        used = [0.0] * gpu_num
        for m in sorted(models, key=lambda m: m.model_size, reverse=True):
            g = min(range(gpu_num), key=lambda x: used[x])
            placement[g].append(m)
            used[g] += m.model_size
        best = placement
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
