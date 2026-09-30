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

    """Multi-start greedy + local search minimizing max KVPR.

    Tries several greedy orderings (by size desc, by req_rate/slo desc, by
    size asc). Each greedy assigns a model to the feasible GPU minimizing the
    resulting KVPR; fallback places it on the GPU with most free memory.
    Then runs local search (single moves + pairwise swaps) on each start and
    returns the placement with the lowest max KVPR.
    """
    def run(order):
        placement = {g: [] for g in range(gpu_num)}
        w = [0.0] * gpu_num   # sum of req_rate/slo per GPU
        mem = [float(GPU_MEM_SIZE)] * gpu_num  # free memory per GPU

        def kvpr(g):
            return w[g] / mem[g] if mem[g] > 0 else float("inf")

        for m in order:
            cands = [g for g in range(gpu_num) if m.model_size <= mem[g]]
            if cands:
                best = min(cands, key=lambda g: (w[g] + m.req_rate / m.slo) / (mem[g] - m.model_size))
            else:
                best = max(range(gpu_num), key=lambda g: mem[g])
            placement[best].append(m)
            w[best] += m.req_rate / m.slo
            mem[best] -= m.model_size

        # Repair: fix any GPU whose memory went negative by moving its
        # smallest model to the GPU with the most free memory where it fits.
        for g in range(gpu_num):
            while mem[g] < -1e-9 and placement[g]:
                fits = [m for m in placement[g]
                        if any(h != g and m.model_size <= mem[h] + 1e-12
                               for h in range(gpu_num))]
                if not fits:
                    break
                m = min(fits, key=lambda x: x.model_size)
                tgt = max((h for h in range(gpu_num)
                           if h != g and m.model_size <= mem[h] + 1e-12),
                          key=lambda h: mem[h])
                do_move(m, g, tgt)

        def do_move(m, src, g):
            r = m.req_rate / m.slo
            placement[src].remove(m); placement[g].append(m)
            w[src] -= r; mem[src] += m.model_size
            w[g] += r; mem[g] -= m.model_size

        for _ in range(300):
            src = max(range(gpu_num), key=kvpr)
            cur = max(kvpr(x) for x in range(gpu_num))
            improved = False
            for m in list(placement[src]):
                r = m.req_rate / m.slo
                for g in range(gpu_num):
                    if g == src or m.model_size > mem[g]:
                        continue
                    new_max = max((w[src] - r) / (mem[src] + m.model_size),
                                  (w[g] + r) / (mem[g] - m.model_size))
                    if new_max < cur - 1e-12:
                        do_move(m, src, g)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                for m in list(placement[src]):
                    r1 = m.req_rate / m.slo
                    for g in range(gpu_num):
                        if g == src:
                            continue
                        for m2 in list(placement[g]):
                            r2 = m2.req_rate / m.slo
                            if m.model_size > mem[g] + m2.model_size or m2.model_size > mem[src] + m.model_size:
                                continue
                            new_src = (w[src] - r1 + r2) / (mem[src] + m.model_size - m2.model_size)
                            new_g = (w[g] - r2 + r1) / (mem[g] + m2.model_size - m.model_size)
                            if max(new_src, new_g) < cur - 1e-12:
                                do_move(m, src, g)
                                do_move(m2, g, src)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                break
        feasible = all(mg >= -1e-9 for mg in mem)
        return placement, max(kvpr(g) for g in range(gpu_num)), feasible

    # Prefer feasible placements; among equals, pick lowest max KVPR.
    best_p, best_v, best_f = None, float("inf"), False
    for key in (lambda x: -x.model_size,
                lambda x: -x.req_rate / x.slo,
                lambda x: x.model_size):
        p, v, f = run(sorted(models, key=key))
        if (f and not best_f) or (f == best_f and v < best_v):
            best_p, best_v, best_f = p, v, f
    return best_p


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
