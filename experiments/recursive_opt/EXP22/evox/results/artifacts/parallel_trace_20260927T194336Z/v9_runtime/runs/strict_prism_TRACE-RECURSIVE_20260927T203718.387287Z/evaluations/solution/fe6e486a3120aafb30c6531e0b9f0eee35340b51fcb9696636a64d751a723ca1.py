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

    """Greedy + local search placement minimizing max KVPR.

    Greedy: sort by req_rate/slo desc, assign each model to the feasible GPU
    minimizing the resulting KVPR; if no GPU fits, raise (never overcommit).
    Then local search with single moves and pairwise swaps that lower the
    global max KVPR, always keeping memory feasibility.
    """
    placement = {g: [] for g in range(gpu_num)}
    w = [0.0] * gpu_num   # sum of req_rate/slo per GPU
    mem = [float(GPU_MEM_SIZE)] * gpu_num  # free memory per GPU

    def kvpr(g):
        return w[g] / mem[g] if mem[g] > 0 else float("inf")

    # Greedy: highest load ratio first, choose GPU minimizing resulting KVPR
    for m in sorted(models, key=lambda x: -x.req_rate / x.slo):
        cands = [g for g in range(gpu_num) if m.model_size <= mem[g]]
        if not cands:
            raise ValueError(f"Model of size {m.model_size} GB does not fit on any GPU")
        best = min(cands, key=lambda g: (w[g] + m.req_rate / m.slo) / (mem[g] - m.model_size))
        placement[best].append(m)
        w[best] += m.req_rate / m.slo
        mem[best] -= m.model_size

    def do_move(m, src, g):
        r = m.req_rate / m.slo
        placement[src].remove(m); placement[g].append(m)
        w[src] -= r; mem[src] += m.model_size
        w[g] += r; mem[g] -= m.model_size

    # Local search: single moves, then pairwise swaps off the most loaded GPU
    for _ in range(200):
        src = max(range(gpu_num), key=kvpr)
        cur = max(kvpr(x) for x in range(gpu_num))
        improved = False
        # Phase 1: single moves that lower the max KVPR of the two GPUs
        for m in list(placement[src]):
            r = m.req_rate / m.slo
            for g in range(gpu_num):
                if g == src or m.model_size > mem[g]:
                    continue
                if max((w[src] - r) / (mem[src] + m.model_size),
                       (w[g] + r) / (mem[g] - m.model_size)) < cur - 1e-12:
                    do_move(m, src, g)
                    improved = True
                    break
            if improved:
                break
        # Phase 2: pairwise swaps between src and other GPUs
        if not improved:
            for m in list(placement[src]):
                r1 = m.req_rate / m.slo
                for g in range(gpu_num):
                    if g == src:
                        continue
                    for m2 in list(placement[g]):
                        r2 = m2.req_rate / m.slo
                        if m.model_size > mem[g] + m2.model_size or \
                           m2.model_size > mem[src] + m.model_size:
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
