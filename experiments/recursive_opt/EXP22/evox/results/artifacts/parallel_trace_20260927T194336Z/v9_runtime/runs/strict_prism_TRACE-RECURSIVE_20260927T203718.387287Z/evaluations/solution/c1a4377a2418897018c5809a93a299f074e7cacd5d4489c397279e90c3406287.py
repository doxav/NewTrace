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

    """Multi-start greedy + single-move local search minimizing max KVPR.

    Tries two greedy orderings (size desc, req_rate/slo desc). Each greedy
    assigns each model to the feasible GPU minimizing the resulting KVPR
    ((w + r/s) / (mem - size)); fallback places on the GPU with most free
    memory to guarantee a valid placement. Then improves with single moves
    off the most loaded GPU that lower the global max KVPR. Returns the
    best placement found.
    """
    def solve(order):
        placement = {g: [] for g in range(gpu_num)}
        w = [0.0] * gpu_num   # sum of req_rate/slo per GPU
        mem = [float(GPU_MEM_SIZE)] * gpu_num  # free memory per GPU

        def kvpr(g):
            return w[g] / mem[g] if mem[g] > 0 else float("inf")

        # Greedy: choose GPU minimizing resulting KVPR among feasible ones
        for m in order:
            cands = [g for g in range(gpu_num) if m.model_size <= mem[g]]
            if cands:
                best = min(cands, key=lambda g: (w[g] + m.req_rate / m.slo) / (mem[g] - m.model_size))
            else:
                # Fallback: place on GPU with most free memory (avoids hard failure)
                best = max(range(gpu_num), key=lambda g: mem[g])
            placement[best].append(m)
            w[best] += m.req_rate / m.slo
            mem[best] -= m.model_size

        # Local search: single moves off the most loaded GPU
        for _ in range(500):
            src = max(range(gpu_num), key=kvpr)
            cur = max(kvpr(x) for x in range(gpu_num))
            improved = False
            for m in list(placement[src]):
                r = m.req_rate / m.slo
                for g in range(gpu_num):
                    if g == src or m.model_size > mem[g] or mem[g] - m.model_size <= 1e-9:
                        continue
                    new_max = max((w[src] - r) / (mem[src] + m.model_size),
                                  (w[g] + r) / (mem[g] - m.model_size))
                    if new_max < cur - 1e-12:
                        placement[src].remove(m); placement[g].append(m)
                        w[src] -= r; mem[src] += m.model_size
                        w[g] += r; mem[g] -= m.model_size
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

        return placement, max(kvpr(g) for g in range(gpu_num))

    best_p, best_v = None, float("inf")
    for key in (lambda x: -x.model_size, lambda x: -x.req_rate / x.slo):
        p, v = solve(sorted(models, key=key))
        if v < best_v:
            best_p, best_v = p, v
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
