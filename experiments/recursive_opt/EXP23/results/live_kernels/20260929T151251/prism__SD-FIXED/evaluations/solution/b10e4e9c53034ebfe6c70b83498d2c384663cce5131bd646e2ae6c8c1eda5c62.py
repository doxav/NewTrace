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

    """Greedy placement (multiple orderings) + local search (moves + swaps)."""
    w = [m.req_rate / m.slo for m in models]
    n = len(models)

    def max_kvpr(load, used):
        return max((load[g] / (GPU_MEM_SIZE - used[g]) if used[g] < GPU_MEM_SIZE else float("inf"))
                   for g in range(gpu_num))

    def greedy(order):
        placement = {g: [] for g in range(gpu_num)}
        load = [0.0] * gpu_num
        used = [0.0] * gpu_num
        for i in order:
            m = models[i]
            best_g, best_kvpr = None, float("inf")
            for g in range(gpu_num):
                if used[g] + m.model_size <= GPU_MEM_SIZE:
                    kvpr = (load[g] + w[i]) / (GPU_MEM_SIZE - used[g] - m.model_size)
                    if kvpr < best_kvpr:
                        best_kvpr, best_g = kvpr, g
            if best_g is None:
                # Fallback: place on GPU with most free space (last resort)
                best_g = max(range(gpu_num), key=lambda g: GPU_MEM_SIZE - used[g])
            placement[best_g].append(m)
            load[best_g] += w[i]
            used[best_g] += m.model_size
        return placement, load, used

    def local_search(placement, load, used):
        cur = max_kvpr(load, used)
        improved = True
        while improved:
            improved = False
            # Single-model moves
            for g in range(gpu_num):
                for m in list(placement[g]):
                    wi = m.req_rate / m.slo
                    for h in range(gpu_num):
                        if h == g or used[h] + m.model_size > GPU_MEM_SIZE:
                            continue
                        nl, nu = load[:], used[:]
                        nl[g] -= wi; nu[g] -= m.model_size
                        nl[h] += wi; nu[h] += m.model_size
                        nm = max_kvpr(nl, nu)
                        if nm < cur - 1e-12:
                            placement[g].remove(m); placement[h].append(m)
                            load, used, cur = nl, nu, nm
                            improved = True
                            break
                    if improved: break
                if improved: break
            if improved: continue
            # Pairwise swaps between GPUs
            for g in range(gpu_num):
                for a in list(placement[g]):
                    wa = a.req_rate / a.slo
                    for h in range(gpu_num):
                        if h == g: continue
                        for b in list(placement[h]):
                            wb = b.req_rate / b.slo
                            nu_g = used[g] - a.model_size + b.model_size
                            nu_h = used[h] - b.model_size + a.model_size
                            if nu_g > GPU_MEM_SIZE or nu_h > GPU_MEM_SIZE:
                                continue
                            nl, nu = load[:], used[:]
                            nl[g] += wb - wa; nl[h] += wa - wb
                            nu[g], nu[h] = nu_g, nu_h
                            nm = max_kvpr(nl, nu)
                            if nm < cur - 1e-12:
                                placement[g].remove(a); placement[h].remove(b)
                                placement[g].append(b); placement[h].append(a)
                                load, used, cur = nl, nu, nm
                                improved = True
                                break
                        if improved: break
                    if improved: break
                if improved: break
        return placement, load, used, cur

    # Try multiple greedy orderings, keep the best after local search
    orders = [
        sorted(range(n), key=lambda i: w[i], reverse=True),
        sorted(range(n), key=lambda i: w[i] / models[i].model_size, reverse=True),
        sorted(range(n), key=lambda i: models[i].model_size, reverse=True),
    ]
    best = None
    for order in orders:
        placement, load, used = greedy(order)
        placement, load, used, cur = local_search(placement, load, used)
        if best is None or cur < best[3]:
            best = (placement, load, used, cur)
    return best[0]


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
