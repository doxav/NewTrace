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

    """Greedy placement minimizing post-assignment KVPR, plus swap refinement."""

    def run(order):
        placement = {g: [] for g in range(gpu_num)}
        load = [0.0] * gpu_num   # sum of req_rate/slo per GPU
        used = [0.0] * gpu_num   # sum of model sizes per GPU
        for m in order:
            best, best_kvpr = None, float("inf")
            for g in range(gpu_num):
                rem = GPU_MEM_SIZE - used[g]
                if m.model_size <= rem and rem > 0:
                    kvpr = (load[g] + m.req_rate / m.slo) / rem
                    if kvpr < best_kvpr:
                        best_kvpr, best = kvpr, g
            if best is None:
                return None
            placement[best].append(m)
            load[best] += m.req_rate / m.slo
            used[best] += m.model_size
        return placement, load, used

    def max_kvpr(load, used):
        return max((l / (GPU_MEM_SIZE - u) if u < GPU_MEM_SIZE else float("inf")
                    for l, u in zip(load, used)), default=0.0)

    # Try a few orderings, keep the best
    best_result = None
    best_score = float("inf")
    orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo / m.model_size, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        models,
        sorted(models, key=lambda m: m.req_rate / m.slo / m.model_size, reverse=True),
    ]
    for order in orders:
        res = run(order)
        if res is None:
            continue
        placement, load, used = res
        score = max_kvpr(load, used)
        if score < best_score:
            best_score, best_result = score, (placement, load, used)

    if best_result is None:
        raise ValueError("Unable to place all models on GPUs")

    # Local search: try moving each model to another GPU if it reduces max KVPR
    placement, load, used = best_result
    improved = True
    while improved:
        improved = False
        for g in range(gpu_num):
            for m in list(placement[g]):
                cur = load[g] / (GPU_MEM_SIZE - used[g]) if used[g] < GPU_MEM_SIZE else float("inf")
                for h in range(gpu_num):
                    if h == g or used[h] + m.model_size > GPU_MEM_SIZE:
                        continue
                    r = m.req_rate / m.slo
                    new_load_g, new_used_g = load[g] - r, used[g] - m.model_size
                    new_load_h, new_used_h = load[h] + r, used[h] + m.model_size
                    kg = new_load_g / (GPU_MEM_SIZE - new_used_g) if new_used_g < GPU_MEM_SIZE else float("inf")
                    kh = new_load_h / (GPU_MEM_SIZE - new_used_h) if new_used_h < GPU_MEM_SIZE else float("inf")
                    others = max((load[i] / (GPU_MEM_SIZE - used[i])
                                  for i in range(gpu_num) if i not in (g, h) and used[i] < GPU_MEM_SIZE),
                                 default=0.0)
                    if max(kg, kh, others) < max_kvpr(load, used) - 1e-12:
                        placement[g].remove(m)
                        placement[h].append(m)
                        load[g], used[g] = new_load_g, new_used_g
                        load[h], used[h] = new_load_h, new_used_h
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Pairwise swap refinement: swap models between GPUs if it lowers max KVPR
    improved = True
    while improved:
        improved = False
        base = max_kvpr(load, used)
        for g in range(gpu_num):
            for m1 in list(placement[g]):
                for h in range(gpu_num):
                    if h <= g:
                        continue
                    for m2 in list(placement[h]):
                        r1, r2 = m1.req_rate / m1.slo, m2.req_rate / m2.slo
                        s1, s2 = m1.model_size, m2.model_size
                        if used[g] - s1 + s2 > GPU_MEM_SIZE or used[h] - s2 + s1 > GPU_MEM_SIZE:
                            continue
                        kg = (load[g] - r1 + r2) / (GPU_MEM_SIZE - (used[g] - s1 + s2))
                        kh = (load[h] - r2 + r1) / (GPU_MEM_SIZE - (used[h] - s2 + s1))
                        others = max((load[i] / (GPU_MEM_SIZE - used[i])
                                      for i in range(gpu_num) if i not in (g, h) and used[i] < GPU_MEM_SIZE),
                                     default=0.0)
                        if max(kg, kh, others) < base - 1e-12:
                            placement[g].remove(m1)
                            placement[h].remove(m2)
                            placement[g].append(m2)
                            placement[h].append(m1)
                            load[g] += r2 - r1
                            load[h] += r1 - r2
                            used[g] += s2 - s1
                            used[h] += s1 - s2
                            base = max(kg, kh, others)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
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
