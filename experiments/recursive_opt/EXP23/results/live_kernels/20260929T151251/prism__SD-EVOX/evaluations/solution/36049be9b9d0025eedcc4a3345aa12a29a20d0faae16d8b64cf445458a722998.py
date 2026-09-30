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

    """Greedy placement: each model goes to the feasible GPU with the lowest
    current KVPR (load / free memory). Several orderings are tried, then a
    local-search phase moves models between GPUs whenever the move strictly
    reduces the maximum KVPR."""

    def run(order):
        placement = {g: [] for g in range(gpu_num)}
        load = [0.0] * gpu_num   # sum of req_rate/slo per GPU
        used = [0.0] * gpu_num   # sum of model sizes per GPU
        for m in order:
            best, best_kvpr = None, float("inf")
            for g in range(gpu_num):
                rem = GPU_MEM_SIZE - used[g]
                if m.model_size <= rem and rem > 0:
                    kvpr = load[g] / rem
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
    best_result, best_score = None, float("inf")
    orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo / m.model_size, reverse=True),
        models,
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

    # Local search: move a model to another GPU if it reduces max KVPR
    placement, load, used = best_result
    improved = True
    while improved:
        improved = False
        for g in range(gpu_num):
            for m in list(placement[g]):
                r, s = m.req_rate / m.slo, m.model_size
                for h in range(gpu_num):
                    if h == g or used[h] + s > GPU_MEM_SIZE:
                        continue
                    ng, nh = used[g] - s, used[h] + s
                    kg = (load[g] - r) / (GPU_MEM_SIZE - ng) if ng < GPU_MEM_SIZE else float("inf")
                    kh = (load[h] + r) / (GPU_MEM_SIZE - nh) if nh < GPU_MEM_SIZE else float("inf")
                    others = max((load[i] / (GPU_MEM_SIZE - used[i])
                                  for i in range(gpu_num) if i not in (g, h) and used[i] < GPU_MEM_SIZE),
                                 default=0.0)
                    if max(kg, kh, others) < max_kvpr(load, used) - 1e-12:
                        placement[g].remove(m)
                        placement[h].append(m)
                        load[g], used[g] = load[g] - r, ng
                        load[h], used[h] = load[h] + r, nh
                        improved = True
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
