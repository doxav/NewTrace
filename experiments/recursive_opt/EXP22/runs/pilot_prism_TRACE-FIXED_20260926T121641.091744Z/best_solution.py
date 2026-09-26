GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def _greedy(order, gpu_num, models):
    """Greedy pass: assign each model (in given order) to the GPU that
    minimizes the resulting KVPR after placement, subject to memory fit."""
    placement = {g: [] for g in range(gpu_num)}
    mem = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num
    for m in order:
        best, best_kvpr = None, float("inf")
        for g in range(gpu_num):
            if m.model_size <= mem[g]:
                kvpr = (load[g] + m.req_rate / m.slo) / (mem[g] - m.model_size)
                if kvpr < best_kvpr:
                    best_kvpr, best = kvpr, g
        if best is None:
            return None
        placement[best].append(m)
        load[best] += m.req_rate / m.slo
        mem[best] -= m.model_size
    max_kvpr = max(load[g] / (GPU_MEM_SIZE - sum(x.model_size for x in placement[g]))
                   for g in range(gpu_num))
    return max_kvpr, placement


def compute_model_placement(gpu_num, models):
    """
    Minimize max KVPR: run greedy placement over several model orderings
    (by load desc/asc, size desc/asc, load-per-size desc) and keep the
    placement with the lowest maximum KVPR.
    """
    """Minimize max KVPR: greedy placement over many orderings (heuristic
    sorts plus seeded random shuffles), keeping the best placement."""
    import random
    rng = random.Random(42)
    orderings = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.model_size),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size),
        sorted(models, key=lambda m: (m.req_rate / m.slo) * m.model_size, reverse=True),
    ] + [rng.sample(models, len(models)) for _ in range(20)]
    best = None
    for order in orderings:
        res = _greedy(order, gpu_num, models)
        if res is not None and (best is None or res[0] < best[0]):
            best = res
    if best is None:
        raise ValueError("Unable to place all models on any GPU.")
    return best[1]


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
