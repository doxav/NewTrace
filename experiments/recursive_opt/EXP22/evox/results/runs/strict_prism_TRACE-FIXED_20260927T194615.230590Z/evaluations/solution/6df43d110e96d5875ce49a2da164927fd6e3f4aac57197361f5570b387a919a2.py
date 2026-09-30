GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def _greedy(gpu_num, models, key):
    """Greedy placement: process models in `key` order (descending); assign each
    model to the feasible GPU with the lowest current KVPR (load / free memory).
    Raises ValueError if a model cannot fit on any GPU (no overcommitment)."""
    placement = {g: [] for g in range(gpu_num)}
    free = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num
    for m in sorted(models, key=key, reverse=True):
        r = m.req_rate / m.slo
        best, best_ratio = None, float("inf")
        for g in range(gpu_num):
            if m.model_size <= free[g] and free[g] > 0:
                ratio = load[g] / free[g]
                if ratio < best_ratio:
                    best_ratio, best = ratio, g
        if best is None:
            raise ValueError(f"Model of size {m.model_size} GB cannot be placed.")
        placement[best].append(m)
        load[best] += r
        free[best] -= m.model_size
    return placement


def compute_model_placement(gpu_num, models):
    """Minimize max KVPR: run the current-KVPR greedy heuristic under several
    sort keys and return the feasible placement with the smallest max KVPR."""
    best, best_kvpr = None, float("inf")
    for key in (
        lambda m: m.req_rate / m.slo,
        lambda m: m.model_size,
        lambda m: (m.req_rate / m.slo) / m.model_size,
    ):
        try:
            p = _greedy(gpu_num, models, key)
        except ValueError:
            continue
        mk = max(
            sum(m.req_rate / m.slo for m in ms) / (GPU_MEM_SIZE - sum(m.model_size for m in ms))
            for ms in p.values()
        ) if any(p.values()) else 0.0
        if mk < best_kvpr:
            best_kvpr, best = mk, p
    if best is None:
        raise ValueError("No feasible placement found for the given models.")
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
