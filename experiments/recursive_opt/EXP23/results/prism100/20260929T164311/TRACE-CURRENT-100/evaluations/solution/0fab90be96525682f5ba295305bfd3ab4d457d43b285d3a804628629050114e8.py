GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Simple greedy placement minimizing max KVPR.

    Sort models by req_rate/slo descending (heaviest load first), then assign
    each model to the feasible GPU with the lowest current load/free-memory
    ratio (a proxy for resulting KVPR). If that ordering fails to fit,
    fall back to size-descending ordering (best-fit for tight memory).
    """

    def max_kvpr(p):
        """Max KVPR across GPUs (exact objective being minimized)."""
        return max(
            sum(m.req_rate / m.slo for m in ms)
            / max(GPU_MEM_SIZE - sum(m.model_size for m in ms), 1e-9)
            for ms in p.values()
        )

    def greedy(order):
        """Assign each model to the GPU minimizing load/free-memory ratio."""
        placement = {g: [] for g in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for model in order:
            best_idx, best_ratio = None, float("inf")
            for g in range(gpu_num):
                if model.model_size <= shared_kv[g] and shared_kv[g] > 0:
                    ratio = load[g] / shared_kv[g]
                    if ratio < best_ratio:
                        best_ratio, best_idx = ratio, g
            if best_idx is None:
                return None
            placement[best_idx].append(model)
            load[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        return placement

    def improve(p):
        """Move models off the max-KVPR GPU when it strictly lowers max KVPR."""
        for _ in range(50):
            src = max(p, key=lambda g: max_kvpr({g: p[g]}))
            cur = max_kvpr(p)
            for m in list(p[src]):
                for g in range(gpu_num):
                    if g == src:
                        continue
                    if m.model_size > GPU_MEM_SIZE - sum(x.model_size for x in p[g]):
                        continue
                    p[src].remove(m)
                    p[g].append(m)
                    if max_kvpr(p) < cur - 1e-12:
                        break
                    p[g].remove(m)
                    p[src].append(m)
                else:
                    continue
                break
            else:
                break
        return p

    def ffd():
        """Fallback: first-fit-decreasing by size for feasibility."""
        placement = {g: [] for g in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE] * gpu_num
        for model in sorted(models, key=lambda m: m.model_size, reverse=True):
            for g in range(gpu_num):
                if model.model_size <= shared_kv[g]:
                    placement[g].append(model)
                    shared_kv[g] -= model.model_size
                    break
            else:
                return None
        return placement

    # Primary: heaviest load first; fallbacks: largest model first, then FFD
    p = greedy(sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True))
    if p is None:
        p = greedy(sorted(models, key=lambda m: m.model_size, reverse=True))
    if p is None:
        p = ffd()
    if p is None:
        raise ValueError("Unable to place all models on GPUs")
    return improve(p)


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
