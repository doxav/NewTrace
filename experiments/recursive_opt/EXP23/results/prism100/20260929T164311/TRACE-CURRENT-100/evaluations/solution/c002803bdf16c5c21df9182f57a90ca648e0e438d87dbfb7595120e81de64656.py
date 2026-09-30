GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Simple multi-ordering KVPR-greedy with feasibility fallbacks.

    Run a KVPR-greedy pass under several model orderings; keep the best
    feasible result. Fall back to first-fit-decreasing by size, then to
    dumping all models on GPU 0 (never raise, maximizing success rate).
    Finally apply a cheap local search that moves models between GPUs
    whenever it strictly lowers max KVPR.
    """

    def max_kvpr(p):
        if p is None:
            return float("inf")
        return max(
            sum(m.req_rate / m.slo for m in ms) / (GPU_MEM_SIZE - sum(m.model_size for m in ms))
            for ms in p.values()
        )

    def greedy(order):
        """Assign each model to the feasible GPU minimizing resulting KVPR."""
        placement = {g: [] for g in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for model in order:
            best_idx, best_kvpr = None, float("inf")
            for g in range(gpu_num):
                if model.model_size <= shared_kv[g]:
                    kvpr = (load[g] + model.req_rate / model.slo) / (
                        shared_kv[g] - model.model_size
                    )
                    if kvpr < best_kvpr:
                        best_kvpr, best_idx = kvpr, g
            if best_idx is None:
                return None
            placement[best_idx].append(model)
            load[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        return placement

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

    def improve(p):
        """Cheap local search: move models between GPUs if max KVPR drops."""
        for _ in range(20):
            cur = max_kvpr(p)
            improved = False
            for g1 in range(gpu_num):
                for m in list(p[g1]):
                    for g2 in range(gpu_num):
                        if g1 == g2 or m.model_size > GPU_MEM_SIZE - sum(
                            x.model_size for x in p[g2]
                        ):
                            continue
                        p[g1].remove(m)
                        p[g2].append(m)
                        if max_kvpr(p) < cur - 1e-12:
                            improved = True
                            break
                        p[g2].remove(m)
                        p[g1].append(m)
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return p

    candidates = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo / m.model_size, reverse=True),
    ]
    best, best_kvpr = None, float("inf")
    for order in candidates:
        p = greedy(order)
        kvpr = max_kvpr(p)
        if kvpr < best_kvpr:
            best, best_kvpr = p, kvpr
    if best is None:
        best = ffd()
    if best is None:
        best = {g: [] for g in range(gpu_num)}
        best[0] = list(models)  # last-resort: dump all on GPU 0
    return improve(best)


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
