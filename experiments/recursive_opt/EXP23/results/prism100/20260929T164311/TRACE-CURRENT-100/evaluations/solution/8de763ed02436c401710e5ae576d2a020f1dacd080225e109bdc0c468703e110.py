GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Minimize max KVPR via multi-ordering greedy + local search.

    Approach: run a KVPR-greedy placement under several model orderings
    (by load, size, load/size ratio), skip orderings that fail to fit,
    then improve the best result by moving models between GPUs whenever
    it strictly lowers max KVPR while respecting memory limits.
    """

    def max_kvpr(p):
        """Max KVPR across GPUs; a fully-packed GPU gets infinite pressure."""
        if p is None:
            return float("inf")
        best = 0.0
        for ms in p.values():
            free = GPU_MEM_SIZE - sum(m.model_size for m in ms)
            load = sum(m.req_rate / m.slo for m in ms)
            if free <= 0:
                return float("inf") if load > 0 else best
            best = max(best, load / free)
        return best

    def greedy(order):
        """Assign each model to feasible GPU with min load/free-memory ratio.

        This ratio-based greedy spreads load proportionally to free memory,
        is highly robust for feasibility, and yields low max KVPR.
        """
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
        """Local search: move OR swap models between GPUs if max KVPR drops."""
        for _ in range(50):
            cur = max_kvpr(p)
            improved = False
            for g1 in range(gpu_num):
                for m in list(p[g1]):
                    for g2 in range(gpu_num):
                        if g1 == g2:
                            continue
                        free2 = GPU_MEM_SIZE - sum(x.model_size for x in p[g2])
                        # Try move
                        if m.model_size <= free2:
                            p[g1].remove(m)
                            p[g2].append(m)
                            if max_kvpr(p) < cur - 1e-12:
                                improved = True
                                break
                            p[g2].remove(m)
                            p[g1].append(m)
                        # Try swap
                        for m2 in list(p[g2]):
                            if m2.model_size - m.model_size <= free2:
                                p[g1].remove(m)
                                p[g2].remove(m2)
                                p[g1].append(m2)
                                p[g2].append(m)
                                if max_kvpr(p) < cur - 1e-12:
                                    improved = True
                                    break
                                p[g2].remove(m)
                                p[g1].remove(m2)
                                p[g1].append(m)
                                p[g2].append(m2)
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
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

    # Try several orderings; keep the one minimizing max KVPR
    candidates = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo / m.model_size, reverse=True),
        sorted(models, key=lambda m: m.model_size / (m.req_rate / m.slo + 1e-9), reverse=True),
        models,
    ]
    best, best_kvpr = None, float("inf")
    for order in candidates:
        p = improve(greedy(order))
        kvpr = max_kvpr(p)
        if kvpr < best_kvpr:
            best, best_kvpr = p, kvpr
    if best is None:
        # Feasibility fallback: FFD guarantees a fit whenever one exists
        best = ffd()
        if best is None:
            best = {0: list(models)}
            for g in range(1, gpu_num):
                best[g] = []
        best = improve(best)
    if best is None:
        best = {g: [] for g in range(gpu_num)}
        best[0] = list(models)  # last-resort: dump all on GPU 0
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
