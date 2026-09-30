GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Approach: run a KVPR-greedy placement under several deterministic orderings
    plus randomized restarts, improve the best result with a (max, sum) local
    search, and fall back to first-fit-decreasing for feasibility. An epsilon
    floor on free memory avoids division-by-zero on exact fits.
    """

    def greedy(order):
        """Greedy: assign each model to GPU minimizing resulting KVPR."""
        placement = {g: [] for g in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for model in order:
            best_idx, best_kvpr = None, float("inf")
            for g in range(gpu_num):
                if model.model_size <= shared_kv[g]:
                    kvpr = (load[g] + model.req_rate / model.slo) / max(
                        shared_kv[g] - model.model_size, 1e-9
                    )
                    if kvpr < best_kvpr:
                        best_kvpr, best_idx = kvpr, g
            if best_idx is None:
                return None
            placement[best_idx].append(model)
            load[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        return placement

    def first_fit():
        """Fallback: best-fit-decreasing by size, maximizing feasibility."""
        placement = {g: [] for g in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE] * gpu_num
        for model in sorted(models, key=lambda m: m.model_size, reverse=True):
            fits = [g for g in range(gpu_num) if model.model_size <= shared_kv[g]]
            if not fits:
                return None
            g = min(fits, key=lambda g: shared_kv[g])
            placement[g].append(model)
            shared_kv[g] -= model.model_size
        return placement

    def score(p):
        """(max KVPR, sum of KVPRs): tie-break on total pressure to escape plateaus."""
        kvs = [
            sum(m.req_rate / m.slo for m in ms)
            / max(GPU_MEM_SIZE - sum(m.model_size for m in ms), 1e-9)
            for ms in p.values()
        ]
        return (max(kvs), sum(kvs))

    def max_kvpr(placement):
        if placement is None:
            return float("inf")
        return max(
            sum(m.req_rate / m.slo for m in ms)
            / max(GPU_MEM_SIZE - sum(m.model_size for m in ms), 1e-9)
            for ms in placement.values()
        )

    def improve(p):
        """Local search: move OR swap models if (max, sum) KVPR score drops."""
        if p is None:
            return None
        for _ in range(30):
            cur = score(p)
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
                            if score(p) < tuple(c - 1e-12 for c in cur):
                                improved = True
                                break
                            p[g2].remove(m)
                            p[g1].append(m)
                        # Try swap with each model on g2
                        for m2 in list(p[g2]):
                            if m2.model_size - m.model_size > free2:
                                continue
                            p[g1].remove(m)
                            p[g2].remove(m2)
                            p[g1].append(m2)
                            p[g2].append(m)
                            if score(p) < tuple(c - 1e-12 for c in cur):
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

    """Try deterministic orderings plus randomized restarts; keep the best."""
    import random

    rng = random.Random(42)
    candidates = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo / m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo / m.model_size),
    ]
    # Randomized restarts to diversify greedy starting points
    for _ in range(8):
        order = models[:]
        rng.shuffle(order)
        candidates.append(order)
    best = None
    best_kvpr = float("inf")
    for order in candidates:
        p = improve(greedy(order))
        kvpr = max_kvpr(p)
        if kvpr < best_kvpr:
            best, best_kvpr = p, kvpr
    if best is None:
        best = first_fit()
        best = improve(best)
    if best is None:
        raise ValueError("Unable to place all models on GPUs")
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
