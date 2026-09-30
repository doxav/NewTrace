GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """
    Multi-start greedy placement (several sort orders) followed by a bounded
    local search (single-model moves and pairwise swaps out of the max-KVPR
    GPU) that minimizes the maximum KVPR across GPUs.
    """

    def greedy(order):
        placement = {g: [] for g in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for model in order:
            best_idx, best_ratio = None, float("inf")
            for g in range(gpu_num):
                if model.model_size <= shared_kv[g]:
                    ratio = load[g] / shared_kv[g]
                    if ratio < best_ratio:
                        best_ratio, best_idx = ratio, g
            if best_idx is None:
                return None
            placement[best_idx].append(model)
            load[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        return placement

    def kvpr_of(gpu_models):
        if not gpu_models:
            return 0.0
        used = sum(m.model_size for m in gpu_models)
        return sum(m.req_rate / m.slo for m in gpu_models) / (GPU_MEM_SIZE - used)

    def max_kvpr(placement):
        return max(kvpr_of(ms) for ms in placement.values())

    def local_search(placement):
        """Move/swap models away from the max-KVPR GPU while max KVPR drops."""
        for _ in range(100):
            kvprs = [kvpr_of(placement[g]) for g in range(gpu_num)]
            src = max(range(gpu_num), key=lambda g: kvprs[g])
            base = max_kvpr(placement)
            improved = False
            # try moving one model out of the most loaded GPU
            for m in list(placement[src]):
                for dst in range(gpu_num):
                    if dst == src:
                        continue
                    if m.model_size <= GPU_MEM_SIZE - sum(x.model_size for x in placement[dst]):
                        placement[src].remove(m)
                        placement[dst].append(m)
                        if max_kvpr(placement) < base - 1e-12:
                            improved = True
                            break
                        placement[dst].remove(m)
                        placement[src].append(m)
                if improved:
                    break
            if improved:
                continue
            # try swapping two models between GPUs
            done = False
            for a in list(placement[src]):
                for b in range(gpu_num):
                    if b == src:
                        continue
                    for c in list(placement[b]):
                        if a.model_size == c.model_size:
                            continue
                        delta = a.model_size - c.model_size
                        if delta > GPU_MEM_SIZE - sum(x.model_size for x in placement[b]):
                            continue
                        if -delta > GPU_MEM_SIZE - sum(x.model_size for x in placement[src]):
                            continue
                        placement[src].remove(a)
                        placement[b].remove(c)
                        placement[src].append(c)
                        placement[b].append(a)
                        if max_kvpr(placement) < base - 1e-12:
                            improved = done = True
                            break
                        placement[b].remove(a)
                        placement[src].remove(c)
                        placement[src].append(a)
                        placement[b].append(c)
                    if done:
                        break
                if done:
                    break
            if not improved:
                break
        return placement

    orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo / m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate, reverse=True),
    ]

    best, best_kv = None, float("inf")
    for order in orders:
        p = greedy(order)
        if p is not None:
            p = local_search(p)
            kv = max_kvpr(p)
            if kv < best_kv:
                best, best_kv = p, kv
    if best is None:
        raise ValueError("Unable to place all models on the given GPUs.")
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
