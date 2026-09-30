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

    def greedy(order):
        """Greedy placement: each model goes to the GPU minimizing resulting KVPR."""
        placement = {g: [] for g in range(gpu_num)}
        free_mem = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num  # sum of req_rate/slo per GPU
        for m in order:
            best, best_kvpr = None, float("inf")
            for g in range(gpu_num):
                if m.model_size <= free_mem[g]:
                    new_kvpr = (load[g] + m.req_rate / m.slo) / (free_mem[g] - m.model_size)
                    if new_kvpr < best_kvpr:
                        best_kvpr, best = new_kvpr, g
            if best is None:
                return None  # infeasible for this ordering
            placement[best].append(m)
            load[best] += m.req_rate / m.slo
            free_mem[best] -= m.model_size
        return placement

    def max_kvpr(placement):
        """Compute max KVPR across GPUs for a placement."""
        worst = 0.0
        for g in range(gpu_num):
            mem = GPU_MEM_SIZE - sum(m.model_size for m in placement[g])
            load = sum(m.req_rate / m.slo for m in placement[g])
            worst = max(worst, load / mem if mem > 0 else float("inf"))
        return worst

    # Try multiple orderings (deterministic + randomized restarts); keep the best
    import random
    rng = random.Random(42)
    orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True),
        sorted(models, key=lambda m: m.model_size / (m.req_rate / m.slo), reverse=True),
    ]
    base = list(models)
    for _ in range(30):
        shuffled = base[:]
        rng.shuffle(shuffled)
        orders.append(shuffled)

    best = None
    best_score = float("inf")
    for order in orders:
        p = greedy(order)
        if p is not None:
            s = max_kvpr(p)
            if s < best_score:
                best_score, best = s, p
    if best is None:
        raise ValueError("Unable to place all models on the GPUs")
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
