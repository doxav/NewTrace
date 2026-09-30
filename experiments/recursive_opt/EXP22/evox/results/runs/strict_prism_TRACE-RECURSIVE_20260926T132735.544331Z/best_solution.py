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

    # Greedy KVPR-minimizing placement: sort by load density r_j / s_j descending,
    # assign each model to the feasible GPU with lowest current KVPR.
    sorted_models = sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True)

    # 2) Initialize per-GPU states
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]  # remaining memory per GPU
    weighted_req_rate = [0.0 for _ in range(gpu_num)]  # sum of r_j / s_j per GPU

    # 3) Assign each model to the GPU that minimizes current KVPR while fitting in memory
    for model in sorted_models:
        best_idx = None
        best_ratio = float("inf")

        for gpu_id in range(gpu_num):
            if model.model_size <= shared_kv[gpu_id] and shared_kv[gpu_id] > 0:
                current_ratio = weighted_req_rate[gpu_id] / shared_kv[gpu_id]
                if current_ratio < best_ratio:
                    best_ratio = current_ratio
                    best_idx = gpu_id

        # Fallback: if no GPU can fit, overcommit to the GPU with most free memory
        if best_idx is None:
            best_idx = max(range(gpu_num), key=lambda g: shared_kv[g])

        placement[best_idx].append(model)
        weighted_req_rate[best_idx] += model.req_rate / model.slo
        shared_kv[best_idx] -= model.model_size

    # --- Local search: single-model moves and pairwise swaps to reduce max KVPR ---
    def kvpr(g):
        return weighted_req_rate[g] / shared_kv[g] if shared_kv[g] > 0 else float("inf")

    def apply(src, dst, m_src, m_dst=None):
        ws, ss = m_src.req_rate / m_src.slo, m_src.model_size
        placement[src].remove(m_src)
        placement[dst].append(m_src)
        weighted_req_rate[src] -= ws
        weighted_req_rate[dst] += ws
        shared_kv[src] += ss
        shared_kv[dst] -= ss
        if m_dst is not None:
            wd, sd = m_dst.req_rate / m_dst.slo, m_dst.model_size
            placement[dst].remove(m_dst)
            placement[src].append(m_dst)
            weighted_req_rate[dst] -= wd
            weighted_req_rate[src] += wd
            shared_kv[dst] += sd
            shared_kv[src] -= sd

    def objective():
        # Lexicographic objective: minimize (max KVPR, sum of squared KVPRs).
        # The second term breaks ties among GPUs at the max, balancing load.
        ks = [kvpr(g) for g in range(gpu_num)]
        return (max(ks), sum(k * k for k in ks))

    def max_after(src, dst, w, s, w2=0.0, s2=0.0):
        vals = [(weighted_req_rate[g], shared_kv[g]) for g in range(gpu_num)]
        vals[src] = (vals[src][0] - w + w2, vals[src][1] + s - s2)
        vals[dst] = (vals[dst][0] + w - w2, vals[dst][1] - s + s2)
        ks = [r / m if m > 0 else float("inf") for r, m in vals]
        return (max(ks), sum(k * k for k in ks))

    base = objective()
    improved = True
    while improved:
        improved = False
        for src in range(gpu_num):
            for m_src in list(placement[src]):
                w, s = m_src.req_rate / m_src.slo, m_src.model_size
                for dst in range(gpu_num):
                    if dst == src:
                        continue
                    # Single-model move
                    if s <= shared_kv[dst] and max_after(src, dst, w, s) < base:
                        apply(src, dst, m_src)
                        base = objective()
                        improved = True
                        break
                    # Pairwise swap with a model on dst
                    for m_dst in list(placement[dst]):
                        sd = m_dst.model_size
                        if sd - s > shared_kv[dst] or s - sd > shared_kv[src]:
                            continue
                        wd = m_dst.req_rate / m_dst.slo
                        if max_after(src, dst, w, s, wd, sd) < base:
                            apply(src, dst, m_src, m_dst)
                            base = objective()
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
