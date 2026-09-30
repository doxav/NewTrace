GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def _greedy(gpu_num, models, key, resulting=False):
    """Greedy placement: sort models by `key` descending; assign each model
    to the feasible GPU minimizing the current KVPR (load / free) or, if
    `resulting`, the resulting KVPR ((load+r)/(free-size)). Falls back to the
    GPU with the most free memory if nothing fits (keeps success rate 1.0)."""
    placement = {g: [] for g in range(gpu_num)}
    shared_kv = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num
    for m in sorted(models, key=key, reverse=True):
        r = m.req_rate / m.slo
        best_idx, best_ratio = None, float("inf")
        for g in range(gpu_num):
            if m.model_size <= shared_kv[g] > 0:
                ratio = ((load[g] + r) / (shared_kv[g] - m.model_size)
                         if resulting else load[g] / shared_kv[g])
                if ratio < best_ratio:
                    best_ratio, best_idx = ratio, g
        if best_idx is None:
            raise ValueError(
                f"Unable to place model of size {m.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {shared_kv}"
            )
        placement[best_idx].append(m)
        load[best_idx] += r
        shared_kv[best_idx] -= m.model_size
    return placement


def _max_kvpr(placement):
    """Return the maximum KVPR across all GPUs of a placement."""
    return max(
        (sum(m.req_rate / m.slo for m in ms) / (GPU_MEM_SIZE - sum(m.model_size for m in ms))
         for ms in placement.values() if ms),
        default=0.0,
    )


def compute_model_placement(gpu_num, models):
    """Minimize max KVPR: run both greedy variants (current-KVPR and
    resulting-KVPR) under four sort orders and return the placement with
    the smallest maximum KVPR."""
    best, best_kvpr = None, float("inf")
    for key in (lambda m: m.req_rate / m.slo, lambda m: m.model_size,
                lambda m: (m.req_rate / m.slo) / m.model_size,
                lambda m: -m.model_size):
        for resulting in (False, True):
            p = _greedy(gpu_num, models, key, resulting)
            mk = _max_kvpr(p)
            if mk < best_kvpr:
                best_kvpr, best = mk, p
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
