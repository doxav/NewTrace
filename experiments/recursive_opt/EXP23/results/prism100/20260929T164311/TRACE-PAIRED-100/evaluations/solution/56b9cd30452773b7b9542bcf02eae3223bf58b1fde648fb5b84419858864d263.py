GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def _kvprs(placement):
    """Per-GPU KVPR list (only non-empty GPUs)."""
    return [
        sum(m.req_rate / m.slo for m in ms) / (GPU_MEM_SIZE - sum(m.model_size for m in ms))
        for ms in placement.values() if ms
    ]


def _refine(placement, gpu_num, rounds=60):
    """Local search: repeatedly move one model from the max-KVPR GPU to
    another GPU (or swap two models across GPUs) if it lowers max KVPR,
    without violating memory limits."""
    placement = {g: list(ms) for g, ms in placement.items()}
    mem = {g: GPU_MEM_SIZE - sum(m.model_size for m in ms) for g, ms in placement.items()}
    for _ in range(rounds):
        if not any(placement.values()):
            break
        cur = max(_kvprs(placement))
        src = max(range(gpu_num), key=lambda g: (
            (sum(m.req_rate / m.slo for m in placement[g]) /
             (GPU_MEM_SIZE - sum(m.model_size for m in placement[g])))
            if placement[g] else -1.0))
        if not placement[src]:
            break
        improved = False
        # Try single-model moves out of src.
        for m in list(placement[src]):
            for g in range(gpu_num):
                if g == src or m.model_size > mem[g]:
                    continue
                placement[src].remove(m)
                placement[g].append(m)
                mem[src] += m.model_size
                mem[g] -= m.model_size
                nk = max(_kvprs(placement)) if any(placement.values()) else 0.0
                if nk < cur - 1e-12:
                    cur, improved = nk, True
                    break
                placement[g].remove(m)
                placement[src].append(m)
                mem[g] += m.model_size
                mem[src] -= m.model_size
            if improved:
                break
        # Try pairwise swaps between src and other GPUs.
        if not improved:
            for a in list(placement[src]):
                for g in range(gpu_num):
                    if g == src:
                        continue
                    for b in list(placement[g]):
                        if a.model_size - b.model_size > mem[g]:
                            continue
                        if b.model_size - a.model_size > mem[src]:
                            continue
                        placement[src].remove(a)
                        placement[g].remove(b)
                        placement[src].append(b)
                        placement[g].append(a)
                        mem[src] += a.model_size - b.model_size
                        mem[g] += b.model_size - a.model_size
                        nk = max(_kvprs(placement)) if any(placement.values()) else 0.0
                        if nk < cur - 1e-12:
                            cur, improved = nk, True
                            break
                        placement[g].remove(a)
                        placement[src].remove(b)
                        placement[src].append(a)
                        placement[g].append(b)
                        mem[src] += b.model_size - a.model_size
                        mem[g] += a.model_size - b.model_size
                    if improved:
                        break
                if improved:
                    break
        if not improved:
            break
    return placement


def _greedy(gpu_num, models, key, reverse):
    """Greedy pass: sort by key, assign each model to the GPU minimizing
    the resulting KVPR (w + r)/(mem - s). Returns placement or None."""
    placement = {g: [] for g in range(gpu_num)}
    mem = [GPU_MEM_SIZE] * gpu_num
    w = [0.0] * gpu_num
    for m in sorted(models, key=key, reverse=reverse):
        r = m.req_rate / m.slo
        best, best_ratio = None, float("inf")
        for g in range(gpu_num):
            if m.model_size <= mem[g]:
                ratio = (w[g] + r) / (mem[g] - m.model_size)
                if ratio < best_ratio:
                    best, best_ratio = g, ratio
        if best is None:
            return None
        placement[best].append(m)
        w[best] += r
        mem[best] -= m.model_size
    return placement


def compute_model_placement(gpu_num, models):
    """
    Minimize max KVPR via greedy placement evaluating post-assignment KVPR
    (w + r)/(mem - s) per GPU, trying multiple sort orders and keeping best.
    """
    best = None
    best_kvpr = float("inf")
    for key, reverse in [
        (lambda m: m.req_rate / m.slo, True),
        (lambda m: m.req_rate / m.slo, False),
        (lambda m: m.model_size, True),
        (lambda m: m.req_rate, True),
    ]:
        p = _greedy(gpu_num, models, key, reverse)
        if p is None:
            continue
        kvpr = max(
            (sum(m.req_rate / m.slo for m in ms) / (GPU_MEM_SIZE - sum(m.model_size for m in ms)))
            for ms in p.values()
            if ms
        ) if any(p.values()) else 0.0
        if kvpr < best_kvpr:
            best_kvpr, best = kvpr, p
    if best is None:
        # Feasibility fallback: size-descending, lowest current KVPR ratio.
        best = {g: [] for g in range(gpu_num)}
        mem = [GPU_MEM_SIZE] * gpu_num
        w = [0.0] * gpu_num
        for m in sorted(models, key=lambda m: m.model_size, reverse=True):
            g_best, g_ratio = None, float("inf")
            for g in range(gpu_num):
                if m.model_size <= mem[g]:
                    ratio = w[g] / mem[g]
                    if ratio < g_ratio:
                        g_best, g_ratio = g, ratio
            if g_best is None:
                raise ValueError("Unable to place all models on GPUs")
            best[g_best].append(m)
            w[g_best] += m.req_rate / m.slo
            mem[g_best] -= m.model_size
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
