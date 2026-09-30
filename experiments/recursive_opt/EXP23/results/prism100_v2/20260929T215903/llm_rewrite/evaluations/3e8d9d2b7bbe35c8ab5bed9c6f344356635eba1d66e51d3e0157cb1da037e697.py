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
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]
        weighted = [0.0 for _ in range(gpu_num)]
        for model in order:
            best_idx = None
            # choose GPU minimizing resulting KVPR after placement
            best_kvpr = float("inf")
            for gpu_id in range(gpu_num):
                if model.model_size <= shared_kv[gpu_id]:
                    new_kvpr = (weighted[gpu_id] + model.req_rate / model.slo) / (
                        shared_kv[gpu_id] - model.model_size
                    )
                    if new_kvpr < best_kvpr:
                        best_kvpr = new_kvpr
                        best_idx = gpu_id
            if best_idx is None:
                return None
            placement[best_idx].append(model)
            weighted[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        return placement

    def max_kvpr(placement):
        worst = 0.0
        for gpu_models in placement.values():
            mem = GPU_MEM_SIZE - sum(m.model_size for m in gpu_models)
            load = sum(m.req_rate / m.slo for m in gpu_models)
            if mem <= 0:
                return float("inf")
            worst = max(worst, load / mem)
        return worst

    def local_search(placement):
        """Iteratively move/swap models out of the worst GPU to reduce max KVPR."""
        for _ in range(30):
            loads = {g: sum(m.req_rate / m.slo for m in ms) for g, ms in placement.items()}
            mems = {g: GPU_MEM_SIZE - sum(m.model_size for m in ms) for g, ms in placement.items()}
            if any(v <= 0 for v in mems.values()):
                return placement
            worst_g = max(placement, key=lambda g: loads[g] / mems[g])
            cur = loads[worst_g] / mems[worst_g]
            best_delta = 1e-9
            best_move = None

            for m in list(placement[worst_g]):
                w = m.req_rate / m.slo
                for g2 in placement:
                    if g2 == worst_g or m.model_size > mems[g2]:
                        continue
                    new_worst = max(
                        (loads[g2] + w) / (mems[g2] - m.model_size),
                        (loads[worst_g] - w) / (mems[worst_g] + m.model_size),
                        max((loads[g] / mems[g]) for g in placement if g not in (worst_g, g2)),
                    )
                    if new_worst - cur < best_delta:
                        best_delta = new_worst - cur
                        best_move = ("move", m, g2)

            for m1 in list(placement[worst_g]):
                w1 = m1.req_rate / m1.slo
                for g2 in placement:
                    if g2 == worst_g:
                        continue
                    for m2 in placement[g2]:
                        w2 = m2.req_rate / m2.slo
                        d = m2.model_size - m1.model_size
                        if d > mems[g2] or -d > mems[worst_g]:
                            continue
                        new_worst = max(
                            (loads[g2] - w2 + w1) / (mems[g2] - d),
                            (loads[worst_g] - w1 + w2) / (mems[worst_g] + d),
                            max((loads[g] / mems[g]) for g in placement if g not in (worst_g, g2)),
                        )
                        if new_worst - cur < best_delta:
                            best_delta = new_worst - cur
                            best_move = ("swap", m1, m2, g2)

            if best_move is None:
                break
            if best_move[0] == "move":
                _, m, g2 = best_move
                placement[worst_g].remove(m)
                placement[g2].append(m)
            else:
                _, m1, m2, g2 = best_move
                placement[worst_g].remove(m1)
                placement[g2].remove(m2)
                placement[worst_g].append(m2)
                placement[g2].append(m1)
        return placement

    # Try multiple orderings; keep the placement with the lowest max KVPR
    keys = [
        lambda m: m.req_rate / m.slo,
        lambda m: m.model_size,
        lambda m: m.req_rate,
        lambda m: m.req_rate / (m.slo * m.model_size),
        lambda m: m.req_rate * m.model_size,
        lambda m: m.model_size / (m.req_rate / m.slo),
        lambda m: m.req_rate / (m.slo * (GPU_MEM_SIZE - m.model_size)),
    ]
    best = None
    best_score = float("inf")
    for key in keys:
        for rev in (True, False):
            try:
                p = greedy(sorted(models, key=key, reverse=rev))
                if p is not None:
                    p = local_search(p)
                    s = max_kvpr(p)
                    if s < best_score:
                        best_score = s
                        best = p
            except Exception:
                continue
    if best is None:
        # Fallback: best-fit decreasing (largest first, tightest fitting GPU).
        # Never overfills a GPU; never raises.
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]
        for model in sorted(models, key=lambda m: m.model_size, reverse=True):
            fits = [g for g in range(gpu_num) if model.model_size <= shared_kv[g]]
            if fits:
                g = min(fits, key=lambda g: shared_kv[g])
            else:
                g = max(range(gpu_num), key=lambda g: shared_kv[g])
            placement[g].append(model)
            shared_kv[g] -= model.model_size
        try:
            placement = local_search(placement)
        except Exception:
            pass
        best = placement
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
