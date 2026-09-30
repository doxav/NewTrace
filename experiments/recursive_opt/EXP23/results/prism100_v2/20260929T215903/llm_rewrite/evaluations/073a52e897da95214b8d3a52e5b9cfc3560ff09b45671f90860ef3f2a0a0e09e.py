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

    # Try multiple orderings; keep the placement with the lowest max KVPR
    keys = [
        lambda m: m.req_rate / m.slo,
        lambda m: m.model_size,
        lambda m: m.req_rate,
        lambda m: m.req_rate / (m.slo * m.model_size),
        lambda m: m.req_rate * m.model_size,
    ]
    def two_worst(loads, mems):
        vals = sorted((loads[g] / mems[g] for g in loads), reverse=True)
        return vals[0], vals[1] if len(vals) > 1 else 0.0

    def local_search(placement):
        """Iteratively move/swap models out of the worst GPU to reduce max KVPR.

        Accepts strict improvements on the max KVPR; if none exist, accepts
        equal-max moves that lower the second-highest KVPR (plateau escape).
        """
        for _ in range(200):
            loads = {g: sum(m.req_rate / m.slo for m in ms) for g, ms in placement.items()}
            mems = {g: GPU_MEM_SIZE - sum(m.model_size for m in ms) for g, ms in placement.items()}
            if any(v <= 0 for v in mems.values()):
                return placement
            worst_g = max(placement, key=lambda g: loads[g] / mems[g])
            cur, cur_second = two_worst(loads, mems)
            best_key = None  # (new_max, new_second) lexicographic
            best_move = None

            for m in list(placement[worst_g]):
                w = m.req_rate / m.slo
                for g2 in placement:
                    if g2 == worst_g or m.model_size > mems[g2]:
                        continue
                    others = [loads[g] / mems[g] for g in placement if g not in (worst_g, g2)]
                    cand = sorted(others + [
                        (loads[worst_g] - w) / (mems[worst_g] + m.model_size),
                        (loads[g2] + w) / (mems[g2] - m.model_size),
                    ], reverse=True)
                    key = (cand[0], cand[1] if len(cand) > 1 else 0.0)
                    if best_key is None or (key < best_key and key < (cur, cur_second)):
                        best_key = key
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
                        others = [loads[g] / mems[g] for g in placement if g not in (worst_g, g2)]
                        cand = sorted(others + [
                            (loads[worst_g] - w1 + w2) / (mems[worst_g] + d),
                            (loads[g2] - w2 + w1) / (mems[g2] - d),
                        ], reverse=True)
                        key = (cand[0], cand[1] if len(cand) > 1 else 0.0)
                        if best_key is None or (key < best_key and key < (cur, cur_second)):
                            best_key = key
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

    best = None
    best_score = float("inf")
    best_second = float("inf")

    def second_kvpr(placement):
        vals = sorted(
            (
                sum(m.req_rate / m.slo for m in ms)
                / (GPU_MEM_SIZE - sum(m.model_size for m in ms))
                for ms in placement.values()
                if sum(m.model_size for m in ms) < GPU_MEM_SIZE
            ),
            reverse=True,
        )
        return vals[1] if len(vals) > 1 else 0.0

    def consider(order):
        nonlocal best, best_score, best_second
        try:
            p = greedy(order)
            if p is not None:
                p = local_search(p)
                s = max_kvpr(p)
                # tie-break on second-highest KVPR to escape plateaus
                if s < best_score - 1e-12 or (
                    abs(s - best_score) <= 1e-12 and second_kvpr(p) < best_second
                ):
                    best_score = s
                    best_second = second_kvpr(p)
                    best = p
        except Exception:
            pass

    for key in keys:
        for rev in (True, False):
            consider(sorted(models, key=key, reverse=rev))

    # Randomized restarts: perturb orderings to escape greedy local optima
    import random
    import time
    rng = random.Random(42)
    deadline = time.time() + 8.0  # hard time budget
    bases = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / (m.slo * m.model_size), reverse=True),
    ]
    if best is not None:
        bases.append([m for ms in best.values() for m in ms])
    for base in bases:
        for _ in range(400):
            if time.time() > deadline:
                break
            order = list(base)
            for _ in range(rng.randint(1, max(1, min(6, len(order))))):
                i, j = rng.randrange(len(order)), rng.randrange(len(order))
                order[i], order[j] = order[j], order[i]
            consider(order)

    if best is None:
        # Fallback: best-fit decreasing (largest first, tightest fitting GPU).
        # Best-effort: never raise; place on GPU with most remaining memory if needed.
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
