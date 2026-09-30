GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START


def compute_model_placement(gpu_num, models):
    """Multi-start greedy + local search minimizing max KVPR, with a
    feasibility guarantee. Greedy restarts use mostly size-descending
    orderings (packing-friendly) with small randomness, so they almost
    always fit. If all restarts fail, a first-fit-decreasing fallback
    guarantees a feasible placement. Local search moves models out of
    the bottleneck GPU whenever it strictly lowers max KVPR."""
    import random

    if not models:
        return {g: [] for g in range(gpu_num)}

    def score(assign):
        load = [0.0] * gpu_num
        mem = [0.0] * gpu_num
        for m, g in assign:
            load[g] += m.req_rate / m.slo
            mem[g] += m.model_size
        return max(load[g] / max(GPU_MEM_SIZE - mem[g], 1e-9)
                   for g in range(gpu_num))

    def greedy(order, eps):
        assign = []
        mem = [0.0] * gpu_num
        load = [0.0] * gpu_num
        for m in order:
            cands = [g for g in range(gpu_num) if mem[g] + m.model_size <= GPU_MEM_SIZE]
            if not cands:
                return None
            if random.random() < eps:
                g = random.choice(cands)
            else:
                g = min(cands, key=lambda g: load[g] / (GPU_MEM_SIZE - mem[g] - m.model_size))
            assign.append((m, g))
            mem[g] += m.model_size
            load[g] += m.req_rate / m.slo
        return assign

    def local_search(assign):
        for _ in range(200):
            cur = score(assign)
            improved = False
            loads = [0.0] * gpu_num
            mems = [0.0] * gpu_num
            for m, g in assign:
                loads[g] += m.req_rate / m.slo
                mems[g] += m.model_size
            bg = max(range(gpu_num),
                     key=lambda g: loads[g] / max(GPU_MEM_SIZE - mems[g], 1e-9))
            # Try moves out of the bottleneck GPU, then swaps.
            for i, (m, g) in enumerate(assign):
                if g != bg:
                    continue
                for h in range(gpu_num):
                    if h == g or mems[h] + m.model_size > GPU_MEM_SIZE:
                        continue
                    assign[i] = (m, h)
                    s = score(assign)
                    if s < cur - 1e-12:
                        cur, improved = s, True
                        mems[g] -= m.model_size; mems[h] += m.model_size
                        loads[g] -= m.req_rate / m.slo; loads[h] += m.req_rate / m.slo
                        bg = max(range(gpu_num),
                                 key=lambda g2: loads[g2] / max(GPU_MEM_SIZE - mems[g2], 1e-9))
                        break
                    assign[i] = (m, g)
            if not improved:
                # Swap phase: swap bottleneck model with a model on another GPU.
                for i, (m, g) in enumerate(assign):
                    if g != bg:
                        continue
                    for j, (m2, g2) in enumerate(assign):
                        if g2 == g or g2 == bg:
                            continue
                        if mems[g2] - m2.model_size + m.model_size > GPU_MEM_SIZE:
                            continue
                        if mems[g] - m.model_size + m2.model_size > GPU_MEM_SIZE:
                            continue
                        assign[i] = (m, g2)
                        assign[j] = (m2, g)
                        s = score(assign)
                        if s < cur - 1e-12:
                            improved = True
                            break
                        assign[i] = (m, g)
                        assign[j] = (m2, g2)
                    if improved:
                        break
            if not improved:
                break
        return assign

    best, best_s = None, float("inf")
    random.seed(42)
    by_size = sorted(models, key=lambda x: x.model_size, reverse=True)
    tries = 0
    while tries < 100:
        # mostly packing-friendly size-descending order with jitter
        order = by_size if tries < 4 else sorted(models, key=lambda x: (random.random(), -x.model_size))
        assign = greedy(order, eps=0.1)
        tries += 1
        if assign is None:
            continue
        assign = local_search(assign)
        s = score(assign)
        if s < best_s:
            best, best_s = assign, s
    if best is None:
        # Fallback: first-fit decreasing guarantees feasibility whenever
        # the models fit at all.
        best = []
        mem = [0.0] * gpu_num
        for m in by_size:
            for g in range(gpu_num):
                if mem[g] + m.model_size <= GPU_MEM_SIZE:
                    best.append((m, g))
                    mem[g] += m.model_size
                    break
    placement = {g: [] for g in range(gpu_num)}
    for m, g in best:
        placement[g].append(m)
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
