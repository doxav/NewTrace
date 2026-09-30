"""Exact optimum of PRISM's 50 cases: minimize max_g W_g / (80 - S_g) by bisection on T with
exact bin-packing feasibility (item weight w_i + T*s_i <= 80*T, and sizes <= 80), branch and bound."""

import importlib.util
import json
import sys
from pathlib import Path
from statistics import mean

spec = importlib.util.spec_from_file_location('ev', '/home/xav/code/evo-compare/repos/skydiscover/benchmarks/ADRS/prism/evaluator/evaluator.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
sys.setrecursionlimit(10000)


def feasible(T, items, n):
    # items: (w, s); need per bin: sum w + T*sum s <= 80 T and sum s < 80
    load = [0.0] * n
    size = [0] * n
    cap = 80 * T
    order = sorted(items, key=lambda it: -(it[0] + T * it[1]))

    def place(k):
        if k == len(order):
            return True
        w, s = order[k]
        cost = w + T * s
        seen = set()
        for g in range(n):
            key = (round(load[g], 12), size[g])
            if key in seen:
                continue  # symmetric bins
            seen.add(key)
            if load[g] + cost <= cap + 1e-12 and size[g] + s < 80:
                load[g] += cost
                size[g] += s
                if place(k + 1):
                    return True
                load[g] -= cost
                size[g] -= s
        return False
    return place(0)


def optimum(n, models):
    items = [(m.req_rate / m.slo, m.model_size) for m in models]
    W, S = sum(w for w, _ in items), sum(s for _, s in items)
    lo = max(W / (80 * n - S), max(w / (80 - s) for w, s in items))
    hi = lo
    while not feasible(hi, items, n):
        hi *= 1.1
    for _ in range(40):
        mid = (lo + hi) / 2
        if feasible(mid, items, n):
            hi = mid
        else:
            lo = mid
    return hi


def main():
    values = []
    for i, (n, models) in enumerate(ev.generate_test_gpu_models()):
        values.append(optimum(n, models))
        print(i, round(values[-1], 6), flush=True)
    report = {'exact_optimal_score': 1 / mean(values) + 1.0, 'per_case': values}
    out = Path(__file__).resolve().parents[1] / 'results' / 'analysis' / 'prism_exact_optimum.json'
    out.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({'exact_optimal_score': report['exact_optimal_score']}))


if __name__ == '__main__':
    main()
