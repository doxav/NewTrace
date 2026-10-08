"""EXP29 P0.2: LLM-free headroom probe of the numeric family (random < seed < hand-written ES?).

Run: python -m experiments.recursive_opt.EXP29.numeric.probe_headroom  (writes EXP29/results/numeric_headroom.json)
"""
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import task as T

RANDOM = '''def propose(history, bounds, seed):
    import random
    rng = random.Random(seed * 1000003 + len(history))
    return [rng.uniform(low, high) for low, high in bounds]
'''
ES = '''def propose(history, bounds, seed):
    """(1+1)-ES with a 1/5-style step rule recomputed from history, random init and restarts."""
    import random
    rng = random.Random(seed * 1000003 + len(history))
    n = len(bounds)
    init = 2 * n + 2
    if len(history) < init:
        return [rng.uniform(low, high) for low, high in bounds]
    incumbent = min(history[:init], key=lambda row: row["value"])
    sigma = 0.2
    for row in history[init:]:
        if row["value"] < incumbent["value"]:
            incumbent, sigma = row, min(0.5, sigma * 1.5)
        else:
            sigma *= 1.5 ** -0.25
    if sigma < 1e-3:
        return [rng.uniform(low, high) for low, high in bounds]
    return [max(low, min(high, x + rng.gauss(0.0, sigma * (high - low)))) for x, (low, high) in zip(incumbent["x"], bounds)]
'''
SURROGATE = '''def propose(history, bounds, seed):
    """Random init, then alternate: centre of a per-coordinate quadratic fit of the best half / Gaussian step around the best."""
    import random
    rng = random.Random(seed * 1000003 + len(history))
    n = len(bounds)
    init = n + 3
    if len(history) < init:
        return [rng.uniform(low, high) for low, high in bounds]
    ranked = sorted(history, key=lambda row: row["value"])
    best = ranked[0]["x"]
    step = len(history) - init
    if step % 2 == 0:
        top = ranked[: max(init, len(ranked) // 2)]
        point = []
        for i, (low, high) in enumerate(bounds):
            xs = [row["x"][i] for row in top]
            ys = [row["value"] for row in top]
            m = len(xs)
            sx = [sum(x ** k for x in xs) for k in range(5)]
            sy = [sum(y * x ** k for x, y in zip(xs, ys)) for k in range(3)]
            a11, a12, a13, a22, a23, a33 = sx[4], sx[3], sx[2], sx[2], sx[1], m
            det = a11 * (a22 * a33 - a23 * a23) - a12 * (a12 * a33 - a23 * a13) + a13 * (a12 * a23 - a22 * a13)
            centre = best[i]
            if abs(det) > 1e-12:
                qa = (sy[2] * (a22 * a33 - a23 * a23) - a12 * (sy[1] * a33 - a23 * sy[0]) + a13 * (sy[1] * a23 - a22 * sy[0])) / det
                qb = (a11 * (sy[1] * a33 - a23 * sy[0]) - sy[2] * (a12 * a33 - a23 * a13) + a13 * (a12 * sy[0] - sy[1] * a13)) / det
                if qa > 0:
                    centre = -qb / (2 * qa)
            point.append(max(low, min(high, centre + rng.gauss(0, 0.02 * (high - low)))))
        return point
    sigma = 0.15 / (1 + step / 4) ** 0.5
    return [max(low, min(high, x + rng.gauss(0.0, sigma * (high - low)))) for x, (low, high) in zip(best, bounds)]
'''
PROGRAMS = {'random': RANDOM, 'seed': T.SEED_SOURCE, 'es_hand': ES, 'surrogate_hand': SURROGATE}


def _one(args):
    name, item = args
    return name, item['instance']['family'], item['instance']['dimension'], T.run_program(PROGRAMS[name], item['instance'], item['seed'])


def main() -> None:
    items = T.items(T.FAMILIES, (2, 5), 'probe', 4)
    with ProcessPoolExecutor(16) as pool:
        rows = list(pool.map(_one, [(n, i) for n in PROGRAMS for i in items]))
    table = {}
    for name, family, dim, result in rows:
        cell = table.setdefault(f'{family}-{dim}d', {}).setdefault(name, [])
        cell.append(result)
    summary = {stratum: {name: {'auc': round(statistics.mean(r['auc'] for r in rs), 4), 'log': round(statistics.mean(r['log_auc'] for r in rs), 3), 'final': round(statistics.median(r['final_regret'] for r in rs), 4),
                                'hits': sum(r['target_evaluations'] is not None for r in rs), 'n': len(rs)} for name, rs in cells.items()}
               for stratum, cells in sorted(table.items())}
    overall = {name: {'auc': round(statistics.mean(r['auc'] for n, _f, _d, r in rows if n == name), 4), 'log_auc': round(statistics.mean(r['log_auc'] for n, _f, _d, r in rows if n == name), 3)} for name in PROGRAMS}
    out = {'budget': T.BUDGET, 'bounds': T.BOUNDS, 'instances_per_stratum': 4, 'overall_auc': overall, 'strata': summary}
    (Path(__file__).resolve().parents[1] / 'results' / 'numeric_headroom.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(overall))
    for stratum, cells in summary.items():
        print(stratum, {n: (c['log'], c['final'], c['hits']) for n, c in cells.items()})


if __name__ == '__main__':
    main()
