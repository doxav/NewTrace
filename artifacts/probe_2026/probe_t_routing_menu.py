"""Probe T — is there a SHARED OPTIMUM across the `select_next_node` routing family?

Precondition for W2 (amortisation). Transfer is vacuous unless the *ranking* of candidate
heuristics agrees across family members. Probe S showed the four routing tasks
(`tsp_construct`, `cvrp_construct`, `ovrp_construct`, `vrptw_construct`) all train the same
entry point `select_next_node`, called POSITIONALLY with a task-specific arg list:

    tsp   : (current_node, destination_node, unvisited, distance_matrix)
    cvrp  : (current_node, depot, unvisited, rest_capacity, demands, distance_matrix)
    ovrp  : (current_node, depot, unvisited, rest_capacity, demands, distance_matrix)
    vrptw : (current_node, depot, unvisited, rest_capacity, current_time, demands,
             distance_matrix, time_windows)

`unvisited` is always positional index 1 of `*a`, and `distance_matrix` is the LAST square
2-D array among the args. So a single `*a`-tolerant artifact text is literally transferable
across all four - which is what makes a code-artifact transfer experiment possible at all.

The menu is deliberately RANKING-DISTINCT (no monotone transforms of one another; only the
argmin/argmax matters for a construction heuristic), which is the collapse mode that made the
Iteration-3 menu effectively size 1 (assessment 19.1b).

Outputs per task: distinct valid scores, range, per-candidate score, wall time, and a
sequential replicate check. Then the Spearman rank correlation of the candidate ranking
between every pair of tasks.
"""
import itertools, json, statistics as st, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from opto.features.recursive_opt import tracebench as TB           # noqa: E402
from opto.features.recursive_opt.levels import LevelConfig         # noqa: E402
from opto.features.recursive_opt.measurement import detect_surface  # noqa: E402

ROUTING = [
    "llm4ad:optimization/tsp_construct",
    "llm4ad:optimization/cvrp_construct",
    "llm4ad:optimization/ovrp_construct",
    "llm4ad:optimization/vrptw_construct",
]

# --- shared, signature-tolerant preamble -----------------------------------------------
PRE = '''import numpy as np


def _ctx(current_node, a):
    u = np.asarray(a[1]).ravel().astype(int)
    D = None
    for x in a:
        try:
            arr = np.asarray(x, dtype=float)
        except Exception:
            continue
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] > 1:
            D = arr
    if D is None or u.size == 0:
        return u, None, None
    d = D[int(current_node)][u]
    return u, d, D


def select_next_node(current_node, *a):
    u, d, D = _ctx(current_node, a)
    if u.size == 0:
        return 0
    if d is None:
        return int(u[0])
'''

MENU = {
    # canonical greedy nearest-neighbour: expected family optimum
    "nearest":        PRE + "    return int(u[int(np.argmin(d))])\n",
    "farthest":       PRE + "    return int(u[int(np.argmax(d))])\n",
    "first_index":    PRE + "    return int(u[0])\n",
    "last_index":     PRE + "    return int(u[-1])\n",
    "median_dist":    PRE + "    return int(u[int(np.argmin(np.abs(d - np.median(d))))])\n",
    "second_nearest": PRE + ("    o = np.argsort(d, kind='stable')\n"
                             "    return int(u[o[1]] if o.size > 1 else u[o[0]])\n"),
    # tie-break among the 3 nearest by lowest node id: NOT a monotone transform of `nearest`
    "near3_lowid":    PRE + ("    o = np.argsort(d, kind='stable')[:3]\n"
                             "    return int(np.min(u[o]))\n"),
    # remote-first / central-first: use the full matrix, not just the current row
    "max_rowsum":     PRE + ("    s = D[np.ix_(u, u)].sum(axis=1)\n"
                             "    return int(u[int(np.argmax(s))])\n"),
    "min_rowsum":     PRE + ("    s = D[np.ix_(u, u)].sum(axis=1)\n"
                             "    return int(u[int(np.argmin(s))])\n"),
}

CONTROLS = {
    "raises":     PRE + "    raise RuntimeError('EXECUTED')\n",
    "syntax_err": "import numpy as np\ndef select_next_node(current_node, *a)\n    return 0\n",
    "wrong_name": PRE.replace("def select_next_node", "def not_the_entry") + "    return 0\n",
}

MAX_EXAMPLES = int(__import__("os").environ.get("PROBE_MAX_EXAMPLES", "2"))
INVALID = 1e5


def score(ad, tid, text):
    b = ad._load_bundle(tid, fresh=True)
    if text is not None:
        ad._apply_starting_artifact(b, LevelConfig(starting_artifact=text))
    t0 = time.time()
    try:
        s = float(TB._score_bundle(b, MAX_EXAMPLES)[0])
    except Exception as exc:
        return f"ERR {type(exc).__name__}: {exc}", round(time.time() - t0, 2)
    return s, round(time.time() - t0, 2)


def spearman(xs, ys):
    """Rank correlation over the shared candidate ordering (ties -> average ranks)."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return None if den == 0 else num / den


def main():
    TB.ensure_default_task_adapter(require=True)
    ad = TB.current_task_adapter()
    out = {"max_examples": MAX_EXAMPLES, "tasks": {}, "rank_corr": {}}

    for tid in ROUTING:
        b = ad._load_bundle(tid, fresh=True)
        surf = detect_surface(b)
        orig, orig_t = score(ad, tid, None)
        scores, times = {}, {}
        for name, code in MENU.items():
            s, t = score(ad, tid, code)
            scores[name], times[name] = s, t
            print(f"  {tid.split('/')[-1]:16s} {name:15s} {s} ({t}s)", flush=True)
        ctl = {k: score(ad, tid, v)[0] for k, v in CONTROLS.items()}
        # sequential replicate control on the single best candidate
        best = max((n for n, v in scores.items() if isinstance(v, float) and abs(v) < INVALID),
                   key=lambda n: scores[n], default=None)
        reps = [score(ad, tid, MENU[best])[0] for _ in range(3)] if best else []
        valid = [v for v in scores.values() if isinstance(v, float) and abs(v) < INVALID]
        out["tasks"][tid] = {
            "surface": surf.kind, "param_name": surf.param_name, "calls_llm": surf.calls_llm,
            "original_score": orig, "original_wall_s": orig_t,
            "scores": scores, "wall_s": times, "controls": ctl,
            "distinct": len(set(valid)), "range": round(max(valid) - min(valid), 4) if valid else 0.0,
            "headroom": len(set(valid)) > 1,
            "best_candidate": best, "replicates": reps,
            "replicate_range": (round(max(reps) - min(reps), 6)
                                if reps and all(isinstance(r, float) for r in reps) else None),
        }

    names = list(MENU)
    for x, y in itertools.combinations(ROUTING, 2):
        sx = [out["tasks"][x]["scores"][n] for n in names]
        sy = [out["tasks"][y]["scores"][n] for n in names]
        keep = [i for i in range(len(names))
                if isinstance(sx[i], float) and isinstance(sy[i], float)
                and abs(sx[i]) < INVALID and abs(sy[i]) < INVALID]
        out["rank_corr"][f"{x.split('/')[-1]} vs {y.split('/')[-1]}"] = {
            "n": len(keep),
            "rho": spearman([sx[i] for i in keep], [sy[i] for i in keep]),
            "argmax_x": max(keep, key=lambda i: sx[i], default=None) is not None
                        and names[max(keep, key=lambda i: sx[i])],
            "argmax_y": max(keep, key=lambda i: sy[i], default=None) is not None
                        and names[max(keep, key=lambda i: sy[i])],
        }

    p = Path(__file__).with_name("probe_t_routing_menu.json")
    p.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print("\n=== per-task ===")
    for tid, r in out["tasks"].items():
        print(f"{tid:44s} distinct={r['distinct']} range={r['range']} "
              f"best={r['best_candidate']} rep_range={r['replicate_range']} "
              f"orig={r['original_score']} ctl={r['controls']}")
    print("\n=== rank correlation (shared optimum?) ===")
    for k, v in out["rank_corr"].items():
        print(f"{k:46s} rho={v['rho']} n={v['n']} argmax: {v['argmax_x']} / {v['argmax_y']}")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
