"""Probe U — Tier 1 W2 (amortisation) budget sweep on the routing family.

Pre-registered in PREREG_W2_routing.md. Tests ONLY W2: total evaluation compute to reach
quality Q across K tasks. Quality-at-equal-budget is not the claim and is not reported as one.

standard(b)  : b candidates drawn uniformly without replacement from the 9-entry menu, best kept.
               Reported as the EXACT expectation over all C(9,b) subsets (no Monte-Carlo error,
               no seed noise) plus the exact distribution.
recursive(b) : leave-one-out. Prior = menu ordered by mean per-source min-max-normalised score
               over the source tasks; the first b in that order are evaluated on the target.
               c_meta = evaluations spent on the sources.

Everything below is arithmetic over a score table measured once from the real benchmark
(deterministic; probe T verified replicate range 0.0).
"""
import itertools, json, math, statistics as st, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from opto.features.recursive_opt import tracebench as TB       # noqa: E402
from opto.features.recursive_opt.levels import LevelConfig     # noqa: E402
from probe_t_routing_menu import MENU, ROUTING                 # noqa: E402

NAMES = list(MENU)
INVALID = 1e5
QS = [1.00, 0.99, 0.95]
DEDUP = [t for t in ROUTING if not t.endswith("ovrp_construct")]  # ovrp is cvrp's rho=1.0 twin


# ---------------------------------------------------------------- measurement
def score_table(reps=1):
    TB.ensure_default_task_adapter(require=True)
    ad = TB.current_task_adapter()
    tbl, replicates = {}, {}
    for tid in ROUTING:
        row = {}
        for n in NAMES:
            vals = []
            for _ in range(reps):
                b = ad._load_bundle(tid, fresh=True)
                ad._apply_starting_artifact(b, LevelConfig(starting_artifact=MENU[n]))
                vals.append(float(TB._score_bundle(b, 2)[0]))
            row[n] = vals[0]
            if reps > 1:
                replicates.setdefault(tid, {})[n] = {
                    "vals": vals, "range": round(max(vals) - min(vals), 12)}
        tbl[tid] = row
    return tbl, replicates


# ---------------------------------------------------------------- helpers
def norm(row):
    """min-max normalise a task's candidate scores to [0,1] (1 = that task's menu optimum)."""
    lo, hi = min(row.values()), max(row.values())
    return {k: (1.0 if hi == lo else (v - lo) / (hi - lo)) for k, v in row.items()}


def std_curve(row_n):
    """Exact E[best-of-b] over all C(9,b) uniform subsets, plus P(reaching the optimum)."""
    v = [row_n[n] for n in NAMES]
    m = len(v)
    out = {}
    for b in range(1, m + 1):
        bests = [max(c) for c in itertools.combinations(v, b)]
        out[b] = {"E": sum(bests) / len(bests), "P_opt": sum(x >= 1.0 for x in bests) / len(bests),
                  "n_subsets": len(bests), "min": min(bests), "max": max(bests)}
    return out


def prior_order(tbl, sources):
    """Meta-optimisation: rank candidates by mean normalised score across the source tasks."""
    agg = {n: st.mean(norm(tbl[s])[n] for s in sources) for n in NAMES}
    return sorted(NAMES, key=lambda n: (-agg[n], NAMES.index(n))), agg


def rec_curve(row_n, order):
    return {b: {"q": max(row_n[n] for n in order[:b])} for b in range(1, len(NAMES) + 1)}


def first_b(curve, key, Q):
    for b in sorted(curve):
        if curve[b][key] >= Q - 1e-12:
            return b
    return None


# ---------------------------------------------------------------- analysis
def analyse(tbl, family, meta_m):
    """meta_m = candidates evaluated per source task when building the prior."""
    res = {"family": family, "meta_candidates_per_source": meta_m, "folds": {}}
    for target in family:
        sources = [s for s in family if s != target]
        # budgeted meta: only the first meta_m menu entries are evaluated on each source.
        sub = NAMES[:meta_m]
        agg = {n: st.mean(norm({k: tbl[s][k] for k in sub})[n] for s in sources) for n in sub}
        order = sorted(sub, key=lambda n: (-agg[n], NAMES.index(n)))
        order = order + [n for n in NAMES if n not in order]   # unranked tail, menu order
        row_n = norm(tbl[target])
        sc, rc = std_curve(row_n), rec_curve(row_n, order)
        res["folds"][target] = {
            "sources": sources, "prior_order": order,
            "prior_top1": order[0], "target_argmax": max(NAMES, key=lambda n: row_n[n]),
            "prior_top1_is_target_argmax": order[0] == max(NAMES, key=lambda n: row_n[n]),
            "standard": {str(b): sc[b] for b in sc},
            "recursive": {str(b): rc[b] for b in rc},
            "raw_scores": tbl[target],
        }
    # pooled curves
    F = res["folds"]
    bs = list(range(1, len(NAMES) + 1))
    res["curve"] = {
        "b": bs,
        "standard_E": [st.mean(F[t]["standard"][str(b)]["E"] for t in family) for b in bs],
        "standard_P_opt": [st.mean(F[t]["standard"][str(b)]["P_opt"] for t in family) for b in bs],
        "recursive_q": [st.mean(F[t]["recursive"][str(b)]["q"] for t in family) for b in bs],
    }
    c_meta = meta_m * (len(family) - 1)
    res["c_meta_evals"] = c_meta
    res["breakeven"] = {}
    for Q in QS:
        pooled_std = {b: {"E": res["curve"]["standard_E"][b - 1]} for b in bs}
        pooled_rec = {b: {"q": res["curve"]["recursive_q"][b - 1]} for b in bs}
        b_std, b_rec = first_b(pooled_std, "E", Q), first_b(pooled_rec, "q", Q)
        gap = None if (b_std is None or b_rec is None) else b_std - b_rec
        res["breakeven"][str(Q)] = {
            "b_std": b_std, "b_rec": b_rec, "compute_saved_per_task": gap,
            "K_star": (math.inf if not gap or gap <= 0 else c_meta / gap),
        }
    # worst-case (max over folds) standard budget, a stricter c_std
    res["breakeven_perfold"] = {
        str(Q): {t: {"b_std": first_b({int(k): v for k, v in F[t]["standard"].items()}, "E", Q),
                     "b_rec": first_b({int(k): v for k, v in F[t]["recursive"].items()}, "q", Q)}
                 for t in family}
        for Q in QS}
    return res


def main():
    t0 = time.time()
    reps = 10
    tbl, replicates = score_table(reps=reps)
    out = {"menu_size": len(NAMES), "names": NAMES, "score_table": tbl,
           "replicates_n": reps,
           "replicate_range_max": max((r["range"] for t in replicates.values()
                                       for r in t.values()), default=None),
           "analyses": []}
    for family, tag in ((ROUTING, "family4"), (DEDUP, "family3_dedup")):
        for meta_m in (3, 5, 9):
            a = analyse(tbl, family, meta_m)
            a["tag"] = f"{tag}_meta{meta_m}"
            out["analyses"].append(a)
    out["wall_s"] = round(time.time() - t0, 1)
    p = Path(__file__).with_name("probe_u_w2_sweep.json")
    p.write_text(json.dumps(out, indent=2, default=str) + "\n")

    print(f"replicate range over {reps} re-scores of every (task,candidate): "
          f"max = {out['replicate_range_max']}")
    for a in out["analyses"]:
        print(f"\n=== {a['tag']}  c_meta={a['c_meta_evals']} evals ===")
        c = a["curve"]
        print("  b            " + " ".join(f"{b:>7d}" for b in c["b"]))
        print("  standard E   " + " ".join(f"{x:7.4f}" for x in c["standard_E"]))
        print("  recursive q  " + " ".join(f"{x:7.4f}" for x in c["recursive_q"]))
        print("  std P(opt)   " + " ".join(f"{x:7.4f}" for x in c["standard_P_opt"]))
        for Q, r in a["breakeven"].items():
            print(f"  Q={Q}: b_std={r['b_std']} b_rec={r['b_rec']} "
                  f"saved/task={r['compute_saved_per_task']} K*={r['K_star']}")
        bad = [t for t, f in a["folds"].items() if not f["prior_top1_is_target_argmax"]]
        print(f"  prior top1 == target argmax on {len(a['folds']) - len(bad)}/{len(a['folds'])} "
              f"folds; misses={[t.split('/')[-1] for t in bad]}")
    print(f"\nwrote {p}  ({out['wall_s']}s)")


if __name__ == "__main__":
    main()
