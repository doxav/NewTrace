"""Probe U2 — probe U re-analysed with the BUNDLE DEFAULT as the common starting point.

Probe U scored both arms purely on menu draws, which ignores that every real run starts from the
task's own template heuristic and keeps it unless something beats it. That baseline is not
neutral: on `tsp_construct` the evaluator hands the heuristic a PRE-SORTED `unvisited_near_nodes`
array, so the template's `unvisited_nodes[0]` IS nearest-neighbour and the default is already the
menu optimum. Reporting `b_std = 9` without saying so overstates the standard arm's cost.

Here both arms start at the default and quality is max(default, best of the b candidates drawn).
Same exact enumeration, same normalisation, same folds.
"""
import itertools, json, math, statistics as st, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from opto.features.recursive_opt import tracebench as TB       # noqa: E402
from opto.features.recursive_opt.levels import LevelConfig     # noqa: E402
from probe_t_routing_menu import MENU, ROUTING                 # noqa: E402
from probe_u_w2_sweep import NAMES, norm, prior_order, first_b, QS  # noqa: E402

FAMILY = [t for t in ROUTING if not t.endswith("ovrp_construct")]


def main():
    TB.ensure_default_task_adapter(require=True)
    ad = TB.current_task_adapter()

    def sc(tid, text):
        b = ad._load_bundle(tid, fresh=True)
        if text is not None:
            ad._apply_starting_artifact(b, LevelConfig(starting_artifact=text))
        return float(TB._score_bundle(b, 2)[0])

    raw = {t: {n: sc(t, MENU[n]) for n in NAMES} for t in FAMILY}
    default = {t: sc(t, None) for t in FAMILY}

    out = {"raw": raw, "default_raw": default, "folds": {}, "curve": {}, "breakeven": {}}
    for target in FAMILY:
        row = dict(raw[target])
        row["__default__"] = default[target]           # normalise on the union
        rn = norm(row)
        d0 = rn.pop("__default__")
        order, _ = prior_order(raw, [s for s in FAMILY if s != target])
        vals = [rn[n] for n in NAMES]
        std, rec = {}, {}
        for b in range(0, len(NAMES) + 1):
            if b == 0:
                std[b] = {"E": d0, "P_opt": float(d0 >= 1.0 - 1e-12)}
                rec[b] = {"q": d0}
                continue
            bests = [max(d0, max(c)) for c in itertools.combinations(vals, b)]
            std[b] = {"E": sum(bests) / len(bests),
                      "P_opt": sum(x >= 1.0 - 1e-12 for x in bests) / len(bests)}
            rec[b] = {"q": max(d0, max(rn[n] for n in order[:b]))}
        out["folds"][target] = {
            "default_q_norm": d0, "prior_order": order, "prior_top1": order[0],
            "target_argmax": max(NAMES, key=lambda n: rn[n]),
            "standard": {str(b): std[b] for b in std},
            "recursive": {str(b): rec[b] for b in rec}}

    bs = list(range(0, len(NAMES) + 1))
    F = out["folds"]
    out["curve"] = {
        "b": bs,
        "standard_E": [st.mean(F[t]["standard"][str(b)]["E"] for t in FAMILY) for b in bs],
        "recursive_q": [st.mean(F[t]["recursive"][str(b)]["q"] for t in FAMILY) for b in bs],
        "standard_P_opt": [st.mean(F[t]["standard"][str(b)]["P_opt"] for t in FAMILY)
                           for b in bs]}
    c_meta = len(NAMES) * (len(FAMILY) - 1)
    out["c_meta_evals"] = c_meta
    for Q in QS:
        ps = {b: {"E": out["curve"]["standard_E"][b]} for b in bs}
        pr = {b: {"q": out["curve"]["recursive_q"][b]} for b in bs}
        bstd, brec = first_b(ps, "E", Q), first_b(pr, "q", Q)
        gap = None if bstd is None or brec is None else bstd - brec
        out["breakeven"][str(Q)] = {
            "b_std": bstd, "b_rec": brec, "compute_saved_per_task": gap,
            "K_star": math.inf if not gap or gap <= 0 else c_meta / gap}

    p = Path(__file__).with_name("probe_u2_default_baseline.json")
    p.write_text(json.dumps(out, indent=2, default=str) + "\n")
    c = out["curve"]
    print("b            " + " ".join(f"{b:>7d}" for b in c["b"]))
    print("standard E   " + " ".join(f"{x:7.4f}" for x in c["standard_E"]))
    print("recursive q  " + " ".join(f"{x:7.4f}" for x in c["recursive_q"]))
    print("std P(opt)   " + " ".join(f"{x:7.4f}" for x in c["standard_P_opt"]))
    for t in FAMILY:
        print(f"  {t.split('/')[-1]:16s} default q_norm={F[t]['default_q_norm']:.4f} "
              f"prior_top1={F[t]['prior_top1']} argmax={F[t]['target_argmax']}")
    print(f"c_meta={c_meta} evals")
    for Q, r in out["breakeven"].items():
        print(f"  Q={Q}: b_std={r['b_std']} b_rec={r['b_rec']} "
              f"saved={r['compute_saved_per_task']} K*={r['K_star']}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
