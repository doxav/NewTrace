"""Probe T2 - how does the routing instrument behave as `max_examples` grows?

`max_examples` selects how many benchmark instances the score averages over. Too few and the
ranking is an artefact of one instance; the W2 experiment must be run on a setting where the
ranking is stable. Also times each setting so the budget sweep can be costed.
"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from opto.features.recursive_opt import tracebench as TB       # noqa: E402
from opto.features.recursive_opt.levels import LevelConfig     # noqa: E402
from probe_t_routing_menu import MENU, ROUTING, spearman       # noqa: E402


def score(ad, tid, text, mx):
    b = ad._load_bundle(tid, fresh=True)
    ad._apply_starting_artifact(b, LevelConfig(starting_artifact=text))
    return float(TB._score_bundle(b, mx)[0])


def main():
    TB.ensure_default_task_adapter(require=True)
    ad = TB.current_task_adapter()
    names = list(MENU)
    out = {}
    for mx in (2, 4, 8, 16):
        t0 = time.time()
        tbl = {tid: {n: score(ad, tid, MENU[n], mx) for n in names} for tid in ROUTING}
        out[str(mx)] = {"scores": tbl, "wall_s": round(time.time() - t0, 1)}
        print(f"max_examples={mx:3d}  {round(time.time()-t0,1)}s  "
              f"argmax={ {t.split('/')[-1]: max(names, key=lambda n: tbl[t][n]) for t in ROUTING} }",
              flush=True)
    # stability of the ranking between consecutive settings, per task
    stab = {}
    keys = list(out)
    for tid in ROUTING:
        stab[tid] = {f"{a}->{b}": spearman([out[a]["scores"][tid][n] for n in names],
                                           [out[b]["scores"][tid][n] for n in names])
                     for a, b in zip(keys, keys[1:])}
    out["ranking_stability_vs_max_examples"] = stab
    p = Path(__file__).with_name("probe_t2_maxexamples.json")
    p.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps(stab, indent=2))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
