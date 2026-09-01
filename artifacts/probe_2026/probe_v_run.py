"""Probe V runner — the (arm x target x budget x seed) matrix, one process per cell.

Each cell is an independent process with its own budget guard and memory root, so cells can be
run concurrently without sharing state. The deterministic evaluator's replicate range was
measured at this concurrency by probe X before any of this is believed.
"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve()
ARMS = ("standard", "recursive")
TARGETS = [0, 1, 2]
BUDGETS = [int(x) for x in os.environ.get("PROBE_V_BUDGETS", "2,4").split(",")]
SEEDS = [int(x) for x in os.environ.get("PROBE_V_SEEDS", "0").split(",")]
CONC = int(os.environ.get("PROBE_V_CONCURRENCY", "6"))


def cell(args):
    arm, ti, b, s = args
    p = subprocess.run([sys.executable, str(HERE.parent / "probe_v_harness.py"),
                        arm, str(ti), str(b), str(s)],
                       capture_output=True, text=True, cwd=str(HERE.parent),
                       env={**os.environ, "PYTHONPATH": str(HERE.parents[2])})
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
            print(f"  {arm:9s} t{ti} b{b} s{s} score={r.get('score')} "
                  f"err={r.get('error')}", flush=True)
            return r
    return {"arm": arm, "target_index": ti, "budget": b, "seed": s,
            "error": (p.stderr or p.stdout)[-300:]}


def main():
    cells = list(product(ARMS, TARGETS, BUDGETS, SEEDS))
    with ThreadPoolExecutor(CONC) as ex:
        rows = list(ex.map(cell, cells))

    by = {}
    for r in rows:
        if r.get("score") is not None:
            by[(r["arm"], r["target"], r["budget"], r["seed"])] = r
    summary = {"budgets": BUDGETS, "seeds": SEEDS, "concurrency": CONC, "rows": rows,
               "paired": [], "artifacts": {}}
    for arm in ARMS:
        summary["artifacts"][arm] = sorted({r.get("artifact", "") for r in rows
                                            if r["arm"] == arm and r.get("score") is not None})
    summary["artifacts_differ"] = (
        len({a for arts in summary["artifacts"].values() for a in arts}) > 1)
    # paired deltas, keyed on (target, budget, seed) so both arms are the same level+task set
    keys = {(r["target"], r["budget"], r["seed"]) for r in rows if r.get("score") is not None}
    for k in sorted(keys):
        a = next((r for r in rows if r.get("score") is not None and r["arm"] == "standard"
                  and (r["target"], r["budget"], r["seed"]) == k), None)
        c = next((r for r in rows if r.get("score") is not None and r["arm"] == "recursive"
                  and (r["target"], r["budget"], r["seed"]) == k), None)
        if a and c:
            summary["paired"].append({
                "target": k[0], "budget": k[1], "seed": k[2],
                "standard": a["score"], "recursive": c["score"],
                "delta": c["score"] - a["score"],
                "same_artifact": a.get("artifact") == c.get("artifact"),
                "scored_task_ids": a["scored_task_ids"],
                "same_scored_task_ids": a["scored_task_ids"] == c["scored_task_ids"]})
    f = HERE.with_name("probe_v_results.json")
    f.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print("\n=== paired (same target, budget, seed; both arms scored on o3_prior) ===")
    for p in summary["paired"]:
        print(f"  {p['target'].split('/')[-1]:16s} b={p['budget']} s={p['seed']} "
              f"std={p['standard']} rec={p['recursive']} delta={p['delta']} "
              f"same_artifact={p['same_artifact']}")
    print(f"artifacts_differ={summary['artifacts_differ']}")
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
