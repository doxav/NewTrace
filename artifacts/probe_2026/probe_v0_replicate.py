"""Probe V0 — in-run replicate control for the PRODUCTION harness.

Probe T showed the raw evaluator is bit-deterministic (replicate range 0.0 over 10 re-scores of
every task x candidate). But two probe-V smoke runs of the IDENTICAL initial artifact on
`cvrp_construct` returned -14.3606 and -14.6461, so the harness-level score is NOT the raw
evaluator score. This measures the harness noise floor directly: n identical runs at budget 1
(which makes zero optimizer LLM calls - `Update/n_iters: 0`), sequentially, same process layout
as the real runs.

Any harness-level effect smaller than the range reported here is noise.
"""
import json, os, statistics as st, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve()
N = int(os.environ.get("PROBE_V0_N", "10"))
TARGETS = [0, 1, 2]
CONC = int(os.environ.get("PROBE_V0_CONCURRENCY", "1"))


def run(args):
    p = subprocess.run([sys.executable, str(HERE.parent / "probe_v_harness.py"), *args],
                       capture_output=True, text=True, cwd=str(HERE.parent),
                       env={**os.environ, "PYTHONPATH": str(HERE.parents[2])})
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"error": (p.stderr or p.stdout)[-300:]}


def main():
    out = {"n": N, "concurrency": CONC, "per_target": {}}
    for ti in TARGETS:
        rows = []
        if CONC <= 1:
            rows = [run(["standard", str(ti), "1", str(s)]) for s in range(N)]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(CONC) as ex:
                rows = list(ex.map(lambda s: run(["standard", str(ti), "1", str(s)]), range(N)))
        sc = [r["score"] for r in rows if r.get("score") is not None]
        arts = {r.get("artifact", "") for r in rows}
        out["per_target"][str(ti)] = {
            "task": rows[0].get("target"), "scores": sc,
            "range": round(max(sc) - min(sc), 6) if sc else None,
            "sd": round(st.pstdev(sc), 6) if len(sc) > 1 else None,
            "mean": round(st.mean(sc), 6) if sc else None,
            "n_ok": len(sc), "distinct_artifacts": len(arts), "artifacts": sorted(arts),
            "errors": [r.get("error") for r in rows if r.get("error")],
        }
        p = out["per_target"][str(ti)]
        print(f"target{ti} {p['task']}: n={p['n_ok']} mean={p['mean']} range={p['range']} "
              f"sd={p['sd']} distinct_artifacts={p['distinct_artifacts']}", flush=True)
    f = HERE.with_name(f"probe_v0_replicate_c{CONC}.json")
    f.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
