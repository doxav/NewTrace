"""Probe K (deliverable #2) — optimization on a ZERO-NOISE certified surface.

Every result in this project so far has been fought over statistics: is the effect above
the noise floor, does the LCB cross zero, how many seeds. On a deterministic evaluator
that argument disappears. `llm4ad:optimization/online_bin_packing` scores by *running the
candidate heuristic*: the same program always scores the same. Certified noise is 0.0.

So the only variance is the optimizer's own proposal sampling, and the question becomes
unambiguous:

    Does the optimizer produce a heuristic that scores better than the baseline?

A single seed answers it for that seed. Multiple seeds characterise proposal variance,
not measurement noise. There is no confidence interval to argue about: if a candidate
scores better, it IS better.

Arms are scored on the identical task with the identical evaluator, so the D3
comparability requirement holds by construction.
"""
import json, statistics as st, time
from pathlib import Path

from opto.features.recursive_opt import measurement as M
from opto.features.recursive_opt import tracebench as TB
from opto.features.recursive_opt.budget import make_budget, reset_budget
from opto.features.recursive_opt.experiments import seed_everything
from opto.features.recursive_opt.spec import run_spec

TASK = "llm4ad:optimization/online_bin_packing"
SEEDS = [0, 1, 2]
OUT = Path(__file__).with_name("probe_k_out")


def spec_for(seed, *, optimize):
    return {
        "families": {"packing": [TASK]},
        "memory_root": str(OUT / f"{'opt' if optimize else 'base'}_seed{seed}"),
        "tracebench": {"max_examples": 2, "inner_steps": 0, "timeout_seconds": 120,
                       "eval_temperature": 0.2, "eval_max_tokens": 2048,
                       "eval_request_timeout": 90},
        "budget": {"optimizer_llm_calls": 30 if optimize else 0,
                   "eval_llm_calls": 60, "candidates": 6 if optimize else 0,
                   "wall_time_s": 1200, "on_exceed": "return_best"},
        "levels": [{
            "id": "o0", "surface": "config", "family": "packing",
            "targets": ["starting_artifact"], "allow_inactive": True,
            "fixed": {"optimizer": "OptoPrimeV2", "trainer": "PrioritySearch"},
            "iterations": 3, "num_candidates": 2,
        }],
    }


rows = []
t_all = time.time()
for seed in SEEDS:
    for arm, optimize in (("baseline", False), ("optimized", True)):
        spec = spec_for(seed, optimize=optimize)
        seed_everything(seed)
        reset_budget(make_budget(spec["budget"]))
        t0 = time.time()
        try:
            out = run_spec(spec)
            if out.get("errors"):
                raise RuntimeError("; ".join(out["errors"])[:180])
            rec = out["results"]["o0"]
            rows.append({"arm": arm, "seed": seed, "score": float(rec["score"]),
                         "artifact": str(rec.get("artifact", ""))[:160],
                         "wall_s": round(time.time() - t0, 1), "error": None})
        except Exception as exc:
            rows.append({"arm": arm, "seed": seed, "score": None, "artifact": "",
                         "wall_s": round(time.time() - t0, 1),
                         "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:150]}"})
        r = rows[-1]
        print(f"  seed{seed} {arm:10s} score={r['score']} ({r['wall_s']}s) {r['error'] or ''}",
              flush=True)

by = {}
for r in rows:
    if r["score"] is not None:
        by.setdefault(r["arm"], {})[r["seed"]] = r["score"]
paired = [by["optimized"][s] - by["baseline"][s]
          for s in SEEDS if s in by.get("optimized", {}) and s in by.get("baseline", {})]

summary = {
    "task": TASK, "evaluator": "deterministic (runs the candidate heuristic)",
    "certified_noise_sd": 0.0, "seeds": SEEDS, "rows": rows,
    "baseline_scores": by.get("baseline", {}), "optimized_scores": by.get("optimized", {}),
    "paired_deltas": paired,
    "paired_mean": st.mean(paired) if paired else None,
    "n_paired": len(paired),
    "improved_seeds": sum(1 for d in paired if d > 0),
    "wall_s": round(time.time() - t_all, 1),
}
print("\n" + json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2, default=str))
Path(__file__).with_name("probe_k_results.json").write_text(json.dumps(summary, indent=2, default=str))
print("\nWROTE probe_k_results.json")
