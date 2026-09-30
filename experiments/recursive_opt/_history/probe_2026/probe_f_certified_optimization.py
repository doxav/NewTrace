"""Probe F (deliverable B) — does optimization move a CERTIFIED surface?

B was specified as the corrected UC4: >=4 families, both arms on the same holdout.
Certification (Probe E) makes that impossible right now: of eight candidate tasks only
`internal:multiobjective_gsm8k` is usable, and a `prior` level needs at least two
families (guard D4), so the O2->O3 structure cannot be built at all.

Running it anyway on broken tasks is exactly the failure this whole analysis documents,
so B is reduced to the strongest question the certified instrument can actually answer:

    Does standard Trace optimization move the score by more than the instrument's
    own resolution limit?

That has never been established, and it is a precondition for any recursive claim.
Certified resolution at n=5 is 0.0410, so an honest win must exceed that.

Seeds run SEQUENTIALLY on purpose: `_BudgetGuard` counters are plain ints with no lock
and the global budget is module-level, so parallel arms would corrupt budget accounting.
"""
import json, statistics as st, time
from pathlib import Path

from opto.features.recursive_opt import tracebench as TB
from opto.features.recursive_opt.budget import make_budget, reset_budget
from opto.features.recursive_opt.experiments import seed_everything
from opto.features.recursive_opt.spec import run_spec

TASK = "internal:multiobjective_gsm8k"
SEEDS = [0, 1, 2, 3, 4]
RESOLUTION_AT_N5 = 0.0410           # from the Probe E certificate
ART_MENU = ["", "Answer directly.", "Plan step by step, then answer.",
            "Plan step by step, then verify the answer before replying.",
            "Use the provided context as evidence, reason briefly, then answer exactly."]
OUT = Path(__file__).with_name("probe_f_out")


def spec_for(seed, *, optimize):
    return {
        "families": {"reasoning": [TASK]},
        "memory_root": str(OUT / f"{'std' if optimize else 'init'}_seed{seed}"),
        "scoring": {"clip": [-2.0, 1.0]},
        "tracebench": {"max_examples": 4, "inner_steps": 0, "timeout_seconds": 120,
                       "eval_temperature": 0.2, "eval_max_tokens": 512,
                       "eval_request_timeout": 60},
        "budget": {"optimizer_llm_calls": 40 if optimize else 0,
                   "eval_llm_calls": 200, "candidates": 4 if optimize else 0,
                   "wall_time_s": 900, "on_exceed": "return_best"},
        "levels": [{
            "id": "o1", "surface": "config", "family": "reasoning",
            "targets": ["starting_artifact"],
            "constraints": {"starting_artifact": ART_MENU},
            "fixed": {"optimizer": "OptoPrimeV2", "trainer": "PrioritySearch"},
            "iterations": 2, "num_candidates": 2,
        }],
    }


rows = []
t_all = time.time()
for seed in SEEDS:
    for arm, optimize in (("initial", False), ("standard", True)):
        spec = spec_for(seed, optimize=optimize)
        seed_everything(seed)
        reset_budget(make_budget(spec["budget"]))
        t0 = time.time()
        try:
            out = run_spec(spec)
            if out.get("errors"):
                raise RuntimeError("; ".join(out["errors"])[:160])
            rec = out["results"]["o1"]
            rows.append({"arm": arm, "seed": seed, "score": float(rec["score"]),
                         "artifact": str(rec.get("artifact", ""))[:120],
                         "wall_s": round(time.time() - t0, 1), "error": None})
        except Exception as exc:
            rows.append({"arm": arm, "seed": seed, "score": None, "artifact": "",
                         "wall_s": round(time.time() - t0, 1),
                         "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:140]}"})
        r = rows[-1]
        print(f"  seed{seed} {arm:9s} score={r['score']} ({r['wall_s']}s) {r['error'] or ''}",
              flush=True)

by = {}
for r in rows:
    if r["score"] is not None:
        by.setdefault(r["arm"], {})[r["seed"]] = r["score"]
paired = [by["standard"][s] - by["initial"][s]
          for s in SEEDS if s in by.get("standard", {}) and s in by.get("initial", {})]

summary = {
    "task": TASK, "seeds": SEEDS, "rows": rows,
    "initial_mean": st.mean(by["initial"].values()) if by.get("initial") else None,
    "standard_mean": st.mean(by["standard"].values()) if by.get("standard") else None,
    "paired_deltas": paired,
    "paired_mean": st.mean(paired) if paired else None,
    "paired_sd": st.pstdev(paired) if len(paired) > 1 else None,
    "n_paired": len(paired),
    "resolution_at_n5": RESOLUTION_AT_N5,
    "artifacts_changed": sorted({r["artifact"] for r in rows if r["arm"] == "standard"}),
    "wall_s": round(time.time() - t_all, 1),
}
if paired:
    summary["exceeds_resolution"] = abs(summary["paired_mean"]) > RESOLUTION_AT_N5
print("\n" + json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2, default=str))
Path(__file__).with_name("probe_f_results.json").write_text(json.dumps(summary, indent=2, default=str))
print("\nWROTE probe_f_results.json")
