"""Iteration 3 — does recursive (o2->o3) beat standard (cold prior) at equal budget?

HYPOTHESIS: a warm family-policy prior transfers to a held-out family better than a cold
prior trained directly on it, at equal candidate budget.
KILL CONDITION: the paired delta lies inside the certified noise floor.

Both arms are scored on the SAME level (`o3_prior`) and therefore the same held-out
family. Scoring `o2_policy` against `o3_prior` is what made UC4's "+0.163" an arithmetic
identity rather than an effect (assessment 5.2), and the D3 gate now rejects it outright.

Runs on deterministic surfaces so the noise floor is zero at the concurrency used, which
is what makes n=3 sufficient rather than the n>=115 an LLM-scored task would need.
"""
import json, os, statistics as st, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from opto.features.recursive_opt import tracebench as TB          # noqa: E402
from opto.features.recursive_opt.budget import make_budget, reset_budget  # noqa: E402
from opto.features.recursive_opt.experiments import seed_everything      # noqa: E402
from opto.features.recursive_opt.spec import run_spec, scored_task_ids   # noqa: E402

HYPOTHESIS = os.environ["PROBE_HYPOTHESIS"]
SEEDS = [0, 1, 2]
OUT = Path(__file__).with_name(f"probe_q_out_{HYPOTHESIS}")

FAMILIES = {
    "numeric": {"a": ["internal:multi_param"], "b": ["internal:code_param"]},
    "packing": {"a": ["llm4ad:optimization/admissible_set"],
                "b": ["llm4ad:optimization/online_bin_packing"]},
    "mixed":   {"a": ["internal:multi_param"],
                "b": ["llm4ad:optimization/admissible_set"]},
}[HYPOTHESIS]

MENU = ["", "Answer directly.", "Plan step by step, then verify the answer before replying."]
FIXED = {"optimizer": "OptoPrimeV2", "trainer": "PrioritySearch"}


def level(level_id, surface, iterations):
    return {"id": level_id, "surface": surface, "family": "*",
            "targets": ["starting_artifact"], "constraints": {"starting_artifact": MENU},
            "fixed": dict(FIXED), "iterations": iterations,
            "trainer_kwargs": {"num_candidates": 2}}


def spec_for(arm, seed):
    """Equal total budget: standard spends 2 iterations on o3; recursive splits 1+1."""
    levels = ([level("o3_prior", "prior", 2)] if arm == "standard"
              else [level("o2_policy", "family_policy", 1), level("o3_prior", "prior", 1)])
    return {
        "families": FAMILIES, "memory_root": str(OUT / f"{arm}_seed{seed}"),
        "reuse_priors": arm == "recursive",
        "tracebench": {"max_examples": 2, "inner_steps": 0, "timeout_seconds": 120,
                       "eval_temperature": 0.2, "eval_max_tokens": 1024,
                       "eval_request_timeout": 90},
        "budget": {"optimizer_llm_calls": 40, "eval_llm_calls": 200, "candidates": 4,
                   "wall_time_s": 1500, "on_exceed": "return_best"},
        "levels": levels,
    }


rows = []
t0 = time.time()
for seed in SEEDS:
    for arm in ("standard", "recursive"):
        spec = spec_for(arm, seed)
        seed_everything(seed)
        reset_budget(make_budget(spec["budget"]))
        started = time.time()
        try:
            out = run_spec(spec)
            if out.get("errors"):
                raise RuntimeError("; ".join(out["errors"])[:170])
            rec = out["results"]["o3_prior"]          # SAME level for both arms
            rows.append({"arm": arm, "seed": seed, "score": float(rec["score"]),
                         "artifact": str(rec.get("artifact", ""))[:120],
                         "wall_s": round(time.time() - started, 1), "error": None})
        except Exception as exc:
            rows.append({"arm": arm, "seed": seed, "score": None, "artifact": "",
                         "wall_s": round(time.time() - started, 1),
                         "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:140]}"})
        r = rows[-1]
        print(f"  [{HYPOTHESIS}] seed{seed} {arm:9s} score={r['score']} "
              f"({r['wall_s']}s) {r['error'] or ''}", flush=True)

by = {}
for r in rows:
    if r["score"] is not None:
        by.setdefault(r["arm"], {})[r["seed"]] = r["score"]
paired = [by["recursive"][s] - by["standard"][s]
          for s in SEEDS if s in by.get("recursive", {}) and s in by.get("standard", {})]
scored = scored_task_ids(level("o3_prior", "prior", 1), FAMILIES)
artifacts = {r["arm"]: {x["artifact"] for x in rows if x["arm"] == r["arm"]} for r in rows}

summary = {
    "hypothesis": HYPOTHESIS, "families": FAMILIES, "scored_task_ids": scored,
    "rows": rows, "paired_deltas": paired,
    "paired_mean": st.mean(paired) if paired else None,
    "paired_sd": st.pstdev(paired) if len(paired) > 1 else None,
    "n_paired": len(paired),
    "artifacts_differ": len(set().union(*artifacts.values())) > 1 if artifacts else False,
    "wall_s": round(time.time() - t0, 1),
}
print(f"\n[{HYPOTHESIS}] scored on (identical for both arms): {scored}")
print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2, default=str))
Path(__file__).with_name(f"probe_q_results_{HYPOTHESIS}.json").write_text(
    json.dumps(summary, indent=2, default=str))
print(f"WROTE probe_q_results_{HYPOTHESIS}.json")
