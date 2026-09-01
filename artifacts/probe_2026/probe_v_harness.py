"""Probe V — Tier 2: does the PRODUCTION harness reproduce probe U's W2 result?

Pre-registered in PREREG_W2_routing.md. One (arm, target, budget, seed) run per process
invocation so folds can be parallelised without sharing a budget guard or a memory root.

    python probe_v_harness.py <arm> <target-index> <budget> <seed>

arm            : "standard" | "recursive"
target-index   : index into TARGETS (leave-one-out; sources are the others)
budget         : num_candidates spent ON THE TARGET level (o3_prior) - equal for both arms
seed           : integer

Both arms are scored on the SAME level (`o3_prior`) and, because `scored_task_ids` for the
`prior` surface returns the HELD-OUT families only, on the SAME single target task. The
recursive arm additionally pays a meta level (`o2_policy`) restricted via `families:` to the
SOURCE families only - probe_q's spec let the family_policy level see every family, which would
leak the target into meta-training.

Emits one JSON line to stdout with the score, the final artifact, budget usage (the measured
`c_meta`), and the scored task ids.
"""
import json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))
for line in (HERE.parents[2] / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
os.environ.setdefault("TRACE_LITELLM_MODEL", "openrouter/deepseek/deepseek-v4-flash-0731")
os.environ.setdefault("RECURSIVE_OPT_TRACEBENCH_MODEL", os.environ["TRACE_LITELLM_MODEL"])

from opto.features.recursive_opt.budget import make_budget, reset_budget, current_budget  # noqa: E402
from opto.features.recursive_opt.experiments import seed_everything   # noqa: E402
from opto.features.recursive_opt.spec import run_spec, scored_task_ids  # noqa: E402
from probe_t_routing_menu import MENU, ROUTING                        # noqa: E402

TARGETS = [t for t in ROUTING if not t.endswith("ovrp_construct")]   # dedup family (probe T)
MENU_LIST = [""] + list(MENU.values())     # "" = bundle default, the legal control arm
FIXED = {"optimizer": "OptoPrimeV2", "trainer": "PrioritySearch"}
OUT = HERE.parent / "probe_v_out"


def level(level_id, surface, budget, families=None):
    """`budget` = candidate proposals spent on this level.

    One optimizer UPDATE happens per `iterations` step (PrioritySearch logs
    `Update/n_iters: 0` at iterations=1, i.e. no LLM proposal at all), so the budget is
    spent as iterations with one candidate each rather than as num_candidates.
    """
    lv = {"id": level_id, "surface": surface, "family": "*",
          "targets": ["starting_artifact"],
          "constraints": {"starting_artifact": MENU_LIST},
          "fixed": dict(FIXED), "iterations": int(budget),
          "trainer_kwargs": {"num_candidates": 1}}
    if families is not None:
        lv["families"] = list(families)
    return lv


def main():
    arm, ti, budget, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    target = TARGETS[ti]
    sources = [t for t in TARGETS if t != target]
    # names[0] = sources, names[1:] = holdout -> `prior` is scored on the target only.
    families = {"src": sources, "tgt": [target]}
    tag = f"{arm}_t{ti}_b{budget}_s{seed}"

    levels = ([level("o3_prior", "prior", budget)] if arm == "standard"
              else [level("o2_policy", "family_policy", budget, families=["src"]),
                    level("o3_prior", "prior", budget)])
    spec = {
        "families": families, "memory_root": str(OUT / tag),
        "reuse_priors": arm == "recursive",
        "tracebench": {"max_examples": 2, "inner_steps": 0, "timeout_seconds": 180,
                       "eval_temperature": 0.2, "eval_max_tokens": 1024,
                       "eval_request_timeout": 120},
        "budget": {"optimizer_llm_calls": 60, "eval_llm_calls": 400,
                   "candidates": 4 * (budget + 1), "wall_time_s": 2400,
                   "on_exceed": "return_best"},
        "levels": levels,
    }
    seed_everything(seed)
    reset_budget(make_budget(spec["budget"]))
    t0 = time.time()
    row = {"arm": arm, "target": target, "sources": sources, "budget": budget, "seed": seed,
           "scored_task_ids": scored_task_ids(level("o3_prior", "prior", budget), families)}
    try:
        out = run_spec(spec)
        if out.get("errors"):
            raise RuntimeError("; ".join(map(str, out["errors"]))[:220])
        rec = out["results"]["o3_prior"]
        row.update(score=float(rec["score"]), artifact=str(rec.get("artifact", "")),
                   reused_prior=rec.get("reused_prior"),
                   meta_score=(float(out["results"]["o2_policy"]["score"])
                               if "o2_policy" in out["results"] else None),
                   meta_artifact=(str(out["results"]["o2_policy"].get("artifact", ""))
                                  if "o2_policy" in out["results"] else None),
                   budget_report=(current_budget().summary() if current_budget() else None),
                   error=None)
    except Exception as exc:
        row.update(score=None, artifact="", error=f"{type(exc).__name__}: "
                                                  f"{str(exc).splitlines()[0][:200]}")
    row["wall_s"] = round(time.time() - t0, 1)
    print("RESULT " + json.dumps(row, default=str), flush=True)


if __name__ == "__main__":
    main()
