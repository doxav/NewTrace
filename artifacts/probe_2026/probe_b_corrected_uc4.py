"""Probe B — the corrected UC4 comparison.

The historical UC4 result scored the standard arm on level `o2_policy`
(FamilyPolicyLevel: mean over ALL families) and the recursive arm on `o3_prior`
(PriorInductionLevel: mean over the HELD-OUT families only), then reported the
difference as a recursive gain. Those are two different measurements.

Here BOTH arms are scored on the same level and therefore the same held-out
family, which is the only way the delta can mean "recursive optimization helped":

  standard  : a single cold `prior` level                (scored on holdout)
  recursive : `family_policy` -> warm `prior` transfer   (scored on the same holdout)

Equal candidate budget. The comparability gate added with D3 must report
comparable=True, otherwise the run is not a valid comparison and says so.
"""
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))

from recursive_opt_three_way import benchmark_uc, markdown_report

FAMILIES = {"gsm8k": ["internal:multiobjective_gsm8k"], "qasper": ["hf:qasper"]}
# Probe A showed `starting_artifact` is the only knob with an unconditional
# artifact->score path; `batch_design`/`batch_size` require inner_steps>0 and
# effects.check_field_effects correctly REFUSED them at inner_steps=0.
TARGETS = ["starting_artifact"]
ART_MENU = ["", "Answer directly.", "Plan step by step, then answer.",
            "Plan step by step, then verify the answer before replying.",
            "Use the provided context as evidence, reason briefly, then answer exactly."]
CONSTRAINTS = {"starting_artifact": ART_MENU}
FIXED = {"optimizer": "OptoPrimeV2", "trainer": "PrioritySearch",
         "trace_type": "internal", "credit_horizon": "step"}

MAX_EXAMPLES = int(os.environ.get("PROBE_MAX_EXAMPLES", "2"))
TOTAL_CANDIDATES = int(os.environ.get("PROBE_TOTAL_CANDIDATES", "4"))
NUM_CANDIDATES = int(os.environ.get("PROBE_NUM_CANDIDATES", "2"))
SEEDS = [int(s) for s in os.environ.get("PROBE_SEEDS", "0,1").split(",")]
WALL_S = int(os.environ.get("PROBE_WALL_S", "1800"))


def level(level_id, surface):
    return {"id": level_id, "surface": surface, "family": "*",
            "families": list(FAMILIES), "targets": TARGETS,
            "constraints": CONSTRAINTS, "fixed": dict(FIXED),
            "iterations": 2}


def spec(levels, *, warm, memory_root):
    return {"families": FAMILIES, "memory_root": memory_root, "reuse_priors": warm,
            "budget": {"wall_time_s": WALL_S, "optimizer_llm_calls": 60,
                       "eval_llm_calls": 200, "candidates": 32, "on_exceed": "return_best"},
            "tracebench": {"max_examples": MAX_EXAMPLES, "inner_steps": 0,
                           "timeout_seconds": 120},
            "scoring": {"clip": [-1.0, 1.0]},
            "levels": levels}


OUT = Path(__file__).with_name("probe_b_out")
t0 = time.time()
report = benchmark_uc(
    "UC4_corrected_same_holdout",
    # every arm is scored on o3_prior => the same held-out family
    initial=spec([level("o3_prior", "prior")], warm=False, memory_root=str(OUT / "mem_init")),
    standard=spec([level("o3_prior", "prior")], warm=False, memory_root=str(OUT / "mem_std")),
    recursive=spec([level("o2_policy", "family_policy"), level("o3_prior", "prior")],
                   warm=True, memory_root=str(OUT / "mem_rec")),
    output_root=str(OUT),
    total_candidates=TOTAL_CANDIDATES, num_candidates=NUM_CANDIDATES,
    optimizer_llm_calls=60, eval_llm_calls=200, wall_time_s=WALL_S,
    seeds=SEEDS,
    primary_level="o3_prior",   # SAME level for every arm -- that is the whole point
    notes="corrected UC4: both arms scored on the same held-out family",
)
print(markdown_report(report))
print("\ncomparability:", json.dumps(report["summary"]["comparability"], indent=2))
print("promotion    :", json.dumps(report["promotion"], indent=2)[:900])
print(f"\nwall={time.time()-t0:.0f}s")
