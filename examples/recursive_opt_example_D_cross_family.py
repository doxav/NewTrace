"""
EXAMPLE D — TRAINABLE O2 / O3 RECURSION ACROSS TRACE-BENCH FAMILIES
==================================================================
GOAL (section D): learn the A/B/C choices PER family (O2) and induce a
TRANSFERABLE prior validated on HELD-OUT families (O3) — as genuine *trainable
trace.Module levels*, not manual Python loops / majority vote.

WHAT CHANGED vs the earlier version
-----------------------------------
Previously O2 was a `max()` loop and O3 a majority vote (a static review flagged
this as "manual, not recursively trainable"). Now:

    O2  FamilyPolicyLevel   — ONE trainable node = a per-family config policy.
                              forward() runs every family and returns mean score
                              + a per-family breakdown naming the weakest family.
    O3  PriorInductionLevel — ONE trainable node = a single shared config, scored
                              ONLY on HELD-OUT families (true transfer objective).

Both are optimized by the SAME machinery as O0/O1 (an LLM optimizer rewrites the
node from the feedback). Offline we drive them with a few candidate proposals to
show the score is climbable; `--live` uses the real optimizer.

MEMORY (M2): every policy/prior is written to the artifact-lineage store, so you
can reconstruct the initial->final chain with scores (printed at the end).

TRACE-BENCH FAMILIES (2 families x 2 tasks):
    combinatorial : llm4ad:online_bin_packing_local , llm4ad:circle_packing
    reasoning_control : internal:multiobjective_gsm8k , internal:multi_param

HOW TO RUN
----------
    PYTHONPATH=/path/to/NewTrace python examples/recursive_opt_example_D_cross_family.py
    OPENAI_API_KEY=... PYTHONPATH=... python examples/...D....py --live
"""
import os, sys

from opto.features.recursive_opt import (
    FamilyPolicyLevel, PriorInductionLevel, RecursiveGuide, MemoryLite,
)
from opto.features.recursive_opt.tracebench import make_task_runner
from opto.features.recursive_opt.runmode import resolve_live, mode_banner

FAMILIES = {
    "combinatorial": ["llm4ad:online_bin_packing_local", "llm4ad:circle_packing"],
    "reasoning_control": ["internal:multiobjective_gsm8k", "internal:multi_param"],
}

# Candidate per-family policies the OFFLINE driver tries (live: the LLM writes these).
POLICY_CANDIDATES = [
    # uniform weak baseline
    ("combinatorial => batch_design=random, memory_policy=none, trainer=MinibatchAlgorithm, trace_type=internal\n"
     "reasoning_control => batch_design=random, memory_policy=none, trainer=MinibatchAlgorithm, trace_type=internal"),
    # family-tuned (should win: each family gets its preferred setup)
    ("combinatorial => batch_design=failure_balanced, memory_policy=typed, trainer=BeamsearchAlgorithm, trace_type=hybrid\n"
     "reasoning_control => batch_design=curriculum, memory_policy=retrieval, trainer=UCBSearchAlgorithm, trace_type=otel"),
]

PRIOR_CANDIDATES = [
    dict(batch_design="failure_balanced", memory_policy="typed", trainer="BeamsearchAlgorithm", trace_type="hybrid"),
    dict(batch_design="curriculum", memory_policy="retrieval", trainer="UCBSearchAlgorithm", trace_type="otel"),
]


def run_offline(mem):
    run_task = make_task_runner()
    guide = RecursiveGuide()

    # ---- O2: trainable per-family policy ---------------------------------- #
    print("O2  FamilyPolicyLevel — learn the per-family config policy")
    o2 = FamilyPolicyLevel(FAMILIES, run_task=run_task, memory=mem)
    best = (-1.0, None, None)
    for pol in POLICY_CANDIDATES:
        o2.propose(pol)
        out = o2.forward()
        s, _ = guide(None, out, None)
        print(f"    policy score={s:.3f}  per-family={ {k: round(v,3) for k,v in out.data['per_family'].items()} }")
        if s > best[0]:
            best = (s, pol, out.data["per_family"])
    o2.propose(best[1])
    print(f"    -> best O2 policy score={best[0]:.3f}\n")

    # ---- O3: trainable transferable prior, scored on HELD-OUT family ------ #
    print("O3  PriorInductionLevel — induce a prior, score it on a HELD-OUT family")
    train_f = {"combinatorial": FAMILIES["combinatorial"]}
    holdout_f = {"reasoning_control": FAMILIES["reasoning_control"]}
    o3 = PriorInductionLevel(train_f, holdout_f, run_task=run_task, memory=mem)
    best3 = (-1.0, None)
    for cand in PRIOR_CANDIDATES:
        o3.propose(**cand)
        t = o3.forward().data["score"]
        print(f"    prior {cand['batch_design']}/{cand['trainer']} -> held-out transfer={t:.3f}")
        if t > best3[0]:
            best3 = (t, cand)
    print(f"    -> best transferable prior: {best3[1]} (held-out transfer={best3[0]:.3f})")
    print("    (note: the combinatorial-tuned prior transfers POORLY to reasoning_control")
    print("     => the trainable O3 objective surfaces the no-universal-default result)\n")


def run_live(mem):
    from opto.features.recursive_opt.optimize import optimize, current_iterations
    run_task = make_task_runner()
    o2 = FamilyPolicyLevel(FAMILIES, run_task=run_task, memory=mem)
    iterations = current_iterations()
    # Trainer = PrioritySearch (or GEPA-Base), optimizer = OptoPrimeV2.
    optimize(
        o2,
        {"inputs": [None] * iterations, "infos": [None] * iterations},
        iterations=iterations,
    )
    print("O2 optimized policy:\n", o2._policy_node.data)


if __name__ == "__main__":
    live = resolve_live()
    print(mode_banner(live))
    print("=== D: TRAINABLE O2/O3 recursion across families ===\n")
    mem = MemoryLite(root="./mem_D")
    if live:
        run_live(mem)
    else:
        run_offline(mem)

    # ---- M2: show the artifact lineage we just built ---------------------- #
    print("M2  artifact history (every policy/prior version recorded this run):")
    for kind in ("policy", "prior"):
        hist = mem.artifact_history(kind=kind)
        if hist:
            print(f"    {kind}: " + " -> ".join(
                f"it{a.iteration}(score={a.score:.3f})" for a in hist))
            best = mem.best_artifact(kind=kind)
            print(f"      best {kind}: it{best.iteration} score={best.score:.3f}")
    print(f"\nmemory summary: {mem.summary()}")
