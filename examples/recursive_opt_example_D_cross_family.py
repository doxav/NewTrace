"""
EXAMPLE D — LEARN A/B/C ACROSS SEVERAL TRACE-BENCH FAMILIES (the full stack)
============================================================================
GOAL (section D): use SEVERAL Trace-Bench problems to learn the A/B/C choices
per problem FAMILY, then INDUCE a transferable prior that holds across families.
This is the "create priors for families of problems" objective and the highest
recursion level we exercise.

THE RECURSION STACK USED HERE
-----------------------------
    O0  task artifact            optimized inside ``make_inner_runner`` (per task)
    O1  MetaLevel                learns the A/B/C config for ONE family
                                 (averaged over that family's tasks)
    O2  pick best O1 config per family
    O3  induce a CROSS-FAMILY prior: any choice that wins in >= 2 families is
        promoted to a transferable default

So O1 answers "what setup is best for THIS family?", O2 compares families, and
O3 extracts "what setup tends to be best ACROSS families?". That O3 prior is
exactly what warm-starts a brand-new, unseen family (zero/one-shot transfer).

WHY THIS IS WORTH RUNNING
-------------------------
The March-2026 "hidden choices" result is that there is NO universal default for
starting artifact / horizon / batch size — the best choice is task-dependent.
This example MEASURES that: if the same config wins everywhere, you have found a
genuine prior; if not, the per-family configs tell you the design is
family-specific (which is itself the scientific result).

TRACE-BENCH FAMILIES (2 families x 2 tasks):
    combinatorial : llm4ad:online_bin_packing_local , llm4ad:circle_packing
    qa_reasoning  : hf:GSM8K , internal:multiobjective_bbeh

INTERPRETING THE OUTPUT
-----------------------
* "best A/B/C setup for family X" : the config (batch design / memory / trainer /
  trace type) that scored highest averaged over that family's tasks.
* "induced cross-family prior"    : the subset of choices that were best in BOTH
  families -> a transferable default. An EMPTY prior is a valid, informative
  result: it means the families need different setups (no universal recipe).
* "memory"                        : how many episodes were recorded and which
  per-family priors MemoryLite promoted (support >= 3 episodes).

HOW TO RUN
----------
    PYTHONPATH=/path/to/OpenTrace python example_D_cross_family.py
"""

from collections import defaultdict

from opto.features.recursive_opt import (
    LevelConfig,
    MetaLevel,
    RecursiveGuide,
    MemoryLite,
)
from opto.features.recursive_opt.tracebench import make_inner_runner

FAMILIES = {
    "combinatorial": ["llm4ad:online_bin_packing_local", "llm4ad:circle_packing"],
    "qa_reasoning": ["hf:GSM8K", "internal:multiobjective_bbeh"],
}

# The A/B/C search space the O1 level explores (kept small for stable search).
# Each row mixes section-A choices (batch/memory/trainer) with a B choice (trace).
SEARCH = [
    dict(
        batch_design="failure_balanced",
        memory_policy="typed",
        trainer="BeamsearchAlgorithm",
        trace_type="hybrid",
    ),  # A.2/A.4/A.7 + B.5
    dict(
        batch_design="curriculum",
        memory_policy="retrieval",
        trainer="UCBSearchAlgorithm",
        trace_type="otel",
    ),  # A.2/A.4/A.7 + B.4
    dict(
        batch_design="random",
        memory_policy="none",
        trainer="MinibatchAlgorithm",
        trace_type="internal",
    ),  # weak baseline
]


def optimize_family(family_name, tasks, mem):
    """O1+O2: find the A/B/C config that maximizes the mean score over `tasks`."""
    guide = RecursiveGuide()
    results = []
    for cand in SEARCH:
        base = LevelConfig(**cand)
        scores = []
        for task in tasks:
            level = MetaLevel(
                cfg=base,
                inner_runner=make_inner_runner(task),
                memory=mem,
                trainable_fields=tuple(cand.keys()),
            )
            out = level.forward(task)  # run inner optimization (O0->O1)
            s, _ = guide(task, out, None)
            scores.append(s)
        results.append((sum(scores) / len(scores), cand))
    return max(results, key=lambda r: r[0])  # best config for this family


def induce_cross_family_prior(per_family_best):
    """O3: a choice that is best in >= 2 families becomes a transferable prior."""
    votes = defaultdict(lambda: defaultdict(int))
    for _, cfg in per_family_best.values():
        for k, v in cfg.items():
            votes[k][v] += 1
    return {k: max(vs, key=vs.get) for k, vs in votes.items() if max(vs.values()) >= 2}


if __name__ == "__main__":
    print("=== D: learn A/B/C per family, then induce a cross-family prior ===\n")
    mem = MemoryLite(root="./mem_D")
    per_family_best = {}
    for fam, tasks in FAMILIES.items():
        score, cfg = optimize_family(fam, tasks, mem)
        per_family_best[fam] = (score, cfg)
        print(f"O1/O2  family '{fam}'")
        print(f"        tasks : {tasks}")
        print(f"        best  : score={score:.3f}  cfg={cfg}\n")

    prior = induce_cross_family_prior(per_family_best)
    print("O3     cross-family prior (choices that win in >= 2 families):")
    print(f"        {prior if prior else '<none — families need different setups>'}\n")
    print(f"memory : {mem.summary()}")
