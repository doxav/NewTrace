"""
EXAMPLE C — LEARN A NEW CAPABILITY FROM A SPEC + MULTIPLE OBJECTIVES
===================================================================
GOAL (section C, the key fix): given (1) a natural-language SPECIFICATION of a
capability we want the agent to acquire, and (2) one or more OBJECTIVES to
maximize/minimize, learn an implementation of that capability and validate it on
the Trace-Bench problems it will be judged on.

This is NOT "pick an existing capability". It is: synthesize/optimize a brand-new
capability artifact so it satisfies the spec while trading off competing
objectives (e.g. maximize accuracy AND minimize token cost).

WHAT IS BEING OPTIMIZED
-----------------------
A ``CapabilityArtifact``: a trainable text node holding the capability's
implementation (a SKILL-style procedure / policy). ``forward(family)`` runs the
current capability on the target Trace-Bench tasks and returns a PER-OBJECTIVE
score dict, e.g. {"accuracy": .., "cost": ..}. The optimizer rewrites the
capability text to push the objectives in the desired directions.

MULTI-OBJECTIVE SELECTION (real API)
------------------------------------
We score candidates as dicts and select with ``opto.trainer.objectives``:
    ObjectiveConfig(mode="pareto", minimize={"cost"})  +  select_best(...)
In LIVE mode the optimizer is ``OptoPrimeMulti`` (multi-candidate / Pareto-aware).

SUPPORTING CAPABILITIES (composed, not the headline):
    C.1  active knowledge building : MemoryLite warm-starts + records each trial
    C.2  agentic optimizer         : tools (run a subset eval / note) enrich feedback
    C.4  HITL                      : a human confirms the Pareto-front pick
    C.3  Tinker                    : reward can instead come from Tinker rollouts
                                     (swap the evaluator; stub shown in example earlier)

TRACE-BENCH PROBLEMS (2) the capability is tested on:
    hf:GSM8K                       (reasoning accuracy)
    internal:multiobjective_bbeh   (already exposes accuracy + cost objectives)

OBJECTIVES:
    accuracy -> maximize
    cost     -> minimize   (token / step budget)

HOW TO RUN
----------
    PYTHONPATH=/path/to/OpenTrace python example_C_learn_capability.py
    OPENAI_API_KEY=... PYTHONPATH=... python example_C_learn_capability.py --live
"""

import os, sys

from opto import trace
from opto.trace import node, Module
from opto.trainer.objectives import ObjectiveConfig, select_best, pareto_rank
from opto.features.recursive_opt import (
    RecursiveGuide,
    MemoryLite,
    AgenticOptimizer,
    default_optimizer_tools,
    HITLGate,
    auto_allow,
)
from opto.features.recursive_opt.tracebench import make_multiobjective_evaluator

# The capability we want the agent to LEARN, stated as a spec the optimizer reads.
CAPABILITY_SPEC = (
    "CAPABILITY: a reusable problem-solving procedure that, for each task, "
    "produces a concise plan, executes it, and VERIFIES the result before "
    "answering. It must generalise across the given problems."
)
OBJECTIVES = {"accuracy": "max", "cost": "min"}
PROBLEMS = ["hf:GSM8K", "internal:multiobjective_bbeh"]


@trace.model
class CapabilityArtifact(Module):
    """Trainable capability implementation (a SKILL-style text/policy).

    The artifact text is the trainable parameter. ``forward`` runs it on the
    target tasks through a multi-objective evaluator and returns a score DICT.
    """

    def __init__(self, seed_impl, evaluator, memory=None):
        super().__init__()
        self.impl = node(seed_impl, trainable=True, name="capability")
        self._eval = evaluator
        self._memory = memory

    def forward(self, family):
        # Build a callable capability from the (trainable) text and evaluate it.
        impl_text = self.impl.data if hasattr(self.impl, "data") else str(self.impl)

        def capability(task):
            # In real mode this would drive the agent with `impl_text` as its
            # procedure; the stub evaluator inspects the text directly.
            return {"answer": impl_text, "task": task}

        score_dict, feedback, scalar = self._eval(capability, family)
        if self._memory is not None:
            self._memory.record(
                level="capability",
                cfg={"len": len(impl_text)},
                family=str(family),
                score=scalar,
                feedback=feedback,
                metrics=score_dict,
            )
        return {"score": scalar, "feedback": feedback, "objectives": score_dict}


# Candidate capability implementations explored by the OFFLINE driver.
# (In LIVE mode the LLM optimizer writes these from the spec + feedback.)
CANDIDATE_IMPLS = [
    "Answer directly.",  # weak: no verify
    "Make a short plan, then answer.",  # decomposes only
    "Make a short plan; execute; then VERIFY/CHECK the answer "  # verify => high acc
    "against the question before responding. Keep it terse.",  # terse => low cost
    "Write an extremely detailed multi-paragraph chain-of-thought "  # verifies but costly
    "that re-derives and verifies everything at length.",
]


def learn_capability():
    evaluator = make_multiobjective_evaluator(PROBLEMS, OBJECTIVES)
    mem = MemoryLite(root="./mem_C_capability")

    # multi-objective selection config: maximize accuracy, minimize cost (Pareto)
    obj_cfg = ObjectiveConfig(
        mode="pareto",
        minimize={"cost"},
        weights={"accuracy": 1.0, "cost": 1.0},
        tie_break="weighted",
    )

    if "--live" in sys.argv and os.environ.get("OPENAI_API_KEY"):
        # ---- LIVE: optimizer rewrites the capability text under multi-objective ----
        from opto.optimizers import OptoPrimeMulti

        art = CapabilityArtifact(
            seed_impl="Answer the task.", evaluator=evaluator, memory=mem
        )
        # C.2: agentic optimizer with a quick subset-eval tool to gather evidence
        tools = default_optimizer_tools(
            memory=mem, run_subset=lambda: art.forward(PROBLEMS[0])["score"]
        )
        opt = AgenticOptimizer(
            art.parameters(),
            tools=tools,
            base_optimizer_cls=OptoPrimeMulti,
            objective=str(OBJECTIVES),
        )
        for step in range(4):
            out = art.forward(PROBLEMS[0])
            opt.zero_feedback()
            opt.backward(out, out["feedback"])  # multi-objective feedback -> rewrite
            opt.step()
        final = art.forward(PROBLEMS[0])
        return art.impl.data, final["objectives"], mem, None

    # ---- OFFLINE: evaluate candidate capabilities, pick the Pareto-best ----
    scored = []  # list of (score_dict, payload)
    for impl in CANDIDATE_IMPLS:
        art = CapabilityArtifact(seed_impl=impl, evaluator=evaluator, memory=mem)
        # average objectives across the 2 target problems
        agg = {"accuracy": 0.0, "cost": 0.0}
        for p in PROBLEMS:
            objs = art.forward(p)["objectives"]
            for k in agg:
                agg[k] += objs[k] / len(PROBLEMS)
        scored.append((agg, impl))
        print(f"  acc={agg['accuracy']:.2f} cost={agg['cost']:.2f}  <- {impl[:48]}...")

    norm = [
        {"accuracy": s["accuracy"], "cost": -s["cost"]} for s, _ in scored
    ]  # cost negated -> higher=better
    ranks = pareto_rank(norm, metrics=("accuracy", "cost"))
    front = [impl for (s, impl), r in zip(scored, ranks) if r == 0]
    best_idx = select_best(scored, obj_cfg)
    best_dict, best_impl = scored[best_idx]

    # C.4: human confirms the Pareto-front selection (auto-allow stub here)
    gate = HITLGate(approver=auto_allow, threshold=0.0, min_support=1)
    gate.decide(
        old_score=0.0,
        new_score=best_dict["accuracy"] - 0.5 * best_dict["cost"],
        support=len(PROBLEMS),
        diff=best_impl,
        evidence=str(best_dict),
        changed_fields=("capability",),
    )
    return best_impl, best_dict, mem, (front, gate)


if __name__ == "__main__":
    live = "--live" in sys.argv and os.environ.get("OPENAI_API_KEY")
    print(
        f"=== C: learning a NEW CAPABILITY from spec + objectives "
        f"({'LIVE' if live else 'offline stub'}) ==="
    )
    print(f"  spec      : {CAPABILITY_SPEC}")
    print(f"  objectives: {OBJECTIVES}")
    print(f"  problems  : {PROBLEMS}\n")
    impl, objs, mem, extra = learn_capability()
    print(f"\n  LEARNED CAPABILITY: {impl}")
    print(
        f"  objectives achieved: accuracy={objs['accuracy']:.2f}  cost={objs['cost']:.2f}"
    )
    if extra:
        front, gate = extra
        print(
            f"  Pareto front had {len(front)} candidate(s); HITL decision: {gate.audit[-1]['via']}"
        )
    print(f"  memory: {mem.summary()['priors']}")
