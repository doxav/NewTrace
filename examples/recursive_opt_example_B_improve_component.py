"""
EXAMPLE B — IMPROVE THE *CODE* OF A LIBRARY COMPONENT  (not just select it)
===========================================================================
GOAL (section B, the key fix): recursive_opt must be able to PROPOSE AN IMPROVED
VERSION of an existing Trace component — a new/better Trainer hot-path, batch
design, or trace representation — by REWRITING ITS SOURCE CODE, not by picking
from a fixed menu.

HOW THIS WORKS (the mechanism)
------------------------------
Trace can make a function's SOURCE CODE a trainable parameter:

    @trace.bundle(trainable=True)
    def batch_design(self, n, k): ...     # <- the body of this function IS the param

We wrap such a function in a ``CodeArtifactLevel``. ``forward(family)`` runs the
CURRENT code on Trace-Bench inner problems (via the component's evaluator) and
returns {"score","feedback"}. An LLM optimizer (OptoPrime) then reads the
execution trace + feedback and writes a BETTER IMPLEMENTATION of the function.
That is genuine component improvement / invention, not selection.

Two components are improved here (>= 2 elements):
    (1) batch_design        -> a NEW batching strategy (B.2 "other trainers": the
                               trainer's sampling policy is part of the trainer)
    (2) trace_summarizer    -> a NEW trace representation (B.5 "traces": how an
                               episode trace is compressed into feedback)
The SAME pattern targets a full Trainer.update() or an OTEL->TGJ projection.

WHY THIS IS THE RIGHT DESIGN FOR FUTURE COMPONENTS
--------------------------------------------------
Because the trainable object is *code*, the optimizer is not limited to existing
classes. Give it a baseline implementation + a feedback signal and it can evolve
a component that did not exist before. This is the answer to "can we optimize the
actual Trainer/batch-design/Trace classes AND future ones?": yes — via the
CODE/IMPLEMENTATION surface (CodeArtifactLevel), complementary to the
SELECTION/CONFIG surface (LevelConfig/MetaLevel) used in recursive_opt_example_A.

TRACE-BENCH PROBLEMS (2):
    llm4ad:online_bin_packing_local   (batching matters: many similar items)
    internal:code_param                 (code-surface validator label; synthetic trace summary)

HOW TO RUN
----------
    PYTHONPATH=/path/to/OpenTrace python recursive_opt_example_B_improve_component.py
    OPENAI_API_KEY=... PYTHONPATH=... python recursive_opt_example_B_improve_component.py --live
"""

import os, sys

from opto import trace
from opto.features.recursive_opt import (
    ComponentSpec,
    CodeArtifactLevel,
    RecursiveGuide,
    MemoryLite,
)
from opto.features.recursive_opt.tracebench import make_code_evaluator
from opto.features.recursive_opt import inspect_utils

# Live mode is resolved loudly in __main__ (see bottom). Safe default for imports.
_LIVE = False

BATCH_DESIGN_GUIDANCE = (
    "Validation uses a pool of n=12, k=4 where hard/failing items are exactly "
    "indices divisible by 3: [0, 3, 6, 9]. A good sampler returns k unique valid "
    "indices, selects hard/failing items first, then fills remaining slots diversely."
)


# --------------------------------------------------------------------------- #
# BASELINE IMPLEMENTATIONS (their SOURCE is what gets optimized).
# They must be module-level functions taking ``self`` first (Trace convention),
# and importable from a file so Trace can read their source via inspect.
# --------------------------------------------------------------------------- #
def batch_design_baseline(self, n, k):
    """Pick which task indices go in a training batch. BASELINE = first k.

    A good rewrite should oversample HARD/FAILING items and keep the batch
    diverse, instead of blindly returning range(k). In this demo validator,
    hard/failing items are indices divisible by 3."""
    return list(range(k))


def trace_summarizer_baseline(self, trace_text):
    """Compress an episode trace into optimizer feedback. BASELINE = truncate.

    A good rewrite should PRESERVE error evidence (failing assertions,
    expected/got) and DROP noise (INFO/DEBUG), staying concise."""
    return trace_text[:200]


# --------------------------------------------------------------------------- #
# A "manually improved" candidate, used only by the OFFLINE driver to prove the
# score is climbable (in LIVE mode the LLM optimizer writes these itself).
# --------------------------------------------------------------------------- #
def batch_design_improved(self, n, k):
    """Oversample hard items (here: indices divisible by 3) then fill diversely."""
    hard = [i for i in range(n) if i % 3 == 0]
    rest = [i for i in range(n) if i % 3 != 0]
    picked = (hard + rest)[:k]
    return picked


def trace_summarizer_improved(self, trace_text):
    """Keep only error-bearing lines; drop INFO/DEBUG; cap length."""
    keep = [
        ln
        for ln in trace_text.splitlines()
        if ("ERROR" in ln or "expected" in ln or "Assertion" in ln)
    ]
    return ("\n".join(dict.fromkeys(keep)))[:160]  # dedupe + cap


def improve_component(problem, name, baseline, improved, objective):
    """Run one component-improvement experiment."""
    mem = MemoryLite(root=f"./mem_B_{name}")
    spec = ComponentSpec(
        name=name,
        baseline=baseline,
        evaluate=make_code_evaluator(
            problem, name
        ),  # runs candidate code on TB problem
        objective=objective,
    )
    level = CodeArtifactLevel(spec, memory=mem)
    guide = RecursiveGuide()

    # 1) score the BASELINE code
    out = level.forward(problem)
    base_score, base_fb = guide(problem, out, None)
    print(f"  [{name}] baseline score={base_score:.3f}")
    print(f"           feedback: {base_fb}")

    if _LIVE:
        # 2-LIVE) a Trainer (PrioritySearch / GEPA-Base) drives the loop and the
        # configured optimizer (OptoPrimeV2) rewrites the function body. No
        # hand-rolled backward()/step() here.
        from opto.features.recursive_opt.optimize import optimize, current_iterations
        from opto.features.recursive_opt.tracebench import make_dataset

        initial_code = level.current_code()
        iterations = current_iterations()
        optimize(
            level,
            make_dataset([problem], repeats=iterations),
            guide=guide,
            iterations=iterations,
        )
        out = level.forward(problem)
        score, _ = guide(problem, out, None)
        print(f"  [{name}] LIVE optimized score={score:.3f}  (Δ={score-base_score:+.3f})")
        print(f"  [{name}] code diff (initial -> optimized):")
        print(inspect_utils.code_diff(initial_code, level.current_code(), name=name))
        return base_score, score
    else:
        # 2-OFFLINE) install a hand-written improved body to show the score climbs
        level._impl = trace.bundle(trainable=True)(improved)
        out = level.forward(problem)
        new_score, new_fb = guide(problem, out, None)
        print(
            f"  [{name}] improved score={new_score:.3f}  (Δ={new_score-base_score:+.3f})"
        )
        print(f"           feedback: {new_fb}")
        return base_score, new_score


if __name__ == "__main__":
    from opto.features.recursive_opt.runmode import resolve_live, mode_banner

    _LIVE = resolve_live()  # raises if --live without a key (no silent fallback)
    print(mode_banner(_LIVE))
    print("\n=== B: improving COMPONENT CODE (rewrite, not select) ===")
    print(
        "\n-- component 1: batch_design  (problem: llm4ad:online_bin_packing_local) --"
    )
    improve_component(
        "llm4ad:online_bin_packing_local",
        "batch_design",
        batch_design_baseline,
        batch_design_improved,
        objective=(
            "maximize held-out pass rate by sampling failing/hard items; keep batch "
            f"diverse. {BATCH_DESIGN_GUIDANCE}"
        ),
    )

    print("\n-- component 2: trace_summarizer  (problem: internal:code_param) --")
    improve_component(
        "internal:code_param",
        "trace_summarizer",
        trace_summarizer_baseline,
        trace_summarizer_improved,
        objective="preserve error evidence (failing assertion + expected/got), drop INFO/DEBUG, stay concise",
    )
