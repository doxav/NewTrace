"""
EXAMPLE A — LEARN THE BEST *TRACE SETUP* FOR A PROBLEM FAMILY
=============================================================
GOAL (section A): given a family of problems, learn which EXISTING optimization
components to use and how to configure them — the "selection/config" surface.

WHAT IS BEING OPTIMIZED
-----------------------
An O1 ``MetaLevel`` whose single trainable parameter is a small config (text)
over existing components. The optimizer SELECTS + CONFIGURES; it does not change
any component's code. (To change a component's *code*, see example_B.)

This example covers >= 2 of the A elements:
    A.2  batch size & design   (random | failure_balanced | curriculum)
    A.4  memory                (none | typed | retrieval)
    A.7  trainer               (MinibatchAlgorithm | Beamsearch | UCBSearch)

WHY THE CONFIG LOOKS LIKE "starting_artifact / initial_knowledge / batch_size..."
---------------------------------------------------------------------------------
``LevelConfig`` (see levels.py) is intentionally a flat, low-dimensional record:
one knob per A-element. Low dimensionality is what makes the O1 search stable
(the classic credit-assignment risk of meta-optimization). The fields are plain
strings/ints, NOT enums, so you can register a NEW trainer or batch design just
by adding its class and allowing its name here — no change to the recursion code.

TRACE-BENCH PROBLEMS (2):
    llm4ad:online_bin_packing_local
    llm4ad:circle_packing

HOW TO RUN
----------
    PYTHONPATH=/path/to/OpenTrace python example_A_learn_setup.py            # offline stub
    OPENAI_API_KEY=... PYTHONPATH=... python example_A_learn_setup.py --live # real LLM optimizer

Offline mode drives the loop by hand (no LLM) so the machinery is testable.
Live mode replaces the hand driver with the real ``opto.trainer.train`` facade
(shown in ``run_live``); the LLM optimizer then proposes configs itself.
"""

import os, sys

from opto.features.recursive_opt import (
    LevelConfig,
    MetaLevel,
    RecursiveGuide,
    MemoryLite,
    best_config_from,
)
from opto.features.recursive_opt.tracebench import make_inner_runner, make_dataset

PROBLEMS = ["llm4ad:online_bin_packing_local", "llm4ad:circle_packing"]

# The search space over the A elements we are learning. Each entry is a full
# setting of the trainable fields; the optimizer (or, offline, this list) explores them.
CANDIDATES = [
    dict(
        batch_size=4,
        batch_design="failure_balanced",
        memory_policy="typed",
        trainer="BeamsearchAlgorithm",
    ),
    dict(
        batch_size=8,
        batch_design="curriculum",
        memory_policy="retrieval",
        trainer="UCBSearchAlgorithm",
    ),
    dict(
        batch_size=1,
        batch_design="random",
        memory_policy="none",
        trainer="MinibatchAlgorithm",
    ),  # weak baseline
]


def _int_env(name: str, default: int) -> int:
    """Read a positive integer environment override for live demo settings."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def build_level(problem, mem):
    """An O1 MetaLevel: its trainable config selects/configures A.2/A.4/A.7."""
    base = LevelConfig(
        batch_size=1,
        batch_design="random",
        memory_policy="none",
        trainer="MinibatchAlgorithm",
    )
    return MetaLevel(
        cfg=base,
        inner_runner=make_inner_runner(problem),  # runs the inner optimization
        # only these four fields are trainable -> tiny, stable search space:
        trainable_fields=("batch_size", "batch_design", "memory_policy", "trainer"),
        memory=mem,  # A.4 / C.1 active knowledge
    )


def run_offline(problem):
    """Hand-driven O1 search (no LLM): evaluate each candidate config, keep best."""
    mem = MemoryLite(root=f"./mem_A_{problem.split(':')[-1]}")
    level = build_level(problem, mem)
    guide = RecursiveGuide()
    best = (-1.0, None)
    for cand in CANDIDATES:
        level.propose(**cand)  # write the candidate into the config node
        out = level.forward(problem)  # run the inner optimization with that config
        score, feedback = guide(problem, out, None)
        print(
            f"    {cand['trainer']:<20} bs={cand['batch_size']} "
            f"{cand['batch_design']:<16} -> score={score:.3f}"
        )
        if score > best[0]:
            best = (score, dict(cand))
    return best, mem


def run_live(problem):
    """Real LLM-driven meta-optimization: the optimizer PROPOSES configs itself."""
    from opto.trainer import train  # GitHub/PR#73 trainer facade

    mem = MemoryLite(root=f"./mem_A_{problem.split(':')[-1]}")
    level = build_level(problem, mem)
    train(
        model=level,
        train_dataset=make_dataset([problem], repeats=8),
        algorithm="BeamsearchAlgorithm",  # A.7 outer search algorithm (exact class name)
        optimizer="OptoPrime",  # A.5 LLM optimizer (needs a key)
        guide=RecursiveGuide(),  # A.6 bridges level output -> (score,fb)
        batch_size=_int_env("RECURSIVE_OPT_LIVE_BATCH_SIZE", 3),
        num_epochs=_int_env("RECURSIVE_OPT_LIVE_NUM_EPOCHS", 4),
        num_threads=_int_env("RECURSIVE_OPT_LIVE_NUM_THREADS", 4),
        beam_width=_int_env("RECURSIVE_OPT_LIVE_BEAM_WIDTH", 3),
        num_proposals=_int_env("RECURSIVE_OPT_LIVE_NUM_PROPOSALS", 4),
        max_depth=_int_env("RECURSIVE_OPT_LIVE_MAX_DEPTH", 2),
        validation_dataset_size=_int_env("RECURSIVE_OPT_LIVE_VALIDATION_SIZE", 5),
    )
    return best_config_from(level), mem


if __name__ == "__main__":
    live = "--live" in sys.argv and os.environ.get("OPENAI_API_KEY")
    for p in PROBLEMS:
        print(
            f"\n=== A: learning best setup for {p} ({'LIVE' if live else 'offline stub'}) ==="
        )
        if live:
            cfg, mem = run_live(p)
            print(f"  optimized config:\n{cfg}")
        else:
            (score, cfg), mem = run_offline(p)
            print(f"  BEST: score={score:.3f}  cfg={cfg}")
        print(f"  memory (M3 priors): {mem.summary()['priors']}")
