"""
EXAMPLE E — DECLARATIVE CONTROL PLANE (one dict) + PRIOR/TOOL TRANSFER
=====================================================================
GOAL: drive the whole recursive stack from a single ``RecursiveSpec`` dict
(depth = the order of ``levels``; plus budget / constraints / allowed-targets),
and *reuse* learned priors and tools by (family, level) so a new project starts
warm instead of cold. This is a thin, transparent compiler over the EXISTING
levels/memory/budget — nothing in Trace core is touched.

WHAT THIS SHOWS
---------------
* ``levels`` ordering IS the recursion depth (O1 config -> O2 family policy).
* ``targets`` -> trainable fields; ``fixed`` -> seed/frozen config; ``constraints``
  -> validated allowed values; ``budget`` -> the global RecursiveOptBudget.
* ``tracebench`` -> real benchmark adapter bounds; ``scoring`` -> normalized
  cross-family interpretation; ``prior_promotion`` -> M1 to M3 capitalization.
* ``reuse_priors`` -> warm-start a config level from the promoted family prior and
  load reusable tools recorded for that family (transfer learning).

RUN
---
    python examples/recursive_opt_example_E_declarative_spec.py            # offline structure + transfer (no LLM)
    python examples/recursive_opt_example_E_declarative_spec.py --live     # full optimized run (needs adapter + key)
"""
from opto.features.recursive_opt import (
    make_level_spec,
    MemoryLite, best_config_from, validate_spec, compile_level, reuse_priors, run_spec,
)
from opto.features.recursive_opt.tracebench import (
    configure_tracebench_adapter,
)
from opto.features.recursive_opt.runmode import resolve_live, mode_banner

FAMILIES = {
    "optimization_control": [
        "llm4ad:online_bin_packing_local",
        "llm4ad:optimization_admissible_set",
    ],
    "reasoning_control": ["internal:multi_param", "internal:numeric_param"],
}


# A single declarative control plane: depth = order of `levels`.
SPEC = {
    "families": FAMILIES,
    "tracebench": {
        "max_examples": 2,
        "inner_steps": 1,
        "inner_candidates": 1,
        "timeout_seconds": 5,
        "allowed_inner_trainers": ["MinibatchAlgorithm"],
        "eval_kwargs": {"n_train": 2, "n_val": 0},
    },
    "scoring": {
        "mode": "relative_delta",
        "baseline": "default_config",
        "clip": [-1.0, 1.0],
        "report_raw": True,
    },
    "prior_promotion": {"enabled": True, "min_support": 2},
    "budget": {"optimizer_llm_calls": 24, "eval_llm_calls": 80,
               "candidates": 12, "wall_time_s": 240,
               "on_exceed": "return_best"},
    "memory_root": "./mem_E",
    "reuse_priors": True,
    "levels": [
        # Built via make_level_spec: keyword args cannot be duplicated, so the
        # duplicate-"constraints" bug (which silently DROPPED the
        # starting_artifact menu) is impossible by construction.
        make_level_spec(
            id="o1_setup", surface="config", family="optimization_control",
            targets=["starting_artifact", "batch_size", "trainer"],
            fixed={"optimizer": "OptoPrime", "guide": "LLMJudge",
                   "trace_type": "internal", "trainer": "MinibatchAlgorithm"},
            constraints={
                "starting_artifact": ["",  # bundle-default control arm
                    "Answer directly.", "Plan step by step, then answer.",
                    "Plan step by step, then verify the answer before replying."],
                # batch_design is declared for VALIDATION of values; it is not a
                # target (its causal path is inactive in the eval-only adapter).
                "batch_design": ["random", "failure_balanced", "curriculum", "diversity"],
            },
            iterations=4,
            tools=["trace_search", "run_subset", "note"],
        ),
        {
            "id": "o2_policy", "surface": "family_policy", "family": "*",
            "targets": ["starting_artifact", "trainer"],
            "iterations": 3, "depends_on": ["o1_setup"], "allow_unplumbed": True,
        },
        {
            "id": "o3_prior", "surface": "prior", "family": "*",
            "targets": ["starting_artifact", "trainer"],
            "iterations": 2, "depends_on": ["o2_policy"],
        },
    ],
}


def run_live():
    out = run_spec(SPEC)
    print("\n=== run_spec results (depth = level order) ===")
    for lid, r in out["results"].items():
        print(f"  [{lid}] surface={r['surface']} score={r['score']:.3f} "
              f"reused_prior={r['reused_prior']} tools={r['tools']}")
        print(f"        artifact: {r['artifact'][:80].splitlines()[0] if r['artifact'] else ''}")
    s = out["memory"].summary()
    print(f"  memory: episodes={s['episodes']} artifacts={s['artifacts']} priors={list(s['priors'])}")

    # Transfer: a NEW project reusing what we just learned for the same family.
    print("\n=== transfer: re-run with reuse_priors (warm start) ===")
    out2 = run_spec({**SPEC, "memory_root": "./mem_E"})
    print("  o1 reused_prior:", out2["results"]["o1_setup"]["reused_prior"],
          "| tools carried:", out2["results"]["o1_setup"]["tools"])


def run_offline():
    # No-LLM: show the control plane compiles, then demonstrate transfer reuse.
    validate_spec(SPEC)
    mem = MemoryLite(root="./mem_E")
    print("\n=== compiled levels (ordered == depth) ===")
    for ls in SPEC["levels"]:
        level = compile_level(ls, mem, FAMILIES)
        targets = ls.get("targets", [])
        print(f"  [{ls['id']}] surface={ls['surface']} -> {type(level).__name__} targets={targets}")

    print("\n=== transfer demo: warm-start a config level from a family prior ===")
    for _ in range(3):  # >=3 episodes -> promotes a FamilyPrior (active knowledge)
        mem.record(level="O1", cfg={"batch_design": "failure_balanced", "batch_size": 4},
                   family="optimization_control", score=0.9, feedback="good config")
    weak = compile_level(
        {"id": "o1_setup", "surface": "config", "family": "optimization_control",
         "targets": ["batch_design", "batch_size"],
         "fixed": {"batch_design": "random", "batch_size": 1}}, mem, FAMILIES)
    print("  before reuse:", best_config_from(weak).replace("\n", ", "))
    info = reuse_priors(mem, weak, {"surface": "config", "family": "optimization_control"})
    print("  after  reuse:", best_config_from(weak).replace("\n", ", "),
          f"(used_prior={info['used_prior']})")
    print("\n  (full optimized run with the Trainer happens under --live)")


if __name__ == "__main__":
    live = resolve_live()
    adapter_cfg = SPEC["tracebench"] if live else {**SPEC["tracebench"], "inner_steps": 0}
    configure_tracebench_adapter(adapter_cfg, require=True)
    print(mode_banner(live))
    print("=== E: declarative control-plane spec ===")
    if live:
        run_live()
    else:
        run_offline()
