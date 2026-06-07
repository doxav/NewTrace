"""
recursive_opt.tracebench  —  Trace-Bench glue  (learn A/B/C on real problems, D)
================================================================================

Turns a Trace-Bench task id into the two callables the recursive stack needs:

    agent_fn(artifact_node, x) -> output           # for O0 ArtifactLevel
    inner_runner(cfg, family)  -> (score, feedback) # for O1+ MetaLevel

Real Trace-Bench task ids (verified in the repo):
    internal:code_param  internal:numeric_param  internal:multi_param
    internal:multiobjective_bbeh   internal:multiobjective_gsm8k
    llm4ad:online_bin_packing_local   llm4ad:circle_packing
    llm4ad:optimization_admissible_set   llm4ad:machine_learning_moon_lander
    llm4ad:optimization_job_shop_scheduling
    hf:GSM8K  hf:BBEH  hf:HotpotQA
    veribench:<name>   kernelbench:<level/prob>

Loading uses Trace-Bench's own registry when installed; otherwise a faithful
STUB scores configs deterministically so the recursive machinery is testable
offline (no API keys, no GPU).
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, List, Tuple

try:
    from trace_bench.registry import load_task_module, discover_tasks

    HAVE_TB = True
except Exception:
    HAVE_TB = False


def list_tasks(suite: str = None) -> List[str]:
    if not HAVE_TB:
        return [
            "internal:code_param",
            "internal:numeric_param",
            "llm4ad:online_bin_packing_local",
            "llm4ad:circle_packing",
            "hf:GSM8K",
            "hf:BBEH",
        ]
    return [spec.id for spec in discover_tasks("benchmarks", bench=suite)]


def make_agent_fn(task_id: str) -> Callable:
    """O0 agent: consume the trainable artifact, produce an answer for input x."""
    if HAVE_TB:
        mod = load_task_module(task_id, "benchmarks")
        # Trace-Bench task modules expose a callable agent/program; adapt it.
        program = getattr(mod, "program", None) or getattr(mod, "agent", None)

        def agent_fn(artifact, x):
            return program(artifact.data if hasattr(artifact, "data") else artifact, x)

        return agent_fn

    # STUB: artifact quality ~ how many "good" keywords it contains.
    def agent_fn(artifact, x):
        text = artifact.data if hasattr(artifact, "data") else str(artifact)
        return {"answer": f"{x}::{text[:24]}", "_artifact": text}

    return agent_fn


def make_inner_runner(task_id: str, n_tasks: int = 6) -> Callable:
    """O1 inner runner: run an inner optimization with `cfg` on `family`,
    return (held_out_score, feedback). Real mode delegates to Trace-Bench's
    evaluator; stub mode scores the config analytically + with noise so that
    *better designs win on average* (the property recursion must exploit)."""
    if HAVE_TB:
        mod = load_task_module(task_id, "benchmarks")
        evaluate = getattr(mod, "evaluate", None) or getattr(mod, "run_eval", None)

        def inner_runner(cfg, family):
            res = evaluate(cfg.to_dict(), n_tasks=n_tasks)
            score = float(res.get("score", res) if isinstance(res, dict) else res)
            fb = res.get("feedback", "") if isinstance(res, dict) else ""
            return score, f"[{task_id}] {fb}"

        return inner_runner

    def inner_runner(cfg, family):
        # Analytic stub: encode known-good design choices from the PDF.
        s = 0.5
        s += 0.10 if cfg.batch_design in ("failure_balanced", "curriculum") else 0.0
        s += 0.08 if 3 <= cfg.batch_size <= 8 else -0.05
        s += 0.06 if cfg.memory_policy in ("typed", "retrieval") else 0.0
        s += 0.05 if cfg.trace_type in ("hybrid", "otel") else 0.0
        s += 0.04 if cfg.optimizer in ("OptoPrime", "OptoPrimeMulti") else 0.0
        s += (
            0.05
            if cfg.trainer in ("BeamsearchAlgorithm", "UCBSearchAlgorithm")
            else 0.0
        )
        # deterministic per-config "noise" so search is non-trivial but reproducible
        h = int(hashlib.md5(str(cfg.to_dict()).encode()).hexdigest(), 16) % 1000
        s += (h / 1000 - 0.5) * 0.06
        s = max(0.0, min(1.0, s))
        fb = (
            f"[stub:{task_id}] design={cfg.batch_design}/bs={cfg.batch_size}/"
            f"mem={cfg.memory_policy}/trainer={cfg.trainer}. "
            f"{'good batch design' if cfg.batch_design!='random' else 'try failure_balanced batches'}; "
            f"{'memory helps here' if cfg.memory_policy=='typed' else 'enable typed memory'}."
        )
        return s, fb

    return inner_runner


def make_dataset(families: List[str], repeats: int = 4) -> dict:
    """Trainer dataset: inputs=families to optimize over, infos unused for meta."""
    inputs = [f for f in families for _ in range(repeats)]
    return {"inputs": inputs, "infos": [None] * len(inputs)}


# =========================================================================== #
# Evaluators for the CODE-improvement and CAPABILITY-synthesis examples.
# In real mode these call Trace-Bench's evaluator on the candidate; in stub
# mode they score analytically so that *better implementations win on average*
# (the property the optimizer must be able to climb). Stub keeps everything
# runnable offline; swap in real eval by installing Trace-Bench.
# =========================================================================== #
def make_code_evaluator(task_id: str, component: str):
    """Return evaluate(component_callable, family) -> (score, feedback) for
    improving the CODE of a library component (example_B).

    component in {"batch_design", "trace_summarizer"}.
    Real mode: run the candidate inside Trace-Bench's training loop on `task_id`
    and read held-out score. Stub mode: reward implementations that exhibit the
    properties the PDF identified as helpful, so code edits are climbable.
    """

    def evaluate(component_callable, family):
        if component == "batch_design":
            # Probe the candidate sampler on a synthetic pool with known "hard"
            # (failing) items at indices divisible by 3.
            pool_scores = [0.2 if i % 3 == 0 else 0.9 for i in range(12)]  # hard=low
            try:
                idx = list(component_callable(n=12, k=4))
            except Exception as e:
                return 0.0, f"[{component}] candidate raised {type(e).__name__}: {e}"
            idx = [i for i in idx if isinstance(i, int) and 0 <= i < 12][:4]
            if not idx:
                return (
                    0.0,
                    f"[{component}] returned no valid indices; must return k ints in [0,n).",
                )
            # Reward batches that (a) include hard items and (b) are diverse.
            hard = sum(1 for i in idx if pool_scores[i] < 0.5)
            diversity = len(set(idx)) / len(idx)
            score = 0.4 + 0.4 * (hard / len(idx)) + 0.2 * diversity
            fb = (
                f"[{component}@{task_id}] picked {idx}; hard_items={hard}/{len(idx)}; "
                f"diversity={diversity:.2f}. "
                f"{'good: targets failing items' if hard else 'tip: oversample failing/hard items (idx%3==0) instead of the first k'}."
            )
            return min(score, 1.0), fb

        if component == "trace_summarizer":
            sample = (
                "ERROR: AssertionError line 42 expected 7 got 5\n"
                "INFO: started\nDEBUG: x=1\nWARN: slow\nINFO: done\n" * 3
            )
            try:
                summary = str(component_callable(trace_text=sample))
            except Exception as e:
                return 0.0, f"[{component}] candidate raised {type(e).__name__}: {e}"
            preserved = ("AssertionError" in summary) + ("expected 7 got 5" in summary)
            concise = max(0.0, 1.0 - len(summary) / max(len(sample), 1))
            score = 0.3 + 0.45 * (preserved / 2) + 0.25 * concise
            fb = (
                f"[{component}@{task_id}] len={len(summary)} (src {len(sample)}); "
                f"error_evidence_preserved={preserved}/2; conciseness={concise:.2f}. "
                f"{'good: keeps the failing assertion' if preserved==2 else 'tip: KEEP the AssertionError + expected/got, DROP INFO/DEBUG noise'}."
            )
            return min(score, 1.0), fb

        return 0.0, f"[{component}] unknown component"

    return evaluate


def make_multiobjective_evaluator(task_ids, objectives, n_tasks: int = 4):
    """Return evaluate(capability_callable, family) -> (score_dict, feedback)
    for learning a NEW CAPABILITY under multiple objectives (example_C).

    objectives: dict like {"accuracy": "max", "cost": "min"}.
    The capability_callable is the (trainable) capability implementation; it is
    run on `task_ids` and produces an answer + a token/cost estimate. We return a
    per-objective score dict (consumed by opto.trainer.objectives / OptoPrimeMulti)
    plus a scalarized score and a directional feedback string.
    """

    def evaluate(capability_callable, family):
        accs, costs, notes = [], [], []
        for tid in task_ids:
            try:
                out = capability_callable(task=tid)
            except Exception as e:
                return (
                    {"accuracy": 0.0, "cost": 1.0},
                    f"[capability@{tid}] raised {type(e).__name__}: {e}",
                )
            spec_text = (
                str(out.get("answer", out)) if isinstance(out, dict) else str(out)
            )
            # STUB scoring: capability that (a) cites a verification step and
            # (b) is concise scores higher accuracy at lower cost.
            verifies = ("verify" in spec_text.lower()) or ("check" in spec_text.lower())
            decomposes = ("step" in spec_text.lower()) or ("plan" in spec_text.lower())
            acc = 0.45 + 0.30 * verifies + 0.20 * decomposes
            cost = 0.3 + 0.0009 * len(spec_text)  # longer => costlier
            accs.append(min(acc, 1.0))
            costs.append(min(cost, 1.0))
            notes.append(f"{tid}:acc={acc:.2f},cost={cost:.2f}")
        score = {"accuracy": sum(accs) / len(accs), "cost": sum(costs) / len(costs)}
        # scalarize for single-objective optimizers (max accuracy - cost)
        scalar = score["accuracy"] - 0.5 * score["cost"]
        fb = (
            "[multi-objective] "
            + "; ".join(notes)
            + f". aggregate acc={score['accuracy']:.2f} cost={score['cost']:.2f}. "
            + (
                "good: includes a verify/check step; "
                if score["accuracy"] > 0.7
                else "tip: ADD an explicit verify/check step to raise accuracy; "
            )
            + (
                "keep it terse to lower cost."
                if score["cost"] > 0.45
                else "cost is acceptable."
            )
        )
        return score, fb, scalar

    return evaluate
