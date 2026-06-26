"""Three-way benchmark helper for the recursive_opt use-case notebook.

Compares, per use case, at EQUAL TOTAL BUDGET:
  A0  initial      - score the artifact/config with no optimizer updates
  A1  standard     - standard Trace optimization (one level, no prior carry), budget N
  A2  recursive    - recursive/multi-level OR specialized sub-optimizer, same total budget N

Design (see recursive_opt_three_way_design.md):
  * ONE runner + ONE result type. No 4-dataclass framework.
  * Fairness = a DETERMINISTIC per-level candidate split that sums to N, with the global
    budget caps set as a BACKSTOP (slightly above N) so a rounding overshoot never crashes.
  * The "recursive" arm must be structurally different (prior carry / extra level / numeric
    route). That difference is supplied per UC by a small builder; it cannot be faked here.
  * Reports learning curves (iterations-to-best, wall-to-best), not just final score, because
    recursion can win on SPEED, and the three artifact diffs.

This is a notebook/example helper, not Trace core.
"""
from __future__ import annotations

import copy
import difflib
import json
import math
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------- #
# Importable code-surface baselines
# --------------------------------------------------------------------------- #
def bbeh_direct_solver_baseline(self: Any, question: str) -> str:
    """Weak BBEH direct-answer baseline used by code-surface benchmarks.

    CodeArtifactLevel trains on a function's source via inspect.getsource(), so
    notebook-local functions are intentionally avoided here: they can be valid
    Python callables while still having no inspectable file-backed source.
    """
    return "True"


# --------------------------------------------------------------------------- #
# Result type (the only dataclass)
# --------------------------------------------------------------------------- #
@dataclass
class ArmResult:
    arm: str                       # "initial" | "standard" | "recursive"
    seed: int
    score: Optional[float]         # final score
    best_score: Optional[float] = None
    best_unit: Optional[int] = None        # candidate index where best reached
    wall_s: Optional[float] = None
    artifact: str = ""
    artifact_id: Optional[str] = None
    curve: List[Dict[str, Any]] = field(default_factory=list)
    budget_used: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def ok(self) -> bool:
        return self.error is None and self.score is not None and math.isfinite(float(self.score))


# --------------------------------------------------------------------------- #
# Budget: deterministic per-level split summing to N, caps as backstop
# --------------------------------------------------------------------------- #
def plan_budget(total_candidates: int, num_candidates: int,
                *, optimizer_llm_calls: int, eval_llm_calls: int,
                wall_time_s: int, margin: float = 1.5) -> Dict[str, Any]:
    """Global budget caps. Candidate cap is a BACKSTOP above the planned total so a
    deterministic per-level split that rounds up by one never raises BudgetExceeded."""
    return {
        "candidates": int(math.ceil(total_candidates * margin)),   # backstop, not the fairness lever
        "optimizer_llm_calls": int(optimizer_llm_calls),
        "eval_llm_calls": int(eval_llm_calls),
        "wall_time_s": int(wall_time_s),
        "on_exceed": "return_best",
    }


def allocate_levels(spec: Dict[str, Any], total_candidates: int, num_candidates: int) -> int:
    """Mutate spec['levels'] so total planned candidate slots is as close as possible to
    total_candidates, distributed evenly across levels. Returns the ACTUAL planned total so
    callers can verify both arms used the same budget (fairness is by equal actual total, and
    the global cap is a backstop). Works in iteration units (slots = iterations*num_candidates).
    """
    levels = spec.get("levels") or []
    L = len(levels)
    if L == 0:
        raise ValueError("spec['levels'] must be non-empty")
    nc = max(1, int(num_candidates))
    # total iterations across all levels, rounded to the nearest achievable with nc-sized steps
    total_iters = max(L, round(int(total_candidates) / nc))   # >= 1 iter per level
    base, extra = divmod(total_iters, L)
    actual = 0
    for i, level in enumerate(levels):
        iters = max(1, base + (1 if i < extra else 0))
        level["iterations"] = iters
        level.setdefault("trainer_kwargs", {})["num_candidates"] = nc
        actual += iters * nc
    return actual


# --------------------------------------------------------------------------- #
# Curve extraction from MemoryLite progress events
# --------------------------------------------------------------------------- #
def curve_from_memory(memory: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        events = memory.progress_events()
    except Exception:
        return rows
    best = None
    for ev in events:
        score = ev.objective_score if ev.objective_score is not None else ev.problem_score
        if score is None:
            continue
        score = float(score)
        best = score if best is None else max(best, score)
        bud = dict(ev.budget or {})
        unit = bud.get("candidates")
        if unit is None:
            unit = ev.global_step if ev.global_step is not None else ev.level_step
        rows.append({
            "unit": int(unit or 0), "score": score, "best_so_far": best,
            "elapsed_s": bud.get("elapsed_s"), "budget": bud,
            "level_id": ev.level_id, "global_step": ev.global_step, "event": ev.event,
        })
    rows.sort(key=lambda r: (r.get("unit") or 0, r.get("global_step") or 0))
    return rows


def curve_stats(curve: Sequence[Dict[str, Any]], final_score: float) -> Tuple[float, Optional[int]]:
    if not curve:
        return float(final_score), None
    best_row = max(curve, key=lambda r: float(r.get("best_so_far", r.get("score", final_score))))
    return float(best_row.get("best_so_far", best_row.get("score", final_score))), best_row.get("unit")


def first_unit_reaching(curve: Sequence[Dict[str, Any]], threshold: float) -> Optional[int]:
    best = -float("inf")
    for row in sorted(curve, key=lambda r: r.get("unit") or 0):
        best = max(best, float(row.get("score", -float("inf"))))
        if best >= threshold:
            return row.get("unit")
    return None


# --------------------------------------------------------------------------- #
# Arm runners: spec-based (default) and a pluggable callable (UC13 numeric)
# --------------------------------------------------------------------------- #
def run_spec_arm(arm: str, spec: Dict[str, Any], seed: int, primary_level: Optional[str],
                 budget_caps: Dict[str, Any], case_root: Path) -> ArmResult:
    from opto.features.recursive_opt.budget import make_budget, reset_budget
    from opto.features.recursive_opt.experiments import seed_everything
    from opto.features.recursive_opt.spec import run_spec

    spec = copy.deepcopy(spec)
    spec["run_id"] = f"{arm}:seed{seed}"
    spec["memory_root"] = str(case_root / f"{arm}_seed{seed}")
    spec["budget"] = budget_caps
    _write_json(case_root / f"{arm}_seed{seed}_spec.json", spec)

    t0 = time.time()
    try:
        seed_everything(seed)
        reset_budget(make_budget(budget_caps))
        out = run_spec(spec)
        lid = _resolve_primary_level(spec, primary_level)
        rec = out["results"][lid]
        memory = out["memory"]
        final = float(rec["score"])
        curve = curve_from_memory(memory)
        best, best_unit = curve_stats(curve, final)
        return ArmResult(arm=arm, seed=seed, score=final, best_score=best, best_unit=best_unit,
                         wall_s=float(rec.get("wall_s") or (time.time() - t0)),
                         artifact=str(rec.get("artifact", "")), artifact_id=rec.get("artifact_id"),
                         curve=curve, budget_used=_budget_snapshot(curve))
    except Exception as exc:
        return ArmResult(arm=arm, seed=seed, score=None, wall_s=time.time() - t0, error=_one_line(exc))


def run_initial_spec_arm(arm: str, spec: Dict[str, Any], seed: int, primary_level: Optional[str],
                         budget_caps: Dict[str, Any], case_root: Path) -> ArmResult:
    """Score a spec artifact once without running any optimizer update."""
    from opto.features.recursive_opt.budget import make_budget, reset_budget
    from opto.features.recursive_opt.experiments import seed_everything
    from opto.features.recursive_opt.memory import MemoryLite
    from opto.features.recursive_opt.spec import (
        _artifact_text,
        _clip_bounds,
        _final_eval,
        compile_level,
        validate_spec,
    )
    from opto.features.recursive_opt import tracebench as TB

    spec = copy.deepcopy(spec)
    spec["run_id"] = f"{arm}:seed{seed}"
    spec["memory_root"] = str(case_root / f"{arm}_seed{seed}")
    spec["budget"] = budget_caps
    _write_json(case_root / f"{arm}_seed{seed}_spec.json", spec)

    t0 = time.time()
    try:
        seed_everything(seed)
        reset_budget(make_budget(budget_caps))
        if isinstance(spec.get("tracebench"), dict):
            TB.configure_tracebench_adapter(spec["tracebench"], require=True)
        validate_spec(spec)
        memory = MemoryLite(root=spec["memory_root"])
        families = spec.get("families", {})
        lid = _resolve_primary_level(spec, primary_level)
        level_spec = _level_spec(spec, lid)
        level = compile_level(level_spec, memory, families, spec.get("scoring"))
        score, _data = _final_eval(level, level_spec, families)
        clip = _clip_bounds(level_spec.get("scoring", spec.get("scoring")))
        final = _clamp(float(score), clip)
        curve = [{
            "unit": 0,
            "score": final,
            "best_so_far": final,
            "elapsed_s": time.time() - t0,
            "budget": _budget_snapshot(),
            "level_id": lid,
            "global_step": 0,
            "event": "initial_eval",
        }]
        return ArmResult(arm=arm, seed=seed, score=final, best_score=final, best_unit=0,
                         wall_s=time.time() - t0, artifact=_artifact_text(level, level_spec["surface"]),
                         curve=curve, budget_used=_budget_snapshot(curve))
    except Exception as exc:
        return ArmResult(arm=arm, seed=seed, score=None, wall_s=time.time() - t0, error=_one_line(exc))


def run_numeric_arm(arm: str, spec: Dict[str, Any], seed: int, primary_level: Optional[str],
                    budget_caps: Dict[str, Any], case_root: Path) -> ArmResult:
    """UC13 recursive/meta arm: route numeric/categorical config search to Optuna/LSQ at the
    same candidate budget. spec must carry a 'numeric' block {level_id, task, fields, optimizer, space}."""
    from opto.features.recursive_opt.budget import make_budget, reset_budget
    from opto.features.recursive_opt.experiments import seed_everything, optimize_config_numeric
    from opto.features.recursive_opt.memory import MemoryLite
    from opto.features.recursive_opt.spec import validate_spec, compile_level
    from opto.features.recursive_opt import tracebench as TB

    spec = copy.deepcopy(spec)
    numeric = dict(spec.pop("numeric"))
    spec["run_id"] = f"{arm}:seed{seed}"
    spec["memory_root"] = str(case_root / f"{arm}_seed{seed}")
    spec["budget"] = budget_caps
    _write_json(case_root / f"{arm}_seed{seed}_spec.json", spec)
    total_trials = int(numeric.get("max_trials") or _planned_total(spec))

    t0 = time.time()
    try:
        seed_everything(seed)
        reset_budget(make_budget(budget_caps))
        if isinstance(spec.get("tracebench"), dict):
            TB.configure_tracebench_adapter(spec["tracebench"], require=True)
        validate_spec(spec)
        memory = MemoryLite(root=spec["memory_root"])
        families = spec.get("families", {})
        ls = _level_spec(spec, numeric.get("level_id") or _resolve_primary_level(spec, primary_level))
        level = compile_level(ls, memory, families, spec.get("scoring"))
        task = numeric.get("task") or ls.get("task") or families[ls["family"]][0]
        fields = list(numeric["fields"])
        assignment, best_score, history = optimize_config_numeric(
            level, task, fields, optimizer=numeric.get("optimizer", "optuna"),
            max_trials=total_trials, space=numeric.get("space"))
        curve = [{"unit": i + 1, "score": float(s),
                  "best_so_far": max(float(x) for _a, x in history[:i + 1]),
                  "budget": {"candidates": i + 1, "optimizer_llm_calls": 0}, "event": "numeric_trial"}
                 for i, (_a, s) in enumerate(history)]
        best, best_unit = curve_stats(curve, float(best_score))
        return ArmResult(arm=arm, seed=seed, score=float(best_score), best_score=best,
                         best_unit=best_unit, wall_s=time.time() - t0,
                         artifact="\n".join(f"{k}: {v}" for k, v in assignment.items()),
                         curve=curve, budget_used=_budget_snapshot(curve))
    except Exception as exc:
        return ArmResult(arm=arm, seed=seed, score=None, wall_s=time.time() - t0, error=_one_line(exc))


# --------------------------------------------------------------------------- #
# The single entry point
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Tier 2/3 scaffolds: numeric landscape (T2.5), tool-call metric (T3.6),
# multi-agent solver+critic (T3.7). Deterministic where possible so they can be
# validated offline; the LLM-dependent parts are clearly marked.
# --------------------------------------------------------------------------- #
def numeric_landscape_evaluator(*, optimum: Dict[str, Any], weights: Optional[Dict[str, float]] = None):
    """T2.5: an evaluator whose score genuinely MOVES with numeric/categorical config fields,
    so numeric routing can show a SCORE win (not only a cost win, as in flat UC13).

    score = 1 - normalized distance from a known optimum over the configured fields.
    Deterministic (no LLM), so the numeric landscape is real and testable offline.
    `optimum` maps field -> best value; numeric fields contribute squared relative error,
    categorical fields contribute 0/1 mismatch. Returns (score, feedback).
    """
    w = weights or {}
    def _evaluate(cfg_or_assignment, _task=None) -> Tuple[float, str]:
        assign = cfg_or_assignment
        if not isinstance(assign, dict):
            assign = {k: getattr(assign, k, None) for k in optimum}
        err, denom, misses = 0.0, 0.0, []
        for field, best in optimum.items():
            wt = float(w.get(field, 1.0)); denom += wt
            val = assign.get(field)
            if isinstance(best, (int, float)) and isinstance(val, (int, float)):
                scale = abs(best) or 1.0
                err += wt * ((float(val) - float(best)) / scale) ** 2
            else:
                if val != best:
                    err += wt; misses.append(field)
        score = max(0.0, 1.0 - (err / denom if denom else 0.0))
        fb = f"numeric landscape score={score:.3f}" + (f"; mismatched: {misses}" if misses else "")
        return float(score), fb
    return _evaluate


def count_tool_calls(memory_root: str | Path) -> int:
    """T3.6: count how many times the optimizer LLM actually INVOKED a tool during a run.

    Reads progress/trace records and counts events tagged as a tool call. The agentic gap
    (tools were policy-as-text, never called) is measured here: tool_calls==0 means no real
    tool use. Looks for 'tool_call'/'tool_use' markers in events.jsonl and any tool_calls.jsonl.
    """
    root = Path(memory_root)
    n = 0
    for fname in ("tool_calls.jsonl", "events.jsonl"):
        f = root / fname
        if not f.exists():
            continue
        try:
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                ev = str(rec.get("event", "")) + str(rec.get("type", ""))
                if "tool_call" in ev or "tool_use" in ev or rec.get("tool"):
                    n += 1
        except Exception:
            continue
    return n



def make_solver_critic_evaluator(
    *,
    solver_fn: Callable[[Any], Any],
    base_evaluate: Callable[[Callable[..., Any], Any], Any],
) -> Callable[[Callable[..., Any], Any], Any]:
    """Return an evaluator that scores a traced solver->critic pipeline.

    ``base_evaluate`` must keep its normal callable contract: it receives a
    candidate callable and invokes it on each example. The wrapper candidate
    first gets a solver draft for that same example, then calls the trainable
    critic with both the question and draft. This keeps the critic call on the
    trace path, so ``CodeArtifactLevel`` can optimize it.
    """
    def _evaluate(critic_component: Callable[..., Any], task: Any) -> Any:
        def _pipeline_component(*args: Any, **kwargs: Any) -> Any:
            question = kwargs.get("question", args[0] if args else task)
            try:
                draft = solver_fn(question)
            except Exception as exc:
                draft = f"<solver_error:{type(exc).__name__}>"
            critic_input = (
                f"question: {question}\n"
                f"solver_draft: {draft}\n"
                "Return the final answer only."
            )
            try:
                revised = critic_component(critic_input)
            except Exception:
                revised = draft
            return draft if revised is None else revised

        return base_evaluate(_pipeline_component, task)
    return _evaluate


def make_code_arm(*, baseline, evaluate, task_id: str, objective: str,
                  warm: bool = False, prior_fraction: float = 0.34,
                  transfer: bool = False, holdout_task_id: Optional[str] = None,
                  holdout_evaluate: Optional[Callable] = None):
    """Build a pluggable code-surface arm runner for benchmark_uc.

    standard (warm=False): cold ComponentSpec optimized for the full candidate budget.
    recursive (warm=True): two-phase at the SAME total budget - a phase-1 'prior discovery'
      run promotes its best artifact, then phase-2 warm-starts from it.

    MOD 1 (structural fix): transfer=True gives the code surface UC4's WINNING objective —
    optimize on `task_id`, but PROMOTE and SCORE on a HELD-OUT family (`holdout_task_id` /
    `holdout_evaluate`). Recursion is then rewarded for producing a GENERALISING artifact, not
    one that over-fits the training task (the warm-restart's weakness). Requires warm=True.
    Returns a callable with the (arm, spec, seed, primary_level, caps, case_root) signature.
    """
    def _runner(arm: str, spec: Dict[str, Any], seed: int, primary_level: Optional[str],
                caps: Dict[str, Any], case_root: Path) -> ArmResult:
        from opto.features.recursive_opt.budget import make_budget, reset_budget
        from opto.features.recursive_opt.experiments import seed_everything
        from opto.features.recursive_opt.memory import MemoryLite
        from opto.features.recursive_opt import (CodeArtifactLevel, ComponentSpec,
                                                 optimize, RecursiveGuide)
        from opto.features.recursive_opt.tracebench import make_dataset

        total = int(spec.get("_total_candidates") or 8)
        nc = int(spec.get("_num_candidates") or 2)
        root = str(case_root / f"{arm}_seed{seed}")
        # transfer mode scores the deployed artifact on a HELD-OUT family (UC4's objective).
        eval_task = holdout_task_id if (transfer and holdout_task_id) else task_id
        eval_fn = holdout_evaluate if (transfer and holdout_evaluate) else evaluate
        t0 = time.time()
        try:
            seed_everything(seed)
            reset_budget(make_budget(caps))
            mem = MemoryLite(root=root)
            guide = RecursiveGuide()
            comp = ComponentSpec(name=spec.get("_component", "component"), baseline=baseline,
                                 evaluate=evaluate, objective=objective)
            level = CodeArtifactLevel(comp, memory=mem)
            ds = make_dataset([task_id], repeats=int(spec.get("_max_examples", 8)))
            # held-out scoring helper: re-evaluate the current deployed code on the held-out task
            def _holdout_score() -> float:
                if not (transfer and eval_task):
                    return float(guide(task_id, level.forward(task_id), None)[0])
                comp_h = ComponentSpec(name=spec.get("_component", "component"), baseline=baseline,
                                       evaluate=eval_fn, objective=objective)
                lvl_h = CodeArtifactLevel(comp_h, memory=None)
                if level.parameters() and lvl_h.parameters():
                    lvl_h.parameters()[0]._data = level.parameters()[0].data
                return float(guide(eval_task, lvl_h.forward(eval_task), None)[0])

            initial_score = _holdout_score()
            initial_code = str(level.current_code())

            if arm == "initial":
                curve = [{
                    "unit": 0,
                    "score": initial_score,
                    "best_so_far": initial_score,
                    "elapsed_s": time.time() - t0,
                    "budget": _budget_snapshot(),
                    "event": "initial_eval",
                }]
                return ArmResult(arm=arm, seed=seed, score=initial_score,
                                 best_score=initial_score, best_unit=0,
                                 wall_s=time.time() - t0, artifact=initial_code,
                                 curve=curve, budget_used=_budget_snapshot(curve))

            if warm:
                p1 = max(nc, round(total * prior_fraction))   # phase-1 candidates (>= one full step)
                p2 = max(nc, total - p1)                       # phase-2 candidates (warm, >= one full step)
                if total < 2 * nc:
                    # too small to split into two real phases; warn via the result note
                    return ArmResult(arm=arm, seed=seed, score=None, wall_s=time.time() - t0,
                                     error=(f"code warm arm needs total_candidates >= 2*num_candidates "
                                            f"({2*nc}) to split into prior+warm phases; got {total}"))
                optimize(level, ds, guide=guide, iterations=max(1, p1 // nc),
                         num_candidates=nc, keep_best_validated=True)
                prior = mem.best_artifact(str(task_id), "code")
                if prior is not None and level.parameters():
                    level.parameters()[0]._data = prior.content   # warm-start from learned prior
                optimize(level, ds, guide=guide, iterations=max(1, p2 // nc),
                         num_candidates=nc, keep_best_validated=True)
            else:
                optimize(level, ds, guide=guide, iterations=max(1, total // nc),
                         num_candidates=nc, keep_best_validated=True)

            best = mem.best_artifact(str(task_id), "code")
            if best is not None and level.parameters():
                level.parameters()[0]._data = best.content
            # MOD 1: in transfer mode the reported score is the HELD-OUT score of the deployed
            # artifact (generalisation), not the training-task score.
            final = _holdout_score()
            artifact = best.content if best is not None else str(level.current_code())
            curve = curve_from_memory(mem)
            if not curve:
                curve = [{"unit": 0, "score": initial_score, "best_so_far": initial_score, "event": "initial"},
                         {"unit": total, "score": final, "best_so_far": max(initial_score, final), "event": "final"}]
            bscore, bunit = curve_stats(curve, final)
            # Bug-fix: a fresh stochastic re-eval of the best artifact can score below the best
            # actually found. Report the realized best (promoted artifact's recorded score), so
            # the arm is judged on the best artifact it produced, not a noisy final sample.
            # In transfer mode, held-out final is authoritative (don't inflate with train-curve best).
            if transfer and eval_task:
                reported, reported_best = final, final
            else:
                promoted = float(best.score) if best is not None else final
                reported, reported_best = max(final, bscore, promoted), max(bscore, promoted)
            return ArmResult(arm=arm, seed=seed, score=reported, best_score=reported_best, best_unit=bunit,
                             wall_s=time.time() - t0, artifact=str(artifact),
                             artifact_id=getattr(best, "artifact_id", None), curve=curve,
                             budget_used=_budget_snapshot(curve))
        except Exception as exc:
            return ArmResult(arm=arm, seed=seed, score=None, wall_s=time.time() - t0, error=_one_line(exc))
    return _runner


# --------------------------------------------------------------------------- #
# The single entry point
# --------------------------------------------------------------------------- #
def benchmark_uc(name: str, *, initial: Dict[str, Any], standard: Dict[str, Any],
                 recursive: Dict[str, Any], output_root: str | Path,
                 total_candidates: int, num_candidates: int,
                 optimizer_llm_calls: int, eval_llm_calls: int, wall_time_s: int,
                 seeds: Sequence[int] = (0,),
                 primary_level: Optional[Any] = None,
                 recursive_runner: Optional[Callable] = None,
                 standard_runner: Optional[Callable] = None,
                 initial_runner: Optional[Callable] = None,
                 notes: str = "") -> Dict[str, Any]:
    """Run the three arms at equal total budget. `initial/standard/recursive` are specs run
    through run_spec by default. Override any arm's runner:
      * recursive_runner=run_numeric_arm  (UC13 numeric; its spec carries a 'numeric' block);
      * {initial,standard,recursive}_runner=make_code_arm(...)  (code-surface UCs).
    `primary_level` may be a str (shared) or a dict {arm: level_id} when arms have different
    level structures (e.g. UC4 o2 arms use 'o2_policy', the recursive arm uses 'o3_prior').
    Code arms read budget from the spec's _total_candidates/_num_candidates keys."""
    case_root = Path(output_root) / _safe(name)
    case_root.mkdir(parents=True, exist_ok=True)
    caps = plan_budget(total_candidates, num_candidates,
                       optimizer_llm_calls=optimizer_llm_calls, eval_llm_calls=eval_llm_calls,
                       wall_time_s=wall_time_s)

    def _prep(spec: Dict[str, Any], runner: Optional[Callable], *, initial_arm: bool) -> Dict[str, Any]:
        s = copy.deepcopy(spec)
        if runner is not None:
            # code/numeric arm: stash budget so the custom runner can read it
            s.setdefault("_total_candidates", total_candidates)
            s.setdefault("_num_candidates", num_candidates)
            return s
        if initial_arm:
            return s
        else:
            allocate_levels(s, total_candidates, num_candidates)
        return s

    a0 = _prep(initial, initial_runner, initial_arm=True)
    a1 = _prep(standard, standard_runner, initial_arm=False)
    a2 = _prep(recursive, recursive_runner, initial_arm=False)

    def _run(arm, spec, runner, seed):
        default_runner = run_initial_spec_arm if arm == "initial" else run_spec_arm
        # primary_level may be a single str (shared) or a dict {arm: level_id} when arms have
        # different level structures (e.g. UC4: o2 arms -> 'o2_policy', recursive -> 'o3_prior').
        pl = primary_level.get(arm) if isinstance(primary_level, dict) else primary_level
        return (runner or default_runner)(arm, spec, seed, pl, caps, case_root)

    rows: List[ArmResult] = []
    for seed in seeds:
        rows.append(_run("initial", a0, initial_runner, seed))
        rows.append(_run("standard", a1, standard_runner, seed))
        rows.append(_run("recursive", a2, recursive_runner, seed))

    report = {"name": name, "notes": notes,
              "rows": [asdict(r) for r in rows], "summary": _summarize(rows)}
    report["promotion"] = promotion_decision(report)
    _write_json(case_root / "promotion_decision.json", report["promotion"])
    _write_json(case_root / "three_way_report.json", report)
    _write_diffs(case_root, rows)
    return report


# --------------------------------------------------------------------------- #
# Summary + verdict (credits higher final/best, speed, or optimizer-call savings)
# --------------------------------------------------------------------------- #
def benchmark_uc_budget_sweep(name: str, *, budget_points: Sequence[int], **kwargs) -> Dict[str, Any]:
    """MOD 2: run a UC at >=2 budget points and only trust a verdict that AGREES across them.

    UC10/UC11 flip with budget; a single point hides that. Returns {points, agreed, verdict,
    reports}. `verdict` is the shared promotion action if all points agree, else 'budget_sensitive'.
    kwargs are forwarded to benchmark_uc (minus total_candidates, set per point).
    """
    reports = {}
    actions = []
    for b in budget_points:
        rep = benchmark_uc(f"{name}_b{b}", total_candidates=int(b), **kwargs)
        reports[int(b)] = rep
        actions.append((rep.get("promotion") or {}).get("action"))
    agreed = len(set(actions)) == 1 and actions[0] is not None
    verdict = actions[0] if agreed else "budget_sensitive"
    return {"name": name, "budget_points": list(budget_points), "agreed": agreed,
            "verdict": verdict, "actions": actions, "reports": reports}


def promotion_decision(report: Dict[str, Any], *, min_support: int = 3,
                       min_gain: float = 0.01, z: float = 1.0) -> Dict[str, Any]:
    """Deterministic promote/hold/retest/reject gate over a three-way report.

    Hard, auditable safety gate: recursive must beat standard by a one-sided lower
    confidence bound (mean - z*std/sqrt(n)) before downstream cells reuse its artifact.
    This is the thin-LCB-reader role (REC #3) — it does NOT re-evaluate or grow decisions.py;
    robustness comes from Trace's keep_best_validated + multi-seed support here.
    """
    summary = report.get("summary") or {}
    arms = summary.get("arms") or {}
    std = arms.get("standard") or {}
    rec = arms.get("recursive") or {}
    name = report.get("name", "")
    std_mean, rec_mean = _num(std.get("mean_final")), _num(rec.get("mean_final"))
    rec_std = _num(rec.get("std_final")) or 0.0
    rec_n, std_n = int(rec.get("n") or 0), int(std.get("n") or 0)

    if std_mean is None or rec_mean is None:
        return {"action": "retest", "name": name, "promote": False,
                "reason": "standard or recursive arm has no valid mean_final",
                "support": {"standard": std_n, "recursive": rec_n}}
    if rec_n < min_support or std_n < min_support:
        return {"action": "retest", "name": name, "promote": False,
                "reason": f"support below min_support={min_support}",
                "delta_mean": rec_mean - std_mean,
                "support": {"standard": std_n, "recursive": rec_n}}

    # FIX (from CURRENT_LIMITS review): use the PAIRED-delta std (per-seed rec-std) for the LCB
    # when available. The previous gate used only the recursive arm's marginal std, which ignores
    # standard-arm + paired uncertainty and is too optimistic (e.g. UC9 promoted on a true tie).
    paired = summary.get("paired_delta") or {}
    delta_mean = rec_mean - std_mean
    paired_mean = _num(paired.get("mean"))
    paired_std = _num(paired.get("std"))
    paired_n = int(paired.get("n") or 0)
    if paired_mean is not None and paired_std is not None and paired_n >= min_support:
        basis = "paired"
        eff_mean, eff_std, eff_n = paired_mean, paired_std, paired_n
        delta_lcb = eff_mean - float(z) * eff_std / math.sqrt(max(1, eff_n))
        # Reliability rule (from CURRENT_LIMITS): a paired win must also clear its own seed
        # variance — if |mean delta| <= paired std, it is a tie regardless of the LCB.
        reliability_tie = abs(eff_mean) <= eff_std
    else:
        basis = "recursive_only"
        eff_std, eff_n = rec_std, rec_n
        delta_lcb = (rec_mean - float(z) * rec_std / math.sqrt(max(1, rec_n))) - std_mean
        reliability_tie = False

    if reliability_tie and delta_mean > 0:
        action, reason = "hold", f"{basis} mean {delta_mean:+.4f} within seed std {eff_std:.4f} (tie)"
    elif delta_lcb >= float(min_gain) and not reliability_tie:
        action, reason = "promote", f"{basis} LCB delta {delta_lcb:+.4f} >= {min_gain}, clears seed std"
    elif delta_mean < -float(min_gain):
        action, reason = "reject", f"recursive mean below standard by {delta_mean:+.4f}"
    elif delta_mean > 0:
        action, reason = "hold", f"recursive mean positive but {basis} LCB insufficient ({delta_lcb:+.4f})"
    else:
        action, reason = "hold", "recursive does not clear standard"
    return {"action": action, "name": name, "promote": action == "promote",
            "delta_mean": delta_mean, "delta_lcb": delta_lcb,
            "basis": basis, "paired_delta_std": paired_std, "recursive_std": rec_std,
            "standard_mean": std_mean, "recursive_mean": rec_mean,
            "support": {"standard": std_n, "recursive": rec_n},
            "min_support": min_support, "min_gain": min_gain, "z": z,
            "verdict": summary.get("verdict"), "reason": reason}


def _summarize(rows: Sequence[ArmResult]) -> Dict[str, Any]:
    by = {}
    for r in rows:
        by.setdefault(r.arm, []).append(r)
    arms = {a: _agg(items) for a, items in by.items()}
    init = arms.get("initial", {}).get("mean_final")
    std, rec = arms.get("standard", {}).get("mean_final"), arms.get("recursive", {}).get("mean_final")
    std_b, rec_b = arms.get("standard", {}).get("mean_best"), arms.get("recursive", {}).get("mean_best")
    std_row = next((r for r in by.get("standard", []) if r.ok()), None)
    rec_row = next((r for r in by.get("recursive", []) if r.ok()), None)
    speed = {}
    if std_row and rec_row:
        std_best = float(std_row.best_score if std_row.best_score is not None else std_row.score)
        speed = {"standard_best_unit": std_row.best_unit, "recursive_best_unit": rec_row.best_unit,
                 "recursive_unit_to_standard_best": first_unit_reaching(rec_row.curve, std_best),
                 "recursive_unit_to_standard_final": first_unit_reaching(rec_row.curve, float(std_row.score))}
    verdict, why = _verdict(init, std, rec, std_b, rec_b, speed, rows)
    paired = _paired_delta(by)
    return {"arms": arms,
            "recursive_minus_standard_final": _sub(rec, std),
            "recursive_minus_standard_best": _sub(rec_b, std_b),
            "standard_gain_final": _sub(std, init), "recursive_gain_final": _sub(rec, init),
            "paired_delta": paired,
            "speed": speed, "verdict": verdict, "why": why}


def _paired_delta(by: Dict[str, List[ArmResult]]) -> Dict[str, Any]:
    """Per-seed recursive-minus-standard deltas + their mean/std. This is the correct
    uncertainty for a paired comparison (it cancels per-seed common noise), and is what the
    reliability gate should use instead of the recursive arm's marginal std alone."""
    std_by_seed = {r.seed: float(r.score) for r in by.get("standard", []) if r.ok()}
    rec_by_seed = {r.seed: float(r.score) for r in by.get("recursive", []) if r.ok()}
    seeds = sorted(set(std_by_seed) & set(rec_by_seed))
    deltas = [rec_by_seed[s] - std_by_seed[s] for s in seeds]
    if not deltas:
        return {"n": 0, "deltas": [], "mean": None, "std": None}
    return {"n": len(deltas), "deltas": deltas,
            "mean": statistics.mean(deltas),
            "std": statistics.pstdev(deltas) if len(deltas) > 1 else 0.0}


def _verdict(init, std, rec, std_b, rec_b, speed, rows) -> Tuple[str, str]:
    if std is None or rec is None:
        errs = [r.arm for r in rows if not r.ok()]
        return "inconclusive", f"arm(s) failed/empty: {sorted(set(errs))}"
    if rec > std + 1e-9:
        return "recursive_wins_final", f"recursive final {rec:.3f} > standard {std:.3f}"
    if rec_b is not None and std_b is not None and rec_b > std_b + 1e-9:
        return "recursive_wins_best", f"recursive best {rec_b:.3f} > standard best {std_b:.3f}"
    u = speed.get("recursive_unit_to_standard_best")
    su = speed.get("standard_best_unit")
    if u is not None and su is not None and u < su:
        return "recursive_wins_speed", f"recursive reached standard's best at candidate {u} < {su}"
    std_calls = _mean_budget(rows, "standard", "optimizer_llm_calls")
    rec_calls = _mean_budget(rows, "recursive", "optimizer_llm_calls")
    if (rec_b is not None and std_b is not None and rec_b >= std_b - 1e-9
            and std_calls is not None and rec_calls is not None
            and rec_calls < std_calls):
        return (
            "recursive_wins_optimizer_calls",
            f"recursive matched standard best {std_b:.3f} with fewer optimizer LLM calls "
            f"({rec_calls:.1f} < {std_calls:.1f})",
        )
    # no win -> diagnostic bucket (the refined, non-blanket interpretation)
    one_point = all(len(r.curve) <= 1 for r in rows if r.arm in ("standard", "recursive") and r.ok())
    if one_point:
        return "no_win_curve_too_short", "runs produced a single curve point; raise iterations to compare speed"
    if abs((rec or 0) - (std or 0)) < 1e-6 and abs((std or 0) - (init or 0)) < 1e-6:
        return "no_win_flat_surface", "initial==standard==recursive; score surface looks flat (low signal)"
    return "no_win_check_design", ("recursive did not beat standard; check: is the recursive arm "
            "structurally different (prior/extra level/numeric route)? budget starved? prior hurt? field inactive?")


def _mean_budget(rows: Sequence[ArmResult], arm: str, key: str) -> Optional[float]:
    """Return the mean reported budget counter for successful rows of one arm."""
    values = [
        float(r.budget_used[key])
        for r in rows
        if r.arm == arm and r.ok() and isinstance(r.budget_used.get(key), (int, float))
    ]
    return statistics.mean(values) if values else None


def markdown_report(report: Dict[str, Any]) -> str:
    s = report["summary"]; a = s["arms"]
    L = [f"### {report['name']}", "",
         "| arm | mean final | mean best | n | err | best@unit |", "|---|---:|---:|---:|---:|---:|"]
    for arm in ("initial", "standard", "recursive"):
        it = a.get(arm, {})
        L.append(f"| {arm} | {_fmt(it.get('mean_final'))} | {_fmt(it.get('mean_best'))} | "
                 f"{it.get('n',0)} | {it.get('errors',0)} | {_fmt(it.get('mean_best_unit'))} |")
    L += ["",
          f"- **verdict:** `{s['verdict']}` — {s['why']}",
          f"- recursive − standard (final): `{_fmt(s.get('recursive_minus_standard_final'))}`  |  (best): `{_fmt(s.get('recursive_minus_standard_best'))}`",
          f"- speed: recursive reaches standard best @ candidate `{s['speed'].get('recursive_unit_to_standard_best')}` (standard best @ `{s['speed'].get('standard_best_unit')}`)",
          f"- diffs in `{_safe(report['name'])}/diffs/` (initial→standard, initial→recursive, standard→recursive)"]
    if report.get("notes"):
        L.append(f"- notes: {report['notes']}")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# diffs + small utils
# --------------------------------------------------------------------------- #
def _write_diffs(case_root: Path, rows: Sequence[ArmResult]) -> None:
    by_seed: Dict[int, Dict[str, ArmResult]] = {}
    for r in rows:
        by_seed.setdefault(r.seed, {})[r.arm] = r
    d = case_root / "diffs"; d.mkdir(parents=True, exist_ok=True)
    for seed, arms in by_seed.items():
        for x, y in (("initial", "standard"), ("initial", "recursive"), ("standard", "recursive")):
            if arms.get(x) and arms.get(y):
                (d / f"seed{seed}_{x}_to_{y}.diff").write_text(
                    _diff(arms[x].artifact, arms[y].artifact, x, y) + "\n")


def _diff(before: str, after: str, fromf: str, tof: str) -> str:
    if str(before or "") == str(after or ""):
        return "(no artifact change)"
    return "\n".join(difflib.unified_diff(str(before or "").splitlines(),
            str(after or "").splitlines(), fromfile=fromf, tofile=tof, lineterm=""))


def _agg(rows: Sequence[ArmResult]) -> Dict[str, Any]:
    ok = [r for r in rows if r.ok()]
    finals = [float(r.score) for r in ok]
    bests = [float(r.best_score if r.best_score is not None else r.score) for r in ok]
    units = [float(r.best_unit) for r in ok if r.best_unit is not None]
    return {"n": len(ok), "errors": len(rows) - len(ok),
            "mean_final": statistics.mean(finals) if finals else None,
            "std_final": statistics.pstdev(finals) if len(finals) > 1 else 0.0,
            "mean_best": statistics.mean(bests) if bests else None,
            "mean_best_unit": statistics.mean(units) if units else None}


def _planned_total(spec: Dict[str, Any]) -> int:
    tot = 0
    for lv in spec.get("levels") or []:
        nc = int((lv.get("trainer_kwargs") or {}).get("num_candidates", 1))
        tot += int(lv.get("iterations") or 1) * nc
    return max(1, tot)


def _level_spec(spec: Dict[str, Any], level_id: Optional[str]) -> Dict[str, Any]:
    if level_id is None:
        return spec["levels"][-1]
    for lv in spec["levels"]:
        if lv["id"] == level_id:
            return lv
    raise KeyError(f"level {level_id!r} not found")


def _resolve_primary_level(spec: Dict[str, Any], level_id: Optional[str]) -> str:
    """Return a valid reporting level, falling back to the spec's last level."""
    levels = spec.get("levels") or []
    if not levels:
        raise ValueError("spec['levels'] must be non-empty")
    if level_id is not None and any(level.get("id") == level_id for level in levels):
        return str(level_id)
    return str(levels[-1]["id"])


def _last_budget(curve: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    for row in reversed(curve):
        if isinstance(row.get("budget"), dict):
            return dict(row["budget"])
    return {}


def _budget_snapshot(curve: Sequence[Dict[str, Any]] = ()) -> Dict[str, Any]:
    """Return current recursive-opt budget usage, merged with curve-reported usage."""
    snap: Dict[str, Any] = {}
    try:
        from opto.features.recursive_opt.budget import current_budget
        budget = current_budget()
        snap = {
            "optimizer_llm_calls": int(budget.used_optimizer_llm_calls),
            "eval_llm_calls": int(budget.used_eval_llm_calls),
            "candidates": int(budget.used_candidates),
            "elapsed_s": float(budget.elapsed_s),
            "enabled": bool(budget.enabled),
        }
    except Exception:
        snap = {}

    for row in curve:
        budget_row = row.get("budget")
        if not isinstance(budget_row, dict):
            continue
        for key in ("optimizer_llm_calls", "eval_llm_calls", "candidates"):
            value = budget_row.get(key)
            if isinstance(value, (int, float)):
                snap[key] = max(int(snap.get(key, 0)), int(value))
        if isinstance(budget_row.get("elapsed_s"), (int, float)):
            snap["elapsed_s"] = max(float(snap.get("elapsed_s", 0.0)), float(budget_row["elapsed_s"]))
    return snap


def _clamp(value: float, clip: Optional[Tuple[float, float]]) -> float:
    """Clamp `value` when the spec declares a scoring clip."""
    if clip is None:
        return value
    lo, hi = clip
    return min(max(value, float(lo)), float(hi))


def _sub(a, b):
    return None if a is None or b is None else a - b


def _num(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except Exception:
        return None
    return f if math.isfinite(f) else None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _safe(text: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("._").lower() or "uc"


def _fmt(v) -> str:
    if v is None:
        return "-"
    try:
        f = float(v)
    except Exception:
        return str(v)
    return f"{f:.3f}" if math.isfinite(f) else "-"


def _one_line(exc: BaseException) -> str:
    lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
    return f"{type(exc).__name__}: {(lines[0] if lines else repr(exc))[:240]}"
