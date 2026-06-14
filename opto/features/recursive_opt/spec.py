"""Declarative control plane for ``recursive_opt`` (steps 1+2).

A ``RecursiveSpec`` is a plain ``dict`` (no new dependency) that describes:

* ``families``     - {family_name: [task_ids]} scope,
* ``budget``       - a global :class:`RecursiveOptBudget` (depth/cost guard),
* ``tracebench``   - real adapter bounds (examples, inner steps, timeouts),
* ``scoring``      - optional score normalization/capping for cross-family runs,
* ``prior_promotion`` - M1 -> M3 family-prior promotion policy,
* ``memory_root``  - where the tiered :class:`MemoryLite` persists,
* ``reuse_priors`` - transfer: warm-start each level from (family, level) memory,
* ``levels``       - an **ordered** list; *the ordering is the recursion depth*.

Each level dict selects a ``surface`` (``config`` | ``code`` | ``family_policy``
| ``prior`` | ``custom``) and is compiled into the **existing** level classes,
memory, budget, and :func:`optimize`. Nothing in Trace core is touched; this is a
thin, transparent compiler that returns the built objects so users can inspect
them.

Minimal usage::

    from opto.features.recursive_opt.spec import run_spec
    out = run_spec(my_spec_dict)            # build + optimize each level in order
    out["results"]["o1_setup"]["artifact"] # the learned config/code/policy text
    out["memory"].summary()                 # episodes/artifacts/priors recorded
"""
from __future__ import annotations

import json
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .levels import (
    CapabilityArtifact,
    TimedGuide,
    RecursiveGuide,
    LevelConfig,
    MetaLevel,
    FamilyPolicyLevel,
    PriorInductionLevel,
    ComponentSpec,
    CodeArtifactLevel,
    DEFAULT_INVALID_FLOOR,
    best_config_from,
    register_config_values,
    validate_level_config,
)
from .memory import MemoryLite
from .budget import RecursiveOptBudget, reset_budget
from .optimize import optimize, current_iterations
from .progress import RecursiveOptProgressLogger, budget_snapshot
from . import tracebench as TB

SURFACES = ("config", "code", "family_policy", "prior", "capability", "custom")


def make_level_spec(*, id: str, surface: str, targets: Optional[List[str]] = None,
                    fixed: Optional[Dict[str, Any]] = None,
                    constraints: Optional[Dict[str, List[str]]] = None,
                    **kwargs: Any) -> Dict[str, Any]:
    """Safe level-spec builder (DRY for examples/notebooks).

    Raw dict literals silently drop duplicate keys (Python keeps the last one);
    that bug shipped in example E, losing the starting_artifact menu. Keyword
    arguments cannot be duplicated, so building specs through this helper makes
    that whole bug class impossible.
    """
    level: Dict[str, Any] = {"id": id, "surface": surface}
    if targets:
        level["targets"] = list(targets)
    if fixed:
        level["fixed"] = dict(fixed)
    if constraints:
        level["constraints"] = {k: list(v) for k, v in constraints.items()}
    level.update(kwargs)
    return level
_DEFAULT_POLICY_FIELDS = ("batch_design", "memory_policy", "trainer", "trace_type")


# --------------------------------------------------------------------------- #
# Step 1: validation + compilation                                            #
# --------------------------------------------------------------------------- #
def validate_spec(spec: dict) -> dict:
    """Validate a RecursiveSpec dict (structure, ids, surfaces, field values).

    Registers any per-level ``constraints`` and validates ``fixed`` config values
    against the existing registry, so illegal values fail *before* optimization.
    Returns the (unchanged) spec for chaining.
    """
    if not isinstance(spec, dict):
        raise TypeError("spec must be a dict")
    families = spec.get("families", {})
    if not isinstance(families, dict):
        raise TypeError("spec['families'] must be {name: [task_ids]}")
    levels = spec.get("levels")
    if not levels or not isinstance(levels, list):
        raise ValueError("spec['levels'] must be a non-empty list (its order is the depth)")
    if not isinstance(spec.get("budget", {}), dict):
        raise TypeError("spec['budget'] must be a dict")
    _validate_tracebench_config(spec.get("tracebench", {}))
    _validate_scoring_config(spec.get("scoring", {}))
    _validate_prior_promotion_config(spec.get("prior_promotion", {}))

    seen: set = set()
    for ls in levels:
        if not isinstance(ls, dict):
            raise TypeError("each level must be a dict")
        lid = ls.get("id")
        if not lid or lid in seen:
            raise ValueError(f"each level needs a unique 'id' (got {lid!r})")
        seen.add(lid)
        surface = ls.get("surface")
        if surface not in SURFACES:
            raise ValueError(f"level {lid}: surface must be one of {SURFACES}, got {surface!r}")

        # register declared constraints, then validate the seed/fixed values
        for field, allowed in (ls.get("constraints") or {}).items():
            register_config_values(field, allowed)
        if surface in ("config", "code", "family_policy", "prior"):
            fixed = ls.get("fixed") or {}
            cfg = LevelConfig(**fixed)
            fields = tuple(ls.get("targets") or ()) + tuple(fixed.keys())
            validate_level_config(cfg, fields)

        if surface in ("config", "family_policy", "prior"):
            _check_plumbing(ls)
        deps = ls.get("depends_on") or []
        earlier = seen - {lid}
        unknown = [d for d in deps if d not in earlier]
        if unknown:
            raise ValueError(
                f"level {lid}: depends_on {unknown} must reference EARLIER level ids "
                f"(seen so far: {sorted(earlier)}). depends_on is enforced, not decorative."
            )
        tasks = ls.get("tasks")
        if tasks is not None:
            if not isinstance(tasks, list) or not tasks or not all(str(t).strip() for t in tasks):
                raise ValueError(f"level {lid}: tasks must be a non-empty list of task ids")
        if surface == "config" and ls.get("family") not in families and not ls.get("task") and not tasks:
            raise ValueError(f"level {lid}: config needs a known 'family', explicit 'task', or non-empty 'tasks'")
        if surface == "code" and not isinstance(ls.get("component"), dict):
            raise ValueError(f"level {lid}: code surface needs a 'component' dict")
        if surface == "capability" and not callable(ls.get("evaluator")):
            raise TypeError(f"level {ls.get('id')}: capability surface requires a callable 'evaluator'")
        if surface == "custom" and not callable(ls.get("builder")):
            raise ValueError(f"level {lid}: custom surface needs a callable 'builder'")
        if surface in ("family_policy", "prior"):
            if not _resolve_families(ls, families):
                raise ValueError(f"level {lid}: {surface} needs at least one family")
    return spec


def compile_level(
    level_spec: dict,
    memory: MemoryLite,
    families: Dict[str, List[str]],
    scoring: Optional[dict] = None,
):
    """Compile one level dict into the matching existing level object."""
    surface = level_spec["surface"]
    score_config = level_spec.get("scoring", scoring)
    clip = _clip_bounds(score_config)
    # Invalid candidates score the worst LEGAL value. When an explicit clip is
    # configured, use its floor; otherwise fall back to a BOUNDED default (-1.0)
    # rather than the raw -1e9 sentinel. A single invalid config must sort last
    # WITHOUT destroying reported means/normalisation baselines (root cause of
    # the -1e9 / -333M leaks in the use-case tables when no scoring.clip is set).
    floor = clip[0] if clip else DEFAULT_INVALID_FLOOR

    if surface == "custom":
        return level_spec["builder"](level_spec, memory)

    if surface == "config":
        task_ids = _config_task_ids(level_spec, families)
        cfg = LevelConfig(**(level_spec.get("fixed") or {}))
        kwargs: Dict[str, Any] = {"memory": memory}
        if level_spec.get("targets"):
            kwargs["trainable_fields"] = tuple(level_spec["targets"])
        inner_runner = (
            _make_inner_runner(task_ids[0], score_config)
            if len(task_ids) == 1
            else _make_task_set_inner_runner(task_ids, score_config)
        )
        return MetaLevel(cfg=cfg, inner_runner=inner_runner,
                         invalid_floor=floor, **kwargs)

    if surface == "code":
        c = level_spec["component"]
        comp = ComponentSpec(
            name=c["name"], baseline=c["baseline"],
            evaluate=c["evaluate"], objective=c.get("objective", ""),
        )
        return CodeArtifactLevel(comp, memory=memory)

    if surface == "capability":
        return CapabilityArtifact(level_spec.get("seed", ""),
                                  evaluator=level_spec["evaluator"], memory=memory)

    if surface == "family_policy":
        fams = _resolve_families(level_spec, families)
        return FamilyPolicyLevel(
            fams, run_task=make_scored_task_runner(score_config), invalid_floor=floor,
            policy_fields=tuple(level_spec.get("targets") or _DEFAULT_POLICY_FIELDS),
            memory=memory,
        )

    if surface == "prior":
        fams = _resolve_families(level_spec, families)
        names = list(fams)
        train = {names[0]: fams[names[0]]}
        holdout = {n: fams[n] for n in names[1:]} or {names[0]: fams[names[0]]}
        return PriorInductionLevel(
            train, holdout, run_task=make_scored_task_runner(score_config), invalid_floor=floor,
            fields=tuple(level_spec.get("targets") or _DEFAULT_POLICY_FIELDS),
            memory=memory,
        )

    raise ValueError(f"unknown surface {surface!r}")


# --------------------------------------------------------------------------- #
# Step 2: prior/tool reuse, budget, and the runner                            #
# --------------------------------------------------------------------------- #
def reuse_priors(memory: MemoryLite, level, level_spec: dict) -> dict:
    """Warm-start a level from (family, level) memory and load reusable tools.

    Config levels warm-start from the promoted family prior (existing hook).
    For any surface, previously learned tools (stored as ``kind='tool'`` artifacts
    for the family) are loaded and returned so an agentic optimizer can reuse
    them. This is the transfer-learning entry point for a new project that points
    at the same family.
    """
    surface = level_spec["surface"]
    family = str(level_spec.get("family") or "*")
    used_prior = False

    if surface == "config" and hasattr(level, "warm_start_from_memory"):
        before = best_config_from(level)
        level.warm_start_from_memory(family)
        used_prior = (best_config_from(level) != before) or (memory.family_prior(family) is not None)
    else:
        # best-effort seed from a previously saved artifact of the same surface
        prev = memory.best_artifact(family=family, kind=surface)
        if prev is not None and hasattr(level, "propose"):
            try:
                _seed_from_text(level, surface, prev.content)
                used_prior = True
            except Exception:
                used_prior = False

    tools = [a.content for a in memory.artifact_history(family, "tool")]
    return {"used_prior": used_prior, "tools": tools}


def save_priors(memory: MemoryLite, level, level_spec: dict, score: float,
                metrics: Optional[dict] = None):
    """Persist the learned artifact (+ declared tools) tagged by family and level.

    Tagging by the spec's family makes the artifact retrievable for transfer to a
    new project via :func:`reuse_priors`. Episode/family-prior promotion already
    happens inside each level's ``forward`` during ``optimize``.
    """
    surface = level_spec["surface"]
    family = str(level_spec.get("family") or "*")
    rec = memory.record_artifact(
        level=surface, family=family, kind=surface,
        content=_artifact_text(level, surface), score=float(score), metrics=metrics,
    )
    for tool in (level_spec.get("tools") or []):
        memory.record_artifact(level=surface, family=family, kind="tool",
                               content=str(tool), score=float(score))
    return rec


def run_spec(spec: dict, *, optimizer=None, trainer: Optional[str] = None) -> dict:
    """Compile and run every level in order (the ordering is the recursion depth).

    ``optimizer``/``trainer`` override the per-level choice (used for offline,
    no-LLM testing). Returns ``{"results", "levels", "memory", "progress"}``;
    the built level objects are returned so the compiler stays transparent and
    debuggable.
    """
    spec = validate_spec(spec)
    if "tracebench" in spec:
        TB.configure_tracebench_adapter(spec.get("tracebench") or {}, require=True)
    families = spec.get("families", {})
    memory = _memory_from_spec(spec)
    _configure_budget(spec)
    do_reuse = bool(spec.get("reuse_priors", False))

    run_id = str(spec.get("run_id") or f"recursive_opt:{int(time.time() * 1000)}")
    results: Dict[str, Any] = {}
    levels: Dict[str, Any] = {}
    progress_summary: Dict[str, Any] = {"run_id": run_id, "levels": {}}
    global_step = 0
    for level_index, ls in enumerate(spec["levels"]):
        lid = ls["id"]
        level = compile_level(ls, memory, families, spec.get("scoring"))
        levels[lid] = level
        reused = reuse_priors(memory, level, ls) if do_reuse else {"used_prior": False, "tools": []}

        iterations = int(ls.get("iterations") or current_iterations())
        # Generic trainer controls (e.g. {"num_threads": 1} for deterministic
        # unit tests) can live at spec or level scope without adding another
        # bespoke argument to run_spec.
        opt_kwargs: Dict[str, Any] = dict(spec.get("trainer_kwargs") or {})
        opt_kwargs.update(ls.get("trainer_kwargs") or {})
        oc = ls.get("objective_config")
        if oc:
            opt_kwargs["objective_config"] = _objective_config(oc)

        level_optimizer = ls.get("optimizer")
        agentic_factory = agentic_optimizer_factory(ls, memory, reused["tools"])
        if agentic_factory is not None and optimizer is None:
            level_optimizer = agentic_factory
        if ls.get("timed_guide"):
            # wall-time as a first-class objective: pair with
            # "objective_config": {"mode": "pareto", "minimize": ["wall_time"]}
            opt_kwargs["guide"] = TimedGuide(RecursiveGuide())
        objective_mode = str((oc or {}).get("mode", "scalar"))
        selected_by = "pareto" if objective_mode == "pareto" else "objective"
        task_ids = _level_task_ids(ls, families)
        progress_logger = RecursiveOptProgressLogger(
            memory=memory,
            run_id=run_id,
            level_id=lid,
            level_index=level_index,
            task_ids=task_ids,
            global_step_offset=global_step,
            echo=True,
        )
        # Use a recursive-opt logger by default so progress is persisted without
        # changing core trainer internals. It still mirrors the ConsoleLogger.
        opt_kwargs["logger"] = progress_logger
        memory.record_progress(
            run_id=run_id,
            level_id=lid,
            level_index=level_index,
            event="level_start",
            level_step=0,
            global_step=global_step,
            metrics={
                "planned_steps": iterations,
                "surface": ls["surface"],
                "objective_mode": objective_mode,
            },
            task_ids=task_ids,
            budget=budget_snapshot(),
            selected_by=selected_by,
        )
        _t0 = time.time()
        trainer_result = optimize(
            level, _dataset_for(ls, families, iterations),
            optimizer=(optimizer if optimizer is not None else level_optimizer),
            trainer=(trainer if trainer is not None else ls.get("trainer")),
            iterations=iterations, **opt_kwargs,
        )
        wall_s = round(time.time() - _t0, 3)

        score, data = _final_eval(level, ls, families)
        score = _clamp(score, _clip_bounds(spec.get("scoring")))  # belt: sentinel can never leak raw
        executed_steps = max(
            progress_logger.executed_steps,
            int(getattr(trainer_result, "n_iters", 0) or 0),
        )
        level_progress = progress_logger.build_summary(
            planned_steps=iterations,
            final_score=float(score),
            selected_by=selected_by,
            objective_mode=objective_mode,
        )
        level_progress["executed_steps"] = executed_steps
        artifact_metrics = dict(data) if isinstance(data, dict) else {}
        artifact_metrics["scores"] = dict(level_progress["scores"])
        artifact_metrics["progress"] = dict(level_progress)
        rec = save_priors(memory, level, ls, score,
                          metrics=artifact_metrics)
        level_progress["artifact_id"] = getattr(rec, "artifact_id", None)
        progress_summary["levels"][lid] = level_progress
        memory.record_progress(
            run_id=run_id,
            level_id=lid,
            level_index=level_index,
            event="level_end",
            level_step=max(0, executed_steps - 1) if executed_steps else None,
            global_step=global_step + executed_steps,
            artifact_id=getattr(rec, "artifact_id", None),
            problem_score=float(score),
            objective_score=float(score),
            metrics={"summary": level_progress, "wall_s": wall_s},
            task_ids=task_ids,
            budget=budget_snapshot(),
            selected_by=selected_by,
        )
        results[lid] = {
            "surface": ls["surface"],
            "score": score,
            "wall_s": wall_s,
            "artifact": _artifact_text(level, ls["surface"]),
            "reused_prior": reused["used_prior"],
            "tools": reused["tools"],
            "artifact_id": getattr(rec, "artifact_id", None),
            "depends_on": list(ls.get("depends_on") or []),  # recorded dependency edges
            "progress": level_progress,
        }
        global_step += executed_steps
    memory.write_run_summary(progress_summary)
    return {"results": results, "levels": levels, "memory": memory,
            "progress": progress_summary}


# --------------------------------------------------------------------------- #
# helpers (all over existing objects)                                         #
# --------------------------------------------------------------------------- #
def _resolve_families(level_spec: dict, families: Dict[str, List[str]]) -> Dict[str, List[str]]:
    sel = level_spec.get("families")
    fam = level_spec.get("family")
    if sel in (None, "*") and fam in (None, "*"):
        return dict(families)
    if isinstance(sel, list):
        return {k: families[k] for k in sel}
    if isinstance(fam, list):
        return {k: families[k] for k in fam}
    if isinstance(fam, str) and fam in families:
        return {fam: families[fam]}
    return dict(families)


def _config_task_ids(level_spec: dict, families: Dict[str, List[str]]) -> List[str]:
    """Return concrete task ids for a config level."""
    tasks = level_spec.get("tasks")
    if tasks is not None:
        return [str(task) for task in tasks]
    task = level_spec.get("task")
    if task:
        return [str(task)]
    return [str(families[level_spec["family"]][0])]


def _level_task_ids(level_spec: dict, families: Dict[str, List[str]]) -> List[str]:
    """Return task ids associated with any level for progress records."""
    if level_spec.get("surface") == "config" and (
        level_spec.get("tasks") or level_spec.get("task") or level_spec.get("family") in families
    ):
        return _config_task_ids(level_spec, families)
    selected = _resolve_families(level_spec, families)
    out: List[str] = []
    for tasks in selected.values():
        out.extend(str(task) for task in tasks)
    return out


def _dataset_for(level_spec: dict, families: Dict[str, List[str]], iterations: int) -> dict:
    fam = level_spec.get("family")
    task = level_spec.get("task")
    tasks = level_spec.get("tasks")
    # family_policy/prior iterate internally; custom/capability levels may be
    # label-free; all of these train on a None-input dataset.
    if level_spec["surface"] in ("family_policy", "prior") or not (fam or task or tasks):
        return {"inputs": [None] * iterations, "infos": [None] * iterations}
    if tasks:
        return TB.make_dataset([f"task_set:{level_spec.get('id', 'config')}"], repeats=iterations)
    return TB.make_dataset([fam or task], repeats=iterations)


def _artifact_text(level, surface: str) -> str:
    if surface == "config":
        return best_config_from(level)
    if surface == "code":
        return level.current_code()
    if surface == "capability":
        return str(level.impl.data)
    if surface == "family_policy":
        return str(getattr(level, "_policy_node").data)
    if surface == "prior":
        return str(getattr(level, "_prior_node").data)
    out = level.forward(None)
    data = out.data if hasattr(out, "data") else out
    return str(data)


def _seed_from_text(level, surface: str, text: str) -> None:
    if surface == "capability":
        level.impl._data = text
    elif surface == "family_policy":
        level.propose(text)
    elif surface == "prior":
        getattr(level, "_prior_node")._data = text
    elif surface == "code":
        getattr(level, "_impl")._data = text


def _final_eval(level, level_spec: dict, families: Dict[str, List[str]]):
    surface = level_spec["surface"]
    if surface == "config":
        label = level_spec.get("family") or level_spec.get("task")
        if label is None and level_spec.get("tasks"):
            label = f"task_set:{level_spec.get('id', 'config')}"
        if label is None:
            label = families[level_spec["family"]][0]
        out = level.forward(label)
    elif surface == "code":
        fam = level_spec.get("family")
        task = level_spec.get("task") or (families.get(fam, [None])[0] if fam else None)
        out = level.forward(task)
    else:
        out = level.forward(None)
    data = out.data if hasattr(out, "data") else out
    score = float(data.get("score", 0.0)) if isinstance(data, dict) else 0.0
    return score, data


def _check_plumbing(level_spec: dict) -> None:
    """Validate targets through the adapter's CAUSAL-EFFECT contract.

    Not binary plumbed/unplumbed: a field is valid when it has an ACTIVE causal
    path (artifact / optimization / feedback / trace / memory / budget / search /
    score) under the adapter's current run mode. Configurable per level:
      - ``allow_inactive: true`` (alias: legacy ``allow_unplumbed``) -> report,
        don't raise (deliberate diagnostic search);
      - ``effect_policy: {"required_effects": ["memory", ...]}`` -> only those
        effect kinds count as relevant for this experiment.
    Raises ``InactiveFieldError`` naming each dead field WITH its activating
    condition, so the error is the documentation.
    """
    from .effects import check_field_effects
    adapter = TB._TASK_ADAPTER
    if adapter is None:
        return
    policy = level_spec.get("effect_policy") or {}
    check_field_effects(
        adapter, level_spec.get("targets") or [],
        required_effects=policy.get("required_effects"),
        allow_inactive=bool(level_spec.get("allow_inactive")
                            or level_spec.get("allow_unplumbed")
                            or policy.get("allow_inactive")),
    )


def score_spread(task_id: str, probes: Optional[List[dict]] = None,
                 scoring: Optional[dict] = None) -> dict:
    """Pre-flight diagnostic: prove the config->score surface is non-flat.

    Evaluates a few probe configs (defaults exercise the artifact path, the one
    field guaranteed plumbed even at inner_steps=0) and reports the spread. Gate
    experiments on ``valid_spread > 0`` and ``catastrophic is False``: a flat
    result means optimization on this task with these probes cannot show gains,
    while catastrophic invalid probes mean the probe set is incompatible with
    the surface and should not be interpreted as signal.
    """
    probes = probes or [
        {},  # adapter/bundle default artifact
        {"starting_artifact": "Answer directly."},
        {"starting_artifact": "Plan step by step, then verify the answer before replying."},
    ]
    runner = make_scored_task_runner(scoring)
    rows = []
    for p in probes:
        try:
            score, _ = runner(LevelConfig(**p), task_id)
            value = float(score)
            if not math.isfinite(value):
                raise ValueError(f"non-finite score {value!r}")
            rows.append({"probe": p, "score": value})
        except Exception as exc:
            # Some plumbed probes are only valid for prompt-like tasks. Keep the
            # gate diagnostic alive and expose the incompatible arm explicitly.
            rows.append({
                "probe": p,
                "score": None,
                "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
            })
    def is_invalid_probe(row: dict) -> bool:
        """Return whether a probe produced no usable score signal."""
        if row.get("score") is None:
            return True
        score = float(row["score"])
        return (not math.isfinite(score)) or score <= -999_999.0

    valid_scores = [
        float(row["score"])
        for row in rows
        if not is_invalid_probe(row)
    ]
    invalid_probes = sum(1 for row in rows if is_invalid_probe(row))
    valid_spread = max(valid_scores) - min(valid_scores) if valid_scores else 0.0
    return {
        "task": task_id,
        "rows": rows,
        "spread": valid_spread,          # backward-compatible alias
        "valid_spread": valid_spread,
        "flat": valid_spread < 1e-9,
        "failed_probes": invalid_probes, # backward-compatible alias
        "invalid_probes": invalid_probes,
        "catastrophic": invalid_probes > 0,
    }


def agentic_optimizer_factory(level_spec: dict, memory: MemoryLite,
                              reused_tools: Optional[List[str]] = None):
    """Build an AgenticOptimizer factory wiring (declared + reused) tools.

    Tool *names* select callables from ``default_optimizer_tools`` (memory-backed
    trace_search etc.), so tools learned/saved for a family are re-armed on reuse.
    Returns an optimizer class usable by Trace's existing ``load_optimizer`` API,
    or None when the level is not agentic.
    """
    agentic = level_spec.get("agentic")
    if not agentic:
        return None
    from .capabilities import (
        AgenticOptimizer,
        default_optimizer_tools,
        select_optimizer_tools,
    )

    cfg = agentic if isinstance(agentic, dict) else {}
    family = level_spec.get("family")
    available = default_optimizer_tools(
        memory=memory, family=family if isinstance(family, str) and family != "*" else None,
    )
    default_names = list(dict.fromkeys((level_spec.get("tools") or []) + list(reused_tools or [])))
    policy = cfg.get("tool_policy", level_spec.get("tool_policy"))
    if policy is None:
        tools = {n: available[n] for n in default_names if n in available} or available
    else:
        tools = select_optimizer_tools(
            available,
            policy,
            default_tools=default_names,
            max_tools=int(cfg.get("tool_budget", 3)),
        ) or ({n: available[n] for n in default_names if n in available} or available)
    configured_kwargs = {"tools": tools, "tool_budget": int(cfg.get("tool_budget", 3))}
    if cfg.get("base_optimizer_cls") is not None:
        configured_kwargs["base_optimizer_cls"] = cfg["base_optimizer_cls"]

    class ConfiguredAgenticOptimizer(AgenticOptimizer):
        """Agentic optimizer class configured from a declarative level spec."""

        keywords = configured_kwargs

        def __init__(self, parameters: list, **optimizer_kwargs: Any) -> None:
            super().__init__(parameters, **{**configured_kwargs, **optimizer_kwargs})

    return ConfiguredAgenticOptimizer


def _memory_from_spec(spec: dict) -> MemoryLite:
    """Create MemoryLite from spec-level prior-promotion controls."""
    promotion = spec.get("prior_promotion") or {}
    return MemoryLite(
        root=spec.get("memory_root", "./trace_memory"),
        promotion_min_support=int(promotion.get("min_support", 3)),
        promote_priors=bool(promotion.get("enabled", True)),
        promotion_min_score=promotion.get("min_score"),
    )


def make_scored_task_runner(
    scoring: Optional[dict] = None,
    *,
    raw_runner: Optional[Callable[[LevelConfig, str], Tuple[float, str]]] = None,
) -> Callable[[LevelConfig, str], Tuple[float, str]]:
    """Wrap a task runner with optional spec-level score normalization.

    Raw Trace-Bench scores remain the default. ``relative_delta`` converts each
    task score into improvement over a baseline config, which keeps cross-family
    policy learning from being dominated by incompatible raw score scales.
    """
    cfg = scoring or {}
    _validate_scoring_config(cfg)
    runner = raw_runner or TB.make_task_runner()
    mode = cfg.get("mode", "raw")
    clip = _clip_bounds(cfg)
    report_raw = bool(cfg.get("report_raw", mode != "raw"))
    baseline_cache: Dict[str, float] = {}
    baseline_cfg = _baseline_config(cfg)

    def run(level_cfg: LevelConfig, task_id: str) -> Tuple[float, str]:
        raw_score, feedback = runner(level_cfg, task_id)
        score = float(raw_score)
        meta: Dict[str, Any] = {"mode": mode, "raw_score": score}
        if mode == "relative_delta":
            key = str(task_id)
            if key not in baseline_cache:
                baseline_cache[key] = float(runner(baseline_cfg, task_id)[0])
            meta["baseline_score"] = baseline_cache[key]
            score = score - baseline_cache[key]
        if clip is not None:
            lo, hi = clip
            score = min(max(score, lo), hi)
            meta["clip"] = [lo, hi]
        meta["score"] = score
        if report_raw:
            feedback = (
                f"{feedback} SCORE_NORMALIZATION_JSON="
                f"{json.dumps(meta, sort_keys=True)}"
            )
        return float(score), feedback

    return run


def _make_inner_runner(
    task_id: str,
    scoring: Optional[dict],
) -> Callable[[LevelConfig, Any], Tuple[float, str]]:
    """Bind a possibly normalized task runner to one Trace-Bench task id."""
    run_task = make_scored_task_runner(scoring)

    def inner_runner(cfg: LevelConfig, _family: Any) -> Tuple[float, str]:
        return run_task(cfg, task_id)

    return inner_runner


def _make_task_set_inner_runner(
    task_ids: List[str],
    scoring: Optional[dict],
) -> Callable[[LevelConfig, Any], Tuple[float, str]]:
    """Bind a normalized task runner to a fixed multi-task evaluation set."""
    if not task_ids:
        raise ValueError("task_ids must be non-empty")
    run_task = make_scored_task_runner(scoring)

    def inner_runner(cfg: LevelConfig, _family: Any) -> Tuple[float, str]:
        scores: List[float] = []
        feedbacks: List[str] = []
        for task_id in task_ids:
            score, feedback = run_task(cfg, task_id)
            scores.append(float(score))
            feedbacks.append(f"{task_id}: {feedback}")
        mean_score = sum(scores) / len(scores)
        return (
            mean_score,
            "[task_set] mean="
            f"{mean_score:.3f} over {len(task_ids)} task(s). "
            + " || ".join(feedbacks),
        )

    return inner_runner


def _baseline_config(scoring: dict) -> LevelConfig:
    """Return the baseline config used by relative score normalization."""
    baseline = scoring.get("baseline", "default_config")
    if baseline in (None, "default_config"):
        return LevelConfig()
    if isinstance(baseline, dict):
        return LevelConfig(**baseline)
    raise ValueError("scoring.baseline must be 'default_config' or a LevelConfig dict")


def _clamp(value: float, clip: Optional[Tuple[float, float]]) -> float:
    """Bound a reported score to the configured clip range (no-op without one)."""
    if clip is None:
        return float(value)
    lo, hi = clip
    return float(min(hi, max(lo, value)))


def _clip_bounds(scoring: Optional[dict]) -> Optional[Tuple[float, float]]:
    """Return optional score clipping bounds from a scoring config."""
    if not scoring:
        return None
    raw = scoring.get("clip")
    if raw is None and scoring.get("mode") == "clip":
        raw = [scoring.get("min", float("-inf")), scoring.get("max", float("inf"))]
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("scoring.clip must be [min, max]")
    lo, hi = float(raw[0]), float(raw[1])
    if lo > hi:
        raise ValueError("scoring.clip minimum cannot exceed maximum")
    return lo, hi


def _validate_tracebench_config(config: dict) -> None:
    """Validate optional top-level Trace-Bench adapter config."""
    if not config:
        return
    if not isinstance(config, dict):
        raise TypeError("spec['tracebench'] must be a dict")
    TB.TraceBenchTaskAdapter.from_config(config)


def _validate_scoring_config(config: dict) -> None:
    """Validate optional score-normalization config."""
    if not config:
        return
    if not isinstance(config, dict):
        raise TypeError("spec['scoring'] must be a dict")
    mode = config.get("mode", "raw")
    if mode not in {"raw", "clip", "relative_delta"}:
        raise ValueError("scoring.mode must be one of raw, clip, relative_delta")
    _clip_bounds(config)
    if mode == "relative_delta":
        _baseline_config(config)


def _validate_prior_promotion_config(config: dict) -> None:
    """Validate optional prior-promotion config."""
    if not config:
        return
    if not isinstance(config, dict):
        raise TypeError("spec['prior_promotion'] must be a dict")
    min_support = int(config.get("min_support", 3))
    if min_support <= 0:
        raise ValueError("prior_promotion.min_support must be positive")
    min_score = config.get("min_score")
    if min_score is not None and not isinstance(min_score, (int, float)):
        raise TypeError("prior_promotion.min_score must be a number")


def _configure_budget(spec: dict):
    b = spec.get("budget") or {}
    if not b:
        return None
    budget = RecursiveOptBudget(
        max_optimizer_llm_calls=b.get("optimizer_llm_calls"),
        max_eval_llm_calls=b.get("eval_llm_calls"),
        max_candidates=b.get("candidates"),
        max_wall_time_s=b.get("wall_time_s"),
        stop_policy=b.get("on_exceed", "return_best"),
    )
    reset_budget(budget)
    return budget


def _objective_config(oc):
    if isinstance(oc, dict):
        from opto.trainer.objectives import ObjectiveConfig
        return ObjectiveConfig(mode=oc.get("mode", "pareto"),
                               minimize=set(oc.get("minimize", [])))
    return oc
