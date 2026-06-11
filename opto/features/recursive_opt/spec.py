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
from typing import Any, Callable, Dict, List, Optional, Tuple

from .levels import (
    LevelConfig,
    MetaLevel,
    FamilyPolicyLevel,
    PriorInductionLevel,
    ComponentSpec,
    CodeArtifactLevel,
    best_config_from,
    register_config_values,
    validate_level_config,
)
from .memory import MemoryLite
from .budget import RecursiveOptBudget, reset_budget
from .optimize import optimize, current_iterations
from . import tracebench as TB

SURFACES = ("config", "code", "family_policy", "prior", "custom")
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
        if surface == "config" and ls.get("family") not in families and not ls.get("task"):
            raise ValueError(f"level {lid}: config needs a known 'family' or an explicit 'task'")
        if surface == "code" and not isinstance(ls.get("component"), dict):
            raise ValueError(f"level {lid}: code surface needs a 'component' dict")
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

    if surface == "custom":
        return level_spec["builder"](level_spec, memory)

    if surface == "config":
        task = level_spec.get("task") or families[level_spec["family"]][0]
        cfg = LevelConfig(**(level_spec.get("fixed") or {}))
        kwargs: Dict[str, Any] = {"memory": memory}
        if level_spec.get("targets"):
            kwargs["trainable_fields"] = tuple(level_spec["targets"])
        return MetaLevel(cfg=cfg, inner_runner=_make_inner_runner(task, score_config), **kwargs)

    if surface == "code":
        c = level_spec["component"]
        comp = ComponentSpec(
            name=c["name"], baseline=c["baseline"],
            evaluate=c["evaluate"], objective=c.get("objective", ""),
        )
        return CodeArtifactLevel(comp, memory=memory)

    if surface == "family_policy":
        fams = _resolve_families(level_spec, families)
        return FamilyPolicyLevel(
            fams, run_task=make_scored_task_runner(score_config),
            policy_fields=tuple(level_spec.get("targets") or _DEFAULT_POLICY_FIELDS),
            memory=memory,
        )

    if surface == "prior":
        fams = _resolve_families(level_spec, families)
        names = list(fams)
        train = {names[0]: fams[names[0]]}
        holdout = {n: fams[n] for n in names[1:]} or {names[0]: fams[names[0]]}
        return PriorInductionLevel(
            train, holdout, run_task=make_scored_task_runner(score_config),
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
    no-LLM testing). Returns ``{"results", "levels", "memory"}`` — the built level
    objects are returned so the compiler stays transparent and debuggable.
    """
    spec = validate_spec(spec)
    if "tracebench" in spec:
        TB.configure_tracebench_adapter(spec.get("tracebench") or {}, require=True)
    families = spec.get("families", {})
    memory = _memory_from_spec(spec)
    _configure_budget(spec)
    do_reuse = bool(spec.get("reuse_priors", False))

    results: Dict[str, Any] = {}
    levels: Dict[str, Any] = {}
    for ls in spec["levels"]:
        lid = ls["id"]
        level = compile_level(ls, memory, families, spec.get("scoring"))
        levels[lid] = level
        reused = reuse_priors(memory, level, ls) if do_reuse else {"used_prior": False, "tools": []}

        iterations = int(ls.get("iterations") or current_iterations())
        opt_kwargs: Dict[str, Any] = {}
        oc = ls.get("objective_config")
        if oc:
            opt_kwargs["objective_config"] = _objective_config(oc)

        level_optimizer = ls.get("optimizer")
        agentic_factory = agentic_optimizer_factory(ls, memory, reused["tools"])
        if agentic_factory is not None and optimizer is None:
            level_optimizer = agentic_factory
        optimize(
            level, _dataset_for(ls, families, iterations),
            optimizer=(optimizer if optimizer is not None else level_optimizer),
            trainer=(trainer if trainer is not None else ls.get("trainer")),
            iterations=iterations, **opt_kwargs,
        )

        score, data = _final_eval(level, ls, families)
        rec = save_priors(memory, level, ls, score,
                          metrics=data if isinstance(data, dict) else None)
        results[lid] = {
            "surface": ls["surface"],
            "score": score,
            "artifact": _artifact_text(level, ls["surface"]),
            "reused_prior": reused["used_prior"],
            "tools": reused["tools"],
            "artifact_id": getattr(rec, "id", None),
        }
    return {"results": results, "levels": levels, "memory": memory}


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


def _dataset_for(level_spec: dict, families: Dict[str, List[str]], iterations: int) -> dict:
    if level_spec["surface"] in ("family_policy", "prior"):
        return {"inputs": [None] * iterations, "infos": [None] * iterations}
    task = level_spec.get("task") or families[level_spec["family"]][0]
    family_label = level_spec.get("family") or task
    return TB.make_dataset([family_label], repeats=iterations)


def _artifact_text(level, surface: str) -> str:
    if surface == "config":
        return best_config_from(level)
    if surface == "code":
        return level.current_code()
    if surface == "family_policy":
        return str(getattr(level, "_policy_node").data)
    if surface == "prior":
        return str(getattr(level, "_prior_node").data)
    out = level.forward(None)
    data = out.data if hasattr(out, "data") else out
    return str(data)


def _seed_from_text(level, surface: str, text: str) -> None:
    if surface == "family_policy":
        level.propose(text)
    elif surface == "prior":
        getattr(level, "_prior_node")._data = text
    elif surface == "code":
        getattr(level, "_impl")._data = text


def _final_eval(level, level_spec: dict, families: Dict[str, List[str]]):
    surface = level_spec["surface"]
    if surface == "config":
        label = level_spec.get("family") or level_spec.get("task") or families[level_spec["family"]][0]
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
    """Fail loud when targets include fields the registered adapter ignores.

    Searching unplumbed fields produces an exactly-flat score surface (the root
    cause of the 0.0 deltas): the optimizer explores knobs that never reach the
    benchmark. Adapters opt in by exposing ``PLUMBED_FIELDS``; set
    ``allow_unplumbed: true`` on the level to override deliberately.
    """
    adapter = TB._TASK_ADAPTER
    plumbed = getattr(adapter, "PLUMBED_FIELDS", None)
    if plumbed is None or level_spec.get("allow_unplumbed"):
        return
    dead = [t for t in (level_spec.get("targets") or []) if t not in plumbed]
    if dead:
        raise ValueError(
            f"level {level_spec.get('id')}: targets {dead} are not plumbed by the "
            f"registered adapter (plumbed: {list(plumbed)}). Searching them yields "
            "a flat score surface. Use plumbed targets (e.g. 'starting_artifact') "
            "or set allow_unplumbed: true to proceed anyway."
        )


def score_spread(task_id: str, probes: Optional[List[dict]] = None,
                 scoring: Optional[dict] = None) -> dict:
    """Pre-flight diagnostic: prove the config->score surface is non-flat.

    Evaluates a few probe configs (defaults exercise the artifact path, the one
    field guaranteed plumbed even at inner_steps=0) and reports the spread. Gate
    experiments on ``spread > 0``: a flat result means optimization on this task
    with these probes cannot show gains, whatever the budget.
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
    scores = [float(r["score"]) for r in rows if r.get("score") is not None]
    return {"task": task_id, "rows": rows,
            "spread": max(scores) - min(scores) if scores else 0.0,
            "flat": (max(scores) - min(scores) < 1e-9) if scores else True,
            "failed_probes": sum(1 for r in rows if r.get("score") is None)}


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
    from .capabilities import AgenticOptimizer, default_optimizer_tools

    cfg = agentic if isinstance(agentic, dict) else {}
    family = level_spec.get("family")
    available = default_optimizer_tools(
        memory=memory, family=family if isinstance(family, str) and family != "*" else None,
    )
    names = list(dict.fromkeys((level_spec.get("tools") or []) + list(reused_tools or [])))
    tools = {n: available[n] for n in names if n in available} or available
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


def _baseline_config(scoring: dict) -> LevelConfig:
    """Return the baseline config used by relative score normalization."""
    baseline = scoring.get("baseline", "default_config")
    if baseline in (None, "default_config"):
        return LevelConfig()
    if isinstance(baseline, dict):
        return LevelConfig(**baseline)
    raise ValueError("scoring.baseline must be 'default_config' or a LevelConfig dict")


def _clip_bounds(scoring: dict) -> Optional[Tuple[float, float]]:
    """Return optional score clipping bounds from a scoring config."""
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
