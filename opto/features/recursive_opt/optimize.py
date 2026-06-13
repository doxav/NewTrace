"""
opto.features.recursive_opt.optimize  —  ONE DRY optimization entry-point
=========================================================================

The examples must NOT hand-roll their own ``for step: backward(); step()`` loop.
They delegate to a real **Trainer** through this single helper, configured by
three knobs (overridable via env, so you set them "at the beginning"):

    RECURSIVE_OPT_TRAINER     trainer type           (default "PrioritySearch";
                                                       falls back to GEPA-Base
                                                       = "ParetobasedPS")
    RECURSIVE_OPT_OPTIMIZER   optimizer the trainer uses (default "OptoPrimeV2")
    RECURSIVE_OPT_ITERATIONS  number of search iterations (default 10)
    RECURSIVE_OPT_NUM_CANDIDATES  candidates per search step (default 4)

Optional global safety budget (across recursive levels):

    RECURSIVE_OPT_MAX_OPTIMIZER_LLM_CALLS  live proposal calls
    RECURSIVE_OPT_MAX_EVAL_LLM_CALLS       known eval LLM calls
    RECURSIVE_OPT_MAX_CANDIDATES           planned outer candidates
    RECURSIVE_OPT_MAX_WALL_TIME_SECONDS    wall-clock limit
    RECURSIVE_OPT_BUDGET_STOP_POLICY       "return_best" (default) or "raise"

``optimize(level, dataset)`` works for EVERY recursive level (MetaLevel,
CodeArtifactLevel, CapabilityArtifact, FamilyPolicyLevel, PriorInductionLevel)
because they are all ``trace.Module`` s. The trainer updates the level's
parameters in place; the helper returns the trainer result.
"""
from __future__ import annotations

import os
import math
from typing import Any, Optional, Union

from opto import trace
from opto.optimizers.optimizer import Optimizer
from opto.trace.nodes import ParameterNode
from opto.trainer.algorithms import Trainer
from opto.trainer.guide import Guide
from opto.trainer.loggers import BaseLogger
from opto.trainer.train import (
    dataset_check,
    load_guide,
    load_logger,
    load_optimizer,
    load_trainer_class,
)

from .levels import RecursiveGuide
from .budget import BudgetExceeded, current_budget
from .runmode import make_live_llm

# --- the three knobs, set once (env-overridable) --------------------------- #
TRAINER = "PrioritySearch"
OPTIMIZER = "OptoPrimeV2"
ITERATIONS = 10
NUM_CANDIDATES = 4


def _positive_int_env(name: str, default: int) -> int:
    """Read a positive integer environment override used by live demos."""
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


def _positive_int(value: int, name: str) -> int:
    """Validate an explicit positive integer option."""
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def current_iterations(default: int = ITERATIONS) -> int:
    """Return the current recursive-opt outer iteration budget."""
    return _positive_int_env("RECURSIVE_OPT_ITERATIONS", default)


def current_num_candidates(default: int = NUM_CANDIDATES) -> int:
    """Return the current candidates-per-step budget."""
    return _positive_int_env("RECURSIVE_OPT_NUM_CANDIDATES", default)


def current_trainer(default: str = TRAINER) -> str:
    """Return the current Trainer name."""
    return os.environ.get("RECURSIVE_OPT_TRAINER", default)


def current_optimizer(default: str = OPTIMIZER) -> str:
    """Return the current optimizer name."""
    return os.environ.get("RECURSIVE_OPT_OPTIMIZER", default)


def resolve_trainer(name: Optional[str] = None) -> str:
    """Return ``name`` if that Trainer exists, else the GEPA-Base fallback.

    GEPA-Base = ``ParetobasedPS`` ("GEPA-style Pareto-based exploration on top of
    the PrioritySearch pipeline").
    """
    import opto.trainer.algorithms as algos

    requested = current_trainer() if name is None else name
    if requested and hasattr(algos, requested):
        return requested
    return "ParetobasedPS"


def optimize(
    model,
    train_dataset: dict,
    *,
    guide: Optional[object] = None,
    trainer: Optional[str] = None,
    optimizer: Optional[str] = None,
    optimizer_kwargs: Optional[dict] = None,
    guide_kwargs: Optional[dict] = None,
    logger: Union[BaseLogger, str] = "ConsoleLogger",
    logger_kwargs: Optional[dict] = None,
    iterations: Optional[int] = None,
    num_candidates: Optional[int] = None,
    batch_size: int = 1,
    keep_best_validated: bool = True,
    **trainer_kwargs,
) -> Any:
    """Optimize a recursive level with a Trainer (no hand-rolled loop).

    Parameters
    ----------
    model : trace.Module
        Any recursive level; its trainable parameters are updated in place.
    train_dataset : dict
        ``{"inputs": [...], "infos": [...]}`` (e.g. ``make_dataset([task], repeats=iterations)``).
    guide : Guide, optional
        Defaults to ``RecursiveGuide`` (maps a level's output to (score, feedback)).
    trainer, optimizer, iterations
        The three configurable knobs (see module docstring / env vars).
    num_candidates, batch_size, **trainer_kwargs
        Passed through to the Trainer (sane, cheap defaults for demos).
    """
    resolved_iterations = (
        current_iterations() if iterations is None else _positive_int(iterations, "iterations")
    )
    resolved_num_candidates = (
        current_num_candidates()
        if num_candidates is None
        else _positive_int(num_candidates, "num_candidates")
    )
    try:
        resolved_iterations = _fit_candidate_budget(
            iterations=resolved_iterations,
            num_candidates=resolved_num_candidates,
        )
    except BudgetExceeded:
        if current_budget().stop_policy == "return_best":
            canonicalize_model(model)
            return None
        raise

    result = _train_returning_trainer(
        model=model,
        train_dataset=train_dataset,
        algorithm=resolve_trainer(trainer),
        optimizer=current_optimizer() if optimizer is None else optimizer,
        guide=guide or RecursiveGuide(),
        optimizer_kwargs=_optimizer_kwargs(optimizer_kwargs),
        guide_kwargs=guide_kwargs,
        logger=logger,
        logger_kwargs=logger_kwargs,
        # search_template loops `while n_epochs < num_epochs or n_iters < num_steps`;
        # num_epochs=0 makes `iterations` (num_steps) the exact, sole stop condition.
        num_steps=resolved_iterations,
        num_epochs=0,
        num_candidates=resolved_num_candidates,
        batch_size=batch_size,
        **trainer_kwargs,
    )
    if keep_best_validated:
        restore_best_validated(result, model)   # write-back to the CALLER's model
    canonicalize_model(model)
    return result


def _optimizer_kwargs(user_kwargs: Optional[dict]) -> dict:
    """Return optimizer kwargs with a live-model LLM adapter when configured."""
    kwargs = dict(user_kwargs or {})
    model_name = os.environ.get("RECURSIVE_OPT_MODEL") or os.environ.get("TRACE_LITELLM_MODEL")
    if model_name and "llm" not in kwargs:
        kwargs["llm"] = make_live_llm(model_name)
    return kwargs


def _train_returning_trainer(
    *,
    model: Union[trace.Module, ParameterNode],
    train_dataset: dict,
    algorithm: Union[Trainer, str],
    optimizer: Union[Optimizer, str, None],
    guide: Union[Guide, str],
    logger: Union[BaseLogger, str],
    optimizer_kwargs: Optional[dict],
    guide_kwargs: Optional[dict],
    logger_kwargs: Optional[dict],
    **trainer_kwargs: Any,
) -> Any:
    """Run a Trace trainer and return the trainer when core ``train`` returns None."""
    dataset_check(train_dataset)
    trainer_class = load_trainer_class(algorithm)
    if not issubclass(trainer_class, Trainer):
        raise TypeError(f"Invalid trainer class: {trainer_class!r}")

    optimizer = optimizer or ("OPROv2" if isinstance(model, ParameterNode) else "OptoPrimeV2")

    if isinstance(model, ParameterNode):
        if not model.trainable:
            raise ValueError("The parameter must be trainable.")

        @trace.model
        class SingleNodeModel:
            def __init__(self, param: ParameterNode) -> None:
                self.param = param

            def forward(self, _x: Any) -> ParameterNode:
                return self.param

        model = SingleNodeModel(model)

    if not model.parameters():
        raise ValueError("Model must have non-empty parameters.")

    optimizer_kwargs = optimizer_kwargs or {}
    guide_kwargs = guide_kwargs or {}
    logger_kwargs = logger_kwargs or {}

    if isinstance(optimizer_kwargs, list):
        if not all(isinstance(item, dict) for item in optimizer_kwargs):
            raise TypeError("optimizer_kwargs list entries must be dictionaries.")
        optimizer_obj = [load_optimizer(optimizer, model, **item) for item in optimizer_kwargs]
        if not all(isinstance(item, Optimizer) for item in optimizer_obj):
            raise TypeError("Loaded optimizer list contains an invalid optimizer.")
    else:
        optimizer_obj = load_optimizer(optimizer, model, **optimizer_kwargs)
        if not isinstance(optimizer_obj, Optimizer):
            raise TypeError(f"Invalid optimizer instance: {optimizer_obj!r}")

    guide_obj = load_guide(guide, **guide_kwargs)
    if not isinstance(guide_obj, Guide):
        raise TypeError(f"Invalid guide instance: {guide_obj!r}")

    logger_obj = load_logger(logger, **logger_kwargs)
    if not isinstance(logger_obj, BaseLogger):
        raise TypeError(f"Invalid logger instance: {logger_obj!r}")

    algo = trainer_class(model, optimizer_obj, logger=logger_obj)
    try:
        result = algo.train(
            guide=guide_obj,
            train_dataset=train_dataset,
            **trainer_kwargs,
        )
    except BudgetExceeded:
        if current_budget().stop_policy != "return_best":
            raise
        return algo
    return result if result is not None else algo


def _fit_candidate_budget(*, iterations: int, num_candidates: int) -> int:
    """Clamp outer search steps to the remaining global candidate budget."""
    budget = current_budget()
    if budget is not None:
        for _res in ("candidates", "optimizer_llm_calls", "eval_llm_calls"):
            try:
                _rem = budget.remaining(_res)
            except Exception:
                _rem = None
            if _rem == 0:
                print(f"[recursive_opt] WARNING: global budget '{_res}' is ALREADY exhausted at "
                      f"optimize() entry — this run will no-op (stop_policy="
                      f"{getattr(budget, 'stop_policy', '?')}). Call reset_budget() first for "
                      "independent measurements (e.g. paired timing runs).")
                break
    remaining = budget.remaining("candidates")
    if remaining is None:
        budget.charge("candidates", iterations * num_candidates)
        return iterations

    allowed_iterations = min(iterations, remaining // num_candidates)
    if allowed_iterations <= 0:
        budget.charge("candidates", num_candidates)
    budget.charge("candidates", allowed_iterations * num_candidates)
    return allowed_iterations


def restore_best_validated(trainer_result: Any, model: Any = None) -> bool:
    """Write the best *evaluated* candidate back into the CALLER's model.

    Root cause of the "trained code lost after optimize()" bug (two defects):
    1. The previous helper applied the candidate onto ``trainer.agent`` — but
       PrioritySearch trains deep-copied candidates, so the caller's model (the
       object users read via ``current_code()``/``best_config_from``) was never
       updated; ``optimize()``'s in-place contract was silently broken.
    2. It trusted ``exploit()``, whose ranking maps unevaluated candidates
       (``mean_score() is None``) to 0 — on surfaces where evaluated means are
       <= 0 (or when rollouts haven't attached yet) that returns a NEVER-SCORED
       candidate, clobbering trained parameters with unvalidated ones.

    Fix: scan trainer memory for candidates with ``num_rollouts > 0``, pick the
    max ``mean_score()``, and ``apply_update(model)`` — ``set_module_parameters``
    remaps by structure, so deep-copied node identities are safe. If nothing was
    ever evaluated, leave the model untouched and return False (never clobber).
    """
    if trainer_result is None:
        return False
    target = model if model is not None else getattr(trainer_result, "agent", None)
    if target is None:
        return False
    try:
        candidates = _validated_candidates(trainer_result)
        if not candidates and hasattr(trainer_result, "exploit"):
            cand, _priority, _info = trainer_result.exploit()
            if _candidate_score(cand) is not None:
                candidates.append(cand)
        if not candidates:
            return False  # nothing validated: never overwrite trained state blindly
        best = max(candidates, key=lambda c: _candidate_score(c) or float("-inf"))
        best.apply_update(target)
        return True
    except Exception:
        return False


def _validated_candidates(trainer_result: Any) -> list[Any]:
    """Return evaluated candidates reachable from common Trainer memory slots."""
    candidates: list[Any] = []
    seen: set[int] = set()
    for candidate in _candidate_pool(trainer_result):
        score = _candidate_score(candidate)
        if score is None:
            continue
        ident = id(candidate)
        if ident in seen:
            continue
        seen.add(ident)
        candidates.append(candidate)
    return candidates


def _candidate_pool(trainer_result: Any) -> list[Any]:
    """Collect candidates from heaps plus active slots popped out for exploration."""
    pool: list[Any] = []
    for name in ("_best_candidate",):
        candidate = getattr(trainer_result, name, None)
        if candidate is not None:
            pool.append(candidate)
    for name in ("_exploration_candidates",):
        for candidate in getattr(trainer_result, name, None) or []:
            if candidate is not None:
                pool.append(candidate)
    for name in ("memory", "long_term_memory", "short_term_memory"):
        memory = getattr(trainer_result, name, None)
        if memory is None:
            continue
        items = getattr(memory, "memory", None)
        if items is None:
            try:
                items = list(memory)
            except TypeError:
                items = []
        for item in list(items or []):
            pool.append(item[1] if isinstance(item, tuple) and len(item) >= 2 else item)
    return pool


def _candidate_score(candidate: Any) -> Optional[float]:
    """Return a finite validated candidate score, or None for unevaluated ones."""
    if not getattr(candidate, "num_rollouts", 0):
        return None
    mean_score = getattr(candidate, "mean_score", None)
    if not callable(mean_score):
        return None
    score = mean_score()
    if score is None:
        return None
    value = float(score)
    return value if math.isfinite(value) else None


def canonicalize_model(model: Any) -> bool:
    """Normalize generated recursive-opt artifacts when the model supports it."""
    canonicalize = getattr(model, "canonicalize", None)
    if not callable(canonicalize):
        return False
    try:
        canonicalize()
        return True
    except ValueError:
        return False
