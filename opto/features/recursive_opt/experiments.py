"""Higher-level experiment orchestration for recursive_opt.

This module sits ABOVE :func:`run_spec`/`optimize`: a multi-seed run re-executes
the whole spec N times with a different RNG seed and an isolated memory root,
then aggregates. It is deliberately NOT a Trainer (a Trainer trains one agent
from a dataset; it cannot re-seed and re-run itself) and NOT an Optimizer — it
adds no optimization logic, only seeding + memory isolation + aggregation, and
REUSES the existing Trace trainer through ``run_spec``.

The one substantive responsibility is **real RNG seeding**: before this module,
a "multi-seed" loop only varied the ``memory_root`` suffix, so it reported
memory-isolation variance mislabeled as seed variance. ``run_spec_repeated``
seeds python/``random``, numpy, and torch (when present) per run, so reported
std is genuine seed variance.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Scores at or below this are treated as invalid candidates and excluded from
# means (kept in sync with levels.DEFAULT_INVALID_FLOOR).
_INVALID_FLOOR = -1.0


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch RNGs (best-effort; missing libs ignored)."""
    import random as _random

    _random.seed(seed)
    try:
        import numpy as _np

        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch as _torch

        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


@dataclass
class RepeatedResult:
    """Aggregate of one spec level evaluated across several seeds.

    ``scores`` keeps every seed's score (including invalid ones); the statistics
    methods exclude invalid candidates (``<= _INVALID_FLOOR``) so a single failed
    seed cannot destroy the reported mean.
    """

    level_id: str
    scores: List[float] = field(default_factory=list)
    wall_s: Optional[float] = None
    artifact: Any = None
    artifact_id: Any = None
    errors: List[str] = field(default_factory=list)
    seeds: List[int] = field(default_factory=list)

    def valid_scores(self) -> List[float]:
        return [s for s in self.scores if s > _INVALID_FLOOR]

    def mean(self) -> Optional[float]:
        v = self.valid_scores()
        return statistics.mean(v) if v else None

    def std(self) -> Optional[float]:
        v = self.valid_scores()
        return statistics.pstdev(v) if len(v) > 1 else None

    def best(self) -> Optional[float]:
        v = self.valid_scores()
        return max(v) if v else None

    def n_valid(self) -> int:
        return len(self.valid_scores())

    def n_invalid(self) -> int:
        return len(self.scores) - len(self.valid_scores())

    def to_rows(self) -> List[Dict[str, Any]]:
        """One flat dict (handy for a DataFrame or a table writer)."""
        return [{
            "level_id": self.level_id,
            "mean": self.mean(),
            "std": self.std(),
            "best": self.best(),
            "n_valid": self.n_valid(),
            "n_invalid": self.n_invalid(),
            "wall_s": self.wall_s,
            "errors": len(self.errors),
        }]

    def to_markdown(self) -> str:
        m = "-" if self.mean() is None else f"{self.mean():.3f}"
        sd = "-" if self.std() is None else f"{self.std():.3f}"
        bst = "-" if self.best() is None else f"{self.best():.3f}"
        note = f"{self.n_invalid()} invalid excluded" if self.n_invalid() else ""
        head = "| level | mean | std | best | n | wall_s | notes |\n|---|---|---|---|---|---|---|"
        return f"{head}\n| {self.level_id} | {m} | {sd} | {bst} | {self.n_valid()} | {self.wall_s} | {note} |"


def run_spec_repeated(
    spec: dict,
    seeds: Iterable[int] = (0, 1, 2),
    *,
    level_id: Optional[str] = None,
    set_seed: bool = True,
    reset_budget_each: bool = True,
    optimizer: Any = None,
    trainer: Optional[str] = None,
    budget: Any = None,
) -> Dict[str, RepeatedResult]:
    """Run ``spec`` once per seed and aggregate per level.

    Returns ``{level_id: RepeatedResult}`` for EVERY level in the spec (so
    multi-level pipelines aggregate at each depth). When ``level_id`` is given,
    only that level is returned. Each seed runs in an isolated memory root
    (``<memory_root>_seed<seed>``) and, when ``reset_budget_each`` is set, with a
    fresh budget so a leftover budget cannot silently no-op the next run.
    """
    # Imported lazily to avoid a circular import with spec.py.
    from .spec import run_spec as _run_spec_once
    from .budget import make_budget, reset_budget

    seeds = list(seeds)
    base_root = spec.get("memory_root", "./mem")
    agg: Dict[str, RepeatedResult] = {}

    for seed in seeds:
        if set_seed:
            seed_everything(seed)
        if reset_budget_each:
            reset_budget(make_budget(budget if budget is not None else spec.get("budget")))
        seed_spec = {**spec, "memory_root": f"{base_root}_seed{seed}"}
        try:
            out = _run_spec_once(seed_spec, optimizer=optimizer, trainer=trainer,
                                 budget=budget)
        except Exception as exc:  # a bad seed must not abort the sweep
            # record the failure against every requested level we know about
            target_ids = [level_id] if level_id else [l["id"] for l in spec["levels"]]
            for lid in target_ids:
                agg.setdefault(lid, RepeatedResult(level_id=lid))
                agg[lid].errors.append(f"seed {seed}: {type(exc).__name__}: {exc}".strip())
                agg[lid].seeds.append(seed)
            continue

        for lid, rec in out["results"].items():
            if level_id and lid != level_id:
                continue
            rr = agg.setdefault(lid, RepeatedResult(level_id=lid))
            rr.scores.append(rec["score"])
            rr.artifact = rec.get("artifact")
            rr.artifact_id = rec.get("artifact_id")
            rr.seeds.append(seed)

    # fill wall_s as the mean across successful seeds (best-effort)
    for rr in agg.values():
        if rr.scores:
            rr.wall_s = None  # per-seed wall not aggregated here; left to caller if needed
    if level_id:
        return {level_id: agg.get(level_id, RepeatedResult(level_id=level_id))}
    return agg


# --------------------------------------------------------------------------- #
# Numeric-optimizer bridge to a real config level (the Item-2 head-to-head seam)
# --------------------------------------------------------------------------- #
def resolve_numeric_search_space(
    fields: Sequence[str],
    space: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a validated numeric/categorical search space for ``fields``.

    ``space`` may narrow the default domains produced by
    :func:`field_search_space`, but it must still cover every numeric/categorical
    field requested by the caller. Free-text fields are ignored here because they
    must be routed to a generative optimizer.
    """
    from .numeric_optimizers import field_search_space, is_numeric_field

    numeric_fields = [field for field in fields if is_numeric_field(field)]
    if not numeric_fields:
        raise ValueError(f"no numeric/categorical fields to optimize in {fields!r}")
    base_space = dict(space) if space is not None else field_search_space(list(numeric_fields))
    search_space = {
        field: base_space[field]
        for field in numeric_fields
        if field in base_space
    }
    missing = [field for field in numeric_fields if field not in search_space]
    if missing:
        raise ValueError(f"missing search-space entries for fields: {missing!r}")
    return search_space


def optimize_config_numeric(
    level: Any,
    task: str,
    fields: Sequence[str],
    *,
    optimizer: str = "optuna",
    max_trials: int = 24,
    base_cfg: Optional["LevelConfig"] = None,
    space: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], float, List[Tuple[Dict[str, Any], float]]]:
    """Optimize numeric/categorical config fields through a real inner runner.

    ``level`` is a compiled ``MetaLevel`` whose ``_inner_runner(cfg, task)``
    returns ``(score, feedback)``. We build an ``evaluate(assignment)->score``
    that sets the assigned fields on a base config and scores it through that
    real inner runner — so this is the apples-to-apples numeric arm for the
    weak-config-surface experiment. ``space`` may narrow the default field
    domains to spec constraints, e.g. ``{"batch_size": ("cat", (2, 4, 8))}``.
    Returns ``(best_assignment, best_score, history)`` where ``history`` is the
    per-trial learning curve.
    """
    import copy as _copy
    from .numeric_optimizers import (OptunaOptimizer, LeastSquaresOptimizer,
                                     is_numeric_field)
    from .levels import LevelConfig

    if max_trials <= 0:
        raise ValueError(f"max_trials must be positive, got {max_trials!r}")
    if optimizer not in {"optuna", "least_squares"}:
        raise ValueError(f"unknown numeric optimizer {optimizer!r}")

    numeric_fields = [f for f in fields if is_numeric_field(f)]
    search_space = resolve_numeric_search_space(numeric_fields, space)
    base = base_cfg if base_cfg is not None else LevelConfig()

    def _evaluate(assignment: Dict[str, Any]) -> float:
        cfg = _copy.deepcopy(base)
        for k, v in assignment.items():
            setattr(cfg, k, v)
        score, _fb = level._inner_runner(cfg, task)
        return float(score)

    params = level.parameters() if hasattr(level, "parameters") else []
    opt_cls = OptunaOptimizer if optimizer == "optuna" else LeastSquaresOptimizer
    opt = opt_cls(params, evaluate=_evaluate, space=search_space, max_trials=max_trials)
    best = opt.step()
    best_score = max((s for _a, s in opt.history), default=float("-inf"))
    return best, best_score, opt.history
