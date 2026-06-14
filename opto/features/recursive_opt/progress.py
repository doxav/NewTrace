"""Progress tracking for recursive optimization runs.

The trainer already emits step-indexed logs through ``BaseLogger``. This module
turns those logs into a small recursive-opt event ledger without changing the
core trainer package.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

from opto.trainer.loggers import BaseLogger, ConsoleLogger

from .budget import current_budget
from .memory import MemoryLite


def budget_snapshot() -> Dict[str, Any]:
    """Return a JSON-safe snapshot of the active recursive optimization budget."""
    try:
        budget = current_budget()
    except Exception:
        return {}
    return {
        "optimizer_llm_calls": budget.used_optimizer_llm_calls,
        "eval_llm_calls": budget.used_eval_llm_calls,
        "candidates": budget.used_candidates,
        "elapsed_s": round(float(budget.elapsed_s), 6),
        "enabled": bool(budget.enabled),
    }


def _finite_float(value: Any) -> Optional[float]:
    """Return ``value`` as a finite float, otherwise ``None``."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _jsonable(value: Any) -> Any:
    """Return a compact JSON-safe scalar for trainer log values."""
    number = _finite_float(value)
    if number is not None:
        return number
    if isinstance(value, (str, bool)) or value is None:
        return value
    return str(value)


def _best_point(
    history: Iterable[tuple[int, float]],
    *,
    fallback_score: float,
    fallback_step: Optional[int],
) -> Dict[str, Any]:
    """Return the highest-scoring point in a step history."""
    points = list(history)
    if points:
        step, score = max(points, key=lambda item: item[1])
        return {"score": score, "level_step": step}
    return {"score": float(fallback_score), "level_step": fallback_step}


class RecursiveOptProgressLogger(BaseLogger):
    """Record trainer progress into ``MemoryLite`` while optionally echoing logs."""

    _PROBLEM_METRICS = {"Test/test_score"}
    _OBJECTIVE_METRICS = {
        "Update/best_candidate_mean_score",
        "Sample/mean_score",
        "Algo/Average train score",
        "Test/test_score",
    }

    def __init__(
        self,
        *,
        memory: MemoryLite,
        run_id: str,
        level_id: str,
        level_index: int,
        task_ids: Optional[List[str]] = None,
        global_step_offset: int = 0,
        echo: bool = True,
        log_dir: str = "./logs",
        **kwargs: Any,
    ) -> None:
        super().__init__(log_dir=log_dir, **kwargs)
        self.memory = memory
        self.run_id = str(run_id)
        self.level_id = str(level_id)
        self.level_index = int(level_index)
        self.task_ids = [str(task) for task in (task_ids or [])]
        self.global_step_offset = int(global_step_offset)
        self._echo_logger = ConsoleLogger(log_dir=log_dir) if echo else None
        self._by_step: Dict[int, Dict[str, Any]] = {}

    def log(self, name: str, data: Any, step: int, **kwargs: Any) -> None:
        """Record one trainer metric and optionally mirror console output."""
        if self._echo_logger is not None:
            self._echo_logger.log(name, data, step, **kwargs)
        if str(name).startswith("Parameter/"):
            return
        level_step = int(step)
        value = _jsonable(data)
        self._by_step.setdefault(level_step, {})[str(name)] = value
        numeric = _finite_float(value)
        problem_score = numeric if name in self._PROBLEM_METRICS else None
        objective_score = numeric if name in self._OBJECTIVE_METRICS else None
        self.memory.record_progress(
            run_id=self.run_id,
            level_id=self.level_id,
            level_index=self.level_index,
            event="trainer_metric",
            level_step=level_step,
            global_step=self.global_step_offset + level_step,
            problem_score=problem_score,
            objective_score=objective_score,
            metrics={"name": str(name), "value": value},
            task_ids=self.task_ids,
            budget=budget_snapshot(),
        )

    @property
    def executed_steps(self) -> int:
        """Number of distinct trainer steps observed by this logger."""
        return max(self._by_step) + 1 if self._by_step else 0

    def build_summary(
        self,
        *,
        planned_steps: int,
        final_score: float,
        selected_by: str = "objective",
        objective_mode: str = "scalar",
    ) -> Dict[str, Any]:
        """Build a compact per-level progress summary."""
        executed = self.executed_steps
        fallback_step = max(0, executed - 1) if executed else None
        problem_history = [
            (step, float(values["Test/test_score"]))
            for step, values in self._by_step.items()
            if _finite_float(values.get("Test/test_score")) is not None
        ]
        objective_history = [
            (step, float(values[name]))
            for step, values in self._by_step.items()
            for name in ("Update/best_candidate_mean_score", "Sample/mean_score", "Algo/Average train score", "Test/test_score")
            if _finite_float(values.get(name)) is not None
        ]
        return {
            "planned_steps": int(planned_steps),
            "executed_steps": int(executed),
            "validated_candidates": int(
                max(
                    (
                        _finite_float(values.get("Update/total_samples")) or 0.0
                        for values in self._by_step.values()
                    ),
                    default=0.0,
                )
            ),
            "scores": {
                "problem_score": float(final_score),
                "objective_score": float(final_score),
            },
            "best_problem_at": _best_point(
                problem_history,
                fallback_score=float(final_score),
                fallback_step=fallback_step,
            ),
            "best_objective_at": _best_point(
                objective_history,
                fallback_score=float(final_score),
                fallback_step=fallback_step,
            ),
            "frontier": [],
            "selected_by": str(selected_by),
            "objective_mode": str(objective_mode),
            "task_ids": list(self.task_ids),
        }
