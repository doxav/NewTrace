"""
Global runtime budget for recursive optimization.

The local knobs (`RECURSIVE_OPT_ITERATIONS`,
`RECURSIVE_OPT_NUM_CANDIDATES`, Trace-Bench inner limits, etc.) still define
the shape of each optimization loop. This module adds an optional global safety
envelope across levels so a live recursive run can fail early, or return the
best currently validated state, before it spends more LLM calls or candidate
slots than intended.

Unset or `none`/`null`/`unlimited` means "no global limit" for that resource.
The value `0` is meaningful and means "allow zero"; for example
`RECURSIVE_OPT_MAX_OPTIMIZER_LLM_CALLS=0` prevents proposal LLM calls while
still allowing offline plumbing and non-LLM evaluation code to run.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

BudgetResource = Literal["optimizer_llm_calls", "eval_llm_calls", "candidates"]

_UNLIMITED = {"", "none", "null", "unlimited", "off", "-1"}
_STOP_POLICIES = {"return_best", "raise"}
_GLOBAL_BUDGET: Optional["RecursiveOptBudget"] = None


class BudgetExceeded(RuntimeError):
    """Raised when a recursive optimization run exceeds its global budget."""


@dataclass
class RecursiveOptBudget:
    """Track optional global limits across recursive-opt levels.

    The budget is intentionally coarse-grained. `candidates` counts planned
    outer candidates (`iterations * num_candidates`) at recursive-opt entry
    points, while LLM counters are charged immediately before wrapped live LLM
    calls. This makes the limit easy to reason about without depending on
    internals of every Trace trainer implementation.
    """

    max_optimizer_llm_calls: Optional[int] = None
    max_eval_llm_calls: Optional[int] = None
    max_candidates: Optional[int] = None
    max_wall_time_s: Optional[float] = None
    stop_policy: str = "return_best"
    used_optimizer_llm_calls: int = 0
    used_eval_llm_calls: int = 0
    used_candidates: int = 0
    started_at: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self._validate_optional_int(self.max_optimizer_llm_calls, "max_optimizer_llm_calls")
        self._validate_optional_int(self.max_eval_llm_calls, "max_eval_llm_calls")
        self._validate_optional_int(self.max_candidates, "max_candidates")
        if self.max_wall_time_s is not None and self.max_wall_time_s < 0:
            raise ValueError("max_wall_time_s must be non-negative or None")
        if self.stop_policy not in _STOP_POLICIES:
            raise ValueError(
                "stop_policy must be one of "
                f"{sorted(_STOP_POLICIES)}, got {self.stop_policy!r}"
            )

    @property
    def enabled(self) -> bool:
        """Return True when at least one global resource has a finite limit."""
        return any(
            value is not None
            for value in (
                self.max_optimizer_llm_calls,
                self.max_eval_llm_calls,
                self.max_candidates,
                self.max_wall_time_s,
            )
        )

    @property
    def elapsed_s(self) -> float:
        """Return wall-clock seconds since this budget was created."""
        return time.monotonic() - self.started_at

    def remaining(self, resource: BudgetResource) -> Optional[int]:
        """Return remaining units for `resource`, or None when unlimited."""
        limit = self._limit_for(resource)
        if limit is None:
            return None
        return max(0, limit - self._used_for(resource))

    def charge(self, resource: BudgetResource, amount: int = 1) -> None:
        """Consume budget for `resource` and raise on overflow.

        Usage is recorded even when a resource is unlimited so notebooks and
        logs can report how much a run actually spent.
        """
        if not isinstance(amount, int):
            raise TypeError(f"budget amount must be an integer, got {type(amount).__name__}")
        if amount < 0:
            raise ValueError(f"budget amount must be non-negative, got {amount}")
        with self._lock:
            self._assert_wall_time_available()
            limit = self._limit_for(resource)
            used = self._used_for(resource)
            if limit is not None and used + amount > limit:
                raise BudgetExceeded(
                    f"recursive optimization budget exhausted for {resource}: "
                    f"requested {amount}, used {used}, limit {limit}. "
                    "Increase the matching RECURSIVE_OPT_MAX_* env var, set it "
                    "to none/unlimited, or lower per-level iteration/candidate limits."
                )
            self._set_used(resource, used + amount)

    def summary(self) -> str:
        """Return a compact human-readable budget status line."""
        parts = [
            self._format_counter(
                "optimizer_llm_calls",
                self.used_optimizer_llm_calls,
                self.max_optimizer_llm_calls,
            ),
            self._format_counter(
                "eval_llm_calls",
                self.used_eval_llm_calls,
                self.max_eval_llm_calls,
            ),
            self._format_counter("candidates", self.used_candidates, self.max_candidates),
        ]
        wall_limit = "unlimited" if self.max_wall_time_s is None else f"{self.max_wall_time_s:g}s"
        parts.append(f"wall_time={self.elapsed_s:.1f}s/{wall_limit}")
        parts.append(f"stop_policy={self.stop_policy}")
        prefix = "enabled" if self.enabled else "off"
        return f"{prefix}: " + ", ".join(parts)

    def _assert_wall_time_available(self) -> None:
        if self.max_wall_time_s is None:
            return
        if self.elapsed_s > self.max_wall_time_s:
            raise BudgetExceeded(
                "recursive optimization budget exhausted for wall_time: "
                f"elapsed {self.elapsed_s:.1f}s, limit {self.max_wall_time_s:g}s."
            )

    def _limit_for(self, resource: BudgetResource) -> Optional[int]:
        if resource == "optimizer_llm_calls":
            return self.max_optimizer_llm_calls
        if resource == "eval_llm_calls":
            return self.max_eval_llm_calls
        if resource == "candidates":
            return self.max_candidates
        raise ValueError(f"unknown budget resource {resource!r}")

    def _used_for(self, resource: BudgetResource) -> int:
        if resource == "optimizer_llm_calls":
            return self.used_optimizer_llm_calls
        if resource == "eval_llm_calls":
            return self.used_eval_llm_calls
        if resource == "candidates":
            return self.used_candidates
        raise ValueError(f"unknown budget resource {resource!r}")

    def _set_used(self, resource: BudgetResource, value: int) -> None:
        if resource == "optimizer_llm_calls":
            self.used_optimizer_llm_calls = value
            return
        if resource == "eval_llm_calls":
            self.used_eval_llm_calls = value
            return
        if resource == "candidates":
            self.used_candidates = value
            return
        raise ValueError(f"unknown budget resource {resource!r}")

    @staticmethod
    def _format_counter(name: str, used: int, limit: Optional[int]) -> str:
        limit_text = "unlimited" if limit is None else str(limit)
        return f"{name}={used}/{limit_text}"

    @staticmethod
    def _validate_optional_int(value: Optional[int], name: str) -> None:
        if value is None:
            return
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer or None")
        if value < 0:
            raise ValueError(f"{name} must be non-negative or None")


class BudgetedLLM:
    """Wrap an LLM callable and charge global budget before each provider call."""

    def __init__(self, llm: Any, resource: BudgetResource) -> None:
        self._llm = llm
        self.resource = resource

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        current_budget().charge(self.resource)
        return self._llm(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


def budgeted_llm(llm: Any, resource: Optional[BudgetResource]) -> Any:
    """Return `llm` wrapped with global budget charging when requested."""
    if resource is None:
        return llm
    return BudgetedLLM(llm, resource)


def configure_budget_from_env() -> RecursiveOptBudget:
    """Create and install the global recursive-opt budget from environment.

    Supported env vars:
      * `RECURSIVE_OPT_BUDGET_PRESET=demo|off|none|unlimited|custom`
      * `RECURSIVE_OPT_MAX_OPTIMIZER_LLM_CALLS`
      * `RECURSIVE_OPT_MAX_EVAL_LLM_CALLS`
      * `RECURSIVE_OPT_MAX_CANDIDATES`
      * `RECURSIVE_OPT_MAX_WALL_TIME_SECONDS`
      * `RECURSIVE_OPT_BUDGET_STOP_POLICY=return_best|raise`
    """
    global _GLOBAL_BUDGET
    _GLOBAL_BUDGET = _budget_from_env()
    return _GLOBAL_BUDGET


def current_budget() -> RecursiveOptBudget:
    """Return the active budget, lazily initialized from environment."""
    global _GLOBAL_BUDGET
    if _GLOBAL_BUDGET is None:
        _GLOBAL_BUDGET = _budget_from_env()
    return _GLOBAL_BUDGET


def reset_budget(budget: Optional[RecursiveOptBudget] = None) -> None:
    """Reset the active budget; pass a budget for tests or custom runners."""
    global _GLOBAL_BUDGET
    _GLOBAL_BUDGET = budget


def budget_status() -> str:
    """Return the active budget summary for banners/notebook output."""
    return current_budget().summary()


def _budget_from_env() -> RecursiveOptBudget:
    preset = os.environ.get("RECURSIVE_OPT_BUDGET_PRESET", "custom").strip().lower()
    defaults: dict[str, Optional[float | int]] = {}
    if preset == "demo":
        defaults = {
            "RECURSIVE_OPT_MAX_OPTIMIZER_LLM_CALLS": 64,
            "RECURSIVE_OPT_MAX_EVAL_LLM_CALLS": 80,
            "RECURSIVE_OPT_MAX_CANDIDATES": 16,
            "RECURSIVE_OPT_MAX_WALL_TIME_SECONDS": 300.0,
        }
    elif preset in {"off", "none", "null", "unlimited"}:
        defaults = {
            "RECURSIVE_OPT_MAX_OPTIMIZER_LLM_CALLS": None,
            "RECURSIVE_OPT_MAX_EVAL_LLM_CALLS": None,
            "RECURSIVE_OPT_MAX_CANDIDATES": None,
            "RECURSIVE_OPT_MAX_WALL_TIME_SECONDS": None,
        }
    elif preset not in {"", "custom"}:
        raise ValueError(
            "RECURSIVE_OPT_BUDGET_PRESET must be demo, custom, off, none, or unlimited"
        )

    stop_policy = os.environ.get("RECURSIVE_OPT_BUDGET_STOP_POLICY", "return_best").strip()
    if stop_policy not in _STOP_POLICIES:
        raise ValueError(
            "RECURSIVE_OPT_BUDGET_STOP_POLICY must be one of "
            f"{sorted(_STOP_POLICIES)}, got {stop_policy!r}"
        )
    return RecursiveOptBudget(
        max_optimizer_llm_calls=_int_limit_env(
            "RECURSIVE_OPT_MAX_OPTIMIZER_LLM_CALLS",
            defaults.get("RECURSIVE_OPT_MAX_OPTIMIZER_LLM_CALLS"),
        ),
        max_eval_llm_calls=_int_limit_env(
            "RECURSIVE_OPT_MAX_EVAL_LLM_CALLS",
            defaults.get("RECURSIVE_OPT_MAX_EVAL_LLM_CALLS"),
        ),
        max_candidates=_int_limit_env(
            "RECURSIVE_OPT_MAX_CANDIDATES",
            defaults.get("RECURSIVE_OPT_MAX_CANDIDATES"),
        ),
        max_wall_time_s=_float_limit_env(
            "RECURSIVE_OPT_MAX_WALL_TIME_SECONDS",
            defaults.get("RECURSIVE_OPT_MAX_WALL_TIME_SECONDS"),
        ),
        stop_policy=stop_policy,
    )


def _int_limit_env(name: str, default: Optional[float | int]) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None:
        return None if default is None else int(default)
    raw = raw.strip().lower()
    if raw in _UNLIMITED:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer or unlimited") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _float_limit_env(name: str, default: Optional[float | int]) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None:
        return None if default is None else float(default)
    raw = raw.strip().lower()
    if raw in _UNLIMITED:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative number or unlimited") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value
