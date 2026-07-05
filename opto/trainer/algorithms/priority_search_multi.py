from __future__ import annotations

from typing import Any

from opto.optimizers.optimizer import Optimizer
from opto.trainer.algorithms.priority_search import ModuleCandidate, PrioritySearch


class PrioritySearchMulti(PrioritySearch):
    """PrioritySearch variant that emits generic optimizer events on search stalls."""

    def __init__(self, *args: Any, emit_optimizer_events: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.emit_optimizer_events = emit_optimizer_events

    def train(self, *args: Any, emit_optimizer_events: bool = True, **kwargs: Any) -> Any:
        """Train like PrioritySearch, optionally notifying optimizers about search stalls."""
        self.emit_optimizer_events = emit_optimizer_events
        return super().train(*args, **kwargs)

    def exploit(self, verbose: bool = False, **kwargs: Any) -> tuple[ModuleCandidate, float, dict[str, Any]]:
        previous_priority = getattr(self, "_best_candidate_priority", None)
        candidate, priority, info = super().exploit(verbose=verbose, **kwargs)
        if self._should_emit_search_event(previous_priority, priority):
            event = "search_regression" if float(priority) < float(previous_priority) else "search_stall"
            self._emit_optimizer_event(
                event,
                previous_best_priority=previous_priority,
                best_priority=priority,
                best_candidate=candidate,
            )
        return candidate, priority, info

    def _should_emit_search_event(self, previous_priority: Any, priority: Any) -> bool:
        """Return True when the best validated priority did not improve."""
        if not self.emit_optimizer_events or previous_priority is None or priority is None:
            return False
        try:
            return float(priority) <= float(previous_priority)
        except (TypeError, ValueError):
            return False

    def _emit_optimizer_event(self, event: str, **context: Any) -> None:
        """Notify every reachable optimizer exposing ``on_search_event``."""
        seen: set[int] = set()

        def emit(optimizer: Any) -> None:
            if not isinstance(optimizer, Optimizer) or id(optimizer) in seen:
                return
            seen.add(id(optimizer))
            hook = getattr(optimizer, "on_search_event", None)
            if callable(hook):
                hook(event=event, trainer=self, **context)

        for optimizer in getattr(self, "_optimizers", []):
            emit(optimizer)
        for candidate in self._reachable_candidates(context.get("best_candidate")):
            emit(candidate.optimizer)

    def _reachable_candidates(self, *candidates: Any) -> list[ModuleCandidate]:
        """Return candidates currently visible to the search policy."""
        found: list[ModuleCandidate] = []
        for candidate in (*candidates, *(getattr(self, "_exploration_candidates", None) or [])):
            if isinstance(candidate, ModuleCandidate):
                found.append(candidate)
        for memory_name in ("short_term_memory", "long_term_memory"):
            memory = getattr(self, memory_name, None)
            if memory is None:
                continue
            found.extend(candidate for _priority, candidate in memory if isinstance(candidate, ModuleCandidate))
        return found
