"""Triggers, deferred (online) policy evaluation, and the strategy archive.

Deferred-evaluation contract: a deployed policy is *pending* until its window closes
(next trigger or end of run). Its score is computed from what happened while it was
active and only then written to its archive entry (status ``scored``). Windows are
non-stationary: they start wherever the search currently is.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def resolve_patience(patience: Union[int, str], horizon: int, ratio: float = 0.10) -> int:
    """'auto' = SkyDiscover's max(1, int(horizon * switch_ratio))."""
    if patience == 'auto':
        return max(1, int(horizon * ratio))
    if not isinstance(patience, int) or isinstance(patience, bool) or patience < 1:
        raise ValueError('patience must be a positive integer or "auto"')
    return patience


class StagnationTrigger:
    """Fires after ``patience`` consecutive updates without a gain > threshold (EvoX _should_evolve_search)."""

    def __init__(self, patience: int, threshold: float = 0.01) -> None:
        self.patience, self.threshold = patience, threshold
        self.count, self.last = 0, None

    def update(self, best: float) -> bool:
        if self.last is None or best - self.last > self.threshold:
            self.count = 0
        else:
            self.count += 1
        self.last = best
        if self.count >= self.patience:
            self.count = 0
            return True
        return False


class PeriodicTrigger:
    """Fires every ``every`` updates."""

    def __init__(self, every: int) -> None:
        self.every, self.count = every, 0

    def update(self, best: float) -> bool:
        self.count += 1
        return self.count % self.every == 0


class NeverTrigger:
    def update(self, best: float) -> bool:
        return False


def make_trigger(kind: str, patience: int, threshold: float):
    if kind == 'stagnation':
        return StagnationTrigger(patience, threshold)
    if kind == 'periodic':
        return PeriodicTrigger(patience)
    if kind == 'never':
        return NeverTrigger()
    raise ValueError(f'unknown trigger {kind!r}')


# ------------------------------------------------------------------ window scorers

class LogWindowScorer:
    """SkyDiscover LogWindowScorer: gain * (1 + log(1 + start)) / sqrt(horizon), horizon fixed."""

    def __init__(self, horizon: int) -> None:
        self.horizon = horizon

    def score(self, start: float, steps: Sequence[float], start_iteration: Optional[int] = None) -> Dict[str, Any]:
        running = start
        for value in steps:
            running = max(running, float(value))
        horizon = int(self.horizon) if self.horizon else max(1, len(steps))
        combined = (running - start) * (1.0 + math.log(1.0 + max(0.0, start))) / math.sqrt(horizon)
        return {'combined_score': combined, 'window_start_iteration': start_iteration, 'search_window_start_score': start,
                'search_window_end_score': running, 'search_horizon': horizon}


class GainScorer(LogWindowScorer):
    """Plain best-score gain over the window."""

    def score(self, start: float, steps: Sequence[float], start_iteration: Optional[int] = None) -> Dict[str, Any]:
        metrics = super().score(start, steps, start_iteration)
        metrics['combined_score'] = metrics['search_window_end_score'] - start
        return metrics


class PairedScorer:
    """Challenger minus incumbent rate of new global bests over one interleaved window."""

    def score_records(self, records: Sequence[Dict[str, Any]]) -> float:
        groups = {tag: [bool(r['new_best']) for r in records if r['tag'] == tag] for tag in ('challenger', 'incumbent')}
        if not groups['challenger'] or not groups['incumbent']:
            return 0.0
        return sum(groups['challenger']) / len(groups['challenger']) - sum(groups['incumbent']) / len(groups['incumbent'])


def interleave(length: int, seed: Any) -> List[str]:
    """Balanced random challenger/incumbent schedule."""
    tags = ['challenger', 'incumbent'] * (length // 2) + (['challenger'] if length % 2 else [])
    random.Random(str(seed)).shuffle(tags)
    return tags


# ------------------------------------------------------------------ archive

@dataclass
class ArchiveEntry:
    id: str
    source: str
    iteration: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = 'pending'  # pending | scored | rejected
    parent_id: Optional[str] = None
    source_kind: str = 'proposal'  # initial | proposal
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> Optional[float]:
        value = self.metrics.get('combined_score')
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def to_dict(self) -> Dict[str, Any]:
        return {'id': self.id, 'iteration': self.iteration, 'metrics': dict(self.metrics), 'status': self.status, 'parent_id': self.parent_id,
                'source_kind': self.source_kind, 'source': self.source}


class StrategyArchive:
    """Scored policies. Parent selection: 'best' (EvoX) or 'current'; context: random others."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.entries: Dict[str, ArchiveEntry] = {}
        self.rng = random.Random(seed)
        self._ids = itertools.count()

    def new_entry(self, source: str, iteration: int, parent_id: Optional[str] = None, source_kind: str = 'proposal') -> ArchiveEntry:
        return ArchiveEntry(id=f'policy_{next(self._ids)}', source=source, iteration=iteration, parent_id=parent_id, source_kind=source_kind)

    def add(self, entry: ArchiveEntry) -> None:
        self.entries[entry.id] = entry

    def __len__(self) -> int:
        return len(self.entries)

    def select(self, num_context: int = 2, mode: str = 'best', current: Optional[ArchiveEntry] = None) -> Tuple[ArchiveEntry, List[ArchiveEntry]]:
        entries = list(self.entries.values())
        if not entries:
            raise ValueError('archive is empty')
        if mode == 'current' and current is not None:
            parent = current
        elif mode == 'best':
            parent = max(entries, key=lambda e: e.score if e.score is not None else float('-inf'))
        else:
            raise ValueError(f'unknown parent selection {mode!r}')
        sample = self.rng.sample(entries, max(0, min(num_context, len(entries))))
        return parent, [e for e in sample if e.id != parent.id]


class DeferredEvaluation:
    """One open scoring window and at most one pending policy entry."""

    def __init__(self, scorer: LogWindowScorer) -> None:
        self.scorer = scorer
        self.start: Optional[float] = None
        self.start_iteration: Optional[int] = None
        self.steps: List[float] = []
        self.pending: Optional[ArchiveEntry] = None

    def reset(self, start: float, start_iteration: Optional[int] = None) -> None:
        self.start, self.start_iteration, self.steps = float(start), start_iteration, []

    def record(self, best: float) -> None:
        if self.start is None:
            self.reset(best)
        self.steps.append(float(best))

    def window_metrics(self, current_best: float, start_override: Optional[float] = None, start_iteration: Optional[int] = None) -> Dict[str, Any]:
        start = self.start if start_override is None else start_override
        steps = self.steps if self.steps else [current_best]
        return self.scorer.score(start or 0.0, steps, self.start_iteration if start_iteration is None else start_iteration)

    def attach(self, entry: ArchiveEntry) -> None:
        entry.status = 'pending'
        self.pending = entry

    def close(self, current_best: float, metrics: Optional[Dict[str, Any]] = None) -> Optional[ArchiveEntry]:
        entry, self.pending = self.pending, None
        if entry is None:
            return None
        entry.metrics.update(metrics if metrics is not None else self.window_metrics(current_best))
        entry.status = 'scored'
        return entry
