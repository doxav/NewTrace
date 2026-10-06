"""Declared shared state for co-evolution: the candidate population.

The population is owned by the co-evolution engine and shared by both levels: the O0
operator writes candidates into it, and every deployed selection policy (O1) reads it
and observes additions. Swapping or rolling back a policy never resets it.
``statistics`` reproduces SkyDiscover's ``ProgramDatabase.get_statistics`` so that a
policy's feedback can include the same per-decision execution trace.
"""

from __future__ import annotations

import copy
import statistics as stats
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple


@dataclass
class Candidate:
    """One evaluated solution and the decision that produced it."""

    id: str
    content: str
    metrics: Dict[str, Any]
    iteration: int
    parent_id: Optional[str] = None
    context_ids: Tuple[str, ...] = ()
    label: str = ''
    context_labels: Tuple[str, ...] = ()
    artifacts: Dict[str, Any] = field(default_factory=dict)
    changes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Population:
    """Insertion-ordered candidate store with snapshot/restore and EvoX-compatible statistics."""

    def __init__(self, score_key: str = 'combined_score') -> None:
        self.score_key = score_key
        self._members: Dict[str, Candidate] = {}

    # ---- access
    @property
    def members(self) -> List[Candidate]:
        return list(self._members.values())

    def __len__(self) -> int:
        return len(self._members)

    def __contains__(self, candidate_id: str) -> bool:
        return candidate_id in self._members

    def get(self, candidate_id: str) -> Optional[Candidate]:
        return self._members.get(candidate_id)

    def score(self, candidate: Candidate) -> Optional[float]:
        value = candidate.metrics.get(self.score_key)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def add(self, candidate: Candidate) -> None:
        if candidate.id in self._members:
            raise ValueError(f'duplicate candidate id {candidate.id!r}')
        self._members[candidate.id] = candidate

    def remove(self, candidate_id: str) -> None:
        self._members.pop(candidate_id, None)

    def best(self) -> Optional[Candidate]:
        """First candidate with the maximum score (ties keep the earliest, like SkyDiscover)."""
        best, best_score = None, None
        for candidate in self._members.values():
            score = self.score(candidate)
            if score is not None and (best_score is None or score > best_score):
                best, best_score = candidate, score
        return best

    def best_score(self, default: float = 0.0) -> float:
        best = self.best()
        return self.score(best) if best is not None else default

    # ---- rollback support
    def snapshot(self) -> Tuple[Tuple[str, Candidate], ...]:
        return tuple((key, copy.deepcopy(value)) for key, value in self._members.items())

    def restore(self, snapshot: Tuple[Tuple[str, Candidate], ...]) -> None:
        self._members = {key: copy.deepcopy(value) for key, value in snapshot}

    # ---- statistics (port of SkyDiscover ProgramDatabase.get_statistics)
    def statistics(self, num_recent_iterations: int = 100, k: int = 20, improvement_threshold: float = 0.10) -> Dict[str, Any]:
        members = self.members
        population_size = len(members)
        last_iteration = max((c.iteration for c in members if isinstance(c.iteration, int)), default=0)
        scores = [s for s in (self.score(c) for c in members) if s is not None]
        if scores:
            ordered = sorted(scores, reverse=True)
            if len(ordered) >= 4:
                q25, q50, q75 = stats.quantiles(scores, n=4)
            else:
                q25 = q50 = q75 = stats.median(scores)
            n = len(scores)
            summary: Dict[str, Any] = {'best': ordered[0], 'q75': q75, 'q50': q50, 'q25': q25, 'worst': ordered[-1]}
            summary['score_tiers'] = {
                'top': {'threshold': f'score >= {q75:.4f}', 'pct_programs': sum(s >= q75 for s in scores) / n * 100},
                'upper_mid': {'threshold': f'{q50:.4f} <= score < {q75:.4f}', 'pct_programs': sum(q50 <= s < q75 for s in scores) / n * 100},
                'lower_mid': {'threshold': f'{q25:.4f} <= score < {q50:.4f}', 'pct_programs': sum(q25 <= s < q50 for s in scores) / n * 100},
                'bottom': {'threshold': f'score < {q25:.4f}', 'pct_programs': sum(s < q25 for s in scores) / n * 100},
            }
            summary['unique_scores'] = len({round(s, 4) for s in scores})
            top_scores = ordered[:k]
        else:
            summary = {'best': None, 'q75': None, 'q50': None, 'q25': None, 'worst': None}
            top_scores = []
        with_parents = [c for c in members if c.parent_id is not None]
        unique_parents = len({c.parent_id for c in with_parents})
        avg_per_parent = len(with_parents) / unique_parents if unique_parents else 0.0
        without_improvement = 0
        if scores:
            best = max(scores)
            near = [c for c in members if self.score(c) is not None and self.score(c) >= best - improvement_threshold]
            if near:
                without_improvement = last_iteration - min(c.iteration for c in near if isinstance(c.iteration, int))
        recent = sorted((c for c in members if self.score(c) is not None and isinstance(c.iteration, int) and c.iteration > last_iteration - num_recent_iterations),
                        key=lambda c: c.iteration)
        recent_stats: Dict[str, Any] = {}
        if members:
            trace, trajectory, parent_scores = [], [], []
            for candidate in recent:
                trajectory.append(self.score(candidate))
                if candidate.parent_id is not None:
                    parent = self.get(candidate.parent_id)
                    parent_score = self.score(parent) if parent is not None else None
                    parent_scores.append(parent_score)
                    parent_tuple = (candidate.label, candidate.parent_id, parent_score)
                else:
                    parent_scores.append(None)
                    parent_tuple = None
                contexts = []
                for index, context_id in enumerate(candidate.context_ids):
                    label = candidate.context_labels[index] if index < len(candidate.context_labels) else ''
                    context = self.get(context_id)
                    contexts.append((label, context_id, self.score(context) if context is not None else None))
                trace.append({'iteration': candidate.iteration, 'program': (candidate.id, self.score(candidate)), 'parent': parent_tuple, 'context': contexts or None})
            parents = [c for c in recent if c.parent_id is not None]
            reuse_parent_ratio, reuse_parent_score, reuse_context_ratio, reuse_context_score = 0.0, None, 0.0, None
            if parents:
                top_id, count = Counter(c.parent_id for c in parents).most_common(1)[0]
                reuse_parent_ratio = count / len(parents)
                reuse_parent_score = self.score(self.get(top_id)) if top_id in self else None
            with_context = [c for c in recent if c.context_ids]
            if with_context:
                top_id, count = Counter(i for c in with_context for i in c.context_ids).most_common(1)[0]
                reuse_context_ratio = count / len(with_context)
                reuse_context_score = self.score(self.get(top_id)) if top_id in self else None
            recent_stats = {'num_recent_iterations': min(num_recent_iterations, len(trajectory)), 'execution_trace': trace, 'score_trajectory': trajectory,
                            'parent_scores': parent_scores, 'iterations_without_improvement': without_improvement, 'improvement_threshold': improvement_threshold,
                            'most_reused_parent_ratio': reuse_parent_ratio, 'most_reused_parent_score': reuse_parent_score,
                            'most_reused_context_ratio': reuse_context_ratio, 'most_reused_context_score': reuse_context_score}
        return {'previous_programs': recent, 'population_size': population_size, 'solution_score_summary': summary,
                'avg_solutions_per_parent': avg_per_parent, 'top_solution_scores': top_scores, 'recent_solution_stats': recent_stats}


def filter_statistics(statistics: Mapping[str, Any], horizon: int) -> Dict[str, Any]:
    """Keep only the last ``horizon`` trajectory entries (SkyDiscover filter_db_stats_by_horizon)."""
    if not statistics or horizon <= 0:
        return dict(statistics or {})
    filtered = dict(statistics)
    recent = statistics.get('recent_solution_stats')
    if recent:
        recent = dict(recent)
        for key in ('execution_trace', 'score_trajectory', 'parent_scores'):
            if recent.get(key) and len(recent[key]) > horizon:
                recent[key] = recent[key][-horizon:]
        recent['num_recent_iterations'] = min(horizon, recent.get('num_recent_iterations', 0))
        filtered['recent_solution_stats'] = recent
    return filtered
