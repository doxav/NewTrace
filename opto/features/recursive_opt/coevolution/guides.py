"""Guides for co-evolution: separate the guiding score from the raw benchmark metric.

A benchmark metric can be exploitable (e.g. averaging only over solved cases rewards
crashing on hard cases). ``GuidedEvaluator`` wraps an evaluator, computes the score the
search should follow (``guided_score`` by default), enforces hard constraints, and keeps
every raw metric for reporting. ``format_case_diagnostics`` turns per-case records from a
white-box evaluator into targeted feedback (worst cases, failures by cause).
"""

from __future__ import annotations

import operator as op
from collections import Counter
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

Evaluate = Callable[[str], Tuple[Dict[str, Any], Dict[str, Any]]]
_OPS = {'>=': op.ge, '>': op.gt, '<=': op.le, '<': op.lt, '==': op.eq}


class GuidedEvaluator:
    """``evaluate(source) -> (metrics, artifacts)`` with a guiding score and hard constraints.

    violation='penalize' keeps the candidate with ``penalty_score``; 'reject' marks it invalid
    (validity=0) so the operator retries. ``score_fn(metrics, artifacts)`` defines the guide score
    (default: ``metrics[source_key]``).
    """

    def __init__(self, evaluate: Evaluate, score_key: str = 'guided_score', source_key: str = 'combined_score',
                 score_fn: Optional[Callable[[Mapping[str, Any], Mapping[str, Any]], float]] = None,
                 hard_constraints: Optional[Mapping[str, Tuple[str, float]]] = None, violation: str = 'penalize',
                 penalty_score: float = 0.0, feedback_fn: Optional[Callable[[Mapping[str, Any], Mapping[str, Any]], str]] = None) -> None:
        if violation not in {'penalize', 'reject'}:
            raise ValueError("violation must be 'penalize' or 'reject'")
        for name, (symbol, _) in (hard_constraints or {}).items():
            if symbol not in _OPS:
                raise ValueError(f'unknown comparison {symbol!r} for constraint {name!r}')
        self.evaluate, self.score_key, self.source_key, self.score_fn = evaluate, score_key, source_key, score_fn
        self.constraints, self.violation, self.penalty_score, self.feedback_fn = dict(hard_constraints or {}), violation, penalty_score, feedback_fn

    def violations(self, metrics: Mapping[str, Any]) -> list:
        failed = []
        for name, (symbol, bound) in self.constraints.items():
            value = metrics.get(name)
            if not isinstance(value, (int, float)) or not _OPS[symbol](value, bound):
                failed.append(f'{name}={value} (required {symbol} {bound})')
        return failed

    def __call__(self, source: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        metrics, artifacts = self.evaluate(source)
        metrics, artifacts = dict(metrics), dict(artifacts or {})
        if metrics.get('validity') in (0, -1):
            return metrics, artifacts  # already a failed evaluation: leave it to the operator
        failed = self.violations(metrics)
        if failed and self.violation == 'reject':
            metrics.update(validity=0, error='hard constraint violated: ' + '; '.join(failed))
            return metrics, artifacts
        score = self.score_fn(metrics, artifacts) if self.score_fn else metrics.get(self.source_key, 0.0)
        metrics[self.score_key] = self.penalty_score if failed else float(score)
        if failed:
            artifacts['constraints'] = 'Hard constraints violated (guide score set to the penalty): ' + '; '.join(failed)
        if self.feedback_fn:
            artifacts['feedback'] = self.feedback_fn(metrics, artifacts)
        return metrics, artifacts


def _fmt(value: Any) -> str:
    return f'{value:.4g}' if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)


def format_case_diagnostics(cases: Sequence[Mapping[str, Any]], worst: int = 5, lower_is_better: bool = True, value_key: str = 'value',
                            bound_key: str = 'bound', detail_keys: Sequence[str] = ()) -> str:
    """Per-case feedback: failures grouped by cause, and the worst solved cases (relative gap to a bound when given)."""
    solved = [c for c in cases if c.get('ok') and isinstance(c.get(value_key), (int, float))]
    failed = [c for c in cases if not c.get('ok')]
    lines = [f'{len(solved)}/{len(cases)} cases solved.']
    if failed:
        causes = Counter(str(c.get('error') or 'unknown').split(':')[0] for c in failed)
        lines.append(f'{len(failed)} case(s) failed: ' + ', '.join(f'{cause} x{count}' for cause, count in causes.most_common())
                     + '. Example: case ' + f"{failed[0].get('case')}: {str(failed[0].get('error'))[:200]}")
    if solved:
        def gap(c: Mapping[str, Any]) -> float:
            bound = c.get(bound_key)
            if isinstance(bound, (int, float)) and bound:
                return c[value_key] / bound if lower_is_better else bound / c[value_key]
            return c[value_key] if lower_is_better else -c[value_key]
        ranked = sorted(solved, key=gap, reverse=True)[:worst]
        if any(isinstance(c.get(bound_key), (int, float)) for c in solved):
            mean_gap = sum(gap(c) for c in solved) / len(solved)
            lines.append(f'Mean ratio to the per-case bound: {mean_gap:.3f} (1.0 = optimal bound reached).')
        lines.append('Worst cases:')
        for c in ranked:
            details = ', '.join(f'{k}={_fmt(c.get(k))}' for k in detail_keys if k in c)
            bound = f", bound={_fmt(c.get(bound_key))}, ratio={gap(c):.3f}" if isinstance(c.get(bound_key), (int, float)) else ''
            lines.append(f"  case {c.get('case')}: {value_key}={_fmt(c[value_key])}{bound}{(' | ' + details) if details else ''}")
    return '\n'.join(lines)
