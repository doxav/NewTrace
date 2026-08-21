"""Multi-objective configuration and selection utilities.

Provides ObjectiveConfig and pure functions for multi-objective candidate
selection: weighted scalarization, Pareto ranking, and backward-compatible
scalar max.

All functions are pure (no side effects) and depend only on numpy, typing,
and dataclasses. No imports from opto.trainer to avoid circular dependencies.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np


# --- Type aliases ---
ScalarScore = float
VectorScore = Dict[str, float]
ScoreLike = Union[int, float, bool, Dict[str, float]]
_USAGE_ROLES = ("forward", "optimizer", "feedback", "judge")


@dataclass(frozen=True)
class EvaluationResult:
    """Canonical evaluator output with explicit validity and role usage."""

    valid: bool
    status: str
    metrics: Dict[str, float] = field(default_factory=dict)
    feedback: Any = ""
    trace: Any = None
    usage: Dict[str, Dict[str, Union[int, float]]] = field(default_factory=dict)
    artifacts: Any = field(default_factory=dict)
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TypeError("EvaluationResult.valid must be a boolean")
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("EvaluationResult.status must be a non-empty string")
        metrics = to_score_dict(self.metrics) if self.metrics else {}
        if self.valid and not metrics:
            raise ValueError("a valid EvaluationResult requires at least one metric")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("EvaluationResult.error must be a string or None")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "usage", _normalize_usage(self.usage))


def normalize_evaluation_result(value: Any) -> EvaluationResult:
    """Adapt canonical results and legacy float/dict/(score, feedback) shapes."""
    if isinstance(value, EvaluationResult):
        return value
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("legacy evaluation tuples must be (score, feedback)")
        normalized = normalize_evaluation_result(value[0])
        return EvaluationResult(
            valid=normalized.valid,
            status=normalized.status,
            metrics=normalized.metrics,
            feedback=value[1],
            trace=normalized.trace,
            usage=normalized.usage,
            artifacts=normalized.artifacts,
            error=normalized.error,
        )
    if isinstance(value, (bool, int, float)):
        return EvaluationResult(valid=True, status="ok", metrics=to_score_dict(value))
    if not isinstance(value, Mapping):
        raise TypeError(
            "evaluation result must be EvaluationResult, score, score dict, or (score, feedback)"
        )

    result_keys = {"valid", "status", "metrics", "feedback", "trace", "usage", "artifacts", "error"}
    if not (set(value) & result_keys):
        return EvaluationResult(valid=True, status="ok", metrics=to_score_dict(dict(value)))

    unknown = set(value) - result_keys - {"score"}
    if unknown:
        raise ValueError(f"unknown evaluation result keys: {sorted(unknown)}")
    raw_metrics = value.get("metrics")
    if raw_metrics is None and "score" in value:
        raw_score = value["score"]
        raw_metrics = raw_score if isinstance(raw_score, Mapping) else {"score": raw_score}
    metrics = to_score_dict(dict(raw_metrics)) if isinstance(raw_metrics, Mapping) else {}
    error = value.get("error")
    valid = value.get("valid")
    if valid is None:
        valid = error is None and value.get("status") not in {"invalid", "error"}
    status = value.get("status") or ("ok" if valid else "invalid")
    return EvaluationResult(
        valid=valid,
        status=status,
        metrics=metrics,
        feedback=value.get("feedback", ""),
        trace=value.get("trace"),
        usage=dict(value.get("usage") or {}),
        artifacts=value.get("artifacts", {}),
        error=error,
    )


def satisfies_hard_constraints(
    result: EvaluationResult, constraints: Sequence[Mapping[str, Any]]
) -> bool:
    """Return whether a valid result satisfies every declared metric constraint."""
    if not result.valid:
        return False
    comparisons = {
        "<": lambda actual, expected: actual < expected,
        "<=": lambda actual, expected: actual <= expected,
        "==": lambda actual, expected: actual == expected,
        "!=": lambda actual, expected: actual != expected,
        ">=": lambda actual, expected: actual >= expected,
        ">": lambda actual, expected: actual > expected,
    }
    for constraint in constraints:
        metric = constraint["metric"]
        operation = constraint["op"]
        if metric not in result.metrics:
            return False
        if operation not in comparisons:
            raise ValueError(f"unsupported hard-constraint operator {operation!r}")
        if not comparisons[operation](result.metrics[metric], float(constraint["value"])):
            return False
    return True


def select_evaluation_result(
    results: Sequence[Any],
    config: "ObjectiveConfig",
    hard_constraints: Sequence[Mapping[str, Any]] = (),
) -> EvaluationResult:
    """Apply validity/constraints before selecting with existing objective logic."""
    normalized = [normalize_evaluation_result(result) for result in results]
    feasible = [
        result for result in normalized
        if satisfies_hard_constraints(result, hard_constraints)
    ]
    if not feasible:
        raise ValueError("no valid evaluation result satisfies the hard constraints")
    index = select_best([(result.metrics, result) for result in feasible], config)
    return feasible[index]


def _normalize_usage(
    usage: Mapping[str, Mapping[str, Union[int, float]]]
) -> Dict[str, Dict[str, Union[int, float]]]:
    """Materialize non-negative runtime token usage for all supported roles."""
    if not isinstance(usage, Mapping):
        raise TypeError("EvaluationResult.usage must be a role mapping")
    unknown = set(usage) - set(_USAGE_ROLES)
    if unknown:
        raise ValueError(f"unknown EvaluationResult usage roles: {sorted(unknown)}")
    normalized: Dict[str, Dict[str, Union[int, float]]] = {}
    for role in _USAGE_ROLES:
        values = usage.get(role, {})
        if not isinstance(values, Mapping):
            raise TypeError(f"usage for role {role!r} must be a mapping")
        merged: Dict[str, Union[int, float]] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            **dict(values),
        }
        for name, amount in merged.items():
            if not isinstance(amount, (int, float)) or amount < 0:
                raise ValueError(f"usage {role}.{name} must be a non-negative number")
        normalized[role] = merged
    return normalized


@dataclass(frozen=True)
class ObjectiveConfig:
    """Immutable configuration for multi-objective candidate selection.

    Attributes:
        mode: Selection strategy.
            - "scalar": existing scalar comparison (default, backward-compatible).
            - "weighted": scalarize via weighted sum, then select max.
            - "pareto": Pareto dominance ranking with configurable tie-break.
        weights: Per-metric weights for weighted scalarization.
            Missing metrics use missing_value. Metrics not in weights are ignored.
            Empty dict in weighted mode -> equal weight 1.0 for all metrics.
        minimize: Frozenset of metric names where lower is better.
            These are negated internally ("higher-is-better" normalization).
            Users can pass a set; it is auto-converted to frozenset.
        missing_value: Score assigned to missing metrics (default: -inf).
        pareto_metrics: Subset of metrics for Pareto dominance.
            None -> use all metrics present across candidates.
        tie_break: Strategy for Pareto-equivalent candidates.
            - "weighted": fall back to weighted scalarization.
            - "lexicographic": sort by metric names alphabetically.
            - "random_seeded": seeded random shuffle.
        seed: Random seed for deterministic tie-breaking.

        scalarize_dict: How to reduce dict scores to a scalar (when mode="scalar").
            - "score": use score_key (default; avoids hidden behavior)
            - "mean": mean(values) (explicitly requested; diagnostic/backcompat)
            - "weighted": weighted_scalarize() (explicitly requested)
        score_key: Key used when scalarize_dict="score" (default: "score")
    """
    mode: str = "scalar"
    weights: Dict[str, float] = field(default_factory=dict)
    minimize: frozenset = field(default_factory=frozenset)
    missing_value: float = float("-inf")
    pareto_metrics: Optional[Tuple[str, ...]] = None
    tie_break: str = "weighted"
    seed: int = 0
    scalarize_dict: str = "score"
    score_key: str = "score"

    def __post_init__(self):
        if isinstance(self.minimize, set):
            object.__setattr__(self, 'minimize', frozenset(self.minimize))
        if self.mode not in ("scalar", "weighted", "pareto"):
            raise ValueError(
                f"mode must be 'scalar', 'weighted', or 'pareto', got '{self.mode}'"
            )
        if self.tie_break not in ("weighted", "lexicographic", "random_seeded"):
            raise ValueError(
                f"tie_break must be 'weighted', 'lexicographic', or "
                f"'random_seeded', got '{self.tie_break}'"
            )
        if self.scalarize_dict not in ("score", "mean", "weighted"):
            raise ValueError(
                f"scalarize_dict must be 'score', 'mean', or 'weighted', "
                f"got '{self.scalarize_dict}'"
            )
        if not isinstance(self.score_key, str) or not self.score_key:
            raise ValueError("score_key must be a non-empty string")
        for k, v in self.weights.items():
            if v < 0:
                raise ValueError(f"Weight for '{k}' must be non-negative, got {v}")
        if self.pareto_metrics is not None and len(self.pareto_metrics) == 0:
            raise ValueError(
                "pareto_metrics must be None (auto) or non-empty tuple"
            )


# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------

def to_score_dict(score: ScoreLike) -> Dict[str, float]:
    """Convert any score to dict form.

    - bool/int/float -> {"score": float(value)}
    - Dict[str, float] -> returned as-is (validated: all values finite)

    Raises:
        TypeError: if score is not int, float, bool, or dict.
        ValueError: if dict is empty or contains non-finite values.
    """
    if isinstance(score, bool):
        return {"score": float(score)}
    if isinstance(score, (int, float)):
        val = float(score)
        if not np.isfinite(val):
            raise ValueError(f"Score must be finite, got {score}")
        return {"score": val}
    if isinstance(score, dict):
        if len(score) == 0:
            raise ValueError("Score dict must not be empty")
        for k, v in score.items():
            if not isinstance(v, (int, float)) or not np.isfinite(float(v)):
                raise ValueError(
                    f"Score dict value for '{k}' must be finite float, got {v}"
                )
        return {k: float(v) for k, v in score.items()}
    raise TypeError(
        f"Score must be int, float, bool, or Dict[str, float], "
        f"got {type(score).__name__}"
    )


# Backward-compatible alias (deprecated name)
normalize_score = to_score_dict


def score_dict_to_scalar(score_dict: Dict[str, float],
                         config: ObjectiveConfig) -> float:
    """Reduce a score dict to a scalar according to ObjectiveConfig.

    Applies apply_minimize first, then reduces per config.scalarize_dict:
      - "score": return sd[config.score_key]
      - "mean": return mean(sd.values())
      - "weighted": return weighted_scalarize(sd, config.weights, ...)

    This exists to avoid hard-coding any dict->scalar behavior in Guide/Evaluator.
    """
    sd = to_score_dict(score_dict)
    sd = apply_minimize(sd, config.minimize)

    if config.scalarize_dict == "score":
        if config.score_key not in sd:
            raise ValueError(
                f"Dict score missing key '{config.score_key}'. "
                "Either include it, or set ObjectiveConfig.scalarize_dict "
                "to 'mean' or 'weighted'."
            )
        return float(sd[config.score_key])

    if config.scalarize_dict == "mean":
        return float(np.mean(list(sd.values())))

    if config.scalarize_dict == "weighted":
        return float(weighted_scalarize(sd, config.weights, config.missing_value))

    raise ValueError(f"Unknown scalarize_dict: {config.scalarize_dict}")


def to_scalar_score(score: ScoreLike,
                    config: Optional[ObjectiveConfig]) -> float:
    """Convert scalar or dict score to scalar using ObjectiveConfig.

    Scalar scores pass through as float(score). Dict scores require
    an explicit ObjectiveConfig to define reduction (no hidden defaults).
    """
    if isinstance(score, dict):
        if config is None:
            raise ValueError(
                "Dict score encountered but ObjectiveConfig is None. "
                "Pass ObjectiveConfig(mode='scalar', scalarize_dict=...) "
                "to define reduction."
            )
        return score_dict_to_scalar(score, config)
    return float(score)


def apply_minimize(score_dict: Dict[str, float],
                   minimize: frozenset) -> Dict[str, float]:
    """Negate values for minimize metrics (higher-is-better normalization).

    Returns a new dict; metrics not in *minimize* are unchanged.
    """
    return {k: -v if k in minimize else v for k, v in score_dict.items()}


def weighted_scalarize(score_dict: Dict[str, float],
                       weights: Dict[str, float],
                       missing_value: float = float("-inf")) -> float:
    """Compute weighted sum of score dict.

    If *weights* is empty, all present metrics get equal weight 1.0.
    Metrics in *score_dict* but NOT in *weights* are ignored.
    """
    if not weights:
        weights = {k: 1.0 for k in score_dict}
    total = 0.0
    for metric, weight in weights.items():
        value = score_dict.get(metric, missing_value)
        total += weight * value
    return total


def dominates(a: Dict[str, float], b: Dict[str, float],
              metrics: Optional[Tuple[str, ...]] = None) -> bool:
    """Check if candidate *a* Pareto-dominates candidate *b*.

    a dominates b iff:
      - a[m] >= b[m] for ALL metrics m, AND
      - a[m] >  b[m] for AT LEAST ONE metric m

    Both dicts must be in "higher-is-better" form (post apply_minimize).
    """
    if metrics is None:
        metrics = tuple(sorted(set(a.keys()) | set(b.keys())))
    at_least_one_better = False
    for m in metrics:
        va = a.get(m, float("-inf"))
        vb = b.get(m, float("-inf"))
        if va < vb:
            return False
        if va > vb:
            at_least_one_better = True
    return at_least_one_better


def pareto_rank(candidates: List[Dict[str, float]],
                metrics: Optional[Tuple[str, ...]] = None) -> List[int]:
    """Assign Pareto rank to each candidate (0 = non-dominated front).

    Uses standard non-dominated sorting.
    """
    n = len(candidates)
    ranks = [0] * n
    remaining = set(range(n))
    current_rank = 0

    while remaining:
        front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i != j and dominates(candidates[j], candidates[i], metrics):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        for i in front:
            ranks[i] = current_rank
            remaining.remove(i)
        current_rank += 1

    return ranks


def select_best(candidates: List[Tuple[ScoreLike, Any]],
                config: Optional[ObjectiveConfig] = None) -> int:
    """Select index of the single best candidate.

    Args:
        candidates: List of (score, payload) tuples.
        config: Selection config. None -> scalar max (backward-compatible).

    Returns:
        Index of the best candidate.

    Notes:
        When *config* is None or mode='scalar', dict scores are collapsed to
        a scalar using ObjectiveConfig.scalarize_dict. If dict scores are
        present and config is None, a ValueError is raised (no hidden
        hard-coded reduction).
    """
    if config is None or config.mode == "scalar":
        scores = [to_scalar_score(score, config) for score, _ in candidates]
        return int(np.argmax(scores))

    score_dicts = [to_score_dict(s) for s, _ in candidates]
    score_dicts = [apply_minimize(sd, config.minimize) for sd in score_dicts]

    if config.mode == "weighted":
        weighted = [
            weighted_scalarize(sd, config.weights, config.missing_value)
            for sd in score_dicts
        ]
        return int(np.argmax(weighted))

    if config.mode == "pareto":
        ranks = pareto_rank(score_dicts, config.pareto_metrics)
        front_indices = [i for i, r in enumerate(ranks) if r == 0]

        if len(front_indices) == 1:
            return front_indices[0]

        # Tie-break among front
        if config.tie_break == "weighted":
            front_scores = [
                weighted_scalarize(
                    score_dicts[i], config.weights, config.missing_value
                )
                for i in front_indices
            ]
            return front_indices[int(np.argmax(front_scores))]

        if config.tie_break == "lexicographic":
            metrics = sorted(score_dicts[front_indices[0]].keys())

            def lex_key(idx):
                return tuple(
                    score_dicts[idx].get(m, config.missing_value) for m in metrics
                )

            return max(front_indices, key=lex_key)

        if config.tie_break == "random_seeded":
            rng = np.random.RandomState(config.seed)
            return front_indices[rng.randint(len(front_indices))]

    raise ValueError(f"Unknown mode: {config.mode}")


def select_top_k(candidates: List[Tuple[ScoreLike, Any]],
                 config: Optional[ObjectiveConfig] = None,
                 k: int = 1) -> List[int]:
    """Select the top-k candidate indices.

    Same logic as select_best but returns *k* indices.
    For Pareto mode: rank-0 front first (up to k), then rank-1, etc.

    Notes:
        When *config* is None or mode='scalar', dict scores are collapsed to
        a scalar using ObjectiveConfig.scalarize_dict. If dict scores are
        present and config is None, a ValueError is raised (no hidden
        hard-coded reduction).
    """
    if config is None or config.mode == "scalar":
        scores = [to_scalar_score(score, config) for score, _ in candidates]
        return list(np.argsort(scores)[::-1][:k])

    score_dicts = [to_score_dict(s) for s, _ in candidates]
    score_dicts = [apply_minimize(sd, config.minimize) for sd in score_dicts]

    if config.mode == "weighted":
        weighted = [
            weighted_scalarize(sd, config.weights, config.missing_value)
            for sd in score_dicts
        ]
        return list(np.argsort(weighted)[::-1][:k])

    if config.mode == "pareto":
        ranks = pareto_rank(score_dicts, config.pareto_metrics)
        result: List[int] = []
        max_rank = max(ranks)
        for rank in range(max_rank + 1):
            rank_indices = [i for i, r in enumerate(ranks) if r == rank]
            if config.tie_break == "weighted":
                rank_indices.sort(
                    key=lambda i: weighted_scalarize(
                        score_dicts[i], config.weights, config.missing_value
                    ),
                    reverse=True,
                )
            elif config.tie_break == "lexicographic":
                metrics = (
                    sorted(score_dicts[rank_indices[0]].keys())
                    if rank_indices else []
                )
                rank_indices.sort(
                    key=lambda i: tuple(
                        score_dicts[i].get(m, config.missing_value)
                        for m in metrics
                    ),
                    reverse=True,
                )
            elif config.tie_break == "random_seeded":
                rng = np.random.RandomState(config.seed + rank)
                rng.shuffle(rank_indices)
            result.extend(rank_indices)
            if len(result) >= k:
                break
        return result[:k]

    raise ValueError(f"Unknown mode: {config.mode}")


def aggregate_score_dicts(score_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute per-metric mean across a list of score dicts.

    This is Objective-side policy (per reviewer): evaluators call this
    rather than defining aggregation logic themselves.
    """
    if not score_dicts:
        return {}
    all_keys = set()
    for sd in score_dicts:
        all_keys.update(sd.keys())
    result: Dict[str, float] = {}
    for key in sorted(all_keys):
        values = [sd[key] for sd in score_dicts if key in sd and sd[key] is not None]
        if values:
            result[key] = float(np.mean(values))
    return result
