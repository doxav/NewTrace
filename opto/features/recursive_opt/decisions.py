"""Structured decision parsing and guard-based scoring for recursive_opt.

The optimizer often emits small policies as text or code-returned strings
(``action: promote`` / JSON / bullet lists).  This module gives examples and
notebooks one DRY parser plus a deterministic guard evaluator, so promotion,
tool-routing, campaign, and trace-design policies can share the same contract.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass
class GuardedDecision:
    """Normalized decision extracted from JSON, key-value text, or free text."""

    action: str = ""
    target: str = ""
    tools: Tuple[str, ...] = ()
    reason: str = ""
    confidence: Optional[float] = None
    values: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""

    def text(self) -> str:
        """Return one lowercase searchable text view of the decision."""
        parts = [self.raw, self.action, self.target, self.reason, " ".join(self.tools)]
        parts.extend(f"{key}: {value}" for key, value in self.values.items())
        return "\n".join(str(part) for part in parts if part).lower()


@dataclass
class GuardedDecisionScore:
    """Score breakdown for a parsed guarded decision."""

    score: float
    decision: GuardedDecision
    action_score: float = 1.0
    target_score: float = 1.0
    avoid_score: float = 1.0
    numeric_score: float = 1.0
    required_score: float = 1.0
    forbidden_score: float = 1.0
    forbidden_hits: Tuple[str, ...] = ()
    matched_actions: Tuple[str, ...] = ()
    matched_targets: Tuple[str, ...] = ()


@dataclass
class GuardedDecisionCase:
    """One guarded policy example with expected actions/targets/terms."""

    name: str
    payload: Any = None
    allowed_actions: Tuple[str, ...] = ()
    required_targets: Tuple[str, ...] = ()
    forbidden_targets: Tuple[str, ...] = ()
    required_terms: Tuple[str, ...] = ()
    forbidden_terms: Tuple[str, ...] = ()
    numeric_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    required_denominator: Optional[int] = None
    hard_forbidden: bool = False
    forbidden_floor: float = 0.0

    def score(self, value: Any) -> GuardedDecisionScore:
        """Score ``value`` against this case's constraints."""

        return score_guarded_decision(
            value,
            allowed_actions=self.allowed_actions,
            required_targets=self.required_targets,
            forbidden_targets=self.forbidden_targets,
            required_terms=self.required_terms,
            forbidden_terms=self.forbidden_terms,
            numeric_ranges=self.numeric_ranges,
            weights=self.weights or None,
            required_denominator=self.required_denominator,
            hard_forbidden=self.hard_forbidden,
            forbidden_floor=self.forbidden_floor,
        )


@dataclass
class GuardedDecisionEvaluator:
    """Callable evaluator for code/policy components that return decisions."""

    cases: Tuple[GuardedDecisionCase, ...]

    def __call__(self, component: Callable[..., Any], _task_id: Any = None) -> Tuple[float, str]:
        """Evaluate ``component`` over all cases and return mean score + feedback."""

        if not self.cases:
            raise ValueError("GuardedDecisionEvaluator requires at least one case")
        scores: List[float] = []
        feedback: List[str] = []
        for case in self.cases:
            raw = _invoke_component(component, case.payload)
            result = case.score(raw)
            scores.append(result.score)
            feedback.append(guarded_feedback(case.name, result))
        return sum(scores) / len(scores), " | ".join(feedback)


@dataclass
class ConfidenceGate:
    """Evidence gate for retest/promote decisions under noisy measurements."""

    min_support: int = 2
    z: float = 1.0
    min_gain: float = 0.0
    mean_key: str = "mean_score"
    baseline_key: str = "initial"
    std_key: str = "std"
    support_key: str = "n"

    def __post_init__(self) -> None:
        if self.min_support <= 0:
            raise ValueError("min_support must be positive")
        if self.z < 0:
            raise ValueError("z must be non-negative")

    def lower_bound(self, evidence: Mapping[str, Any]) -> Optional[float]:
        """Return a one-sided lower confidence bound for the measured mean."""

        mean = _optional_float(evidence.get(self.mean_key))
        if mean is None:
            return None
        support = self.support(evidence)
        std = _optional_float(evidence.get(self.std_key)) or 0.0
        return float(mean - self.z * std / math.sqrt(max(1, support)))

    def support(self, evidence: Mapping[str, Any]) -> int:
        """Return positive integer support count from evidence."""

        value = _optional_float(evidence.get(self.support_key))
        return max(0, int(value or 0))

    def needs_retest(self, evidence: Mapping[str, Any]) -> bool:
        """Whether support is too weak for a final promote/reject decision."""

        return self.support(evidence) < self.min_support

    def clears_promotion(self, evidence: Mapping[str, Any]) -> bool:
        """Whether lower bound clears baseline by ``min_gain``."""

        lower = self.lower_bound(evidence)
        baseline = _optional_float(evidence.get(self.baseline_key))
        if lower is None or baseline is None:
            return False
        return lower >= baseline + self.min_gain


def parse_guarded_decision(value: Any) -> GuardedDecision:
    """Parse a generated decision from dict, JSON, key-value text, or free text.

    The parser is intentionally permissive: unknown fields are preserved in
    ``values`` and free text remains searchable through ``raw``.  Strictness
    belongs in :func:`score_guarded_decision`, where use cases can define their
    own guards without forcing one schema on all policies.

    #TODO: Impose a stricter schema for structured decisions, e.g. JSON or key-value text
    """

    raw = _raw_text(value)
    data = _load_mapping(value)
    if data is None:
        data = _parse_key_value_text(raw)

    action = _first_text(data, ("action", "decision", "operation", "route"))
    target = _first_text(data, ("target", "task", "family", "trace_type", "format", "strategy"))
    reason = _first_text(data, ("reason", "rationale", "why", "hint", "notes"))
    tools = _parse_tools(data.get("tools") or data.get("tool_policy") or data.get("use"))
    confidence = _optional_float(
        data.get("confidence", data.get("score", data.get("probability")))
    )

    return GuardedDecision(
        action=action,
        target=target,
        tools=tools,
        reason=reason,
        confidence=confidence,
        values=dict(data),
        raw=raw,
    )


def score_guarded_decision(
    value: Any,
    *,
    allowed_actions: Sequence[str] = (),
    required_targets: Sequence[str] = (),
    forbidden_targets: Sequence[str] = (),
    required_terms: Sequence[str] = (),
    forbidden_terms: Sequence[str] = (),
    numeric_ranges: Optional[Mapping[str, Tuple[float, float]]] = None,
    weights: Optional[Mapping[str, float]] = None,
    required_denominator: Optional[int] = None,
    hard_forbidden: bool = False,
    forbidden_floor: float = 0.0,
) -> GuardedDecisionScore:
    """Score a generated decision against reusable guard constraints.

    Parameters are deliberately generic:
    ``allowed_actions`` covers promote/retest/reject/control or probe/switch;
    ``required_targets`` covers task ids, tools, trace types, or formats;
    ``numeric_ranges`` covers fields such as ``max_examples``.
    ``hard_forbidden`` makes a forbidden hit cap the total score, which is useful
    for safety gates such as "never promote invalid syntax".
    """

    decision = parse_guarded_decision(value)
    text = decision.text()
    weights_dict = _normalized_weights(
        weights,
        active={
            "action": bool(allowed_actions),
            "target": bool(required_targets),
            "avoid": bool(forbidden_targets),
            "numeric": bool(numeric_ranges),
            "required": bool(required_terms),
            "forbidden": bool(forbidden_terms),
        },
    )

    matched_actions = _matched_terms(text, allowed_actions)
    action_score = 1.0 if not allowed_actions else float(bool(matched_actions))

    matched_targets = _matched_targets(decision, required_targets)
    target_score = (
        1.0
        if not required_targets
        else min(1.0, len(matched_targets) / max(1, len(tuple(required_targets))))
    )

    forbidden_target_hits = _matched_targets(decision, forbidden_targets)
    avoid_score = 1.0 if not forbidden_target_hits else 0.0

    numeric_score = _score_numeric_ranges(decision, numeric_ranges or {})

    required_hits = _matched_terms(text, required_terms)
    denominator = required_denominator
    if denominator is None:
        denominator = min(2, len(tuple(required_terms))) if required_terms else 0
    required_score = (
        1.0
        if not required_terms
        else min(1.0, len(required_hits) / max(1, int(denominator)))
    )

    forbidden_hits = tuple(
        dict.fromkeys((*_matched_terms(text, forbidden_terms), *forbidden_target_hits))
    )
    forbidden_score = 0.0 if forbidden_hits else 1.0

    score = (
        weights_dict["action"] * action_score
        + weights_dict["target"] * target_score
        + weights_dict["avoid"] * avoid_score
        + weights_dict["numeric"] * numeric_score
        + weights_dict["required"] * required_score
        + weights_dict["forbidden"] * forbidden_score
    )
    if hard_forbidden and forbidden_hits:
        score = min(score, float(forbidden_floor))

    return GuardedDecisionScore(
        score=float(score),
        decision=decision,
        action_score=action_score,
        target_score=target_score,
        avoid_score=avoid_score,
        numeric_score=numeric_score,
        required_score=required_score,
        forbidden_score=forbidden_score,
        forbidden_hits=forbidden_hits,
        matched_actions=matched_actions,
        matched_targets=matched_targets,
    )


def guarded_feedback(label: str, result: GuardedDecisionScore) -> str:
    """Return compact optimizer feedback from a guarded decision score."""

    decision = result.decision
    return (
        f"{label}: score={result.score:.2f}; action={decision.action or '-'}; "
        f"target={decision.target or '-'}; tools={list(decision.tools)}; "
        f"action_score={result.action_score:.2f}; target_score={result.target_score:.2f}; "
        f"avoid_score={result.avoid_score:.2f}; numeric_score={result.numeric_score:.2f}; "
        f"required_score={result.required_score:.2f}; forbidden_hits={list(result.forbidden_hits)}; "
        f"reason={decision.reason[:120]!r}"
    )


def make_guarded_decision_evaluator(
    cases: Sequence[GuardedDecisionCase],
) -> GuardedDecisionEvaluator:
    """Return a typed evaluator from reusable guarded decision cases."""

    return GuardedDecisionEvaluator(tuple(cases))


def keyword_present(text: str, term: str) -> bool:
    """Return whether ``term`` appears without false positives like do_not_promote."""

    key = str(term).strip().lower()
    haystack = str(text).lower()
    if not key:
        return False
    if len(key) <= 5 or any(ch in key for ch in " _:/-"):
        return key in haystack
    return re.search(rf"(?<![a-z0-9_]){re.escape(key)}(?![a-z0-9_])", haystack) is not None


def _raw_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def _load_mapping(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return {}
    try:
        loaded = json.loads(text)
    except Exception:
        return None
    return dict(loaded) if isinstance(loaded, Mapping) else None


def _parse_key_value_text(text: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for line in str(text).splitlines():
        clean = line.strip().strip("-* ")
        if not clean or ":" not in clean:
            continue
        key, rest = clean.split(":", 1)
        normalized_key = key.strip().lower().replace(" ", "_")
        if normalized_key:
            data[normalized_key] = rest.strip().strip("'\"`")
    if "raw" not in data:
        data["raw"] = text
    return data


def _first_text(data: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return str(value).strip()
    return ""


def _parse_tools(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        tokens = re.split(r"[\s,]+", value)
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, Mapping)):
        tokens = [str(item) for item in value]
    else:
        tokens = [str(value)]
    out = []
    for token in tokens:
        clean = token.strip().strip("'\"`")
        if clean and clean not in out:
            out.append(clean)
    return tuple(out)


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _matched_terms(text: str, terms: Sequence[str]) -> Tuple[str, ...]:
    return tuple(term for term in terms if keyword_present(text, term))


def _matched_targets(decision: GuardedDecision, targets: Sequence[str]) -> Tuple[str, ...]:
    if not targets:
        return ()
    search = decision.text()
    return tuple(target for target in targets if keyword_present(search, target))


def _score_numeric_ranges(
    decision: GuardedDecision,
    numeric_ranges: Mapping[str, Tuple[float, float]],
) -> float:
    if not numeric_ranges:
        return 1.0
    hits = 0
    for field_name, (lower, upper) in numeric_ranges.items():
        value = _optional_float(decision.values.get(field_name))
        if value is not None and float(lower) <= value <= float(upper):
            hits += 1
    return hits / len(numeric_ranges)


def _invoke_component(component: Callable[..., Any], payload: Any) -> Any:
    """Call a policy component with optional payload."""

    if payload is None:
        return component()
    return component(payload)


def _normalized_weights(
    weights: Optional[Mapping[str, float]],
    *,
    active: Mapping[str, bool],
) -> Dict[str, float]:
    keys = ("action", "target", "avoid", "numeric", "required", "forbidden")
    if weights is None:
        enabled = [key for key in keys if active.get(key)]
        if not enabled:
            return {key: 0.0 for key in keys}
        share = 1.0 / len(enabled)
        return {key: share if key in enabled else 0.0 for key in keys}
    out = {key: float(weights.get(key, 0.0)) for key in keys}
    total = sum(value for value in out.values() if value > 0)
    if total <= 0:
        return {key: 0.0 for key in keys}
    return {key: (value / total if value > 0 else 0.0) for key, value in out.items()}
