"""Causal-effect contract for trainable fields (configurable, adapter-driven).

Replaces the binary "plumbed / not plumbed" notion: a field does NOT need to
directly change the final score to be a valid recursive knob; it needs a real
causal path to at least one relevant intermediate object:

    knob -> artifact / optimization trajectory / feedback / trace / memory /
            budget|search -> candidate or selected artifact -> (maybe) score

So validation asks "does this field have an ACTIVE declared effect under the
current run mode?", never "does it directly move the score?". Fields like
``trace_type`` (feedback-plumbed) or ``num_threads`` (budget/search-plumbed)
are first-class; only fields with NO active path are rejected, and even that is
configurable (``allow_inactive`` / required-effects policy per experiment).

Adapters opt in by exposing ``field_effects() -> dict[str, FieldEffect]``.
Adapters that only expose the legacy ``PLUMBED_FIELDS`` tuple are auto-mapped
(each field -> ARTIFACT+SCORE), so nothing existing breaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple


class Effect(str, Enum):
    ARTIFACT = "artifact"          # changes the prompt/code/context evaluated
    OPTIMIZATION = "optimization"  # changes the inner training trajectory
    FEEDBACK = "feedback"          # changes optimizer-visible evidence
    TRACE = "trace"                # changes trace construction/recording
    MEMORY = "memory"              # changes retrieval/promotion/warm-start
    BUDGET = "budget"              # changes cost/runtime
    SEARCH = "search"              # changes schedule/coverage/parallelism
    SCORE = "score"                # can change the reported score
    NONE = "none"


@dataclass(frozen=True)
class FieldEffect:
    """Declared causal contract for one config field.

    ``active`` reports whether the path is live under the adapter's CURRENT
    run mode (e.g. trainer is conditional on ``inner_steps > 0``);
    ``condition`` documents when it becomes active.
    """

    field: str
    effects: Tuple[Effect, ...]
    active: bool = True
    condition: str = ""
    probe_values: Tuple[Any, ...] = ()
    notes: str = ""


@dataclass
class EffectReport:
    """Outcome of checking requested trainable fields against a contract."""

    requested: List[str]
    active: Dict[str, Tuple[Effect, ...]] = field(default_factory=dict)
    inactive: Dict[str, str] = field(default_factory=dict)   # field -> condition
    undeclared: List[str] = field(default_factory=list)

    def ok(self) -> bool:
        return not self.inactive and not self.undeclared


class InactiveFieldError(ValueError):
    """A trainable field has no active causal path under the current run mode."""


def effects_for(adapter: Any) -> Dict[str, FieldEffect]:
    """Adapter contract with graceful fallbacks (keeps old adapters working)."""
    if adapter is None:
        return {}
    fe = getattr(adapter, "field_effects", None)
    if callable(fe):
        return dict(fe())
    plumbed = getattr(adapter, "PLUMBED_FIELDS", None)
    if plumbed:
        # Legacy binary contract: declared fields are artifact+score active.
        return {
            f: FieldEffect(f, effects=(Effect.ARTIFACT, Effect.SCORE))
            for f in plumbed
        }
    return {}


def check_field_effects(
    adapter: Any,
    fields: Iterable[str],
    *,
    required_effects: Optional[Iterable[Effect]] = None,
    allow_inactive: bool = False,
) -> EffectReport:
    """Validate that requested trainable fields have an active causal path.

    - ``required_effects``: optionally restrict which effect kinds count as
      relevant for THIS experiment (e.g. {Effect.MEMORY} for a memory study);
      a field is then "active" only if it has an active effect in that set.
    - ``allow_inactive``: report instead of raise (diagnostic / opt-out mode).

    Raises ``InactiveFieldError`` listing each dead field WITH the condition
    that would activate it (so the error is the documentation).
    """
    contract = effects_for(adapter)
    req = {Effect(e) for e in required_effects} if required_effects else None
    report = EffectReport(requested=list(fields))
    for f in report.requested:
        decl = contract.get(f)
        if decl is None:
            if contract:           # adapter declared a contract; field unknown
                report.undeclared.append(f)
            else:                  # no contract at all: nothing to enforce
                report.active[f] = (Effect.NONE,)
            continue
        effs = tuple(decl.effects)
        relevant = effs if req is None else tuple(e for e in effs if e in req)
        if decl.active and relevant and Effect.NONE not in relevant:
            report.active[f] = relevant
        else:
            why = decl.condition or "no declared effect"
            if req is not None and decl.active and not relevant:
                why = f"active, but none of its effects {[e.value for e in effs]} is in required {sorted(e.value for e in req)}"
            report.inactive[f] = why
    if not allow_inactive and not report.ok():
        dead = {**{f: c for f, c in report.inactive.items()},
                **{f: "not declared by the adapter contract" for f in report.undeclared}}
        raise InactiveFieldError(
            "trainable fields with no ACTIVE causal path under the current run mode: "
            + "; ".join(f"{f!r} ({why})" for f, why in dead.items())
            + ". Activate the path (see each condition), pick other fields, or set "
              "allow_inactive/allow_unplumbed to proceed deliberately."
        )
    return report
