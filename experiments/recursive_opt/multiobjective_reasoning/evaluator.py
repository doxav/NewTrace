"""Strict exact-output evaluator for Experiment 0."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from opto.trainer.objectives import EvaluationResult


EVALUATOR_REF = "recursive_experiments.evaluator.exact_reasoning@1"
EVALUATOR_MODE = "output"
EVALUATOR_EVENTS: list[dict[str, Any]] = []
_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_CHOICE = re.compile(r"\([A-E]\)", re.IGNORECASE)


def _normalize_numeric(value: str) -> str:
    matches = _NUMBER.findall(value)
    if not matches:
        return ""
    try:
        number = Decimal(matches[-1].replace(",", ""))
    except InvalidOperation:
        return ""
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"-0", "+0"} else normalized


def _extract(content: str, task_kind: str) -> str:
    final_lines = re.findall(r"(?im)^\s*FINAL(?:\s+ANSWER)?\s*:\s*(.+?)\s*$", content)
    candidate = final_lines[-1] if final_lines else content
    if task_kind == "choice":
        matches = _CHOICE.findall(candidate)
        return matches[-1].upper() if matches else ""
    if task_kind == "numeric":
        return _normalize_numeric(candidate)
    return " ".join(candidate.strip().lower().split())


def _sum_usage(calls: list[Mapping[str, Any]]) -> dict[str, float | int]:
    usage: dict[str, float | int] = {"calls": len(calls)}
    for call in calls:
        for name, amount in call.get("usage", {}).items():
            if isinstance(amount, (int, float)):
                usage[name] = usage.get(name, 0) + amount
    return usage


def exact_reasoning_evaluator(
    output: Any, example: Any, context: Mapping[str, Any]
) -> EvaluationResult:
    """Score one sample; an invalid extraction counts as INCORRECT, not as absent.

    The sample stays in the accuracy denominator. Excluding invalid samples instead
    would let an optimizer raise measured accuracy by driving hard questions to invalid
    output; keeping them makes that strategy strictly self-defeating. `invalid_rate` is
    reported alongside accuracy so the two are never conflated.
    """
    data = getattr(output, "data", output)
    item = getattr(example, "data", example)
    if not isinstance(data, Mapping) or not isinstance(item, Mapping):
        raise TypeError("exact reasoning evaluator requires mapping output/example")
    calls = [data["analysis"], data["answer_response"]]
    usage = _sum_usage(calls)
    content = str(data["answer_response"]["content"])
    produced = _extract(content, str(item["task_kind"]))
    expected = _extract(str(item["expected"]), str(item["task_kind"]))
    invalid = not produced
    correct = (not invalid) and produced == expected
    forward_tokens = int(usage.get("total_tokens", 0))
    denominator = int(item.get("baseline_forward_tokens", max(1, forward_tokens)))
    latency = sum(float(call["latency_s"]) for call in calls)
    event = {
        "sample_id": str(item["id"]),
        "phase": str(context["phase"]),
        "output_identity": id(output),
        "workflow_call": int(data["workflow_call"]),
    }
    EVALUATOR_EVENTS.append(event)
    return EvaluationResult(
        valid=not invalid,
        status="invalid" if invalid else "ok",
        metrics={
            "accuracy": float(correct),
            "invalid_rate": float(invalid),
            "forward_tokens": float(forward_tokens),
            "forward_tokens_per_example": float(forward_tokens),
            "forward_token_ratio": float(forward_tokens / max(1, denominator)),
            "latency_s": float(latency),
        },
        feedback=f"Expected answer: {expected}. Produced answer: {produced or '<invalid>'}.",
        trace={
            "sample_id": str(item["id"]),
            "workflow_call": int(data["workflow_call"]),
            "output_identity": id(output),
            "provider_calls": [dict(call["provider"]) for call in calls],
            # Persist the raw text ONLY when extraction failed. The 2026-08-24 main run
            # stopped on `invalid_rate <= 0` after one empty extraction, and its own
            # report had to record that "the raw provider response text is not persisted,
            # so the exact upstream formatting cause is unknown". An unanalysable stop
            # costs a whole run; a bounded excerpt costs a few hundred bytes.
            **({"invalid_raw_content": content[:2000],
                "invalid_raw_length": len(content)} if invalid else {}),
        },
        usage={"forward": usage},
        artifacts={"answer": produced, "expected": expected},
        error="deterministic answer extraction failed" if invalid else None,
    )


def clear_evaluator_events() -> None:
    EVALUATOR_EVENTS.clear()
