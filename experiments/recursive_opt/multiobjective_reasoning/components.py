"""Compound prompt-only reasoning module for Experiment 0."""

from __future__ import annotations

import time
from typing import Any, Mapping

from opto.trace import bundle, node
from opto.trace.modules import Module


MODULE_REF = "recursive_experiments.module.compound_reasoning@1"
ARTIFACT_KEYS = frozenset({"analysis_instruction", "answer_instruction"})
FORWARD_EVENTS: list[dict[str, Any]] = []


def _plain(value: Any) -> Any:
    return getattr(value, "data", value)


def _usage_dict(response: Any) -> dict[str, float | int]:
    raw = getattr(response, "usage", None)
    if raw is None:
        return {}
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    elif not isinstance(raw, Mapping) and hasattr(raw, "dict"):
        raw = raw.dict()
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, float | int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = raw.get(key)
        if isinstance(value, (int, float)) and value >= 0:
            result[key] = int(value)
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, Mapping):
        cost = hidden.get("response_cost")
        if isinstance(cost, (int, float)) and cost >= 0:
            result["cost_usd"] = float(cost)
    return result


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if content is not None:
            return str(content)
    if isinstance(response, Mapping):
        try:
            return str(response["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError):
            pass
    return str(response)


def _provider_metadata(response: Any) -> dict[str, Any]:
    hidden = getattr(response, "_hidden_params", None)
    hidden = hidden if isinstance(hidden, Mapping) else {}
    return {
        "actual_model": str(getattr(response, "model", "") or hidden.get("model_id", "")),
        "actual_provider": str(hidden.get("custom_llm_provider", "")),
        "cache_hit": hidden.get("cache_hit"),
    }


@bundle()
def _analysis_prompt(instruction: str, question: str) -> str:
    return f"{instruction.strip()}\n\nQuestion:\n{question.strip()}"


@bundle()
def _answer_prompt(instruction: str, question: str, analysis_call: Mapping[str, Any]) -> str:
    analysis = str(analysis_call["content"])
    return (
        f"{instruction.strip()}\n\nQuestion:\n{question.strip()}\n\n"
        f"Analysis from the first stage:\n{analysis}\n\n"
        "Return exactly one final line in the form FINAL: <answer>."
    )


@bundle()
def _record_call(prompt: str, content: str, usage: Mapping[str, Any], latency_s: float,
                 metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "content": content,
        "usage": dict(usage),
        "latency_s": float(latency_s),
        "provider": dict(metadata),
    }


@bundle()
def _structured_output(example: Mapping[str, Any], analysis_call: Mapping[str, Any],
                       answer_call: Mapping[str, Any], workflow_call: int) -> dict[str, Any]:
    return {
        "sample_id": str(example["id"]),
        "task_kind": str(example["task_kind"]),
        "analysis": dict(analysis_call),
        "answer_response": dict(answer_call),
        "workflow_call": int(workflow_call),
    }


class CompoundReasoningModule(Module):
    """Two-stage traced workflow with two trainable prompt parameters."""

    def __init__(self, analysis_instruction: str, answer_instruction: str, forward_client: Any) -> None:
        self.analysis_instruction = node(
            analysis_instruction,
            name="analysis_instruction",
            trainable=True,
            description="Instructions for reasoning about the question before answering.",
        )
        self.answer_instruction = node(
            answer_instruction,
            name="answer_instruction",
            trainable=True,
            description="Instructions for converting the analysis into one exact final answer.",
        )
        self.forward_client = forward_client

    def _call(self, prompt: Any) -> Any:
        if self.forward_client is None:
            raise RuntimeError("compound reasoning requires the forward LLM role")
        started = time.perf_counter()
        response = self.forward_client(
            messages=[{"role": "user", "content": str(_plain(prompt))}]
        )
        latency = time.perf_counter() - started
        return _record_call(
            prompt,
            _response_text(response),
            _usage_dict(response),
            latency,
            _provider_metadata(response),
        )

    def forward(self, example: Any) -> Any:
        item = _plain(example)
        if not isinstance(item, Mapping) or not isinstance(item.get("question"), str):
            raise TypeError("compound reasoning examples require a question mapping")
        analysis_prompt = _analysis_prompt(self.analysis_instruction, item["question"])
        analysis_call = self._call(analysis_prompt)
        answer_prompt = _answer_prompt(
            self.answer_instruction, item["question"], analysis_call
        )
        answer_call = self._call(answer_prompt)
        event = {"sample_id": str(item["id"]), "output": None}
        FORWARD_EVENTS.append(event)
        output = _structured_output(item, analysis_call, answer_call, len(FORWARD_EVENTS))
        event["output"] = output
        return output


def build_module(level: Mapping[str, Any], resources: Mapping[str, Any]) -> Module:
    config = level["module"]["config"]
    return CompoundReasoningModule(
        str(config["analysis_instruction"]),
        str(config["answer_instruction"]),
        resources.get("llm_clients", {}).get("forward"),
    )


def snapshot_module(module: Module) -> dict[str, Any]:
    if not isinstance(module, CompoundReasoningModule):
        raise TypeError("compound reasoning snapshot requires CompoundReasoningModule")
    return {
        "analysis_instruction": str(module.analysis_instruction.data),
        "answer_instruction": str(module.answer_instruction.data),
    }


def restore_module(module: Module, artifact: Mapping[str, Any]) -> None:
    validate_artifact(artifact)
    if not isinstance(module, CompoundReasoningModule):
        raise TypeError("compound reasoning restore requires CompoundReasoningModule")
    module.analysis_instruction._set(artifact["analysis_instruction"])
    module.answer_instruction._set(artifact["answer_instruction"])


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    if not isinstance(artifact, Mapping) or set(artifact) != ARTIFACT_KEYS:
        raise ValueError("compound artifact requires exactly both instruction fields")
    if any(not isinstance(artifact[key], str) or not artifact[key].strip() for key in ARTIFACT_KEYS):
        raise TypeError("compound artifact instructions must be non-empty strings")


def validate_config(config: Mapping[str, Any]) -> None:
    validate_artifact(config)


def clear_forward_events() -> None:
    FORWARD_EVENTS.clear()
