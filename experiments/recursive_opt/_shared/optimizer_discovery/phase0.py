"""Frozen Phase-0 calibration and a portable optimizer evaluation entry point."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from opto.features.recursive_opt import spec as control
from opto.features.recursive_opt.measurement import (
    is_transient_provider_error,
    menu_evidence,
)
from opto.features.recursive_opt.optimizer_program import (
    optimizer_spec,
    parse_program,
    propose_point,
)
from opto.features.recursive_opt.runmode import _response_usage, make_live_llm

ROOT = Path(__file__).resolve().parent
SPEC = json.loads((ROOT / "phase0_spec.json").read_text())
ENGINEERING_SPEC = json.loads((ROOT / "engineering_smoke_spec.json").read_text())
CALIBRATION_SPEC = json.loads((ROOT / "generation_calibration_spec.json").read_text())
REQUESTS = [
    *SPEC["live"]["requests"],
    ENGINEERING_SPEC["label"],
    *CALIBRATION_SPEC["requests"],
]
PROMPT = """Write a complete portable optimizer.py file exporting exactly
propose(history, bounds, seed). It proposes one point for a black-box MINIMIZATION
problem. history is a list of past observations, each with exactly x (a list of
coordinates) and value (a finite number; lower is better). bounds is a nonempty
list of finite [low, high] pairs. seed is an integer held fixed across the run.
Return one finite in-bounds list of coordinates, of length len(bounds).
Use only the Python standard library. The function must have exactly these three
positional parameters, without defaults. Each invocation is a fresh process, so
all state comes from history and seed. Use random.Random(seed + len(history)) if
randomness is needed; repeated identical arguments must give identical points.
You cannot see or call the objective, an LLM, or any train/validation/holdout data.
Do not read files, network, environment, clocks, or process state. Keep execution
under two seconds. Generate a simple reasonable optimizer using prior observations
when available. No performance threshold is required. Return only one Python code
block containing the entire file, with no explanations."""


def _safe(text: str) -> str:
    """Remove known parent credentials and recognizable secret strings from evidence."""
    for name, value in os.environ.items():
        if value and any(
            marker in name.upper()
            for marker in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
        ):
            text = text.replace(value, "<redacted>")
    return re.sub(r"sk-[A-Za-z0-9_-]+", "<redacted>", text)


def _write(path: Path, value: Any) -> None:
    """Persist sanitized JSON evidence in an already reserved run directory."""
    path.write_text(_safe(json.dumps(value, indent=2, allow_nan=False)) + "\n")


def run_request(
    client: Any, label: str, directory: Path, *, seeds: list[int], budget: int
) -> dict[str, Any]:
    """Run one preregistered request, retaining every retry and invalid candidate."""
    if label not in REQUESTS:
        raise ValueError("request label must be preregistered")
    calibration = CALIBRATION_SPEC["requests"].get(label)
    if calibration and (
        seeds != CALIBRATION_SPEC["fixture_seeds"]
        or budget != CALIBRATION_SPEC["fixture_budget"]
    ):
        raise ValueError("calibration fixture seeds and budget must remain frozen")
    if (
        calibration
        and hashlib.sha256(PROMPT.encode()).hexdigest()
        != CALIBRATION_SPEC["prompt_sha256"]
    ):
        raise ValueError("calibration prompt must remain frozen")
    settings = {
        name: SPEC["live"][name]
        for name in ("temperature", "top_p", "max_tokens", "seed")
    }
    timeout = SPEC["live"]["request_timeout_s"]
    if calibration:
        settings.update(CALIBRATION_SPEC["configs"][calibration["config"]])
        settings["seed"] = calibration["seed"]
        timeout = CALIBRATION_SPEC["request_timeout_s"]
    provider_settings = dict(settings)
    effort = provider_settings.pop("reasoning_effort", None)
    if effort is not None:
        provider_settings["extra_body"] = {"reasoning": {"effort": effort}}
    directory.mkdir(parents=True, exist_ok=False)
    request = {
        "model": SPEC["live"]["model"],
        "provider": "openrouter",
        "messages": [
            {
                "role": "user",
                "content": (
                    ENGINEERING_SPEC["prompt"]
                    if label == ENGINEERING_SPEC["label"]
                    else PROMPT
                ),
            }
        ],
        **settings,
        "request_timeout_s": timeout,
        "concurrency": 1,
    }
    _write(directory / "request.json", request)
    result: dict[str, Any] = {
        "label": label,
        "provider_status": "pending",
        "attempts": [],
        "evaluations": [],
    }
    response = None
    for index in range(4):
        stream = io.StringIO()
        started = time.monotonic()
        try:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                response = client(
                    messages=request["messages"], timeout=timeout, **provider_settings
                )
            attempt = {"attempt": index + 1, "status": "success"}
        except Exception as error:  # noqa: BLE001 - preserve every provider failure
            attempt = {
                "attempt": index + 1,
                "status": "provider_error",
                "error_type": type(error).__name__,
                "error": _safe(str(error)),
                "transient": is_transient_provider_error(error),
                "status_code": getattr(error, "status_code", None),
            }
        attempt["wall_s"] = round(time.monotonic() - started, 3)
        attempt["logs"] = _safe(stream.getvalue())[:16000]
        result["attempts"].append(attempt)
        _write(directory / f"attempt_{index + 1}.json", attempt)
        _write(directory / "result.json", result)
        if response is not None:
            break
        if not attempt.get("transient") or index == 3:
            result["provider_status"] = "provider_error"
            _write(directory / "result.json", result)
            return result
        time.sleep(SPEC["live"]["retry_delays_s"][index])
    result["provider_status"] = "success"
    content = control._optimizer_response_text(response)
    metadata = {
        name: getattr(response, name, None)
        for name in ("id", "model", "created", "system_fingerprint")
    }
    metadata["usage"] = _response_usage(response)
    raw_usage = getattr(response, "usage", None)
    if hasattr(raw_usage, "model_dump"):
        raw_usage = raw_usage.model_dump()
    if isinstance(raw_usage, dict):
        details = raw_usage.get("completion_tokens_details") or {}
        for name, value in {
            "reasoning_tokens": details.get("reasoning_tokens"),
            "cost_usd": raw_usage.get("cost_usd", raw_usage.get("cost")),
        }.items():
            if value is not None:
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(
                        f"provider usage {name} must be finite and non-negative"
                    )
                metadata["usage"][name] = value
    metadata["finish_reason"] = getattr(response.choices[0], "finish_reason", None)
    metadata["content"] = content
    _write(directory / "response.json", metadata)
    result["response_metadata"] = {
        key: value for key, value in metadata.items() if key != "content"
    }
    try:
        source = parse_program(content)
    except ValueError as error:
        result["parse_status"] = "invalid"
        result["parse_error"] = str(error)
    else:
        result["parse_status"] = "valid"
        source = _safe(source)
        (directory / "optimizer.py").write_text(source)
        result["artifact_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        for seed in seeds:
            raw = optimizer_spec(source, seed=seed, budget=budget)
            raw["outputs"] = {"directory": str(directory / f"seed_{seed}")}
            evaluation = control.execute_plan(control.compile_plan(raw))[0]
            result["evaluations"].append({"seed": seed, **evaluation.to_dict()})
            _write(directory / "result.json", result)
        if calibration:
            result["history_probe"] = history_probe(source)
    _write(directory / "result.json", result)
    return result


def history_probe(source: str) -> dict[str, Any]:
    """Measure response to reversed value rankings without changing length or seed."""
    pairs = []
    for length in CALIBRATION_SPEC["probe_lengths"]:
        histories = [
            [
                {
                    "x": [-4 + 8 * i / (length - 1), 3 - 6 * i / (length - 1)],
                    "value": length - 1 - i if reverse else i,
                }
                for i in range(length)
            ]
            for reverse in (False, True)
        ]
        for seed in CALIBRATION_SPEC["probe_seeds"]:
            proposals = [
                propose_point(source, h, [[-5, 5], [-5, 5]], seed) for h in histories
            ]
            pairs.append(
                {
                    "length": length,
                    "seed": seed,
                    "histories": histories,
                    "proposals": [
                        {"status": p.status, "point": p.point} for p in proposals
                    ],
                    "valid": all(p.valid for p in proposals),
                    "changed": all(p.valid for p in proposals)
                    and proposals[0].point != proposals[1].point,
                }
            )
    valid = all(pair["valid"] for pair in pairs)
    return {
        "valid": valid,
        "responsive": valid and any(p["changed"] for p in pairs),
        "pairs": pairs,
    }


def readiness_gate(
    rows: list[dict[str, Any]], menu: dict[str, Any], expected: list[str], *, phase: str
) -> dict[str, Any]:
    """Apply the preregistered batch gate without excluding failures or missing runs."""
    if phase not in ("pilot", "confirmation"):
        raise ValueError("readiness phase must be pilot or confirmation")
    thresholds = CALIBRATION_SPEC[f"{phase}_gate"]
    complete = sorted(row["label"] for row in rows) == sorted(expected)
    valid = sum(bool(row["generation_valid"]) for row in rows)
    responsive = sum(
        bool(row["generation_valid"] and row["history_responsive"]) for row in rows
    )
    passed = (
        complete
        and valid >= thresholds["valid"]
        and responsive >= thresholds["history_responsive"]
        and menu.get("behavior_equivalence_known") is True
        and (menu.get("effective_menu_size") or 0) >= thresholds["effective_menu_size"]
    )
    return {
        "passed": passed,
        "complete": complete,
        "expected_count": len(expected),
        "valid_count": valid,
        "history_responsive_count": responsive,
        "thresholds": thresholds,
        "menu_evidence": menu,
    }


def summarize_readiness(directory: Path, config: str, phase: str) -> dict[str, Any]:
    """Recompute readiness from retained canonical evaluations and provider evidence."""
    if config not in CALIBRATION_SPEC["pilot_order"] or phase not in (
        "pilot",
        "confirmation",
    ):
        raise ValueError("unknown readiness config or phase")
    expected = [
        label
        for label, item in CALIBRATION_SPEC["requests"].items()
        if item["config"] == config and item["phase"] == phase
    ]
    rows, observations = [], []
    for label in expected:
        path = directory / label / "result.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        metadata = result.get("response_metadata", {})
        tokens = metadata.get("usage", {}).get("completion_tokens")
        evaluations = result["evaluations"]
        fixture_valid = [row["seed"] for row in evaluations] == CALIBRATION_SPEC[
            "fixture_seeds"
        ] and all(
            row["valid"]
            and row["evaluation"]["artifacts"][0]["evaluated_count"]
            == CALIBRATION_SPEC["fixture_budget"]
            for row in evaluations
        )
        generation_valid = (
            result["provider_status"] == "success"
            and metadata.get("finish_reason") == "stop"
            and fixture_valid
            and result.get("history_probe", {}).get("valid", False)
            and type(tokens) is int
            and 0 < tokens <= CALIBRATION_SPEC["configs"][config]["max_tokens"]
            and bool(result["attempts"])
            and result["attempts"][-1]["wall_s"]
            <= CALIBRATION_SPEC["request_timeout_s"]
        )
        rows.append(
            {
                "label": label,
                "generation_valid": generation_valid,
                "history_responsive": result.get("history_probe", {}).get(
                    "responsive", False
                ),
                "usage": metadata.get("usage", {}),
                "wall_s": sum(attempt["wall_s"] for attempt in result["attempts"]),
                "finish_reason": metadata.get("finish_reason"),
            }
        )
        for row in evaluations:
            observation = {
                "candidate": {"sha256": result["artifact_sha256"]},
                "example": {"seed": row["seed"]},
                "phase": "fit",
                "valid": generation_valid,
                "metrics": {},
            }
            if generation_valid:
                observation["behavior_signature"] = row["evaluation"]["artifacts"][0][
                    "behavior_signature"
                ]
            observations.append(observation)
    menu = menu_evidence(observations, declared_menu_size=len(expected))
    return {
        "config": config,
        "phase": phase,
        "rows": rows,
        **readiness_gate(rows, menu, expected, phase=phase),
    }


def _load_key() -> None:
    """Read the local credential source privately; never persist its path or contents."""
    from dotenv import dotenv_values

    values = dotenv_values(Path.cwd() / ".env")
    key = os.environ.get("OPENROUTER_API_KEY") or values.get("OPENROUTER_API_KEY")
    if not key and values.get("OPENROUTER_API_KEY_SOURCE"):
        try:
            match = re.search(
                r"sk-or-v1-[A-Za-z0-9_-]+",
                Path(values["OPENROUTER_API_KEY_SOURCE"]).read_text(),
            )
            key = match.group(0) if match else None
        except OSError:
            raise RuntimeError(
                "local OpenRouter credential source is inaccessible"
            ) from None
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is unavailable")
    os.environ["OPENROUTER_API_KEY"] = key


def main() -> None:
    """Run frozen live labels or evaluate one optimizer artifact offline."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    live = subparsers.add_parser("calibrate")
    live.add_argument("--requests", nargs="+", choices=REQUESTS, required=True)
    live.add_argument("--output", type=Path, default=ROOT / "live")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--program", type=Path, required=True)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--budget", type=int, default=8)
    summary = subparsers.add_parser("readiness-summary")
    summary.add_argument("--output", type=Path, default=ROOT / "live_generation")
    summary.add_argument(
        "--config", choices=CALIBRATION_SPEC["pilot_order"], required=True
    )
    summary.add_argument("--phase", choices=["pilot", "confirmation"], required=True)
    args = parser.parse_args()
    if args.command == "readiness-summary":
        result = summarize_readiness(args.output, args.config, args.phase)
        _write(args.output / f"{args.phase}_{args.config}_summary.json", result)
        print(json.dumps(result, indent=2))
        return
    if args.command == "evaluate":
        result = control.execute_plan(
            control.compile_plan(
                optimizer_spec(
                    args.program.read_text(), seed=args.seed, budget=args.budget
                )
            )
        )[0]
        print(json.dumps(result.to_dict(), indent=2))
        return
    _load_key()
    client = make_live_llm(
        model="openrouter/" + SPEC["live"]["model"],
        cache=False,
        max_retries=1,
        empty_response_retries=0,
        request_timeout_s=SPEC["live"]["request_timeout_s"],
        allow_env_overrides=False,
        budget_resource=None,
    )
    for label in args.requests:
        result = run_request(
            client,
            label,
            args.output / label,
            seeds=SPEC["fixture"]["seeds"],
            budget=SPEC["fixture"]["budget"],
        )
        print(
            json.dumps(
                {
                    "request": label,
                    "provider_status": result["provider_status"],
                    "valid_evaluations": sum(
                        row["valid"] for row in result["evaluations"]
                    ),
                }
            )
        )


if __name__ == "__main__":
    main()
