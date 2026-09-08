"""Frozen Phase-0 calibration and a portable optimizer evaluation entry point."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from opto.features.recursive_opt import spec as control
from opto.features.recursive_opt.measurement import is_transient_provider_error
from opto.features.recursive_opt.optimizer_program import optimizer_spec, parse_program
from opto.features.recursive_opt.runmode import _response_usage, make_live_llm

ROOT = Path(__file__).resolve().parent
SPEC = json.loads((ROOT / "phase0_spec.json").read_text())
ENGINEERING_SPEC = json.loads((ROOT / "engineering_smoke_spec.json").read_text())
REQUESTS = [*SPEC["live"]["requests"], ENGINEERING_SPEC["label"]]
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
    directory.mkdir(parents=True, exist_ok=False)
    settings = {
        name: SPEC["live"][name]
        for name in ("temperature", "top_p", "max_tokens", "seed")
    }
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
        "request_timeout_s": SPEC["live"]["request_timeout_s"],
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
                response = client(messages=request["messages"], **settings)
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
    _write(directory / "result.json", result)
    return result


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
    args = parser.parse_args()
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
