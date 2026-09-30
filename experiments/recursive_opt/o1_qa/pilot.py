"""Resumable readiness pilot; this command does not launch comparative optimization."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from opto.features.recursive_opt.runmode import make_live_llm, _response_usage

from . import prepare, task


def failure_evidence(error: Exception) -> dict[str, Any]:
    """Extract only fixed diagnostic labels; never persist provider exception text."""
    statuses: set[int] = set()
    messages = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = getattr(current, "status_code", None)
        if type(status) is int and 400 <= status <= 599:
            statuses.add(status)
        messages.append(str(current))
        current = current.__cause__ or current.__context__
    text = " ".join(messages)
    statuses.update(int(s) for s in re.findall(r'"code"\s*:\s*([45]\d\d)\b', text))
    return {
        "completed_response": False,
        "http_status_codes": sorted(statuses),
        "provider_name": "DeepInfra" if "DeepInfra" in text else None,
        "provider_error_code": (
            "engine_overloaded" if "engine_overloaded" in text else None
        ),
        "limit_source": (
            "upstream_provider_shared_pool"
            if "upstream_provider_shared_pool" in text
            else None
        ),
        "remote_completion": "explicit_rejection" if statuses == {429} else "uncertain",
        "resume_requires_reconciliation": True,
    }


def write_once(path: Path, value: Any) -> None:
    """Persist immutable JSON; a partial file blocks resume rather than replacing evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def request_once(
    folder: Path, request: dict[str, Any], client: Callable[..., Any]
) -> dict[str, Any]:
    """Reuse a completed response; an ambiguous in-flight request requires reconciliation."""
    identity = task.digest(request)
    if (folder / "response.json").exists():
        result = json.loads((folder / "response.json").read_text())
        if result["request_hash"] != identity:
            raise ValueError("completed slot request mismatch")
        return result
    if (folder / "request.json").exists():
        raise RuntimeError(
            "unresolved request: reconcile remote completion before resuming"
        )
    write_once(folder / "request.json", {"request_hash": identity, "request": request})
    started = time.monotonic()
    try:
        response = client(**request)
    except Exception as error:
        write_once(folder / "failure.json", failure_evidence(error))
        raise
    raw = response.model_dump() if hasattr(response, "model_dump") else response
    usage = _response_usage(response)
    raw_usage = raw.get("usage") or {}
    if raw_usage.get("cost") is not None:
        usage["cost_usd"] = raw_usage["cost"]
    result = {
        "request_hash": identity,
        "response": raw,
        "wall_s": time.monotonic() - started,
        "usage": usage,
    }
    write_once(folder / "response.json", result)
    return result


def solve_slot(
    row: dict[str, Any], variant: str, folder: Path, profile: dict[str, Any]
) -> dict[str, Any]:
    """One reader response per slot, including empty/incorrect responses; no repairs."""
    if (folder / "result.json").exists():
        return json.loads((folder / "result.json").read_text())
    artifact = dict(task.INITIAL)
    diagnostic = dict(row)
    if variant == "all_documents":
        artifact["top_k"] = 10
    elif variant == "oracle_documents":
        titles = {title for title, _ in row["supporting_facts"]}
        diagnostic["context"] = [item for item in row["context"] if item[0] in titles]
        artifact["top_k"] = 2
    elif variant not in {"seed", "repeat_seed"}:
        raise ValueError("unknown pilot variant")
    payload = task.public_input(diagnostic)
    retrieval = task.retrieve(
        artifact["ranker_source"],
        payload,
        artifact["top_k"],
        artifact["bridge_expansion"],
    )
    prompt = task.make_prompt(artifact["answer_instruction"], payload, retrieval).data
    events: list[dict[str, Any]] = []

    def record(event: str, kind: str | None) -> None:
        """Persist safe retry lifecycle labels, never exception text or credentials."""
        item = {"event": event, "failure_kind": kind}
        write_once(folder / f"transport_{len(events)}.json", item)
        events.append(item)

    client = make_live_llm(
        "openrouter/" + profile["model"],
        max_retries=profile["transport_max_attempts"],
        base_delay=profile["transport_base_delay_s"],
        request_timeout_s=profile["request_timeout_s"],
        allow_env_overrides=False,
        cache=False,
        empty_response_retries=0,
        retry_event_callback=record,
        budget_resource=None,
    )
    request = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": profile["temperature"],
        "max_tokens": profile["max_tokens"],
        **profile["request_params"],
    }
    evidence = request_once(folder, request, client)
    content = task._response_text(evidence["response"])
    output = task.pack_answer({"content": content}, retrieval).data
    em, f1 = task.answer_metrics(output["answer"], row["answer"])
    selected = {p["title"] for p in retrieval.data["documents"]}
    required = {title for title, _ in row["supporting_facts"]}
    result = {
        "id": row["id"],
        "variant": variant,
        "type": row["type"],
        "answer": output["answer"],
        "answer_EM": em,
        "answer_F1": f1,
        "format_valid": output["format_valid"],
        "support_recall": len(selected & required) / len(required),
        "reader_completed": True,
        "usage": evidence["usage"],
        "wall_s": evidence["wall_s"],
    }
    write_once(folder / "result.json", result)
    return result


def summarize(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    """Require every paired slot; evaluate gates without treating oracle as an eligible policy."""
    ids = manifest["splits"]["pilot_diagnostic"]["ids"]
    variants = manifest["pilot"]["variants"]
    expected = {(identity, variant) for identity in ids for variant in variants}
    by_key = {(r["id"], r["variant"]): r for r in rows}
    if len(rows) != len(expected) or set(by_key) != expected:
        raise ValueError(
            "every pilot identity and slot must be represented exactly once"
        )
    scores = {
        v: statistics.mean(by_key[i, v]["answer_EM"] for i in ids) for v in variants
    }
    solved = sum(
        by_key[i, "seed"]["answer_EM"] == 0
        and by_key[i, "oracle_documents"]["answer_EM"] == 1
        for i in ids
    )
    flips = sum(
        by_key[i, "seed"]["answer_EM"] != by_key[i, "repeat_seed"]["answer_EM"]
        for i in ids
    )
    invalid = statistics.mean(not r["format_valid"] for r in rows)
    gates = manifest["pilot"]["gates"]
    checks = {
        "not_saturated": scores["seed"] <= gates["maximum_seed_EM"],
        "all_documents_not_saturated": scores["all_documents"]
        <= gates["maximum_seed_EM"],
        "reader_can_use_evidence": scores["oracle_documents"]
        >= gates["minimum_oracle_EM"],
        "retrieval_headroom": solved >= gates["minimum_seed_errors_solved_by_oracle"],
        "format_usable": invalid <= gates["maximum_format_invalid_fraction"],
        "repeat_stable": flips <= gates["maximum_seed_repeat_correctness_flips"],
    }
    paid = sum(r["reader_completed"] for r in rows)
    mean_wall = statistics.mean(r["wall_s"] for r in rows)
    return {
        "status": (
            "READY_FOR_FIT_PILOT" if all(checks.values()) else "REVIEW_TASK_OR_READER"
        ),
        "checks": checks,
        "EM": scores,
        "oracle_solves_seed_errors": solved,
        "repeat_correctness_flips": flips,
        "format_invalid_fraction": invalid,
        "completed_reader_slots": paid,
        "optimizer_calls": 0,
        "usage": {
            key: sum(r["usage"].get(key, 0) for r in rows)
            for key in {k for r in rows for k in r["usage"]}
        },
        "mean_reader_wall_s": mean_wall,
        "uncached_confirmation_reader_seconds_at_8_workers": manifest[
            "confirmation_draft"
        ]["call_planning_estimate"]["reader_uncached"]
        * mean_wall
        / 8,
        "timing_caveat": "forecast excludes optimizer critical path, retries and local overhead; not a deadline guarantee",
    }


def worker(args: argparse.Namespace) -> None:
    """Run only pilot slots with eight bounded workers and immutable resume identity."""
    manifest = json.loads(args.manifest.read_text())
    panels = prepare.verify_manifest(manifest, args.dataset)
    run_id = {"manifest_hash": task.digest(manifest), "mode": "READINESS_PILOT"}
    if (args.output / "run.json").exists():
        if json.loads((args.output / "run.json").read_text()) != run_id:
            raise ValueError("run identity mismatch")
    else:
        write_once(args.output / "run.json", run_id)
    from experiments.recursive_opt._shared.optimizer_discovery.phase0 import _load_key

    _load_key()
    jobs = []
    variants = manifest["pilot"]["variants"]
    for index, row in enumerate(panels["pilot_diagnostic"]):
        # Rotation prevents every baseline request preceding every oracle request.
        for variant in variants[index % 4 :] + variants[: index % 4]:
            jobs.append(
                (
                    row,
                    variant,
                    args.output / row["id"] / variant,
                    manifest["llm_profiles"]["reader"],
                )
            )
    workers = manifest["pilot"]["workers"]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, len(jobs), workers):
            # Finish at most this wave after a failure; do not enqueue all 96 slots.
            futures = [
                pool.submit(solve_slot, *job) for job in jobs[offset : offset + workers]
            ]
            results.extend(future.result() for future in futures)
    summary = summarize(results, manifest)
    if not (args.output / "summary.json").exists():
        write_once(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    """Require explicit live launch; supervisor kills the process group at its total deadline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=prepare.MANIFEST)
    parser.add_argument("--dataset", type=Path, default=prepare.DATA_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=prepare.ROOT / "experiments/recursive_opt/_shared/o1_learning/exp20/pilot",
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.live:
        parser.error("no calls made; --live is required after reviewing EXP20.md")
    if args.worker:
        worker(args)
        return
    manifest = json.loads(args.manifest.read_text())
    prepare.verify_manifest(manifest, args.dataset)
    command = [
        sys.executable,
        "-m",
        "experiments.recursive_opt.o1_qa.pilot",
        "--live",
        "--worker",
        "--manifest",
        str(args.manifest),
        "--dataset",
        str(args.dataset),
        "--output",
        str(args.output),
    ]

    def clock() -> float:
        """Include machine suspension in the externally enforced deadline."""
        return time.clock_gettime(time.CLOCK_BOOTTIME)

    deadline = clock() + manifest["pilot"]["deadline_s"]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        while process.poll() is None and clock() < deadline:
            time.sleep(0.5)
        if process.poll() is None:
            raise TimeoutError(
                "pilot deadline: completed evidence retained; unresolved requests require reconciliation"
            )
        if process.returncode:
            raise RuntimeError(
                "pilot incomplete; inspect retained evidence before resuming"
            )
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


if __name__ == "__main__":
    main()
