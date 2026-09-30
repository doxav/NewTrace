"""Prospective objective-instruction and trace-information intervention, EXP-16/F1."""

from __future__ import annotations

import argparse
import contextlib
import gzip
import io
import json
import random
import time
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.evidence import collect_metadata, environment
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import generation as G
from experiments.recursive_opt._shared.optimizer_discovery.investigation16.feedback import rich_feedback as R
from experiments.recursive_opt._shared.optimizer_discovery.phase0 import _load_key
from opto.features.recursive_opt.optimizer_program import _validate_inputs
from opto.features.recursive_opt.runmode import make_live_llm

ROOT = G.ROOT / "feedback_experiment"
TASK_NAMESPACE = "F1-R1"
ANYTIME_OBJECTIVE = R.ANYTIME_OBJECTIVE
CONDITIONS = ("legacy_code", "anytime_code", "anytime_sparse", "anytime_rich")
VALIDITY_CLARIFICATION = (
    "The static protocol rejects these identifiers anywhere, including local "
    "variable names: open, input, eval, exec, compile, getattr, setattr, delattr, "
    "globals, locals, vars, dir, breakpoint, help, exit, quit. Do not use them. "
    "Names starting with double underscores and attribute names starting with "
    "an underscore are also prohibited."
)


def choose_cap(groups: dict[str, dict[str, int]]) -> int:
    """Use only G1's registered truncation/eligibility criterion to choose the cap."""
    if set(groups) != {"8000", "32000"} or any(
        type(group.get(key)) is not int or not 0 <= group[key] <= 12
        for group in groups.values()
        for key in ("length", "eligible")
    ):
        raise ValueError("G1 cap choice requires bounded reliability counts")
    lower, higher = groups["8000"], groups["32000"]
    return (
        32000
        if higher["length"] < lower["length"] or higher["eligible"] > lower["eligible"]
        else 8000
    )


def block_requests(block: dict[str, Any], cap: int) -> list[dict[str, Any]]:
    """Build four factor-separated requests around the same fixed parent."""
    if cap not in (8000, 32000):
        raise ValueError("F1 cap must be chosen from the registered G1 levels")
    parent = block["parent"]
    settings = {
        "temperature": 0.6,
        "top_p": 1.0,
        "max_tokens": cap,
        "extra_body": {"reasoning": {"effort": "low"}},
        "timeout": 300,
        "seed": block["block"],
        "num_retries": 0,
    }
    result = []
    for condition in CONDITIONS:
        invariant = E.INVARIANT + B.SEED_SOURCE + "\n" + VALIDITY_CLARIFICATION
        if condition != "legacy_code":
            invariant += "\nSELECTION OBJECTIVE:\n" + ANYTIME_OBJECTIVE
        current = "Improve the current optimizer.\nCURRENT SOURCE:\n" + parent
        if condition in ("anytime_sparse", "anytime_rich"):
            payload = (
                {"current": block["sparse"]}
                if condition == "anytime_sparse"
                else block["rich"]
            )
            current += "\nTRAINING FEEDBACK:\n" + R.serialize_feedback(
                payload, max_chars=64000
            )
        result.append(
            {
                "slot_id": f'{block["block"]}/A{condition}/slot_00',
                "block": block["block"],
                "condition": condition,
                "model": G.MODEL,
                "messages": [
                    {"role": "user", "content": invariant},
                    {"role": "user", "content": current},
                ],
                "settings": settings,
                "parent_sha256": B.source_hash(parent),
            }
        )
    return result


def evaluate_panel(
    source: str, tasks: list[dict[str, Any]], block: int, *, deployment: bool = False
) -> list[dict[str, Any]]:
    """Reuse the unchanged evaluator with F1-local randomness and common fallback."""
    return [
        B.evaluate(
            source,
            task,
            G.local_seed(TASK_NAMESPACE, block, task),
            deployment=deployment,
        )
        for task in tasks
    ]


def verify_panel(
    rows: list[dict[str, Any]],
    source: str,
    tasks: list[dict[str, Any]],
    block: int,
    *,
    deployment: bool,
) -> None:
    """Check exact source/task/randomness allocations and recompute preserved metrics."""
    if not isinstance(rows, list) or len(rows) != len(tasks):
        raise RuntimeError("F1 panel has missing or extra trajectories")
    for row, task in zip(rows, tasks):
        expected = {
            "source_sha256": B.source_hash(source),
            "task_identity": B.task_identity(task),
            "local_seed": G.local_seed(TASK_NAMESPACE, block, task),
            "budget": 32,
            "stratum": f'{task["family"]}/{task["dimension"]}',
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise RuntimeError("F1 trajectory identity or budget mismatch")
        observations = row.get("observations")
        _validate_inputs(
            observations, [[-5, 5]] * task["dimension"], seed=0, timeout_s=1.0
        )
        count = len(observations)
        if (
            type(row.get("valid")) is not bool
            or type(row.get("candidate_valid")) is not bool
            or type(row.get("fallback_used")) is not bool
            or count > 32
            or row.get("objective_calls") != count
            or row.get("unused_objective_allocation") != 32 - count
            or row["valid"] != (count == 32)
            or (deployment and not row["valid"])
            or (row.get("fallback_used") and not deployment)
            or (row.get("status") == "valid") != row["valid"]
            or row["candidate_valid"]
            != (not row["fallback_used"] if deployment else row["valid"])
        ):
            raise RuntimeError("F1 trajectory validity or accounting mismatch")
        metrics = (
            B.metrics(
                [item["value"] for item in observations], B.normalization(task), 32
            )
            if row["valid"]
            else None
        )
        if row.get("metrics") != metrics:
            raise RuntimeError("F1 trajectory metric integrity failure")


def _g1_result() -> dict[str, Any]:
    """Verify all G1 slots and recompute only the reliability fields choosing the cap."""
    frozen = G.preflight()
    result = E.read(G.ROOT / "generation/results.json")
    rows = result.get("rows", [])
    requests = {request["slot_id"]: request for request in frozen["requests"]}
    if (
        len(rows) != 24
        or len({row["slot_id"] for row in rows}) != 24
        or {row["slot_id"] for row in rows} != set(requests)
    ):
        raise RuntimeError("F1 requires every completed registered G1 slot")
    for row in rows:
        request = requests[row["slot_id"]]
        if (
            row["cap"] != request["settings"]["max_tokens"]
            or row["block"] != request["block"]
            or row["context"] != request["context"]
            or type(row["eligible"]) is not bool
        ):
            raise RuntimeError(
                "G1 summary slot identity differs from its frozen request"
            )
    for cap in (8000, 32000):
        selected = [row for row in rows if row["cap"] == cap]
        expected = {
            "responses": len(selected),
            "length": sum(row["finish_reason"] == "length" for row in selected),
            "eligible": sum(row["eligible"] for row in selected),
        }
        if expected["responses"] != 12 or any(
            result["caps"][str(cap)].get(key) != value
            for key, value in expected.items()
        ):
            raise RuntimeError(
                "G1 reliability aggregate differs from its retained slots"
            )
    return result


def _response(request: dict[str, Any]) -> dict[str, Any]:
    """Verify exact request, completion metadata and extracted-source provenance."""
    directory = ROOT / "raw" / request["slot_id"]
    if (
        not E.exists(directory / "request.json")
        or E.read(directory / "request.json") != request
    ):
        raise RuntimeError("F1 persisted request differs from the freeze")
    response = E.read(directory / "response.json")
    if (
        response.get("completed") is not True
        or response.get("model") != G.MODEL
        or type(response.get("completed_ns")) is not int
        or response["completed_ns"] <= 0
        or not isinstance(response.get("id"), str)
        or not response["id"]
    ):
        raise RuntimeError("F1 response completion or model integrity failure")
    try:
        source, status = G.parse_program(response.get("content")), "parsed"
    except ValueError:
        source, status = "", "unparsable"
    if (
        response.get("source") != source
        or response.get("source_sha256") != B.source_hash(source)
        or response.get("parse_status") != status
        or response.get("source_status") != B.source_status(source)
    ):
        raise RuntimeError("F1 response source integrity failure")
    return response


def _completed(
    frozen: dict[str, Any], *, require_all: bool
) -> dict[str, dict[str, Any]]:
    """Audit completed slots and prohibit regeneration after the validation barrier."""
    if require_all and not all(
        E.exists(ROOT / "raw" / request["slot_id"] / "response.json")
        for request in frozen["requests"]
    ):
        raise RuntimeError("validation requires every response to be frozen")
    responses = {
        request["slot_id"]: _response(request)
        for request in frozen["requests"]
        if E.exists(ROOT / "raw" / request["slot_id"] / "response.json")
    }
    if len({value["id"] for value in responses.values()}) != len(responses):
        raise RuntimeError("F1 response IDs are not unique")
    barrier = ROOT / "validation_opened.json"
    if E.exists(barrier):
        record = E.read(barrier)
        if (
            len(responses) != len(frozen["requests"])
            or record.get("sources")
            != {slot: response["source_sha256"] for slot, response in responses.items()}
            or record.get("responses")
            != {slot: B.digest(response) for slot, response in responses.items()}
            or any(
                response["completed_ns"] > record["time_ns"]
                for response in responses.values()
            )
        ):
            raise RuntimeError("F1 validation barrier provenance or chronology differs")
    return responses


def prepare() -> dict[str, Any]:
    """Freeze every request before generation and before validation evaluation."""
    if E.exists(ROOT / "freeze.json"):
        frozen = preflight(allow_unsealed_preparation=True)
        I.persist(ROOT / "freeze_sha256.json", {"sha256": B.digest(frozen)})
        return frozen
    generation = _g1_result()
    cap = choose_cap(generation["caps"])
    tasks = {
        split: G.fresh_tasks(TASK_NAMESPACE, split, 1)
        for split in ("train", "validation")
    }
    old = {
        B.task_identity(task)
        for phase in ("pilot", "confirmation")
        for split in ("train", "validation", "holdout")
        for task in B.make_tasks(phase, split)
    }
    new = [B.task_identity(t) for values in tasks.values() for t in values]
    if len(set(new)) != 12 or set(new).intersection(old):
        raise RuntimeError("F1 split identity overlaps earlier evidence")
    representative = gzip.decompress(
        (B.ROOT / "exp15/selected/A2_seed_41.py.gz").read_bytes()
    ).decode()
    if (
        B.source_hash(representative)
        != "1684f91acdc36c0ca6aac70afeb9cc2c4eed7ab847926d5880590e059266abb7"
    ):
        raise RuntimeError("fixed representative source differs")
    blocks, schedule, context_hashes = [], [], {}
    rng = random.Random(163000)
    for index, seed in enumerate(range(16301, 16307)):
        parent = B.SEED_SOURCE if index % 2 == 0 else representative
        context_path = ROOT / f"contexts/{seed}.json"
        evaluations = (
            E.read(context_path)
            if E.exists(context_path)
            else evaluate_panel(parent, tasks["train"], seed)
        )
        verify_panel(evaluations, parent, tasks["train"], seed, deployment=False)
        if not all(row["valid"] for row in evaluations):
            I.persist(ROOT / f"preparation_failure_{seed}.json", evaluations)
            raise RuntimeError("fixed parent failed prospective context evaluation")
        I.persist(context_path, evaluations)
        context_hashes[str(seed)] = B.digest(evaluations)
        block = {
            "block": seed,
            "parent": parent,
            "parent_kind": "seed" if index % 2 == 0 else "representative",
            "sparse": G.sparse_feedback(evaluations),
            "rich": R.build_feedback(
                parent,
                evaluations,
                [[[-5, 5]] * t["dimension"] for t in tasks["train"]],
            ),
        }
        requests = block_requests(block, cap)
        rng.shuffle(requests)
        schedule.extend(requests)
        blocks.append({k: block[k] for k in ("block", "parent", "parent_kind")})
    files = [
        Path(__file__),
        Path(G.__file__),
        Path(R.__file__),
        Path(B.__file__),
        Path(E.__file__),
        Path(I.__file__),
        B.ROOT / "exp15_manifest.json",
        G.ROOT / "feedback/PROTOCOL_F1.md",
        Path("opto/features/recursive_opt/optimizer_program.py"),
    ]
    frozen = {
        "experiment": "EXP-16",
        "stage": "F1",
        "task_namespace": TASK_NAMESPACE,
        "status": "FROZEN_DIAGNOSTIC",
        "created_ns": time.time_ns(),
        "environment": environment(),
        "files": {
            str(p.relative_to(Path.cwd()) if p.is_absolute() else p): B.source_hash(
                p.read_text()
            )
            for p in files
        },
        "cap": cap,
        "g1_results_sha256": B.digest(generation),
        "tasks": tasks,
        "contexts": context_hashes,
        "blocks": blocks,
        "requests": schedule,
        "budget": 32,
        "bootstrap": B.MANIFEST["bootstrap"],
    }
    I.persist(ROOT / "freeze.json", frozen)
    I.persist(ROOT / "freeze_sha256.json", {"sha256": B.digest(frozen)})
    return frozen


def preflight(*, allow_unsealed_preparation: bool = False) -> dict[str, Any]:
    """Verify scientific code, protocol and environment before execution or resume."""
    frozen = E.read(ROOT / "freeze.json")
    seal = ROOT / "freeze_sha256.json"
    if not E.exists(seal):
        raw = ROOT / "raw"
        if (
            not allow_unsealed_preparation
            or any(raw.rglob("started_*.json"))
            or any(raw.rglob("response.json*"))
            or E.exists(ROOT / "validation_opened.json")
        ):
            raise RuntimeError("F1 freeze is unsealed after preparation or execution")
    elif E.read(seal) != {"sha256": B.digest(frozen)}:
        raise RuntimeError("F1 freeze manifest checksum mismatch")
    for path, expected in frozen["files"].items():
        if B.source_hash(Path(path).read_text()) != expected:
            raise RuntimeError(f"F1 freeze mismatch: {path}")
    if frozen["environment"] != environment():
        raise RuntimeError("F1 environment mismatch")
    if any(
        frozen.get(key) != value
        for key, value in {
            "experiment": "EXP-16",
            "stage": "F1",
            "status": "FROZEN_DIAGNOSTIC",
        }.items()
    ):
        raise RuntimeError("F1 frozen experiment identity differs")
    generation = _g1_result()
    if B.digest(generation) != frozen["g1_results_sha256"] or frozen[
        "cap"
    ] != choose_cap(generation["caps"]):
        raise RuntimeError("F1 G1 provenance or cap decision differs")
    expected_tasks = {
        split: G.fresh_tasks(TASK_NAMESPACE, split, 1)
        for split in ("train", "validation")
    }
    if (
        frozen["tasks"] != expected_tasks
        or frozen.get("task_namespace") != TASK_NAMESPACE
        or frozen["budget"] != 32
        or frozen["bootstrap"] != B.MANIFEST["bootstrap"]
    ):
        raise RuntimeError("F1 frozen task, budget or analysis configuration differs")
    if [block["block"] for block in frozen["blocks"]] != list(range(16301, 16307)):
        raise RuntimeError("F1 requires six fixed parent blocks")
    schedule = []
    rng = random.Random(163000)
    for index, block in enumerate(frozen["blocks"]):
        expected_hash = (
            B.source_hash(B.SEED_SOURCE)
            if index % 2 == 0
            else "1684f91acdc36c0ca6aac70afeb9cc2c4eed7ab847926d5880590e059266abb7"
        )
        if B.source_hash(block["parent"]) != expected_hash or block["parent_kind"] != (
            "seed" if index % 2 == 0 else "representative"
        ):
            raise RuntimeError("F1 fixed parent source integrity failure")
        rows = E.read(ROOT / f'contexts/{block["block"]}.json')
        if B.digest(rows) != frozen["contexts"][str(block["block"])]:
            raise RuntimeError("F1 frozen context integrity failure")
        requests = block_requests(
            {
                **block,
                "sparse": G.sparse_feedback(rows),
                "rich": R.build_feedback(
                    block["parent"],
                    rows,
                    [[[-5, 5]] * task["dimension"] for task in expected_tasks["train"]],
                ),
            },
            frozen["cap"],
        )
        rng.shuffle(requests)
        schedule.extend(requests)
    if (
        frozen["requests"] != schedule
        or len({request["slot_id"] for request in schedule}) != 24
    ):
        raise RuntimeError("F1 request schedule differs from the registered factors")
    return frozen


def validate() -> None:
    """Open the validation panel only after every frozen response is preserved."""
    frozen = preflight()
    responses = _completed(frozen, require_all=True)
    for request in frozen["requests"]:
        target = ROOT / "raw" / request["slot_id"] / "training.json"
        if not E.exists(target):
            raise RuntimeError("F1 validation requires every training panel")
        verify_panel(
            E.read(target),
            responses[request["slot_id"]]["source"],
            frozen["tasks"]["train"],
            request["block"],
            deployment=False,
        )
    barrier = ROOT / "validation_opened.json"
    if not E.exists(barrier):
        opened = time.time_ns()
        if any(response["completed_ns"] > opened for response in responses.values()):
            raise RuntimeError("F1 response completion chronology is in the future")
        I.persist(
            barrier,
            {
                "time_ns": opened,
                "sources": {
                    slot: response["source_sha256"]
                    for slot, response in responses.items()
                },
                "responses": {
                    slot: B.digest(response) for slot, response in responses.items()
                },
            },
        )
    for request in frozen["requests"]:
        directory = ROOT / "raw" / request["slot_id"]
        target = directory / "validation.json"
        if not E.exists(target):
            result = E.read(directory / "response.json")
            rows = evaluate_panel(
                result["source"],
                frozen["tasks"]["validation"],
                request["block"],
                deployment=True,
            )
            I.persist(target, rows)
        verify_panel(
            E.read(target),
            responses[request["slot_id"]]["source"],
            frozen["tasks"]["validation"],
            request["block"],
            deployment=True,
        )
    for block in frozen["blocks"]:
        for name, source in (("parent", block["parent"]), ("seed", B.SEED_SOURCE)):
            target = ROOT / f'controls/{block["block"]}_{name}.json'
            if not E.exists(target):
                rows = evaluate_panel(
                    source,
                    frozen["tasks"]["validation"],
                    block["block"],
                    deployment=True,
                )
                if any(row["fallback_used"] or not row["valid"] for row in rows):
                    I.persist(target, rows)
                    raise RuntimeError("trusted fixed F1 control failed validation")
                I.persist(target, rows)
            verify_panel(
                E.read(target),
                source,
                frozen["tasks"]["validation"],
                block["block"],
                deployment=True,
            )
    paths = [
        ROOT / "raw" / request["slot_id"] / name
        for request in frozen["requests"]
        for name in ("training.json", "validation.json")
    ]
    paths.extend(
        ROOT / f'controls/{block["block"]}_{name}.json'
        for block in frozen["blocks"]
        for name in ("parent", "seed")
    )
    I.persist(
        ROOT / "evaluations_frozen.json",
        {str(path.relative_to(ROOT)): B.digest(E.read(path)) for path in paths},
    )


def summarize() -> dict[str, Any]:
    """Report all responses and paired intervention deltas without dropping failures."""
    frozen = preflight()
    if not E.exists(ROOT / "validation_opened.json") or not E.exists(
        ROOT / "evaluations_frozen.json"
    ):
        raise RuntimeError("F1 analysis requires completed, frozen validation")
    _completed(frozen, require_all=True)
    for path, digest in E.read(ROOT / "evaluations_frozen.json").items():
        if B.digest(E.read(ROOT / path)) != digest:
            raise RuntimeError("F1 frozen evaluation integrity failure")
    rows = []
    for request in frozen["requests"]:
        directory = ROOT / "raw" / request["slot_id"]
        response = E.read(directory / "response.json")
        train = E.read(directory / "training.json")
        validation = E.read(directory / "validation.json")
        verify_panel(
            train,
            response["source"],
            frozen["tasks"]["train"],
            request["block"],
            deployment=False,
        )
        verify_panel(
            validation,
            response["source"],
            frozen["tasks"]["validation"],
            request["block"],
            deployment=True,
        )
        rows.append(
            {
                "block": request["block"],
                "condition": request["condition"],
                "parent_kind": next(
                    block["parent_kind"]
                    for block in frozen["blocks"]
                    if block["block"] == request["block"]
                ),
                "source_sha256": response["source_sha256"],
                "source_status": response["source_status"],
                "train_valid": all(row["valid"] for row in train),
                "candidate_valid_trajectories": sum(
                    row["candidate_valid"] for row in validation
                ),
                "fallback_trajectories": sum(
                    row["fallback_used"] for row in validation
                ),
                "validation_auc": B.aggregate(validation, "auc"),
                "final_regret": B.aggregate(validation, "final_regret"),
                "usage": response["usage"],
            }
        )
    contrasts = {}
    for treated, control in (
        ("anytime_code", "legacy_code"),
        ("anytime_sparse", "anytime_code"),
        ("anytime_rich", "anytime_sparse"),
        ("anytime_rich", "anytime_code"),
    ):
        deltas = []
        for block in frozen["blocks"]:
            scores = {
                row["condition"]: row["validation_auc"]
                for row in rows
                if row["block"] == block["block"]
            }
            deltas.append(scores[treated] - scores[control])
        contrasts[f"{treated}-{control}"] = {
            **E.paired(deltas),
            "replication_unit": "fixed_parent_generation_block",
            "interpretation_scope": "exploratory n=6; multiple contrasts",
        }
    result = {
        "experiment": "EXP-16",
        "stage": "F1",
        "rows": rows,
        "contrasts": contrasts,
    }
    I.persist(ROOT / "results.json", result)
    print(json.dumps(contrasts, indent=2), flush=True)
    return result


def run() -> None:
    """Generate all fixed-parent proposals, then evaluate separate validation tasks."""
    frozen = preflight()
    after_validation = E.exists(ROOT / "validation_opened.json")
    _completed(frozen, require_all=after_validation)
    if after_validation:
        validate()
        summarize()
        return
    _load_key()
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        client = make_live_llm(
            "openrouter/" + G.MODEL,
            cache=False,
            max_retries=1,
            request_timeout_s=300,
            allow_env_overrides=False,
            empty_response_retries=0,
            budget_resource=None,
        )
    for request in frozen["requests"]:
        directory = ROOT / "raw" / request["slot_id"]
        response = I.complete_slot(directory, request, client)
        response = _response(request)
        target = directory / "training.json"
        if not E.exists(target):
            I.persist(
                target,
                evaluate_panel(
                    response["source"], frozen["tasks"]["train"], request["block"]
                ),
            )
        verify_panel(
            E.read(target),
            response["source"],
            frozen["tasks"]["train"],
            request["block"],
            deployment=False,
        )
        print(
            json.dumps(
                {
                    "stage": "F1",
                    "slot": request["slot_id"],
                    "status": response["source_status"],
                    "usage": response["usage"],
                }
            ),
            flush=True,
        )
    collect_metadata(ROOT / "raw")
    validate()
    summarize()


def main() -> None:
    """Prepare or execute the prospective fixed-parent information ablation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    args = parser.parse_args()
    if args.command == "prepare":
        frozen = prepare()
        print(
            json.dumps(
                {
                    "stage": "F1",
                    "cap": frozen["cap"],
                    "requests": len(frozen["requests"]),
                }
            )
        )
    elif args.command == "run":
        run()
    else:
        summarize()


if __name__ == "__main__":
    main()
