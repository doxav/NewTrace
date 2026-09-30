"""Read-only G1 generation-mechanism analysis; frozen experiment code is untouched."""

from __future__ import annotations

import argparse
import ast
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.evidence import environment
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import generation as G

FORBIDDEN = frozenset(
    {
        "open",
        "input",
        "eval",
        "exec",
        "compile",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "breakpoint",
        "help",
        "exit",
        "quit",
    }
)


def source_diagnostics(source: str) -> dict[str, Any]:
    """Inspect syntax and declared source-screen mechanisms without executing code."""
    result: dict[str, Any] = {
        "source_bytes": len(source.encode()),
        "source_status": B.source_status(source),
    }
    if not source:
        return result
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError) as error:
        result["parse_error"] = {
            "type": type(error).__name__,
            "line": getattr(error, "lineno", None),
            "offset": getattr(error, "offset", None),
        }
        return result
    nodes = list(ast.walk(tree))
    forbidden_names = sorted(
        {
            node.id
            for node in nodes
            if isinstance(node, ast.Name) and node.id in FORBIDDEN
        }
    )
    forbidden_calls = sorted(
        {
            node.func.id
            for node in nodes
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FORBIDDEN
        }
    )
    private_names = sorted(
        {
            node.id
            for node in nodes
            if isinstance(node, ast.Name) and node.id.startswith("__")
        }
    )
    private_attributes = sorted(
        {
            node.attr
            for node in nodes
            if isinstance(node, ast.Attribute) and node.attr.startswith("_")
        }
    )
    imports = [
        alias.name
        for node in nodes
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    imports.extend(
        node.module or "" for node in nodes if isinstance(node, ast.ImportFrom)
    )
    disallowed = sorted(
        {
            name
            for name in imports
            if name.split(".")[0] not in B.MANIFEST["allowed_imports"]
        }
    )
    restricted_aliases = sorted(
        {
            node.name
            for node in nodes
            if isinstance(node, ast.alias)
            and (
                node.name.startswith("_")
                or node.name
                in {"os", "sys", "builtins", "subprocess", "socket", "pathlib"}
            )
        }
    )
    relative_imports = sum(
        isinstance(node, ast.ImportFrom) and node.level > 0 for node in nodes
    )
    lexical_only = bool(forbidden_names) and not (
        forbidden_calls
        or private_names
        or private_attributes
        or disallowed
        or restricted_aliases
        or relative_imports
    )
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "propose"
    ]
    signature = None
    if definitions:
        arguments = definitions[-1].args
        signature = {
            "positional_only": [arg.arg for arg in arguments.posonlyargs],
            "positional": [arg.arg for arg in arguments.args],
            "keyword_only": [arg.arg for arg in arguments.kwonlyargs],
            "defaults": len(arguments.defaults),
            "vararg": arguments.vararg.arg if arguments.vararg else None,
            "kwarg": arguments.kwarg.arg if arguments.kwarg else None,
        }
    result.update(
        {
            "ast_nodes": len(nodes),
            "forbidden_names": forbidden_names,
            "forbidden_name_occurrences": [
                {
                    "name": node.id,
                    "line": node.lineno,
                    "context": type(node.ctx).__name__,
                }
                for node in nodes
                if isinstance(node, ast.Name) and node.id in FORBIDDEN
            ],
            "forbidden_calls": forbidden_calls,
            "private_names": private_names,
            "private_attributes": private_attributes,
            "disallowed_imports": disallowed,
            "restricted_aliases": restricted_aliases,
            "relative_imports": relative_imports,
            "protocol_class": (
                "reserved_identifier_without_forbidden_call"
                if lexical_only
                else "other_or_no_source_screen_violation"
            ),
            "propose_signature": signature,
        }
    )
    return result


def usage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Retain missing usage explicitly; reasoning is a subset of completion tokens."""
    result = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",
        "cost_usd",
    ):
        values = [
            row["usage"][key] for row in rows if row["usage"].get(key) is not None
        ]
        result[key] = {
            "reported_sum": sum(values) if values else None,
            "reported_responses": len(values),
            "missing_responses": len(rows) - len(values),
        }
    return result


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe generation reliability and resources without conditional-score selection."""
    eligible = sum(row["eligible"] for row in rows)
    return {
        "responses": len(rows),
        "eligible": eligible,
        "eligible_fraction": eligible / len(rows),
        "source_valid": sum(row["source_status"] == "valid" for row in rows),
        "length": sum(row["finish_reason"] == "length" for row in rows),
        "source_statuses": dict(Counter(row["source_status"] for row in rows)),
        "execution_failure_candidates": sum(
            row["source_status"] == "valid" and not row["eligible"] for row in rows
        ),
        "wall_s_mean": statistics.mean(row["wall_s"] for row in rows),
        "wall_s_median": statistics.median(row["wall_s"] for row in rows),
        "wall_s_max": max(row["wall_s"] for row in rows),
        "transport_attempts": sum(row["transport_attempts"] for row in rows),
        "transport_failures": sum(row["transport_failures"] for row in rows),
        "attempt_measured_s": sum(row["attempt_measured_s"] for row in rows),
        "slot_elapsed_s": sum(row["slot_elapsed_s"] for row in rows),
        "objective_allocations": sum(row["objective_allocations"] for row in rows),
        "objective_calls": sum(row["objective_calls"] for row in rows),
        "unused_objective_allocations": sum(
            row["unused_objective_allocations"] for row in rows
        ),
        "subprocess_executions": sum(row["subprocess_executions"] for row in rows),
        "usage": usage_summary(rows),
        "providers": dict(
            Counter(
                (
                    row["receipt"].get("provider_name", "unreported")
                    if row["receipt"]
                    else "missing_receipt"
                )
                for row in rows
            )
        ),
    }


def paired_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Count all paired binary outcomes, without imputing scores for failed programs."""
    deltas = [pair["eligibility_delta_32000_minus_8000"] for pair in pairs]
    return {
        "pairs": len(pairs),
        "eligibility_deltas": deltas,
        "eligibility_rate_difference": statistics.mean(deltas),
        "eligibility_gain_pairs": sum(value > 0 for value in deltas),
        "eligibility_loss_pairs": sum(value < 0 for value in deltas),
        "both_eligible_pairs": sum(pair["both_eligible"] for pair in pairs),
        "neither_eligible_pairs": sum(pair["neither_eligible"] for pair in pairs),
        "length_rate_difference": statistics.mean(
            pair["length_delta_32000_minus_8000"] for pair in pairs
        ),
        "source_validity_rate_difference": statistics.mean(
            pair["source_validity_delta_32000_minus_8000"] for pair in pairs
        ),
    }


def analyze(root: Path) -> dict[str, Any]:
    """Verify and analyze every frozen G1 slot only after all 24 evaluations complete."""
    frozen = E.read(root / "freeze.json")
    requests = frozen["requests"]
    expected_slots = {
        f"{block}/A{context}_{cap}/slot_00"
        for block in range(16001, 16007)
        for context in ("I", "L")
        for cap in (8000, 32000)
    }
    if (
        len(requests) != 24
        or {request["slot_id"] for request in requests} != expected_slots
    ):
        raise RuntimeError("G1 frozen schedule does not contain all 24 unique slots")
    if not all(
        E.exists(root / "raw" / request["slot_id"] / name)
        for request in requests
        for name in ("request.json", "response.json", "evaluation.json")
    ):
        raise RuntimeError(
            "G1 analysis requires all 24 completed responses and evaluations"
        )
    for path, expected in frozen["files"].items():
        if B.source_hash(Path(path).read_text()) != expected:
            raise RuntimeError("G1 frozen source/protocol changed")
    if frozen["environment"] != environment():
        raise RuntimeError("G1 environment differs from its freeze")
    hashes = {str(root / "freeze.json"): B.digest(frozen)}

    def read(path: Path) -> Any:
        """Record canonical content identity whenever an evidence object is read."""
        value = E.read(path)
        hashes[str(path)] = B.digest(value)
        return value

    rows = []
    for position, request in enumerate(requests):
        directory = root / "raw" / request["slot_id"]
        actual = read(directory / "request.json")
        response = read(directory / "response.json")
        evaluations = read(directory / "evaluation.json")
        if actual != request or request["model"] != G.MODEL:
            raise RuntimeError("G1 request differs from the exact frozen request")
        if (
            response.get("completed") is not True
            or response.get("model") != G.MODEL
            or not response.get("id")
        ):
            raise RuntimeError("G1 response completion/model identity differs")
        source = response["source"]
        try:
            extracted, parse_status = G.parse_program(response["content"]), "parsed"
        except ValueError:
            extracted, parse_status = "", "unparsable"
        if (
            B.source_hash(source) != response["source_sha256"]
            or source != extracted
            or response["parse_status"] != parse_status
            or response["source_status"] != B.source_status(source)
        ):
            raise RuntimeError(
                "G1 source hash/extraction mismatch; raw evidence must not be replaced"
            )
        if len(evaluations) != 6 or len(frozen["tasks"]) != 6:
            raise RuntimeError("G1 evaluation panel does not cover all six tasks")
        for evaluation, task in zip(evaluations, frozen["tasks"]):
            expected = {
                "source_sha256": B.source_hash(source),
                "task_identity": B.task_identity(task),
                "budget": 32,
                "local_seed": G.local_seed("G1", request["block"], task),
                "stratum": f'{task["family"]}/{task["dimension"]}',
            }
            if any(evaluation.get(key) != value for key, value in expected.items()):
                raise RuntimeError("G1 evaluation source/task/seed/budget mismatch")
            count = len(evaluation["observations"])
            if (
                evaluation["objective_calls"] != count
                or evaluation["unused_objective_allocation"] != 32 - count
                or evaluation["valid"] != (count == 32)
            ):
                raise RuntimeError("G1 objective-allocation accounting differs")
            expected_metrics = (
                B.metrics(
                    [
                        observation["value"]
                        for observation in evaluation["observations"]
                    ],
                    B.normalization(task),
                    32,
                )
                if evaluation["valid"]
                else None
            )
            if evaluation["metrics"] != expected_metrics:
                raise RuntimeError("G1 metrics differ from preserved observations")
        attempt_paths = sorted(
            directory.glob("attempt_*.json"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
        attempts = [read(path) for path in attempt_paths]
        started = [
            read(directory / path.name.replace("attempt_", "started_"))
            for path in attempt_paths
        ]
        if (
            not 1 <= len(attempts) <= 4
            or len(attempts) != response["attempt"]
            or attempts[-1]["status"] != "completed"
            or attempts[-1]["id"] != response["id"]
            or any(
                attempt["status"] != "transport_failure" for attempt in attempts[:-1]
            )
        ):
            raise RuntimeError(
                "G1 transport attempt allocation differs from its frozen policy"
            )
        receipt_path = directory / "provider_generation.json"
        receipt = read(receipt_path) if E.exists(receipt_path) else None
        if receipt is not None and receipt.get("id") != response["id"]:
            raise RuntimeError("G1 provider receipt identity mismatch")
        eligible = all(evaluation["valid"] for evaluation in evaluations)
        rows.append(
            {
                "slot_id": request["slot_id"],
                "position": position,
                "block": request["block"],
                "context": request["context"],
                "cap": request["settings"]["max_tokens"],
                "id": response["id"],
                "model": response["model"],
                "source_sha256": response["source_sha256"],
                "source_status": response["source_status"],
                "finish_reason": response["finish_reason"],
                "eligible": eligible,
                "trajectory_statuses": dict(
                    Counter(evaluation["status"] for evaluation in evaluations)
                ),
                "source_diagnostics": source_diagnostics(source),
                "auc": B.aggregate(evaluations, "auc") if eligible else None,
                "usage": response["usage"],
                "wall_s": response["wall_s"],
                "transport_attempts": len(attempts),
                "transport_failures": len(attempts) - 1,
                "unresolved_remote_completion_attempts": sum(
                    attempt.get(
                        "possible_remote_completion_or_duplicate_billing", False
                    )
                    for attempt in attempts
                ),
                "transport_failure_types": [
                    attempt.get("error_type", "unreported")
                    for attempt in attempts
                    if attempt["status"] == "transport_failure"
                ],
                "attempt_measured_s": sum(attempt["wall_s"] for attempt in attempts),
                "slot_elapsed_s": (response["completed_ns"] - started[0]["time_ns"])
                / 1e9,
                "objective_allocations": 192,
                "objective_calls": sum(
                    evaluation["objective_calls"] for evaluation in evaluations
                ),
                "unused_objective_allocations": sum(
                    evaluation["unused_objective_allocation"]
                    for evaluation in evaluations
                ),
                "subprocess_executions": sum(
                    evaluation["subprocess_executions"] for evaluation in evaluations
                ),
                "execution_s": sum(
                    evaluation["execution_s"] for evaluation in evaluations
                ),
                "receipt": receipt,
            }
        )
    if len({row["id"] for row in rows}) != 24:
        raise RuntimeError("G1 response IDs are not unique")
    pairs = []
    for block in range(16001, 16007):
        for context in ("I", "L"):
            values = {
                row["cap"]: row
                for row in rows
                if row["block"] == block and row["context"] == context
            }
            lower, higher = values[8000], values[32000]
            left_request, right_request = (
                requests[lower["position"]],
                requests[higher["position"]],
            )
            if left_request["messages"] != right_request["messages"] or {
                key: value
                for key, value in left_request["settings"].items()
                if key != "max_tokens"
            } != {
                key: value
                for key, value in right_request["settings"].items()
                if key != "max_tokens"
            }:
                raise RuntimeError("G1 paired request changes factors besides the cap")
            pairs.append(
                {
                    "block": block,
                    "context": context,
                    "first_cap": (
                        8000 if lower["position"] < higher["position"] else 32000
                    ),
                    "lower_slot": lower["slot_id"],
                    "higher_slot": higher["slot_id"],
                    "eligibility_delta_32000_minus_8000": int(higher["eligible"])
                    - int(lower["eligible"]),
                    "source_validity_delta_32000_minus_8000": int(
                        higher["source_status"] == "valid"
                    )
                    - int(lower["source_status"] == "valid"),
                    "length_delta_32000_minus_8000": int(
                        higher["finish_reason"] == "length"
                    )
                    - int(lower["finish_reason"] == "length"),
                    "both_eligible": lower["eligible"] and higher["eligible"],
                    "neither_eligible": not lower["eligible"]
                    and not higher["eligible"],
                    "auc_delta_if_both_eligible": (
                        higher["auc"] - lower["auc"]
                        if lower["eligible"] and higher["eligible"]
                        else None
                    ),
                }
            )
    caps = {
        str(cap): group_summary([row for row in rows if row["cap"] == cap])
        for cap in (8000, 32000)
    }
    conditioned = [
        pair["auc_delta_if_both_eligible"] for pair in pairs if pair["both_eligible"]
    ]
    receipts = [row["receipt"] for row in rows if row["receipt"] is not None]
    costs = [
        receipt["total_cost"]
        for receipt in receipts
        if receipt.get("total_cost") is not None
    ]
    return {
        "experiment": "EXP-16",
        "stage": "G1",
        "classification": "EXPLORATORY generation-reliability intervention; not optimizer superiority",
        "rows": rows,
        "pairs": pairs,
        "caps": caps,
        "contexts": {
            context: {
                str(cap): group_summary(
                    [
                        row
                        for row in rows
                        if row["cap"] == cap and row["context"] == context
                    ]
                )
                for cap in (8000, 32000)
            }
            for context in ("I", "L")
        },
        "paired": {
            "pooled": paired_summary(pairs),
            **{
                context: paired_summary(
                    [pair for pair in pairs if pair["context"] == context]
                )
                for context in ("I", "L")
            },
        },
        "order": {
            context: {
                f"{cap}_first": sum(
                    pair["context"] == context and pair["first_cap"] == cap
                    for pair in pairs
                )
                for cap in (8000, 32000)
            }
            for context in ("I", "L")
        },
        "usage_total": usage_summary(rows),
        "receipt_coverage": {
            "present": len(receipts),
            "missing": 24 - len(receipts),
            "reported_cost_sum": sum(costs) if costs else None,
            "cost_receipts": len(costs),
            "model_identifiers": sorted(
                {receipt["model"] for receipt in receipts if receipt.get("model")}
            ),
        },
        "conditional_performance": {
            "both_eligible_pairs": len(conditioned),
            "mean_auc_delta_if_both_eligible": (
                statistics.mean(conditioned) if conditioned else None
            ),
            "warning": "Selected on successful generation under both caps; not an overall policy or task-performance contrast",
        },
        "recommended_cap_by_frozen_reliability_rule": (
            32000
            if caps["32000"]["length"] < caps["8000"]["length"]
            or caps["32000"]["eligible"] > caps["8000"]["eligible"]
            else 8000
        ),
        "limitations": [
            "Only six request-seed blocks and two fixed contexts",
            "Cap order is confounded within each context: I has 8000 first, L has 32000 first",
            "Routing and generation are stochastic; matching request seeds do not reproduce an identical response",
            "Reasoning tokens are included in completion tokens, not additive",
            "User-reported machine suspension contaminates timing; recorded durations must not be interpreted as provider endpoint latency",
            "A timed-out attempt may have completed remotely; the unique completed response count does not establish absence of duplicate billing",
            "No confidence bound establishes near-perfect generation reliability",
            "Validity improvements do not establish usefulness of recursive feedback or task-performance superiority",
        ],
        "input_canonical_json_hashes": hashes,
    }


def main() -> None:
    """Write a final complete-run audit after G1's own recorder and summary finish."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=G.ROOT / "generation")
    arguments = parser.parse_args()
    if not E.exists(arguments.root / "results.json"):
        raise RuntimeError("G1 recorder and summary must finish before final analysis")
    result = analyze(arguments.root)
    I.persist(arguments.root / "analysis_results.json", result)
    print(
        {
            "responses": len(result["rows"]),
            "pairs": len(result["pairs"]),
            "recommended_cap": result["recommended_cap_by_frozen_reliability_rule"],
        }
    )


if __name__ == "__main__":
    main()
