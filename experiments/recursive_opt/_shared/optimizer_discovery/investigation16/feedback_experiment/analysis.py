"""Read-only final F1 audit; refuses efficacy analysis before all registered evidence exists."""

from __future__ import annotations

import importlib.util
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import feedback_experiment as F
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import generation as G

CONTRASTS = (
    ("anytime_code", "legacy_code"),
    ("anytime_sparse", "anytime_code"),
    ("anytime_rich", "anytime_sparse"),
    ("anytime_rich", "anytime_code"),
)


def source_diagnostics(source: str) -> dict[str, Any]:
    """Reuse G1's AST-only mechanism inspection without executing candidate code."""
    spec = importlib.util.spec_from_file_location(
        "g1_ast_helper", G.ROOT / "generation/analysis.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("G1 AST helper unavailable")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper.source_diagnostics(source)


def verify_attempts(row: dict[str, Any], *, deployment: bool) -> dict[str, int]:
    """Verify the objective cursor, permanent fallback and deterministic replay accounting."""
    cursor = executions = failures = 0
    fallback = False
    attempts = row["proposal_attempts"]
    for index, attempt in enumerate(attempts):
        if attempt["evaluation_index"] != cursor or cursor >= row["budget"]:
            raise RuntimeError("F1 proposal cursor or budget accounting differs")
        if attempt["policy"] != ("fallback" if fallback else "candidate"):
            raise RuntimeError("F1 fallback policy switched back or skipped failure")
        if "stdout" in attempt or "stderr" in attempt:
            if any(
                not isinstance(attempt.get(name), str) or len(attempt[name]) > 8192
                for name in ("stdout", "stderr")
            ):
                raise RuntimeError(
                    "F1 captured candidate output exceeds its declared boundary"
                )
            executions += 2 if attempt["status"] in {"valid", "nondeterministic"} else 1
        elif attempt["status"] == "valid":
            raise RuntimeError("F1 successful proposal lacks subprocess evidence")
        if attempt["status"] == "valid":
            cursor += 1
        else:
            failures += 1
            if (
                fallback
                or failures > 1
                or (not deployment and index != len(attempts) - 1)
            ):
                raise RuntimeError(
                    "F1 trusted fallback failed or invalid training continued"
                )
            fallback = deployment
    if (
        not attempts
        or cursor != row["objective_calls"]
        or executions != row["subprocess_executions"]
        or row["fallback_used"] != fallback
        or row["candidate_valid"] != (failures == 0)
    ):
        raise RuntimeError("F1 proposal/fallback/subprocess accounting differs")
    return {
        "attempts": len(attempts),
        "failures": failures,
        "objective_calls": cursor,
        "subprocess_executions": executions,
    }


def usage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report known usage with coverage; unavailable cost never becomes a zero charge."""
    result = {}
    for name in (
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",
        "cost_usd",
    ):
        values = [
            row["usage"][name] for row in rows if row["usage"].get(name) is not None
        ]
        result[name] = {
            "reported_sum": sum(values) if values else None,
            "reported_responses": len(values),
            "missing_responses": len(rows) - len(values),
        }
    return result


def usage_issues(usage: dict[str, Any]) -> list[str]:
    """Flag inconsistent provider counters without changing raw usage or scientific validity."""
    issues = []
    if (
        all(
            type(usage.get(key)) is int
            for key in ("reasoning_tokens", "completion_tokens")
        )
        and usage["reasoning_tokens"] > usage["completion_tokens"]
    ):
        issues.append("reasoning_exceeds_completion")
    if (
        all(
            type(usage.get(key)) is int
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
        and usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]
    ):
        issues.append("prompt_plus_completion_differs_from_total")
    return issues


def termination_category(finish_reason: str | None) -> str:
    """Annotate provider termination without changing slot, source or execution validity."""
    return {
        "error": "provider_error",
        "length": "token_limit",
        "stop": "normal_stop",
        None: "unreported",
    }.get(finish_reason, "other_reported_finish")


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate complete deployment comparisons while retaining candidate-only invalidity."""
    return {
        "responses": len(rows),
        "mean_auc": statistics.mean(row["validation_auc"] for row in rows),
        "median_auc": statistics.median(row["validation_auc"] for row in rows),
        "per_block_auc": {str(row["block"]): row["validation_auc"] for row in rows},
        "training_eligible": sum(row["train_valid"] for row in rows),
        "source_statuses": dict(Counter(row["source_status"] for row in rows)),
        "finish_reasons": dict(
            Counter(row["finish_reason"] or "unreported" for row in rows)
        ),
        "termination_categories": dict(
            Counter(row["termination_category"] for row in rows)
        ),
        "length_finishes": sum(row["finish_reason"] == "length" for row in rows),
        "candidate_valid_trajectories": sum(
            row["candidate_valid_trajectories"] for row in rows
        ),
        "fallback_trajectories": sum(row["fallback_trajectories"] for row in rows),
        "validation_trajectories": 6 * len(rows),
        "mean_final_regret": statistics.mean(row["final_regret"] for row in rows),
        "mean_target_attainment": statistics.mean(
            row["target_attainment"] for row in rows
        ),
        "mean_capped_target_evaluations": statistics.mean(
            row["capped_target_evaluations"] for row in rows
        ),
        "mean_parent_delta": statistics.mean(
            row["validation_auc"] - row["parent_auc"] for row in rows
        ),
        "mean_seed_delta": statistics.mean(
            row["validation_auc"] - row["seed_auc"] for row in rows
        ),
        "actual_objective_calls": sum(row["actual_objective_calls"] for row in rows),
        "unused_objective_allocations": sum(
            row["unused_objective_allocations"] for row in rows
        ),
        "subprocess_executions": sum(row["subprocess_executions"] for row in rows),
        "execution_s": sum(row["execution_s"] for row in rows),
        "response_wall_s": [row["wall_s"] for row in rows],
        "transport_attempts": sum(row["transport_attempts"] for row in rows),
        "transport_failures": sum(row["transport_attempts"] - 1 for row in rows),
        "usage": usage_summary(rows),
        "usage_issue_counts": dict(
            Counter(issue for row in rows for issue in row["usage_issues"])
        ),
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


def analyze() -> dict[str, Any]:
    """Audit a complete F1 run without generating, evaluating policies, or writing evidence."""
    root = F.ROOT
    if not all(
        E.exists(root / name)
        for name in (
            "results.json",
            "evaluations_frozen.json",
            "validation_opened.json",
        )
    ):
        raise RuntimeError("F1 final audit requires complete generation and validation")
    frozen = F.preflight()
    F._completed(frozen, require_all=True)
    expected_paths = {
        f'raw/{request["slot_id"]}/{name}.json'
        for request in frozen["requests"]
        for name in ("training", "validation")
    } | {
        f'controls/{block["block"]}_{name}.json'
        for block in frozen["blocks"]
        for name in ("parent", "seed")
    }
    evaluation_seal = E.read(root / "evaluations_frozen.json")
    if set(evaluation_seal) != expected_paths or not all(
        E.exists(root / path) for path in expected_paths
    ):
        raise RuntimeError(
            "F1 complete evaluation seal must include every registered allocation"
        )
    hashes: dict[str, str] = {}

    def read(path: Path) -> Any:
        """Record the canonical identity of every evidence object used by this audit."""
        value = E.read(path)
        hashes[str(path.relative_to(root))] = B.digest(value)
        return value

    for name in (
        "freeze.json",
        "freeze_sha256.json",
        "validation_opened.json",
        "evaluations_frozen.json",
    ):
        read(root / name)
    for path, digest in evaluation_seal.items():
        if B.digest(read(root / path)) != digest:
            raise RuntimeError("F1 evaluation seal differs from preserved observations")
    parents = {block["block"]: block for block in frozen["blocks"]}
    controls = {}
    preparation = []
    for block in frozen["blocks"]:
        context = read(root / f'contexts/{block["block"]}.json')
        F.verify_panel(
            context,
            block["parent"],
            frozen["tasks"]["train"],
            block["block"],
            deployment=False,
        )
        for row in context:
            verify_attempts(row, deployment=False)
        preparation.extend(context)
        for name, source in (("parent", block["parent"]), ("seed", B.SEED_SOURCE)):
            panel = read(root / f'controls/{block["block"]}_{name}.json')
            F.verify_panel(
                panel,
                source,
                frozen["tasks"]["validation"],
                block["block"],
                deployment=True,
            )
            for row in panel:
                verify_attempts(row, deployment=True)
            if any(row["fallback_used"] for row in panel):
                raise RuntimeError("F1 trusted parent/seed control failed")
            controls[block["block"], name] = panel
    rows = []
    summary_rows = []
    previous_completed = frozen["created_ns"]
    for position, request in enumerate(frozen["requests"]):
        directory = root / "raw" / request["slot_id"]
        read(directory / "request.json")
        response = read(directory / "response.json")
        train = read(directory / "training.json")
        validation = read(directory / "validation.json")
        for panel, split, deployment in (
            (train, "train", False),
            (validation, "validation", True),
        ):
            F.verify_panel(
                panel,
                response["source"],
                frozen["tasks"][split],
                request["block"],
                deployment=deployment,
            )
            for row in panel:
                verify_attempts(row, deployment=deployment)
        paths = sorted(
            directory.glob("attempt_*.json"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
        if not 1 <= len(paths) <= 4 or [path.name for path in paths] != [
            f"attempt_{i}.json" for i in range(1, len(paths) + 1)
        ]:
            raise RuntimeError(
                "F1 transport attempt allocation differs from the bounded policy"
            )
        attempts = [read(path) for path in paths]
        starts = [
            read(directory / path.name.replace("attempt_", "started_"))
            for path in paths
        ]
        if (
            response["attempt"] != len(attempts)
            or attempts[-1]["status"] != "completed"
            or attempts[-1]["id"] != response["id"]
            or any(
                attempt["status"] != "transport_failure" for attempt in attempts[:-1]
            )
            or starts[0]["time_ns"] < previous_completed
            or not all(start["time_ns"] <= response["completed_ns"] for start in starts)
        ):
            raise RuntimeError("F1 transport/request chronology or identity differs")
        previous_completed = response["completed_ns"]
        receipt = (
            read(directory / "provider_generation.json")
            if E.exists(directory / "provider_generation.json")
            else None
        )
        if receipt is not None and receipt.get("id") != response["id"]:
            raise RuntimeError(
                "F1 provider receipt does not match its completed response"
            )
        base = {
            "block": request["block"],
            "condition": request["condition"],
            "parent_kind": parents[request["block"]]["parent_kind"],
            "source_sha256": response["source_sha256"],
            "source_status": response["source_status"],
            "train_valid": all(row["valid"] for row in train),
            "candidate_valid_trajectories": sum(
                row["candidate_valid"] for row in validation
            ),
            "fallback_trajectories": sum(row["fallback_used"] for row in validation),
            "validation_auc": B.aggregate(validation, "auc"),
            "final_regret": B.aggregate(validation, "final_regret"),
            "usage": response["usage"],
        }
        summary_rows.append(base)
        rows.append(
            {
                **base,
                "slot_id": request["slot_id"],
                "position": position,
                "parent_sha256": request["parent_sha256"],
                "response_id": response["id"],
                "finish_reason": response["finish_reason"],
                "termination_category": termination_category(response["finish_reason"]),
                "parse_status": response["parse_status"],
                "raw_content_present": isinstance(response["content"], str),
                "raw_content_characters": (
                    len(response["content"])
                    if isinstance(response["content"], str)
                    else None
                ),
                "raw_fence_markers": (response["content"] or "").count("```"),
                "raw_contains_propose_definition_text": "def propose("
                in (response["content"] or ""),
                "usage_issues": usage_issues(response["usage"]),
                "source_diagnostics": source_diagnostics(response["source"]),
                "prompt_characters": sum(
                    len(message["content"]) for message in request["messages"]
                ),
                "condition_order": position % 4,
                "parent_auc": B.aggregate(controls[request["block"], "parent"], "auc"),
                "seed_auc": B.aggregate(controls[request["block"], "seed"], "auc"),
                "target_attainment": B.aggregate(validation, "attained"),
                "capped_target_evaluations": B.aggregate(
                    validation, "capped_target_evaluations"
                ),
                "target_evaluations": [
                    row["metrics"]["target_evaluations"] for row in validation
                ],
                "training_statuses": dict(Counter(row["status"] for row in train)),
                "validation_proposal_statuses": dict(
                    Counter(
                        attempt["status"]
                        for row in validation
                        for attempt in row["proposal_attempts"]
                    )
                ),
                "fallback_indices": [
                    next(
                        (
                            attempt["evaluation_index"]
                            for attempt in row["proposal_attempts"]
                            if attempt["status"] != "valid"
                        ),
                        None,
                    )
                    for row in validation
                ],
                "actual_objective_calls": sum(
                    row["objective_calls"] for row in train + validation
                ),
                "unused_objective_allocations": sum(
                    row["unused_objective_allocation"] for row in train + validation
                ),
                "subprocess_executions": sum(
                    row["subprocess_executions"] for row in train + validation
                ),
                "execution_s": sum(row["execution_s"] for row in train + validation),
                "wall_s": response["wall_s"],
                "transport_attempts": len(attempts),
                "attempt_measured_s": sum(attempt["wall_s"] for attempt in attempts),
                "slot_wall_s": (response["completed_ns"] - starts[0]["time_ns"]) / 1e9,
                "possible_remote_completion_attempts": sum(
                    attempt.get(
                        "possible_remote_completion_or_duplicate_billing", False
                    )
                    for attempt in attempts
                ),
                "receipt": receipt,
            }
        )
    contrasts = {}
    for treated, control in CONTRASTS:
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
    saved = read(root / "results.json")
    if saved != {
        "experiment": "EXP-16",
        "stage": "F1",
        "rows": summary_rows,
        "contrasts": contrasts,
    }:
        raise RuntimeError(
            "F1 saved efficacy summary differs from complete raw recomputation"
        )
    receipt_costs = [
        row["receipt"]["total_cost"]
        for row in rows
        if row["receipt"] and row["receipt"].get("total_cost") is not None
    ]
    control_rows = [row for panel in controls.values() for row in panel]
    return {
        "experiment": "EXP-16",
        "stage": "F1",
        "task_namespace": frozen["task_namespace"],
        "classification": "exploratory fixed-parent one-proposal deployment-policy contrasts",
        "cap": frozen["cap"],
        "rows": rows,
        "contrasts": contrasts,
        "conditions": {
            condition: group_summary(
                [row for row in rows if row["condition"] == condition]
            )
            for condition in F.CONDITIONS
        },
        "by_parent": {
            kind: {
                condition: group_summary(
                    [
                        row
                        for row in rows
                        if row["condition"] == condition and row["parent_kind"] == kind
                    ]
                )
                for condition in F.CONDITIONS
            }
            for kind in ("seed", "representative")
        },
        "order": {
            condition: dict(
                Counter(
                    str(row["condition_order"])
                    for row in rows
                    if row["condition"] == condition
                )
            )
            for condition in F.CONDITIONS
        },
        "usage": usage_summary(rows),
        "receipt_coverage": {
            "present": sum(row["receipt"] is not None for row in rows),
            "missing": sum(row["receipt"] is None for row in rows),
            "cost_receipts": len(receipt_costs),
            "reported_cost_sum": sum(receipt_costs) if receipt_costs else None,
            "model_identifiers": sorted(
                {
                    row["receipt"]["model"]
                    for row in rows
                    if row["receipt"] and row["receipt"].get("model")
                }
            ),
        },
        "total_allocation": {
            "response_slots": 24,
            "candidate_train_validation_trajectories": 288,
            "candidate_train_validation_objective_calls": 9216,
            "actual_candidate_objective_calls": sum(
                row["actual_objective_calls"] for row in rows
            ),
            "unused_candidate_objective_allocations": sum(
                row["unused_objective_allocations"] for row in rows
            ),
            "parent_training_preparation_trajectories": len(preparation),
            "parent_training_preparation_objective_calls": sum(
                row["objective_calls"] for row in preparation
            ),
            "validation_control_trajectories": len(control_rows),
            "validation_control_objective_calls": sum(
                row["objective_calls"] for row in control_rows
            ),
            "subprocess_executions": sum(
                row["subprocess_executions"]
                for row in rows + preparation + control_rows
            ),
            "reference_normalization": "128 shared deterministic reference evaluations per unique task, separate from policy budgets",
        },
        "chronology": {
            "unique_response_ids": len({row["response_id"] for row in rows}),
            "all_responses_precede_validation_barrier": True,
            "validation_boundary_guarantee": "frozen control flow and immutable response barrier; trajectory files have no independent evaluation-start timestamps",
        },
        "host_clock_observations": [
            read(path) for path in sorted(root.glob("clock_observation_*.json"))
        ],
        "limitations": [
            "Only six stochastic request blocks, three per fixed parent class; no new parent selected by these outcomes",
            "All four contrasts are exploratory; fragile bootstrap intervals have no multiple-comparison guarantee",
            "The primary estimand includes common seed fallback; candidate-only validity is reported separately",
            "Code-only arms still receive a fixed parent; this is not independent best-of-N or a complete recursive search",
            "Rich feedback here is raw-only; it contains no aggregate normalized training AUC scalar",
            "Request seeds do not establish deterministic LLM generation; routing and unequal realized tokens remain sources of variation",
            "Wall and monotonic timing cannot identify host suspension without an independent boot-clock record",
            "Unknown transport-attempt billing is not included in known-response receipt totals",
            "Provider reasoning/completion counters are retained verbatim even when inconsistent; reasoning is not added to reported total tokens",
            "A completed provider-error response consumes its frozen slot and is reported separately from source syntax or token-limit failures; deployment comparisons include this provider reliability variation",
            "No OS filesystem security isolation, novelty, amortization, or additional recursion-depth benefit is tested",
        ],
        "input_canonical_json_hashes": hashes,
    }


def main() -> None:
    """Persist one exact final audit after the complete registered F1 results exist."""
    result = analyze()
    I.persist(F.ROOT / "analysis_results.json", result)
    print(
        {
            "responses": len(result["rows"]),
            "contrasts": len(result["contrasts"]),
            "receipts": result["receipt_coverage"]["present"],
        }
    )


if __name__ == "__main__":
    main()
