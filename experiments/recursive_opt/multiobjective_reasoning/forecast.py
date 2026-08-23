"""Post-micro-smoke token and monetary cost forecast for Experiment 0."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .live import MICRO_REPORT_PATH, PILOT_REPORT_PATH
from .preflight import _load_json


PACKAGE_ROOT = Path(__file__).resolve().parent


def _role_tokens(run: Mapping[str, Any]) -> dict[str, float]:
    totals = {"prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0}
    for role in ("forward", "optimizer"):
        usage = run["usage"].get(role, {})
        for name in totals:
            totals[name] += float(usage.get(name, 0))
    return totals


def _priced_cost(tokens: Mapping[str, float], pricing: Mapping[str, Any]) -> float:
    input_rate = float(pricing["input_usd_per_million_tokens"])
    output_rate = float(pricing["output_usd_per_million_tokens"])
    return (
        float(tokens["prompt_tokens"]) * input_rate
        + float(tokens["completion_tokens"]) * output_rate
    ) / 1_000_000


def _scaled(tokens: Mapping[str, float], multiplier: float) -> dict[str, float]:
    return {name: float(value) * multiplier for name, value in tokens.items()}


def _sum_tokens(values: list[Mapping[str, float]]) -> dict[str, float]:
    return {
        name: sum(float(value[name]) for value in values)
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _pilot_tokens_by_arm(runs: list[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """Sum actual forward and optimizer tokens for every completed pilot arm."""
    totals: dict[str, dict[str, float]] = {}
    for run in runs:
        arm = str(run["arm"])
        target = totals.setdefault(
            arm,
            {"prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0},
        )
        for name, value in _role_tokens(run).items():
            target[name] += value
    return totals


def _retry_costs(
    retry_statistics: Mapping[str, Mapping[str, Any]], pricing: Mapping[str, Any]
) -> dict[str, float]:
    """Price only additional semantic-retry attempts from recorded token usage."""
    return {
        arm: _priced_cost(
            {
                "prompt_tokens": float(values.get("semantic_retry_prompt_tokens", 0)),
                "completion_tokens": float(
                    values.get("semantic_retry_completion_tokens", 0)
                ),
            },
            pricing,
        )
        for arm, values in retry_statistics.items()
    }


def build_cost_forecast() -> dict[str, Any]:
    """Forecast from smoke before the pilot, then from complete pilot usage."""
    micro = _load_json(MICRO_REPORT_PATH)
    if not micro.get("passed"):
        raise RuntimeError("cost forecast requires a passing A/B/C micro-smoke")
    preregistration = _load_json(PACKAGE_ROOT / "manifests/preregistration_v2.json")
    pricing = _load_json(PACKAGE_ROOT / "manifests/provider_pricing.json")
    smoke_tokens = {arm: _role_tokens(run) for arm, run in micro["arms"].items()}
    smoke_costs = {
        arm: _priced_cost(tokens, pricing) for arm, tokens in smoke_tokens.items()
    }
    micro_examples = sum(
        int(value)
        for value in preregistration["dataset_pools"]["micro_smoke_subset"].values()
    )
    pilot_examples = sum(
        int(value)
        for value in preregistration["dataset_pools"]["pilot_subset"].values()
    )
    full_examples = sum(
        int(value)
        for value in preregistration["dataset_pools"]["minimum_sizes"].values()
    )
    pilot_sample_scale = pilot_examples / micro_examples
    main_sample_scale = full_examples / micro_examples
    pilot_to_main_sample_scale = full_examples / pilot_examples
    pilot_proposal_units = 3 * sum(
        int(value) for value in preregistration["pilot"]["candidate_budgets"]
    )
    main_proposal_units = 5 * (6 + 12)
    projected_pilot_by_arm = {
        "A": _scaled(smoke_tokens["A"], pilot_sample_scale * 6),
        "B": _scaled(smoke_tokens["B"], pilot_sample_scale * pilot_proposal_units),
        "C": _scaled(smoke_tokens["C"], pilot_sample_scale * pilot_proposal_units),
        "D": _scaled(smoke_tokens["B"], pilot_sample_scale * pilot_proposal_units),
    }
    pilot = _load_json(PILOT_REPORT_PATH) if PILOT_REPORT_PATH.exists() else None
    pilot_complete = bool(pilot and pilot.get("passed"))
    pilot_by_arm = (
        _pilot_tokens_by_arm(pilot["runs"])
        if pilot_complete and pilot is not None
        else projected_pilot_by_arm
    )
    pilot_tokens = _sum_tokens(list(pilot_by_arm.values()))
    if pilot_complete:
        main_by_arm = {
            arm: _scaled(
                tokens,
                pilot_to_main_sample_scale
                * (10 / 6 if arm == "A" else main_proposal_units / pilot_proposal_units),
            )
            for arm, tokens in pilot_by_arm.items()
        }
    else:
        main_by_arm = {
            "A": _scaled(smoke_tokens["A"], main_sample_scale * 10),
            "B": _scaled(smoke_tokens["B"], main_sample_scale * main_proposal_units),
            "C": _scaled(smoke_tokens["C"], main_sample_scale * main_proposal_units),
            "D": _scaled(smoke_tokens["B"], main_sample_scale * main_proposal_units),
        }
    main_tokens = _sum_tokens(list(main_by_arm.values()))
    main_cost = _priced_cost(main_tokens, pricing)
    retry_statistics = pilot.get("retry_statistics_by_arm", {}) if pilot_complete and pilot else {}
    return {
        "schema_version": "recursive-opt-cost-forecast/v3",
        "method": (
            "actual complete pilot usage, including semantic retries, scaled by frozen sample and proposal units"
            if pilot_complete
            else "linear projection from actual one-unit smoke usage; optimized arms scale by frozen proposal units and stage sample counts; D uses the measured Trace-B usage profile"
        ),
        "pricing": pricing,
        "micro_smoke": {
            "measured_tokens_by_arm": smoke_tokens,
            "estimated_cost_usd_by_arm": smoke_costs,
            "estimated_total_cost_usd": sum(smoke_costs.values()),
        },
        "pilot": {
            "complete": pilot_complete,
            "tokens_by_arm": pilot_by_arm,
            "tokens": pilot_tokens,
            "cost_usd": _priced_cost(pilot_tokens, pricing),
            "retry_statistics_by_arm": retry_statistics,
            "retry_cost_usd_by_arm": _retry_costs(retry_statistics, pricing),
        },
        "main_full_v2_pool": {
            "assumed_seeds": 5,
            "assumed_candidate_budgets": [6, 12],
            "projected_tokens_by_arm": main_by_arm,
            "projected_tokens": main_tokens,
            "projected_cost_usd": main_cost,
        },
        "pilot_forecast_complete": True,
        "main_monetary_ceiling_usd": preregistration["main_monetary_ceiling_usd"],
        "main_run_authorized": preregistration["main_monetary_ceiling_usd"] is not None,
        "main_stop_reason": "no explicit main-run monetary ceiling is acknowledged",
        "recommended_main_cost_ceiling_usd": main_cost * 1.2,
        "recommended_ceiling_safety_margin_fraction": 0.2,
    }
