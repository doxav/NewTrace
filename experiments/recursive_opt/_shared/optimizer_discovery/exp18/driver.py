"""Bounded live CLI for the registered EXP-18 memory/Pareto mechanism study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.exp17 import analysis, verify_numerics
from experiments.recursive_opt._shared.optimizer_discovery.exp17 import driver as SHARED
from experiments.recursive_opt._shared.optimizer_discovery.exp17 import study as N
from experiments.recursive_opt._shared.optimizer_discovery.exp18 import (
    memory_projection,
    pareto_selection,
    pareto_trainer,
    study,
)
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I

ROOT = Path(__file__).resolve().parent
MEMORY = ROOT / "engineering_memory"
SHORT = ROOT / "engineering_short"
RUN = ROOT / "run"


def engineering_config(kind: str) -> dict[str, Any]:
    """Return the disjoint long-memory or short multi-arm engineering allocation."""
    if kind not in {"memory", "short"}:
        raise ValueError("unknown EXP-18 engineering stage")
    return N.configuration(
        experiment="EXP-18",
        namespace="EXP18-E1-M-v1" if kind == "memory" else "EXP18-E1-SHORT-v1",
        arms=["M"] if kind == "memory" else ["L", "P", "PM"],
        outer_seeds=[18901] if kind == "memory" else [18903],
        slots=10 if kind == "memory" else 2,
    )


def main_config() -> dict[str, Any]:
    """Return the exact six-pair, four-arm, sixteen-slot exploratory study."""
    config = N.configuration(
        experiment="EXP-18",
        namespace="EXP18-MECHANISMS-v1",
        arms=["L", "M", "P", "PM"],
        outer_seeds=[18011, 18023, 18037, 18041, 18053, 18067],
        slots=16,
    )
    config["arm_orders"] = [
        ["L", "M", "P", "PM"],
        ["PM", "P", "M", "L"],
        ["M", "PM", "L", "P"],
        ["P", "L", "PM", "M"],
        ["P", "L", "M", "PM"],
        ["PM", "M", "L", "P"],
    ]
    N.validate_config(config)
    return config


def _pilot_root(kind: str) -> Path:
    """Resolve a declared engineering stage without falling back to another run."""
    if kind not in {"memory", "short"}:
        raise ValueError("unknown EXP-18 engineering stage")
    return MEMORY if kind == "memory" else SHORT


def engineering_evidence(kind: str) -> dict[str, Any]:
    """Authenticate complete real-pilot records, causal contexts and no-work resume."""
    root = _pilot_root(kind)
    frozen = N.preflight(root)
    if frozen["config"] != engineering_config(kind):
        raise RuntimeError(
            "engineering configuration differs from its registered scope"
        )
    value = SHARED.engineering_evidence(root)
    value["stage"] = frozen["config"]["namespace"]
    audit_rows = 0
    for path in (root / "cache").glob("*.json*"):
        if path.suffix not in {".json", ".gz"}:
            continue
        logical = path.with_suffix("") if path.suffix == ".gz" else path
        audit_rows += E.read(logical)["key"]["split"] == "audit"
    if audit_rows:
        raise RuntimeError("engineering accessed audit trajectories")
    contexts = 0
    decisions = 0
    unused = 0
    non_scalar = 0
    natural_memory = None
    for outer in frozen["config"]["outer_seeds"]:
        for arm in frozen["config"]["arms"]:
            owner = study.MechanismStudy(root, arm)
            directory = root / "raw" / str(outer) / arm
            for slot in range(frozen["config"]["slots"]):
                context = E.read(directory / f"slot_{slot:02d}/current_context.json")
                contexts += 1
                if kind == "memory" and slot == 9:
                    memory = context["memory"]
                    if (
                        arm != "M"
                        or memory is None
                        or memory["snapshot"]["before_slot"] != 9
                        or memory["counts"]["prior_slots"] != 9
                        or memory["snapshot"]["snapshot_ns"] != context["snapshot_ns"]
                        or {row["slot"] for row in memory["provenance"]}
                        != set(range(9))
                    ):
                        raise RuntimeError(
                            "long pilot did not preserve natural prior-attempt history"
                        )
                    natural_memory = {
                        "slot": slot,
                        **memory["counts"],
                        "context_sha256": N.B.digest(context),
                        "memory_text_sha256": memory["text_sha256"],
                    }
            if arm in {"P", "PM"}:
                for slot in range(frozen["config"]["slots"] + 1):
                    saved = E.read(directory / f"parent_decisions/slot_{slot:02d}.json")
                    reconstructed = pareto_selection.select_pareto_parent(
                        owner._archive(outer, slot),
                        namespace=frozen["config"]["namespace"],
                        outer_seed=outer,
                        next_slot=slot,
                    )
                    if any(
                        saved.get(key) != item for key, item in reconstructed.items()
                    ) or saved["will_generate"] != (slot < frozen["config"]["slots"]):
                        raise RuntimeError("engineering Pareto decision failed replay")
                    decisions += 1
                    unused += not saved["will_generate"]
                    non_scalar += (
                        saved["will_generate"]
                        and saved["selected_source_sha256"]
                        != saved["scalar_best_source_sha256"]
                    )
    if kind == "memory" and natural_memory is None:
        raise RuntimeError(
            "long engineering pilot did not reach its ninth prior-attempt context"
        )
    return {
        **value,
        "audit_cache_rows": audit_rows,
        "current_contexts_authenticated": contexts,
        "pareto_decisions_authenticated": decisions,
        "pareto_unused_terminal_decisions": unused,
        "pareto_non_scalar_parent_updates": non_scalar,
        "natural_memory_context": natural_memory,
    }


def require_pilot(kind: str) -> dict[str, Any]:
    """Reject absent, changed or unusable pilot evidence before allocating main calls."""
    path = _pilot_root(kind) / "engineering_results.json"
    if not E.exists(path):
        raise RuntimeError("completed EXP-18 engineering evidence is required")
    saved = E.read(path)
    current = engineering_evidence(kind)
    if (
        saved.get("passed") is not True
        or current["eligible_generated"] < 1
        or any(saved.get(key) != value for key, value in current.items())
    ):
        raise RuntimeError("EXP-18 engineering gate or evidence integrity failed")
    return saved


def require_engineering() -> dict[str, dict[str, Any]]:
    """Require both independent engineering stages, preserving each one's outcome."""
    return {kind: require_pilot(kind) for kind in ("memory", "short")}


def require_main() -> dict[str, Any]:
    """Compare the current run against the exact registered 384-response grid."""
    frozen = N.preflight(RUN)
    if frozen["config"] != main_config():
        raise RuntimeError("EXP-18 configuration differs from the registered grid")
    return frozen


def prepare(kind: str | None = None) -> dict[str, Any]:
    """Freeze the complete implementation with immutable pilot-protocol copies."""
    if kind is None:
        require_engineering()
    root = RUN if kind is None else _pilot_root(kind)
    protocol = ROOT / ("PREREG_EXP18.md" if kind is None else "PILOT_PROTOCOL.md")
    if kind is not None and not protocol.exists():
        protocol.write_bytes((ROOT / "PREREG_EXP18.md").read_bytes())
    implementations = [
        study,
        memory_projection,
        pareto_selection,
        pareto_trainer,
        analysis,
        verify_numerics,
        SHARED,
    ]
    frozen = N.prepare(
        root,
        main_config() if kind is None else engineering_config(kind),
        protocol,
        extra_frozen_paths=[
            Path(__file__),
            *(Path(module.__file__) for module in implementations),
            N.ROOT / "test_analysis.py",
            N.ROOT / "test_verify_numerics.py",
            N.ROOT / "test_source_archive.py",
            *Path(__file__).parent.glob("test_*.py"),
        ],
    )
    SHARED.source_archive(root)
    return frozen


def run_engineering(kind: str, *, client: Any = None) -> dict[str, Any]:
    """Run the registered responses and selection, with no pilot audit evaluation."""
    root = _pilot_root(kind)
    if E.exists(root / "engineering_results.json"):
        return require_pilot(kind)
    if N.preflight(root)["config"] != engineering_config(kind):
        raise RuntimeError(
            "engineering configuration differs from its registered scope"
        )
    N.run_generation(root, client=SHARED.live_client() if client is None else client)
    SHARED.OLD.collect_receipts(root / "raw")
    N.select_all(root)
    value = engineering_evidence(kind)
    I.persist(
        root / "engineering_results.json",
        {"passed": value["eligible_generated"] > 0, **value},
    )
    return require_pilot(kind)


def execute(action: str) -> None:
    """Execute one declared stage under the CLI's exclusive run lease."""
    if action in {"prepare_memory", "prepare_short", "prepare"}:
        kind = action.removeprefix("prepare_") if action != "prepare" else None
        frozen = prepare(kind)
        print(
            json.dumps(
                {"freeze_sha256": N.B.digest(frozen), "config": frozen["config"]}
            ),
            flush=True,
        )
    elif action in {"memory", "short"}:
        result = run_engineering(action)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("passed", "completed_responses", "eligible_generated")
                }
            ),
            flush=True,
        )
    elif action == "generate":
        require_engineering()
        require_main()
        N.run_generation(RUN, client=SHARED.live_client())
        SHARED.OLD.collect_receipts(RUN / "raw")
    elif action in {"select", "audit", "analyze"}:
        require_main()
        if action == "select":
            N.select_all(RUN)
        elif action == "audit":
            N.run_audit(RUN)
        else:
            I.persist(
                RUN / "analysis_results.json",
                analysis.summarize(analysis.read_bundle(RUN)),
            )
    elif action == "receipts":
        SHARED.OLD.collect_receipts(RUN / "raw")
    else:
        raise ValueError("unknown EXP-18 action")


def main() -> None:
    """Hold the shared process lease around all mutations and reuse the provider lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "prepare_memory",
            "memory",
            "prepare_short",
            "short",
            "prepare",
            "generate",
            "select",
            "audit",
            "analyze",
            "receipts",
        ],
    )
    action = parser.parse_args().action
    root = (
        MEMORY
        if action in {"prepare_memory", "memory"}
        else SHORT if action in {"prepare_short", "short"} else RUN
    )
    with SHARED.run_lease(root):
        execute(action)


if __name__ == "__main__":
    main()
