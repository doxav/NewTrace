"""The EXP-18 CLI cannot bypass its frozen live grids or engineering gates."""

from pathlib import Path
from typing import Any

import pytest

from experiments.recursive_opt._shared.optimizer_discovery.exp17 import study as N
from experiments.recursive_opt._shared.optimizer_discovery.exp18 import driver as D
from experiments.recursive_opt._shared.optimizer_discovery.exp18.test_study import prepared
from tests.unit_tests.test_investigation16_search_experiment import _client


def test_exact_registered_engineering_and_main_allocations() -> None:
    """Main contains 384 slots and balanced pairwise whole-arm precedence."""
    memory, short = D.engineering_config("memory"), D.engineering_config("short")
    assert (
        memory["namespace"],
        memory["arms"],
        memory["outer_seeds"],
        memory["slots"],
    ) == ("EXP18-E1-M-v1", ["M"], [18901], 10)
    assert (
        short["namespace"],
        short["arms"],
        short["outer_seeds"],
        short["slots"],
    ) == ("EXP18-E1-SHORT-v1", ["L", "P", "PM"], [18903], 2)
    config = D.main_config()
    assert config["outer_seeds"] == [18011, 18023, 18037, 18041, 18053, 18067]
    assert config["arms"] == ["L", "M", "P", "PM"]
    assert config["slots"] * len(config["arms"]) * len(config["outer_seeds"]) == 384
    orders = list(N.arm_order(config).values())
    for first in config["arms"]:
        for second in config["arms"]:
            if first != second:
                assert sum(row.index(first) < row.index(second) for row in orders) == 3
    assert config["memory_max_sources"] == 7
    assert config["memory_max_chars"] == 65536


def test_both_engineering_results_are_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing proof cannot be replaced by a passing label or a model constructor."""
    monkeypatch.setattr(D, "MEMORY", tmp_path / "memory")
    monkeypatch.setattr(D, "SHORT", tmp_path / "short")
    with pytest.raises(RuntimeError, match="engineering"):
        D.require_engineering()
    D.I.persist(D.MEMORY / "engineering_results.json", {"passed": True})
    with pytest.raises((RuntimeError, KeyError, FileNotFoundError)):
        D.require_engineering()


def test_memory_gate_authenticates_naturally_accumulated_ninth_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The long pilot proves original slot-9 history, including all earlier responses."""
    root, state = prepared(tmp_path, monkeypatch, slots=10, arms=["M"])
    state["invalid_at"] = 2
    monkeypatch.setattr(D, "MEMORY", root)
    monkeypatch.setattr(
        D, "engineering_config", lambda kind: N.preflight(root)["config"]
    )
    N.run_generation(root, client=_client(state))
    N.select_all(root)
    before = len(state["evaluations"])
    value = D.engineering_evidence("memory")
    assert value["completed_responses"] == 10
    assert value["stage"] == N.preflight(root)["config"]["namespace"]
    assert value["audit_cache_rows"] == 0
    assert value["natural_memory_context"]["prior_slots"] == 9
    assert value["natural_memory_context"]["no_source_slots"] == 1
    assert value["resume_without_client_passed"]
    assert len(state["evaluations"]) == before
    D.I.persist(root / "engineering_results.json", {"passed": True, **value})
    assert D.require_pilot("memory")["passed"]
    response = root / "raw/18011/M/slot_00/response.json"
    response.write_text("{}")
    with pytest.raises((RuntimeError, KeyError)):
        D.require_pilot("memory")


def test_short_gate_authenticates_scalar_and_both_pareto_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every P/PM terminal selection is present and marked as generating no extra slot."""
    root, state = prepared(tmp_path, monkeypatch, arms=["L", "P", "PM"])
    monkeypatch.setattr(D, "SHORT", root)
    monkeypatch.setattr(
        D, "engineering_config", lambda kind: N.preflight(root)["config"]
    )
    N.run_generation(root, client=_client(state))
    N.select_all(root)
    value = D.engineering_evidence("short")
    assert value["completed_responses"] == 6
    assert value["current_contexts_authenticated"] == 6
    assert value["pareto_decisions_authenticated"] == 6
    assert value["pareto_unused_terminal_decisions"] == 2


def test_main_rejects_a_smaller_generic_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid two-slot unit run cannot be presented as the 384-slot design."""
    root, _ = prepared(tmp_path, monkeypatch)
    monkeypatch.setattr(D, "RUN", root)
    with pytest.raises(RuntimeError, match="registered grid"):
        D.require_main()


def test_engineering_rejects_audit_cache_even_without_an_audit_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unexported audit evaluation must still invalidate the engineering gate."""
    root, state = prepared(tmp_path, monkeypatch, arms=["L"])
    monkeypatch.setattr(D, "SHORT", root)
    monkeypatch.setattr(
        D, "engineering_config", lambda kind: N.preflight(root)["config"]
    )
    N.run_generation(root, client=_client(state))
    N.select_all(root)
    D.study.MechanismStudy(root, "L").panel(
        N.B.SEED_SOURCE, 18011, "audit", deployment=True
    )
    assert not D.E.exists(root / "audit_results.json")
    with pytest.raises(RuntimeError, match="accessed audit"):
        D.engineering_evidence("short")


def test_generate_cannot_create_live_client_before_both_pilot_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI checks completed engineering evidence before touching model credentials."""
    created: list[bool] = []

    def missing() -> dict[str, dict[str, Any]]:
        """Represent a genuinely unavailable prerequisite, without running a model."""
        raise RuntimeError("engineering is incomplete")

    monkeypatch.setattr(D, "require_engineering", missing)
    monkeypatch.setattr(D.SHARED, "live_client", lambda: created.append(True))
    with pytest.raises(RuntimeError, match="engineering"):
        D.execute("generate")
    assert not created


def test_prepare_preserves_pilot_protocol_and_freezes_all_adapters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Later main-report edits cannot overwrite the protocol used by either pilot."""
    original_root = D.ROOT
    monkeypatch.setattr(D, "ROOT", tmp_path)
    monkeypatch.setattr(D, "MEMORY", tmp_path / "memory")
    monkeypatch.setattr(D, "SHORT", tmp_path / "short")
    (tmp_path / "PREREG_EXP18.md").write_text("First protocol.\n")
    seen: list[dict[str, Any]] = []

    def freeze(
        root: Path, config: dict[str, Any], protocol: Path, **kwargs: Any
    ) -> dict[str, Any]:
        """Inspect preparation arguments; source archive implementation is tested upstream."""
        seen.append({"root": root, "config": config, "protocol": protocol, **kwargs})
        return {"config": config}

    monkeypatch.setattr(N, "prepare", freeze)
    monkeypatch.setattr(D.SHARED, "source_archive", lambda root: None)
    D.prepare("memory")
    (tmp_path / "PREREG_EXP18.md").write_text("Later main protocol.\n")
    D.prepare("short")
    assert (tmp_path / "PILOT_PROTOCOL.md").read_text() == "First protocol.\n"
    assert all(row["protocol"].name == "PILOT_PROTOCOL.md" for row in seen)
    names = {path.name for path in seen[0]["extra_frozen_paths"]}
    assert {
        "study.py",
        "memory_projection.py",
        "pareto_selection.py",
        "pareto_trainer.py",
        "driver.py",
        "analysis.py",
        "test_analysis.py",
    } <= names
    assert any(path.parent == original_root for path in seen[0]["extra_frozen_paths"])


def test_prepare_main_cannot_run_with_one_missing_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Main preparation does not allocate new scientific work before both gates pass."""
    calls: list[str] = []

    def pilot(kind: str) -> dict[str, Any]:
        """Present one available pilot and one unavailable pilot."""
        calls.append(kind)
        if kind == "short":
            raise RuntimeError("short engineering incomplete")
        return {"passed": True}

    monkeypatch.setattr(D, "require_pilot", pilot)
    with pytest.raises(RuntimeError, match="short engineering"):
        D.prepare()
    assert calls == ["memory", "short"]
