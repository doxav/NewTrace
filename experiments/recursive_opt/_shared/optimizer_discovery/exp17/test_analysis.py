"""Synthetic evidence and real local integration tests; no live model calls."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery import exp17
from experiments.recursive_opt._shared.optimizer_discovery.exp17 import analysis as A
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import search_experiment as S
from experiments.recursive_opt._shared.optimizer_discovery.investigation16.production import test_analysis as T


def bundle(kind: str = "exp17", seeds: list[int] | None = None) -> dict[str, Any]:
    """Adapt the established synthetic evidence fixture to two studies and B2."""
    raw = T.bundle(seeds or [17101, 17103])
    old_source = "invalid candidate ("
    invalid_source = "def propose(history, bounds, seed):\n    return (\n"
    old_hash, invalid_hash = B.source_hash(old_source), B.source_hash(invalid_source)

    def replace_source(value: Any) -> Any:
        """Give the invalid fixture an extractable but syntactically invalid API."""
        if isinstance(value, dict):
            return {key: replace_source(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace_source(item) for item in value]
        if isinstance(value, str):
            return {old_source: invalid_source, old_hash: invalid_hash}.get(
                value, value
            )
        return value

    raw = replace_source(raw)
    cache_keys = {key: B.digest(entry["key"]) for key, entry in raw["cache"].items()}
    raw["cache"] = {
        cache_keys[key]: {**entry, "row_sha256": B.digest(entry["row"])}
        for key, entry in raw["cache"].items()
    }
    for event in raw["events"]:
        event["key"] = cache_keys[event["key"]]
    config = raw["freeze"]["config"]
    config.update(
        analysis_kind=kind,
        representative_arm="C" if kind == "exp17" else "PM",
        max_tokens=32000,
        model=S.G.MODEL,
        bootstrap=B.MANIFEST["bootstrap"],
    )
    names = (
        {"I": "I", "C": "C"}
        if kind == "exp17"
        else {"I": "L", "C": "M", "R": "P", "W": "PM"}
    )
    config["arms"] = list(names.values())
    for key in ("pools", "selections", "slots"):
        entries = {}
        for name, value in raw[key].items():
            outer, old_arm, *suffix = name.split("/")
            if old_arm in names:
                entries["/".join([outer, names[old_arm], *suffix])] = value
        raw[key] = entries
    control = B.SEED_SOURCE + "\n# fixed control fixture\n"
    raw["freeze"]["fixed_controls"] = {
        "B2": {"source": control, "source_sha256": B.source_hash(control)}
    }
    for outer, results in raw["audit"]["per_seed"].items():
        new = {"A0": results["A0"]}
        new.update({names[arm]: results[arm] for arm in names})
        new["B2"] = copy.deepcopy(results["A0"])
        new["B2"]["source_sha256"] = B.source_hash(control)
        for row in new["B2"]["rows"]:
            old_identity = (row["task_identity"], row["local_seed"])
            old = next(
                entry
                for entry in raw["cache"].values()
                if entry["key"]["split"] == "audit"
                and entry["key"]["outer"] == int(outer)
                and entry["key"]["source_sha256"] == raw["freeze"]["seed_sha256"]
                and (entry["row"]["task_identity"], entry["row"]["local_seed"])
                == old_identity
            )
            row["source_sha256"] = B.source_hash(control)
            key = {**old["key"], "source_sha256": row["source_sha256"]}
            digest = B.digest(key)
            raw["cache"][digest] = {
                "key": key,
                "row": copy.deepcopy(row),
                "row_sha256": B.digest(row),
            }
            raw["events"].append(
                {"event": "evaluation_cache", "key": digest, "hit": False}
            )
        raw["audit"]["per_seed"][outer] = new
    for name, slot in raw["slots"].items():
        outer, arm, index = name.split("/")
        request, response = slot["request"], slot["response"]
        request.update(
            outer=int(outer),
            arm=arm,
            slot=int(index),
            slot_id=name,
            model=config["model"],
            freeze_sha256=B.digest(raw["freeze"]),
            settings=S._generation_settings(config, int(outer), int(index)),
            messages=[{"role": "user", "content": S._invariant(config)}],
        )
        slot["path"] = f"raw/{outer}/{arm}/slot_{int(index):02d}"
        response.update(
            content=f"```python\n{response['source']}\n```",
            completed_ns=1000,
            attempt=2,
        )
        response["source_status"] = B.source_status(response["source"])
    return raw


def test_exp17_preserves_all_negative_invalid_and_allocated_rows() -> None:
    """The primary and secondary outputs retain failures and unfavorable controls."""
    raw = bundle(seeds=list(range(17101, 17147)))
    result = A.summarize(raw)
    assert len(result["per_seed"]) == 46
    assert set(result["arms"]) == {"A0", "B2", "I", "C"}
    assert result["contrast_roles"]["C-I"] == "confirmatory_primary"
    assert result["contrasts"]["C-B2"]["mean"] == pytest.approx(0.1)
    assert result["contrasts"]["C-B2"]["interpretation"] == "negative signal"
    assert result["arms"]["C"]["candidate_eligibility"]["ineligible_generated"] == 46
    assert result["resources"]["allocated_proposal_slots"] == 184
    assert result["resources"]["completed_responses"] == 184
    assert result["resources"]["transport_attempts"] == 368
    assert result["resources"]["logical"]["train"]["unused_objective_allocation"] > 0
    assert (
        result["resources"]["physical_cache"]["train"]["objective_calls"]
        < result["resources"]["logical"]["train"]["objective_calls"]
    )
    assert (
        result["resources"]["known_usage"]["usage"]["cost_usd"]["reported_sum"] is None
    )
    assert result["representative"]["arm"] == "C"


def test_exp18_factorial_uses_outer_seed_joint_differences() -> None:
    """Main effects and interaction use paired rows rather than pooled trajectories."""
    raw = bundle("exp18")
    result = A.summarize(raw)
    assert result["contrast_roles"]["memory"] == "exploratory_factorial"
    values = {
        "1": {
            a: {"auc": x}
            for a, x in zip(
                ("L", "M", "P", "PM", "A0", "B2"), (1.0, 2.0, 4.0, 8.0, 9.0, 10.0)
            )
        },
        "2": {
            a: {"auc": x}
            for a, x in zip(
                ("L", "M", "P", "PM", "A0", "B2"), (2.0, 4.0, 5.0, 9.0, 9.0, 10.0)
            )
        },
    }
    contrasts, roles = A.contrasts(values, [1, 2], "exp18")
    assert contrasts["memory"] == E.paired([2.5, 3.0])
    assert contrasts["pareto"] == E.paired([4.5, 4.0])
    assert contrasts["interaction"] == E.paired([3.0, 2.0])
    assert roles["PM-L"] == "descriptive_secondary"
    assert "I" not in result["arms"]


@pytest.mark.parametrize(
    "remove", ["seed", "arm", "slot", "candidate", "trajectory", "cache", "B2"]
)
def test_refuses_missing_evidence(remove: str) -> None:
    """No missing item can be silently omitted from an aggregate."""
    raw = bundle()
    if remove == "seed":
        raw["audit"]["per_seed"].pop("17101")
    elif remove in ("arm", "B2"):
        raw["audit"]["per_seed"]["17101"].pop("C" if remove == "arm" else "B2")
    elif remove == "slot":
        raw["slots"].pop("17101/C/1")
    elif remove == "candidate":
        raw["pools"]["17101/C"].pop()
    elif remove == "trajectory":
        raw["pools"]["17101/C"][2]["train"].pop()
    else:
        raw["cache"].pop(next(iter(raw["cache"])))
    with pytest.raises(ValueError):
        A.summarize(raw)


@pytest.mark.parametrize(
    "change",
    [
        "settings",
        "model",
        "invariant",
        "parse",
        "source",
        "duplicate_id",
        "attempt_id",
        "freeze",
        "control",
        "bootstrap",
    ],
)
def test_refuses_request_response_or_freeze_inconsistency(change: str) -> None:
    """Models, prompts, parsing and raw source identities remain part of integrity."""
    raw = bundle()
    slot = raw["slots"]["17101/C/0"]
    if change == "settings":
        slot["request"]["settings"]["max_tokens"] = 8000
    elif change == "model":
        slot["request"]["model"] = "another-model"
    elif change == "invariant":
        slot["request"]["messages"][0]["content"] = "altered objective"
    elif change == "parse":
        slot["response"]["content"] = "no code"
    elif change == "source":
        slot["response"]["source"] += "\n# changed"
    elif change == "duplicate_id":
        slot["response"]["id"] = raw["slots"]["17101/I/0"]["response"]["id"]
    elif change == "attempt_id":
        slot["attempts"][-1]["id"] = "wrong-attempt"
    elif change == "freeze":
        slot["request"]["freeze_sha256"] = "wrong"
    elif change == "control":
        raw["freeze"]["fixed_controls"]["B2"]["source"] += "edited"
    else:
        raw["freeze"]["config"]["bootstrap"] = {**B.MANIFEST["bootstrap"], "seed": 18}
    with pytest.raises(ValueError):
        A.summarize(raw)


def test_refuses_fabricated_invalid_metric_and_holdout_selected_winner() -> None:
    """Invalidity and validation choice cannot be repaired by analysis."""
    raw = bundle()
    raw["pools"]["17101/C"][2]["train"][0]["metrics"] = {"auc": 1000.0}
    with pytest.raises(ValueError):
        A.summarize(raw)
    raw = bundle()
    raw["selections"]["17101/C"]["index"] = -1
    with pytest.raises(ValueError, match="selection"):
        A.summarize(raw)


def test_rejects_aggregate_or_curve_corruption() -> None:
    """Saved audit means and anytime curves must agree with preserved rows."""
    raw = bundle()
    raw["audit"]["per_seed"]["17101"]["C"]["auc"] += 0.1
    with pytest.raises(ValueError, match="aggregate"):
        A.summarize(raw)
    raw = bundle()
    entry = next(iter(raw["cache"].values()))
    entry["row"]["metrics"]["curve"] = [0.1, 0.5]
    entry["row_sha256"] = B.digest(entry["row"])
    with pytest.raises(ValueError, match="nonincreasing|AUC"):
        A.summarize(raw)


def test_receipts_fill_unknown_cost_without_double_counting() -> None:
    """Optional receipts fill missing fields only and retain missing coverage."""
    raw = bundle()
    slot = raw["slots"]["17101/C/0"]
    slot["response"]["usage"]["cost_usd"] = 0.01
    slot["metadata"] = {"id": slot["response"]["id"], "total_cost": 999.0}
    result = A.summarize(raw)
    assert (
        result["arms"]["C"]["known_usage"]["usage"]["cost_usd"]["reported_sum"] == 0.01
    )
    slot["metadata"]["id"] = "different"
    with pytest.raises(ValueError, match="receipt"):
        A.summarize(raw)


def write_fixture(root: Path, raw: dict[str, Any]) -> None:
    """Persist synthetic evidence with realistic barriers for file-reader checks."""
    raw["freeze"]["files"][str(Path(A.__file__).resolve())] = B.source_hash(
        Path(A.__file__).read_text()
    )
    for slot in raw["slots"].values():
        slot["request"]["freeze_sha256"] = B.digest(raw["freeze"])
    representative_arm = raw["freeze"]["config"]["representative_arm"]
    outer = raw["freeze"]["config"]["outer_seeds"][0]
    barrier = {
        "completed_ns": 2000,
        "representative_outer": outer,
        "representative": raw["selections"][f"{outer}/{representative_arm}"],
    }
    raw["audit"]["selections_frozen_ns"] = 2000
    for name, value in (
        ("freeze.json", raw["freeze"]),
        ("audit_results.json", raw["audit"]),
        ("selections_frozen.json", barrier),
        ("generation_frozen.json", {"completed_ns": 1500}),
    ):
        I.persist(root / name, value)
    for name, pool in raw["pools"].items():
        I.persist(root / "raw" / name / "pool.json", pool)
        I.persist(root / "raw" / name / "selection.json", raw["selections"][name])
    for slot in raw["slots"].values():
        directory = root / slot["path"]
        for field in ("request", "response"):
            I.persist(directory / f"{field}.json", slot[field])
        for index, attempt in enumerate(slot["attempts"], 1):
            I.persist(directory / f"attempt_{index}.json", attempt)
    for key, entry in raw["cache"].items():
        I.persist(root / "cache" / f"{key}.json", entry)
    for index, event in enumerate(raw["events"]):
        I.persist(root / "events" / f"{index:05d}.json", event)


def test_file_reader_checks_barriers_extra_responses_and_ambiguous_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reading preserves complete evidence while delegating live-code guards lazily."""
    raw = bundle()
    write_fixture(tmp_path, raw)
    study = SimpleNamespace(
        preflight=lambda root: E.read(root / "freeze.json"),
        verify_chronology=lambda root: {"verified": True},
    )
    monkeypatch.setitem(sys.modules, "experiments.recursive_opt._shared.optimizer_discovery.exp17.study", study)
    monkeypatch.setattr(exp17, "study", study, raising=False)
    read = A.read_bundle(tmp_path)
    assert A.summarize(read)["resources"]["completed_responses"] == 8
    directory = tmp_path / "raw/17101/C/slot_00"
    I.persist(directory / "started_3.json", {"status": "in_flight"})
    with pytest.raises(ValueError, match="unreconciled"):
        A.read_bundle(tmp_path)
    (directory / "started_3.json").unlink()
    I.persist(
        tmp_path / "raw/17101/C/slot_99/response.json",
        raw["slots"]["17101/C/0"]["response"],
    )
    with pytest.raises(ValueError, match="unexpected completed"):
        A.read_bundle(tmp_path)


def test_analysis_never_evaluates_objectives_or_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Analysis depends only on stored curves and hashes, not new scientific work."""
    raw = bundle()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("scientific evaluation called from analysis")

    monkeypatch.setattr(B, "evaluate", forbidden)
    monkeypatch.setattr(B, "objective", forbidden)
    A.summarize(raw)


def synchronize_cache(raw: dict[str, Any]) -> None:
    """Rebuild matching synthetic cache identities after a deliberate fixture change."""
    cache = {}
    for entry in raw["cache"].values():
        key = entry["key"]
        if key["split"] == "audit":
            panels = [
                value["rows"]
                for value in raw["audit"]["per_seed"][str(key["outer"])].values()
            ]
        else:
            panels = [
                candidate[key["split"]]
                for name, pool in raw["pools"].items()
                if name.startswith(f"{key['outer']}/")
                for candidate in pool
            ]
        rows = [
            row
            for panel in panels
            for row in panel
            if all(
                row[field] == key[field]
                for field in ("source_sha256", "task_identity", "local_seed")
            )
        ]
        if rows:
            assert all(row == rows[0] for row in rows)
            cache[B.digest(key)] = {
                "key": key,
                "row": copy.deepcopy(rows[0]),
                "row_sha256": B.digest(rows[0]),
            }
    raw["cache"] = cache
    raw["events"] = [
        {"event": "evaluation_cache", "key": key, "hit": False} for key in cache
    ]


def test_deployment_fallback_keeps_actual_metric_and_reports_candidate_failure() -> (
    None
):
    """Fallback does not erase candidate invalidity or reset evaluated observations."""
    raw = bundle()
    for arm in ("I", "C"):
        deployment = raw["audit"]["per_seed"]["17101"][arm]
        row = deployment["rows"][0]
        row.update(candidate_valid=False, fallback_used=True, subprocess_executions=5)
        row["proposal_attempts"].insert(
            0,
            {
                "evaluation_index": 0,
                "status": "exception",
                "policy": "candidate",
                "stdout": "",
                "stderr": "",
            },
        )
        deployment["fallback_trajectories"] = 1
    synchronize_cache(raw)
    result = A.summarize(raw)
    assert result["per_seed"]["17101"]["C"]["auc"] == pytest.approx(0.3)
    assert result["arms"]["C"]["deployment"]["fallback_fraction"] == pytest.approx(
        1 / 12
    )
    assert result["arms"]["C"]["deployment"][
        "candidate_invalid_fraction"
    ] == pytest.approx(1 / 12)
    assert result["resources"]["logical"]["audit"]["objective_calls"] == 96


def test_no_eligible_replacement_retains_seed_in_every_search() -> None:
    """Failed generated searches remain complete deployment results via seed choice."""
    raw = bundle()
    for name, pool in raw["pools"].items():
        for candidate in pool[1:]:
            candidate.update(eligible=False, validation_auc=None)
            for split in ("train", "validation"):
                for row in candidate[split]:
                    row.update(
                        valid=False,
                        candidate_valid=False,
                        status="exception",
                        metrics=None,
                        observations=[],
                        proposal_attempts=[],
                        objective_calls=0,
                        unused_objective_allocation=2,
                        subprocess_executions=1,
                    )
        raw["selections"][name] = {
            key: pool[0][key]
            for key in ("index", "source", "source_sha256", "validation_auc")
        }
        outer, arm = name.split("/")
        raw["audit"]["per_seed"][outer][arm] = copy.deepcopy(
            raw["audit"]["per_seed"][outer]["A0"]
        )
    synchronize_cache(raw)
    result = A.summarize(raw)
    assert (
        result["arms"]["C"]["candidate_eligibility"]["no_eligible_replacement_count"]
        == 2
    )
    assert (
        result["arms"]["C"]["candidate_eligibility"]["seed_index_selection_count"] == 2
    )
    assert result["arms"]["C"]["auc"]["mean"] == pytest.approx(0.2)
    assert result["resources"]["completed_responses"] == 8


def test_file_reader_refuses_wrong_selection_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An audit cannot be attached to another freeze of the selections."""
    raw = bundle()
    write_fixture(tmp_path, raw)
    study = SimpleNamespace(
        preflight=lambda root: E.read(root / "freeze.json"),
        verify_chronology=lambda root: {"verified": True},
    )
    monkeypatch.setitem(sys.modules, "experiments.recursive_opt._shared.optimizer_discovery.exp17.study", study)
    monkeypatch.setattr(exp17, "study", study, raising=False)
    audit = E.read(tmp_path / "audit_results.json")
    audit["selections_frozen_ns"] = 1999
    (tmp_path / "audit_results.json").unlink()
    I.persist(tmp_path / "audit_results.json", audit)
    with pytest.raises(ValueError, match="barrier"):
        A.read_bundle(tmp_path)


def test_resumed_attempts_are_loaded_in_numeric_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Several bounded retry batches can reach attempt ten without reordering it."""
    raw = bundle()
    slot = raw["slots"]["17101/C/0"]
    slot["response"]["attempt"] = 10
    slot["attempts"] = [copy.deepcopy(slot["attempts"][0]) for _ in range(9)] + [
        slot["attempts"][-1]
    ]
    write_fixture(tmp_path, raw)
    study = SimpleNamespace(
        preflight=lambda root: E.read(root / "freeze.json"),
        verify_chronology=lambda root: {"verified": True},
    )
    monkeypatch.setitem(sys.modules, "experiments.recursive_opt._shared.optimizer_discovery.exp17.study", study)
    monkeypatch.setattr(exp17, "study", study, raising=False)
    read = A.read_bundle(tmp_path)
    assert read["slots"]["17101/C/0"]["attempts"][-1]["status"] == "completed"
    assert A.summarize(read)["resources"]["transport_attempts"] == 24


def test_real_evaluator_production_search_and_analysis_roundtrip(
    tmp_path: Path,
) -> None:
    """Real subprocess trajectories match the reader; only model replies are scripted."""
    from experiments.recursive_opt._shared.optimizer_discovery.exp17 import study as N
    from tests.unit_tests.test_investigation16_search_experiment import _client

    config = N.configuration(
        experiment="EXP-17",
        namespace="UNIT-EXP17-ANALYSIS-INTEGRATION",
        arms=["I", "C"],
        outer_seeds=[17991],
        slots=2,
        budget=2,
        local_replicates=1,
        task_replicates={"train": 1, "validation": 1, "audit": 1},
        workers=2,
    )
    protocol = tmp_path / "protocol.md"
    protocol.write_text(
        "UNIT: scripted model only; real deterministic evaluator and subprocesses.\n"
    )
    root = tmp_path / "run"
    N.prepare(
        root, config, protocol, extra_frozen_paths=[Path(A.__file__), Path(__file__)]
    )
    state: dict[str, Any] = {"calls": [], "invalid_at": 1}
    N.run_generation(root, client=_client(state))
    N.select_all(root)
    N.run_audit(root)
    result = A.summarize(A.read_bundle(root))
    assert len(state["calls"]) == 4
    assert result["resources"]["completed_responses"] == 4
    assert result["resources"]["transport_attempts"] == 4
    assert result["arms"]["I"]["candidate_eligibility"]["ineligible_generated"] == 1
    assert result["resources"]["physical_cache"]["train"]["subprocess_executions"] > 0
    assert result["resources"]["logical"]["train"]["allocated_objective_calls"] == 72
    assert set(result["arms"]) == {"A0", "B2", "I", "C"}
    before = {
        str(path.relative_to(root)): B.source_hash(path.read_text())
        for path in root.rglob("*.json")
    }
    N.run_generation(root, client=None)
    N.select_all(root)
    N.run_audit(root)
    assert A.summarize(A.read_bundle(root)) == result
    assert before == {
        str(path.relative_to(root)): B.source_hash(path.read_text())
        for path in root.rglob("*.json")
    }
