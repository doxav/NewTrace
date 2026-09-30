"""Read-only G1 analysis retains every slot and separates generation failure types."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import generation as G


def load_analysis() -> ModuleType:
    """Load the report script without shadowing the frozen generation.py module."""
    path = G.ROOT / "generation/analysis.py"
    spec = importlib.util.spec_from_file_location("exp16_g1_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = load_analysis()
SOURCE = "def propose(history, bounds, seed):\n    return [0.0] * len(bounds)\n"


@pytest.fixture
def run_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create twenty-four explicit unit responses with no model or generated-code execution."""
    tasks = G.fresh_tasks("G1_UNIT_ANALYSIS", "train", 1)
    contexts = {
        "I": [{"role": "user", "content": "unit"}],
        "L": [
            {"role": "user", "content": "unit"},
            {"role": "user", "content": "long unit context"},
        ],
    }
    requests = G.requests(contexts)
    monkeypatch.setattr(A, "environment", dict)
    I.persist(
        tmp_path / "freeze.json",
        {
            "requests": requests,
            "tasks": tasks,
            "files": {},
            "environment": {},
            "budget": 32,
        },
    )
    for index, request in enumerate(requests):
        directory = tmp_path / "raw" / request["slot_id"]
        valid = request["settings"]["max_tokens"] == 32000
        source = SOURCE if valid else ""
        start = (index + 1) * 10_000_000_000
        I.persist(directory / "request.json", request)
        I.persist(
            directory / "response.json",
            {
                "completed": True,
                "completed_ns": start + 1_000_000_000,
                "id": f"unit-{index}",
                "model": G.MODEL,
                "source": source,
                "source_sha256": B.source_hash(source),
                "source_status": B.source_status(source),
                "content": source,
                "parse_status": "parsed" if valid else "unparsable",
                "finish_reason": "stop" if valid else "length",
                "usage": {},
                "wall_s": 1.0,
                "attempt": 1,
            },
        )
        I.persist(
            directory / "started_1.json", {"time_ns": start, "status": "in_flight"}
        )
        I.persist(
            directory / "attempt_1.json",
            {"status": "completed", "id": f"unit-{index}", "wall_s": 1.0},
        )
        rows = []
        for task in tasks:
            value = B.objective(task, [0.0] * task["dimension"])
            rows.append(
                {
                    "valid": valid,
                    "candidate_valid": valid,
                    "status": "valid" if valid else "missing_source",
                    "source_sha256": B.source_hash(source),
                    "task_identity": B.task_identity(task),
                    "stratum": f'{task["family"]}/{task["dimension"]}',
                    "local_seed": G.local_seed("G1", request["block"], task),
                    "budget": 32,
                    "observations": (
                        [{"x": [0.0] * task["dimension"], "value": value}] * 32
                        if valid
                        else []
                    ),
                    "metrics": (
                        B.metrics([value] * 32, B.normalization(task), 32)
                        if valid
                        else None
                    ),
                    "objective_calls": 32 if valid else 0,
                    "unused_objective_allocation": 0 if valid else 32,
                    "subprocess_executions": 64 if valid else 0,
                    "execution_s": 1.0 if valid else 0.0,
                    "proposal_attempts": [
                        {"status": "valid" if valid else "missing_source"}
                    ]
                    * (32 if valid else 1),
                    "fallback_used": False,
                }
            )
        I.persist(directory / "evaluation.json", rows)
    return tmp_path


def test_all_pairs_and_missing_usage_are_retained(run_root: Path) -> None:
    """The generation contrast includes invalid slots, while unreported usage stays missing."""
    result = A.analyze(run_root)
    assert len(result["rows"]) == 24 and len(result["pairs"]) == 12
    assert result["caps"]["8000"]["eligible"] == 0
    assert result["caps"]["32000"]["eligible"] == 12
    assert result["paired"]["pooled"]["eligibility_gain_pairs"] == 12
    assert result["caps"]["8000"]["usage"]["total_tokens"]["reported_sum"] is None
    assert result["receipt_coverage"]["present"] == 0
    assert all(row["auc"] is None for row in result["rows"] if row["cap"] == 8000)
    assert result["conditional_performance"]["both_eligible_pairs"] == 0
    assert result["order"]["I"]["8000_first"] == 6
    assert result["order"]["L"]["32000_first"] == 6


def test_incomplete_run_is_not_analyzed(run_root: Path) -> None:
    """No comparison is produced while a registered response or evaluation is missing."""
    path = next((run_root / "raw").glob("*/A*/slot_*/evaluation.json"))
    path.unlink()
    with pytest.raises(RuntimeError, match="complete"):
        A.analyze(run_root)


def test_source_hash_or_request_corruption_is_rejected(run_root: Path) -> None:
    """Never reinterpret sanitized or edited source as the exact evaluated program."""
    path = next((run_root / "raw").glob("*/A*/slot_*/response.json"))
    response = json.loads(path.read_text())
    response["source"] += "# corrupted\n"
    path.write_text(json.dumps(response))
    with pytest.raises(RuntimeError, match="source"):
        A.analyze(run_root)


def test_changed_request_is_not_absorbed_into_the_analysis(run_root: Path) -> None:
    """A matched-response label cannot conceal an altered generative setting."""
    path = next((run_root / "raw").glob("*/A*/slot_*/request.json"))
    request = json.loads(path.read_text())
    request["settings"]["temperature"] = 1.0
    path.write_text(json.dumps(request))
    with pytest.raises(RuntimeError, match="request"):
        A.analyze(run_root)


def test_receipt_alias_and_partial_cost_reporting_are_explicit(run_root: Path) -> None:
    """A provider canonical model alias is recorded separately from the request model."""
    directory = next((run_root / "raw").glob("*/A*/slot_*"))
    response = json.loads((directory / "response.json").read_text())
    I.persist(
        directory / "provider_generation.json",
        {
            "id": response["id"],
            "model": "deepseek/deepseek-v4-flash-20260731",
            "provider_name": "unit",
            "total_cost": 0.125,
        },
    )
    result = A.analyze(run_root)
    assert result["receipt_coverage"]["present"] == 1
    assert result["receipt_coverage"]["missing"] == 23
    assert result["receipt_coverage"]["reported_cost_sum"] == 0.125
    assert result["receipt_coverage"]["model_identifiers"] == [
        "deepseek/deepseek-v4-flash-20260731"
    ]


def test_transport_retry_is_not_counted_as_a_new_proposal(run_root: Path) -> None:
    """A failed attempt and its completed response retain one slot and both costs in time."""
    directory = next((run_root / "raw").glob("*/A*/slot_*"))
    response = json.loads((directory / "response.json").read_text())
    start = json.loads((directory / "started_1.json").read_text())["time_ns"]
    (directory / "attempt_1.json").write_text(
        json.dumps(
            {
                "status": "transport_failure",
                "wall_s": 0.5,
                "possible_remote_completion_or_duplicate_billing": True,
            }
        )
    )
    I.persist(
        directory / "started_2.json",
        {"status": "in_flight", "time_ns": start + 3_000_000_000},
    )
    I.persist(
        directory / "attempt_2.json",
        {"status": "completed", "id": response["id"], "wall_s": 1.0},
    )
    response["attempt"] = 2
    response["completed_ns"] = start + 4_000_000_000
    (directory / "response.json").write_text(json.dumps(response))
    result = A.analyze(run_root)
    assert len(result["rows"]) == 24
    assert sum(group["transport_attempts"] for group in result["caps"].values()) == 25
    assert sum(group["transport_failures"] for group in result["caps"].values()) == 1
    assert (
        sum(row["unresolved_remote_completion_attempts"] for row in result["rows"]) == 1
    )
    assert any("suspension" in limitation for limitation in result["limitations"])


def test_lexical_protocol_rejection_is_distinct_from_external_import() -> None:
    """A local reserved name is not reported as proof of introspection or I/O."""
    lexical = A.source_diagnostics(
        "def propose(history,bounds,seed):\n return [dir for dir in [0]*len(bounds)]\n"
    )
    assert lexical["protocol_class"] == "reserved_identifier_without_forbidden_call"
    assert lexical["forbidden_calls"] == []
    external = A.source_diagnostics(
        "import numpy as np\ndef propose(history,bounds,seed):\n return np.array([0]).__dict__\n"
    )
    assert external["disallowed_imports"] == ["numpy"]
    assert external["private_attributes"] == ["__dict__"]


def test_signature_diagnostics_never_execute_candidate() -> None:
    """AST inspection explains incompatible signatures without running candidate code."""
    result = A.source_diagnostics(
        "while True: pass\ndef propose(history,bounds,seed,extra):\n return []\n"
    )
    assert result["propose_signature"]["positional"] == [
        "history",
        "bounds",
        "seed",
        "extra",
    ]
