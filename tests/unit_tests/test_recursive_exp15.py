"""EXP-15 accounting, isolation and real production Trace integration checks."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from artifacts.optimizer_discovery import benchmark as B
from artifacts.optimizer_discovery import exp15 as E


def response(source: str | None) -> Any:
    """Return a completed model response; unit tests never replace live evidence."""
    return SimpleNamespace(
        id="unit",
        model=B.MANIFEST["model"]["model"],
        usage={"completion_tokens": 10, "prompt_tokens": 20, "total_tokens": 30},
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=source), finish_reason="stop"
            )
        ],
    )


def test_slots_invalid_responses_and_resume(tmp_path: Path, monkeypatch: Any) -> None:
    """Completed empty responses consume slots and cannot be replaced on resume."""
    calls = []

    def client(**kwargs: Any) -> Any:
        """Capture an actual request shape and simulate a completed empty response."""
        calls.append(kwargs)
        return response(None)

    exp = E.Experiment(tmp_path, "pilot", client=client)
    first = exp.proposal(701, "A1", 0, B.SEED_SOURCE)
    assert first["source"] == "" and first["completed"]
    assert exp.proposal(701, "A1", 0, B.SEED_SOURCE) == first and len(calls) == 1
    (tmp_path / "701/A1/slot_00/attempt_1.json").unlink()
    exp.proposal(701, "A1", 0, B.SEED_SOURCE)
    assert (tmp_path / "701/A1/slot_00/attempt_1.json").exists()
    assert len(calls) == 1
    assert calls[0]["extra_body"] == {"reasoning": {"effort": "low"}}
    assert calls[0]["num_retries"] == 0
    with pytest.raises(ValueError, match="slot"):
        exp.proposal(701, "A1", 2, B.SEED_SOURCE)


def test_retry_and_uncertain_inflight_are_distinct(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Transient attempts are bounded; an interrupted remote call is never blindly replayed."""
    monkeypatch.setattr(E.time, "sleep", lambda delay: None)
    calls = []

    def client(**kwargs: Any) -> Any:
        """Simulate transient failures with no hidden retries."""
        calls.append(1)
        raise RuntimeError("connection reset")

    exp = E.Experiment(tmp_path, "pilot", client=client)
    with pytest.raises(RuntimeError, match="transport"):
        exp.proposal(701, "A1", 0, B.SEED_SOURCE)
    assert len(calls) == 4
    slot = tmp_path / "701/A1/slot_01"
    slot.mkdir(parents=True)
    (slot / "attempt_1.json").write_text(json.dumps({"status": "in_flight"}))
    with pytest.raises(RuntimeError, match="uncertain"):
        exp.proposal(701, "A1", 1, B.SEED_SOURCE)
    assert len(calls) == 4


def test_production_trace_budgets_isolation_and_selection(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Both arms allocate identical tasks; production A2 performs exactly two real updates."""
    monkeypatch.setitem(B.MANIFEST, "inner_budget", 2)
    calls = []

    def client(**kwargs: Any) -> Any:
        """Return one valid source and one invalid source per arm."""
        calls.append(kwargs)
        return response(B.SEED_SOURCE if len(calls) % 2 else "def :")

    exp = E.Experiment(tmp_path, "pilot", client=client)
    with pytest.raises(RuntimeError, match="selection"):
        exp.holdout()
    exp.generate(701, "A1")
    exp.generate(701, "A2")
    assert len(calls) == 4
    trace = json.loads((tmp_path / "701/A2/trace.json").read_text())
    assert (
        trace["result"]["portable"]
        and trace["result"]["level_results"][0]["metadata"]["trace_optimize_path"]
    )
    assert not trace["plan"]["runtime"].get("test_mode", False)
    a1 = [
        json.loads(p.read_text())
        for p in sorted((tmp_path / "701/A1").glob("slot_*/request.json"))
    ]
    a2 = [
        json.loads(p.read_text())
        for p in sorted((tmp_path / "701/A2").glob("slot_*/request.json"))
    ]
    assert a1[0]["messages"] == a1[1]["messages"]
    assert len(a2[0]["messages"]) == 2
    assert "TRAINING FEEDBACK" in a2[0]["messages"][1]["content"]
    assert all("normalization" not in p["messages"][-1]["content"] for p in a2)
    assert not list((tmp_path / "cache").glob("*holdout*"))
    exp.select(701)
    pools = [
        json.loads((tmp_path / "701" / arm / "pool.json").read_text())
        for arm in ("A1", "A2")
    ]
    panels = [
        [(r["task_identity"], r["local_seed"]) for r in pool[0]["validation"]]
        for pool in pools
    ]
    assert panels[0] == panels[1]
    assert (
        json.loads((tmp_path / "701/selection.json").read_text())["A1"]["index"] == -1
    )
    exp.seeds = [701, 702]
    with pytest.raises(RuntimeError, match="every selection"):
        exp.freeze_selections()
    exp.seeds = [701]
    exp.freeze_selections()
    exp.holdout()
    before = len(calls)
    exp.generate(701, "A2")
    assert len(calls) == before
    from artifacts.optimizer_discovery import evidence

    assert evidence.verify(exp)["completed_responses"] == 4
    result = exp.analyze()
    assert len(result["per_seed"]) == 1
    assert result["contrasts"]["A2-A1"]["mean"] == 0
    assert exp.audit()["proposal_slots"] == 4
    assert exp.audit()["ineligible_generated_candidates"] == 2
    exp.seeds = [701, 702]
    with pytest.raises(RuntimeError, match="missing outer seeds"):
        exp.analyze()
    exp.seeds = [701]
    (tmp_path / "701/selection.json").write_text("{}")
    with pytest.raises(RuntimeError, match="modified"):
        exp.freeze_selections()


def test_freeze_and_hash_mismatch_fail_closed(tmp_path: Path) -> None:
    """An unregistered confirmation configuration cannot reach the provider."""
    with pytest.raises(RuntimeError, match="freeze"):
        E.preflight(tmp_path / "absent.json")
    E.persist(tmp_path / "evidence.json", {"x": 1})
    with pytest.raises(RuntimeError, match="overwrite"):
        E.persist(tmp_path / "evidence.json", {"x": 2})


def test_paired_analysis_preserves_negative_values_and_all_seeds() -> None:
    """Uncertainty resamples paired outer seeds and does not omit unfavorable deltas."""
    result = E.paired([1, 2, 3, 4, 5])
    assert result["mean"] == 3 and result["interpretation"] == "negative signal"
    assert E.paired([0, 0, 0, 0, 0])["interpretation"] == "no detectable difference"
    with pytest.raises(ValueError):
        E.paired([])


def test_environment_freeze_checks_versions(tmp_path: Path, monkeypatch: Any) -> None:
    """Configuration hashes alone cannot authorize a changed execution environment."""
    monkeypatch.setitem(B.MANIFEST, "status", "FROZEN_CONFIRMATORY")
    target = tmp_path / "freeze.json"
    E.persist(target, {"files": {}, "environment": {"python": "impossible"}})
    with pytest.raises(RuntimeError, match="environment"):
        E.preflight(target)


def test_descriptive_audit_preserves_invalidity_and_missing_usage() -> None:
    """Missing usage stays missing; source failure is distinct from trajectory quality."""
    from artifacts.optimizer_discovery import evidence

    proposals = [
        {
            "source": B.SEED_SOURCE,
            "source_status": "valid",
            "usage": {"total_tokens": 9},
        },
        {"source": "def :", "source_status": "syntax_error", "usage": {}},
    ]
    trajectories = [
        {
            "valid": False,
            "candidate_valid": False,
            "fallback_used": False,
            "execution_s": 0.2,
            "proposal_attempts": [{"status": "timeout"}],
            "metrics": None,
        },
    ]
    result = evidence.describe(proposals, trajectories)
    assert result["source_invalid_fraction"] == 0.5
    assert result["usage"]["total_tokens"] == {
        "reported_sum": 9,
        "reported_responses": 1,
        "missing_responses": 1,
    }
    assert result["usage"]["cost_usd"]["reported_sum"] is None
    assert result["proposal_status_counts"] == {"timeout": 1}
    assert result["trajectory_invalid_fraction"] == 1
    assert result["source_complexity"][1]["ast_nodes"] is None


def test_provider_metadata_filter_does_not_persist_unknown_fields() -> None:
    """Only declared safe provider metadata may enter scientific artifacts."""
    from artifacts.optimizer_discovery import evidence

    assert evidence.safe_metadata(
        {"id": "x", "provider_name": "p", "secret": "private"}
    ) == {"id": "x", "provider_name": "p"}


def test_large_evidence_packaging_preserves_exact_json(tmp_path: Path) -> None:
    """Large scientific records compress losslessly and remain readable on resume."""
    value = {"verbatim": "a" * 600000}
    path = tmp_path / "large.json"
    E.persist(path, value)
    assert not path.exists() and path.with_suffix(".json.gz").exists()
    assert E.read(path) == value and E.exists(path)
    E.persist(path, value)
    with pytest.raises(RuntimeError, match="overwrite"):
        E.persist(path, {"different": True})


def test_production_a2_resumes_mid_search_without_replacing_completed_response(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Replay production search state after interruption, without a second completed slot zero."""
    monkeypatch.setitem(B.MANIFEST, "inner_budget", 2)
    completed_source = (
        "def propose(history, bounds, seed):\n    return [(a+b)/2 for a,b in bounds]\n"
    )
    calls = []

    def client(**kwargs: Any) -> Any:
        """Interrupt the second scientific slot once, then complete it on explicit resume."""
        calls.append(kwargs["seed"])
        if len(calls) == 2:
            raise RuntimeError("unit nontransient interruption")
        return response(completed_source)

    first = E.Experiment(tmp_path, "pilot", client=client)
    with pytest.raises(RuntimeError, match="proposal count"):
        first.generate(701, "A2")
    retained = (tmp_path / "701/A2/slot_00/response.json").read_bytes()
    resumed = E.Experiment(tmp_path, "pilot", client=client)
    resumed.generate(701, "A2")
    assert (tmp_path / "701/A2/slot_00/response.json").read_bytes() == retained
    assert calls == [
        B.stable_seed("request", "pilot", 701, index) for index in [0, 1, 1]
    ]
    assert len(list(tmp_path.glob("701/A2/slot_*/response.json"))) == 2
    assert len(list(tmp_path.glob("701/A2/slot_*/attempt_*.json"))) == 3
    assert E.read(tmp_path / "701/A2/generation_complete.json")["slots"] == 2


def test_preflight_rejects_modified_frozen_source(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Changing a frozen byte must fail before a confirmatory request can execute."""
    monkeypatch.setitem(B.MANIFEST, "status", "FROZEN_CONFIRMATORY")
    source = tmp_path / "source.py"
    source.write_text("original\n")
    frozen = tmp_path / "freeze.json"
    E.persist(
        frozen,
        {
            "files": {str(source): B.source_hash(source.read_text())},
            "environment": E.environment(),
        },
    )
    E.preflight(frozen)
    source.write_text("changed\n")
    with pytest.raises(RuntimeError, match="freeze mismatch"):
        E.preflight(frozen)


def test_reported_latency_matches_the_completed_attempt(tmp_path: Path) -> None:
    """A resumed slot's total wall duration must not be attributed to its final attempt."""
    from artifacts.optimizer_discovery.reporting import attempt_timing

    E.persist(tmp_path / "started_1.json", {"time_ns": 100_000_000_000})
    E.persist(
        tmp_path / "attempt_1.json", {"status": "transport_failure", "wall_s": 10.0}
    )
    E.persist(tmp_path / "started_2.json", {"time_ns": 1_000_000_000_000})
    E.persist(tmp_path / "attempt_2.json", {"status": "completed", "wall_s": 50.0})
    E.persist(
        tmp_path / "response.json",
        {"attempt": 2, "completed_ns": 1_050_000_000_000, "wall_s": 50.0},
    )
    result = attempt_timing(tmp_path)
    assert result["successful_attempt_wall_s"] == 50
    assert result["slot_wall_s"] == 950
    assert result["measured_attempts_s"] == 60
    assert result["unattributed_wall_gap_s"] == 890
    assert result["transport_attempts"] == 2 and result["transport_failures"] == 1


def test_selected_source_export_preserves_bytes_and_frozen_hash(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Portable export preserves even raw formatting and refuses altered selections."""
    import gzip

    from artifacts.optimizer_discovery.reporting import export_programs

    root = tmp_path / "raw"
    source = B.SEED_SOURCE + "\n  \n"
    candidate = {
        "source": source,
        "source_sha256": B.source_hash(source),
        "index": 0,
        "validation_auc": 0.1,
    }
    selection = {"A2": candidate}
    E.persist(root / "11/selection.json", selection)
    E.persist(
        root / "selections_frozen.json",
        {
            "selection_hashes": {"11": B.digest(selection)},
            "representative_outer_seed": 11,
            "representative": candidate,
        },
    )
    E.persist(
        root / "11/A2/pool.json",
        [
            {
                **candidate,
                "eligible": True,
                "train": [
                    {"valid": True, "metrics": {"auc": 0.1}, "stratum": "sphere/2"}
                ],
            }
        ],
    )
    E.persist(
        root / "11/A2/slot_00/request.json",
        {"slot": 0, "parent_sha256": B.MANIFEST["seed_sha256"]},
    )
    response_record = {"source": source, "source_sha256": B.source_hash(source)}
    E.persist(root / "11/A2/slot_00/response.json", response_record)
    result = export_programs(root)
    exported = Path(result["programs"]["11"]["gzip_path"])
    assert gzip.decompress(exported.read_bytes()).decode() == source
    assert result["programs"]["11"]["source_sha256"] == B.source_hash(source)
    assert result["representative_outer_seed"] == 11
    assert export_programs(root) == result
    with monkeypatch.context() as relative_context:
        relative_context.chdir(tmp_path)
        assert export_programs(Path("raw")) == result
    lineage = E.read(Path(result["programs"]["11"]["lineage_path"]))
    assert lineage[0]["selected"] and lineage[0]["source_sha256"] == B.source_hash(
        source
    )
    assert lineage[0]["parent_sha256"] == B.MANIFEST["seed_sha256"]
    assert "+  \n" in lineage[0]["diff"]
    response_path = root / "11/A2/slot_00/response.json"
    original_response = response_path.read_bytes()
    response_path.write_text(json.dumps({**response_record, "source": "changed"}))
    with pytest.raises(RuntimeError, match="lineage"):
        export_programs(root)
    response_path.write_bytes(original_response)
    (root / "11/selection.json").write_text("{}")
    with pytest.raises(RuntimeError, match="selection"):
        export_programs(root)


def test_trace_archive_is_lossless_and_requires_completed_generation(
    tmp_path: Path,
) -> None:
    """Post-run compression preserves every byte and never removes an active trace."""
    import gzip
    import hashlib
    import lzma

    from artifacts.optimizer_discovery.reporting import archive_trace, read_trace

    original = b'{"source": "raw  \\n", "all_data": [1, 2, 3]}\n'
    compressed = gzip.compress(original, mtime=0)
    (tmp_path / "trace.json.gz").write_bytes(compressed)
    with pytest.raises(RuntimeError, match="completed generation"):
        archive_trace(tmp_path)
    E.persist(tmp_path / "generation_complete.json", {"slots": 8})
    record = archive_trace(tmp_path)
    assert lzma.decompress((tmp_path / "trace.json.xz").read_bytes()) == original
    assert not (tmp_path / "trace.json.gz").exists()
    assert read_trace(tmp_path) == json.loads(original)
    assert record["original_gzip_sha256"] == hashlib.sha256(compressed).hexdigest()
    assert archive_trace(tmp_path) == record
    (tmp_path / "trace.json.xz").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="integrity"):
        read_trace(tmp_path)


def test_completed_event_archive_replays_exact_log_and_cleans_up(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Oversized event evidence stays lossless and reconstructs for unchanged analysis."""
    from artifacts.optimizer_discovery.reporting import analyze_archived, archive_events

    payload = b'{"event":"evaluation","value":-1}\n' * 20000
    path = tmp_path / "events.jsonl"
    path.write_bytes(payload)
    with pytest.raises(RuntimeError, match="completed"):
        archive_events(tmp_path)
    expected = {"all_rows_preserved": True}
    E.persist(tmp_path / "results.json", expected)
    record = archive_events(tmp_path)
    assert not path.exists() and record["jsonl_bytes"] == len(payload)
    assert archive_events(tmp_path) == record

    class Reader:
        """Stand in only for the scientific reader while checking exact log bytes."""

        def analyze(self) -> dict[str, bool]:
            """Read the original stream, including every negative row."""
            assert path.read_bytes() == payload
            return expected

    monkeypatch.setattr(E, "Experiment", lambda root, phase: Reader())
    assert analyze_archived(tmp_path) == expected
    assert not path.exists()
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="differs"):
        analyze_archived(tmp_path)
    assert path.read_bytes() == b"changed"
    path.unlink()
    (tmp_path / "events.jsonl.gz").write_bytes(b"damaged")
    with pytest.raises(RuntimeError, match="integrity"):
        analyze_archived(tmp_path)
