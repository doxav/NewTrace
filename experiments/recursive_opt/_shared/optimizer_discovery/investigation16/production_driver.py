"""Frozen CLI for the P1 engineering check and prospective production diagnostic."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from experiments.recursive_opt._shared.optimizer_discovery import benchmark as B
from experiments.recursive_opt._shared.optimizer_discovery import evidence
from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import evidence_io as I
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import feedback_experiment as F
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import generation as G
from experiments.recursive_opt._shared.optimizer_discovery.investigation16 import search_experiment as S
from experiments.recursive_opt._shared.optimizer_discovery.phase0 import _load_key
from opto.features.recursive_opt.runmode import make_live_llm

ENGINEERING_ROOT = G.ROOT / "production_engineering"
ROOT = G.ROOT / "production_run"
PROTOCOL_ROOT = G.ROOT / "production"


def selected_cap() -> int:
    """Reuse the completed G1 reliability decision, without inspecting efficacy outcomes."""
    return F.choose_cap(F._g1_result()["caps"])


def study_config(*, engineering: bool) -> dict[str, Any]:
    """Construct the exact declared stage allocations with disjoint scientific namespaces."""
    config = S.configuration(
        max_tokens=selected_cap(),
        namespace="P1-E1" if engineering else "P1",
        outer_seeds=[16601] if engineering else None,
        slots=2 if engineering else 8,
        budget=32,
        task_replicates={"train": 4, "validation": 2, "audit": 2},
        local_replicates=2,
        workers=8,
        include_aggregate_auc=True,
        max_prompt_chars=524288,
    )
    config["run_kind"] = "engineering_R_only" if engineering else "prospective_P1"
    return config


def response_paths(raw_root: Path) -> list[Path]:
    """Discover exact response artifacts for every arm, including compressed records."""
    paths = sorted(raw_root.glob("*/*/slot_*/response.json*"))
    result = [
        path for path in paths if path.name in ("response.json", "response.json.gz")
    ]
    if len({path.parent for path in result}) != len(result):
        raise RuntimeError("duplicate compressed and ordinary response artifacts")
    return result


def read_record(path: Path) -> Any:
    """Read a discovered physical JSON/gzip artifact through the existing logical reader."""
    return E.read(path.with_suffix("") if path.suffix == ".gz" else path)


def collect_receipts(raw_root: Path) -> None:
    """Reuse the provider metadata whitelist for P1's four non-A-prefixed arm paths."""
    key_loaded = False
    for path in response_paths(raw_root):
        target = path.parent / "provider_generation.json"
        result = read_record(path)
        identifier = result.get("id")
        if (
            result.get("completed") is not True
            or not isinstance(identifier, str)
            or not identifier
        ):
            raise RuntimeError("completed response lacks its provider identifier")
        if E.exists(target):
            if E.read(target).get("id") != identifier:
                raise RuntimeError(
                    "provider receipt identity differs from its response"
                )
            continue
        if not key_loaded:
            _load_key()
            key_loaded = True
        url = "https://openrouter.ai/api/v1/generation?" + urllib.parse.urlencode(
            {"id": identifier}
        )
        request = urllib.request.Request(
            url, headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]}
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                value = evidence.safe_metadata(json.load(response)["data"])
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            directory = path.parent / "metadata_attempts"
            attempt = len(list(directory.glob("*.json"))) + 1
            I.persist(
                directory / f"{attempt}.json",
                {
                    "id": identifier,
                    "error_type": type(error).__name__,
                    "http_status": getattr(error, "code", None),
                },
            )
        else:
            if value.get("id") != identifier:
                raise RuntimeError(
                    "provider receipt identity differs from its response"
                )
            I.persist(target, value)


def live_client() -> Any:
    """Load the credential privately and use the existing frozen compatibility path."""
    _load_key()
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        return make_live_llm(
            "openrouter/" + G.MODEL,
            cache=False,
            max_retries=1,
            request_timeout_s=300,
            allow_env_overrides=False,
            empty_response_retries=0,
            budget_resource=None,
        )


def engineering_evidence(frozen: dict[str, Any]) -> dict[str, Any]:
    """Verify complete R-only training evidence and capture immutable semantic records."""
    if frozen["config"] != study_config(engineering=True):
        raise RuntimeError(
            "engineering configuration differs from its registered scope"
        )
    paths = response_paths(ENGINEERING_ROOT / "raw")
    expected = {ENGINEERING_ROOT / f"raw/16601/R/slot_{slot:02d}" for slot in range(2)}
    if {path.parent for path in paths} != expected:
        raise RuntimeError(
            "engineering requires exactly the two registered R responses"
        )
    if any(
        E.exists(ENGINEERING_ROOT / name)
        for name in (
            "generation_frozen.json",
            "selections_frozen.json",
            "audit_results.json",
        )
    ):
        raise RuntimeError("engineering crossed its training-only boundary")
    caches = [
        path
        for path in (ENGINEERING_ROOT / "cache").glob("*.json*")
        if path.name.endswith((".json", ".json.gz"))
    ]
    if not caches or any(
        read_record(path)["key"]["split"] != "train" for path in caches
    ):
        raise RuntimeError(
            "engineering cache does not contain exclusively training evidence"
        )
    directory = ENGINEERING_ROOT / "raw/16601/R"
    completion = E.read(directory / "generation_complete.json")
    S.verify_arm_responses(
        ENGINEERING_ROOT, 16601, "R", completed_before_ns=completion["completed_ns"]
    )
    # Reuse production's saved-Trace schedule/callback verification without a client.
    S.T.generate_recursive(S.SearchExperiment(ENGINEERING_ROOT, "R"), 16601)
    allocation = E.read(directory / "allocations_train.json")
    trajectories = frozen["panels"]["train"]["trajectories"]
    pool_hashes = [
        frozen["seed_sha256"],
        *(read_record(path)["source_sha256"] for path in paths),
    ]
    if allocation["candidate_slots"] != 3 or len(allocation["rows"]) != 3:
        raise RuntimeError("engineering training allocation omits a pool entry")
    for index, row in enumerate(allocation["rows"]):
        if any(
            row.get(key) != value
            for key, value in {
                "index": index - 1,
                "source_sha256": pool_hashes[index],
                "trajectories": trajectories,
                "allocated_objective_calls": trajectories * frozen["config"]["budget"],
            }.items()
        ):
            raise RuntimeError("engineering training allocation identity differs")
    eligible = [row["valid_trajectories"] == trajectories for row in allocation["rows"]]
    if not eligible[0]:
        raise RuntimeError("trusted seed failed engineering training evaluation")
    records = [
        path
        for path in directory.rglob("*.json*")
        if path.name.endswith((".json", ".json.gz"))
        and path.name != "provider_generation.json"
        and "metadata_attempts" not in path.parts
    ] + caches
    return {
        "response_hashes": {
            str(path.relative_to(ENGINEERING_ROOT)): B.digest(read_record(path))
            for path in paths
        },
        "evidence_hashes": {
            str(path.relative_to(ENGINEERING_ROOT)): B.digest(read_record(path))
            for path in sorted(records)
        },
        "eligible_generated": sum(eligible[1:]),
    }


def require_engineering() -> dict[str, Any]:
    """Require complete, unchanged, passing engineering evidence before the efficacy stage."""
    path = ENGINEERING_ROOT / "engineering_results.json"
    if not E.exists(path):
        raise RuntimeError("P1 requires a completed passing engineering check")
    result = E.read(path)
    if any(
        result.get(key) != value
        for key, value in {
            "passed": True,
            "stage": "P1-E1",
            "completed_responses": 2,
            "resume_without_client_passed": True,
        }.items()
    ):
        raise RuntimeError("P1 requires a completed passing engineering check")
    frozen = S.preflight(ENGINEERING_ROOT)
    verified = engineering_evidence(frozen)
    if (
        result.get("freeze_sha256") != B.digest(frozen)
        or any(result.get(key) != value for key, value in verified.items())
        or verified["eligible_generated"] < 1
    ):
        raise RuntimeError("engineering completed evidence or provenance changed")
    return result


def prepare(*, engineering: bool) -> dict[str, Any]:
    """Freeze all stage code and registered analysis before any scientific generation."""
    if not engineering:
        require_engineering()
    root = ENGINEERING_ROOT if engineering else ROOT
    protocol = PROTOCOL_ROOT / (
        "PROTOCOL_P1_E1.md" if engineering else "PROTOCOL_P1.md"
    )
    analysis = PROTOCOL_ROOT / "analysis.py"
    if not analysis.is_file():
        raise RuntimeError("analysis implementation must exist before the stage freeze")
    return S.prepare(
        root,
        study_config(engineering=engineering),
        protocol,
        extra_frozen_paths=[
            Path(__file__),
            analysis,
            PROTOCOL_ROOT / "analysis_protocol.md",
            PROTOCOL_ROOT / "test_analysis.py",
        ],
    )


def run_engineering(*, client: Any = None) -> dict[str, Any]:
    """Run exactly two R callbacks, then prove completed-response resume makes no calls."""
    frozen = S.preflight(ENGINEERING_ROOT)
    if frozen["config"] != study_config(engineering=True):
        raise RuntimeError(
            "engineering configuration differs from its registered scope"
        )
    target = ENGINEERING_ROOT / "engineering_results.json"
    if E.exists(target):
        collect_receipts(ENGINEERING_ROOT / "raw")
        return require_engineering()
    owner = S.SearchExperiment(
        ENGINEERING_ROOT, "R", client=client if client is not None else live_client()
    )
    owner.generate(16601)
    before_replay = engineering_evidence(frozen)
    paths = response_paths(ENGINEERING_ROOT / "raw")
    owner.client = None
    owner.generate(16601)
    verified = engineering_evidence(frozen)
    if before_replay != verified:
        raise RuntimeError("engineering resume altered completed evidence")
    result = {
        "stage": "P1-E1",
        "freeze_sha256": B.digest(frozen),
        "passed": verified["eligible_generated"] > 0,
        "completed_responses": 2,
        **verified,
        "resume_without_client_passed": True,
        "prompt_characters": [
            sum(
                len(message["content"])
                for message in E.read(path.parent / "request.json")["messages"]
            )
            for path in paths
        ],
        "usage": [read_record(path)["usage"] for path in paths],
    }
    I.persist(target, result)
    collect_receipts(ENGINEERING_ROOT / "raw")
    return result


def main() -> None:
    """Execute only the requested registered stage; keep generation and audit separable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare_engineering",
            "engineering",
            "prepare",
            "generate",
            "select",
            "audit",
            "receipts",
        ),
    )
    action = parser.parse_args().action
    if action.startswith("prepare"):
        value = prepare(engineering=action == "prepare_engineering")
        print(
            json.dumps({"freeze_sha256": B.digest(value), "config": value["config"]}),
            flush=True,
        )
    elif action == "engineering":
        print(json.dumps(run_engineering(), indent=2), flush=True)
    elif action == "generate":
        require_engineering()
        S.run_generation(ROOT, client=live_client())
        collect_receipts(ROOT / "raw")
    elif action == "select":
        S.select_all(ROOT)
    elif action == "audit":
        S.run_audit(ROOT)
        print(json.dumps(S.verify_chronology(ROOT)), flush=True)
    else:
        collect_receipts(ROOT / "raw")


if __name__ == "__main__":
    main()
