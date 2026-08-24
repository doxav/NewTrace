"""Separate control-plane and Experiment 0 v2 provenance locking."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from opto.features.recursive_opt import spec as control_plane

from .components import MODULE_REF
from .datasets import DATASET_REFS, DATASET_REFS_V2
from .evaluator import EVALUATOR_REF
from .preflight import _canonical_json, _load_json
from .registration import register_experiment_components
from .specs import build_spec


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def experiment_source_provenance() -> dict[str, Any]:
    """Hash versioned experiment code and preregistered v2 inputs."""
    paths = sorted(
        [path for path in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in path.parts]
        + [
            PACKAGE_ROOT / "manifests/preregistration_v2.json",
            PACKAGE_ROOT / "manifests/preregistration_frozen.json",
            PACKAGE_ROOT / "manifests/dataset_manifest_v2.json",
            PACKAGE_ROOT / "preflight_skips.json",
        ]
    )
    records = [
        {
            "path": str(path.relative_to(REPOSITORY_ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    ]
    return {
        "sha256": hashlib.sha256(_canonical_json(records).encode("utf-8")).hexdigest(),
        "files": records,
    }


def experiment_registry_provenance() -> dict[str, Any]:
    """Hash all versioned Experiment 0 registry entries independently of engines."""
    register_experiment_components()
    entries = {
        "modules": [
            control_plane._registry_record(
                "module", MODULE_REF, control_plane._MODULE_REGISTRY[MODULE_REF]
            )
        ],
        "evaluators": [
            control_plane._registry_record(
                "evaluator",
                EVALUATOR_REF,
                control_plane._EVALUATOR_REGISTRY[EVALUATOR_REF],
            )
        ],
        "datasets": [
            control_plane._registry_record(
                "dataset", ref, control_plane._DATASET_REGISTRY[ref]
            )
            for ref in sorted({*DATASET_REFS.values(), *DATASET_REFS_V2.values()})
        ],
    }
    return {
        "sha256": hashlib.sha256(_canonical_json(entries).encode("utf-8")).hexdigest(),
        "entries": entries,
    }


def build_control_plane_lock_after_gepa_reflection_fix() -> dict[str, Any]:
    """Lock Experiment 0 to the CI-verified GEPA reflection implementation."""
    register_experiment_components()
    old_lock_path = PACKAGE_ROOT / "control_plane_lock_v2.json"
    old_lock = _load_json(old_lock_path)
    readiness = _load_json(
        REPOSITORY_ROOT / "artifacts/control_plane_v2/prompt18_readiness.json"
    )
    golden = _load_json(
        REPOSITORY_ROOT
        / "artifacts/control_plane_v2/golden_specs/uc4_positive.normalized.json"
    )
    control_provenance = control_plane.compile_plan(golden).code_provenance
    expected_runtime = readiness["verified_runtime_tree_sha256"]
    expected_registry = readiness["verified_registry_sha256"]
    if control_provenance["runtime_tree_sha256"] != expected_runtime:
        raise RuntimeError("readiness runtime digest diverges from production")
    if control_provenance["registry_sha256"] != expected_registry:
        raise RuntimeError("readiness registry digest diverges from production")
    required_ci = readiness.get("required_ci_run")
    if not readiness.get("ready_for_prompt_18") or not isinstance(required_ci, dict):
        raise RuntimeError("GEPA reflection control plane is not ready")
    if required_ci.get("status") != "completed" or required_ci.get("conclusion") != "success":
        raise RuntimeError("required GEPA reflection CI is not green")
    changed_control_files = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            "opto/features/recursive_opt",
            "opto/features/graph",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if changed_control_files:
        raise RuntimeError("locked control-plane files have local modifications")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != required_ci["head_sha"]:
        raise RuntimeError("current HEAD does not match the CI-verified implementation")
    eligibility_path = PACKAGE_ROOT / "reports/task_eligibility_v2.json"
    selected_task = None
    if eligibility_path.exists():
        selected_task = _load_json(eligibility_path).get("selected_task")
    plan_task = selected_task if isinstance(selected_task, str) else "gsm8k"
    plan_registry = {
        engine: control_plane.compile_plan(
            build_spec(
                task=plan_task,
                engine=engine,
                seed=0,
                output_directory=None,
            )
        ).code_provenance["registry_sha256"]
        for engine in ("fixed", "trace", "gepa_optimize_anything")
    }
    return {
        "schema_version": "recursive-opt-experiment-lock/v3",
        "experiment_version": "experiment-0-v2",
        "git_head": head,
        "branch": old_lock["branch"],
        "control_plane": {
            "runtime_tree_sha256": expected_runtime,
            "registry_sha256": expected_registry,
            "runtime_files_locked": True,
            "local_modifications": False,
        },
        "experiment": {
            "source": experiment_source_provenance(),
            "registry": experiment_registry_provenance(),
            "plan_registry_sha256_by_engine": plan_registry,
            "selected_task": selected_task,
        },
        "workflow": {
            "name": "recursive-opt v2 contracts",
            "run_id": required_ci["id"],
            "job": required_ci["job"],
            "job_id": required_ci["job_id"],
            "head_sha": required_ci["head_sha"],
            "status": required_ci["status"],
            "conclusion": required_ci["conclusion"],
            "url": required_ci["url"],
        },
        "gepa_public_evaluator_contract": old_lock["gepa_public_evaluator_contract"],
        "gepa_reflection_protocol": (
            "text prompt -> guarded optimizer chat messages -> textual response"
        ),
        "supersedes": {
            "path": str(old_lock_path.relative_to(PACKAGE_ROOT)),
            "git_head": old_lock["git_head"],
            "runtime_tree_sha256": old_lock["control_plane"]["runtime_tree_sha256"],
            "registry_sha256": old_lock["control_plane"]["registry_sha256"],
            "reason": "GEPA 0.1.4 reflection protocol hotfix",
        },
        "environment": {
            "python": platform.python_version(),
            "os": platform.platform(),
            "dependencies": {
                name: _package_version(name)
                for name in (
                    "gepa",
                    "ipykernel",
                    "langgraph",
                    "litellm",
                    "nbclient",
                    "nbformat",
                    "numpy",
                    "pytest",
                    "pytest-socket",
                    "trace-opt",
                )
            },
        },
    }


def build_experiment_protocol_lock_after_proposal_gate_fix() -> dict[str, Any]:
    """Version the proposal-gate correction against the frozen control plane."""
    previous_path = PACKAGE_ROOT / "control_plane_lock_after_gepa_reflection_fix.json"
    previous = _load_json(previous_path)
    golden = _load_json(
        REPOSITORY_ROOT
        / "artifacts/control_plane_v2/golden_specs/uc4_positive.normalized.json"
    )
    control_provenance = control_plane.compile_plan(golden).code_provenance
    expected_control = previous["control_plane"]
    if (
        control_provenance["runtime_tree_sha256"]
        != expected_control["runtime_tree_sha256"]
    ):
        raise RuntimeError("frozen control-plane runtime digest changed")
    if control_provenance["registry_sha256"] != expected_control["registry_sha256"]:
        raise RuntimeError("frozen control-plane registry digest changed")
    changed_control_files = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            "opto/features/recursive_opt",
            "opto/features/graph",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if changed_control_files:
        raise RuntimeError("frozen control-plane files have local modifications")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "schema_version": "recursive-opt-experiment-lock/v4",
        "experiment_version": previous["experiment_version"],
        "git_head": head,
        "branch": previous["branch"],
        "control_plane": previous["control_plane"],
        "experiment": {
            **previous["experiment"],
            "source": experiment_source_provenance(),
            "registry": experiment_registry_provenance(),
        },
        "workflow": previous["workflow"],
        "environment": previous["environment"],
        "gepa_public_evaluator_contract": previous[
            "gepa_public_evaluator_contract"
        ],
        "gepa_reflection_protocol": previous["gepa_reflection_protocol"],
        "protocol_amendment": {
            "name": "proposal-gate-semantics-correction",
            "proposal_gate": "optimizer usage plus proposed and evaluated candidates",
            "selection_diagnostic": "selected artifact differs from P0",
            "scientific_success_criteria_changed": False,
        },
        "supersedes": {
            "path": str(previous_path.relative_to(PACKAGE_ROOT)),
            "git_head": previous["git_head"],
            "experiment_source_sha256": previous["experiment"]["source"]["sha256"],
            "reason": "separate proposal execution from selected-artifact change",
        },
    }


def build_control_plane_lock_after_empty_text_retry() -> dict[str, Any]:
    """Lock the metered empty-text retry without changing Experiment 0 science."""
    register_experiment_components()
    previous_path = PACKAGE_ROOT / "experiment_protocol_lock_after_proposal_gate_fix.json"
    previous = _load_json(previous_path)
    readiness = _load_json(
        REPOSITORY_ROOT / "artifacts/control_plane_v2/prompt18_readiness.json"
    )
    golden = _load_json(
        REPOSITORY_ROOT
        / "artifacts/control_plane_v2/golden_specs/uc4_positive.normalized.json"
    )
    control_provenance = control_plane.compile_plan(golden).code_provenance
    if (
        control_provenance["runtime_tree_sha256"]
        != readiness["verified_runtime_tree_sha256"]
    ):
        raise RuntimeError("readiness runtime digest diverges from production")
    if control_provenance["registry_sha256"] != readiness["verified_registry_sha256"]:
        raise RuntimeError("readiness registry digest diverges from production")
    changed_control_files = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            "opto/features/recursive_opt",
            "opto/optimizers/optoprime_v2.py",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    required_ci = readiness.get("required_ci_run")
    ready = bool(readiness.get("ready_for_prompt_18"))
    if ready:
        if not isinstance(required_ci, dict):
            raise RuntimeError("ready control plane lacks required CI evidence")
        if required_ci.get("status") != "completed" or required_ci.get("conclusion") != "success":
            raise RuntimeError("required empty-text retry CI is not green")
        if required_ci.get("head_sha") != head:
            raise RuntimeError("current HEAD does not match required CI evidence")
        if changed_control_files:
            raise RuntimeError("CI-verified control-plane files have local modifications")
    selected_task = previous["experiment"]["selected_task"]
    plan_registry = {
        engine: control_plane.compile_plan(
            build_spec(
                task=selected_task,
                engine=engine,
                seed=0,
                output_directory=None,
            )
        ).code_provenance["registry_sha256"]
        for engine in ("fixed", "trace", "gepa_optimize_anything")
    }
    workflow = None
    if isinstance(required_ci, dict):
        workflow = {
            "name": "recursive-opt v2 contracts",
            "run_id": required_ci["id"],
            "job": required_ci["job"],
            "job_id": required_ci["job_id"],
            "head_sha": required_ci["head_sha"],
            "status": required_ci["status"],
            "conclusion": required_ci["conclusion"],
            "url": required_ci["url"],
        }
    return {
        "schema_version": "recursive-opt-experiment-lock/v5",
        "experiment_version": previous["experiment_version"],
        "git_head": head,
        "branch": previous["branch"],
        "ready_for_live_experiment": ready and not changed_control_files,
        "control_plane": {
            "runtime_tree_sha256": control_provenance["runtime_tree_sha256"],
            "registry_sha256": control_provenance["registry_sha256"],
            "runtime_files_locked": ready and not changed_control_files,
            "local_modifications": bool(changed_control_files),
        },
        "experiment": {
            **previous["experiment"],
            "source": experiment_source_provenance(),
            "registry": experiment_registry_provenance(),
            "plan_registry_sha256_by_engine": plan_registry,
        },
        "workflow": workflow,
        "environment": previous["environment"],
        "gepa_public_evaluator_contract": previous[
            "gepa_public_evaluator_contract"
        ],
        "gepa_reflection_protocol": previous["gepa_reflection_protocol"],
        "optimizer_semantic_response_contract": {
            "final_text_required": True,
            "maximum_metered_attempts": 2,
            "maximum_semantic_retries": 1,
            "trace_and_gepa_shared": True,
            "reasoning_text_is_not_final_output": True,
        },
        "scientific_protocol_changed": False,
        "supersedes": {
            "path": str(previous_path.relative_to(PACKAGE_ROOT)),
            "git_head": previous["git_head"],
            "runtime_tree_sha256": previous["control_plane"][
                "runtime_tree_sha256"
            ],
            "experiment_source_sha256": previous["experiment"]["source"][
                "sha256"
            ],
            "reason": "metered optimizer empty-text semantic retry",
        },
    }


def build_main_experiment_lock(
    *,
    ci_run_id: int,
    ci_job_id: int,
    ci_head_sha: str,
    ci_url: str,
) -> dict[str, Any]:
    """Freeze the CI-verified source and user authorization for the main run."""
    if ci_run_id <= 0 or ci_job_id <= 0:
        raise ValueError("main lock requires positive CI run and job IDs")
    if len(ci_head_sha) != 40:
        raise ValueError("main lock requires a full CI head SHA")
    if not ci_url.startswith("https://github.com/"):
        raise ValueError("main lock requires the GitHub Actions run URL")
    previous_path = PACKAGE_ROOT / "control_plane_lock_after_empty_text_retry.json"
    previous = _load_json(previous_path)
    frozen_path = PACKAGE_ROOT / "manifests/preregistration_frozen.json"
    authorization_path = PACKAGE_ROOT / "reports/main_cost_authorization.json"
    frozen = _load_json(frozen_path)
    authorization = _load_json(authorization_path)
    if not authorization.get("authorized") or not authorization.get(
        "numeric_ceiling_waived"
    ):
        raise RuntimeError("main lock requires explicit user cost authorization")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != ci_head_sha:
        raise RuntimeError("current HEAD does not match the supplied CI head")
    changed_control_files = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            "opto/features/recursive_opt",
            "opto/optimizers/optoprime_v2.py",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if changed_control_files:
        raise RuntimeError("frozen control-plane files have local modifications")
    golden = _load_json(
        REPOSITORY_ROOT
        / "artifacts/control_plane_v2/golden_specs/uc4_positive.normalized.json"
    )
    control_provenance = control_plane.compile_plan(golden).code_provenance
    expected_control = previous["control_plane"]
    if control_provenance["runtime_tree_sha256"] != expected_control[
        "runtime_tree_sha256"
    ]:
        raise RuntimeError("frozen control-plane runtime digest changed")
    if control_provenance["registry_sha256"] != expected_control["registry_sha256"]:
        raise RuntimeError("frozen control-plane registry digest changed")
    task = str(frozen["task"])
    plan_registry = {
        engine: control_plane.compile_plan(
            build_spec(task=task, engine=engine, seed=0, output_directory=None)
        ).code_provenance["registry_sha256"]
        for engine in ("fixed", "trace", "gepa_optimize_anything")
    }
    return {
        "schema_version": "recursive-opt-experiment-lock/v6",
        "experiment_version": previous["experiment_version"],
        "git_head": head,
        "branch": previous["branch"],
        "ready_for_main_experiment": True,
        "control_plane": expected_control,
        "experiment": {
            **previous["experiment"],
            "source": experiment_source_provenance(),
            "registry": experiment_registry_provenance(),
            "plan_registry_sha256_by_engine": plan_registry,
            "selected_task": task,
        },
        "main_preregistration_sha256": hashlib.sha256(
            frozen_path.read_bytes()
        ).hexdigest(),
        "main_authorization_sha256": hashlib.sha256(
            authorization_path.read_bytes()
        ).hexdigest(),
        "workflow": {
            "name": "recursive-opt v2 contracts",
            "run_id": ci_run_id,
            "job": "recursive-opt v2 offline (required)",
            "job_id": ci_job_id,
            "head_sha": ci_head_sha,
            "status": "completed",
            "conclusion": "success",
            "url": ci_url,
        },
        "environment": previous["environment"],
        "scientific_protocol_changed": False,
        "monetary_authorization": {
            "numeric_ceiling_waived": True,
            "forecast_cost_usd": authorization["forecast_cost_usd"],
            "authorization_source": authorization["authorization_source"],
        },
        "supersedes": {
            "path": str(previous_path.relative_to(PACKAGE_ROOT)),
            "git_head": previous["git_head"],
            "experiment_source_sha256": previous["experiment"]["source"]["sha256"],
            "reason": "freeze main matrix and explicit user monetary-gate waiver",
        },
    }
