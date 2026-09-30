"""Operational pipeline tests: scripted processes only, never live stage execution."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.recursive_opt._shared.optimizer_discovery import exp15 as E
from experiments.recursive_opt._shared.optimizer_discovery.exp17 import driver as D
from experiments.recursive_opt._shared.optimizer_discovery.exp17 import run_pipeline as P


def scripted_process(returncode: int = 0) -> SimpleNamespace:
    """Stand in for one started child without executing a registered study stage."""
    return SimpleNamespace(pid=8100, wait=lambda: returncode)


def fake_driver(root: Path, experiment: str = "exp17") -> tuple[Any, list[str]]:
    """Provide exact-grid and engineering guards without accessing real run files."""
    root.mkdir(exist_ok=True)
    calls: list[str] = []

    def grid() -> dict[str, Any]:
        """Represent the existing driver's completed exact-configuration preflight."""
        calls.append("grid")
        return {"config": {"experiment": experiment.replace("exp", "EXP-")}}

    def engineering() -> dict[str, Any]:
        """Represent authenticated completed engineering gates."""
        calls.append("engineering")
        return {"passed": True, "fixture": experiment}

    return (
        SimpleNamespace(
            RUN=root,
            require_confirmation=grid,
            require_main=grid,
            require_engineering=engineering,
        ),
        calls,
    )


@pytest.mark.parametrize("experiment", ["exp17", "exp18"])
def test_exact_stage_order_and_immutable_launch_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, experiment: str
) -> None:
    driver, gates = fake_driver(tmp_path, experiment)
    monkeypatch.setattr(P, "_driver", lambda name: driver)
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        """Observe each existing CLI command without executing a subprocess."""
        assert gates == ["grid", "engineering"]
        assert kwargs == {"cwd": P.REPOSITORY}
        assert "stdout" not in kwargs and "stderr" not in kwargs
        commands.append(command)
        step = len(commands)
        folder = tmp_path / "runtime/launch_001"
        assert E.exists(folder / f"step_{step:02d}_started.json")
        assert not E.exists(folder / f"step_{step:02d}_finished.json")
        return scripted_process()

    monkeypatch.setattr(P.subprocess, "Popen", run)
    report = P.run_pipeline(experiment)
    assert report["status"] == "complete"
    assert report["completed_steps"] == 6
    assert [command[-1] for command in commands[:5]] == [
        "generate",
        "receipts",
        "select",
        "audit",
        "analyze",
    ]
    assert all(
        command[2] == f"experiments.recursive_opt._shared.optimizer_discovery.{experiment}.driver"
        for command in commands[:5]
    )
    assert commands[-1][2:] == [
        "experiments.recursive_opt._shared.optimizer_discovery.exp17.verify_numerics",
        str(tmp_path),
    ]
    launch = tmp_path / "runtime/launch_001"
    source = E.read(launch / "helper_source.json")
    assert source["source"] == Path(P.__file__).read_text()
    assert (
        source["source_sha256"] == hashlib.sha256(source["source"].encode()).hexdigest()
    )
    assert E.read(launch / "launch_finished.json") == report
    assert E.read(launch / "launch_started.json")["commands"] == commands
    for step in range(1, 7):
        assert E.read(launch / f"step_{step:02d}_finished.json")["returncode"] == 0


@pytest.mark.parametrize("failed_step", [1, 3, 6])
def test_nonzero_return_stops_without_retry_or_later_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_step: int
) -> None:
    driver, _ = fake_driver(tmp_path)
    monkeypatch.setattr(P, "_driver", lambda name: driver)
    commands = []

    def run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        """Fail one existing CLI once and count any unapproved continuation."""
        commands.append(command)
        return scripted_process(7 if len(commands) == failed_step else 0)

    monkeypatch.setattr(P.subprocess, "Popen", run)
    report = P.run_pipeline("exp17")
    assert report["status"] == "failed"
    assert report["returncode"] == 7
    assert report["completed_steps"] == failed_step - 1
    assert len(commands) == failed_step
    assert not E.exists(
        tmp_path / f"runtime/launch_001/step_{failed_step + 1:02d}_started.json"
    )


@pytest.mark.parametrize("first_failure", [False, True])
def test_explicit_rerun_delegates_every_stage_and_preserves_old_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_failure: bool
) -> None:
    driver, _ = fake_driver(tmp_path)
    monkeypatch.setattr(P, "_driver", lambda name: driver)
    commands = []

    def run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        """Let frozen child CLIs own idempotence rather than skipping journaled stages."""
        commands.append(command)
        return scripted_process(7 if first_failure and len(commands) == 3 else 0)

    monkeypatch.setattr(P.subprocess, "Popen", run)
    first = P.run_pipeline("exp17")
    folder = tmp_path / "runtime/launch_001"
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    second = P.run_pipeline("exp17")
    assert first["launch"] == "launch_001"
    assert second["launch"] == "launch_002"
    first_count = 3 if first_failure else 6
    assert len(commands) == first_count + 6
    assert commands[:first_count] == commands[first_count : 2 * first_count]
    assert commands[first_count][-1] == "generate"
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before


def test_pipeline_lock_is_exclusive_and_does_not_hold_child_process_lock(
    tmp_path: Path,
) -> None:
    with P.pipeline_lease(tmp_path):
        with (
            pytest.raises(RuntimeError, match="pipeline already"),
            P.pipeline_lease(tmp_path),
        ):
            pytest.fail("second pipeline was admitted")
        with D.run_lease(tmp_path):
            pass
    with pytest.raises(ValueError), P.pipeline_lease(tmp_path):
        raise ValueError("unit interruption")
    with P.pipeline_lease(tmp_path):
        pass


@pytest.mark.parametrize("failure", ["grid", "engineering", "identity"])
def test_failed_gate_precedes_journals_or_child_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    driver, _ = fake_driver(tmp_path)

    def reject() -> dict[str, Any]:
        """Represent a real study gate rejection without weakening the guard."""
        raise RuntimeError("registered study gate rejected")

    if failure == "grid":
        driver.require_confirmation = reject
    elif failure == "engineering":
        driver.require_engineering = reject
    else:
        driver.require_confirmation = lambda: {"config": {"experiment": "EXP-18"}}
    monkeypatch.setattr(P, "_driver", lambda name: driver)
    monkeypatch.setattr(
        P.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("child started before gates"),
    )
    with pytest.raises(RuntimeError):
        P.run_pipeline("exp17")
    assert not (tmp_path / "runtime").exists()


def test_unknown_experiment_is_rejected_before_import_or_work() -> None:
    with pytest.raises(ValueError, match="exp17 or exp18"):
        P.run_pipeline("another-study")


def test_process_start_failure_is_recorded_without_error_text_or_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver, _ = fake_driver(tmp_path)
    monkeypatch.setattr(P, "_driver", lambda name: driver)

    def fail(command: list[str], **kwargs: Any) -> SimpleNamespace:
        """Provide sensitive-looking exception text that must never reach a journal."""
        raise OSError("PRIVATE_DIAGNOSTIC_TEXT")

    monkeypatch.setattr(P.subprocess, "Popen", fail)
    report = P.run_pipeline("exp17")
    assert report["status"] == "failed"
    assert report["returncode"] == 1
    step = E.read(tmp_path / "runtime/launch_001/step_01_finished.json")
    assert step["returncode"] is None
    assert step["error_type"] == "OSError"
    assert not (tmp_path / "runtime/launch_001/step_01_process.json").exists()
    assert all(
        "PRIVATE_DIAGNOSTIC_TEXT" not in path.read_text()
        for path in (tmp_path / "runtime/launch_001").iterdir()
    )


def test_parent_never_loads_live_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.recursive_opt._shared.optimizer_discovery import phase0

    driver, _ = fake_driver(tmp_path)
    monkeypatch.setattr(P, "_driver", lambda name: driver)
    monkeypatch.setattr(phase0, "_load_key", lambda: pytest.fail("parent loaded a key"))
    monkeypatch.setattr(
        P.subprocess,
        "Popen",
        lambda command, **kwargs: scripted_process(),
    )
    assert P.run_pipeline("exp17")["status"] == "complete"


def test_child_pid_is_persisted_before_wait_and_retained_after_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver, _ = fake_driver(tmp_path)
    monkeypatch.setattr(P, "_driver", lambda name: driver)
    processes: list[dict[str, Any]] = []
    waited: list[int] = []

    def popen(command: list[str], **kwargs: Any) -> SimpleNamespace:
        """Require child identity on disk before blocking for process completion."""
        assert kwargs == {"cwd": P.REPOSITORY}
        step = len(processes) + 1
        pid = 8000 + step
        path = tmp_path / f"runtime/launch_001/step_{step:02d}_process.json"
        assert not path.exists()
        processes.append({"step": step, "child_pid": pid, "command": command})

        def wait() -> int:
            """Check the immutable identity while the child would still be active."""
            record = E.read(path)
            assert record["child_pid"] == pid
            assert record["parent_pid"] == P.os.getpid()
            assert record["step"] == step
            assert record["stage"] == P.STAGES[step - 1]
            assert record["command"] == command
            assert record["recorded_ns"] > 0
            processes[-1]["record_bytes"] = path.read_bytes()
            waited.append(pid)
            return 0

        return SimpleNamespace(pid=pid, wait=wait)

    monkeypatch.setattr(P.subprocess, "Popen", popen)
    monkeypatch.setattr(
        P.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("pipeline must record a Popen child PID"),
    )
    assert P.run_pipeline("exp17")["status"] == "complete"
    assert waited == list(range(8001, 8007))
    for record in processes:
        path = tmp_path / f"runtime/launch_001/step_{record['step']:02d}_process.json"
        assert path.read_bytes() == record["record_bytes"]


def test_wait_interruption_preserves_child_identity_without_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver, _ = fake_driver(tmp_path)
    monkeypatch.setattr(P, "_driver", lambda name: driver)
    started: list[list[str]] = []

    def popen(command: list[str], **kwargs: Any) -> SimpleNamespace:
        """Leave an identifiable possibly active child after parent interruption."""
        started.append(command)

        def wait() -> int:
            """Interrupt without exposing exception text or starting the next stage."""
            raise KeyboardInterrupt("PRIVATE_INTERRUPT_TEXT")

        return SimpleNamespace(pid=9001, wait=wait)

    monkeypatch.setattr(P.subprocess, "Popen", popen)
    monkeypatch.setattr(
        P.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("pipeline must record a Popen child PID"),
    )
    with pytest.raises(KeyboardInterrupt):
        P.run_pipeline("exp17")
    folder = tmp_path / "runtime/launch_001"
    assert len(started) == 1
    assert E.read(folder / "step_01_process.json")["child_pid"] == 9001
    assert E.read(folder / "step_01_finished.json")["returncode"] is None
    assert E.read(folder / "launch_finished.json")["status"] == "failed"
    assert not (folder / "step_02_started.json").exists()
    assert all("PRIVATE_INTERRUPT_TEXT" not in p.read_text() for p in folder.iterdir())
