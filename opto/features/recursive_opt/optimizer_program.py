"""Portable black-box optimizer programs; process isolation is NOT a security sandbox.

Only standard library imports are needed by the worker. Objective evaluation stays
in the parent; candidates receive only prior observations, bounds and a local seed.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opto.trainer.objectives import EvaluationResult


@dataclass(frozen=True)
class ProposalResult:
    """Typed proposal validity; invalid execution has no numeric point or score."""

    status: str
    point: list[float] | None = None
    stdout: str = ""
    stderr: str = ""

    @property
    def valid(self) -> bool:
        """Whether the proposal satisfied the checked contract."""
        return self.status == "valid"


@dataclass(frozen=True)
class ProgramEvaluation:
    """Fixture trajectory with exact objective accounting and typed invalidity."""

    status: str
    evaluations: list[dict[str, Any]]
    proposals: list[ProposalResult]
    budget: int

    @property
    def valid(self) -> bool:
        """Whether every budgeted evaluation completed successfully."""
        return self.status == "valid"

    @property
    def evaluated_count(self) -> int:
        """Number of objective calls actually consumed."""
        return len(self.evaluations)

    @property
    def best_value(self) -> float | None:
        """Report a numeric result only for a valid complete trajectory."""
        return min(row["value"] for row in self.evaluations) if self.valid else None

    @property
    def behavior_signature(self) -> list[list[float]]:
        """Actual ordered proposals, independent of optimizer source bytes."""
        return [row["x"] for row in self.evaluations]

    def to_dict(self) -> dict[str, Any]:
        """Serialize execution evidence without imputing invalid scores."""
        return {
            **asdict(self),
            "valid": self.valid,
            "evaluated_count": self.evaluated_count,
            "best_value": self.best_value,
            "behavior_signature": self.behavior_signature,
        }


def parse_program(text: str) -> str:
    """Extract plain source or a single Python fence, without code repair."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("optimizer response must contain nonempty source")
    if "```" in text:
        matches = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
        if len(matches) != 1 or text.count("```") != 2:
            raise ValueError(
                "optimizer response must contain exactly one Python code fence"
            )
        text = matches[0]
    return text.strip() + "\n"


def _finite_number(value: Any) -> bool:
    """Accept real JSON numbers while excluding booleans and nonfinite values."""
    return type(value) in (int, float) and math.isfinite(value)


def _point_status(point: Any, bounds: Sequence[Sequence[float]]) -> str:
    """Validate the returned point independently from objective evaluation."""
    if not isinstance(point, (list, tuple)) or len(point) != len(bounds):
        return "shape_error"
    if not all(_finite_number(value) for value in point):
        return "nonfinite"
    if any(not low <= value <= high for value, (low, high) in zip(point, bounds)):
        return "out_of_bounds"
    return "valid"


def _validate_inputs(history: Any, bounds: Any, seed: int, timeout_s: float) -> None:
    """Validate the host API before starting any candidate process."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not _finite_number(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be positive and finite")
    if not isinstance(bounds, (list, tuple)) or not bounds:
        raise ValueError("bounds must be a nonempty sequence of finite ordered pairs")
    for pair in bounds:
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or not all(_finite_number(value) for value in pair)
            or pair[0] >= pair[1]
        ):
            raise ValueError("bounds must contain finite strictly increasing pairs")
    if not isinstance(history, list):
        raise TypeError("history must be a list of prior observations")
    for row in history:
        if (
            not isinstance(row, dict)
            or set(row) != {"x", "value"}
            or not _finite_number(row["value"])
            or _point_status(row["x"], bounds) != "valid"
        ):
            raise ValueError(
                "history entries must contain only a legal x and finite value"
            )


def _execute_once(
    source: str, payload: dict[str, Any], timeout_s: float
) -> ProposalResult:
    """Execute in a fresh credential-free process, killing its group on timeout."""
    with tempfile.TemporaryDirectory(prefix="optimizer-program-") as directory:
        root = Path(directory)
        # Ship only proposal validation to the child, never the objective implementation.
        worker_source = (
            "from __future__ import annotations\n"
            "import inspect, json, math\nfrom pathlib import Path\n"
            + "\n".join(
                inspect.getsource(function)
                for function in (_finite_number, _point_status, _worker)
            )
            + "\n_worker()\n"
        )
        (root / "worker.py").write_text(worker_source, encoding="utf-8")
        (root / "optimizer.py").write_text(source, encoding="utf-8")
        (root / "request.json").write_text(json.dumps(payload), encoding="utf-8")
        with (
            (root / "stdout.txt").open("wb") as stdout,
            (root / "stderr.txt").open("wb") as stderr,
        ):
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", "worker.py"],
                cwd=root,
                env={"PATH": os.defpath, "LANG": "C.UTF-8"},
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            status = None
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                status = "timeout"
            finally:
                # Also reap descendants left by a completed candidate.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        logs = [
            (root / name).read_bytes()[:8192].decode("utf-8", errors="replace")
            for name in ("stdout.txt", "stderr.txt")
        ]
        logs = [re.sub(r"sk-[A-Za-z0-9_-]+", "<redacted>", value) for value in logs]
        if status is None:
            try:
                result = json.loads((root / "result.json").read_text())
                status = result["status"]
                point = result.get("point")
                if status == "valid":
                    status = _point_status(point, payload["bounds"])
                return ProposalResult(
                    status,
                    [float(x) for x in point] if status == "valid" else None,
                    *logs,
                )
            except (OSError, ValueError, KeyError, TypeError):
                status = "process_error"
        return ProposalResult(status, None, *logs)


def propose_point(
    source: str,
    history: list[dict[str, Any]],
    bounds: list[list[float]],
    seed: int,
    *,
    timeout_s: float = 2.0,
) -> ProposalResult:
    """Validate one proposal and replay it in a second fresh process for determinism."""
    _validate_inputs(history, bounds, seed, timeout_s)
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be nonempty Python text")
    payload = {"history": history, "bounds": bounds, "seed": seed}
    first = _execute_once(source, payload, timeout_s)
    if not first.valid:
        return first
    second = _execute_once(source, payload, timeout_s)
    if not second.valid or first.point != second.point:
        return ProposalResult(
            "nondeterministic",
            None,
            first.stdout + second.stdout,
            first.stderr + second.stderr,
        )
    return first


def evaluate_program(
    source: str, *, seed: int, budget: int = 8, timeout_s: float = 2.0
) -> ProgramEvaluation:
    """Run only the public 2-D calibration sphere; no final/holdout API exists."""
    if type(budget) is not int or budget <= 0:
        raise ValueError("budget must be a positive integer")
    history: list[dict[str, Any]] = []
    proposals: list[ProposalResult] = []
    for _ in range(budget):
        proposal = propose_point(
            source, history, [[-5.0, 5.0], [-5.0, 5.0]], seed, timeout_s=timeout_s
        )
        proposals.append(proposal)
        if not proposal.valid:
            return ProgramEvaluation(proposal.status, history, proposals, budget)
        point = proposal.point
        value = sum((x - shift) ** 2 for x, shift in zip(point, [1.25, -0.75]))
        history.append({"x": point, "value": value})
    return ProgramEvaluation("valid", history, proposals, budget)


def _worker() -> None:
    """Trusted launcher for candidate code, with bounded file output and no secrets."""
    import resource

    resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    payload = json.loads(Path("request.json").read_text())
    result: dict[str, Any] = {}
    try:
        compiled = compile(Path("optimizer.py").read_text(), "optimizer.py", "exec")
        namespace: dict[str, Any] = {"__name__": "optimizer"}
        exec(compiled, namespace)  # noqa: S102 - explicit candidate execution in child
        function = namespace.get("propose")
        if not callable(function):
            result["status"] = "missing_propose"
        else:
            parameters = list(inspect.signature(function).parameters.values())
            if (
                len(parameters) != 3
                or [p.name for p in parameters] != ["history", "bounds", "seed"]
                or any(
                    p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                    or p.default is not p.empty
                    for p in parameters
                )
            ):
                result["status"] = "signature_error"
            else:
                point = function(payload["history"], payload["bounds"], payload["seed"])
                result["status"] = _point_status(point, payload["bounds"])
                if result["status"] == "valid":
                    result["point"] = list(point)
    except SyntaxError:
        result["status"] = "syntax_error"
    except ImportError:
        result["status"] = "import_error"
    except BaseException:  # noqa: BLE001 - candidate SystemExit is typed invalid
        result["status"] = "exception"
    Path("result.json").write_text(json.dumps(result, allow_nan=False))


def optimizer_evaluator(output: Any, example: Any, context: Any) -> EvaluationResult:
    """Adapt portable source to the canonical typed evaluator without executing it in Trace."""
    from opto.trainer.objectives import EvaluationResult

    data = getattr(output, "data", output)
    source = data["components"]["optimizer"]
    if not isinstance(example, dict) or set(example) != {"seed", "budget"}:
        raise ValueError("optimizer fixture examples require only seed and budget")
    result = evaluate_program(source, seed=example["seed"], budget=example["budget"])
    return EvaluationResult(
        valid=result.valid,
        status="ok" if result.valid else "invalid",
        metrics={"value": result.best_value} if result.valid else {},
        feedback=f"OptimizerProgramV0: {result.status}; evaluations={result.evaluated_count}/{result.budget}",
        artifacts=result.to_dict(),
        error=None if result.valid else result.status,
    )


def optimizer_spec(
    source: str, *, seed: int, budget: int = 8, engine: str = "fixed"
) -> dict[str, Any]:
    """Build a minimal canonical fixture spec using the existing trainable component module."""
    from opto.features.recursive_opt import spec as control

    _validate_inputs([], [[-5, 5], [-5, 5]], seed, 2.0)
    if type(budget) is not int or budget <= 0:
        raise ValueError("budget must be a positive integer")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be nonempty Python text")
    if engine not in {"fixed", "trace", "gepa"}:
        raise ValueError("engine must be fixed, trace or gepa")
    reference = "recursive_opt.evaluator.optimizer_program@1"
    control.register_evaluator(reference, optimizer_evaluator)
    example = {"seed": seed, "budget": budget}
    return {
        "schema_version": control.SCHEMA_VERSION,
        "kind": control.SPEC_KIND,
        "runtime": {"offline": True, "seed": seed},
        "module": {
            "ref": "recursive_opt.module.reasoning_workflow@1",
            "config": {"components": {"optimizer": source}},
        },
        "surface": {"kind": "module", "targets": ["optimizer"]},
        "engine": (
            {"name": engine, "config": {"iterations": 2, "num_candidates": 1}}
            if engine == "trace"
            else {"name": engine}
        ),
        "objective": {
            "evaluator_ref": reference,
            "intent": "Minimize black-box objective within the fixed evaluation budget.",
            "metrics": {
                "value": {"direction": "minimize", "source": "evaluation.metrics.value"}
            },
            "selection": {"mode": "scalar", "score_key": "value"},
        },
        "datasets": {"train": [example], "validation": [example], "holdout": []},
    }
