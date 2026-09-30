"""Use the stock evaluator unchanged as the EXP22 hybrid evaluation boundary."""

import asyncio
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

from skydiscover.optimize.config import load_config
from skydiscover.optimize.evaluation.evaluation_result import EvaluationResult
from skydiscover.optimize.evaluation.evaluator import Evaluator

SKY = Path('/home/xav/code/evo-compare/repos/skydiscover')
TASKS = {"prism": "benchmarks/ADRS/prism", "signal_processing": "benchmarks/math/signal_processing"}
EVALUATOR_PROTOCOL = 'process-stage-v1'


def start_worker(arguments: list[str], env: dict[str, str], log: BinaryIO) -> subprocess.Popen[bytes]:
    """Create the owned process synchronously so cancellation cannot orphan it."""
    return subprocess.Popen(arguments, env=env, stdout=log, stderr=log, start_new_session=True)


class ProcessEvaluator(Evaluator):
    """Preserve stock cascade/retries while enforcing each stage's timeout."""

    async def evaluate_program(self, program_solution: str, program_id: str = '', mode: str = 'train') -> EvaluationResult:
        """Count evaluator invocations separately from cascaded stage attempts."""
        audit = getattr(self, 'audit_directory', None)
        if audit:
            audit.mkdir(parents=True, exist_ok=True)
            with (audit / 'calls.jsonl').open('a') as stream:
                stream.write(json.dumps({'source_sha256': hashlib.sha256(program_solution.encode()).hexdigest(), 'program_id': program_id, 'mode': mode}) + '\n')
        return await super().evaluate_program(program_solution, program_id, mode)

    @classmethod
    def replace(cls, original: Evaluator, audit_directory: Path | None = None) -> 'ProcessEvaluator':
        """Reuse the exact evaluator configuration and discard its unused module."""
        replacement = cls(original.config, original.llm_judge, max_concurrent=1, env_vars=original.env_vars)
        replacement.audit_directory = audit_directory
        original.close()
        return replacement

    async def _run_stage(self, func: Callable[[str], Any], program_path: str) -> EvaluationResult:
        """Kill and reap the worker group before returning, including cancellation."""
        name = func.__name__
        if name not in ('evaluate', 'evaluate_stage1', 'evaluate_stage2'):
            raise ValueError('Unsupported evaluation stage')
        started = time.monotonic()
        source = Path(program_path).read_bytes()
        record: dict[str, Any] = {'stage': name, 'source_sha256': hashlib.sha256(source).hexdigest(), 'status': 'error'}
        audit = getattr(self, 'audit_directory', None)
        if audit:
            audit.mkdir(parents=True, exist_ok=True)
            (audit / f"{record['source_sha256']}.py").write_bytes(source)
        # Workers do not need provider credentials; never put them in input files.
        env = {key: value for key, value in {**os.environ, **self.env_vars}.items() if not any(token in key.upper() for token in ('API_KEY', 'TOKEN', 'SECRET', 'PASSWORD'))}
        with tempfile.TemporaryDirectory(prefix='exp22-eval-') as temporary:
            result_path = Path(temporary) / 'result.json'
            log_path = audit / f"{record['source_sha256']}_{name}.log" if audit else Path(temporary) / 'worker.log'
            with log_path.open('ab') as log:
                process = start_worker([sys.executable, '-I', str(Path(__file__).with_name('evaluation_worker.py')), str(Path(self.evaluation_file).resolve()), name, program_path, str(result_path)], env, log)
                record['pid'] = process.pid
                try:
                    while process.poll() is None:
                        if time.monotonic() - started >= self.config.timeout:
                            record['status'] = 'timeout'
                            raise asyncio.TimeoutError
                        await asyncio.sleep(0.02)
                    if process.returncode != 0 or not result_path.exists():
                        raise RuntimeError('Evaluation worker failed without a valid result')
                    result = json.loads(result_path.read_text())
                    if result.get('exception') == 'TimeoutError':
                        record['status'] = 'timeout'
                        raise asyncio.TimeoutError
                    if 'exception' in result:
                        raise RuntimeError(f"Evaluation worker raised {result['exception']}")
                    if not isinstance(result.get('metrics'), dict) or not isinstance(result.get('artifacts'), dict):
                        raise TypeError('Evaluation worker returned malformed metrics or artifacts')
                    record['status'] = 'completed'
                    return EvaluationResult(**result)
                except asyncio.CancelledError:
                    record['status'] = 'cancelled'
                    raise
                finally:
                    # SIGKILL also handles workers that ignore SIGTERM. Kill the
                    # group even after normal exit, to remove leftover children.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    record.update(returncode=process.returncode, reaped=True, wall_seconds=time.monotonic() - started)
                    if audit:
                        with (audit / 'stages.jsonl').open('a') as stream:
                            stream.write(json.dumps(record) + '\n')


def stock_evaluator(task: str) -> Evaluator:
    """Load stock config with one evaluator worker and no LLM judge."""
    if task not in TASKS:
        raise ValueError("Unknown EXP22 benchmark")
    directory = SKY / TASKS[task]
    config = load_config(directory / 'config.yaml')
    config.evaluator.evaluation_file = str(directory / 'evaluator/evaluator.py')
    return Evaluator(config.evaluator, max_concurrent=1)


class TraceEvaluatorAdapter:
    """Delegate the entire cascade and error contract to SkyDiscover."""

    def __init__(self, task: str) -> None:
        """Retain one stock evaluator for sequential candidate evaluation."""
        self.evaluator = ProcessEvaluator.replace(stock_evaluator(task))

    async def evaluate(self, source: str) -> dict[str, Any]:
        """Return unchanged metrics without remapping invalid-candidate fitness."""
        if not isinstance(source, str):
            raise TypeError("Candidate source must be a string")
        return (await self.evaluator.evaluate_program(source)).metrics

    def close(self) -> None:
        """Release the stock dynamically loaded benchmark module."""
        self.evaluator.close()
