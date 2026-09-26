"""Prove timeout, cancellation and worker failure cannot overlap later work."""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skydiscover.optimize.config import EvaluatorConfig
from src.evaluation import ProcessEvaluator


class ProcessEvaluationTests(unittest.IsolatedAsyncioTestCase):
    """Exercise actual subprocesses with bounded, deliberately broken candidates."""

    def setUp(self) -> None:
        """Build a tiny evaluator so failures do not depend on benchmark quality."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        fixture = self.directory / 'evaluator.py'
        fixture.write_text('import runpy\ndef evaluate(path):\n    return runpy.run_path(path)["run"]()\n')
        self.evaluator = ProcessEvaluator(EvaluatorConfig(evaluation_file=str(fixture), timeout=10, max_retries=0, cascade_evaluation=False))
        self.evaluator.audit_directory = self.directory / 'audit'
        self.addCleanup(self.evaluator.close)

    def records(self) -> list[dict[str, Any]]:
        """Read completed stage audits without relying on output buffering."""
        return [json.loads(line) for line in (self.evaluator.audit_directory / 'stages.jsonl').read_text().splitlines()]

    def assert_stopped(self, pid: int) -> None:
        """Reject a live process; an adopted zombie cannot execute or overlap."""
        stat = Path(f'/proc/{pid}/stat')
        self.assertTrue(not stat.exists() or stat.read_text().split(') ', 1)[1].startswith('Z'))

    async def test_success_error_crash_and_credentials(self) -> None:
        """Preserve metrics, contain exceptions/crashes and exclude credentials."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fixture-secret'}):
            result = await self.evaluator.evaluate_program('import os\ndef run():\n    assert "OPENROUTER_API_KEY" not in os.environ\n    return {"combined_score": 2.5}\n')
        self.assertEqual(result.metrics, {'combined_score': 2.5})
        for body in ('raise ValueError("fixture")', 'import os; os._exit(9)', 'return object()'):
            result = await self.evaluator.evaluate_program(f'def run():\n    {body}\n')
            self.assertEqual(result.metrics, {'error': 0.0})
        for row in self.records():
            self.assertTrue(row['reaped'])
            self.assert_stopped(row['pid'])

    async def test_timeout_kills_group_then_next_candidate_runs(self) -> None:
        """Kill a TERM-ignoring worker and its child before the next evaluation."""
        marker = self.directory / 'child.pid'
        source = f'import subprocess, sys, signal, time\nfrom pathlib import Path\ndef run():\n    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])\n    Path({str(marker)!r}).write_text(str(child.pid))\n    while True: time.sleep(.01)\n'
        task = asyncio.create_task(self.evaluator.evaluate_program(source))
        for _ in range(500):
            if marker.exists():
                break
            await asyncio.sleep(.02)
        self.assertTrue(marker.exists())
        # Expire the real parent boundary only after the child exists.
        self.evaluator.config.timeout = .01
        result = await asyncio.wait_for(task, 5)
        self.assertEqual(result.metrics, {'error': 0.0, 'timeout': True})
        self.assert_stopped(int(marker.read_text()))
        self.assert_stopped(self.records()[-1]['pid'])
        self.evaluator.config.timeout = 10
        result = await self.evaluator.evaluate_program('def run(): return {"combined_score": 3}\n')
        self.assertEqual(result.metrics, {'combined_score': 3})
        self.assertEqual([row['status'] for row in self.records()], ['timeout', 'completed'])

    async def test_cancellation_reaps_worker(self) -> None:
        """Cancellation performs the same mandatory cleanup as timeout."""
        marker = self.directory / 'started'
        task = asyncio.create_task(self.evaluator.evaluate_program(f'from pathlib import Path\nimport time\ndef run():\n    Path({str(marker)!r}).touch()\n    time.sleep(60)\n'))
        for _ in range(500):
            if marker.exists():
                break
            await asyncio.sleep(.02)
        self.assertTrue(marker.exists())
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        row = self.records()[-1]
        self.assertEqual(row['status'], 'cancelled')
        self.assertTrue(row['reaped'])
        self.assert_stopped(row['pid'])

    async def test_normal_completion_cleans_descendant(self) -> None:
        """A successful evaluator may not leave background children behind."""
        marker = self.directory / 'child.pid'
        result = await self.evaluator.evaluate_program(f'import subprocess, sys\nfrom pathlib import Path\ndef run():\n    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])\n    Path({str(marker)!r}).write_text(str(child.pid))\n    return {{"combined_score": 1}}\n')
        self.assertEqual(result.metrics, {'combined_score': 1})
        for _ in range(50):
            stat = Path(f'/proc/{int(marker.read_text())}/stat')
            if not stat.exists() or stat.read_text().split(') ', 1)[1].startswith('Z'):
                break
            await asyncio.sleep(.01)
        self.assert_stopped(int(marker.read_text()))

    async def test_cascade_timeout_preserves_stage_one(self) -> None:
        """Retain the stock stage-two timeout fallback and failure artifact."""
        fixture = self.directory / 'evaluator.py'
        fixture.write_text(fixture.read_text() + '\ndef evaluate_stage1(path):\n    return {"combined_score": 1.0}\ndef evaluate_stage2(path):\n    import time\n    time.sleep(60)\n')
        self.evaluator.close()
        self.evaluator = ProcessEvaluator(EvaluatorConfig(evaluation_file=str(fixture), timeout=2, max_retries=0, cascade_evaluation=True))
        self.evaluator.audit_directory = self.directory / 'audit'
        self.addCleanup(self.evaluator.close)
        result = await self.evaluator.evaluate_program('pass\n')
        self.assertEqual(result.metrics, {'combined_score': 1.0, 'timeout': True})
        self.assertEqual(result.artifacts, {'failure_stage': 'stage2'})
        self.assertEqual([row['stage'] for row in self.records()], ['evaluate_stage1', 'evaluate_stage2'])
        self.assert_stopped(self.records()[-1]['pid'])
