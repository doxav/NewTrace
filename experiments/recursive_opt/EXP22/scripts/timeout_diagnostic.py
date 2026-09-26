"""Reproduce stock evaluator timeout leakage with bounded, unpaid fixtures."""

import asyncio
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.preflight import write_json
from src.evaluation import stock_evaluator


async def diagnose() -> dict[str, Any]:
    """Show outer timeout overlap and inner shutdown waiting without hanging."""
    evaluator = stock_evaluator('prism')
    evaluator.config.timeout = 0.05
    release, finished, started = threading.Event(), threading.Event(), threading.Event()

    def slow(path: str) -> dict[str, float]:
        """Wait for explicit cleanup, with a hard fixture-side upper bound."""
        started.set()
        release.wait(2)
        finished.set()
        return {'combined_score': 1.0}

    def subsequent(path: str) -> dict[str, bool]:
        """Observe whether the earlier evaluator is still running."""
        return {'earlier_worker_active': started.is_set() and not finished.is_set()}

    timed_out = False
    try:
        try:
            await evaluator._run_stage(slow, 'bounded-fixture')
        except TimeoutError:
            timed_out = True
        overlap = await evaluator._run_stage(subsequent, 'subsequent-fixture')
    finally:
        release.set()
        await asyncio.to_thread(finished.wait, 2)
    before = time.monotonic()
    inner_timeout = False
    try:
        evaluator._eval_module.run_with_timeout(time.sleep, args=(0.15,), timeout_seconds=0.01)
    except TimeoutError:
        inner_timeout = True
    elapsed = time.monotonic() - before
    evaluator.close()
    return {'outer_timeout_raised': timed_out, 'subsequent_call_overlapped_worker': overlap['earlier_worker_active'],
            'fixture_worker_cleaned_up': finished.is_set(), 'inner_timeout_raised': inner_timeout,
            'inner_nominal_timeout_seconds': 0.01, 'inner_sleep_seconds': 0.15, 'inner_actual_return_seconds': elapsed,
            'inner_context_waited_for_worker': elapsed >= 0.14,
            'scope': 'Bounded unpaid fixtures using unmodified stock implementations; test-instance outer deadline shortened to 0.05s. No benchmark score is measured.'}


if __name__ == '__main__':
    result = asyncio.run(diagnose())
    write_json(ROOT/'artifacts/timeout_diagnostic.json', result)
    print(result)
