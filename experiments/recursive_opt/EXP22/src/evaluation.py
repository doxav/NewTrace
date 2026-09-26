"""Use the stock evaluator unchanged as the EXP22 hybrid evaluation boundary."""

from pathlib import Path
from typing import Any

from skydiscover.optimize.config import load_config
from skydiscover.optimize.evaluation.evaluator import Evaluator

SKY = Path('/home/xav/code/evo-compare/repos/skydiscover')
TASKS = {"prism": "benchmarks/ADRS/prism", "signal_processing": "benchmarks/math/signal_processing"}


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
        self.evaluator = stock_evaluator(task)

    async def evaluate(self, source: str) -> dict[str, Any]:
        """Return unchanged metrics without remapping invalid-candidate fitness."""
        if not isinstance(source, str):
            raise TypeError("Candidate source must be a string")
        return (await self.evaluator.evaluate_program(source)).metrics

    def close(self) -> None:
        """Release the stock dynamically loaded benchmark module."""
        self.evaluator.close()
