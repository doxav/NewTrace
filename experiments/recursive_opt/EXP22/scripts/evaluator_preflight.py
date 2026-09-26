"""Run ten sequential golden/parity checks without any LLM calls."""

import asyncio
import contextlib
import io
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.preflight import write_json
from src.evaluation import (
    EVALUATOR_PROTOCOL,
    SKY,
    TASKS,
    TraceEvaluatorAdapter,
    stock_evaluator,
)

GOLDEN = {
    "prism": {"max_kvpr": 20.891622105209393, "success_rate": 1.0, "combined_score": 21.891622105209393},
    "signal_processing": {"combined_score": 0.49904861783269006, "composite_score": 0.4518100150842093, "correlation": 0.8420816095203836, "noise_reduction": 0.3347443615622788, "success_rate": 1.0},
}


def stable(metrics: dict[str, Any]) -> dict[str, Any]:
    """Exclude timing only; preserve all quality and failure fields."""
    timing = {'execution_time', 'avg_execution_time', 'mean_execution_time', 'total_execution_time'}
    return {key: value for key, value in metrics.items() if key not in timing}


def equivalent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Compare every deterministic metric with absolute tolerance 1e-12."""
    if left.keys() != right.keys():
        return False
    return all(
        math.isclose(value, right[key], rel_tol=0, abs_tol=1e-12)
        if isinstance(value, (int, float)) and isinstance(right[key], (int, float))
        else value == right[key]
        for key, value in left.items()
    )


async def run() -> dict[str, Any]:
    """Measure stock full evaluation and identical stock-cascade adapter paths."""
    from skydiscover.optimize.utils.metrics import get_score

    report: dict[str, Any] = {"passed": True, "evaluator_protocol": EVALUATOR_PROTOCOL, "repetitions": 10, "tolerance": 1e-12, "tasks": {}}
    for task, relative in TASKS.items():
        stock = stock_evaluator(task)
        adapter = TraceEvaluatorAdapter(task)
        source = (SKY / relative / "initial_program.py").read_text()
        observations = []
        for _ in range(10):
            direct = stable(stock.evaluate_function(str(SKY / relative / "initial_program.py")))
            native = stable((await stock.evaluate_program(source)).metrics)
            trace = stable(await adapter.evaluate(source))
            observations.append({"direct_full": direct, "stock_cascade": native, "trace_adapter": trace})
        golden_pass = all(equivalent({k: row['direct_full'].get(k) for k in GOLDEN[task]}, GOLDEN[task]) for row in observations)
        parity_pass = all(equivalent(row['stock_cascade'], row['trace_adapter']) for row in observations)
        deterministic = all(equivalent(row[path], observations[0][path]) for row in observations for path in row)
        broken = "# Deliberately missing the benchmark entry point.\n"
        invalid_stock = stable((await stock.evaluate_program(broken)).metrics)
        invalid_trace = stable(await adapter.evaluate(broken))
        invalid_pass = equivalent(invalid_stock, invalid_trace) and get_score(invalid_stock) == get_score(invalid_trace) and get_score(invalid_stock) < get_score(observations[0]['stock_cascade'])
        stage1 = None
        stage1_pass = True
        if task == 'signal_processing':
            stage1 = stable(stock._eval_module.evaluate_stage1(str(SKY / relative / "initial_program.py")))
            stage1_pass = equivalent(stage1, {"runs_successfully": 1.0, "composite_score": 0.7, "output_length": 91})
        passed = golden_pass and parity_pass and deterministic and invalid_pass and stage1_pass
        report['passed'] = report['passed'] and passed
        report['tasks'][task] = {
            "passed": passed, "golden_pass": golden_pass, "parity_pass": parity_pass,
            "deterministic": deterministic, "invalid_pass": invalid_pass, "stage1_pass": stage1_pass,
            "observations": observations, "stage1": stage1,
            "invalid_stock": invalid_stock, "invalid_trace": invalid_trace,
            "invalid_rank": get_score(invalid_stock),
        }
        stock.close()
        adapter.close()
    return report


if __name__ == '__main__':
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = asyncio.run(run())
    write_json(ROOT / 'artifacts/evaluator_parity.json', result)
    print({"evaluator_parity_passed": result['passed']})
    sys.exit(0 if result['passed'] else 2)
