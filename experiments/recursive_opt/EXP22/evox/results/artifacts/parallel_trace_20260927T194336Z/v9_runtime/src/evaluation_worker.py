"""Execute one unchanged stock evaluator stage in an expendable process."""

import json
import random
import sys
from pathlib import Path

import numpy as np
from skydiscover.optimize.config import EvaluatorConfig
from skydiscover.optimize.evaluation.evaluator import Evaluator


def main() -> None:
    """Return structured metrics through a file, separate from candidate output."""
    evaluation_file, stage, program_path, output = sys.argv[1:]
    if stage not in ('evaluate', 'evaluate_stage1', 'evaluate_stage2'):
        raise ValueError('Unsupported evaluator stage')
    random.seed(42)
    np.random.seed(42)
    evaluator = Evaluator(EvaluatorConfig(evaluation_file=evaluation_file), max_concurrent=1)
    try:
        result = evaluator._normalize_result(getattr(evaluator._eval_module, stage)(program_path))
        payload = {'metrics': result.metrics, 'artifacts': result.artifacts}
    except TimeoutError:
        payload = {'exception': 'TimeoutError'}
    finally:
        evaluator.close()
    Path(output).write_text(json.dumps(payload))


if __name__ == '__main__':
    main()
