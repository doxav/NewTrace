"""Measure frozen PrioritySearch scheduling without model or benchmark requests."""

import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'worktrees/trace_cp_b'))

from opto import trace
from opto.optimizers.optoprime_v2 import OptoPrimeV2
from opto.trainer.algorithms.priority_search import PrioritySearch
from opto.trainer.guide import Guide
from opto.utils.llm import DummyLLM

COUNTS: dict[str, Any] = {}


@trace.model
class Agent:
    """Represent a policy with a cheap, measurable scalar parameter."""

    def __init__(self) -> None:
        self.param = trace.node(1.0, trainable=True, name='policy')

    def forward(self, x: float) -> Any:
        """Count evaluations while retaining a real Trace dependency graph."""
        COUNTS['evaluations'] += 1
        return self.param + x


class ProbeGuide(Guide):
    """Score the scalar without external evaluation."""

    def get_feedback(self, query: Any, response: Any, reference: Any = None, **kwargs: Any) -> tuple[float, str]:
        """Return a deterministic score and optimizer feedback."""
        return float(response), 'Increase the policy value.'


def fake_step(optimizer: OptoPrimeV2, **kwargs: Any) -> dict[Any, float]:
    """Propose improvement, regression, then improvement; record chosen parents."""
    COUNTS['proposals'] += 1
    COUNTS['parents'].append(float(optimizer.parameters[0].data))
    value = [2.0, 0.0, 3.0][min(COUNTS['proposals'] - 1, 2)]
    return {optimizer.parameters[0]: value}


def forbid_llm(*args: Any, **kwargs: Any) -> str:
    """Reject any unexpected model invocation."""
    raise AssertionError('Unexpected LLM call in offline probe')


def main() -> None:
    """Assert initialization, proposal counts, and archived-parent selection."""
    rows = []
    for steps in (1, 4, 10):
        COUNTS.clear()
        COUNTS.update(evaluations=0, proposals=0, parents=[])
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink), patch.object(OptoPrimeV2, '_step', fake_step):
            agent = Agent()
            optimizer = OptoPrimeV2(agent.parameters(), llm=DummyLLM(forbid_llm), log=False)
            trainer = PrioritySearch(agent, optimizer, num_threads=1)
            trainer.train(
                guide=ProbeGuide(), train_dataset={'inputs': [0.0], 'infos': [None]},
                num_steps=steps, num_epochs=0, num_candidates=1, num_proposals=1,
                batch_size=1, num_batches=1, num_threads=1, test_frequency=None,
                log_frequency=1, save_frequency=None, validate_exploration_candidates=False,
                use_best_candidate_to_explore=True, decouple_optimizers=False,
            )
        rows.append({'num_steps': steps, 'trainer_iterations': trainer.n_iters, **COUNTS,
                     'final_parameter': agent.param.data})
    assert [row['proposals'] for row in rows] == [0, 3, 9]
    assert rows[1]['parents'] == [1.0, 2.0, 2.0]
    report = {'method': 'Frozen PrioritySearch; real OptoPrimeV2 backward, mocked _step; no API or benchmark calls', 'results': rows}
    Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
