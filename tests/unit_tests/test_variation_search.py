"""VariationSearch: scheduled REFINE / DIVERGE / combine instructions on top of PrioritySearch (DummyLLM, no API key)."""
import re

import pytest

from opto import trace
from opto.optimizers import OptoPrimeV2
from opto.trainer.algorithms import VariationSearch
from opto.trainer.algorithms.variation_search import VARIATION_INSTRUCTIONS
from opto.trainer.guide import Guide
from opto.utils.llm import DummyLLM


class DistanceGuide(Guide):
    """Score = -|x - 10| for a numeric parameter; never reaches the optimum, so the search stagnates."""

    def get_feedback(self, query, response, reference=None, **kwargs):
        value = float(response)
        return -abs(value - 10.0), f'value {value} is {abs(value - 10.0)} away from the target'


@trace.model
class NumericAgent:
    def __init__(self):
        self.x = trace.node(1.0, trainable=True, description='a number')

    def forward(self, _):
        return self.x


def make(proposal='2.0'):
    prompts = []

    def llm(messages, **kwargs):
        prompts.append(messages[1]['content'])
        name = re.findall(r'<variable name="\s*(.*?)" type=.*>', messages[1]['content'])
        return f'<reasoning>r</reasoning><variable><name>{name[0] if name else "x"}</name><value>{proposal}</value></variable>'
    agent = NumericAgent()
    optimizer = OptoPrimeV2(agent.parameters(), llm=DummyLLM(llm))
    return VariationSearch(agent, optimizer), optimizer, prompts


DATA = {'inputs': [None], 'infos': [None]}


def run(algo, **kwargs):
    algo.train(DistanceGuide(), DATA, num_steps=7, num_candidates=1, num_proposals=1, batch_size=1, num_threads=1,
               test_frequency=None, verbose=False, **kwargs)


def test_periodic_schedule_injects_and_restores_instruction():
    algo, optimizer, prompts = make()
    base = optimizer.objective
    run(algo, variation_schedule='periodic', period=3)
    modes = [r['mode'] for r in algo.variation_log]
    assert modes[2] == 'diverge' and modes[0] == 'free'
    diverge_prompts = [p for p, m in zip(prompts, modes) if m == 'diverge']
    assert diverge_prompts and all('VARIATION MODE: DIVERGE' in p for p in diverge_prompts)
    assert all('VARIATION MODE' not in p for p, m in zip(prompts, modes) if m == 'free')
    assert optimizer.objective == base  # restored after every step


def test_stagnation_schedule_diverges_after_patience_and_numeric_parameter_still_updates():
    algo, optimizer, prompts = make(proposal='2.0')
    run(algo, variation_schedule='stagnation', patience=2)
    modes = [r['mode'] for r in algo.variation_log]
    assert 'diverge' in modes  # 2.0 stops improving after the first step -> stagnation
    assert float(algo.agent.x.data) == pytest.approx(2.0)


def test_combine_shows_inspirations_when_memory_has_them():
    algo, _, prompts = make()
    run(algo, variation_schedule='periodic', period=2, num_inspirations=1)
    modes = [r['mode'] for r in algo.variation_log]
    assert 'combine' in modes
    shown = [p for p, m in zip(prompts, modes) if m == 'combine']
    assert any('VARIATION MODE: DIVERGE + COMBINE' in p and 'Inspiration 1' in p for p in shown) or \
        any(VARIATION_INSTRUCTIONS['diverge'].splitlines()[0] in p for p in shown)


def test_free_schedule_matches_priority_search_prompts():
    algo, optimizer, prompts = make()
    run(algo, variation_schedule='free')
    assert all(r['mode'] == 'free' for r in algo.variation_log)
    assert all('VARIATION MODE' not in p for p in prompts)


def test_bad_arguments():
    algo, _, _ = make()
    with pytest.raises(ValueError):
        run(algo, variation_schedule='sometimes')
