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
    run(algo, variation_schedule='periodic', period=2, num_inspirations=1, inspiration_mode='always')
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


def test_default_is_stagnation_with_plain_diverge_and_matches_explicit_never():
    default, _, default_prompts = make()
    run(default, patience=2)
    explicit, _, explicit_prompts = make()
    run(explicit, variation_schedule='stagnation', patience=2, refine_after_gain=2, inspiration_mode='never')
    assert [r['mode'] for r in default.variation_log] == [r['mode'] for r in explicit.variation_log]
    norm = lambda ps: [re.sub(r'\d+', '#', s) for s in ps]  # trace node names carry a process-global counter
    assert norm(default_prompts) == norm(explicit_prompts)
    assert 'combine' not in [r['mode'] for r in default.variation_log] and 'diverge' in [r['mode'] for r in default.variation_log]


def test_alternate_switches_exploration_steps_between_diverge_and_combine():
    algo, _, _ = make()
    run(algo, variation_schedule='periodic', period=1, inspiration_mode='alternate', num_inspirations=1)
    explore = [r['mode'] for r in algo.variation_log if r['mode'] in ('diverge', 'combine')]
    assert explore[:4] == ['diverge', 'combine', 'diverge', 'combine']


def test_context_style_lists_inspirations_without_combine_instruction():
    algo, _, prompts = make()
    run(algo, variation_schedule='periodic', period=1, inspiration_mode='always', inspiration_style='context', num_inspirations=1)
    shown = [p for p, r in zip(prompts, algo.variation_log) if r['mode'] == 'combine' and 'Other evaluated solution 1' in p]
    assert shown and all('VARIATION MODE: DIVERGE.' in p and 'COMBINE' not in p for p in shown)


def test_inspiration_arguments_are_validated():
    for kwargs in ({'inspiration_mode': 'sometimes'}, {'inspiration_style': 'merge'}, {'inspiration_mode': 'always', 'num_inspirations': 0}):
        algo, _, _ = make()
        with pytest.raises(ValueError):
            run(algo, **kwargs)


def test_instruction_never_leaks_into_later_free_steps():
    """Regression (EXP28): candidates created on a diverge step carry an optimizer copy; expanding them later must not
    reuse that step's instruction. Proposals improve every step, so each new candidate becomes the next parent."""
    prompts, counter = [], {'n': 1.0}

    def llm(messages, **kwargs):
        prompts.append(messages[1]['content'])
        counter['n'] += 1.0
        name = re.findall(r'<variable name="\s*(.*?)" type=.*>', messages[1]['content'])
        return f'<reasoning>r</reasoning><variable><name>{name[0] if name else "x"}</name><value>{counter["n"]}</value></variable>'
    agent = NumericAgent()
    algo = VariationSearch(agent, OptoPrimeV2(agent.parameters(), llm=DummyLLM(llm)))
    algo.train(DistanceGuide(), DATA, num_steps=9, num_candidates=1, num_proposals=1, batch_size=1, num_threads=1, test_frequency=None,
               variation_schedule='periodic', period=2, refine_after_gain=0)
    modes = [r['mode'] for r in algo.variation_log]
    assert 'diverge' in modes and len(prompts) == len(modes)
    for prompt, mode in zip(prompts, modes):
        assert prompt.count('VARIATION MODE') == (0 if mode == 'free' else 1), (mode, prompt.count('VARIATION MODE'))
