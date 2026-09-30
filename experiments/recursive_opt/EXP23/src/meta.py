"""Online meta-optimization arms inside one 100-call solution run (simulator).

Every arm spends exactly ``horizon`` solution calls on one shared population.
Arms differ only in how policies are scored, which parent policy is mutated,
and how the proposal is produced:

  fixed:<name>     reference policy held fixed (no meta)
  evox_log         EvoX as shipped: stagnation trigger, sequential windows scored by
                   LogWindowScorer, archive parent = argmax score, direct mutation
  trace_exp22      EXP22 as run: same trigger/score, parent = currently active policy,
                   fresh OptoPrimeV2 per trigger, identity graph, raw JSON feedback
  evox_paired      fixed score: challenger interleaved with the incumbent on the live
                   population, promoted only if paired score > 0; parent = incumbent
  trace_paired     same as evox_paired, proposal by one persistent OptoPrimeV2 whose
                   graph contains the executed decisions, compact feedback
"""

import json
import random
from typing import Any

from opto import trace
from opto.optimizers.optoprime_v2 import OptoPrimeV2
from opto.utils.llm import DummyLLM

from src.mock_llm import MockOptimizerLLM, knobs_to_code, mutate, read_policy
from src.policies import (
    REFERENCE,
    PolicyInvalid,
    compile_policy,
    dump_knobs,
    stock_text,
)
from src.search import log_window, step
from src.traced import Arena, SelectionPolicy, feedback_text
from src.world import STAGNATION_EPS, Population, World

ONLINE_ARMS = ('evox_log', 'trace_exp22', 'evox_paired', 'trace_paired')


@trace.bundle()
def measured_window(policy_source, observation):
    """Associate measured downstream feedback with its actual deployed policy (EXP22 graph)."""
    return observation


def _direct_proposal(parent_text: str, surface: str, rng: random.Random) -> str:
    knobs = mutate(read_policy(parent_text), rng)
    return dump_knobs(knobs) if surface == 'knobs' else knobs_to_code(knobs)


def run_online(world: World, seed: int, arm: str, *, surface: str = 'knobs', horizon: int = 100, window: int = 10,
               patience: int = 10, trigger: str = 'stagnation', credit: str = 'new_best', memory_size: int = 5,
               llm: Any = None) -> dict:
    """One solution run of ``horizon`` calls with online meta-optimization."""
    if arm.startswith('fixed:'):
        text = dump_knobs(REFERENCE[arm.split(':', 1)[1]])
        select, population = compile_policy('knobs', text), Population.start(world.init)
        for iteration in range(1, horizon + 1):
            step(world, population, select, seed, iteration)
        return {'final_best': population.best, 'proposals': 0, 'promotions': 0, 'meta_llm_calls': 0, 'mean_prompt_chars': 0}
    if arm not in ONLINE_ARMS:
        raise ValueError(arm)
    paired, traced = arm.endswith('paired'), arm.startswith('trace')
    llm = llm or MockOptimizerLLM(seed)
    client = DummyLLM(llm) if isinstance(llm, MockOptimizerLLM) else llm  # control-plane clients are already LLM-shaped
    rng = random.Random(f'proposer:{seed}')
    population = Population.start(world.init)
    incumbent_text = stock_text(surface)
    incumbent = compile_policy(surface, incumbent_text)
    active_text, active = incumbent_text, incumbent
    archive: list[dict] = []           # sequential arms: {'text', 'score'} per finished window
    window_start, last, stagnant = population.best, population.best, 0
    module = SelectionPolicy(surface, incumbent_text) if traced else None
    optimizer = OptoPrimeV2(module.parameters(), llm=client, memory_size=memory_size, log=False) if arm == 'trace_paired' else None
    pending_feedback = None            # (output node, feedback text) of the last traced paired window
    proposals = promotions = invalid = 0
    iteration = 0

    def propose(parent_text: str, observation: dict | None) -> str:
        nonlocal invalid
        if arm == 'trace_exp22':
            module.selection_policy._data = parent_text
            fresh = OptoPrimeV2(module.parameters(), llm=client, memory_size=0, log=False, initial_var_char_limit=100000)
            output = measured_window(module.selection_policy, observation)
            fresh.zero_feedback()
            fresh.backward(output, json.dumps(observation))
            fresh.step()
            text = str(module.selection_policy.data)
        elif arm == 'trace_paired':
            module.selection_policy._data = parent_text
            if pending_feedback is None:  # first proposal: the graph of an empty challenge would be meaningless; seed from the stock
                arena = Arena(world, population.copy(), seed, 0, 0, surface, 'solo')
                output, feedback = module(arena), 'No challenge has run yet; propose a policy likely to find new global bests faster than uniform random parent selection.'
            else:
                output, feedback = pending_feedback
            module.selection_policy._data = parent_text
            optimizer.zero_feedback()
            optimizer.backward(output, feedback)
            optimizer.step()
            text = str(module.selection_policy.data)
        else:
            text = _direct_proposal(parent_text, surface, rng)
        try:
            compile_policy(surface, text)
        except PolicyInvalid:
            invalid += 1
            return ''
        return text

    while iteration < horizon:
        step(world, population, active, seed, iteration + 1)
        iteration += 1
        stagnant = 0 if population.best - last > STAGNATION_EPS else stagnant + 1
        last = population.best
        fire = (stagnant >= patience) if trigger == 'stagnation' else (iteration % patience == 0)
        if not fire or iteration >= horizon:
            continue
        stagnant = 0
        if not paired:
            archive.append({'text': active_text, 'score': log_window(window_start, population.best, window)})
            parent = max(archive, key=lambda a: a['score'])['text'] if arm == 'evox_log' else active_text
            observation = {'window_metrics': archive[-1], 'policy_history': archive, 'population': [vars(m) for m in population.members]}
            proposals += 1
            text = propose(parent, observation)
            if text:
                active_text, active = text, compile_policy(surface, text)
                promotions += 1
            window_start = population.best
            continue
        proposals += 1
        text = propose(incumbent_text, None)
        if not text:
            continue
        calls = min(2 * window, horizon - iteration)
        arena = Arena(world, population, seed, iteration, calls, surface, 'paired', incumbent, credit)
        if traced:
            module.selection_policy._data = text
            output = module(arena)
            summary = output.data
            pending_feedback = (output, feedback_text(summary, incumbent_text) + f'\nchallenger policy: {text}')
        else:
            from src.search import interleave, paired_score
            challenger = compile_policy(surface, text)
            for offset, tag in enumerate(interleave(calls, seed, iteration)):
                arena.records.append(step(world, population, challenger if tag == 'challenger' else incumbent, seed, iteration + offset + 1, tag))
            summary = {'score': paired_score(arena.records, credit, world.scale)['score']}
        iteration += calls
        last = population.best
        if summary['score'] > 0:
            incumbent_text, incumbent = text, compile_policy(surface, text)
            active_text, active = incumbent_text, incumbent
            promotions += 1
    chars = getattr(llm, 'prompt_chars', [])
    return {'final_best': population.best, 'proposals': proposals, 'promotions': promotions, 'invalid_proposals': invalid,
            'meta_llm_calls': getattr(llm, 'calls', 0), 'mean_prompt_chars': sum(chars) / len(chars) if chars else 0,
            'final_policy': active_text}
