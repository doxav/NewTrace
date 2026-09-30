"""(1 + updates) x block schedule: stock block, then `updates` x (proposal, block).

Default 4-step blocks and 2 updates = 12 solution steps and 2 optimizer LLM calls per run.
Every proposal comes from the given LLM callable (live GLM or mock):

  evox_log      EvoX-style prompt, parent = best archived block score, LogWindowScorer block score
  llm_blind     same prompt without any score/statistics (LLM-prior control)
  trace_exp22   fresh OptoPrimeV2 per update, identity graph, raw JSON observation as feedback
  evox_paired   EvoX-style prompt from the incumbent; block = 2 challenger + 2 incumbent steps; promote if paired score > 0
  trace_paired  persistent OptoPrimeV2; blocks execute inside the traced module; compact decision feedback
"""

import json
from collections.abc import Callable
from typing import Any

from opto import trace
from opto.optimizers.optoprime_v2 import OptoPrimeV2
from opto.utils.llm import DummyLLM

from src.live import evox_prompt, extract_policy
from src.policies import PolicyInvalid, compile_policy, stock_text
from src.search import interleave, log_window, paired_score, step
from src.traced import Arena, SelectionPolicy, feedback_text
from src.world import Population, World

ARMS = ('evox_log', 'llm_blind', 'trace_exp22', 'evox_paired', 'trace_paired')


@trace.bundle()
def measured_window(policy_source, observation):
    """Associate measured downstream feedback with its actual deployed policy (EXP22 graph)."""
    return observation


def run_schedule(world: World, seed: int, arm: str, surface: str, llm: Callable[..., str], block: int = 4, updates: int = 2) -> dict[str, Any]:
    """One live-schedule run; returns the trajectory including every proposed policy text."""
    if arm not in ARMS:
        raise ValueError(arm)
    population, iteration = Population.start(world.init), 0
    stock = stock_text(surface)
    incumbent_text, incumbent = stock, compile_policy(surface, stock)
    traced = arm.startswith('trace')
    module = SelectionPolicy(surface, stock) if traced else None
    optimizer = OptoPrimeV2(module.parameters(), llm=DummyLLM(llm), memory_size=5, log=False) if arm == 'trace_paired' else None
    trajectory: list[dict[str, Any]] = []

    # Block 0: stock policy alone (traced for trace arms so the first update has real evidence).
    start = population.best
    if traced:
        arena = Arena(world, population, seed, iteration, block, surface, 'solo')
        output = module(arena)
        last = (output, feedback_text({**output.data, 'score': 0.0}, stock).replace('challenger minus incumbent new-best rate; >0 means the proposal beat the incumbent in the same stage', 'no challenger yet: stock policy alone'))
    else:
        for _ in range(block):
            step(world, population, incumbent, seed, iteration + 1)
            iteration += 1
    iteration = block
    archive = [{'text': stock, 'score': log_window(start, population.best, block)}]
    trajectory.append({'block': 0, 'policy': stock, 'block_score': archive[0]['score'], 'best_after': population.best})
    active_text = stock

    for update in range(1, updates + 1):
        stats = {'block': update, 'solution_steps_done': iteration, 'best_score': population.best, 'population_size': len(population.members)}
        if arm in ('evox_log', 'llm_blind', 'evox_paired'):
            parent = max(archive, key=lambda a: a['score']) if arm == 'evox_log' else {'text': incumbent_text, 'score': archive[-1]['score']} if arm == 'evox_paired' else {'text': stock}
            context = [a for a in archive if a['text'] != parent['text']][-3:]
            text = extract_policy(llm(messages=evox_prompt(surface, parent, context, stats, feedback=arm != 'llm_blind')))
        elif arm == 'trace_exp22':
            module.selection_policy._data = active_text
            fresh = OptoPrimeV2(module.parameters(), llm=DummyLLM(llm), memory_size=0, log=False, initial_var_char_limit=100000)
            observation = {'window_metrics': archive[-1], 'search_stats': stats, 'policy_history': archive, 'population': [vars(m) for m in population.members]}
            output = measured_window(module.selection_policy, observation)
            fresh.zero_feedback()
            fresh.backward(output, json.dumps(observation))
            fresh.step()
            text = str(module.selection_policy.data)
        else:
            module.selection_policy._data = incumbent_text
            optimizer.zero_feedback()
            optimizer.backward(*last)
            optimizer.step()
            text = str(module.selection_policy.data)
        record: dict[str, Any] = {'block': update, 'proposed': text, 'valid': True, 'error': None}
        try:
            challenger = compile_policy(surface, text)
        except PolicyInvalid as error:
            record.update(valid=False, error=str(error)[:300])
            challenger, text = None, None
        start = population.best
        if arm in ('evox_paired', 'trace_paired') and challenger is not None:
            if arm == 'trace_paired':
                module.selection_policy._data = text
                arena = Arena(world, population, seed, iteration, block, surface, 'paired', incumbent, 'new_best')
                output = module(arena)
                score = output.data['score']
                last = (output, feedback_text(output.data, incumbent_text) + f'\nchallenger policy: {text}')
            else:
                records = [step(world, population, challenger if tag == 'challenger' else incumbent, seed, iteration + k + 1, tag) for k, tag in enumerate(interleave(block, seed, iteration))]
                score = paired_score(records, 'new_best', world.scale)['score']
            record['promoted'] = score > 0
            if score > 0:
                incumbent_text, incumbent = text, challenger
        else:
            select = challenger or compile_policy(surface, active_text)
            for k in range(block):
                step(world, population, select, seed, iteration + k + 1)
            score = log_window(start, population.best, block)
            if challenger is not None:
                active_text = text
                archive.append({'text': text, 'score': score})
            if traced:  # trace_exp22 needs no traced evidence; keep last for symmetry
                last = None
        iteration += block
        record.update(block_score=score, best_after=population.best)
        trajectory.append(record)
    return {'arm': arm, 'surface': surface, 'seed': seed, 'solution_steps': iteration, 'final_best': population.best, 'trajectory': trajectory}
