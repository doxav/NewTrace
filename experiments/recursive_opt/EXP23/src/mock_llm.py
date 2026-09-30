"""Offline stand-in for the optimizer LLM: a feedback-blind random local proposer.

It reads the current policy from the real OptoPrimeV2 prompt and answers in the
real response format, so Trace's graph, prompt, parser and update all execute.
It ignores feedback on purpose: offline runs then measure only the score and
parent-selection mechanics, never "feedback quality", which needs a real LLM
(see alternative A3 in README). The EvoX-style arms use the same ``mutate``.
"""

import hashlib
import json
import math
import random
import re

from src.policies import KNOBS, PolicyInvalid, dump_knobs, parse_knobs

VARIABLE = re.compile(r'<variable name="(?P<name>\w+)" type="str">\s*<value>\n(?P<value>.*?)\n</value>', re.DOTALL)
CODE_TEMPLATE = '''def select_parent(members, rng):
    """Rank-softmax parent selection generated from knobs {knobs}."""
    if rng.random() < {epsilon!r}:
        return rng.randrange(len(members))
    elite = sorted(range(len(members)), key=lambda i: members[i]['rank'])[:{elite_k!r}]
    logits = [members[i]['rank_pct'] / {temperature!r} - {reuse_penalty!r} * members[i]['uses'] for i in elite]
    top = max(logits)
    weights = [math.exp(v - top) for v in logits]
    return rng.choices(elite, weights=weights)[0]
'''
KNOB_LINE = re.compile(r'generated from knobs (\{.*?\})')


def mutate(knobs: dict, rng: random.Random, global_rate: float = 0.5) -> dict:
    """Half fresh global draws, half local perturbations within bounds.

    Local-only proposals from the stock policy (epsilon=1, clipped) stay uniform for
    many steps; real EXP22 proposals jumped straight to fitness-biased selection.
    """
    (t_lo, t_hi), (r_lo, r_hi), (k_lo, k_hi) = KNOBS['temperature'], KNOBS['reuse_penalty'], KNOBS['elite_k']
    if rng.random() < global_rate:
        return {'temperature': round(math.exp(rng.uniform(math.log(t_lo), math.log(t_hi))), 4), 'epsilon': round(rng.random(), 4),
                'reuse_penalty': round(rng.uniform(r_lo, r_hi), 4), 'elite_k': round(math.exp(rng.uniform(0, math.log(k_hi))))}
    return {
        'temperature': round(min(t_hi, max(t_lo, knobs['temperature'] * math.exp(rng.gauss(0, 0.8)))), 4),
        'epsilon': round(min(1.0, max(0.0, knobs['epsilon'] + rng.gauss(0, 0.25))), 4),
        'reuse_penalty': round(min(r_hi, max(r_lo, knobs['reuse_penalty'] + rng.gauss(0, 0.6))), 4),
        'elite_k': int(min(k_hi, max(k_lo, round(knobs['elite_k'] * math.exp(rng.gauss(0, 0.8)))))),
    }


def knobs_to_code(knobs: dict) -> str:
    """Render knobs as a code-surface policy."""
    return CODE_TEMPLATE.format(knobs=json.dumps(knobs, sort_keys=True), **knobs)


def read_policy(text: str) -> dict:
    """Recover knobs from either surface; unknown code falls back to the stock knobs."""
    try:
        return parse_knobs(text)
    except PolicyInvalid:
        match = KNOB_LINE.search(text)
        return parse_knobs(match.group(1)) if match else {'temperature': 5.0, 'epsilon': 1.0, 'reuse_penalty': 0.0, 'elite_k': 64}


class MockOptimizerLLM:
    """Callable with the DummyLLM/OpenAI-compatible signature used by OptoPrimeV2."""

    def __init__(self, seed: int = 0) -> None:
        self.seed, self.calls, self.prompt_chars = seed, 0, []

    def __call__(self, *args, **kwargs) -> str:
        messages = kwargs.get('messages') or args[0]
        prompt = messages[-1]['content']
        self.calls += 1
        self.prompt_chars.append(sum(len(str(m.get('content', ''))) for m in messages))
        match = VARIABLE.search(prompt.split('# Variables', 1)[-1])
        if match is None and '# Parent policy to improve' in prompt:  # EvoX-style prompt: answer with a fenced block
            parent = prompt.split('# Parent policy to improve\n', 1)[1].split('\n(score:', 1)[0].split('\n\n#', 1)[0]
            rng = random.Random(int(hashlib.sha256(f'{self.seed}:{self.calls}:{parent}'.encode()).hexdigest()[:16], 16))
            proposal = mutate(read_policy(parent), rng)
            return f'```json\n{dump_knobs(proposal)}\n```' if parent.lstrip().startswith('{') else f'```python\n{knobs_to_code(proposal)}```'
        if match is None:
            return '<reasoning>no variable found</reasoning>'
        current = match.group('value')
        rng = random.Random(int(hashlib.sha256(f'{self.seed}:{self.calls}:{current}'.encode()).hexdigest()[:16], 16))
        proposal = mutate(read_policy(current), rng)
        value = dump_knobs(proposal) if current.lstrip().startswith('{') else knobs_to_code(proposal)
        return f"<reasoning>mock random local proposal</reasoning>\n<variable>\n<name>{match.group('name')}</name>\n<value>\n{value}\n</value>\n</variable>"
