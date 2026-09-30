"""Deterministic LLM scripts shared by both runs (outputs depend only on the call index)."""

import os
import random

SEED = int(os.environ.get('EQUIV_SEED', '1234'))
_rng = random.Random(SEED)
SOLUTION_ACTIONS = []
for _ in range(1000):
    roll = _rng.random()
    if roll < 0.10:
        SOLUTION_ACTIONS.append(('nodiff', None))
    elif roll < 0.15:
        SOLUTION_ACTIONS.append(('miss', None))
    elif roll < 0.28:
        SOLUTION_ACTIONS.append(('bad', None))
    else:
        SOLUTION_ACTIONS.append(('add', _rng.choice([-0.4, -0.1, 0.0, 0.003, 0.05, 0.2, 0.6])))

META_SEQUENCE = os.environ.get('EQUIV_META', 'invalid,greedy_refine,invalid,topk_diverge,late_raise,uniform_label,greedy_refine,topk_diverge,uniform_label').split(',')


def solution_reply(n):
    kind, delta = SOLUTION_ACTIONS[(n - 1) % len(SOLUTION_ACTIONS)]
    if kind == 'nodiff':
        return 'I have no idea.'
    if kind == 'miss':
        return '<<<<<<< SEARCH\nTHIS LINE DOES NOT EXIST\n=======\nSCORE = 0\n>>>>>>> REPLACE'
    if kind == 'bad':
        return '<<<<<<< SEARCH\n# end\n=======\nSCORE = SCORE + "x"\n# end\n>>>>>>> REPLACE'
    return f'<<<<<<< SEARCH\n# end\n=======\nSCORE = SCORE + {delta}\n# end\n>>>>>>> REPLACE'


def meta_policy(n):
    return META_SEQUENCE[(n - 1) % len(META_SEQUENCE)]


# Policy bodies: identical logic, written once per framework.
# Names available: candidates (insertion order), k, score(p) -> float (-inf if missing), rng, REFINE, DIVERGE.
BODIES = {
    'greedy_refine': 'parent = max(candidates, key=score)\nexamples = [p for p in candidates if p.id != parent.id][:k]\nlabel = REFINE',
    'topk_diverge': 'top = sorted(candidates, key=score, reverse=True)[:3]\nparent = rng.choice(top)\nexamples = [p for p in rng.sample(candidates, min(k, len(candidates))) if p.id != parent.id]\nlabel = DIVERGE if rng.random() < 0.5 else ""',
    'late_raise': 'if len(candidates) > 30:\n    raise RuntimeError("late failure")\nparent = max(candidates, key=score)\nexamples = []\nlabel = ""',
    'uniform_label': 'parent = rng.choice(candidates)\nexamples = [p for p in rng.sample(candidates, min(k + 1, len(candidates))) if p.id != parent.id][:k]\nlabel = REFINE',
}


def indent(text, spaces):
    return '\n'.join(' ' * spaces + line for line in text.splitlines())


def stock_policy_source(name):
    if name == 'invalid':
        return 'class EvolvedProgramDatabase(:\n    pass\n'
    body = indent(BODIES[name].replace('REFINE', 'self.REFINE_LABEL').replace('DIVERGE', 'self.DIVERGE_LABEL'), 8)
    return f'''# EVOLVE-BLOCK-START
from dataclasses import dataclass
from typing import Optional

from skydiscover.optimize.config import DatabaseConfig
from skydiscover.optimize.search.base_database import Program, ProgramDatabase


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


class EvolvedProgramDatabase(ProgramDatabase):
    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(self, num_context_programs: Optional[int] = 4, **kwargs):
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")
        k = num_context_programs or 0
        rng = self.rng

        def score(p):
            value = p.metrics.get("combined_score") if p.metrics else None
            return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else float("-inf")

{body}
        return {{label: parent}}, {{"": examples}}
# EVOLVE-BLOCK-END
'''


def v2_policy_source(name):
    if name == 'invalid':
        return 'class Policy(:\n    pass\n'
    body = indent(BODIES[name].replace('REFINE', '"refine"').replace('DIVERGE', '"diverge"'), 8)
    return f'''class Policy:
    def __init__(self, labels):
        self.labels = labels

    def observe(self, candidate):
        pass

    def sample(self, population, rng, num_context):
        candidates = population.members
        if not candidates:
            raise ValueError("No candidates available for sampling")
        k = num_context or 0

        def score(p):
            value = population.score(p)
            return value if value is not None else float("-inf")

{body}
        return parent, examples, label
'''
