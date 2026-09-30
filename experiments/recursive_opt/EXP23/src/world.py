"""Empirical-replay solution-search world calibrated on EXP22 logs (no LLM).

A child's score is its parent's score plus a delta drawn from the real EXP22
child-minus-parent deltas observed for parents in the same score bin, so
improvement gets harder near the top as it did live. Positive deltas are
shrunk by ``wear ** parent.uses`` to model parents wearing out with reuse.
Invalid children occur at the observed per-task rate.

World randomness is keyed by (seed, iteration, parent id, parent reuse), so two policies
that choose the same parent in the same state draw the same child (common random numbers).

Limitations: children are conditionally independent given the parent score;
context programs and LLM memory are not modelled. Absolute Signal levels are
optimistic. Claims must hold across all WORLDS, not in one.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'data' / 'deltas.json'
DATA_KEY = {'prism': 'prism', 'signal_processing': 'signal'}
TASKS = {
    'prism': {'init': 21.891622105209393, 'invalid': 0.30},
    'signal_processing': {'init': 0.49904861783269006, 'invalid': 0.03},
}
# Multiplier applied to positive deltas per prior reuse of the parent.
# W1 is a rough fit to PRISM (0 new bests in 39 children of parents reused 4+ times).
WORLDS = {'W0': 1.0, 'W1': 0.8, 'W2': 0.5}
N_BINS = 5
STAGNATION_EPS = 0.01  # stock EvoX DEFAULT_IMPROVEMENT_THRESHOLD (absolute, both tasks)


@dataclass
class Member:
    """One evaluated solution in the population."""

    id: int
    score: float
    born: int
    uses: int = 0


@dataclass
class Population:
    """Solution population shared by every policy; never reset on policy switches."""

    members: list[Member] = field(default_factory=list)
    best: float = float('-inf')
    next_id: int = 0

    @classmethod
    def start(cls, score: float) -> 'Population':
        """Seed with the task's initial program."""
        population = cls()
        population.add(score, 0)
        return population

    def add(self, score: float, iteration: int) -> Member:
        """Insert one valid child."""
        member = Member(self.next_id, score, iteration)
        self.next_id += 1
        self.members.append(member)
        self.best = max(self.best, score)
        return member

    def view(self, iteration: int) -> list[dict]:
        """Describe members to a selection policy without exposing internals."""
        order = sorted(range(len(self.members)), key=lambda i: -self.members[i].score)
        rank = {index: position for position, index in enumerate(order)}
        size = max(1, len(self.members) - 1)
        return [{'score': m.score, 'rank': rank[i], 'rank_pct': 1 - rank[i] / size, 'uses': m.uses, 'age': iteration - m.born} for i, m in enumerate(self.members)]

    def percentile(self, score: float) -> float:
        """Fraction of the current population strictly below ``score``."""
        return sum(m.score < score for m in self.members) / len(self.members)

    def copy(self) -> 'Population':
        """Deep copy for forked evaluation."""
        return Population([Member(m.id, m.score, m.born, m.uses) for m in self.members], self.best, self.next_id)


class World:
    """Stochastic child generator calibrated on EXP22 deltas."""

    def __init__(self, task: str, world: str = 'W1', data: Path = DATA) -> None:
        if task not in TASKS or world not in WORLDS:
            raise ValueError(f'unknown task/world {task}/{world}')
        rows = [r for r in json.loads(data.read_text())[DATA_KEY[task]] if r['parent'] is not None and r['child'] is not None]
        parents = sorted(r['parent'] for r in rows)
        self.task, self.name = task, world
        self.init, self.invalid = TASKS[task]['init'], TASKS[task]['invalid']
        self.wear = WORLDS[world]
        self.edges = [parents[int(i * (len(parents) - 1) / N_BINS)] for i in range(1, N_BINS)]
        self.bins: list[list[float]] = [[] for _ in range(N_BINS)]
        for row in rows:
            self.bins[self._bin(row['parent'])].append(row['child'] - row['parent'])
        spread = sorted(abs(d) for b in self.bins for d in b)
        self.scale = spread[len(spread) // 2] or 1.0  # median |delta|, for scale-free credit

    def _bin(self, score: float) -> int:
        return sum(score > edge for edge in self.edges)

    def generate(self, parent: Member, seed: int, iteration: int) -> float | None:
        """Draw one child with randomness keyed by (seed, iteration, parent id, parent reuse)."""
        rng = random.Random(f'{self.task}:{seed}:{iteration}:{parent.id}:{parent.uses}')
        if rng.random() < self.invalid:
            return None
        delta = rng.choice(self.bins[self._bin(parent.score)])
        if delta > 0:
            delta *= self.wear ** parent.uses
        return parent.score + delta
