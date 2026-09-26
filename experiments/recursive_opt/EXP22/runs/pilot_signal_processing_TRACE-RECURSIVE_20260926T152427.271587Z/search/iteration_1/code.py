# EVOLVE-BLOCK-START
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from skydiscover.optimize.config import DatabaseConfig
from skydiscover.optimize.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


def _score_of(program: EvolvedProgram) -> float:
    """Best available scalar score for a program (defaults to 0.0)."""
    metrics = getattr(program, "metrics", None) or {}
    for key in ("combined_score", "overall_score", "composite_score"):
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])
    return 0.0


class EvolvedProgramDatabase(ProgramDatabase):
    """Score-biased search strategy database.

    Sampling strategy (informed by measured window feedback):
    - Parents are drawn with a softmax-style bias toward high-scoring
      programs, so mutations mostly build on the strongest ancestors while
      weaker programs still receive occasional exploration pressure.
    - Context programs mix elite (top-scored) examples with random ones to
      give the mutator both quality anchors and population diversity.
    - Multiobjective mode still prefers the global Pareto front, but weights
      front members by their scalar score as a tie-breaker.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs
    ) -> str:
        """Add a program to the database."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def _score_biased_choice(self, candidates: List[EvolvedProgram]) -> EvolvedProgram:
        """Roulette-wheel selection with softmax temperature over scores."""
        if len(candidates) == 1:
            return candidates[0]
        scores = [_score_of(p) for p in candidates]
        m = max(scores)
        # Temperature scaled to score spread; avoids degenerate all-or-nothing.
        spread = max(m - min(scores), 1e-9)
        temperature = max(spread, 1e-6)
        weights = [math.exp((s - m) / temperature) for s in scores]
        total = sum(weights)
        r = self.rng.random() * total
        acc = 0.0
        for p, w in zip(candidates, weights):
            acc += w
            if r <= acc:
                return p
        return candidates[-1]

    def _elite_programs(
        self, candidates: List[EvolvedProgram], k: int
    ) -> List[EvolvedProgram]:
        """Top-k programs by scalar score."""
        return sorted(candidates, key=_score_of, reverse=True)[:k]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """
        Picks a parent and set of context programs.

        Parent: score-biased (Pareto front in multiobjective mode).
        Context: elite anchors plus random exploration, excluding the parent.
        """
        candidates = list(self.programs.values())

        if len(candidates) == 0:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 0

        if self.is_multiobjective_enabled():
            front = [p for p in self.get_pareto_front() if p.id in self.programs]
            parent_pool = front if front else candidates
        else:
            parent_pool = candidates

        parent = self._score_biased_choice(list(parent_pool))

        # Context: elite anchors first (quality guidance), then random fill
        # from the rest of the population (diversity), excluding the parent.
        others = [p for p in candidates if p.id != parent.id]
        examples: List[EvolvedProgram] = []
        if n_ctx > 0 and others:
            num_elite = max(1, n_ctx // 2)
            for elite in self._elite_programs(others, num_elite):
                if len(examples) >= n_ctx:
                    break
                if elite.id not in {e.id for e in examples}:
                    examples.append(elite)
            remaining = [p for p in others if p.id not in {e.id for e in examples}]
            self.rng.shuffle(remaining)
            for p in remaining:
                if len(examples) >= n_ctx:
                    break
                examples.append(p)

        parent_dict = {"": parent}
        context_programs_dict = {"": examples}

        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END
