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


class EvolvedProgramDatabase(ProgramDatabase):
    """Score-biased search strategy database.

    Sampling policy (informed by measured window feedback):
    - Parents are drawn with rank/softmax weighting on ``combined_score``
      instead of uniformly, so weak programs are rarely chosen as parents
      while exploration is preserved.
    - Context programs mix elite (top-scoring) members with random members
      of the population for diversity.
    - When ``config.pareto_objectives`` is set, multiobjective mode prefers
      members of the global Pareto front, weighted by scalarized score.
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

    def _score_of(self, program: EvolvedProgram) -> float:
        """Scalar score used for weighting; falls back to 0.0."""
        metrics = getattr(program, "metrics", None) or {}
        score = metrics.get("combined_score", 0.0)
        try:
            return float(score)
        except (TypeError, ValueError):
            return 0.0

    def _weighted_choice(self, candidates: List[EvolvedProgram]) -> EvolvedProgram:
        """Softmax-weighted choice over scores (temperature-scaled)."""
        if len(candidates) == 1:
            return candidates[0]

        scores = [self._score_of(p) for p in candidates]
        m = max(scores)
        # Temperature: spread of scores determines sharpness; keep it gentle.
        spread = max(m - min(scores), 1e-9)
        temperature = max(spread, 1e-9)
        weights = [math.exp((s - m) / temperature) for s in scores]
        total = sum(weights)
        r = self.rng.random() * total
        acc = 0.0
        for p, w in zip(candidates, weights):
            acc += w
            if r <= acc:
                return p
        return candidates[-1]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """
        Picks a parent and set of context programs.

        Parent: score-weighted draw (Pareto-front-restricted in
        multiobjective mode). Context: elites first, then random fill.
        """
        candidates = list(self.programs.values())

        if len(candidates) == 0:
            raise ValueError("No candidates available for sampling")

        pool = candidates
        if self.is_multiobjective_enabled():
            front = [p for p in self.get_pareto_front() if p.id in self.programs]
            if front:
                pool = front

        # Parent: score-weighted selection from the pool.
        parent = self._weighted_choice(pool)

        n_ctx = num_context_programs or 0
        # Context: prefer other high scorers (elites), then random fill.
        others = [p for p in candidates if p.id != parent.id]
        others_sorted = sorted(others, key=self._score_of, reverse=True)

        examples: List[EvolvedProgram] = []
        seen = set()
        # Take up to half the context slots from elites.
        num_elites = (n_ctx + 1) // 2
        for p in others_sorted[:num_elites]:
            examples.append(p)
            seen.add(p.id)
        # Fill remaining slots randomly for diversity.
        remaining = [p for p in others if p.id not in seen]
        self.rng.shuffle(remaining)
        for p in remaining:
            if len(examples) >= n_ctx:
                break
            examples.append(p)
            seen.add(p.id)

        parent_dict = {"": parent}
        context_programs_dict = {"": examples}

        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END
