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
    """Search strategy database with elite-biased parent sampling.

    Strategy (informed by measured window feedback):
    - Parents are sampled with a score-weighted distribution so high-scoring
      programs are exploited more often, while a bounded exploration
      probability keeps sampling from the broader population (uniform
      fallback in scalar mode).
    - Context programs mix the global best / top scorers with random
      population members, mirroring the observed pattern where the best
      result came from a weak parent paired with a strong context program.
    """

    # Exploration probability: chance of picking a purely random parent.
    _EXPLORE_PROB = 0.2
    # Temperature for score-weighted parent sampling.
    _TEMPERATURE = 1.0

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

    def _score(self, program: EvolvedProgram) -> float:
        """Return the scalar score used for weighting (0.0 if missing)."""
        metrics = getattr(program, "metrics", None) or {}
        score = metrics.get("combined_score", 0.0)
        try:
            return float(score)
        except (TypeError, ValueError):
            return 0.0

    def _weighted_parent(self, candidates: List[EvolvedProgram]) -> EvolvedProgram:
        """Score-weighted parent choice with bounded exploration."""
        if self.rng.random() < self._EXPLORE_PROB or len(candidates) == 1:
            return self.rng.choice(candidates)

        scores = [self._score(p) for p in candidates]
        m = max(scores)
        # Shift for numerical stability, then softmax.
        exp_scores = [math.exp((s - m) / self._TEMPERATURE) for s in scores]
        total = sum(exp_scores)
        r = self.rng.random() * total
        cum = 0.0
        for p, w in zip(candidates, exp_scores):
            cum += w
            if r <= cum:
                return p
        return candidates[-1]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """
        Picks a parent and set of context programs.

        Parent: score-weighted (elite-biased) sample of the population, or a
        Pareto-front member in multiobjective mode. Context: the best program
        plus top scorers, backfilled with random population members.
        """
        candidates = list(self.programs.values())

        if len(candidates) == 0:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 0

        front: List[EvolvedProgram] = []
        if self.is_multiobjective_enabled():
            front = [p for p in self.get_pareto_front() if p.id in self.programs]

        if front:
            parent = self.rng.choice(front)
        else:
            parent = self._weighted_parent(candidates)

        # Build context: dedupe, exclude parent.
        pool = [p for p in candidates if p.id != parent.id]

        # Deterministic elite part: best program first, then top scorers.
        elite = sorted(pool, key=self._score, reverse=True)
        examples: List[EvolvedProgram] = []
        seen_ids = set()
        for p in elite[: max(1, n_ctx // 2)]:
            examples.append(p)
            seen_ids.add(p.id)

        # Exploratory part: random fill from the rest of the population.
        rest = [p for p in pool if p.id not in seen_ids]
        self.rng.shuffle(rest)
        examples.extend(rest[: max(0, n_ctx - len(examples))])

        # If population is tiny, allow duplicate-free truncation only.
        examples = examples[:n_ctx]

        parent_dict = {"": parent}
        context_programs_dict = {"": examples}

        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END
