# EVOLVE-BLOCK-START
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from skydiscover.optimize.config import DatabaseConfig
from skydiscover.optimize.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


class EvolvedProgramDatabase(ProgramDatabase):
    """Search strategy database with explicit exploration/exploitation balance.

    Anti-stagnation policy:
    - Parent drawn from the Pareto front with probability ``_FRONT_PARENT_PROB``
      (lowered to 0.35 to reduce elite over-reuse), otherwise fitness-biased
      random over the population with a novelty (usage-count) penalty so
      rarely-sampled parents get a fair chance.
    - Context programs maximize score diversity and mix front/non-front
      members, and are also novelty-aware.
    """

    _FRONT_PARENT_PROB = 0.35

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self._usage: Dict[str, int] = {}

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

    def _score(self, p: EvolvedProgram) -> float:
        try:
            return float(p.metrics.get("combined_score", 0.0)) if p.metrics else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _diverse_context(
        self,
        candidates: List[EvolvedProgram],
        exclude_id: Optional[str],
        num_context: int,
        front_ids: Optional[set] = None,
    ) -> List[EvolvedProgram]:
        """Pick context programs favoring score diversity and coverage."""
        pool = [p for p in candidates if p.id != exclude_id]
        if not pool:
            return []

        by_score: Dict[float, List[EvolvedProgram]] = {}
        for p in pool:
            by_score.setdefault(self._score(p), []).append(p)

        scores = sorted(by_score.keys())
        # Interleave low/high scores to expose both ends of the distribution.
        ordered: List[float] = []
        lo, hi = 0, len(scores) - 1
        while lo <= hi:
            ordered.append(scores[lo])
            if lo != hi:
                ordered.append(scores[hi])
            lo += 1
            hi -= 1

        examples: List[EvolvedProgram] = []
        for score in ordered:
            bucket = by_score[score]
            # Novelty-aware pick within the bucket: prefer least-used members.
            min_use = min(self._usage.get(p.id, 0) for p in bucket)
            least_used = [p for p in bucket if self._usage.get(p.id, 0) == min_use]
            pick = self.rng.choice(least_used)
            examples.append(pick)
            if len(examples) >= num_context:
                break

        if len(examples) < num_context:
            chosen_ids = {p.id for p in examples}
            extras = [p for p in pool if p.id not in chosen_ids]
            self.rng.shuffle(extras)
            examples.extend(extras[: num_context - len(examples)])

        return examples[:num_context]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """
        Picks a parent and set of context programs.

        Exploitation from the Pareto front with probability
        ``_FRONT_PARENT_PROB``; otherwise fitness-biased choice over the
        whole population penalized by parent usage (novelty). Context is
        diversity-oriented.
        """
        candidates = list(self.programs.values())

        if len(candidates) == 0:
            raise ValueError("No candidates available for sampling")

        front: List[EvolvedProgram] = []
        front_ids: set = set()
        if self.is_multiobjective_enabled():
            front = [p for p in self.get_pareto_front() if p.id in self.programs]
            front_ids = {p.id for p in front}

        n_ctx = num_context_programs or 0
        parent: Optional[EvolvedProgram] = None

        if front and self.rng.random() < self._FRONT_PARENT_PROB:
            # Exploitation: novelty-aware choice within the front.
            min_use = min(self._usage.get(p.id, 0) for p in front)
            least_used = [p for p in front if self._usage.get(p.id, 0) == min_use]
            parent = self.rng.choice(least_used)
        else:
            # Exploration: fitness-biased with a usage penalty.
            pool = [p for p in candidates if p.id not in front_ids] or candidates
            scores = [self._score(p) for p in pool]
            lo, hi = min(scores), max(scores)
            if hi > lo:
                weights = [
                    (0.1 + 0.9 * (s - lo) / (hi - lo))
                    / (1.0 + self._usage.get(p.id, 0))
                    for s, p in zip(scores, pool)
                ]
            else:
                weights = [1.0 / (1.0 + self._usage.get(p.id, 0)) for p in pool]
            parent = self.rng.choices(pool, weights=weights, k=1)[0]

        # Record usage so future samples penalize over-reuse.
        self._usage[parent.id] = self._usage.get(parent.id, 0) + 1

        examples = self._diverse_context(candidates, parent.id, n_ctx, front_ids)

        parent_dict = {"": parent}
        context_programs_dict = {"": examples}

        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END
