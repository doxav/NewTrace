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

    Multiobjective mode: parent drawn from the Pareto front with probability
    ``_FRONT_PARENT_PROB`` (exploitation), otherwise from the broader
    population with a fitness-biased random choice (exploration). Context
    programs are chosen to maximize diversity (distinct scores, mixed
    front/non-front membership) so offspring see varied information instead
    of a homogeneous elite set, combating stagnation.
    """

    _FRONT_PARENT_PROB = 0.6

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

    def _diverse_context(
        self,
        candidates: List[EvolvedProgram],
        exclude_id: Optional[str],
        num_context: int,
        front_ids: Optional[set] = None,
    ) -> List[EvolvedProgram]:
        """Pick context programs favoring score diversity and coverage.

        Strategy: greedily pick programs whose scores differ most from the
        already-picked ones, mixing Pareto-front and non-front members.
        """
        pool = [p for p in candidates if p.id != exclude_id]
        if not pool:
            return []

        # Deduplicate by score so repeated identical-score programs do not
        # dominate the context window.
        by_score: Dict[float, List[EvolvedProgram]] = {}
        for p in pool:
            score = p.metrics.get("combined_score", 0.0) if p.metrics else 0.0
            by_score.setdefault(score, []).append(p)

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
            # Prefer a random member; if front info available, alternate
            # between front and non-front members across buckets.
            pick = self.rng.choice(bucket)
            examples.append(pick)
            if len(examples) >= num_context:
                break

        # Fill remaining slots from unused pool members.
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

        Multiobjective: parent from the Pareto front with probability
        ``_FRONT_PARENT_PROB``; otherwise fitness-biased from the whole
        population. Context is diversity-oriented in both modes.
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
            # Exploitation: sample from the front, but bias away from the
            # single most-reused elite by sampling uniformly.
            parent = self.rng.choice(front)
        else:
            # Exploration: fitness-biased choice over the whole population.
            def _score(p: EvolvedProgram) -> float:
                try:
                    return (
                        float(p.metrics.get("combined_score", 0.0))
                        if p.metrics
                        else 0.0
                    )
                except (TypeError, ValueError):
                    return 0.0

            pool = [p for p in candidates if p.id not in front_ids] or candidates
            scores = [_score(p) for p in pool]
            lo, hi = min(scores), max(scores)
            if hi > lo:
                # Rank-linear weights: better programs get more weight but
                # weaker ones are never excluded.
                weights = [0.1 + 0.9 * (s - lo) / (hi - lo) for s in scores]
            else:
                weights = [1.0] * len(pool)
            parent = self.rng.choices(pool, weights=weights, k=1)[0]

        examples = self._diverse_context(candidates, parent.id, n_ctx, front_ids)

        parent_dict = {"": parent}
        context_programs_dict = {"": examples}

        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END
