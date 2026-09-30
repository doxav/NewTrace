# EVOLVE-BLOCK-START
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from skydiscover.optimize.config import DatabaseConfig
from skydiscover.optimize.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


class EvolvedProgramDatabase(ProgramDatabase):
    """Search strategy database with score-biased sampling.

    Parent selection favors high-scoring programs (top-k, score-weighted)
    while context sampling mixes the best program with diverse mid/low
    scorers to preserve exploration. Scalar mode uses combined_score;
    multiobjective mode prefers the Pareto front.
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
        """Scalar score used for biasing; falls back to 0 when absent."""
        try:
            return float(program.combined_score)
        except (AttributeError, TypeError, ValueError):
            metrics = getattr(program, "metrics", None) or {}
            try:
                return float(metrics.get("combined_score", 0.0))
            except (TypeError, ValueError):
                return 0.0

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """
        Picks a parent and set of context programs.

        Parent: sampled from the top half of the population by score with
        weights proportional to score (exploitation of strong ancestors).
        Context: always includes the best program, then fills with diverse
        programs from the rest of the population.
        """
        candidates = list(self.programs.values())

        if len(candidates) == 0:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 0

        # ---- Parent selection: score-weighted among top half ----
        scored = sorted(candidates, key=self._score_of, reverse=True)
        top_k = max(1, len(scored) // 2)
        pool = scored[:top_k]
        weights = [max(self._score_of(p), 1e-6) for p in pool]
        parent = self.rng.choices(pool, weights=weights, k=1)[0]

        # ---- Context selection: best + diverse others ----
        examples: List[EvolvedProgram] = []
        best = scored[0]
        if n_ctx > 0 and best.id != parent.id:
            examples.append(best)

        others = [
            p
            for p in candidates
            if p.id != parent.id and all(p.id != e.id for e in examples)
        ]
        self.rng.shuffle(others)
        examples.extend(others[: max(0, n_ctx - len(examples))])

        parent_dict = {"": parent}
        context_programs_dict = {"": examples}

        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END
