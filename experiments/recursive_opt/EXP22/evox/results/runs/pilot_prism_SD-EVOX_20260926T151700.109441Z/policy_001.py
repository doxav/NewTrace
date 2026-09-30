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
    """Simple adaptive search database.

    Parent selection is score-weighted (exploit the best) with occasional
    random picks (explore). When progress stalls, the best program is
    refined; after deeper stagnation, divergence is triggered from a
    random non-best program. Context shows a mix of top programs and
    diverse alternatives.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.best_score: float = float("-inf")
        self.best_id: Optional[str] = None
        self.stagnation: int = 0
        self.total_added: int = 0

    @staticmethod
    def _score(program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score") if program.metrics else None
        return float(v) if isinstance(v, (int, float)) else float("-inf")

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program
        self.total_added += 1

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        s = self._score(program)
        if s > self.best_score + max(0.01, 0.01 * abs(self.best_score)):
            self.best_score = s
            self.best_id = program.id
            self.stagnation = 0
        else:
            self.stagnation += 1

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = [(self._score(p), p) for p in candidates]
        scored.sort(key=lambda t: t[0], reverse=True)
        ranked = [p for _, p in scored]
        best = ranked[0]

        n_ctx = num_context_programs or 0
        label = ""

        # Stagnation-driven strategy
        if self.stagnation >= 4 and len(ranked) > 1:
            # Deeply stuck: diverge from a random non-best program
            parent = self.rng.choice(ranked[1:])
            label = self.DIVERGE_LABEL
            parent_dict = {label: parent}
            return parent_dict, {}
        elif self.stagnation >= 2:
            # Mildly stuck: refine the best program
            parent = best
            label = self.REFINE_LABEL
            parent_dict = {label: parent}
            return parent_dict, {}

        # Default: mostly exploit best, sometimes explore others
        if self.rng.random() < 0.7 or len(ranked) == 1:
            parent = best
        else:
            parent = self.rng.choice(ranked[1:])

        # Context: top programs + a couple diverse/lower ones
        pool = [p for p in ranked if p.id != parent.id]
        top = pool[: max(1, n_ctx // 2)]
        rest = pool[len(top):]
        self.rng.shuffle(rest)
        examples = (top + rest)[:n_ctx]

        return {"": parent}, {"": examples}


# EVOLVE-BLOCK-END