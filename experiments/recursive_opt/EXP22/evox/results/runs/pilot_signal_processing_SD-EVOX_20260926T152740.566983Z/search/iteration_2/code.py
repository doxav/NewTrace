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


def _score(program: EvolvedProgram) -> float:
    v = program.metrics.get("combined_score") if program.metrics else None
    return float(v) if isinstance(v, (int, float)) else 0.0


class EvolvedProgramDatabase(ProgramDatabase):
    """Simple adaptive search database.

    - Mutates the best program with small context by default (exploit).
    - Tracks best-score progress in add(); when stuck (no meaningful
      improvement), alternates REFINE-on-best and DIVERGE-on-a-contrasting
      program to escape plateaus.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.best_score: float = float("-inf")
        self.best_id: Optional[str] = None
        self.stagnation: int = 0
        self.escape_counter: int = 0
        self.iteration: int = 0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0 and self.best_id is None:
            self.best_score = _score(program)
            self.best_id = program.id

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
            self.iteration = max(self.iteration, iteration)

        s = _score(program)
        if self.best_id is None or s > self.best_score:
            if s > self.best_score + max(0.01, 0.01 * abs(self.best_score)) if self.best_id is not None else True:
                self.stagnation = 0
            self.best_score = s
            self.best_id = program.id
        else:
            self.stagnation += 1

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(self, num_context_programs: Optional[int] = 4, **kwargs):
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        num_context = num_context_programs if num_context_programs else 0

        # Rank programs by score.
        ranked = sorted(candidates, key=_score, reverse=True)
        best = ranked[0]

        parent_label = ""
        parent = best
        context: List[EvolvedProgram] = []

        if self.stagnation >= 2:
            self.escape_counter += 1
            mode = self.escape_counter % 2
            if mode == 1:
                # REFINE the best: focused context around the top performers.
                parent_label = self.REFINE_LABEL
                pool = [p for p in ranked if p.id != best.id]
                context = pool[: min(2, num_context or 2)]
            else:
                # DIVERGE: pick a contrasting (non-best, non-worst) parent with
                # diverse context spanning the score range.
                parent_label = self.DIVERGE_LABEL
                if len(ranked) >= 3:
                    parent = ranked[len(ranked) // 2]
                else:
                    parent = ranked[-1]
                pool = [p for p in candidates if p.id != parent.id]
                # Diverse spread: top, bottom, and one random middle.
                picks: List[EvolvedProgram] = []
                seen = set()
                for p in [ranked[0]] + pool[::-1][: len(pool)]:
                    if p.id not in seen and p.id != parent.id:
                        picks.append(p)
                        seen.add(p.id)
                context = picks[: max(1, min(3, num_context or 3))]
        else:
            # Default: mutate best with small, light context.
            pool = [p for p in ranked if p.id != best.id]
            self.rng.shuffle(pool)
            context = pool[: min(2, num_context)]

        parent_dict = {parent_label: parent}
        context_programs_dict = {"": context}
        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END